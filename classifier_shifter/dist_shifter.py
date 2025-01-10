import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

class DistributionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DistributionMapper(nn.Module):
    def __init__(self, input_dim):
        super(DistributionMapper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
            # Remove Softmax here - we'll handle it separately
        )
    
    def forward(self, x):
        logits = self.model(x)
        # Add small epsilon before softmax for numerical stability
        return torch.softmax(logits + 1e-15, dim=1)

def compute_distribution_metrics(pred, target, epsilon=1e-15):
    pred_np = pred.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()
    
    # Clip values to ensure they're positive
    pred_np = np.clip(pred_np, epsilon, 1.0)
    target_np = np.clip(target_np, epsilon, 1.0)
    
    # Renormalize
    pred_np = pred_np / pred_np.sum(axis=1, keepdims=True)
    target_np = target_np / target_np.sum(axis=1, keepdims=True)
    
    metrics = {
        'js_distances': [],
        'wasserstein_distances': [],
        'l1_distances': [],
        'l2_distances': []
    }
    
    for p, t in zip(pred_np, target_np):
        # Jensen-Shannon divergence
        js_dist = jensenshannon(p, t)
        metrics['js_distances'].append(0.0 if np.isnan(js_dist) else js_dist)
        
        # Wasserstein distance
        w_dist = wasserstein_distance(p, t)
        metrics['wasserstein_distances'].append(0.0 if np.isnan(w_dist) else w_dist)
        
        # L1 distance
        metrics['l1_distances'].append(np.sum(np.abs(p - t)))
        
        # L2 distance
        metrics['l2_distances'].append(np.sqrt(np.sum((p - t) ** 2)))
    
    return {k: np.mean(v) for k, v in metrics.items()}

def train_model(model, train_loader, val_loader, num_epochs=100):
    wandb.init(
        project="distribution-mapper",
        config={
            "architecture": "MLP",
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam",
            "learning_rate": 0.001
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use KLDivLoss with log_target=True for better numerical stability
    criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    epsilon = 1e-15
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_metrics = {
            'train_loss': 0,
            'train_js_dist': 0,
            'train_wasserstein_dist': 0,
            'train_l1_dist': 0,
            'train_l2_dist': 0
        }
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Add epsilon and take log for KL divergence
            outputs_log = torch.log(outputs + epsilon)
            target_log = torch.log(batch_y + epsilon)
            
            loss = criterion(outputs_log, target_log)
            loss.backward()
            optimizer.step()
            
            batch_metrics = compute_distribution_metrics(outputs, batch_y)
            
            train_metrics['train_loss'] += loss.item()
            train_metrics['train_js_dist'] += batch_metrics['js_distances']
            train_metrics['train_wasserstein_dist'] += batch_metrics['wasserstein_distances']
            train_metrics['train_l1_dist'] += batch_metrics['l1_distances']
            train_metrics['train_l2_dist'] += batch_metrics['l2_distances']
        
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        
        # Validation
        model.eval()
        val_metrics = {
            'val_loss': 0,
            'val_js_dist': 0,
            'val_wasserstein_dist': 0,
            'val_l1_dist': 0,
            'val_l2_dist': 0
        }
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                outputs_log = torch.log(outputs + epsilon)
                target_log = torch.log(batch_y + epsilon)
                
                val_loss = criterion(outputs_log, target_log)
                batch_metrics = compute_distribution_metrics(outputs, batch_y)
                
                val_metrics['val_loss'] += val_loss.item()
                val_metrics['val_js_dist'] += batch_metrics['js_distances']
                val_metrics['val_wasserstein_dist'] += batch_metrics['wasserstein_distances']
                val_metrics['val_l1_dist'] += batch_metrics['l1_distances']
                val_metrics['val_l2_dist'] += batch_metrics['l2_distances']
        
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
        
        wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})
        
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save(model.state_dict(), 'best_distribution_mapper.pt')
            wandb.save('best_distribution_mapper.pt')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}:')
            print(f'Train - Loss: {train_metrics["train_loss"]:.4f}, JS: {train_metrics["train_js_dist"]:.4f}')
            print(f'Val - Loss: {val_metrics["val_loss"]:.4f}, JS: {val_metrics["val_js_dist"]:.4f}')

def load_data(query_path, doc_path, epsilon=1e-15):
    # Read CSV files
    query_df = pd.read_csv(query_path)
    doc_df = pd.read_csv(doc_path)
    
    # Extract probability columns (excluding 'text' column)
    prob_columns = [col for col in query_df.columns if col != 'text']
    
    # Convert to numpy arrays
    X = query_df[prob_columns].values  # Query probabilities
    y = doc_df[prob_columns].values    # Document probabilities
    
    # Clip values to ensure they're positive
    X = np.clip(X, epsilon, 1.0)
    y = np.clip(y, epsilon, 1.0)
    
    # Renormalize
    X = X / X.sum(axis=1, keepdims=True)
    y = y / y.sum(axis=1, keepdims=True)
    
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    print(f"Sample input distribution sum: {X[0].sum():.6f}")
    print(f"Sample output distribution sum: {y[0].sum():.6f}")
    
    return X, y

def main():
    wandb.login()
    
    query_path = '/shared/eng/pj20/firas_data/datasets/classifier_labeling_data/query_class_probabilities.csv'
    doc_path = '/shared/eng/pj20/firas_data/datasets/classifier_labeling_data/document_class_probabilities.csv'
    
    X, y = load_data(query_path, doc_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    train_dataset = DistributionDataset(X_train, y_train)
    val_dataset = DistributionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model = DistributionMapper(input_dim=X.shape[1])
    train_model(model, train_loader, val_loader)
    
    wandb.finish()

if __name__ == "__main__":
    main()
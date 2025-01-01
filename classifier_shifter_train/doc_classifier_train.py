import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import wandb
from tqdm import tqdm

class DBPediaDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, model):
        self.texts = texts
        self.labels = labels
        self.model = model
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        embedding = self.model.encode(text, convert_to_tensor=True)
        return {
            'embedding': embedding,
            'labels': torch.FloatTensor(self.labels[idx])
        }

class MultilabelClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, embeddings):
        return torch.sigmoid(self.classifier(embeddings))

class ModelSaver:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(
        self,
        classifier: torch.nn.Module,
        encoder: torch.nn.Module,
        label_mapping: Dict[str, int],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        epoch: int,
        optimizer_state: dict,
        is_best: bool = False
    ) -> None:
        checkpoint_dir = self.base_path / f"checkpoint_epoch_{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        model_path = checkpoint_dir / "model_state.pt"
        torch.save({
            'classifier_state': classifier.state_dict(),
            'encoder_state': encoder.state_dict(),
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'metrics': metrics
        }, model_path, _use_new_zipfile_serialization=True)
        
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': config,
                'label_mapping': label_mapping,
                'metrics': metrics
            }, f, indent=2)
        
        if is_best:
            best_path = self.base_path / "best_model"
            if best_path.exists():
                best_path.unlink()
            os.symlink(checkpoint_dir, best_path)
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        classifier: torch.nn.Module,
        encoder: torch.nn.Module,
        device: torch.device
    ) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        
        model_state = torch.load(
            checkpoint_path / "model_state.pt",
            map_location=device
        )
        
        classifier.load_state_dict(model_state['classifier_state'])
        encoder.load_state_dict(model_state['encoder_state'])
        
        with open(checkpoint_path / "config.json", 'r') as f:
            config = json.load(f)
            
        return {
            'epoch': model_state['epoch'],
            'metrics': model_state['metrics'],
            'model_config': config['model_config'],
            'label_mapping': config['label_mapping'],
            'optimizer_state': model_state.get('optimizer_state', None)
        }

def load_data(file_path: str) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    
    label_cols = ['l1', 'l2', 'l3']
    unique_labels = set()
    for col in label_cols:
        unique_labels.update(df[col].unique())
    
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    print(f"Number of unique labels: {len(label_to_idx)}")
    
    labels = np.zeros((len(df), len(label_to_idx)))
    for i, row in df.iterrows():
        for col in label_cols:
            label_idx = label_to_idx[row[col]]
            labels[i, label_idx] = 1
            
    return texts, labels, label_to_idx

def train_model(model, sentence_transformer, train_dataloader, val_dataloader, device, label_to_idx, 
                num_epochs=10, start_epoch=0, optimizer_state=None, run_id=None):
    model_saver = ModelSaver('/shared/eng/pj20/firas_data/classifiers')
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 2e-4},
        {'params': sentence_transformer.parameters(), 'lr': 2e-5}
    ])
    
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    
    criterion = nn.BCELoss()
    best_val_f1 = 0
    
    wandb_config = {
        "architecture": "nomic-embed-text-v1",
        "learning_rate_classifier": 2e-4,
        "learning_rate_encoder": 2e-5,
        "epochs": num_epochs,
        "batch_size": train_dataloader.batch_size,
        "continued_training": start_epoch > 0
    }
    
    if run_id:
        wandb.init(project="dbpedia-classification", id=run_id, resume="must")
    else:
        wandb.init(project="dbpedia-classification", config=wandb_config)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training
        model.train()
        sentence_transformer.train()
        total_loss = 0
        train_steps = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
        
        for batch in progress_bar:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_steps += 1
            
            if train_steps % 1 == 0:
                wandb.log({"training_loss": loss.item()})
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        sentence_transformer.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                embeddings = batch['embedding'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_f1 = f1_score(val_labels, val_predictions, average='micro')
        val_precision = precision_score(val_labels, val_predictions, average='micro')
        val_recall = recall_score(val_labels, val_predictions, average='micro')
        
        metrics = {
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        
        config = {
            'architecture': 'nomic-embed-text-v1',
            'learning_rate_classifier': 2e-4,
            'learning_rate_encoder': 2e-5,
            'batch_size': train_dataloader.batch_size,
            'embedding_dim': model.classifier[0].in_features,
            'num_labels': model.classifier[-1].out_features
        }
        
        wandb.log({
            "epoch": epoch + 1,
            **metrics
        })
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation F1: {val_f1:.4f}')
        
        model_saver.save_checkpoint(
            classifier=model,
            encoder=sentence_transformer,
            label_mapping=label_to_idx,
            metrics=metrics,
            config=config,
            epoch=epoch + 1,
            optimizer_state=optimizer.state_dict(),
            is_best=(val_f1 > best_val_f1)
        )
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    wandb.finish()
    return best_val_f1

def load_trained_model(checkpoint_path, device):
    sentence_transformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    embed_dim = sentence_transformer.get_sentence_embedding_dimension()
    
    with open(Path(checkpoint_path) / "config.json", 'r') as f:
        config = json.load(f)
    
    classifier = MultilabelClassifier(embed_dim, len(config['label_mapping']))
    
    checkpoint_info = ModelSaver.load_checkpoint(
        checkpoint_path,
        classifier,
        sentence_transformer,
        device
    )
    
    return classifier, sentence_transformer, checkpoint_info

def evaluate_model(model, sentence_transformer, dataloader, device):
    model.eval()
    sentence_transformer.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            embeddings = batch['embedding'].to(device)
            outputs = model(embeddings)
            predictions = (outputs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
    
    metrics = {
        'f1': f1_score(all_labels, all_predictions, average='micro'),
        'precision': precision_score(all_labels, all_predictions, average='micro'),
        'recall': recall_score(all_labels, all_predictions, average='micro')
    }
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the last checkpoint if it exists
    last_checkpoint = None
    checkpoint_dir = Path('/shared/eng/pj20/firas_data/classifiers')
    checkpoints = [d for d in checkpoint_dir.glob("checkpoint_epoch_*")]
    if checkpoints:
        last_checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split('_')[-1])))
        print(f"Continuing training from checkpoint: {last_checkpoint}")
    
    # Initialize models and load state if checkpoint exists
    sentence_transformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    embed_dim = sentence_transformer.get_sentence_embedding_dimension()
    
    start_epoch = 0
    optimizer_state = None
    wandb_run_id = None
    
    if last_checkpoint:
        classifier, sentence_transformer, checkpoint_info = load_trained_model(last_checkpoint, device)
        start_epoch = checkpoint_info['epoch']
        optimizer_state = checkpoint_info.get('optimizer_state')
        
        # Get the wandb run ID from the checkpoint directory if it exists
        wandb_dir = Path(last_checkpoint) / "wandb"
        if wandb_dir.exists():
            try:
                with open(wandb_dir / "run-id.txt", "r") as f:
                    wandb_run_id = f.read().strip()
            except:
                pass
    else:
        classifier = MultilabelClassifier(embed_dim, num_labels=0)  # Will be set after loading data
    
    # Load datasets
    train_texts, train_labels, label_to_idx = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_train.csv')
    val_texts, val_labels, _ = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_val.csv')
    test_texts, test_labels, _ = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_test.csv')
    
    # If starting fresh, initialize the classifier with the correct number of labels
    if not last_checkpoint:
        classifier = MultilabelClassifier(embed_dim, num_labels=len(label_to_idx))
    
    # Move models to device
    classifier = classifier.to(device)
    sentence_transformer.to(device)
    
    # Create datasets and dataloaders
    train_dataset = DBPediaDataset(train_texts, train_labels, sentence_transformer)
    val_dataset = DBPediaDataset(val_texts, val_labels, sentence_transformer)
    test_dataset = DBPediaDataset(test_texts, test_labels, sentence_transformer)
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Continue training
    additional_epochs = 10  # Set the number of additional epochs you want to train
    best_val_f1 = train_model(
        classifier,
        sentence_transformer,
        train_dataloader,
        val_dataloader,
        device,
        label_to_idx,
        num_epochs=additional_epochs,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
        run_id=wandb_run_id
    )
    
    # Load best model and evaluate
    best_checkpoint_path = '/shared/eng/pj20/firas_data/classifiers/best_model'
    classifier, sentence_transformer, checkpoint_info = load_trained_model(best_checkpoint_path, device)
    classifier.to(device)
    sentence_transformer.to(device)
    
    test_metrics = evaluate_model(classifier, sentence_transformer, test_dataloader, device)
    
    print("\nTest Results:")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")

if __name__ == "__main__":
    main()
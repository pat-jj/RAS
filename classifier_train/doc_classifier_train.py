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

def train_model(
    model, 
    sentence_transformer, 
    train_dataloader, 
    val_dataloader, 
    device, 
    label_to_idx, 
    start_epoch=0,  # New parameter for continued training
    num_epochs=10,
    checkpoint_dir='/shared/eng/pj20/firas_data/classifiers'
):
    model_saver = ModelSaver(checkpoint_dir)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 2e-4},
        {'params': sentence_transformer.parameters(), 'lr': 2e-5}
    ])
    criterion = nn.BCELoss()
    
    # Load best validation F1 from previous training if it exists
    best_val_f1 = 0
    if start_epoch > 0:
        best_model_path = Path(checkpoint_dir) / "best_model"
        if best_model_path.exists():
            with open(best_model_path / "config.json", 'r') as f:
                best_config = json.load(f)
                best_val_f1 = best_config['metrics']['val_f1']
    
    wandb.init(
        project="dbpedia-classification",
        config={
            "architecture": "nomic-embed-text-v1",
            "learning_rate_classifier": 2e-4,
            "learning_rate_encoder": 2e-5,
            "epochs": num_epochs,
            "batch_size": train_dataloader.batch_size,
            "continued_from_epoch": start_epoch  # Track continued training
        },
        resume=True if start_epoch > 0 else False  # Resume wandb logging if continuing
    )
    
    for epoch in range(start_epoch, num_epochs):
        # [Rest of training loop remains the same...]
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
            is_best=(val_f1 > best_val_f1)
        )
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    wandb.finish()
    return best_val_f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for existing checkpoints
    checkpoint_dir = '/shared/eng/pj20/firas_data/classifiers'
    start_epoch = 0
    total_epochs = 20  # Set your desired total epochs here
    
    # Look for the latest checkpoint
    checkpoints = list(Path(checkpoint_dir).glob("checkpoint_epoch_*"))
    if checkpoints:
        latest_epoch = max(int(cp.name.split('_')[-1]) for cp in checkpoints)
        if latest_epoch < total_epochs:  # Only continue if we haven't reached total_epochs
            start_epoch = latest_epoch
            print(f"Continuing training from epoch {start_epoch}")
            
            # Load the latest checkpoint
            classifier, sentence_transformer, checkpoint_info = load_trained_model(
                str(Path(checkpoint_dir) / f"checkpoint_epoch_{start_epoch}"),
                device
            )
        else:
            print(f"Training already completed ({latest_epoch} epochs)")
            return
    else:
        # Initialize new model
        sentence_transformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        sentence_transformer.to(device)
        embed_dim = sentence_transformer.get_sentence_embedding_dimension()
        
        # Load datasets and create classifier
        train_texts, train_labels, label_to_idx = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_train.csv')
        classifier = MultilabelClassifier(embed_dim, num_labels=len(label_to_idx))
        classifier.to(device)
    
    # Load datasets
    train_texts, train_labels, label_to_idx = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_train.csv')
    val_texts, val_labels, _ = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_val.csv')
    
    # Create datasets and dataloaders
    train_dataset = DBPediaDataset(train_texts, train_labels, sentence_transformer)
    val_dataset = DBPediaDataset(val_texts, val_labels, sentence_transformer)
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Continue training
    best_val_f1 = train_model(
        classifier,
        sentence_transformer,
        train_dataloader,
        val_dataloader,
        device,
        label_to_idx,
        start_epoch=start_epoch,
        num_epochs=total_epochs
    )
    
    print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import os
from typing import List, Dict, Tuple
import wandb
from tqdm import tqdm

class DBPediaDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': str(self.texts[idx]),
            'label': torch.FloatTensor(self.labels[idx])
        }

class MultilabelEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, sentences: List[str], labels: List[List[float]], batch_size: int = 32, name: str = ''):
        self.sentences = sentences
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.name = name

    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        model.eval()
        
        all_predictions = []
        
        for start_idx in range(0, len(self.sentences), self.batch_size):
            batch_sentences = self.sentences[start_idx:start_idx + self.batch_size]
            
            with torch.no_grad():
                embeddings = model.encode(batch_sentences, convert_to_tensor=True)
                logits = model.classification_head(embeddings)
                predictions = torch.sigmoid(logits)
                all_predictions.extend(predictions.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        predictions_binary = (all_predictions > 0.5).astype(float)
        
        f1 = f1_score(self.labels, predictions_binary, average='micro')
        precision = precision_score(self.labels, predictions_binary, average='micro')
        recall = recall_score(self.labels, predictions_binary, average='micro')
        
        logging.info(f"{self.name} evaluation{out_txt}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        
        return f1

def load_data(file_path: str) -> Tuple[List[str], np.ndarray]:
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

class WandbCallback:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.best_score = -1

    def on_epoch_end(self, score, epoch, steps):
        wandb.log({
            "epoch": epoch,
            "validation_f1": score,
        })
        
        if score > self.best_score:
            self.best_score = score
            wandb.log({"best_validation_f1": score})

def main():
    # Initialize wandb
    wandb.init(
        project="dbpedia-classification",
        config={
            "architecture": "sentence-transformer",
            "learning_rate": 2e-5,
            "epochs": 5,
            "batch_size": 32
        }
    )

    # Load data
    train_texts, train_labels, label_to_idx = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_train.csv')
    val_texts, val_labels, _ = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_val.csv')
    test_texts, test_labels, _ = load_data('/shared/eng/pj20/firas_data/datasets/DBPedia/DBPEDIA_test.csv')

    # Initialize model
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    
    # Add classification head
    model.add_module('classification_head', 
                    nn.Sequential(
                        nn.Linear(model.get_sentence_embedding_dimension(), 512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, len(label_to_idx))
                    ))
    
    # Create datasets
    train_dataset = DBPediaDataset(train_texts, train_labels)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create loss function
    train_loss = losses.BCEWithLogitsLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(label_to_idx)
    )

    # Create evaluator
    evaluator = MultilabelEvaluator(
        val_texts,
        val_labels.tolist(),
        name='dbpedia-validation'
    )

    # Create WandB callback
    wandb_callback = WandbCallback(evaluator)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=5,
        evaluation_steps=1000,
        warmup_steps=100,
        output_path='/shared/eng/pj20/firas_data/classifiers/dbpedia_st',
        callback=lambda score, epoch, steps: wandb_callback.on_epoch_end(score, epoch, steps)
    )

    # Test evaluation
    test_evaluator = MultilabelEvaluator(
        test_texts,
        test_labels.tolist(),
        name='dbpedia-test'
    )
    
    test_score = test_evaluator(model)
    
    print("\nTest Results:")
    print(f"F1 Score: {test_score:.4f}")
    
    # Save label mapping
    torch.save({
        'label_to_idx': label_to_idx
    }, os.path.join('/shared/eng/pj20/firas_data/classifiers/dbpedia_st', 'label_mapping.pt'))

    wandb.log({"test_f1": test_score})
    wandb.finish()

if __name__ == "__main__":
    main()
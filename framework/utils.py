import torch
import torch.nn.parallel
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
import json
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from functools import partial
import time
from datetime import datetime
import numpy as np


class MultilabelClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_labels)
        )
        
    def forward(self, embeddings):
        return torch.sigmoid(self.classifier(embeddings))

def load_theme_classifier(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    
    # Load model configuration
    with open(checkpoint_path / "config.json", 'r') as f:
        config = json.load(f)
    
    # Initialize models
    sentence_transformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    embed_dim = sentence_transformer.get_sentence_embedding_dimension()
    classifier = MultilabelClassifier(embed_dim, len(config['label_mapping']))
    
    # Load saved states to CPU first
    model_state = torch.load(
        checkpoint_path / "model_state.pt",
        map_location='cpu'    # Load to CPU regardless of where it was saved
    )
    
    classifier.load_state_dict(model_state['classifier_state'])
    sentence_transformer.load_state_dict(model_state['encoder_state'])
    
    # Models can be moved to the desired device later when needed
    # classifier.to(device)
    # sentence_transformer.to(device)
    
    return classifier, sentence_transformer, config['label_mapping']
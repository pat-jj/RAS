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
import torch.nn as nn
import json
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from claude_api import get_claude_response
from threading import Lock
import queue
import logging
from concurrent.futures import ThreadPoolExecutor



def get_planner_instruction(model_name):
    if model_name != 'sonnet':
        # return """You are a planner to determine if the question can be answered with current information.
        # You will be output [NO_RETRIEVAL] if the question can be directly answered with the question itself.
        # You will be output [SUBQ] with the subquery if the question needs a subquery.
        # You will be output [SUFFICIENT] if the question can be answered with provided information.
        # """
        return """You are a planner to determine if the question can be answered with current information and output the appropriate label as well as the subquery if needed.
Output [NO_RETRIEVAL] if the question can be directly answered with the question itself without any retrieval.
Output [SUBQ] with an subquery for retrieval if still needs a subquery.
Output [SUFFICIENT] if the question can be answered with the provided information.
"""


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
        )
    
    def forward(self, x):
        logits = self.model(x)
        # Add small epsilon before softmax for numerical stability
        return torch.softmax(logits + 1e-15, dim=1)
    
    
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


def load_theme_distribution_shifter(checkpoint_path: str, input_dim: int):
    """Load the theme distribution shifter model from checkpoint"""
    distribution_mapper = DistributionMapper(input_dim)
    
    # Load saved states to CPU first
    model_state = torch.load(
        checkpoint_path,
        map_location='cpu'    # Load to CPU regardless of where it was saved
    )
    
    distribution_mapper.load_state_dict(model_state)
    
    return distribution_mapper



class GraphProcessor:
    def __init__(self, bert_model='sentence-transformers/all-roberta-large-v1'):
        """
        Initialize the data processor.
        Args:
            bert_model (str): Name of the SentenceBERT model to use for embeddings
        """
        self.sentence_model = SentenceTransformer(bert_model)
        self.embed_dim = self.sentence_model.get_sentence_embedding_dimension()
    

    def create_graph_from_triples(self, triple_strs):
        """Convert a list of triple strings into a PyG graph with predicate encodings"""
        nodes = set()
        edge_triples = []
        
        # Collect unique nodes and edges
        for triple_str in triple_strs:
            # Keep original triple string for description
            triple_str = triple_str.strip('()')
            parts = triple_str.split('|')
            
            # Extract subject, predicate, object
            subject = parts[0].replace('S>', '').strip()
            predicate = parts[1].replace('P>', '').strip()
            object_ = parts[2].replace('O>', '').strip()
            
            nodes.add(subject)
            nodes.add(object_)
            edge_triples.append((subject, predicate, object_))
        
        # Create node mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge index and collect predicates
        edge_index = []
        predicates = []
        
        for subj, pred, obj in edge_triples:
            # Add forward edge
            edge_index.append([node_to_idx[subj], node_to_idx[obj]])
            predicates.append(pred)  # Original predicate
            
            # Add reverse edge
            edge_index.append([node_to_idx[obj], node_to_idx[subj]])
            predicates.append(f"inverse_{pred}")  # Inverse predicate
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create node features (embeddings)
        node_texts = list(nodes)
        node_embeddings = self.sentence_model.encode(node_texts)
        node_features = torch.tensor(node_embeddings, dtype=torch.float)
        
        # Create edge features (only encode predicates)
        predicate_embeddings = self.sentence_model.encode(predicates)
        edge_features = torch.tensor(predicate_embeddings, dtype=torch.float)
        
        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )
        
        return graph
    
    
def generate_triples_claude(text: str) -> str:
    """
    Generates relationship triples from input text.
    
    Args:
        text (str): Input text to extract relationships from
        
    Returns:
        str: Comma-separated triples in format (S> subject| P> predicate| O> object)
    """
    prompt = f"""Extract relationship triples from the given text. Each triple should have exactly one subject (S>), one predicate (P>), and one object (O>).

Rules:
1. Extract as many meaningful triples as possible
2. Each triple must be in format: (S> subject| P> predicate| O> object)
3. Multiple triples should be separated by commas
4. Avoid using pronouns (it/he/she) - always use the actual names
5. Keep all entities in their original case (uppercase/lowercase)
6. Make predicates clear and specific

Example Input:
"William Gerald Standridge (November 27, 1953 – April 12, 2014) was an American stock car racing driver. He was a competitor in the NASCAR Winston Cup Series and Busch Series."

Example Output:
(S> William gerald standridge| P> Nationality| O> American),
(S> William gerald standridge| P> Occupation| O> Stock car racing driver),
(S> William gerald standridge| P> Competitor| O> Busch series),
(S> William gerald standridge| P> Competitor| O> Nascar winston cup series),
(S> William gerald standridge| P> Birth date| O> November 27, 1953),
(S> William gerald standridge| P> Death date| O> April 12, 2014)

Input Text: {text}

Output only the triples, nothing else."""

    # Here you would add your actual implementation to get response from Claude API
    # For example:
    response = get_claude_response(prompt)
    return response.strip()
    
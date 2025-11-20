#!/usr/bin/env python3
"""
Baseline-compatible retriever for KnowTrace.
Uses the same dense retrieval approach as other baselines.
"""

import faiss
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import orjson
import gc
from typing import List, Dict, Any
import os


class BaselineRetriever:
    """Retriever using baseline's dense retrieval approach"""
    
    def __init__(self, knowledge_path: str, dense_encoder: str = 'facebook/contriever-msmarco', 
                 num_splits: int = 5, topk: int = 5, device: str = 'cpu'):
        """
        Initialize the retriever
        
        Args:
            knowledge_path: Path to knowledge base indices
            dense_encoder: Dense encoder model name
            num_splits: Number of index splits
            topk: Number of top results to return
            device: Device to use ('cpu' or 'cuda')
        """
        self.knowledge_path = knowledge_path
        self.num_splits = num_splits
        self.topk = topk
        self.device = device
        
        print("Loading dense encoder model...")
        self.tokenizer = AutoTokenizer.from_pretrained(dense_encoder)
        self.model = AutoModel.from_pretrained(dense_encoder)
        if device == 'cpu':
            self.model = self.model.cpu()
        else:
            self.model = self.model.cuda()
        self.model.eval()
        
        # Cache for loaded indices (load lazily on first use)
        self._text_mappings = {}
        self._faiss_indices = {}
    
    def _load_knowledge_split(self, split_idx: int):
        """Load FAISS indices and text mappings for a specific split"""
        if split_idx in self._text_mappings:
            return self._text_mappings[split_idx], self._faiss_indices[split_idx]
        
        # Load text mappings
        mapping_path = f"{self.knowledge_path}/embedding/text_mapping_{split_idx}.json"
        with open(mapping_path, 'rb') as f:
            text_mapping = orjson.loads(f.read())
        
        # Load FAISS indices
        index_path = f"{self.knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss"
        faiss_index = faiss.read_index(index_path)
        
        if hasattr(faiss_index, 'nprobe'):
            faiss_index.nprobe = 256
        
        # Cache
        self._text_mappings[split_idx] = text_mapping
        self._faiss_indices[split_idx] = faiss_index
        
        return text_mapping, faiss_index
    
    def _dense_retrieve(self, query: str, tokenizer, model, dense_index, text_mapping, top_k: int = 30):
        """Dense retrieval function for a single query"""
        query_inputs = tokenizer(
            query, padding=True, truncation=True, return_tensors="pt")
        
        if self.device == 'cuda':
            query_inputs = {k: v.cuda() for k, v in query_inputs.items()}
        
        with torch.no_grad():
            query_embedding = model(**query_inputs).last_hidden_state[:, 0]
            if self.device == 'cuda':
                query_embedding = query_embedding.cpu()
            query_embedding = query_embedding.numpy()
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        dense_scores, dense_doc_ids = dense_index.search(query_embedding, k=top_k)
        
        result = []
        for doc_id, score in zip(dense_doc_ids[0], dense_scores[0]):
            # text_mapping is a list, use doc_id as index
            try:
                doc_id_int = int(doc_id)
                if 0 <= doc_id_int < len(text_mapping):
                    result.append((text_mapping[doc_id_int], float(score)))
            except (ValueError, IndexError, TypeError):
                # Fallback: try as key if it's actually a dict
                if isinstance(text_mapping, dict):
                    doc_key = str(doc_id) if str(doc_id) in text_mapping else doc_id
                    if doc_key in text_mapping:
                        result.append((text_mapping[doc_key], float(score)))
        
        return result
    
    def retrieve(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a list of questions/entities
        
        Args:
            questions: List of questions or entity descriptions
            
        Returns:
            List of dictionaries with 'question' and 'contexts' keys
        """
        import time
        all_results = []
        
        for question in questions:
            start = time.time()
            # Retrieve from all splits
            split_results = []
            
            for split_idx in range(self.num_splits):
                try:
                    text_mapping, faiss_index = self._load_knowledge_split(split_idx)
                    split_result = self._dense_retrieve(
                        question, self.tokenizer, self.model, 
                        faiss_index, text_mapping, top_k=100
                    )
                    split_results.extend(split_result)
                except Exception as e:
                    print(f"Error retrieving from split {split_idx}: {e}")
                    continue
            
            # Sort by score and take top k
            split_results.sort(key=lambda x: x[1], reverse=True)
            top_results = split_results[:self.topk]
            
            # Format as KnowTrace expects
            # KnowTrace expects contexts with 'title' and 'text' keys
            # Our text_mapping contains plain strings (Wikipedia articles)
            # Wikipedia format: usually starts with article title, then text
            contexts = []
            for text, score in top_results:
                text_str = str(text) if not isinstance(text, str) else text
                
                # Wikipedia articles typically have the title as the first part
                # Extract first sentence or first 100 chars as title
                # Split by periods to get first sentence
                first_period = text_str.find('.')
                if first_period > 0 and first_period < 150:
                    # Use first sentence as title
                    title = text_str[:first_period].strip()
                    text_content = text_str
                else:
                    # Use first 100 chars as title, full text as content
                    title = text_str[:100].strip()
                    text_content = text_str
                
                # Ensure title is not too long
                if len(title) > 200:
                    title = title[:200]
                
                contexts.append({
                    'title': title,
                    'text': text_content,
                    'score': float(score)
                })
            
            all_results.append({
                'question': question,
                'contexts': contexts
            })
        
        return all_results
    
    def clear_cache(self):
        """Clear cached indices to free memory"""
        self._text_mappings.clear()
        self._faiss_indices.clear()
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()


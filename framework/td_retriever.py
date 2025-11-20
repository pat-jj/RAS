import torch
import numpy as np
import faiss
import faiss.contrib.torch_utils
from faiss import get_num_gpus
import orjson
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict
import os
from dataclasses import dataclass
from tqdm import tqdm
# Import clean_document if available, otherwise use a simple fallback
try:
    from utils import clean_document
except ImportError:
    def clean_document(doc):
        """Simple fallback if utils is not available"""
        return doc if doc else ""

@dataclass
class KnowledgeBase:
    """Container for knowledge base components of a single split."""
    text_mapping: Dict
    dense_faiss_index: faiss.Index
    idx_mapping: Dict


class DenseRetriever:
    def __init__(
        self,
        knowledge_path: str = '/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018',
        num_splits: int = 5,
        dense_encoder: str = 'facebook/contriever-msmarco',
        debug: bool = False,
        device: str = None,
        faiss_gpu_ids: List[int] = [1, 2, 3, 4, 6, 7],  # Use GPUs 1-4 for indices
    ):
        self.knowledge_path = knowledge_path
        self.num_splits = num_splits
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.faiss_gpu_ids = faiss_gpu_ids
        self.gpu_indices = {}
        
        # Check if GPU FAISS is available
        self.use_gpu_faiss = hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available()
        
        if self.use_gpu_faiss:
            # Initialize multiple GPU resources for FAISS
            self.res_list = [faiss.StandardGpuResources() for _ in faiss_gpu_ids]
            self.co = faiss.GpuClonerOptions()
            self.co.useFloat16 = True  # Use FP16 to save GPU memory
        else:
            # Use CPU FAISS
            self.res_list = None
            self.co = None
            print("⚠️  GPU FAISS not available, using CPU FAISS")
        
        # Dictionary to store knowledge bases for each split
        self.knowledge_bases: Dict[int, KnowledgeBase] = {}
        self.debug = debug
        
        # Load knowledge bases and models
        self._load_all_knowledge()
        self._load_models(dense_encoder)
        
            

    def _load_all_knowledge(self):
        """Load FAISS indices and mappings for all splits."""
        # Check if using combined format (wiki_2020) or split format (wiki_2018)
        combined_mapping_path = f"{self.knowledge_path}/embedding/text_mapping_cleaned.json"
        combined_index_path = f"{self.knowledge_path}/embedding/wikipedia_embeddings_cleaned.faiss"
        
        if os.path.exists(combined_mapping_path) and os.path.exists(combined_index_path):
            # Combined format (single file for all data)
            print("Detected combined format (wiki_2020 style)...")
            print("Loading combined indices and mappings...")
            
            # Load text mappings
            with open(combined_mapping_path, 'rb') as f:
                text_mapping = orjson.loads(f.read())
            
            idx_mapping = {i: i for i in range(len(text_mapping))}
            
            # Load dense index
            dense_faiss_index = faiss.read_index(combined_index_path)
            
            # Distribute indices across available GPUs or use CPU
            if self.use_gpu_faiss:
                gpu_idx = self.faiss_gpu_ids[0]  # Use first GPU for combined index
                res = self.res_list[0]
                
                try:
                    gpu_index = faiss.index_cpu_to_gpu(res, gpu_idx, dense_faiss_index)
                    print(f"Successfully moved combined index to GPU {gpu_idx}")
                    self.gpu_indices[0] = gpu_index
                    del dense_faiss_index  # Free CPU memory
                except RuntimeError as e:
                    print(f"Warning: Could not move combined index to GPU {gpu_idx}: {e}")
                    self.gpu_indices[0] = dense_faiss_index
            else:
                # Use CPU FAISS
                print("Using CPU FAISS for combined index")
                self.gpu_indices[0] = dense_faiss_index
            
            # Store knowledge base components (use split_idx=0 for combined format)
            self.knowledge_bases[0] = KnowledgeBase(
                text_mapping=text_mapping,
                dense_faiss_index=None,
                idx_mapping=idx_mapping
            )
            
            # Update num_splits to 1 for combined format
            self.num_splits = 1
            
            if self.debug:
                print(f"Loaded combined knowledge base with {len(text_mapping)} entries")
        else:
            # Split format (wiki_2018 style) - original logic
            for split_idx in tqdm(range(self.num_splits)):
                print(f"Loading indices and mappings for split {split_idx}...")
                
                # Load text mappings
                with open(f"{self.knowledge_path}/embedding/text_mapping_{split_idx}.json", 'rb') as f:
                    text_mapping = orjson.loads(f.read())
                
                idx_mapping = {i: i for i in range(len(text_mapping))}
                
                # Load dense index
                dense_faiss_index = faiss.read_index(
                    f"{self.knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss"
                )
                
                # Distribute indices across available GPUs or use CPU
                if self.use_gpu_faiss:
                    gpu_idx = self.faiss_gpu_ids[split_idx % len(self.faiss_gpu_ids)]
                    res = self.res_list[split_idx % len(self.faiss_gpu_ids)]
                    
                    try:
                        gpu_index = faiss.index_cpu_to_gpu(res, gpu_idx, dense_faiss_index)
                        print(f"Successfully moved split {split_idx} to GPU {gpu_idx}")
                        self.gpu_indices[split_idx] = gpu_index
                        del dense_faiss_index  # Free CPU memory
                    except RuntimeError as e:
                        print(f"Warning: Could not move split {split_idx} to GPU {gpu_idx}: {e}")
                        self.gpu_indices[split_idx] = dense_faiss_index
                else:
                    # Use CPU FAISS
                    print(f"Using CPU FAISS for split {split_idx}")
                    self.gpu_indices[split_idx] = dense_faiss_index
                
                # Store knowledge base components
                self.knowledge_bases[split_idx] = KnowledgeBase(
                    text_mapping=text_mapping,
                    dense_faiss_index=None,
                    idx_mapping=idx_mapping
                )
                
                if self.debug:
                    print(f"Knowledge base for split {split_idx} loaded for debug")

    def _load_models(self, dense_encoder: str):
        """Load dense encoder model."""
        self.dense_encoder_tokenizer = AutoTokenizer.from_pretrained(dense_encoder)
        self.dense_encoder_model = AutoModel.from_pretrained(dense_encoder).to(self.device)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query across all splits using dense retrieval.
        
        Args:
            query: Input query string
            top_k: Number of final results to return
            
        Returns:
            List of (text, score) tuples
        """
        query_inputs = self.dense_encoder_tokenizer(
            query, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            query_dense_embedding = self.dense_encoder_model(**query_inputs).last_hidden_state[:, 0]
            query_dense_embedding = query_dense_embedding / torch.norm(query_dense_embedding, p=2, dim=1)
        
        all_results = []
        
        # Process each split using pre-loaded GPU indices
        for split_idx, kb in self.knowledge_bases.items():
            gpu_index = self.gpu_indices[split_idx]
            
            dense_scores, dense_doc_ids = gpu_index.search(
                query_dense_embedding.cpu().numpy(), 
                k=top_k
            )
            
            split_results = [(kb.text_mapping[doc_id], float(score)) 
                        for doc_id, score in zip(dense_doc_ids[0], dense_scores[0])]
            all_results.extend([item for item in split_results if clean_document(item[0])])
        
        # Sort all results and return top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def __del__(self):
        """Cleanup GPU resources when the retriever is destroyed."""
        for split_idx, gpu_index in self.gpu_indices.items():
            try:
                del gpu_index
            except:
                pass
        torch.cuda.empty_cache()
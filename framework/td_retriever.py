import torch
import numpy as np
import faiss
import faiss.contrib.torch_utils
from faiss import get_num_gpus
import orjson
from transformers import AutoModel, AutoTokenizer
from utils import load_theme_classifier, load_theme_distribution_shifter
from typing import List, Tuple, Dict
import os
from dataclasses import dataclass
from tqdm import tqdm
from utils import clean_document

@dataclass
class KnowledgeBase:
    """Container for knowledge base components of a single split."""
    text_mapping: Dict
    dense_faiss_index: faiss.Index
    theme_faiss_index: faiss.Index
    idx_mapping: Dict


class ThemeScopedRetriever:
    def __init__(
        self,
        knowledge_path: str = '/shared/eng/pj20/firas_data/knowledge_source/wiki_2018',
        num_splits: int = 5,
        dense_encoder: str = 'facebook/contriever-msmarco',
        theme_encoder_path: str = '/shared/eng/pj20/firas_data/classifiers/best_model',
        theme_shifter_path: str = '/shared/eng/pj20/firas_data/classifiers/best_distribution_mapper.pt',
        retrieval_mode: str = 'theme_scoped',
        debug: bool = False,
        device: str = None,
        faiss_gpu_ids: List[int] = [1, 2, 3, 4, 6, 7],  # Use GPUs 1-4 for indices
    ):
        self.knowledge_path = knowledge_path
        self.num_splits = num_splits
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.faiss_gpu_ids = faiss_gpu_ids
        self.gpu_indices = {}
        
        # Initialize multiple GPU resources for FAISS
        self.res_list = [faiss.StandardGpuResources() for _ in faiss_gpu_ids]
        self.co = faiss.GpuClonerOptions()
        self.co.useFloat16 = True  # Use FP16 to save GPU memory
        
        # Dictionary to store knowledge bases for each split
        self.knowledge_bases: Dict[int, KnowledgeBase] = {}
        self.retrieval_mode = retrieval_mode
        self.debug = debug
        
        # Load knowledge bases and models
        self._load_all_knowledge()
        self._load_models(dense_encoder, theme_encoder_path, theme_shifter_path)
        
        if self.retrieval_mode == 'dense_only':
            self.retrieve = self._dense_only_retrieve
        else:
            self.retrieve = self._retrieve
            

    def _load_all_knowledge(self):
        """Load FAISS indices and mappings for all splits."""
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
            
            # Distribute indices across available GPUs
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
            
            # Store knowledge base components
            self.knowledge_bases[split_idx] = KnowledgeBase(
                text_mapping=text_mapping,
                dense_faiss_index=None,
                theme_faiss_index=None,
                idx_mapping=idx_mapping
            )
            
            if self.debug:
                print(f"Knowledge base for split {split_idx} loaded for debug")
                break

    def _load_models(self, dense_encoder: str, theme_encoder_path: str, theme_shifter_path: str):
        """Load all required models."""
        # Dense encoder
        self.dense_encoder_tokenizer = AutoTokenizer.from_pretrained(dense_encoder)
        self.dense_encoder_model = AutoModel.from_pretrained(dense_encoder).to(self.device)
        
        # Theme models
        self.theme_classifier, self.theme_encoder, self.theme_label_mapping = load_theme_classifier(theme_encoder_path)
        self.theme_classifier = self.theme_classifier.to(self.device)
        self.theme_encoder = self.theme_encoder.to(self.device)
        
        # Theme shifter
        self.theme_shifter = load_theme_distribution_shifter(
            theme_shifter_path, 
            input_dim=len(self.theme_label_mapping)
        ).to(self.device)

    def _convert_index_to_l2(self, index):
        """Convert an IP index to L2 index."""
        if isinstance(index, faiss.IndexFlatL2):
            return index
            
        dimension = index.d
        nvecs = index.ntotal
        vectors = index.reconstruct_batch(np.arange(nvecs))
        
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(vectors)
        return new_index

    def _retrieve(self, query: str, top_k: int = 10, theme_top_k: int = 100000) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query across all splits.
        
        Args:
            query: Input query string
            top_k: Number of final results to return
            theme_top_k: Number of candidates to consider from theme retrieval
            
        Returns:
            List of (text, score) tuples
        """
        # Get theme distribution
        query_theme_embedding = self.theme_encoder.encode(
            [query], 
            convert_to_tensor=True,
            batch_size=1
        ).to(self.device)
        
        with torch.no_grad():
            query_theme_probs = self.theme_classifier(query_theme_embedding)
            predicted_theme_distribution = self.theme_shifter(query_theme_probs)
            theme_norm = torch.norm(predicted_theme_distribution, p=2, dim=1)
            
            # Check if theme distribution is valid
            if theme_norm < 1e-8:
                return self._dense_only_retrieve(query, top_k)
                
            predicted_theme_distribution = predicted_theme_distribution / theme_norm.unsqueeze(1)
        
        # Get dense embeddings for query
        query_inputs = self.dense_encoder_tokenizer(
            query, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            query_dense_embedding = self.dense_encoder_model(**query_inputs).last_hidden_state[:, 0]
            query_dense_embedding = query_dense_embedding / torch.norm(query_dense_embedding, p=2, dim=1, keepdim=True)

        all_results = []
        
        # Process each split
        for split_idx, kb in self.knowledge_bases.items():
            # Theme-based retrieval
            theme_scores, theme_doc_ids = kb.theme_faiss_index.search(
                predicted_theme_distribution.cpu().numpy(), 
                k=theme_top_k
            )
            theme_scores = -theme_scores  # Convert L2 distances to similarities
            
            # Get candidate vectors from dense index
            dense_candidate_ids = np.array([
                kb.idx_mapping[int(idx)]
                for idx in theme_doc_ids[0]
                if int(idx) in kb.idx_mapping
            ], dtype=np.int64)
            
            # Create temporary index for dense search
            temp_index = faiss.IndexFlatIP(kb.dense_faiss_index.d)
            candidate_vectors = kb.dense_faiss_index.reconstruct_batch(dense_candidate_ids)
            temp_index.add(candidate_vectors)
            
            # Search in temporary index
            dense_scores, temp_doc_ids = temp_index.search(
                query_dense_embedding.cpu().numpy(),
                k=min(top_k * 2, len(dense_candidate_ids))
            )
            
            # Map back to original indices
            dense_doc_ids = dense_candidate_ids[temp_doc_ids[0]]
            
            # Get theme scores for selected documents
            theme_id_to_score = dict(zip(theme_doc_ids[0], theme_scores[0]))
            selected_theme_scores = np.array([
                theme_id_to_score[kb.idx_mapping[doc_id]]
                for doc_id in dense_doc_ids
            ])
            
            # Combine scores
            final_scores = 0.9 * dense_scores[0] + 0.1 * selected_theme_scores
            
            # Add results from this split
            split_results = [(kb.text_mapping[dense_doc_ids[i]], float(final_scores[i]))
                           for i in range(len(dense_doc_ids))]
            all_results.extend(split_results)
        
        # Sort all results and return top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _dense_only_retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
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
            all_results = [item for item in split_results if clean_document(item[0])]
        
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
    
    # def _dense_only_retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    #     """Fallback method for pure dense retrieval across all splits."""
    #     query_inputs = self.dense_encoder_tokenizer(
    #         query, 
    #         padding=True, 
    #         truncation=True, 
    #         return_tensors="pt"
    #     ).to(self.device)
        
    #     with torch.no_grad():
    #         query_dense_embedding = self.dense_encoder_model(**query_inputs).last_hidden_state[:, 0]
    #         query_dense_embedding = query_dense_embedding / torch.norm(query_dense_embedding, p=2, dim=1)
        
    #     all_results = []
        
    #     # Process each split
    #     for split_idx, kb in self.knowledge_bases.items():
    #         dense_scores, dense_doc_ids = kb.dense_faiss_index.search(
    #             query_dense_embedding.cpu().numpy(), 
    #             k=top_k
    #         )
            
    #         split_results = [(kb.text_mapping[doc_id], float(score)) 
    #                        for doc_id, score in zip(dense_doc_ids[0], dense_scores[0])]
    #         all_results.extend(split_results)
        
    #     # Sort all results and return top_k
    #     all_results.sort(key=lambda x: x[1], reverse=True)
    #     return all_results[:top_k]
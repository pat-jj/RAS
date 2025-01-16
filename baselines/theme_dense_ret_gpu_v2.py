import argparse
import json
try:
    import faiss
    import faiss.contrib.torch_utils
    USE_GPU = False  # Will be set to True if GPU FAISS is available
except ImportError:
    print("Warning: FAISS import failed. Please ensure FAISS is installed.")
    raise
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import MultilabelClassifier, load_theme_distribution_shifter, load_theme_classifier, load_file, save_file_jsonl
from copy import deepcopy
import orjson
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import torch.multiprocessing as mp

# Global variables for thread safety
gpu_lock = Lock()
results_lock = Lock()


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    return torch.device('cpu')

def clear_memory():
    """Enhanced memory clearing with GPU support"""
    try:
        if torch.cuda.is_available():
            with gpu_lock:  # Thread-safe GPU operations
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
        
        gc.collect()
        gc.collect()
    except Exception as e:
        print(f"Warning: Memory clearing encountered an error: {e}")
        pass

import os

def convert_index_to_l2(index):
    """Convert an IP index to L2 index"""
    if isinstance(index, faiss.IndexFlatL2):
        return index
        
    print("Converting IP index to L2 index...")
    dimension = index.d
    nvecs = index.ntotal
    
    # Get vectors using reconstruct_batch for all index types
    vectors = index.reconstruct_batch(np.arange(nvecs))
    
    new_index = faiss.IndexFlatL2(dimension)
    new_index.add(vectors)
    return new_index

def save_faiss_index_safely(index, filepath):
    """Safely save FAISS index with proper file handling"""
    temp_filepath = filepath + '.temp'
    try:
        # Write to temporary file first
        with open(temp_filepath, 'wb') as f:
            faiss.write_index(index, f)
            f.flush()  # Ensure all data is written
            os.fsync(f.fileno())  # Force sync to disk
        
        # Atomic rename to final filename
        os.replace(temp_filepath, filepath)
        
        # Verify the file was written correctly
        try:
            test_load = faiss.read_index(filepath)
            if test_load.ntotal != index.ntotal:
                raise ValueError("Saved index verification failed")
            return True
        except Exception as e:
            print(f"Warning: Index verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"Warning: Failed to save index to {filepath}: {e}")
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass
        return False
    
def load_knowledge(knowledge_path, split_idx, device):
    """Load FAISS indices with GPU acceleration if available"""
    print(f"Loading indices and mappings for split {split_idx}...")
    
    clear_memory()
    
    # Load text mappings
    with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'rb') as f:
        text_mapping = orjson.loads(f.read())
    
    idx_mapping = {i: i for i in range(len(text_mapping))}
    
    # Try to load L2 index first, if not found, convert and save
    theme_index_path = f"{knowledge_path}/theme_dist/theme_embeddings_{split_idx}"
    l2_index_path = f"{theme_index_path}_l2.faiss"
    
    if os.path.exists(l2_index_path):
        print(f"Loading existing L2 theme index for split {split_idx}")
        theme_faiss_index = faiss.read_index(l2_index_path)
        if not verify_index(theme_faiss_index, l2_index_path):
            print("Existing L2 index appears corrupted, recreating...")
            os.remove(l2_index_path)
            theme_faiss_index = None
    else:
        theme_faiss_index = None
        
    if theme_faiss_index is None:
        print(f"Converting to L2 index for split {split_idx}...")
        original_index = faiss.read_index(f"{theme_index_path}.faiss")
        theme_faiss_index = convert_index_to_l2(original_index)
        
        if save_faiss_index_safely(theme_faiss_index, l2_index_path):
            print(f"Successfully saved L2 index to {l2_index_path}")
        else:
            print("Warning: Failed to save L2 index, will use in-memory version")
    
    # Load dense index (keeping original IP similarity for dense retrieval)
    dense_faiss_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
    
    # Optimize FAISS parameters
    if hasattr(theme_faiss_index, 'nprobe'):
        theme_faiss_index.nprobe = 128
    if hasattr(dense_faiss_index, 'nprobe'):
        dense_faiss_index.nprobe = 256
    
    return (text_mapping, dense_faiss_index, 
            text_mapping, theme_faiss_index,
            idx_mapping, idx_mapping)

def load_models(args):
    """Load models with GPU support"""
    clear_memory()
    device = get_device()
    
    # Load models to GPU if available
    dense_encoder_tokenizer = AutoTokenizer.from_pretrained(args.dense_encoder)
    dense_encoder_model = AutoModel.from_pretrained(args.dense_encoder).to(device)
    
    theme_classifier, theme_encoder, theme_label_mapping = load_theme_classifier(args.theme_encoder_path)
    theme_classifier = theme_classifier.to(device)
    theme_encoder = theme_encoder.to(device)
    theme_shifter = load_theme_distribution_shifter(args.theme_shifter_path, input_dim=len(theme_label_mapping))
    theme_shifter = theme_shifter.to(device)
    
    return (dense_encoder_tokenizer, dense_encoder_model,
            theme_classifier, theme_encoder, theme_label_mapping, theme_shifter)

def convert_to_serializable(item):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(item, (np.int32, np.int64)):
        return int(item)
    elif isinstance(item, (np.float32, np.float64)):
        return float(item)
    elif isinstance(item, dict):
        return {key: convert_to_serializable(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert_to_serializable(element) for element in item]
    elif isinstance(item, tuple):
        return tuple(convert_to_serializable(element) for element in item)
    elif isinstance(item, np.ndarray):
        return convert_to_serializable(item.tolist())
    return item

def dense_only_retrieve(query, tokenizer, model, dense_index, text_mapping, top_k=30):
    """Fallback function for pure dense retrieval"""
    query_inputs = tokenizer(
        query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_dense_embedding = model(**query_inputs).last_hidden_state[:, 0]
        query_dense_embedding = query_dense_embedding / np.linalg.norm(query_dense_embedding)
    
    dense_scores, dense_doc_ids = dense_index.search(query_dense_embedding, k=top_k)
    
    result = [(text_mapping[doc_id], float(score)) 
              for doc_id, score in zip(dense_doc_ids[0], dense_scores[0])]
    
    clear_memory()
    return result

def process_query_batch(query_batch, models, knowledge_index, device, theme_top_k=300000, dense_top_k=100):
    """Process a batch of queries in parallel"""
    (dense_encoder_tokenizer, dense_encoder_model,
     theme_classifier, theme_encoder, theme_label_mapping, theme_shifter) = models
    
    (dense_text_mapping, dense_faiss_index,
     theme_text_mapping, theme_faiss_index,
     theme_to_dense_idx, dense_to_theme_idx) = knowledge_index
    
    batch_results = []
    
    # Batch encode queries for theme distribution
    query_theme_embeddings = theme_encoder.encode(query_batch, 
                                                convert_to_tensor=True,
                                                batch_size=len(query_batch)).to(device)
    
    with torch.no_grad():
        query_theme_probs = theme_classifier(query_theme_embeddings)
        predicted_theme_distributions = theme_shifter(query_theme_probs)
        theme_norms = torch.norm(predicted_theme_distributions, p=2, dim=1)
        
        # Normalize theme distributions
        valid_mask = theme_norms >= 1e-15
        predicted_theme_distributions[valid_mask] = predicted_theme_distributions[valid_mask] / theme_norms[valid_mask].unsqueeze(1)
    
    # Batch process dense embeddings
    query_inputs = dense_encoder_tokenizer(
        query_batch, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        query_dense_embeddings = dense_encoder_model(**query_inputs).last_hidden_state[:, 0]
        query_dense_embeddings = query_dense_embeddings / torch.norm(query_dense_embeddings, p=2, dim=1, keepdim=True)
    
    # Process each query in the batch
    for i, query in enumerate(query_batch):
        if not valid_mask[i]:
            # Fallback to dense-only retrieval
            result = dense_only_retrieve(query, dense_encoder_tokenizer, dense_encoder_model,
                                      dense_faiss_index, dense_text_mapping)
            batch_results.append(result)
            continue
        
        # Theme-based retrieval (now using L2 distance)
        theme_scores, theme_doc_ids = theme_faiss_index.search(
            predicted_theme_distributions[i:i+1].cpu().numpy(), k=theme_top_k)
        # Convert L2 distances to similarity scores (smaller L2 distance = higher similarity)
        theme_scores = -theme_scores
        
        # Convert theme doc ids to dense doc ids
        dense_candidate_ids = np.array([
            theme_to_dense_idx[int(idx)]
            for idx in theme_doc_ids[0]
            if int(idx) in theme_to_dense_idx
        ], dtype=np.int64)
        
        # Create temporary index with candidate vectors
        temp_index = faiss.IndexFlatIP(dense_faiss_index.d)
        candidate_vectors = dense_faiss_index.reconstruct_batch(dense_candidate_ids)
        temp_index.add(candidate_vectors)
        
        # Search in temporary index
        dense_scores, temp_doc_ids = temp_index.search(
            query_dense_embeddings[i:i+1].cpu().numpy(),
            k=min(dense_top_k, len(dense_candidate_ids))
        )
        
        # Map back to original dense indices
        dense_doc_ids = dense_candidate_ids[temp_doc_ids[0]]
        
        # Get theme scores
        theme_id_to_score = dict(zip(theme_doc_ids[0], theme_scores[0]))
        selected_theme_scores = np.array([
            theme_id_to_score[dense_to_theme_idx[doc_id]]
            for doc_id in dense_doc_ids
        ])
        
        # Combine scores and get top 10
        final_scores = 0.9 * dense_scores[0] + 0.1 * selected_theme_scores
        # final_scores = dense_scores[0]
        top_k = 10
        top_indices = np.argpartition(-final_scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-final_scores[top_indices])]
        
        result = [(dense_text_mapping[dense_doc_ids[i]], float(final_scores[i]))
                 for i in top_indices]
        
        batch_results.append(result)
    
    return batch_results

def process_split(split_idx, knowledge_path, models, all_batches, device):
    """Process all batches for a single split"""
    try:
        knowledge_index = load_knowledge(knowledge_path, split_idx, device)
        split_results = []
        
        for batch in tqdm(all_batches, desc=f"Processing batches for split {split_idx}", position=1):
            batch_queries = [item["instruction"] if "instruction" in item else item["question"]
                           for item in batch]
            retrieved_docs = process_query_batch(batch_queries, models, knowledge_index, device)
            
            for item, docs in zip(batch, retrieved_docs):
                item_copy = item.copy()
                item_copy['ctxs_theme'] = docs
                split_results.append(item_copy)
        
        return split_results
    finally:
        clear_memory()

def verify_index(index, filepath):
    """Verify that a FAISS index is valid"""
    try:
        # Basic sanity checks
        if index.ntotal == 0:
            return False
            
        # Try a simple search operation
        dummy_query = np.zeros((1, index.d), dtype=np.float32)
        index.search(dummy_query, k=1)
        
        return True
    except Exception as e:
        print(f"Index verification failed: {e}")
        return False
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--knowledge_path', type=str, required=True)
    parser.add_argument('--dense_encoder', type=str, default='facebook/contriever-msmarco')
    parser.add_argument('--theme_encoder_path', type=str, required=True)
    parser.add_argument('--theme_shifter_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()

    # Set up device and load models
    device = get_device()
    print(f"Using device: {device}")
    models = load_models(args)
    
    # Load and prepare data
    input_data = load_file(args.query_file)
    all_batches = [input_data[i:i + args.batch_size] 
                  for i in range(0, len(input_data), args.batch_size)]
    
    # Process all splits using ThreadPoolExecutor
    all_results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_split = {
            executor.submit(process_split, split_idx, args.knowledge_path, 
                          models, all_batches, device): split_idx
            for split_idx in range(args.num_splits)
        }
        
        for future in tqdm(as_completed(future_to_split), 
                         total=len(future_to_split),
                         desc="Processing splits"):
            try:
                split_results = future.result()
                with results_lock:
                    all_results.extend(split_results)
            except Exception as e:
                print(f"Split {future_to_split[future]} failed: {str(e)}")
    
    # Sort and save final results
    all_results = [convert_to_serializable(result) for result in all_results]
    all_results.sort(key=lambda x: max(doc[1] for doc in x['ctxs_theme']), reverse=True)
    save_file_jsonl(all_results, args.output_file)
    
    # Final cleanup
    clear_memory()

if __name__ == "__main__":
    # Initialize multiprocessing with spawn method for better compatibility
    mp.set_start_method('spawn', force=True)
    main()
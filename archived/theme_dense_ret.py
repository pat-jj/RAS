import argparse
import json
import faiss
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import MultilabelClassifier, load_theme_distribution_shifter, load_theme_classifier, load_file, save_file_jsonl
from copy import deepcopy
import orjson
import gc


def clear_memory():
    """Safely clear unused memory without affecting crucial data
    
    This function only removes unused cache and garbage collects unreferenced objects.
    It will not affect:
    - Active model weights and parameters
    - Data currently being processed
    - Loaded FAISS indices
    - Any variables currently in scope
    """
    try:
        # Only clear CUDA cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection to clean up unreferenced objects
        gc.collect()
        
        # Additional collection for harder to reach cycles
        # This won't affect objects still in use
        gc.collect()
    except Exception as e:
        print(f"Warning: Memory clearing encountered an error: {e}")
        # Continue execution even if memory clearing fails
        pass


def load_knowledge(knowledge_path, split_idx):
    """Load FAISS indices and text mappings for a specific split"""
    print(f"Loading indices and mappings for split {split_idx}...")
    
    clear_memory()
    
    # Load text mappings for the split
    print("Loading text mappings ...")
    with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'rb') as f:
        text_mapping = orjson.loads(f.read())
    
    # Create index mappings (since text mappings are identical, indices will match 1:1)
    print("Creating index mappings...")
    idx_mapping = {i: i for i in range(len(text_mapping))}
    
    # Load FAISS indices with memory-efficient settings
    print("Loading FAISS indices...")
    theme_faiss_index = faiss.read_index(f"{knowledge_path}/theme_dist/theme_embeddings_{split_idx}.faiss")
    dense_faiss_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
    
    # Enable FAISS optimizations
    if hasattr(theme_faiss_index, 'nprobe'):
        theme_faiss_index.nprobe = 128
    if hasattr(dense_faiss_index, 'nprobe'):
        dense_faiss_index.nprobe = 256
    
    return (text_mapping, dense_faiss_index, 
            text_mapping, theme_faiss_index,
            idx_mapping, idx_mapping)


def load_models(args):
    """Load necessary models and ensure they're on CPU"""
    clear_memory()
    
    # Dense encoder
    dense_encoder_tokenizer = AutoTokenizer.from_pretrained(args.dense_encoder)
    dense_encoder_model = AutoModel.from_pretrained(args.dense_encoder).cpu()
    
    # Theme models
    theme_classifier, theme_encoder, theme_label_mapping = load_theme_classifier(args.theme_encoder_path)
    theme_classifier = theme_classifier.cpu()
    theme_encoder = theme_encoder.cpu()
    theme_shifter = load_theme_distribution_shifter(args.theme_shifter_path, input_dim=len(theme_label_mapping))
    theme_shifter = theme_shifter.cpu()
    
    clear_memory()
    
    return (dense_encoder_tokenizer, dense_encoder_model,
            theme_classifier, theme_encoder, theme_label_mapping, theme_shifter)


def retrieve(query, models, knowledge_index, theme_top_k=50000, dense_top_k=100):
    """Retrieval with compatibility for older FAISS versions"""
    (dense_encoder_tokenizer, dense_encoder_model,
     theme_classifier, theme_encoder, theme_label_mapping, theme_shifter) = models
    
    (dense_text_mapping, dense_faiss_index,
     theme_text_mapping, theme_faiss_index,
     theme_to_dense_idx, dense_to_theme_idx) = knowledge_index

    
    # Get theme distribution
    query_theme_embedding = theme_encoder.encode(query, convert_to_tensor=True).cpu()
    
    with torch.no_grad():
        query_theme_probs = theme_classifier(query_theme_embedding.unsqueeze(0))
        predicted_theme_distribution = theme_shifter(query_theme_probs)
        theme_norm = torch.norm(predicted_theme_distribution, p=2, dim=1)
        
        if theme_norm < 1e-8:
            print(f"Warning: Zero theme distribution detected for query: {query[:100]}...")
            result = dense_only_retrieve(query, dense_encoder_tokenizer, dense_encoder_model, 
                                      dense_faiss_index, dense_text_mapping)
            clear_memory()
            return result
            
        predicted_theme_distribution = predicted_theme_distribution / theme_norm.unsqueeze(1)
    
    # Theme-based retrieval
    theme_scores, theme_doc_ids = theme_faiss_index.search(
        predicted_theme_distribution.cpu().numpy(), k=theme_top_k)
    
    # Convert theme doc ids to dense doc ids
    dense_candidate_ids = np.array([
        theme_to_dense_idx[int(idx)] 
        for idx in theme_doc_ids[0] 
        if int(idx) in theme_to_dense_idx
    ], dtype=np.int64)
    
    # Dense retrieval
    query_inputs = dense_encoder_tokenizer(
        query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_dense_embedding = dense_encoder_model(**query_inputs).last_hidden_state[:, 0].cpu().numpy()
        query_dense_embedding = query_dense_embedding / np.linalg.norm(query_dense_embedding)
    
    # Create a temporary index with only the candidate vectors
    temp_index = faiss.IndexFlatIP(dense_faiss_index.d)
    candidate_vectors = dense_faiss_index.reconstruct_batch(dense_candidate_ids)
    temp_index.add(candidate_vectors)
    
    # Search in the temporary index
    dense_scores, temp_doc_ids = temp_index.search(query_dense_embedding, k=min(dense_top_k, len(dense_candidate_ids)))
    
    # Map back to original dense indices
    dense_doc_ids = dense_candidate_ids[temp_doc_ids[0]]
    
    # Get theme scores
    theme_id_to_score = dict(zip(theme_doc_ids[0], theme_scores[0]))
    selected_theme_scores = np.array([
        theme_id_to_score[dense_to_theme_idx[doc_id]]
        for doc_id in dense_doc_ids
    ])
    
    # Combine scores and get top 10
    final_scores = 0.8 * dense_scores[0] + 0.2 * selected_theme_scores
    top_k = 10
    top_indices = np.argpartition(-final_scores, top_k)[:top_k]
    top_indices = top_indices[np.argsort(-final_scores[top_indices])]
    
    result = [(dense_text_mapping[dense_doc_ids[i]], final_scores[i]) 
              for i in top_indices]
    
    # Clean up temporary objects
    del temp_index, candidate_vectors, dense_scores, temp_doc_ids
    clear_memory()
    
    return result


def dense_only_retrieve(query, tokenizer, model, dense_index, text_mapping, top_k=30):
    """Fallback function for pure dense retrieval"""
    query_inputs = tokenizer(
        query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_dense_embedding = model(**query_inputs).last_hidden_state[:, 0].cpu().numpy()
        query_dense_embedding = query_dense_embedding / np.linalg.norm(query_dense_embedding)
    
    dense_scores, dense_doc_ids = dense_index.search(query_dense_embedding, k=top_k)
    
    result = [(text_mapping[doc_id], float(score)) 
              for doc_id, score in zip(dense_doc_ids[0], dense_scores[0])]
    
    clear_memory()
    return result


def retrieve_from_splits(query, models, knowledge_path, num_splits=5, theme_top_k=50000, dense_top_k=100):
    """Retrieve results from all splits and combine them"""
    all_results = []
    
    for split_idx in tqdm(range(num_splits), desc="Processing splits", position=0):
        print(f"\nProcessing split {split_idx}/{num_splits-1}")
        # Load knowledge index for this split
        knowledge_index = load_knowledge(knowledge_path, split_idx)
        
        # Get results from this split
        split_results = retrieve(query, models, knowledge_index, 
                               theme_top_k=theme_top_k, 
                               dense_top_k=dense_top_k)
        all_results.extend(split_results)
        
        # Clean up split data
        del knowledge_index
        clear_memory()
    
    # Sort all results by score and take top 10
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:10]


def process_batch(batch_items, models, knowledge_path, num_splits=5):
    """Process a batch of queries with memory optimization"""
    results = []
    for item in tqdm(batch_items, desc="Processing queries in batch", position=1):
        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]
        retrieved_docs = retrieve_from_splits(item["instruction"], models, 
                                           knowledge_path, num_splits)
        item['ctxs_theme'] = retrieved_docs
        results.append(item)
        clear_memory()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, required=True,
                      help='Path to file containing queries (one per line)')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to save retrieval results')
    parser.add_argument('--knowledge_path', type=str, required=True,
                      help='Path to knowledge base indices')
    parser.add_argument('--dense_encoder', type=str, 
                      default='facebook/contriever-msmarco')
    parser.add_argument('--theme_encoder_path', type=str, required=True,
                      help='Path to theme encoder model')
    parser.add_argument('--theme_shifter_path', type=str, required=True,
                      help='Path to theme distribution shifter model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Number of queries to process in parallel')
    parser.add_argument('--num_splits', type=int, default=5,
                      help='Number of index splits to process')
    args = parser.parse_args()

    # Initial memory cleanup
    clear_memory()

    # Load only models initially (indices will be loaded per split)
    print("Loading models...")
    models = load_models(args)

    # Load queries
    input_data = load_file(args.query_file)
    
    # Process queries in batches
    print("Processing queries...")
    batch_size = args.batch_size
    processed_items = []
    
    for i in tqdm(range(0, len(input_data), batch_size), desc="Processing batches", position=2):
        batch = input_data[i:i + batch_size]
        batch_results = process_batch(batch, models, args.knowledge_path, args.num_splits)
        processed_items.extend(batch_results)
        clear_memory()

    # Save results
    print(f"Saving results to {args.output_file}")
    save_file_jsonl(processed_items, args.output_file)

    # Final cleanup
    clear_memory()


if __name__ == "__main__":
    main()
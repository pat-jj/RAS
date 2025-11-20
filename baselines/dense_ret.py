import argparse
import json
import faiss
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from utils import load_file, save_file_jsonl
from copy import deepcopy
import orjson
import gc


def clear_memory():
    """Safely clear unused memory without affecting crucial data"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
    except Exception as e:
        print(f"Warning: Memory clearing encountered an error: {e}")
        pass


def load_knowledge(knowledge_path, split_idx):
    """Load FAISS indices and text mappings for a specific split"""
    print(f"Loading indices and mappings for split {split_idx}...")
    
    clear_memory()
    
    # Load text mappings for the split
    print("Loading text mappings ...")
    with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'rb') as f:
        text_mapping = orjson.loads(f.read())
    
    # Load FAISS indices with memory-efficient settings
    print("Loading FAISS indices...")
    dense_faiss_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
    
    # Enable FAISS optimizations
    if hasattr(dense_faiss_index, 'nprobe'):
        dense_faiss_index.nprobe = 256
    
    return text_mapping, dense_faiss_index


def load_models(args):
    """Load necessary models and ensure they're on CPU"""
    clear_memory()
    
    # Dense encoder
    dense_encoder_tokenizer = AutoTokenizer.from_pretrained(args.dense_encoder)
    dense_encoder_model = AutoModel.from_pretrained(args.dense_encoder).cpu()
    
    clear_memory()
    
    return dense_encoder_tokenizer, dense_encoder_model


def dense_retrieve(query, tokenizer, model, dense_index, text_mapping, top_k=30):
    """Dense retrieval function"""
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


def retrieve_from_splits(query, models, knowledge_path, num_splits=5, dense_top_k=100):
    """Retrieve results from all splits and combine them"""
    all_results = []
    
    for split_idx in tqdm(range(num_splits), desc="Processing splits", position=0):
        print(f"\nProcessing split {split_idx}/{num_splits-1}")
        # Load knowledge index for this split
        text_mapping, dense_faiss_index = load_knowledge(knowledge_path, split_idx)
        
        tokenizer, model = models
        
        # Get results from this split
        split_results = dense_retrieve(query, tokenizer, model, dense_faiss_index, 
                                      text_mapping, top_k=dense_top_k)
        all_results.extend(split_results)
        
        # Clean up split data
        del text_mapping, dense_faiss_index
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
        item['ctxs'] = retrieved_docs
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


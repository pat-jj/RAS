import faiss
import numpy as np
import orjson
from tqdm import tqdm
import gc
import torch

def clear_memory():
    """Safely clear unused memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
    except Exception as e:
        print(f"Warning: Memory clearing encountered an error: {e}")
        pass

def split_dense_index(knowledge_path, num_splits=5):
    """Split dense index into equal parts and save them"""
    print("Step 1: Loading dense index and mapping...")
    
    # Load dense index and mapping
    dense_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings.faiss")
    with open(f"{knowledge_path}/embedding/text_mapping.json", 'rb') as f:
        dense_text_mapping = orjson.loads(f.read())
    
    # Convert mapping to list and shuffle
    texts = list(dense_text_mapping)
    np.random.shuffle(texts)
    
    # Split texts into chunks
    chunk_size = len(texts) // num_splits
    text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    # Handle any remainder to ensure exactly num_splits chunks
    if len(text_chunks) > num_splits:
        text_chunks[-2].extend(text_chunks[-1])
        text_chunks.pop()
    
    # Create lookup dictionary for efficient index finding
    dense_text_to_idx = {text: idx for idx, text in enumerate(dense_text_mapping)}
    
    print(f"Processing {num_splits} splits...")
    
    for split_idx, split_texts in enumerate(text_chunks):
        print(f"\nProcessing split {split_idx + 1}/{num_splits}")
        
        # Get indices for this split
        dense_indices = [dense_text_to_idx[text] for text in split_texts]
        
        # Create new index
        new_dense_index = faiss.IndexFlatIP(dense_index.d)
        dense_vectors = dense_index.reconstruct_batch(dense_indices)
        new_dense_index.add(dense_vectors)
        
        # Save new index and mapping
        print(f"Saving split {split_idx + 1}...")
        faiss.write_index(new_dense_index, 
                         f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
        with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'wb') as f:
            f.write(orjson.dumps(split_texts))
        
        print(f"Split {split_idx + 1} complete: {len(split_texts)} documents")
        del new_dense_index, dense_vectors
        clear_memory()
    
    print("All splits completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_path', type=str, required=True,
                      help='Path to knowledge base indices')
    parser.add_argument('--num_splits', type=int, default=5,
                      help='Number of splits to create')
    args = parser.parse_args()
    
    split_dense_index(args.knowledge_path, args.num_splits)
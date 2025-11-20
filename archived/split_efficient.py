import faiss
import numpy as np
import orjson
from tqdm import tqdm
import gc
from collections import OrderedDict

def clear_memory():
    gc.collect()
    gc.collect()

def prepare_split_mappings(knowledge_path, num_splits):
    """Prepare text mappings and verify alignment before processing vectors"""
    print("Loading text mappings...")
    with open(f"{knowledge_path}/theme_dist/text_mapping.json", 'rb') as f:
        theme_text_mapping = orjson.loads(f.read())
    with open(f"{knowledge_path}/embedding/text_mapping.json", 'rb') as f:
        dense_text_mapping = orjson.loads(f.read())
        
    # Create lookup dictionaries
    theme_text_to_idx = {text: idx for idx, text in enumerate(theme_text_mapping)}
    dense_text_to_idx = {text: idx for idx, text in enumerate(dense_text_mapping)}
    
    print("Processing theme index to find valid vectors...")
    theme_index = faiss.read_index(f"{knowledge_path}/theme_dist/theme_embeddings.faiss")
    
    # Process theme vectors in batches to find valid ones
    batch_size = 100000
    valid_theme_texts = set()
    
    for start_idx in tqdm(range(0, len(theme_text_mapping), batch_size)):
        end_idx = min(start_idx + batch_size, len(theme_text_mapping))
        batch_vectors = theme_index.reconstruct_batch(range(start_idx, end_idx))
        norms = np.linalg.norm(batch_vectors, axis=1)
        valid_indices = [i + start_idx for i, norm in enumerate(norms) if norm > 1e-8]
        valid_theme_texts.update(theme_text_mapping[i] for i in valid_indices)
    
    del theme_index
    clear_memory()
    
    # Find common valid texts
    common_texts = list(set(dense_text_mapping) & set(theme_text_mapping) & valid_theme_texts)
    np.random.shuffle(common_texts)
    
    # Split into chunks
    chunk_size = len(common_texts) // num_splits
    text_chunks = [common_texts[i:i + chunk_size] for i in range(0, len(common_texts), chunk_size)]
    if len(text_chunks) > num_splits:
        text_chunks[-2].extend(text_chunks[-1])
        text_chunks.pop()
    
    # Prepare split mappings
    split_mappings = []
    for split_idx, texts in enumerate(text_chunks):
        # Create ordered mappings for this split
        split_map = {
            'texts': texts,
            'dense_indices': [dense_text_to_idx[text] for text in texts],
            'theme_indices': [theme_text_to_idx[text] for text in texts]
        }
        split_mappings.append(split_map)
        
        # Verify the mapping lengths match
        assert len(split_map['texts']) == len(split_map['dense_indices']) == len(split_map['theme_indices']), \
            f"Mapping length mismatch in split {split_idx}"
    
    print(f"Prepared {len(split_mappings)} splits with perfect alignment")
    return split_mappings

def process_split(knowledge_path, split_idx, split_map, batch_size=100000):
    """Process one split for both indices with guaranteed alignment"""
    texts = split_map['texts']
    dense_indices = split_map['dense_indices']
    theme_indices = split_map['theme_indices']
    
    # Process dense index
    print(f"\nProcessing dense index for split {split_idx}...")
    dense_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings.faiss")
    new_dense_index = faiss.IndexFlatIP(dense_index.d)
    
    for start_idx in tqdm(range(0, len(dense_indices), batch_size)):
        end_idx = min(start_idx + batch_size, len(dense_indices))
        batch_indices = dense_indices[start_idx:end_idx]
        batch_vectors = dense_index.reconstruct_batch(batch_indices)
        new_dense_index.add(batch_vectors)
    
    faiss.write_index(new_dense_index, 
                     f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
    del dense_index, new_dense_index
    clear_memory()
    
    # Process theme index
    print(f"Processing theme index for split {split_idx}...")
    theme_index = faiss.read_index(f"{knowledge_path}/theme_dist/theme_embeddings.faiss")
    new_theme_index = faiss.IndexFlatIP(theme_index.d)
    
    for start_idx in tqdm(range(0, len(theme_indices), batch_size)):
        end_idx = min(start_idx + batch_size, len(theme_indices))
        batch_indices = theme_indices[start_idx:end_idx]
        batch_vectors = theme_index.reconstruct_batch(batch_indices)
        new_theme_index.add(batch_vectors)
    
    faiss.write_index(new_theme_index, 
                     f"{knowledge_path}/theme_dist/theme_embeddings_{split_idx}.faiss")
    del theme_index, new_theme_index
    clear_memory()
    
    # Save identical text mappings
    with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'wb') as f:
        f.write(orjson.dumps(texts))
    with open(f"{knowledge_path}/theme_dist/text_mapping_{split_idx}.json", 'wb') as f:
        f.write(orjson.dumps(texts))
    
    # Verify the saved indices
    print("Verifying split alignment...")
    new_dense = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
    new_theme = faiss.read_index(f"{knowledge_path}/theme_dist/theme_embeddings_{split_idx}.faiss")
    assert new_dense.ntotal == new_theme.ntotal == len(texts), \
        f"Size mismatch in split {split_idx}"
    del new_dense, new_theme
    clear_memory()
    
    print(f"Split {split_idx} complete and verified: {len(texts)} documents")

def split_and_save_indices(knowledge_path, num_splits=5, batch_size=100000):
    """Main function to split indices with guaranteed alignment"""
    # First prepare all split mappings
    split_mappings = prepare_split_mappings(knowledge_path, num_splits)
    
    # Process each split
    for split_idx, split_map in enumerate(split_mappings):
        process_split(knowledge_path, split_idx, split_map, batch_size)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_path', type=str, required=True,
                      help='Path to knowledge base indices')
    parser.add_argument('--num_splits', type=int, default=5,
                      help='Number of splits to create')
    parser.add_argument('--batch_size', type=int, default=100000,
                      help='Batch size for processing vectors')
    args = parser.parse_args()
    
    split_and_save_indices(args.knowledge_path, args.num_splits, args.batch_size)
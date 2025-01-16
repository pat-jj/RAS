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

def load_and_filter_indices(knowledge_path):
    """Load indices and filter out zero vectors from theme index"""
    print("Loading original indices and mappings...")
    
    # Load text mappings
    with open(f"{knowledge_path}/embedding/text_mapping.json", 'rb') as f:
        dense_text_mapping = orjson.loads(f.read())
    with open(f"{knowledge_path}/theme_dist/text_mapping.json", 'rb') as f:
        theme_text_mapping = orjson.loads(f.read())
    
    print(f"Original sizes - Dense: {len(dense_text_mapping)}, Theme: {len(theme_text_mapping)}")
    
    # Load FAISS indices
    theme_index = faiss.read_index(f"{knowledge_path}/theme_dist/theme_embeddings.faiss")
    dense_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings.faiss")
    
    # Get vectors from theme index
    theme_vectors = faiss.vector_to_array(theme_index.reconstruct_n(0, theme_index.ntotal))
    theme_vectors = theme_vectors.reshape(theme_index.ntotal, theme_index.d)
    
    # Find non-zero vectors
    norms = np.linalg.norm(theme_vectors, axis=1)
    valid_indices = norms > 1e-8
    
    # Create dictionaries for efficient lookup
    valid_theme_texts = {theme_text_mapping[i] for i in range(len(theme_text_mapping)) 
                        if valid_indices[i]}
    
    # Find common texts between indices that also have non-zero theme vectors
    common_texts = set(dense_text_mapping) & set(theme_text_mapping) & valid_theme_texts
    
    print(f"Common texts after filtering: {len(common_texts)}")
    return dense_index, theme_index, dense_text_mapping, theme_text_mapping, common_texts

# def split_and_save_indices(knowledge_path, num_splits=5):
#     """Split indices into equal parts and save them"""
#     dense_index, theme_index, dense_text_mapping, theme_text_mapping, common_texts = \
#         load_and_filter_indices(knowledge_path)
    
#     # Convert common_texts to list and shuffle
#     common_texts = list(common_texts)
#     np.random.shuffle(common_texts)
    
#     # Split texts into chunks
#     chunk_size = len(common_texts) // num_splits
#     text_chunks = [common_texts[i:i + chunk_size] for i in range(0, len(common_texts), chunk_size)]
    
#     # Ensure we have exactly num_splits chunks
#     if len(text_chunks) > num_splits:
#         text_chunks[-2].extend(text_chunks[-1])
#         text_chunks.pop()
    
#     # Create lookup dictionaries for efficient index finding
#     dense_text_to_idx = {text: idx for idx, text in enumerate(dense_text_mapping)}
#     theme_text_to_idx = {text: idx for idx, text in enumerate(theme_text_mapping)}
    
#     print(f"Processing {num_splits} splits...")
    
#     for split_idx, texts in enumerate(text_chunks):
#         print(f"\nProcessing split {split_idx + 1}/{num_splits}")
        
#         # Get indices for both dense and theme
#         dense_indices = []
#         theme_indices = []
        
#         # Verify each text exists in both mappings
#         for text in tqdm(texts, desc="Collecting indices"):
#             dense_idx = dense_text_to_idx[text]
#             theme_idx = theme_text_to_idx[text]
            
#             dense_indices.append(dense_idx)
#             theme_indices.append(theme_idx)
        
#         # Create new indices
#         print("Creating new dense index...")
#         new_dense_index = faiss.IndexFlatIP(dense_index.d)
#         dense_vectors = dense_index.reconstruct_batch(dense_indices)
#         new_dense_index.add(dense_vectors)
        
#         print("Creating new theme index...")
#         new_theme_index = faiss.IndexFlatIP(theme_index.d)
#         theme_vectors = theme_index.reconstruct_batch(theme_indices)
#         new_theme_index.add(theme_vectors)
        
#         # Verify sizes match
#         assert len(texts) == new_dense_index.ntotal == new_theme_index.ntotal, \
#             "Mismatch in number of vectors and texts"
        
#         # Save new indices and mappings
#         print("Saving split files...")
#         faiss.write_index(new_dense_index, 
#                          f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
#         faiss.write_index(new_theme_index, 
#                          f"{knowledge_path}/theme_dist/theme_embeddings_{split_idx}.faiss")
        
#         # Save exactly the same text mapping for both indices
#         with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'wb') as f:
#             f.write(orjson.dumps(texts))
#         with open(f"{knowledge_path}/theme_dist/text_mapping_{split_idx}.json", 'wb') as f:
#             f.write(orjson.dumps(texts))
        
#         print(f"Split {split_idx + 1} complete: {len(texts)} documents")
#         print("Verifying text mappings match...")
#         # Verify saved mappings
#         with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'rb') as f:
#             dense_texts = orjson.loads(f.read())
#         with open(f"{knowledge_path}/theme_dist/text_mapping_{split_idx}.json", 'rb') as f:
#             theme_texts = orjson.loads(f.read())
#         assert dense_texts == theme_texts, "Text mappings don't match!"
#         print(f"✓ Text mappings verified for split {split_idx}")
        
#         clear_memory()
        
#         print(f"Split {split_idx + 1} complete: {len(texts)} documents")


# def split_and_save_indices(knowledge_path, num_splits=5):
#     """Split indices into equal parts and save them, processing one index at a time"""
#     print("Step 1: Loading and processing text mappings...")
#     with open(f"{knowledge_path}/embedding/text_mapping.json", 'rb') as f:
#         dense_text_mapping = orjson.loads(f.read())
#     with open(f"{knowledge_path}/theme_dist/text_mapping.json", 'rb') as f:
#         theme_text_mapping = orjson.loads(f.read())
    
#     print("Step 2: Loading theme index and filtering zero vectors...")
#     theme_index = faiss.read_index(f"{knowledge_path}/theme_dist/theme_embeddings.faiss")
    
#     # Get vectors one by one to avoid memory issues
#     print("Extracting theme vectors...")
#     theme_vectors = []
#     for i in tqdm(range(theme_index.ntotal)):
#         vec = theme_index.reconstruct(i)
#         theme_vectors.append(vec)
#     theme_vectors = np.array(theme_vectors)
    
#     # Find non-zero vectors
#     norms = np.linalg.norm(theme_vectors, axis=1)
#     valid_indices = norms > 1e-8
#     valid_theme_texts = {theme_text_mapping[i] for i in range(len(theme_text_mapping)) 
#                         if valid_indices[i]}
    
#     # Free memory
#     del theme_vectors
#     clear_memory()
    
#     common_texts = set(dense_text_mapping) & set(theme_text_mapping) & valid_theme_texts
#     common_texts = list(common_texts)
#     np.random.shuffle(common_texts)
    
#     chunk_size = len(common_texts) // num_splits
#     text_chunks = [common_texts[i:i + chunk_size] for i in range(0, len(common_texts), chunk_size)]
#     if len(text_chunks) > num_splits:
#         text_chunks[-2].extend(text_chunks[-1])
#         text_chunks.pop()
    
#     print("Step 3: Processing theme index splits...")
#     theme_text_to_idx = {text: idx for idx, text in enumerate(theme_text_mapping)}
    
#     for split_idx, texts in enumerate(text_chunks):
#         theme_indices = [theme_text_to_idx[text] for text in texts]
        
#         new_theme_index = faiss.IndexFlatIP(theme_index.d)
#         theme_vectors = theme_index.reconstruct_batch(theme_indices)
#         new_theme_index.add(theme_vectors)
        
#         faiss.write_index(new_theme_index, 
#                          f"{knowledge_path}/theme_dist/theme_embeddings_{split_idx}.faiss")
#         with open(f"{knowledge_path}/theme_dist/text_mapping_{split_idx}.json", 'wb') as f:
#             f.write(orjson.dumps(texts))
        
#         del new_theme_index, theme_vectors
#         clear_memory()
    
#     # Free theme index memory
#     del theme_index
#     clear_memory()
    
#     print("Step 4: Loading dense index and processing splits...")
#     dense_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings.faiss")
#     dense_text_to_idx = {text: idx for idx, text in enumerate(dense_text_mapping)}
    
#     for split_idx, texts in enumerate(text_chunks):
#         dense_indices = [dense_text_to_idx[text] for text in texts]
        
#         new_dense_index = faiss.IndexFlatIP(dense_index.d)
#         dense_vectors = dense_index.reconstruct_batch(dense_indices)
#         new_dense_index.add(dense_vectors)
        
#         faiss.write_index(new_dense_index, 
#                          f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
#         with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'wb') as f:
#             f.write(orjson.dumps(texts))
        
#         # Verify text mappings match
#         with open(f"{knowledge_path}/theme_dist/text_mapping_{split_idx}.json", 'rb') as f:
#             theme_texts = orjson.loads(f.read())
#         assert texts == theme_texts, f"Text mapping mismatch in split {split_idx}"
        
#         del new_dense_index, dense_vectors
#         clear_memory()
    
#     print("All splits completed successfully!")
def split_and_save_indices(knowledge_path, num_splits=5):
    """Continue with dense index splitting using existing theme splits"""
    print("Step 1: Loading text chunks from existing theme splits...")
    text_chunks = []
    for split_idx in range(num_splits):
        with open(f"{knowledge_path}/theme_dist/text_mapping_{split_idx}.json", 'rb') as f:
            texts = orjson.loads(f.read())
            text_chunks.append(texts)
            print(f"Found theme split {split_idx} with {len(texts)} documents")
    
    print("Step 2: Loading dense index and processing splits...")
    dense_index = faiss.read_index(f"{knowledge_path}/embedding/wikipedia_embeddings.faiss")
    with open(f"{knowledge_path}/embedding/text_mapping.json", 'rb') as f:
        dense_text_mapping = orjson.loads(f.read())
    dense_text_to_idx = {text: idx for idx, text in enumerate(dense_text_mapping)}
    
    for split_idx, texts in enumerate(text_chunks):
        print(f"\nProcessing dense split {split_idx + 1}/{num_splits}")
        dense_indices = [dense_text_to_idx[text] for text in texts]
        
        new_dense_index = faiss.IndexFlatIP(dense_index.d)
        dense_vectors = dense_index.reconstruct_batch(dense_indices)
        new_dense_index.add(dense_vectors)
        
        faiss.write_index(new_dense_index, 
                         f"{knowledge_path}/embedding/wikipedia_embeddings_{split_idx}.faiss")
        with open(f"{knowledge_path}/embedding/text_mapping_{split_idx}.json", 'wb') as f:
            f.write(orjson.dumps(texts))
        
        print(f"Split {split_idx + 1} complete: {len(texts)} documents")
        del new_dense_index, dense_vectors
        clear_memory()
    
    print("Dense index splits completed successfully!")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_path', type=str, required=True,
                      help='Path to knowledge base indices')
    parser.add_argument('--num_splits', type=int, default=5,
                      help='Number of splits to create')
    args = parser.parse_args()
    
    split_and_save_indices(args.knowledge_path, args.num_splits)
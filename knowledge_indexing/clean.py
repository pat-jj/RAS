import orjson
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gc
import torch
import os
import shutil

VERSION = "2018"

def load_wiki_text(file_path):
    """Load wiki text data from jsonl file"""
    data = []
    with open(file_path, "r") as f:
        for line in tqdm(f):
            data.append(orjson.loads(line)["text"])
    return data

def save_json(data, file_path):
    """Save data to json file with error handling and disk space check"""
    try:
        # Check available disk space
        directory = os.path.dirname(file_path)
        free_space = shutil.disk_usage(directory).free
        
        # Estimate required space (rough estimation)
        estimated_size = len(str(data)) * 2  # rough estimation in bytes
        
        if estimated_size > free_space:
            print(f"Warning: Insufficient disk space. Need {estimated_size/1e9:.2f}GB, "
                  f"have {free_space/1e9:.2f}GB free")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Try to save the file
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(data))
        return True
        
    except OSError as e:
        print(f"Error saving file {file_path}: {e}")
        print("Available disk space:", shutil.disk_usage(directory).free / 1e9, "GB")
        return False

def create_index_mapping(new_texts, old_texts):
    """Create mapping between old and new indices"""
    text_to_new_index = {text: idx for idx, text in enumerate(new_texts)}
    old_to_new_indices = {}
    for old_idx, old_text in tqdm(enumerate(old_texts)):
        if old_text in text_to_new_index:
            old_to_new_indices[old_idx] = text_to_new_index[old_text]
    return old_to_new_indices

def create_new_vectors_and_mapping(old_to_new_indices, text_mapping, index, new_data_len, vector_dim):
    """Create new text mapping and vectors"""
    new_text_mapping = [None] * new_data_len
    new_vectors = np.zeros((new_data_len, vector_dim), dtype=np.float32)
    
    old_indices = list(old_to_new_indices.keys())
    print("Getting all vectors...")
    all_vectors = index.reconstruct_n(0, index.ntotal)  # Get all vectors at once
    
    print("Creating new mapping and vectors...")
    for j, old_idx in tqdm(enumerate(old_indices)):
        new_idx = old_to_new_indices[old_idx]
        new_text_mapping[new_idx] = text_mapping[old_idx]
        new_vectors[new_idx] = all_vectors[old_idx]
    
    del all_vectors
    clear_memory()
    return new_text_mapping, new_vectors

def split_and_save_index(base_path, new_vectors, new_text_mapping, num_splits=5):
    """Split the cleaned index into parts and save them"""
    print(f"Creating {num_splits} splits...")
    
    # Shuffle the data
    indices = np.arange(len(new_text_mapping))
    np.random.shuffle(indices)
    
    # Split into chunks
    chunk_size = len(indices) // num_splits
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
    
    # Handle remainder to ensure exactly num_splits chunks
    if len(chunks) > num_splits:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    
    # Process each split
    for split_idx, chunk_indices in enumerate(chunks):
        print(f"\nProcessing split {split_idx + 1}/{num_splits}")
        
        # Create split vectors and mapping
        split_vectors = new_vectors[chunk_indices]
        split_mapping = [new_text_mapping[i] for i in chunk_indices]
        
        # Create and save split index
        split_index = faiss.IndexFlatIP(new_vectors.shape[1])
        split_index.add(split_vectors)
        
        # Save split files
        print(f"Saving split {split_idx + 1}...")
        faiss.write_index(split_index, 
            str(base_path / f"embedding/wikipedia_embeddings_cleaned_{split_idx}.faiss"))
        with open(base_path / f"embedding/text_mapping_cleaned_{split_idx}.json", "wb") as f:
            f.write(orjson.dumps(split_mapping))
        
        print(f"Split {split_idx + 1} complete: {len(split_mapping)} documents")
        del split_index, split_vectors, split_mapping
        clear_memory()

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

def main():
    base_path = Path(f"/shared/eng/pj20/firas_data/knowledge_source/wiki_{VERSION}")
    
    print("Loading original wiki text data...")
    data = load_wiki_text(base_path / f"corpora/wiki/enwiki-dec{VERSION}/text-list-100-sec.jsonl")
    
    print("Saving cleaned wiki text...")
    save_json(data, base_path / "all_wiki_text_cleaned.json")
    # print("Loading cleaned wiki text...")
    # with open(base_path / "all_wiki_text_cleaned.json", "rb") as f:
    #     data = orjson.loads(f.read())
    
    print("Loading original text mapping and FAISS index...")
    with open(base_path / "embedding/text_mapping.json", "rb") as f:
        text_mapping = orjson.loads(f.read())
    
    index = faiss.read_index(str(base_path / "embedding/wikipedia_embeddings.faiss"))
    
    print("Creating index mapping...")
    with open(base_path / "all_wiki_text.json", "rb") as f:
        old_texts = orjson.loads(f.read())
    
    old_to_new_indices = create_index_mapping(data, old_texts)
    
    print("Creating new text mapping and vectors...")
    vector_dim = index.d
    new_text_mapping, new_vectors = create_new_vectors_and_mapping(
        old_to_new_indices, text_mapping, index, len(data), vector_dim
    )
    
    print("Saving new text mapping...")
    save_json(new_text_mapping, base_path / "embedding/text_mapping_cleaned.json")
    
    print("Saving new FAISS index...")
    new_index = faiss.IndexFlatIP(vector_dim)
    new_index.add(new_vectors)
    faiss.write_index(new_index, str(base_path / "embedding/wikipedia_embeddings_cleaned.faiss"))
    
    print("Creating splits...")
    split_and_save_index(base_path, new_vectors, new_text_mapping, num_splits=5)
    
    print("Done!")

if __name__ == "__main__":
    def run_tests():
        print("Running tests...")
        test_texts = ["text1", "text2", "text3"]
        test_old_texts = ["text1", "text4", "text2"]
        mapping = create_index_mapping(test_texts, test_old_texts)
        assert mapping == {0: 0, 2: 1}, "Index mapping test failed"
        print("Tests passed!")
    
    run_tests()
    main()
import orjson
from pathlib import Path
import faiss
import numpy as np
from tqdm import tqdm


VERSION = "2020"

def load_data(base_path):
    """Load all necessary data files"""
    base_path = Path(base_path)
    
    print("Loading text mappings...")
    with open(base_path / "embedding/text_mapping_old.json", "rb") as f:
        old_mapping = orjson.loads(f.read())
    with open(base_path / "embedding/text_mapping_cleaned.json", "rb") as f:
        new_mapping = orjson.loads(f.read())
        
    print("Loading original and cleaned texts...")
    with open(base_path / "all_wiki_text.json", "rb") as f:
        old_texts = orjson.loads(f.read())
    with open(base_path / "all_wiki_text_cleaned.json", "rb") as f:
        new_texts = orjson.loads(f.read())
        
    print("Loading FAISS indices...")
    old_index = faiss.read_index(str(base_path / "embedding/wikipedia_embeddings_old.faiss"))
    new_index = faiss.read_index(str(base_path / "embedding/wikipedia_embeddings_cleaned.faiss"))
    
    return {
        'old_mapping': old_mapping,
        'new_mapping': new_mapping,
        'old_texts': old_texts,
        'new_texts': new_texts,
        'old_index': old_index,
        'new_index': new_index
    }

def verify_mappings(data):
    """Verify the consistency of mappings and texts"""
    print("\n=== Verification Results ===")
    
    # Basic counts
    print("\nCounts:")
    print(f"Old texts: {len(data['old_texts'])}")
    print(f"New texts: {len(data['new_texts'])}")
    print(f"Old mapping: {len(data['old_mapping'])}")
    print(f"New mapping: {len(data['new_mapping'])}")
    print(f"Old index size: {data['old_index'].ntotal}")
    print(f"New index size: {data['new_index'].ntotal}")
    
    # Sample verification
    print("\nSampling 5 random texts to verify consistency...")
    sample_indices = np.random.choice(len(data['new_mapping']), 5, replace=False)
    
    for idx in sample_indices:
        new_text = data['new_texts'][idx]
        new_mapping = data['new_mapping'][idx]
        
        # Find this text in old data
        try:
            old_idx = data['old_texts'].index(new_text)
            old_mapping = data['old_mapping'][old_idx]
            
            print(f"\nSample {idx}:")
            print(f"Text match: {'YES' if new_text == data['old_texts'][old_idx] else 'NO'}")
            print(f"Mapping match: {'YES' if new_mapping == old_mapping else 'NO'}")
            
            # Compare embeddings
            old_vector = data['old_index'].reconstruct(old_idx)
            new_vector = data['new_index'].reconstruct(idx)
            similarity = np.dot(old_vector, new_vector)
            print(f"Embedding similarity: {similarity:.4f}")
            
            # Print first 100 chars of text
            print(f"Text preview: {new_text[:100]}...")
            
        except ValueError:
            print(f"\nSample {idx}: Text not found in old dataset!")

def main():
    base_path = f"/shared/eng/pj20/firas_data/knowledge_source/wiki_{VERSION}"
    data = load_data(base_path)
    verify_mappings(data)

if __name__ == "__main__":
    main()
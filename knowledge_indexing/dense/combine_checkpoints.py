import faiss
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict

def setup_logging(output_dir):
    """Setup logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'combine_checkpoints.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_checkpoint(checkpoint_path, logger):
    """Load embeddings and texts from a checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load FAISS index
    index_path = checkpoint_path.with_suffix('.faiss')
    index = faiss.read_index(str(index_path))
    
    # Extract embeddings
    n_vectors = index.ntotal
    dimension = index.d
    embeddings = np.zeros((n_vectors, dimension), dtype=np.float32)
    index.reconstruct_n(0, n_vectors, embeddings)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Load corresponding texts
    text_path = checkpoint_path.with_suffix('.json')
    with open(text_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    logger.info(f"Loaded {len(texts)} texts")
    
    return embeddings, texts

def get_text_to_indices_mapping(texts, logger):
    """Create a mapping from text content to list of indices where it appears"""
    text_to_indices = defaultdict(list)
    for idx, text in enumerate(texts):
        text_to_indices[text].append(idx)
    
    # Log duplicate statistics
    duplicate_counts = [len(indices) for indices in text_to_indices.values()]
    max_duplicates = max(duplicate_counts)
    avg_duplicates = sum(duplicate_counts) / len(duplicate_counts)
    logger.info(f"Text statistics:")
    logger.info(f"  Total unique texts: {len(text_to_indices)}")
    logger.info(f"  Maximum duplicates for a text: {max_duplicates}")
    logger.info(f"  Average duplicates per text: {avg_duplicates:.2f}")
    
    return text_to_indices

def combine_checkpoints(checkpoint_dir, output_dir, input_texts_path, logger):
    """Combine all checkpoints"""
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original input texts
    logger.info(f"Loading original input texts from {input_texts_path}")
    with open(input_texts_path, 'r', encoding='utf-8') as f:
        original_texts = json.load(f)
    total_texts = len(original_texts)
    logger.info(f"Total texts in original file: {total_texts}")
    
    # Create mapping of text content to indices in original file
    text_to_indices = get_text_to_indices_mapping(original_texts, logger)
    
    # Find and sort checkpoints
    def get_checkpoint_number(path):
        name = path.stem
        num = int(name.split('_')[1])
        return num
    
    checkpoints = sorted(
        list(checkpoint_dir.glob("checkpoint_*_of_*.faiss")),
        key=get_checkpoint_number
    )
    checkpoint_bases = [cp.with_suffix('') for cp in checkpoints]
    logger.info("Processing checkpoints in order:")
    for cp in checkpoint_bases:
        logger.info(f"  {cp.name}")
    
    # Process checkpoints
    final_embeddings = []
    final_texts = []
    processed_unique_texts = set()
    
    for checkpoint_base in tqdm(checkpoint_bases, desc="Processing checkpoints"):
        embeddings, texts = load_checkpoint(checkpoint_base, logger)
        
        # Process each text in this checkpoint
        for idx, text in enumerate(texts):
            if text not in processed_unique_texts:
                processed_unique_texts.add(text)
                # Get all positions where this text appears in original file
                original_positions = text_to_indices[text]
                # Replicate the embedding for each occurrence
                for _ in original_positions:
                    final_embeddings.append(embeddings[idx])
                    final_texts.append(text)
    
    # Convert to numpy array
    final_embeddings = np.array(final_embeddings)
    
    # Verify dimensions
    if len(final_texts) != total_texts:
        raise ValueError(f"Number of processed texts ({len(final_texts)}) "
                        f"doesn't match original texts ({total_texts})")
    
    # Create and save final FAISS index
    logger.info("Creating final FAISS index")
    dimension = final_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(final_embeddings)
    index.add(final_embeddings)
    
    # Save outputs
    logger.info("Saving final outputs")
    faiss.write_index(index, str(output_dir / "wikipedia_embeddings.faiss"))
    with open(output_dir / "text_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(final_texts, f, ensure_ascii=False, indent=2)
    
    logger.info("Combination complete!")
    logger.info(f"Final embeddings shape: {final_embeddings.shape}")
    logger.info(f"Final number of texts: {len(final_texts)}")

def main():
    parser = argparse.ArgumentParser(description='Combine checkpoint embeddings')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                      help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save combined outputs')
    parser.add_argument('--input_texts', type=str, required=True,
                      help='Path to original input texts JSON file')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    try:
        combine_checkpoints(args.checkpoint_dir, args.output_dir, args.input_texts, logger)
    except Exception as e:
        logger.error(f"Error combining checkpoints: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
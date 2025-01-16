import pandas as pd
import numpy as np
import faiss
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_csv_files(probabilities_csv, debug_rows=None):
    """
    Load theme probability distributions from CSV with proper text handling
    """
    logger.info("Reading CSV file...")
    
    # For debugging, use small chunk
    if debug_rows:
        logger.info(f"Debug mode: Loading only {debug_rows} rows")
        df = pd.read_csv(probabilities_csv, nrows=debug_rows)
        prob_columns = [col for col in df.columns if col != 'text']
        df[prob_columns] = df[prob_columns].astype(np.float32)
    else:
        # Get columns from first chunk
        chunk_iterator = pd.read_csv(probabilities_csv, chunksize=1)
        first_chunk = next(chunk_iterator)
        prob_columns = [col for col in first_chunk.columns if col != 'text']
        
        # Read in chunks with proper dtypes from the start
        dtype_dict = {col: np.float32 for col in prob_columns}
        dtype_dict['text'] = str
        
        chunks = []
        chunk_iterator = pd.read_csv(
            probabilities_csv,
            dtype=dtype_dict,
            chunksize=100000,  # 100K rows per chunk
            na_values=['nan', 'NaN', 'NULL'],
            keep_default_na=False
        )
        
        # Process chunks
        for chunk_num, chunk in enumerate(tqdm(chunk_iterator, desc="Loading CSV")):
            # Only handle invalid/NaN texts, no stripping
            chunk['text'] = chunk['text'].fillna('[INVALID_TEXT]')
            chunks.append(chunk)
        
        df = pd.concat(chunks, axis=0, copy=False)
        
    # Get embeddings and texts
    embeddings = np.ascontiguousarray(df[prob_columns].values)
    texts = df['text'].tolist()
    
    logger.info(f"Loaded {len(texts):,} samples with {len(prob_columns)} dimensions")
    return embeddings, texts

def convert_to_faiss(embeddings, texts=None, output_dir="faiss_index"):
    """
    Convert theme probability vectors to FAISS index with clean text mapping
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Checking vector normalization...")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-8):
        logger.info("Normalizing vectors...")
        # Handle zero-norm vectors
        zero_norms = norms == 0
        if zero_norms.any():
            logger.warning(f"Found {zero_norms.sum()} zero-norm vectors")
            # Replace zero norms with 1 to avoid division by zero
            norms[zero_norms] = 1.0
        embeddings = embeddings / norms.reshape(-1, 1)
        embeddings = np.ascontiguousarray(embeddings)
    
    dimension = embeddings.shape[1]
    logger.info("Creating FlatIP index for exact search...")
    index = faiss.IndexFlatIP(dimension)
    
    logger.info("Adding vectors to index...")
    batch_size = 1000000
    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch = embeddings[i:min(i + batch_size, len(embeddings))]
        index.add(batch)
    
    logger.info("Saving FAISS index...")
    faiss.write_index(index, str(output_dir / "theme_embeddings.faiss"))
    
    if texts:
        logger.info("Saving text mapping...")
        # Just convert to string, no stripping
        text_mapping = [
            '[INVALID_TEXT]' if pd.isna(text) else str(text)
            for text in texts
        ]
        
        text_mapping_file = output_dir / "text_mapping.json"
        # Save with proper JSON formatting
        with open(text_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(text_mapping, f, ensure_ascii=False, indent=2)
        
        # Verify the saved file
        try:
            with open(text_mapping_file, 'r', encoding='utf-8') as f:
                _ = json.load(f)
            logger.info("JSON format verified successfully")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to verify text mapping JSON format: {e}")
            raise
    
    metadata = {
        "num_vectors": len(embeddings),
        "dimension": dimension,
        "index_type": "flat",
        "target_retrieval_size": 50000,
        "has_text_mapping": texts is not None
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities_csv", type=str, required=True,
                      help="Path to CSV file containing theme probabilities")
    parser.add_argument("--output_dir", type=str, default="theme_index",
                      help="Directory to save FAISS index and mapping")
    parser.add_argument("--debug", action="store_true",
                      help="Debug mode: only process first 1000 rows")
    parser.add_argument("--debug_rows", type=int,
                      help="Number of rows to process in debug mode")
    parser.add_argument("--verify_rows", type=int, default=1000,
                      help="Number of rows to verify for alignment check")
    args = parser.parse_args()
    
    print("Converting CSV to FAISS...")
    print("Data source: ", args.probabilities_csv)
    
    try:
        debug_rows = 1000 if args.debug else args.debug_rows
        
        logger.info("Starting theme index conversion...")
        embeddings, texts = load_csv_files(args.probabilities_csv, debug_rows)
        
        logger.info("Converting to FAISS format...")
        convert_to_faiss(embeddings, texts, args.output_dir)
        
        logger.info(f"Conversion complete. Files saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
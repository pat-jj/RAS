import pandas as pd
import numpy as np
import faiss
import argparse
from pathlib import Path
import json
from tqdm import tqdm

def load_csv_files(embeddings_csv, texts_csv=None):
    """
    Load embeddings and texts from CSV files
    """
    # Load embeddings
    df_embeddings = pd.read_csv(embeddings_csv)
    embeddings = df_embeddings.values.astype(np.float32)
    
    # Load texts if provided
    texts = None
    if texts_csv:
        df_texts = pd.read_csv(texts_csv)
        texts = df_texts['text'].tolist()  # Adjust column name as needed
    
    return embeddings, texts

def convert_to_faiss(embeddings, texts=None, output_dir="faiss_index"):
    """
    Convert embeddings to FAISS index and save with optional text mapping
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "embeddings.faiss"))
    
    # Save text mapping if provided
    if texts:
        with open(output_dir / "text_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_csv", type=str, required=True,
                      help="Path to CSV file containing embeddings")
    parser.add_argument("--texts_csv", type=str,
                      help="Optional: Path to CSV file containing original texts")
    parser.add_argument("--output_dir", type=str, default="faiss_index",
                      help="Directory to save FAISS index and mapping")
    args = parser.parse_args()
    
    print("Loading CSV files...")
    embeddings, texts = load_csv_files(args.embeddings_csv, args.texts_csv)
    
    print("Converting to FAISS format...")
    convert_to_faiss(embeddings, texts, args.output_dir)
    
    print(f"Conversion complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import json
from pathlib import Path
import os
from tqdm import tqdm
import time

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

def mean_pooling(token_embeddings, attention_mask):
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_gpu_memory():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
    return "GPU not available"

def encode_texts(texts, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    start_time = time.time()
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(device)
    model.eval()
    print(f"Model loaded. GPU Memory: {get_gpu_memory()}")
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate total batches for progress bar
    total_batches = len(dataloader)
    print(f"\nStarting encoding of {len(texts)} texts in {total_batches} batches")
    print(f"Batch size: {batch_size}")
    
    # Encode texts
    all_embeddings = []
    processed_texts = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding texts", 
                         unit="batch", ncols=100):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings = mean_pooling(outputs[0], batch['attention_mask'])
            all_embeddings.append(embeddings.cpu())
            
            processed_texts += len(batch['input_ids'])
            if processed_texts % (batch_size * 10) == 0:  # Log every 10 batches
                print(f"\nProcessed {processed_texts}/{len(texts)} texts. "
                      f"GPU Memory: {get_gpu_memory()}")
    
    # Combine all embeddings
    print("\nCombining embeddings...")
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    # Print statistics
    elapsed_time = time.time() - start_time
    texts_per_second = len(texts) / elapsed_time
    print(f"\nEncoding completed in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {texts_per_second:.2f} texts/second")
    
    return embeddings

def main():
    # Paths
    checkpoint_dir = "/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding/checkpoints"
    checkpoint_name = "checkpoint_5_of_21"
    
    # Load texts from JSON
    json_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
    print(f"Loading texts from: {json_path}")
    start_time = time.time()
    with open(json_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    load_time = time.time() - start_time
    print(f"Loaded {len(texts)} texts in {load_time:.2f} seconds")
    
    # Encode texts
    print("\nStarting encoding process...")
    embeddings = encode_texts(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Create and save FAISS index
    print("\nCreating FAISS index...")
    start_time = time.time()
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index
    faiss_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.faiss")
    print(f"Saving FAISS index to: {faiss_path}")
    faiss.write_index(index, faiss_path)
    save_time = time.time() - start_time
    print(f"Index creation and saving completed in {save_time:.2f} seconds")
    print("Done!")

if __name__ == "__main__":
    main()
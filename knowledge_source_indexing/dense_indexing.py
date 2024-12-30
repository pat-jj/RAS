import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import os

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
    """
    Perform mean pooling on token embeddings using attention mask
    """
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings

def debug_gpu_info():
    print(f"[Rank {os.environ.get('LOCAL_RANK', 'N/A')}] Number of GPUs available: {torch.cuda.device_count()}")
    print(f"[Rank {os.environ.get('LOCAL_RANK', 'N/A')}] Current device: {torch.cuda.current_device()}")
    print(f"[Rank {os.environ.get('LOCAL_RANK', 'N/A')}] Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if dist.is_initialized():
        print(f"[Rank {os.environ.get('LOCAL_RANK', 'N/A')}] World size: {dist.get_world_size()}")
        print(f"[Rank {os.environ.get('LOCAL_RANK', 'N/A')}] Global rank: {dist.get_rank()}")
    else:
        print(f"[Rank {os.environ.get('LOCAL_RANK', 'N/A')}] Distributed training not initialized!")

class ContrieverEncoder:
    def __init__(self, model_name="facebook/contriever-msmarco"):
        # Get local rank from environment variable
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        # Important: Set the device before initializing the model
        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}' if self.local_rank != -1 else 'cuda:0')
        
        print(f"[Rank {self.local_rank}] Initializing model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        if self.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32):
        dataset = TextDataset(texts, self.tokenizer)
        sampler = DistributedSampler(dataset) if self.local_rank != -1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        all_embeddings = []
        
        for batch in tqdm(dataloader, disable=self.local_rank not in [-1, 0]):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            embeddings = mean_pooling(outputs[0], batch['attention_mask'])
            all_embeddings.append(embeddings.cpu())
            
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if self.local_rank != -1:
            gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_embeddings, embeddings)
            if self.local_rank == 0:
                embeddings = torch.cat(gathered_embeddings, dim=0)
        
        return embeddings.numpy()

def setup_distributed():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return local_rank
    return -1

def save_to_faiss(embeddings, texts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, str(output_dir / "wikipedia_embeddings.faiss"))
    
    # Save texts mapping
    with open(output_dir / "text_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank = setup_distributed()
    debug_gpu_info()
    
    # Load Wikipedia texts
    with open(args.input_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)
        if not isinstance(texts, list):
            raise ValueError("Input JSON should contain a list of texts")
    
    # Initialize encoder and encode texts
    encoder = ContrieverEncoder()
    embeddings = encoder.encode_texts(texts, batch_size=args.batch_size)
    
    # Save embeddings and mapping (only on main process)
    if local_rank in [-1, 0]:
        save_to_faiss(embeddings, texts, args.output_dir)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
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
import datetime
import math

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, start_idx=0):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_idx = start_idx
        self.indices = list(range(start_idx, start_idx + len(texts)))
    
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
        return {
            **{k: v.squeeze(0) for k, v in inputs.items()},
            'original_idx': torch.tensor(self.indices[idx])
        }

def mean_pooling(token_embeddings, attention_mask):
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings

def save_checkpoint(embeddings, indices, texts, checkpoint_dir, checkpoint_name):
    """Save intermediate results"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}")
    
    # Normalize and save embeddings
    embeddings_np = embeddings.numpy()
    faiss.normalize_L2(embeddings_np)
    
    # Create and save FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, f"{checkpoint_path}.faiss")
    
    # Save corresponding texts
    text_indices = indices.numpy()
    checkpoint_texts = [texts[idx] for idx in text_indices]
    with open(f"{checkpoint_path}.json", 'w', encoding='utf-8') as f:
        json.dump(checkpoint_texts, f, ensure_ascii=False, indent=2)
    
    print(f"Saved checkpoint: {checkpoint_path}")

class ContrieverEncoder:
    def __init__(self, model_name="facebook/contriever-msmarco"):
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}' if self.local_rank != -1 else 'cuda:0')
        
        print(f"[Rank {self.local_rank}] Initializing model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        if self.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
        
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32, checkpoint_dir=None):
        total_size = len(texts)
        checkpoint_sizes = [
            int(0.25 * total_size),
            int(0.50 * total_size),
            int(0.75 * total_size),
            total_size
        ]
        
        all_embeddings = []
        all_indices = []
        processed_count = 0
        
        for checkpoint_idx, end_idx in enumerate(checkpoint_sizes, 1):
            start_idx = processed_count
            chunk_size = end_idx - start_idx
            
            if chunk_size == 0:
                continue
                
            # Process chunk
            chunk_texts = texts[start_idx:end_idx]
            dataset = TextDataset(chunk_texts, self.tokenizer, start_idx=start_idx)
            sampler = DistributedSampler(dataset, shuffle=False) if self.local_rank != -1 else None
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
            
            chunk_embeddings = []
            chunk_indices = []
            
            for batch in tqdm(dataloader, 
                            disable=self.local_rank not in [-1, 0],
                            desc=f"Processing chunk {checkpoint_idx}/4"):
                original_idx = batch.pop('original_idx')
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                embeddings = mean_pooling(outputs[0], batch['attention_mask'])
                
                chunk_embeddings.append(embeddings.cpu())
                chunk_indices.append(original_idx.cpu())
            
            # Gather chunk results
            embeddings = torch.cat(chunk_embeddings, dim=0)
            indices = torch.cat(chunk_indices, dim=0)
            
            if self.local_rank != -1:
                # Gather from all processes
                local_size = torch.tensor([embeddings.shape[0]], device=self.device)
                sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
                dist.all_gather(sizes, local_size)
                
                gathered_embeddings = []
                gathered_indices = []
                
                for i, size in enumerate(sizes):
                    size = size.item()
                    emb_tensor = torch.zeros(size, embeddings.shape[1], dtype=embeddings.dtype)
                    idx_tensor = torch.zeros(size, dtype=indices.dtype)
                    
                    if i == self.local_rank:
                        emb_tensor = embeddings
                        idx_tensor = indices
                    
                    dist.broadcast(emb_tensor, i)
                    dist.broadcast(idx_tensor, i)
                    
                    gathered_embeddings.append(emb_tensor)
                    gathered_indices.append(idx_tensor)
                
                if self.local_rank == 0:
                    embeddings = torch.cat(gathered_embeddings, dim=0)
                    indices = torch.cat(gathered_indices, dim=0)
                    
                    # Sort by original indices
                    sorted_order = torch.argsort(indices)
                    embeddings = embeddings[sorted_order]
                    indices = indices[sorted_order]
                    
                    # Save checkpoint
                    if checkpoint_dir:
                        checkpoint_name = f"checkpoint_{checkpoint_idx}_of_4"
                        save_checkpoint(embeddings, indices, texts, checkpoint_dir, checkpoint_name)
            
            # Accumulate results
            all_embeddings.append(embeddings)
            all_indices.append(indices)
            processed_count = end_idx
            
            # Clear memory
            torch.cuda.empty_cache()
        
        # Combine all checkpoints
        if self.local_rank in [-1, 0]:
            final_embeddings = torch.cat(all_embeddings, dim=0)
            return final_embeddings.numpy()
        return None

def setup_distributed():
    if 'LOCAL_RANK' in os.environ:
        try:
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
            return local_rank
        except Exception as e:
            print(f"Error setting up distributed training: {str(e)}")
            raise
    return -1

def save_to_faiss(embeddings, texts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        faiss.write_index(index, str(output_dir / "wikipedia_embeddings.faiss"))
        with open(output_dir / "text_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving FAISS index: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    try:
        local_rank = setup_distributed()
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
            if not isinstance(texts, list):
                raise ValueError("Input JSON should contain a list of texts")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        
        encoder = ContrieverEncoder()
        embeddings = encoder.encode_texts(texts, 
                                      batch_size=args.batch_size,
                                      checkpoint_dir=checkpoint_dir)
        
        if local_rank in [-1, 0]:
            save_to_faiss(embeddings, texts, args.output_dir)
            
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
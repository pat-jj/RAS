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
import logging

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

def setup_logging(output_dir):
    """Setup logging to file and console"""
    import logging
    log_file = os.path.join(output_dir, "encoding.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('encoder')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(embeddings, indices, texts, checkpoint_dir, checkpoint_name, logger):
    """Save intermediate results with error handling"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}")
        
        logger.info(f"Starting checkpoint save: {checkpoint_name}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Convert to numpy safely
        if torch.is_tensor(embeddings):
            logger.info("Converting embeddings from torch tensor to numpy")
            embeddings_np = embeddings.numpy()
        else:
            embeddings_np = embeddings
            
        # Save embeddings first
        logger.info("Normalizing embeddings")
        faiss.normalize_L2(embeddings_np)
        
        logger.info("Creating FAISS index")
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)
        
        logger.info(f"Saving FAISS index to {checkpoint_path}.faiss")
        faiss.write_index(index, f"{checkpoint_path}.faiss")
        
        # Save texts mapping
        logger.info("Processing text indices")
        if torch.is_tensor(indices):
            text_indices = indices.numpy()
        else:
            text_indices = indices
            
        checkpoint_texts = [texts[int(idx)] for idx in text_indices]
        
        logger.info(f"Saving text mapping to {checkpoint_path}.json")
        with open(f"{checkpoint_path}.json", 'w', encoding='utf-8') as f:
            json.dump(checkpoint_texts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved checkpoint: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Error saving checkpoint {checkpoint_name}: {str(e)}", exc_info=True)
        raise

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
        logger = logging.getLogger('encoder')
        total_size = len(texts)
        logger.info(f"Starting encoding of {total_size} texts")
        
        # Calculate checkpoint sizes
        checkpoint_sizes = [
            int(0.01 * total_size),
            int(0.05 * total_size),
            int(0.10 * total_size),
            int(0.15 * total_size),
            int(0.20 * total_size),
            int(0.25 * total_size),
            int(0.30 * total_size),
            int(0.35 * total_size),
            int(0.40 * total_size),
            int(0.45 * total_size),
            int(0.50 * total_size),
            int(0.55 * total_size),
            int(0.60 * total_size),
            int(0.65 * total_size),
            int(0.70 * total_size),
            int(0.75 * total_size),
            int(0.80 * total_size),
            int(0.85 * total_size),
            int(0.90 * total_size),
            int(0.95 * total_size),
            total_size
        ]
        logger.info(f"Checkpoint sizes: {checkpoint_sizes}")
        
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
                num_workers=6,
                pin_memory=True,
                drop_last=False
            )
            
            chunk_embeddings = []
            chunk_indices = []
            
            for batch in tqdm(dataloader, 
                            disable=self.local_rank not in [-1, 0],
                            desc=f"Processing chunk {checkpoint_idx}/21"):
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
                try:
                    logger.info(f"[Rank {self.local_rank}] Starting gathering process")
                    # Move tensors to GPU for gathering
                    embeddings = embeddings.to(self.device)
                    indices = indices.to(self.device)
                    
                    # Gather sizes first
                    local_size = torch.tensor([embeddings.shape[0]], device=self.device)
                    sizes = [torch.zeros_like(local_size, device=self.device) for _ in range(dist.get_world_size())]
                    dist.all_gather(sizes, local_size)
                    
                    # Pre-allocate tensors on GPU
                    gathered_embeddings = []
                    gathered_indices = []
                    
                    for i, size in enumerate(sizes):
                        size = size.item()
                        logger.info(f"[Rank {self.local_rank}] Allocating tensor for rank {i} with size {size}")
                        emb_tensor = torch.zeros(size, embeddings.shape[1], 
                                               dtype=embeddings.dtype, 
                                               device=self.device)
                        idx_tensor = torch.zeros(size, 
                                               dtype=indices.dtype, 
                                               device=self.device)
                        
                        if i == self.local_rank:
                            emb_tensor.copy_(embeddings)
                            idx_tensor.copy_(indices)
                        
                        # Synchronize before broadcast
                        torch.cuda.synchronize()
                        
                        dist.broadcast(emb_tensor, i)
                        dist.broadcast(idx_tensor, i)
                        
                        # Move to CPU after broadcast to save GPU memory
                        gathered_embeddings.append(emb_tensor.cpu())
                        gathered_indices.append(idx_tensor.cpu())
                        
                        logger.info(f"[Rank {self.local_rank}] Gathered tensors from rank {i}")
                    
                except Exception as e:
                    logger.error(f"[Rank {self.local_rank}] Error during gathering: {str(e)}", exc_info=True)
                    raise
                
                if self.local_rank == 0:
                    embeddings = torch.cat(gathered_embeddings, dim=0)
                    indices = torch.cat(gathered_indices, dim=0)
                    
                    # Sort by original indices
                    sorted_order = torch.argsort(indices)
                    embeddings = embeddings[sorted_order]
                    indices = indices[sorted_order]
                    
                    # Save checkpoint
                    if checkpoint_dir:
                        checkpoint_name = f"checkpoint_{checkpoint_idx}_of_21"
                        save_checkpoint(embeddings, indices, texts, checkpoint_dir, checkpoint_name, logger=logger)
            
            # Accumulate results only if we are rank 0 or not distributed
            if self.local_rank in [-1, 0]:
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

def log_gpu_memory(local_rank, logger, step=""):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(local_rank) / (1024 * 1024 * 1024)
        cached = torch.cuda.memory_reserved(local_rank) / (1024 * 1024 * 1024)
        logger.info(f"[Rank {local_rank}] {step} - GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    try:
        # Create output directory and setup logging
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logging(args.output_dir)
        
        local_rank = setup_distributed()
        logger.info(f"Process initialized with local_rank: {local_rank}")
        
        logger.info(f"Loading texts from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
            if not isinstance(texts, list):
                raise ValueError("Input JSON should contain a list of texts")
        logger.info(f"Loaded {len(texts)} texts")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Created checkpoint directory: {checkpoint_dir}")
        
        logger.info("Initializing encoder")
        encoder = ContrieverEncoder()
        
        logger.info(f"Starting encoding with batch size {args.batch_size}")
        embeddings = encoder.encode_texts(texts, 
                                      batch_size=args.batch_size,
                                      checkpoint_dir=checkpoint_dir)
        
        # if local_rank in [-1, 0]:
        #     save_to_faiss(embeddings, texts, args.output_dir)
            
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
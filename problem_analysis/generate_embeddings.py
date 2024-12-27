import faiss
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from pathlib import Path
import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def mean_pooling(token_embeddings, mask):
    """Compute mean pooling over token embeddings"""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class EmbeddingGenerator:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        
        # Initialize model
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
        self.model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(self.device)
        self.model = DDP(self.model, device_ids=[rank])
        
        self.batch_size = 512
        self.save_every = 100000
        
    def generate_embeddings(self, texts):
        """Generate embeddings for a batch of texts using mean pooling"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            return embeddings.cpu().numpy()
            
    def process_corpus(self, corpus_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Load corpus
        if self.rank == 0:
            print("Loading corpus...")
        with open(corpus_path) as f:
            corpus = json.load(f)
            
        # Split corpus among GPUs
        per_gpu = len(corpus) // self.world_size
        start_idx = self.rank * per_gpu
        end_idx = start_idx + per_gpu if self.rank != self.world_size - 1 else len(corpus)
        local_corpus = corpus[start_idx:end_idx]
        
        embeddings = []
        for i in tqdm(range(0, len(local_corpus), self.batch_size), 
                     disable=self.rank != 0):
            # Process batch
            batch = local_corpus[i:i + self.batch_size]
            batch_embeddings = self.generate_embeddings(batch)
            embeddings.append(batch_embeddings)

            # Periodic saving
            if len(embeddings) * self.batch_size >= self.save_every:
                embeddings_array = np.vstack(embeddings)
                save_path = output_dir / f'embeddings_rank{self.rank}_part{i//self.save_every}.npy'
                np.save(str(save_path), embeddings_array)
                embeddings = []  # Clear list after saving
                
                if self.rank == 0:
                    print(f"Saved embeddings until batch {i}")

        # Save remaining embeddings
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            save_path = output_dir / f'embeddings_rank{self.rank}_final.npy'
            np.save(str(save_path), embeddings_array)

def combine_and_index(output_dir, total_docs):
    """Combine all partial embeddings and create FAISS index"""
    output_dir = Path(output_dir)
    
    # Load and combine all partial embeddings
    all_embeddings = []
    for emb_file in sorted(output_dir.glob('embeddings_rank*_*.npy')):
        embeddings = np.load(emb_file)
        all_embeddings.append(embeddings)
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Create and save FAISS index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)
    
    faiss.write_index(index, str(output_dir / "wiki_index.faiss"))
    
    # Save metadata
    metadata = {
        'total_docs': total_docs,
        'embedding_dim': dimension,
        'final_embedding_count': len(all_embeddings)
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)

def main_worker(rank, world_size, corpus_path, output_dir):
    setup(rank, world_size)
    
    generator = EmbeddingGenerator(rank, world_size)
    generator.process_corpus(corpus_path, output_dir)  # Changed from generate_embeddings to process_corpus
    
    cleanup()

def main():
    corpus_path = "/shared/eng/pj20/hotpotqa/data/all_wiki_text.json"
    output_dir = "/shared/eng/pj20/hotpotqa/data/embeddings"
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    
    # Start multiple processes
    mp.spawn(
        main_worker,
        args=(world_size, corpus_path, output_dir),
        nprocs=world_size,
        join=True
    )
    
    # After all processes complete, combine embeddings and create index
    with open(corpus_path) as f:
        total_docs = len(json.load(f))
    combine_and_index(output_dir, total_docs)

if __name__ == "__main__":
    main()
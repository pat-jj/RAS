### This script retrieves documents from the hotpotqa dataset using the contriever model
### It is distributed across multiple GPUs

import faiss
import json
import torch
import torch.distributed as dist
import numpy as np
import argparse
import os
from typing import List, Dict
import logging
try:
    # Import FAISS GPU
    import faiss.contrib.torch_utils
    from faiss import get_num_gpus
except ImportError:
    raise ImportError("FAISS GPU support not found. Please install faiss-gpu")
import signal
import sys
import time

def setup_logging(output_dir: str, rank: int) -> logging.Logger:
    """Setup logging configuration with rank-specific files"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [Rank %(rank)d] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, f'retrieval_rank_{rank}.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.addFilter(lambda record: setattr(record, 'rank', rank) or True)
    return logger

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11335'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def custom_collate_fn(batch):
    """Custom collate function to properly batch the data"""
    if not batch:
        raise ValueError("Empty batch received")
        
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    queries = [item['query'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'queries': queries
    }

class WikiRetriever:
    def __init__(
        self,
        faiss_path: str,
        text_mapping_path: str,
        batch_size: int = 32,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.rank = rank
        self.world_size = world_size
        self.logger = logging.getLogger()
        self.text_mapping_path = text_mapping_path
        
        # Load full text mapping first to ensure proper alignment
        if self.rank == 0:
            self.logger.info(f"Loading text mapping from {text_mapping_path}")
        with open(text_mapping_path, 'r', encoding='utf-8') as f:
            self.full_mapping = json.load(f)
    
        # Load and validate FAISS index
        if rank == 0:
            self.logger.info(f"Loading FAISS index from {faiss_path}")
        
        try:
            # Load CPU index first
            cpu_index = faiss.read_index(faiss_path)
            self.dim = cpu_index.d
            self.size = cpu_index.ntotal
            
            # Validate index and text mapping alignment
            if self.size != len(self.full_mapping):
                raise ValueError(
                    f"Mismatch between FAISS index size ({self.size} vectors) and "
                    f"text mapping length ({len(self.full_mapping)} entries)"
                )
            
            # Calculate shard information
            self.shard_size = self.size // world_size
            self.start_idx = rank * self.shard_size
            self.end_idx = self.start_idx + self.shard_size if rank != world_size - 1 else self.size
            
            # Initialize GPU resources
            res = faiss.StandardGpuResources()
            res.setTempMemory(4 * 1024 * 1024 * 1024)  # 4GB temp memory
            
            # Configure GPU index options
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            co.deviceId = rank
            
            # Create empty GPU index
            base_index = faiss.IndexFlatIP(self.dim)
            self.gpu_index = faiss.index_cpu_to_gpu(res, rank, base_index, co)
            
            # Load vectors in chunks
            chunk_size = 25000
            for chunk_start in range(self.start_idx, self.end_idx, chunk_size):
                chunk_end = min(chunk_start + chunk_size, self.end_idx)
                if rank == 0 or chunk_start % (5 * chunk_size) == 0:  # Log less frequently
                    self.logger.info(
                        f"Rank {rank}: Loading vectors {chunk_start} to {chunk_end} "
                        f"({((chunk_start - self.start_idx) / (self.end_idx - self.start_idx) * 100):.1f}%)"
                    )
                
                # Extract vectors for this chunk
                chunk_vectors = cpu_index.reconstruct_n(chunk_start, chunk_end - chunk_start)
                
                # Convert to correct format
                if torch.is_tensor(chunk_vectors):
                    chunk_vectors = chunk_vectors.float()
                else:
                    chunk_vectors = torch.tensor(chunk_vectors, dtype=torch.float32)
                
                # Add to GPU index
                self.gpu_index.add(chunk_vectors.numpy())
                
                # Clean up
                del chunk_vectors
                torch.cuda.empty_cache()
                time.sleep(1.0)  # Delay between chunks
            
            # Clean up CPU index
            del cpu_index
            torch.cuda.empty_cache()
            
            # Store shard-specific text mapping
            self.text_mapping = self.full_mapping[self.start_idx:self.end_idx]
            
            if rank == 0:
                self.logger.info(
                    f"Successfully initialized WikiRetriever on rank {rank}:\n"
                    f"- Total vectors: {self.size}\n"
                    f"- Shard size: {self.shard_size}\n"
                    f"- This shard: {self.end_idx - self.start_idx} vectors\n"
                    f"- Vector dimension: {self.dim}"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU index on rank {rank}: {e}")
            raise
        
        self.batch_size = batch_size


    def retrieve(self, query_embeddings: torch.Tensor, queries, top_k: int = 5) -> List[Dict]:
        if self.rank == 0:
            self.logger.info(f"Query embeddings shape: {query_embeddings.shape}")
        
        # Search on local shard
        search_vectors = query_embeddings.cpu().numpy().astype('float32')
        
        # Process in small batches for memory efficiency
        batch_size = 100
        # Search vectors in chunks
        all_local_scores = []
        all_local_indices = []
        
        for i in range(0, len(search_vectors), batch_size):
            batch_end = min(i + batch_size, len(search_vectors))
            batch_vectors = search_vectors[i:batch_end]
            
            # Get raw search results without adding start_idx
            local_scores, local_indices = self.gpu_index.search(batch_vectors, top_k)
            all_local_scores.append(local_scores)
            all_local_indices.append(local_indices)

        # Combine batch results
        local_scores = np.concatenate(all_local_scores, axis=0)
        local_indices = np.concatenate(all_local_indices, axis=0)
        
        # Add shard offset to indices AFTER concatenation
        if self.world_size > 1:
            local_indices = local_indices + (self.rank * self.shard_size)
        
        # Convert to tensors for gathering
        scores_tensor = torch.from_numpy(local_scores).cuda(self.rank)
        indices_tensor = torch.from_numpy(local_indices).cuda(self.rank)
        
        # Gather all results
        gathered_scores = [torch.zeros_like(scores_tensor) for _ in range(self.world_size)]
        gathered_indices = [torch.zeros_like(indices_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_scores, scores_tensor)
        dist.all_gather(gathered_indices, indices_tensor)
        
        if self.rank == 0:
            self.logger.info(f"First batch results indices: {local_indices[0]}")
            # Combine results from all shards
            all_scores = torch.cat(gathered_scores, dim=1)
            all_indices = torch.cat(gathered_indices, dim=1)
            
            # Get global top-k
            final_scores, top_k_indices = torch.topk(
                all_scores,
                k=min(top_k, all_scores.shape[1]),
                dim=1,
                largest=True,
                sorted=True
            )
            
            # Map back to document indices
            final_indices = torch.gather(all_indices, 1, top_k_indices)
            
            # Create results using full_mapping
            results = []
            for query_idx, (query, scores, indices) in enumerate(
                zip(queries, final_scores.cpu().numpy(), final_indices.cpu().numpy())
            ):
                retrieved_docs = []
                for doc_idx, (score, idx) in enumerate(zip(scores, indices)):
                    if 0 <= idx < len(self.full_mapping):
                        retrieved_docs.append({
                            'text': self.full_mapping[idx],
                            'score': float(score)
                        })
                
                results.append({
                    'query': query,
                    'retrieved_docs': retrieved_docs
                })
                
            
            return results
            
        return None
            

def main_worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    setup_distributed(rank, world_size)
    logger = setup_logging(args.output_dir, rank)
    
    queries = torch.load(args.queries_embeddings_path)['queries']
    query_embeddings = torch.load(args.queries_embeddings_path)['embeddings'].cuda(rank)

    retriever = WikiRetriever(
        args.faiss_path,
        args.text_mapping_path,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
    )

    logger.info(f"Rank {rank}: Starting retrieval for {len(queries)} queries")
    retrieval_results = retriever.retrieve(query_embeddings,queries, top_k=5)
    
    if retrieval_results is not None and rank == 0:
        output_path = os.path.join(args.output_dir, "wiki_retrieval_results.json")
        logger.info(f"Saving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(retrieval_results, f, indent=2)
        
        logger.info(f"Retrieval process completed successfully. Saved results for {len(retrieval_results)} questions")



def main():
    parser = argparse.ArgumentParser(description='Distributed Wikipedia document retrieval for HotpotQA queries')
    parser.add_argument('--faiss_path', type=str,
                      default="/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding/wikipedia_embeddings.faiss")
    parser.add_argument('--text_mapping_path', type=str,
                      default="/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding/text_mapping.json")
    parser.add_argument('--output_dir', type=str,
                      default="/shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                      help='Number of GPUs to use')
    parser.add_argument('--queries_embeddings_path', type=str,
                      default="/shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval/queries.pt")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    def signal_handler(sig, frame):
        print('Received interrupt signal. Cleaning up...')
        dist.destroy_process_group()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    # Launch distributed processes
    torch.multiprocessing.spawn(
        main_worker,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    )

if __name__ == "__main__":
    main()
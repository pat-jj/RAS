import faiss
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import argparse
import os
from typing import List, Dict
import logging
from pathlib import Path
import hashlib
from typing import Optional
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
    os.environ['MASTER_PORT'] = '11334'
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
        model_name: str = "facebook/contriever-msmarco",
        batch_size: int = 32,
        cache_dir: str = "/shared/eng/pj20/hf_cache",
        rank: int = 0,
        world_size: int = 1,
        embeddings_cache_dir: str = None
    ):
        self.rank = rank
        self.world_size = world_size
        self.logger = logging.getLogger()
        self.embeddings_cache_dir = embeddings_cache_dir
        
        if embeddings_cache_dir and rank == 0:
            os.makedirs(embeddings_cache_dir, exist_ok=True)
            self.logger.info(f"Using embeddings cache directory: {embeddings_cache_dir}")
        
        # Initialize model and tokenizer
        if rank == 0:
            self.logger.info(f"Initializing Contriever model and tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Move model to GPU and wrap with DDP
        self.model = self.model.cuda(rank)
        self.model = DDP(self.model, device_ids=[rank])
        self.model.eval()
        
        # Modified GPU index initialization
        if rank == 0:
            self.logger.info(f"Loading FAISS index from {faiss_path}")
            

        try:
            # Load CPU index first
            cpu_index = faiss.read_index(faiss_path)
            self.dim = cpu_index.d
            self.size = cpu_index.ntotal
            
            # Calculate shard information
            self.shard_size = self.size // world_size
            self.start_idx = rank * self.shard_size
            self.end_idx = self.start_idx + self.shard_size if rank != world_size - 1 else self.size
            
            # Initialize GPU resources with explicit device selection
            res = faiss.StandardGpuResources()
            
            # Increase temp memory allocation for large-scale search
            res.setTempMemory(4 * 1024 * 1024 * 1024)  # 4GB temp memory
            
            # Modified GPU index configuration
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True  # Keep FP16 for memory efficiency
            co.deviceId = rank
            
            # Create empty GPU index with different index type
            base_index = faiss.IndexFlatIP(self.dim)
            self.gpu_index = faiss.index_cpu_to_gpu(res, rank, base_index, co)
            
            # Reduced chunk size for safer memory handling
            chunk_size = 25000  # Smaller chunks
            
            for chunk_start in range(self.start_idx, self.end_idx, chunk_size):
                chunk_end = min(chunk_start + chunk_size, self.end_idx)
                self.logger.info(f"Rank {rank}: Loading vectors {chunk_start} to {chunk_end} on GPU {rank}")
                
                chunk_vectors = cpu_index.reconstruct_n(chunk_start, chunk_end - chunk_start)
                if torch.is_tensor(chunk_vectors):
                    chunk_vectors = chunk_vectors.float()
                else:
                    chunk_vectors = torch.tensor(chunk_vectors, dtype=torch.float32)
                
                # Add directly to GPU index without moving to GPU first
                self.gpu_index.add(chunk_vectors.numpy())
                
                del chunk_vectors
                torch.cuda.empty_cache()
                time.sleep(1.0)  # Increased delay between chunks
            
            del cpu_index
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU index on rank {rank}: {e}")
            raise
        
        # Load text mapping for this shard
        with open(text_mapping_path, 'r', encoding='utf-8') as f:
            full_mapping = json.load(f)
            self.text_mapping = full_mapping[self.start_idx:self.end_idx]
        
        self.batch_size = batch_size

    def _get_cache_path(self, queries: List[str]) -> Path:
        """Generate a deterministic cache path for a list of queries"""
        if not self.embeddings_cache_dir:
            return None
        queries_str = '\n'.join(queries).encode('utf-8')
        queries_hash = hashlib.md5(queries_str).hexdigest()
        model_name = self.model.module.__class__.__name__
        cache_path = Path(self.embeddings_cache_dir) / f"{model_name}_{queries_hash}.npy"
        return cache_path

    def _load_cached_embeddings(self, cache_path: Path) -> Optional[torch.Tensor]:
        """Load cached embeddings if they exist"""
        if not cache_path or not cache_path.exists():
            return None
        try:
            if self.rank == 0:
                self.logger.info(f"Loading cached embeddings from {cache_path}")
            embeddings = torch.from_numpy(np.load(str(cache_path)))
            return embeddings
        except Exception as e:
            self.logger.warning(f"Failed to load cached embeddings: {e}")
            return None

    def _save_embeddings_cache(self, embeddings: torch.Tensor, cache_path: Path):
        """Save embeddings to cache (only on rank 0)"""
        if self.rank == 0 and cache_path:
            try:
                self.logger.info(f"Saving embeddings cache to {cache_path}")
                np.save(str(cache_path), embeddings.cpu().numpy())
            except Exception as e:
                self.logger.warning(f"Failed to save embeddings cache: {e}")

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean pooling of token embeddings"""
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries using Contriever model with distributed processing and caching"""
        if self.rank == 0:
            self.logger.info(f"Starting query encoding for {len(queries)} queries...")
        
        # Check cache first
        cache_path = self._get_cache_path(queries)
        cached_embeddings = self._load_cached_embeddings(cache_path)
        if cached_embeddings is not None:
            self.logger.info(f"Rank {self.rank}: Using cached embeddings")
            return cached_embeddings
        
        # If no cache, encode queries
        dataset = QueryDataset(queries, self.tokenizer, self.logger)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=custom_collate_fn
        )
        
        self.logger.info(f"Rank {self.rank}: Processing {len(dataset)} queries in {len(dataloader)} batches")
        
        all_embeddings = []
        total_encoding_time = 0
        
        # Only rank 0 shows progress
        pbar = None
        if self.rank == 0:
            pbar = tqdm(total=len(dataloader), desc="Encoding queries", unit='batch')
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            inputs = {
                'input_ids': batch['input_ids'].cuda(self.rank),
                'attention_mask': batch['attention_mask'].cuda(self.rank)
            }
            
            # Track memory before encoding
            if self.rank == 0 and batch_idx % 10 == 0:  # Log every 10 batches
                memory_before = torch.cuda.memory_allocated(self.rank) / 1024**2
                self.logger.info(f"GPU memory before encoding batch {batch_idx}: {memory_before:.2f}MB")
            
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

            # Calculate batch statistics
            batch_time = time.time() - batch_start_time
            total_encoding_time += batch_time
            avg_time_per_query = batch_time / len(batch['queries'])
            
            if self.rank == 0 and batch_idx % 10 == 0:  # Log every 10 batches
                memory_after = torch.cuda.memory_allocated(self.rank) / 1024**2
                self.logger.info(
                    f"Batch {batch_idx}/{len(dataloader)}: "
                    f"Time={batch_time:.2f}s, "
                    f"Queries={len(batch['queries'])}, "
                    f"Time/Query={avg_time_per_query*1000:.1f}ms, "
                    f"Memory={memory_after:.2f}MB"
                )
                
                # Log sample queries in this batch
                sample_queries = batch['queries'][:2]  # Show first 2 queries
                self.logger.info(f"Sample queries in batch: {sample_queries}")

            if pbar:
                pbar.update(1)
                
            # Clear memory after each batch
            torch.cuda.empty_cache()

        if pbar:
            pbar.close()

        # Concatenate all embeddings
        local_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Gather embeddings from all GPUs
        gathered_embeddings = [torch.zeros_like(local_embeddings) for _ in range(self.world_size)]
        dist.all_gather(gathered_embeddings, local_embeddings)
        
        final_embeddings = torch.cat(gathered_embeddings, dim=0)
        
        # Log final statistics
        if self.rank == 0:
            avg_time_per_query = total_encoding_time / len(queries)
            self.logger.info(
                f"Query encoding completed: "
                f"Total time={total_encoding_time:.2f}s, "
                f"Queries={len(queries)}, "
                f"Avg time/query={avg_time_per_query*1000:.1f}ms, "
                f"Final embedding shape={final_embeddings.shape}"
            )
        
        # Save to cache
        self._save_embeddings_cache(final_embeddings, cache_path)
        
        return final_embeddings

    def retrieve(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Retrieval method with detailed progress tracking"""
        try:
            if self.rank == 0:
                self.logger.info(f"Starting retrieval for {len(queries)} queries...")
            
            dist.barrier()
            
            # Track overall progress
            total_progress = torch.zeros(self.world_size, dtype=torch.float32, device=f'cuda:{self.rank}')
            progress_tensor = torch.zeros(1, dtype=torch.float32, device=f'cuda:{self.rank}')
            
            # Encode queries (with caching)
            query_embeddings = self.encode_queries(queries)
            
            # Shard queries across GPUs
            queries_per_gpu = len(queries) // self.world_size
            start_query = self.rank * queries_per_gpu
            end_query = start_query + queries_per_gpu if self.rank != self.world_size - 1 else len(queries)
            
            local_query_embeddings = query_embeddings[start_query:end_query]
            local_queries = queries[start_query:end_query]
            num_local_queries = len(local_queries)
            
            self.logger.info(f"Rank {self.rank}: Processing {num_local_queries} queries ({start_query} to {end_query})")
            
            success_tensor = torch.zeros(1, dtype=torch.bool, device=f'cuda:{self.rank}')
            search_batch_size = 2
            all_local_scores = []
            all_local_indices = []
            
            try:
                # Progress logging setup
                log_interval = max(1, num_local_queries // 20)  # Log roughly 20 times per GPU
                last_log_time = time.time()
                start_time = time.time()
                
                for i in range(0, num_local_queries, search_batch_size):
                    batch_end = min(i + search_batch_size, num_local_queries)
                    batch_embeddings = local_query_embeddings[i:batch_end]
                    
                    # Update progress
                    current_progress = (i + search_batch_size) / num_local_queries * 100
                    progress_tensor.fill_(min(current_progress, 100.0))
                    
                    # Gather progress from all GPUs periodically
                    if i % log_interval == 0 or batch_end == num_local_queries:
                        dist.all_gather(list(total_progress.split(1)), progress_tensor)
                        
                        if self.rank == 0:
                            avg_progress = total_progress.mean().item()
                            elapsed_time = time.time() - start_time
                            queries_processed = sum(min(p.item() / 100 * n, n) 
                                                for p, n in zip(total_progress, [queries_per_gpu] * self.world_size))
                            
                            # Calculate processing speed and ETA
                            if elapsed_time > 0:
                                qps = queries_processed / elapsed_time
                                remaining_queries = len(queries) - queries_processed
                                eta = remaining_queries / qps if qps > 0 else 0
                                
                                self.logger.info(
                                    f"\nOverall Progress: {avg_progress:.1f}% | "
                                    f"Speed: {qps:.1f} queries/sec | "
                                    f"ETA: {eta/60:.1f} minutes\n"
                                    f"Individual GPU Progress: " + 
                                    " | ".join(f"GPU {j}: {p.item():.1f}%" for j, p in enumerate(total_progress))
                                )
                    
                    # Perform search
                    search_vectors = batch_embeddings.cpu().numpy().astype('float32')
                    local_scores, local_indices = self.gpu_index.search(search_vectors, top_k)
                    
                    local_indices += self.start_idx
                    all_local_scores.append(local_scores)
                    all_local_indices.append(local_indices)
                    
                    # Memory management
                    torch.cuda.empty_cache()
                    
                    # Detailed batch statistics (every 10 batches)
                    if i % (10 * search_batch_size) == 0:
                        current_time = time.time()
                        batch_time = current_time - last_log_time
                        if batch_time > 0:
                            batch_qps = (batch_end - i) / batch_time
                            self.logger.info(
                                f"Rank {self.rank} - Batch {i//search_batch_size}: "
                                f"Processing {batch_qps:.1f} queries/sec"
                            )
                        last_log_time = current_time
                
                # Mark successful completion
                success_tensor.fill_(True)
                local_scores = np.concatenate(all_local_scores, axis=0)
                local_indices = np.concatenate(all_local_indices, axis=0)
                
            except Exception as e:
                self.logger.error(f"Error during search on rank {self.rank}: {str(e)}")
                success_tensor.fill_(False)
                local_scores = np.array([])
                local_indices = np.array([])
            
            # Final status check
            success_list = [torch.zeros_like(success_tensor) for _ in range(self.world_size)]
            dist.all_gather(success_list, success_tensor)
            all_succeeded = all(t.item() for t in success_list)
            
            if not all_succeeded:
                if self.rank == 0:
                    failed_ranks = [i for i, t in enumerate(success_list) if not t.item()]
                    self.logger.error(f"Failed ranks: {failed_ranks}")
                return None
            
            # Prepare for gathering results
            max_queries = queries_per_gpu + (1 if self.rank < len(queries) % self.world_size else 0)
            padding_size = max_queries - len(local_scores)
            
            if padding_size > 0:
                local_scores = np.pad(local_scores, ((0, padding_size), (0, 0)), mode='constant')
                local_indices = np.pad(local_indices, ((0, padding_size), (0, 0)), mode='constant')
            
            # Final gathering of results
            gathered_scores = [torch.zeros((max_queries, top_k), dtype=torch.float32) for _ in range(self.world_size)]
            gathered_indices = [torch.zeros((max_queries, top_k), dtype=torch.int64) for _ in range(self.world_size)]
            
            dist.barrier()
            dist.all_gather(gathered_scores, torch.from_numpy(local_scores))
            dist.all_gather(gathered_indices, torch.from_numpy(local_indices))
            
            if self.rank == 0:
                # Log final statistics
                total_time = time.time() - start_time
                overall_qps = len(queries) / total_time if total_time > 0 else 0
                self.logger.info(
                    f"\nRetrieval completed:"
                    f"\nTotal time: {total_time:.1f} seconds"
                    f"\nOverall speed: {overall_qps:.1f} queries/sec"
                    f"\nTotal queries processed: {len(queries)}"
                )
                
                # Merge results
                results = []
                query_idx = 0
                
                for gpu_idx in range(self.world_size):
                    gpu_queries = queries_per_gpu + (1 if gpu_idx < len(queries) % self.world_size else 0)
                    scores = gathered_scores[gpu_idx][:gpu_queries]
                    indices = gathered_indices[gpu_idx][:gpu_queries]
                    
                    for q_scores, q_indices in zip(scores, indices):
                        query_results = []
                        for score, idx in zip(q_scores, q_indices):
                            if 0 <= idx < len(self.text_mapping):
                                query_results.append({
                                    'text': self.text_mapping[idx],
                                    'score': float(score)
                                })
                        results.append({
                            'query': queries[query_idx],
                            'retrieved_docs': query_results
                        })
                        query_idx += 1
                
                return results
            return None
            
        except Exception as e:
            self.logger.error(f"Critical error in retrieve on rank {self.rank}: {str(e)}")
            dist.barrier()
            return None

class QueryDataset(Dataset):
    """Dataset for processing queries with robust validation"""
    def __init__(self, queries: List[str], tokenizer, logger=None):
        self.logger = logger or logging.getLogger()
        self.tokenizer = tokenizer
        self.queries = []
        skipped = 0
        
        for idx, q in enumerate(queries):
            if not isinstance(q, str):
                self.logger.warning(f"Skipping non-string query at index {idx}: {type(q)}")
                skipped += 1
                continue
                
            cleaned = q.strip()
            if not cleaned:
                self.logger.warning(f"Skipping empty query at index {idx}")
                skipped += 1
                continue
                
            self.queries.append(cleaned)

        if skipped > 0:
            self.logger.warning(f"Skipped {skipped} invalid queries out of {len(queries)}")

        if not self.queries:
            raise ValueError("No valid queries found after filtering")
            
        self.logger.info(f"Initialized dataset with {len(self.queries)} valid queries")
        
        if self.queries:
            self.logger.info(f"First 3 queries: {self.queries[:3]}")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        if idx >= len(self.queries):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.queries)}")
            
        query = self.queries[idx]
        try:
            inputs = self.tokenizer(
                query,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors=None
            )
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'query': query
            }
        except Exception as e:
            self.logger.error(f"Error tokenizing query at index {idx}: '{query}'. Error: {str(e)}")
            raise

def main_worker(rank, world_size, args):
    # Set the device for this process
    torch.cuda.set_device(rank)
    setup_distributed(rank, world_size)
    logger = setup_logging(args.output_dir, rank)
    
    try:
        # Load HotpotQA data and extract queries on all ranks
        logger.info(f"Loading HotpotQA data from {args.hotpot_path}")
        with open(args.hotpot_path, 'r') as f:
            hotpot_data = json.load(f)
        
        # Extract queries and metadata
        queries = []
        query_info = []
        
        for item_idx, item in enumerate(hotpot_data):
            # Process main question
            question = item.get('question', '')
            if isinstance(question, str) and question.strip():
                clean_question = question.strip()
                queries.append(clean_question)
                query_info.append({
                    'type': 'main',
                    'question_idx': item_idx,
                    'sub_idx': None
                })
                
                # Process subqueries
                for sub_idx, sub_query in enumerate(item.get('sub_queries', [])):
                    sub_q = sub_query.get('sub_query', '').strip()
                    if isinstance(sub_q, str) and sub_q:
                        queries.append(sub_q)
                        query_info.append({
                            'type': 'sub',
                            'question_idx': item_idx,
                            'sub_idx': sub_idx
                        })
        
        # Log query statistics
        main_queries = sum(1 for info in query_info if info['type'] == 'main')
        sub_queries = sum(1 for info in query_info if info['type'] == 'sub')
        logger.info(f"Rank {rank}: Extracted {main_queries} main queries and {sub_queries} sub-queries")
        if queries:
            logger.info(f"Rank {rank}: Sample queries: {queries[:3]}")

        # Initialize retriever and process queries
        retriever = WikiRetriever(
            args.faiss_path,
            args.text_mapping_path,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            embeddings_cache_dir=args.embeddings_cache_dir
        )

        # Retrieve documents
        logger.info(f"Rank {rank}: Starting retrieval for {len(queries)} queries")
        retrieval_results = retriever.retrieve(queries, top_k=5)

        # Save results on rank 0
        if retrieval_results is not None and rank == 0:
            # Validate results
            if len(retrieval_results) != len(queries):
                raise ValueError(f"Retrieved {len(retrieval_results)} results but had {len(queries)} queries")
                
            output_path = os.path.join(args.output_dir, "wiki_retrieval_results.json")
            logger.info(f"Saving results to {output_path}")
            
            # Map results back using query_info
            result_map = {}  # Map to store results by question_idx
            for idx, (result, info) in enumerate(zip(retrieval_results, query_info)):
                if info['type'] == 'main':
                    result_map[info['question_idx']] = result['retrieved_docs']
            
            # Update original data
            for idx, item in enumerate(hotpot_data):
                if idx in result_map:
                    item['wiki_retrieved_docs'] = result_map[idx]
                else:
                    logger.warning(f"No results found for question {idx}")
            
            with open(output_path, 'w') as f:
                json.dump(hotpot_data, f, indent=2)
            
            logger.info(f"Retrieval process completed successfully. Saved results for {len(result_map)} questions")

    except Exception as e:
        logger.error(f"Error during retrieval on rank {rank}: {str(e)}", exc_info=True)
        raise
    finally:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Distributed Wikipedia document retrieval for HotpotQA queries')
    parser.add_argument('--hotpot_path', type=str,
                      default="/shared/eng/pj20/firas_data/datasets/hotpotqa/hotpot_with_subqueries.json")
    parser.add_argument('--faiss_path', type=str,
                      default="/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding/wikipedia_embeddings.faiss")
    parser.add_argument('--text_mapping_path', type=str,
                      default="/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding/text_mapping.json")
    parser.add_argument('--output_dir', type=str,
                      default="/shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                      help='Number of GPUs to use')
    parser.add_argument('--embeddings_cache_dir', type=str,
                      default="/shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval/embeddings_cache",
                      help='Directory to cache query embeddings')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.embeddings_cache_dir, exist_ok=True)
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # Use GPUs with most free memory
        # In your case, might want to set: "3,6,7"
        os.environ['CUDA_VISIBLE_DEVICES'] = "3,6,7"
    
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
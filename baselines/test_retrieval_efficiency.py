# Move these to the very top of the file, before any other imports
import os
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads

import time
import numpy as np
from typing import Dict
import faiss
from tqdm import tqdm
import argparse

# After FAISS import, verify thread settings
faiss.omp_set_num_threads(1)         # FAISS threads
print(f"FAISS number of threads: {faiss.omp_get_max_threads()}")
print(f"OpenMP number of threads: {os.environ.get('OMP_NUM_THREADS')}")

class RetrievalExperiment:
    def __init__(self, num_docs: int = 35_000_000, dense_dim: int = 768, theme_dim: int = 298):
        """
        Initialize experiment with synthetic data matching real-world scale
        
        Args:
            num_docs: Number of documents (default: 35M to match Wikipedia corpus)
            dense_dim: Dimension of dense embeddings (default: 768 for Contriever)
            theme_dim: Dimension of theme embeddings (default: 298 for theme classifier)
        """
        print(f"Initializing synthetic indices with {num_docs:,} documents...")
        
        # Create synthetic document embeddings
        print("Creating dense index...")
        dense_vectors = np.random.randn(num_docs, dense_dim).astype('float32')
        dense_vectors = dense_vectors / np.linalg.norm(dense_vectors, axis=1)[:, np.newaxis]
        self.dense_index = faiss.IndexFlatIP(dense_dim)
        self.dense_index.add(dense_vectors)
        
        print("Creating theme index...")
        theme_vectors = np.random.randn(num_docs, theme_dim).astype('float32')
        theme_vectors = theme_vectors / np.linalg.norm(theme_vectors, axis=1)[:, np.newaxis]
        self.theme_index = faiss.IndexFlatL2(theme_dim)
        self.theme_index.add(theme_vectors)
        
        self.num_docs = num_docs
        self.dense_dim = dense_dim
        self.theme_dim = theme_dim
        
        print("Initialization complete!")
        print(f"Dense index: {dense_dim}d vectors, {self.dense_index.ntotal:,} total")
        print(f"Theme index: {theme_dim}d vectors, {self.theme_index.ntotal:,} total")
    
    def run_experiment(self, 
                      n_queries: int = 1000, 
                      theme_top_k: int = 100000, 
                      dense_top_k: int = 100, 
                      n_trials: int = 5,
                      dense_only_time: float = None) -> Dict:
        """
        Run comparative experiment between dense-only and theme-scoped retrieval
        
        Args:
            n_queries: Number of random queries to test
            theme_top_k: Number of candidates to retrieve in theme phase
            dense_top_k: Number of final results to retrieve
            n_trials: Number of trials for timing
            dense_only_time: If provided, skip dense-only testing
        """
        # Generate random normalized queries
        dense_queries = np.random.randn(n_queries, self.dense_dim).astype('float32')
        dense_queries = dense_queries / np.linalg.norm(dense_queries, axis=1)[:, np.newaxis]
        
        theme_queries = np.random.randn(n_queries, self.theme_dim).astype('float32')
        theme_queries = theme_queries / np.linalg.norm(theme_queries, axis=1)[:, np.newaxis]
        
        dense_times = [dense_only_time] if dense_only_time is not None else []
        scoped_times = []
        
        print(f"\nRunning {n_trials} trials with {n_queries:,} queries each...")
        
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            
            # Skip dense-only if time was provided
            # if dense_only_time is None:
            # Dense-only retrieval timing
            start_time = time.perf_counter()
            for i, query in enumerate(tqdm(dense_queries, desc="Dense-only retrieval")):
                query_start = time.perf_counter()
                _, _ = self.dense_index.search(query.reshape(1, -1), dense_top_k)
                query_time = time.perf_counter() - query_start
                if i == 0:  # Print timing for first query
                    print(f"\nFirst dense query time: {query_time:.3f}s")
                    print(f"Index size: {self.dense_index.ntotal:,} vectors")
                    print(f"FAISS threads: {faiss.omp_get_max_threads()}")
            dense_time = time.perf_counter() - start_time
            dense_times.append(dense_time)
            print(f"Dense-only time: {dense_time:.2f}s")
            print(f"Average time per query: {dense_time/n_queries:.3f}s")
            
            # Theme-scoped retrieval timing
            start_time = time.perf_counter()
            for dense_q, theme_q in tqdm(zip(dense_queries, theme_queries), 
                                       desc="Theme-scoped retrieval", 
                                       total=len(dense_queries)):
                # First get theme candidates
                _, theme_doc_ids = self.theme_index.search(
                    theme_q.reshape(1, -1), k=theme_top_k
                )
                
                # Then search in dense space using reconstruct_n for batch reconstruction
                dense_candidates = np.empty((len(theme_doc_ids[0]), self.dense_dim), dtype='float32')
                self.dense_index.reconstruct_n(
                    int(theme_doc_ids[0][0]),  # starting index
                    len(theme_doc_ids[0]),     # number of vectors
                    dense_candidates           # output array
                )
                
                temp_index = faiss.IndexFlatIP(self.dense_dim)
                temp_index.add(dense_candidates)
                _, _ = temp_index.search(dense_q.reshape(1, -1), k=min(dense_top_k, len(dense_candidates)))
            
            scoped_time = time.perf_counter() - start_time
            scoped_times.append(scoped_time)
            print(f"Theme-scoped time: {scoped_time:.2f}s")
        
        # Calculate statistics
        avg_dense_time = np.mean(dense_times) if dense_times else dense_only_time
        avg_scoped_time = np.mean(scoped_times)
        time_saved = (avg_dense_time - avg_scoped_time) / avg_dense_time * 100
        
        results = {
            'dense_time': avg_dense_time,
            'scoped_time': avg_scoped_time,
            'time_saved_percentage': time_saved,
            'dense_std': np.std(dense_times) if len(dense_times) > 1 else 0,
            'scoped_std': np.std(scoped_times),
            'n_queries': n_queries,
            'theme_top_k': theme_top_k,
            'dense_top_k': dense_top_k,
            'n_trials': n_trials,
            'corpus_size': self.num_docs
        }
        
        return results

def print_results(results: Dict):
    """Pretty print the experimental results"""
    print("\nRetrieval Performance Comparison")
    print("-" * 60)
    print(f"Configuration:")
    print(f"  Corpus size: {results['corpus_size']:,} documents")
    print(f"  Number of queries per trial: {results['n_queries']:,}")
    print(f"  Theme candidates (top-k): {results['theme_top_k']:,}")
    print(f"  Dense results (top-k): {results['dense_top_k']}")
    print(f"  Number of trials: {results['n_trials']}")
    print("\nResults:")
    print(f"  Dense-only retrieval time:     {results['dense_time']:.2f}s ± {results['dense_std']:.2f}s")
    print(f"  Theme-scoped retrieval time:   {results['scoped_time']:.2f}s ± {results['scoped_std']:.2f}s")
    print(f"  Time saved:                    {results['time_saved_percentage']:.2f}%")
    print(f"  Average time per query (dense):     {(results['dense_time']/results['n_queries'])*1000:.2f}ms")
    print(f"  Average time per query (scoped):    {(results['scoped_time']/results['n_queries'])*1000:.2f}ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_docs', type=int, default=35_000_000)
    parser.add_argument('--n_queries', type=int, default=100)
    parser.add_argument('--theme_top_k', type=int, default=100000)
    parser.add_argument('--dense_top_k', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--dense_only_time', type=float)  # Add previous dense-only time
    args = parser.parse_args()
    
    experiment = RetrievalExperiment(num_docs=args.num_docs)
    results = experiment.run_experiment(
        n_queries=args.n_queries,
        theme_top_k=args.theme_top_k,
        dense_top_k=args.dense_top_k,
        n_trials=args.n_trials,
        dense_only_time=args.dense_only_time
    )
    print_results(results)

if __name__ == "__main__":
    main()
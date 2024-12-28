import json
import logging
from tqdm import tqdm
import time
import os
from datetime import datetime
from claude_api import get_claude_response
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import queue

class TripleVerifier:
    def __init__(self,
                 input_file: str,
                 output_dir: str,
                 num_threads: int = 6,
                 checkpoint_interval: int = 100,
                 llm: str = "sonnet"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.num_threads = num_threads
        self.checkpoint_interval = checkpoint_interval
        self.llm = llm
        
        # Output files
        self.detailed_output = os.path.join(output_dir, "verified_triples_detailed.json")
        self.clean_output = os.path.join(output_dir, "verified_triples_clean.json")
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        
        # Processing state
        self.results = []
        self.total_samples = 0
        self.processed_samples = 0
        self.last_checkpoint = 0
        
        # Thread safety
        self.results_lock = Lock()
        self.checkpoint_lock = Lock()
        self.sample_queue = queue.Queue()
        self.progress_lock = Lock()
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_prompt(self, text: str, triples: str) -> str:
        return f"""Task: Verify and complete relationship triples extracted from text. Ensure all triples are accurate and capture important relationships mentioned in the text. Keep correct triples, fix any errors, and add any missing important relationships.

Text: {text}
Triples: {triples}

Output only the final, corrected, and complete set of triples in the format (S> subject| P> predicate| O> object). Include both verified correct triples and any important missing relationships. Separate triples with commas. Do not include any explanations or additional text."""

    def process_sample(self, sample: dict) -> dict:
        try:
            prompt = self.create_prompt(
                text=sample['text'],
                triples=sample['generated_triple']
            )
            
            response = get_claude_response(llm=self.llm, prompt=prompt)
            response = response.strip()
            
            # Create new sample with results
            processed_sample = {
                'text': sample['text'],
                'original_triple': sample['generated_triple'],
                'generated_triple': response,
                'modification_type': 'modified' if response != sample['generated_triple'] else 'unchanged'
            }
            
            return processed_sample
            
        except Exception as e:
            self.logger.error(f"Error processing sample: {str(e)}")
            return {**sample, 'error': str(e)}

    def save_checkpoint(self, force: bool = False):
        with self.checkpoint_lock:
            if force or (self.processed_samples - self.last_checkpoint) >= self.checkpoint_interval:
                # Save detailed output
                checkpoint_file = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                
                try:
                    # Save detailed results
                    with open(self.detailed_output, 'w', encoding='utf-8') as f:
                        json.dump(self.results, f, indent=2, ensure_ascii=False)
                    
                    # Save clean results
                    clean_results = [{
                        'text': item['text'],
                        'generated_triple': item['generated_triple']
                    } for item in self.results]
                    with open(self.clean_output, 'w', encoding='utf-8') as f:
                        json.dump(clean_results, f, indent=2, ensure_ascii=False)
                    
                    # Save checkpoint
                    checkpoint_data = {
                        'processed_samples': self.processed_samples,
                        'results': self.results,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    
                    self.last_checkpoint = self.processed_samples
                    self.logger.info(f"Checkpoint saved: {checkpoint_file}")
                    self._cleanup_old_checkpoints()
                    
                except Exception as e:
                    self.logger.error(f"Error saving checkpoint: {str(e)}")

    def _cleanup_old_checkpoints(self, keep_num: int = 5):
        try:
            checkpoints = sorted([
                os.path.join(self.checkpoint_dir, f)
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith('checkpoint_')
            ], reverse=True)
            
            for old_checkpoint in checkpoints[keep_num:]:
                os.remove(old_checkpoint)
        except Exception as e:
            self.logger.error(f"Error cleaning up old checkpoints: {str(e)}")

    def find_latest_checkpoint(self) -> dict:
        try:
            checkpoints = [
                f for f in os.listdir(self.checkpoint_dir)
                if f.startswith('checkpoint_')
            ]
            
            if not checkpoints:
                return None
            
            latest_checkpoint = max(checkpoints)
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"Found checkpoint: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return None

    def worker(self, worker_id: int, pbar: tqdm):
        """Worker function for processing samples in parallel"""
        while True:
            try:
                # Get next sample from queue
                sample = self.sample_queue.get_nowait()
            except queue.Empty:
                break
                
            try:
                processed_sample = self.process_sample(sample)
                
                # Safely add results and update progress
                with self.results_lock:
                    self.results.append(processed_sample)
                    self.processed_samples += 1
                    self.save_checkpoint()
                
                with self.progress_lock:
                    pbar.update(1)
                    
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error processing sample: {str(e)}")
            
            finally:
                self.sample_queue.task_done()

    def process_dataset(self, resume: bool = True):
        try:
            # Load data
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.total_samples = len(data)
            start_idx = 0
            
            # Try to resume from checkpoint if requested
            if resume:
                checkpoint_data = self.find_latest_checkpoint()
                if checkpoint_data:
                    self.results = checkpoint_data['results']
                    self.processed_samples = checkpoint_data['processed_samples']
                    start_idx = self.processed_samples
                    self.last_checkpoint = self.processed_samples
                    
                    self.logger.info(f"Resuming from checkpoint. "
                                   f"Processed samples: {self.processed_samples}")
            
            # Add remaining samples to queue
            for sample in data[start_idx:]:
                self.sample_queue.put(sample)
            
            # Process samples with thread pool
            with tqdm(total=self.total_samples, initial=self.processed_samples,
                     desc="Processing Triples") as pbar:
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    # Start worker threads
                    workers = [
                        executor.submit(self.worker, i, pbar)
                        for i in range(self.num_threads)
                    ]
                    
                    # Wait for all workers to complete
                    for worker in workers:
                        worker.result()
            
            # Save final results
            self.save_checkpoint(force=True)
            self.logger.info(f"Completed processing. Results saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in process_dataset: {str(e)}")
            raise

def main():
    verifier = TripleVerifier(
        input_file="/shared/eng/pj20/firas_data/datasets/hotpotqa/generated_triples/hotpot_triples.json",
        output_dir="/shared/eng/pj20/firas_data/datasets/hotpotqa/verified_triples",
        num_threads=10,  # Adjust based on your system
        checkpoint_interval=100,
        llm="sonnet"
    )
    verifier.process_dataset(resume=True)

if __name__ == "__main__":
    main()
import json
import logging
import os
from datetime import datetime
from typing import Dict
from tqdm import tqdm
from claude_api import get_claude_response
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import queue
import time

class TripleGenerator:
    def __init__(self, 
                 input_file: str, 
                 output_dir: str,
                 num_threads: int = 8,
                 checkpoint_interval: int = 100):
        self.input_file = input_file
        self.output_dir = output_dir
        self.num_threads = num_threads
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, "selfrag_triples.json")
        
        # Thread safety mechanisms
        self.results_lock = Lock()
        self.progress_lock = Lock()
        self.sample_queue = queue.Queue()
        self.results = []
        self.processed_count = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'generation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = 0
        self.checkpoint_lock = Lock()
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def create_prompt(self, text: str) -> str:
#         return f"""Extract relationship triples from the given text. Each triple should have exactly one subject (S>), one predicate (P>), and one object (O>).

# Rules:
# 1. Extract as many meaningful triples as possible
# 2. Each triple must be in format: (S> subject| P> predicate| O> object)
# 3. Multiple triples should be separated by commas
# 4. Avoid using pronouns (it/he/she) - always use the actual names
# 5. Keep all entities in their original case (uppercase/lowercase)
# 6. Make predicates clear and specific

# Example Input:
# "William Gerald Standridge (November 27, 1953 – April 12, 2014) was an American stock car racing driver. He was a competitor in the NASCAR Winston Cup Series and Busch Series."

# Example Output:
# (S> William gerald standridge| P> Nationality| O> American),
# (S> William gerald standridge| P> Occupation| O> Stock car racing driver),
# (S> William gerald standridge| P> Competitor| O> Busch series),
# (S> William gerald standridge| P> Competitor| O> Nascar winston cup series),
# (S> William gerald standridge| P> Birth date| O> November 27, 1953),
# (S> William gerald standridge| P> Death date| O> April 12, 2014)

# Input Text: {text}

# Output only the triples, nothing else."""
        return f"""Extract relationship triples from the text. Format: (S> subject| P> predicate| O> object)
Rules:
1. Extract all meaningful triples
2. Each triple must be in format: (S> subject| P> predicate| O> object)
3. No pronouns, Avoid using pronouns (it/he/she) - always use the actual names
4. Keep original case
5. Separate triples with commas
6. Output only the triples, nothing else.

Example Input: "William Gerald Standridge was an American racing driver."
Example Output: (S> William gerald standridge| P> Nationality| O> American), (S> William gerald standridge| P> Occupation| O> Racing driver)

Input Text: {text}
Output triples:"""


    def process_sample(self, text: str) -> Dict:
        """Process a single sample with retry mechanism"""
        max_retries = 3
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                prompt = self.create_prompt(text)
                response = get_claude_response(llm="sonnet", prompt=prompt)
                return {
                    "text": text,
                    "generated_triple": response.strip()
                }
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
        
        self.logger.error(f"Error processing text after {max_retries} attempts: {str(last_exception)}")
        return {
            "text": text,
            "generated_triple": "",
            "error": str(last_exception)
        }

    def worker(self, worker_id: int):
        """Worker function for processing individual samples"""
        while True:
            try:
                # Get next sample from queue
                try:
                    text = self.sample_queue.get_nowait()
                except queue.Empty:
                    break
                
                # Process single sample
                result = self.process_sample(text)
                
                # Update results and progress
                with self.results_lock:
                    self.results.append(result)
                    self.processed_count += 1
                    
                    # Check for checkpoint
                    if self.processed_count % self.checkpoint_interval == 0:
                        self.save_checkpoint()
                
                # Mark task as done
                self.sample_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
                break

    def save_checkpoint(self, force: bool = False):
        """Enhanced checkpoint saving with cleanup"""
        with self.checkpoint_lock:
            if force or (self.processed_count - self.last_checkpoint) >= self.checkpoint_interval:
                checkpoint_file = os.path.join(
                    self.checkpoint_dir, 
                    f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                
                checkpoint_data = {
                    'processed_count': self.processed_count,
                    'results': self.results,
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    
                    self.last_checkpoint = self.processed_count
                    self.logger.info(f"Checkpoint saved: {checkpoint_file}")
                    
                    # Cleanup old checkpoints
                    self._cleanup_old_checkpoints()
                except Exception as e:
                    self.logger.error(f"Error saving checkpoint: {str(e)}")

    def _cleanup_old_checkpoints(self, keep_num: int = 5):
        """Cleanup old checkpoints keeping only the most recent ones"""
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
        """Find and load the latest checkpoint"""
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

    def process_dataset(self, resume: bool = True):
        try:
            # Load documents
            with open(self.input_file, 'r') as f:
                documents = json.load(f)
            
            start_idx = 0
            
            # Try to resume from checkpoint if requested
            if resume:
                checkpoint_data = self.find_latest_checkpoint()
                if checkpoint_data:
                    self.results = checkpoint_data['results']
                    self.processed_count = checkpoint_data['processed_count']
                    start_idx = self.processed_count
                    self.last_checkpoint = self.processed_count
                    
                    self.logger.info(f"Resuming from checkpoint. "
                                   f"Processed documents: {self.processed_count}")
            
            self.logger.info(f"Processing {len(documents)} documents with {self.num_threads} threads...")
            
            # Add remaining documents to queue
            for doc in documents[start_idx:]:
                self.sample_queue.put(doc)
            
            # Create progress bar
            pbar = tqdm(total=len(documents), initial=self.processed_count, 
                       desc="Processing documents")
            last_count = self.processed_count
            
            # Start worker threads
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit workers
                futures = [
                    executor.submit(self.worker, i)
                    for i in range(self.num_threads)
                ]
                
                # Update progress bar
                while self.sample_queue.unfinished_tasks > 0:
                    current_count = self.processed_count
                    if current_count > last_count:
                        pbar.update(current_count - last_count)
                        last_count = current_count
                    time.sleep(0.1)
                
                # Wait for all workers to complete
                for future in futures:
                    future.result()
            
            pbar.close()
            
            # Save final results
            with open(self.output_file, 'w') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Processing completed. Results saved to {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Error in process_dataset: {str(e)}")
            raise

def main():
    generator = TripleGenerator(
        input_file="/shared/eng/pj20/firas_data/datasets/selfrag/documents.json",
        output_dir="/shared/eng/pj20/firas_data/datasets/selfrag/generated_triples",
        num_threads=12,
        checkpoint_interval=2000
    )
    generator.process_dataset(resume=True)

if __name__ == "__main__":
    main()
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
import shutil

class TripleVerifier:
    def __init__(self,
                 input_file: str,
                 output_dir: str,
                 num_threads: int = 6,
                 checkpoint_interval: int = 1000,
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
        self.error_count = 0
        self.max_errors = 50  # Maximum number of consecutive errors before raising alert
        
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
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'verifier.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_prompt(self, text: str, triples: str) -> str:
        return f"""Task: Verify and complete relationship triples extracted from text. Ensure all triples are accurate and capture important relationships mentioned in the text. Keep correct triples, fix any errors, and add any missing important relationships.

Text: {text}
Triples: {triples}

Output only the final, corrected, and complete set of triples in the format (S> subject| P> predicate| O> object). Include both verified correct triples and any important missing relationships. Separate triples with commas. Do not include any explanations or additional text."""

    def safe_file_write(self, filepath: str, data: any, max_retries: int = 3):
        """Safely write data to file with retry logic"""
        temp_file = filepath + '.tmp'
        
        for attempt in range(max_retries):
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Atomic replace
                shutil.move(temp_file, filepath)
                return True
                
            except Exception as e:
                wait_time = 2 ** attempt
                self.logger.warning(f"Write attempt {attempt + 1} failed for {filepath}: {str(e)}")
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to write file after {max_retries} attempts: {filepath}")
                    return False

    def check_disk_space(self, required_mb: int = 100) -> bool:
        """Check if there's enough disk space"""
        try:
            stats = os.statvfs(self.output_dir)
            available_mb = (stats.f_bavail * stats.f_frsize) / (1024 * 1024)
            if available_mb < required_mb:
                self.logger.error(f"Low disk space: {available_mb}MB available, {required_mb}MB required")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking disk space: {str(e)}")
            return False

    def process_sample(self, sample: dict) -> dict:
        try:
            prompt = self.create_prompt(
                text=sample['text'],
                triples=sample['generated_triple']
            )
            
            response = get_claude_response(llm=self.llm, prompt=prompt)
            response = response.strip()
            
            processed_sample = {
                'text': sample['text'],
                'original_triple': sample['generated_triple'],
                'generated_triple': response,
                'modification_type': 'modified' if response != sample['generated_triple'] else 'unchanged',
                'timestamp': datetime.now().isoformat()
            }
            
            self.error_count = 0  # Reset error count on successful processing
            return processed_sample
            
        except Exception as e:
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.logger.critical(f"Too many consecutive errors ({self.error_count}). Please check the system.")
            
            self.logger.error(f"Error processing sample: {str(e)}")
            return {**sample, 'error': str(e)}

    def save_checkpoint(self, force: bool = False):
        """Save checkpoint with improved error handling"""
        with self.checkpoint_lock:
            if not force and (self.processed_samples - self.last_checkpoint) < self.checkpoint_interval:
                return
                
            if not self.check_disk_space():
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{timestamp}.json"
            )
            
            # Prepare checkpoint data
            checkpoint_data = {
                'processed_samples': self.processed_samples,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save files
            files_to_save = [
                (self.detailed_output, self.results),
                (self.clean_output, [{
                    'text': item['text'],
                    'generated_triple': item['generated_triple']
                } for item in self.results]),
                (checkpoint_file, checkpoint_data)
            ]
            
            for filepath, data in files_to_save:
                if not self.safe_file_write(filepath, data):
                    self.logger.error(f"Failed to save {filepath}")
                    return
            
            self.last_checkpoint = self.processed_samples
            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self, keep_num: int = 5):
        """Clean up old checkpoints while keeping the specified number"""
        try:
            checkpoints = sorted([
                os.path.join(self.checkpoint_dir, f)
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith('checkpoint_')
            ], reverse=True)
            
            for old_checkpoint in checkpoints[keep_num:]:
                try:
                    os.remove(old_checkpoint)
                except Exception as e:
                    self.logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {str(e)}")
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
            
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.logger.info(f"Found checkpoint: {checkpoint_path}")
                return checkpoint_data
            except Exception as e:
                self.logger.error(f"Error reading checkpoint {checkpoint_path}: {str(e)}")
                # Try the next most recent checkpoint
                if len(checkpoints) > 1:
                    checkpoints.remove(latest_checkpoint)
                    return self.find_latest_checkpoint()
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding latest checkpoint: {str(e)}")
            return None

    def worker(self, worker_id: int, pbar: tqdm):
        """Worker function for processing samples in parallel"""
        while True:
            try:
                sample = self.sample_queue.get_nowait()
            except queue.Empty:
                break
                
            try:
                processed_sample = self.process_sample(sample)
                
                with self.results_lock:
                    self.results.append(processed_sample)
                    self.processed_samples += 1
                    self.save_checkpoint()
                
                with self.progress_lock:
                    pbar.update(1)
                    
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
            
            finally:
                self.sample_queue.task_done()

    def process_dataset(self, resume: bool = True):
        """Process the entire dataset with improved error handling"""
        try:
            # Load input data
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.total_samples = len(data)
            start_idx = 0
            
            # Resume from checkpoint if requested
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
                    workers = [
                        executor.submit(self.worker, i, pbar)
                        for i in range(self.num_threads)
                    ]
                    
                    for worker in workers:
                        worker.result()
            
            # Save final results
            self.save_checkpoint(force=True)
            self.logger.info(f"Processing completed. Results saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in process_dataset: {str(e)}")
            raise

def main():
    verifier = TripleVerifier(
        input_file="/shared/eng/pj20/firas_data/datasets/hotpotqa/generated_triples/hotpot_triples.json",
        output_dir="/shared/eng/pj20/firas_data/datasets/hotpotqa/verified_triples",
        num_threads=10,
        checkpoint_interval=1000,
        llm="sonnet"
    )
    verifier.process_dataset(resume=True)

if __name__ == "__main__":
    main()
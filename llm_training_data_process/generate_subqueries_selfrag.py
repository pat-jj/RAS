#/shared/eng/pj20/firas_data/datasets/selfrag/train.jsonl

import json
import logging
from copy import deepcopy
from tqdm import tqdm
import time
import os
from datetime import datetime
from claude_api import get_claude_response
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import queue

class SubqueryGenerator:
    def __init__(self, 
                 input_file: str, 
                 output_file: str = "selfrag_with_subqueries.json",
                 checkpoint_dir: str = "checkpoints",
                 checkpoint_interval: int = 10,
                 num_threads: int = 4):
        self.input_file = input_file
        self.output_file = output_file
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.num_threads = num_threads
        
        self.results = []
        self.total_samples = 0
        self.processed_samples = 0
        self.last_checkpoint = 0
        
        # Thread safety mechanisms
        self.results_lock = Lock()
        self.checkpoint_lock = Lock()
        self.sample_queue = queue.Queue()
        self.progress_lock = Lock()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_paragraphs_and_answer(self, output: str) -> tuple:
        """Extract paragraphs and answer from SelfRAG output format"""
        paragraphs = []
        answer = None
        
        if '[No Retrieval]' in output:
            # Extract answer for no-retrieval cases
            start_idx = output.find('[No Retrieval]') + len('[No Retrieval]')
            end_idx = output.find('[Utility:')
            answer = output[start_idx:end_idx].strip()
            return [], answer
            
        # Extract paragraphs
        current_pos = 0
        while True:
            start_idx = output.find('<paragraph>', current_pos)
            if start_idx == -1:
                break
            end_idx = output.find('</paragraph>', start_idx)
            if end_idx == -1:
                break
            paragraph = output[start_idx + len('<paragraph>'):end_idx].strip()
            paragraphs.append(paragraph)
            current_pos = end_idx + 1

        # Extract answer
        for marker in ['[Relevant]', '[Irrelevant]']:
            if marker in output:
                start_idx = output.find(marker) + len(marker)
                end_idx = output.find('[Utility:', start_idx)
                if end_idx != -1:
                    answer = output[start_idx:end_idx].strip()
                    break

        return paragraphs, answer

    def generate_prompt(self, instruction: str, paragraphs: list) -> str:
        prompt = f"""Given this instruction and retrieved paragraphs, determine if we need multiple sub-queries to retrieve these paragraphs or if a single query would suffice.

Instruction: {instruction}

Retrieved Paragraphs:
{chr(10).join(f'[{i+1}] {p}' for i, p in enumerate(paragraphs))}

If a single query can retrieve all paragraphs effectively, respond with:
[SINGLE]
<query>Your proposed query here</query>

If multiple queries are needed, respond with:
[MULTIPLE]
Then generate ONE sub-query per paragraph that would help retrieve that specific information.
Write only the sub-queries, one per line.

Respond with either [SINGLE] or [MULTIPLE] format, nothing else."""
        return prompt

    def process_sample(self, sample: dict) -> dict:
        try:
            processed_sample = deepcopy(sample)
            paragraphs, answer = self.extract_paragraphs_and_answer(sample['output'])
            
            if not paragraphs:  # No retrieval needed
                processed_sample['retrieval_type'] = 'no_retrieval'
                processed_sample['answer'] = answer
                processed_sample['sub_queries'] = []
                return processed_sample

            # Generate sub-queries for paragraphs
            prompt = self.generate_prompt(sample['instruction'], paragraphs)
            response = get_claude_response(llm="sonnet", prompt=prompt)
            
            if '[SINGLE]' in response:
                # Extract single query
                start_idx = response.find('<query>') + len('<query>')
                end_idx = response.find('</query>')
                query = response[start_idx:end_idx].strip()
                
                processed_sample['retrieval_type'] = 'single_query'
                processed_sample['main_query'] = query
                processed_sample['sub_queries'] = []
                
            else:  # [MULTIPLE] case
                queries = [q.strip() for q in response.split('\n') if q.strip() and not q.startswith('[')]
                sub_queries = []
                
                for query, paragraph in zip(queries, paragraphs):
                    sub_queries.append({
                        'sub_query': query,
                        'supporting_doc': paragraph
                    })
                
                processed_sample['retrieval_type'] = 'multiple_queries'
                processed_sample['sub_queries'] = sub_queries
            
            processed_sample['answer'] = answer
            return processed_sample
            
        except Exception as e:
            self.logger.error(f"Error processing sample: {str(e)}")
            return sample

    def save_checkpoint(self, force: bool = False):
        with self.checkpoint_lock:
            if (force or 
                (self.processed_samples - self.last_checkpoint) >= self.checkpoint_interval):
                checkpoint_file = os.path.join(
                    self.checkpoint_dir, 
                    f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                
                checkpoint_data = {
                    'processed_samples': self.processed_samples,
                    'results': self.results,
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
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
            with open(self.input_file, "r") as f:
                    data = [json.loads(line) for line in f]
            
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
                     desc="Overall Progress") as pbar:
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
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Completed processing. Results saved to {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Error in process_dataset: {str(e)}")
            raise

def main():
    generator = SubqueryGenerator(
        input_file="/shared/eng/pj20/firas_data/datasets/selfrag/train.jsonl",
        output_file="/shared/eng/pj20/firas_data/datasets/selfrag/selfrag_with_subqueries.json",
        checkpoint_dir="/shared/eng/pj20/firas_data/datasets/selfrag/checkpoints",
        checkpoint_interval=5000,
        num_threads=6
    )
    generator.process_dataset(resume=True)

if __name__ == "__main__":
    main()
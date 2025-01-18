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
                 output_file: str = "hotpot_with_subqueries.json",
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

    def generate_prompt(self, 
                    main_question: str, 
                    current_doc: str,
                    topic: str,
                    previous_queries: list = None) -> str:
        prompt = f"""Given this main question and a supporting document, generate a simple sub-query (a question) that will help retrieve information from the document to answer the main question.

    Main Question: {main_question}

    Current Document ({topic}):
    {current_doc}

    """
        if previous_queries:
            prompt += "\nPreviously generated sub-queries:\n"
            for prev in previous_queries:
                prompt += f"- {prev['sub_query']}\n"

        prompt += """\nWrite ONE clear and specific question that:
    1. Can be answered using ONLY this document
    2. Helps retrieve information needed for the main question
    3. Is direct and focused on key information from this document

    Write only the question, without any explanations or formatting."""
        return prompt

    def process_document(self, 
                        sample: dict, 
                        topic: str, 
                        doc: str, 
                        previous_queries: list) -> dict:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                prompt = self.generate_prompt(
                    main_question=sample['question'],
                    current_doc=doc,
                    topic=topic,
                    previous_queries=previous_queries
                )
                
                response = get_claude_response(llm="sonnet", prompt=prompt)
                
                # Clean the response
                sub_query = response.strip().strip('"').strip()
                
                # Basic validation
                if sub_query and '?' in sub_query and len(sub_query) > 10:
                    return {
                        'sub_query': sub_query,
                        'doc_topic': topic,
                        'supporting_doc': doc
                    }
                    
            except Exception as e:
                self.logger.error(f"Error processing document: {str(e)}")
            
            retry_count += 1
            time.sleep(1)
        
        # If all retries failed, return a fallback query
        return {
            'sub_query': f"What does this document tell us about {topic}?",
            'doc_topic': topic,
            'supporting_doc': doc,
            'is_fallback': True
        }

    def process_sample(self, sample: dict) -> dict:
        try:
            processed_sample = deepcopy(sample)
            processed_sample['sub_queries'] = []
            
            for topic, docs in sample['supporting_docs'].items():
                for doc in docs:
                    sub_query = self.process_document(
                        sample=sample,
                        topic=topic,
                        doc=doc,
                        previous_queries=processed_sample['sub_queries']
                    )
                    
                    processed_sample['sub_queries'].append(sub_query)
                    time.sleep(0.1)
        
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
        input_file="/shared/eng/pj20/hotpotqa/data/processed_hotpot/processed_hotpot.json",
        output_file="/shared/eng/pj20/hotpotqa/data/processed_hotpot/hotpot_with_subqueries.json",
        checkpoint_dir="/shared/eng/pj20/hotpotqa/data/processed_hotpot/checkpoints",
        checkpoint_interval=5000,
        num_threads=6  # Adjust based on your system's capabilities
    )
    generator.process_dataset(resume=True)

if __name__ == "__main__":
    main()
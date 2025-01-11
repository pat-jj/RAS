import json
from typing import Dict, List, Any
from claude_api import get_claude_response
import concurrent.futures
from tqdm import tqdm
import threading
import time
from queue import Queue
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hotpot_filter.log'),
        logging.StreamHandler()
    ]
)

class ProgressTracker:
    def __init__(self, total_samples):
        self.total = total_samples
        self.completed = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.pbar = tqdm(total=total_samples, desc="Processing samples")
        
    def update(self, success=True):
        with self.lock:
            self.completed += 1
            if not success:
                self.errors += 1
            self.pbar.update(1)
            
    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        return {
            "completed": self.completed,
            "errors": self.errors,
            "elapsed_time": elapsed_time,
            "samples_per_second": self.completed / elapsed_time if elapsed_time > 0 else 0
        }

class HotpotQAFilter:
    def __init__(self, llm, max_workers=4, batch_size=1000):
        self.llm = llm
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.result_queue = Queue()

    def create_prompt(self, question: str, supporting_docs: Dict[str, List[str]]) -> str:
        """Create a prompt for Claude to identify helpful documents."""
        # First validate the question
        if not question or not isinstance(question, str):
            logging.error(f"Invalid question format: {question}")
            return None, []

        prompt = """Identify which documents are HELPFUL to answer the question. Output only the document numbers separated by commas.

Examples:

Example 1 (Some documents are not helpful):
Question: What nationality was James Henry Miller's wife?
Supporting docs:
1. Margaret "Peggy" Seeger (born June 17, 1935) is an American folksinger. She is also well known in Britain, where she has lived for more than 30 years, and was married to the singer and songwriter Ewan MacColl until his death in 1989.
2. Seeger's father was Charles Seeger (1886–1979), an important folklorist and musicologist; her mother was Seeger's second wife, Ruth Porter Crawford.
3. James Henry Miller, better known by his stage name Ewan MacColl, was an English folk singer and songwriter.
Output: 1,3
Explanation: Only docs 1 and 3 are helpful - doc 1 shows Peggy Seeger (who is American) was married to Ewan MacColl, and doc 3 confirms Ewan MacColl is James Henry Miller. Doc 2 about Seeger's parents is not helpful.

Example 2 (All documents are helpful):
Question: The Oberoi family is part of a hotel company that has a head office in what city?
Supporting docs:
1. The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
2. The Oberoi Group is a hotel company with its head office in Delhi.
Output: 1,2
Explanation: Both docs are helpful - doc 1 links the Oberoi family to The Oberoi Group, and doc 2 provides the head office location.

Now for the actual question:

Question: """ + question + """

Supporting docs:
"""
        # Flatten and enumerate docs
        numbered_docs = []
        for topic, docs in supporting_docs.items():
            for doc in docs:
                numbered_docs.append(doc)
        
        if not numbered_docs:
            logging.error("No supporting documents found")
            return None, []
            
        for i, doc in enumerate(numbered_docs, 1):
            prompt += f"{i}. {doc}\n"
            
        prompt += "\nOutput only the helpful document numbers separated by commas:"
        
        # Add debug logging
        logging.debug(f"Generated prompt with question: {question}")
        logging.debug(f"Number of supporting docs: {len(numbered_docs)}")
        
        return prompt, numbered_docs

    def filter_docs(self, sample: Dict[str, Any], tracker: ProgressTracker = None) -> Dict[str, Any]:
        """Filter a single HotpotQA sample based on Claude's response."""
        try:
            prompt, numbered_docs = self.create_prompt(
                sample.get("question", ""),  # Use get() with default value
                sample.get("supporting_docs", {})
            )
            
            # If prompt creation failed, return None to filter out this sample
            if prompt is None or not numbered_docs:
                logging.warning(f"Failed to create prompt for sample with question: {sample.get('question', '')}")
                if tracker:
                    tracker.update(success=True)
                return None
                
            # Log the actual question being processed
            logging.debug(f"Processing question: {sample.get('question', '')}")
            
            # Get necessary doc indices from Claude
            response = get_claude_response(llm=self.llm, prompt=prompt)
            
            # Extract just the numbers from the response
            try:
                # Find the first sequence of comma-separated numbers
                import re
                number_matches = re.findall(r'(?:^|[^\d])(\d+(?:\s*,\s*\d+)*)', response)
                if not number_matches:
                    logging.warning(f"No numbers found in response: {response}")
                    if tracker:
                        tracker.update(success=True)  # Still counts as successful processing
                    return None  # Indicate this sample should be filtered out
                    
                # Take the first match and split it into numbers
                numbers_str = number_matches[0]
                necessary_indices = [int(idx.strip()) - 1 for idx in numbers_str.split(",")]
                
                # Validate indices
                if not all(0 <= idx < len(numbered_docs) for idx in necessary_indices):
                    logging.warning(f"Invalid indices in response: {response}")
                    if tracker:
                        tracker.update(success=False)
                    return sample
                    
            except Exception as e:
                logging.error(f"Error parsing response: {response}\nError: {e}")
                if tracker:
                    tracker.update(success=False)
                return sample

            # Get necessary docs
            necessary_docs = [numbered_docs[i] for i in necessary_indices]
            
            # Filter supporting_docs
            filtered_supporting_docs = {}
            for topic, docs in sample["supporting_docs"].items():
                filtered_docs = [doc for doc in docs if doc in necessary_docs]
                if filtered_docs:
                    filtered_supporting_docs[topic] = filtered_docs

            # Filter sub_queries based on necessary docs
            filtered_sub_queries = []
            for query in sample["sub_queries"]:
                if any(doc in necessary_docs for doc in sample["supporting_docs"][query["doc_topic"]]):
                    filtered_sub_queries.append(query)

            # Create filtered sample
            filtered_sample = {
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample["type"],
                "supporting_docs": filtered_supporting_docs,
                "sub_queries": filtered_sub_queries
            }
            
            if tracker:
                tracker.update(success=True)
            return filtered_sample

        except Exception as e:
            logging.error(f"Error processing sample: {e}")
            if tracker:
                tracker.update(success=False)
            return sample

    def process_batch(self, batch: List[Dict], tracker: ProgressTracker) -> List[Dict]:
        """Process a batch of samples using ThreadPoolExecutor."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.filter_docs, sample, tracker)
                for sample in batch
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return results

def process_dataset(input_file: str, output_file: str, llm, max_workers: int = 4, batch_size: int = 1000, resume: bool = False) -> None:
    """Process entire HotpotQA dataset using multiple threads."""
    # Initialize filter
    hotpot_filter = HotpotQAFilter(llm, max_workers=max_workers, batch_size=batch_size)
    
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Find the last intermediate file if resuming
    filtered_data = []
    start_idx = 0
    
    if resume:
        intermediate_files = [f for f in os.listdir(os.path.dirname(output_file)) 
                            if f.startswith(os.path.basename(output_file) + '.partial_')]
        if intermediate_files:
            # Get the latest intermediate file
            latest_file = max(intermediate_files, 
                            key=lambda x: int(x.split('_')[-1]))
            latest_path = os.path.join(os.path.dirname(output_file), latest_file)
            
            # Load the data from latest intermediate file
            with open(latest_path, 'r') as f:
                filtered_data = json.load(f)
            
            # Calculate where to resume from
            start_idx = int(latest_file.split('_')[-1])
            logging.info(f"Resuming from {latest_file} with {len(filtered_data)} samples processed")
    
    # Create progress tracker for remaining samples
    tracker = ProgressTracker(len(data) - start_idx)
    
    # Process remaining data in batches
    for i in range(start_idx, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_results = hotpot_filter.process_batch(batch, tracker)
        # Filter out None values (samples that should be removed)
        valid_results = [result for result in batch_results if result is not None]
        filtered_data.extend(valid_results)
        
        # Log statistics about filtered samples
        filtered_count = len(batch_results) - len(valid_results)
        if filtered_count > 0:
            logging.info(f"Filtered out {filtered_count} samples from current batch")
        
        # Save intermediate results
        if (i + batch_size) % (batch_size * 10) == 0:
            intermediate_file = f"{output_file}.partial_{i + batch_size}"
            with open(intermediate_file, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            logging.info(f"Saved intermediate results to {intermediate_file}")

    # Write final output file
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Log final statistics
    stats = tracker.get_stats()
    logging.info(f"""
    Processing complete:
    - Total samples processed: {len(filtered_data)}
    - Completed in this session: {stats['completed']}
    - Errors in this session: {stats['errors']}
    - Time elapsed: {stats['elapsed_time']:.2f} seconds
    - Processing speed: {stats['samples_per_second']:.2f} samples/second
    - Output saved to: {output_file}
    """)

# Example usage
if __name__ == "__main__":
    input_file = "/shared/eng/pj20/firas_data/datasets/hotpotqa/hotpot_with_subqueries.json"
    output_file = "/shared/eng/pj20/firas_data/datasets/hotpotqa/filtered/hotpot_filtered.json"
    
    # Initialize your LLM here
    llm = "sonnet"  # Replace with your Claude API initialization
    
    # Process with 4 workers and batches of 1000 samples
    process_dataset(
        input_file=input_file,
        output_file=output_file,
        llm=llm,
        max_workers=5,
        batch_size=500,
        resume=True  # Set to True to resume from last intermediate file
    )
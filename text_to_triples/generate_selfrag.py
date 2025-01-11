import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from tqdm import tqdm
import logging
import os
from typing import List
import time
from pathlib import Path
import multiprocessing
from functools import partial

# Enhanced logging setup
log_dir = "/shared/eng/pj20/firas_data/datasets/selfrag/generated_triples/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'processing_{time.strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_batch(texts: List[str], tokenizer, model, batch_size: int = 8) -> List[dict]:
    """Process a batch of texts using pre-loaded model and tokenizer"""
    try:
        results = []
        
        # Process in smaller batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=0.6
                )
            
            # Decode
            decoded_triples = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Store results
            for text, triple in zip(batch_texts, decoded_triples):
                results.append({
                    "text": text,
                    "generated_triple": triple
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in process_batch: {str(e)}")
        return []

def save_results(results: List[dict], output_path: str, worker_id: int):
    """Save results from a worker"""
    filename = f"results_worker_{worker_id}_{int(time.time())}.json"
    output_file = os.path.join(output_path, filename)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return output_file

def worker_function(texts: List[str], model_path: str, output_dir: str, worker_id: int):
    """Worker process function"""
    try:
        logger.info(f"Worker {worker_id} starting with {len(texts)} texts")
        
        # Load model and tokenizer once per worker
        logger.info(f"Loading model and tokenizer for worker {worker_id}")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.eval()
        logger.info(f"Model and tokenizer loaded for worker {worker_id}")
        
        results = process_batch(texts, tokenizer, model)
        
        if results:
            output_file = save_results(results, output_dir, worker_id)
            logger.info(f"Worker {worker_id} saved results to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in worker {worker_id}: {str(e)}")
        return False

def merge_results(output_dir: str, final_output: str):
    """Merge all worker results into a single file"""
    all_results = []
    
    result_files = list(Path(output_dir).glob("results_worker_*.json"))
    for result_file in tqdm(result_files, desc="Merging results"):
        with open(result_file, 'r') as f:
            results = json.load(f)
            all_results.extend(results)
        os.remove(result_file)
    
    with open(final_output, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Merged {len(all_results)} results into {final_output}")

def main():
    # Paths
    model_path = "/shared/eng/pj20/firas_data/text2triple_model/best_model"
    input_file = "/shared/eng/pj20/firas_data/datasets/selfrag/documents.json"
    output_dir = "/shared/eng/pj20/firas_data/datasets/selfrag/generated_triples"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load documents
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Configuration
    num_workers = 15
    chunk_size = len(documents) // num_workers
    if len(documents) % num_workers:
        chunk_size += 1
    
    logger.info(f"Processing {len(documents)} documents with {num_workers} workers")
    
    # Split documents into chunks
    document_chunks = [
        documents[i:i + chunk_size] 
        for i in range(0, len(documents), chunk_size)
    ]
    
    # Create pool and process
    with multiprocessing.Pool(num_workers) as pool:
        # Create worker tasks without using partial
        tasks = [
            (chunk, model_path, output_dir, i) 
            for i, chunk in enumerate(document_chunks)
        ]
        
        results = pool.starmap(worker_function, tasks)
    
    # Merge results
    if all(results):
        merge_results(
            output_dir,
            os.path.join(output_dir, "selfrag_triples.json")
        )
    else:
        logger.error("Some workers failed to complete")

if __name__ == "__main__":
    main()
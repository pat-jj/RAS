import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import json
from tqdm import tqdm
import logging
import os
from typing import List
import multiprocessing
from pathlib import Path
import time
import shutil

# Enhanced logging setup
log_dir = "/shared/eng/pj20/firas_data/datasets/DBPedia/generated_triples/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'processing_{time.strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()  # This will still print to console
    ]
)
logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: T5Tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        input_encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'text': text,
            'idx': idx
        }

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def save_accumulated_batches(accumulated_results, output_dir: str, split: str, batch_start: int, rank: int):
    """
    Save accumulated batch results with unique identifiers
    """
    if not accumulated_results:
        return None
        
    timestamp = int(time.time() * 1000)
    filename = f"{split}_batches_{batch_start}_to_{batch_start+len(accumulated_results)-1}_rank_{rank}_{timestamp}.json"
    output_path = os.path.join(output_dir, "batch_results", filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(accumulated_results, f, ensure_ascii=False)
    
    logger.info(f"Saved accumulated batches {batch_start} to {batch_start+len(accumulated_results)-1} on rank {rank}")
    return output_path

def process_csv_file_distributed(
    rank: int,
    world_size: int,
    model_path: str,
    input_file: str,
    output_dir: str,
    split: str,
    batch_size: int = 32
):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    logger.info(f"Process {rank}/{world_size} using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    model = DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    
    df = pd.read_csv(input_file)
    texts = df['text'].tolist()
    total_examples = len(texts)
    
    dataset = InferenceDataset(texts, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=min(4, multiprocessing.cpu_count() // world_size),
        pin_memory=True
    )
    
    accumulated_results = []
    save_frequency = 10  # Save every 10 batches
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating triples (GPU {rank})")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                original_indices = batch['idx'].tolist()
                
                generated_ids = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=0.6,
                    use_cache=True
                )
                
                decoded_triples = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                batch_results = [
                    {
                        "text": text,
                        "generated_triple": triple,
                        "original_idx": idx
                    }
                    for text, triple, idx in zip(batch['text'], decoded_triples, original_indices)
                ]
                
                accumulated_results.extend(batch_results)
                
                # Save accumulated results every save_frequency batches
                if (batch_idx + 1) % save_frequency == 0:
                    save_accumulated_batches(
                        accumulated_results,
                        output_dir,
                        split,
                        batch_idx - save_frequency + 1,
                        rank
                    )
                    accumulated_results = []  # Clear accumulated results
                    
                logger.info(f"Rank {rank}: Completed batch {batch_idx + 1}/{len(dataloader)}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} on rank {rank}: {str(e)}")
                continue
    
    # Save any remaining accumulated results
    if accumulated_results:
        save_accumulated_batches(
            accumulated_results,
            output_dir,
            split,
            len(dataloader) - len(accumulated_results),
            rank
        )
    
    dist.barrier()
    
    if rank == 0:
        try:
            output_file = os.path.join(output_dir, f"DBPEDIA_{split}_triples.json")
            batch_dir = os.path.join(output_dir, "batch_results")
            merge_results(batch_dir, output_file, total_examples)
            logger.info(f"Results merged and saved to {output_file}")
            
            shutil.rmtree(batch_dir)
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
    
    cleanup_distributed()

def merge_results(batch_dir: str, output_file: str, total_examples: int):
    """
    Merge all batch results into a single file, maintaining original order
    """
    all_results = [None] * total_examples
    
    batch_files = list(Path(batch_dir).glob("*.json"))
    for batch_file in tqdm(batch_files, desc="Merging results"):
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
            for result in batch_results:
                original_idx = result.pop('original_idx')
                all_results[original_idx] = result
        
        os.remove(batch_file)
    
    assert all(x is not None for x in all_results), "Some examples were not processed"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

def main():
    model_path = "/shared/eng/pj20/firas_data/text2triple_model/best_model"
    data_dir = "/shared/eng/pj20/firas_data/datasets/DBPedia"
    output_dir = "/shared/eng/pj20/firas_data/datasets/DBPedia/generated_triples"
    batch_size = 32
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "batch_results"), exist_ok=True)
    
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUs")
    
    splits = ['train', 'test', 'val']
    for split in splits:
        input_file = os.path.join(data_dir, f"DBPEDIA_{split}.csv")
        
        if os.path.exists(input_file):
            try:
                torch.multiprocessing.spawn(
                    process_csv_file_distributed,
                    args=(world_size, model_path, input_file, output_dir, split, batch_size),
                    nprocs=world_size,
                    join=True
                )
                logger.info(f"Successfully processed {split} split")
            except Exception as e:
                logger.error(f"Error processing {split} split: {str(e)}")
        else:
            logger.warning(f"Input file not found: {input_file}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
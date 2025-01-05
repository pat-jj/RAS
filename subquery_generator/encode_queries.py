import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import argparse
import os

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


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute mean pooling of token embeddings"""
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]  


def encode_and_save_queries(queries: List[str], output_path: str, batch_size: int = 32):
    # Initialize model on single GPU
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    model = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
    model.eval()
    
    # Create dataloader
    dataset = QueryDataset(queries, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    # Encode in batches
    all_embeddings = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding queries")):
            if batch_idx % 100 == 0:
                print(f"Encoding batch {batch_idx}/{len(dataloader)}")
                
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            
            if batch_idx % 100 == 0:
                print(f"Sample queries: {batch['queries'][:2]}")

    # Concatenate and save
    final_embeddings = torch.cat(all_embeddings, dim=0)
    assert len(final_embeddings) == len(queries), "Mismatch in number of embeddings"
    assert final_embeddings.shape == (len(queries), 768), f"Expected shape {(len(queries), 768)}, got {final_embeddings.shape}"
    
    torch.save({
        'embeddings': final_embeddings,
        'queries': queries
    }, output_path)
    print(f"Saved {len(queries)} query embeddings to {output_path}")
    

def main():
    parser = argparse.ArgumentParser(description='Encode queries for dense retrieval')
    parser.add_argument('--hotpot_path', type=str, required=True, help='Path to the HotpotQA dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the query embeddings')
    args = parser.parse_args()
    
    if os.path.exists(args.output_path):
        print(f"Output path {args.output_path} already exists. Skipping encoding.")
        return
    
    with open(args.hotpot_path, "r") as f:
        hotpot_data = json.load(f)
    
    # Extract queries and metadata with improved tracking
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

    encode_and_save_queries(queries, args.output_path)
    


if __name__ == "__main__":
    main()
import torch
import torch.nn.parallel
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
import json
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from functools import partial
import time
from datetime import datetime
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts: List[str], model):
        self.texts = texts
        self.model = model
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        embedding = self.model.encode(text, convert_to_tensor=True)
        return {
            'embedding': embedding,
            'text': text,
            'idx': idx
        }

class MultilabelClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_labels)
        )
        
    def forward(self, embeddings):
        return torch.sigmoid(self.classifier(embeddings))

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def save_checkpoint(results_dict, output_path, checkpoint_name):
    """Save intermediate results to a checkpoint file."""
    checkpoint_dir = Path(output_path).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    pd.to_pickle(results_dict, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoints(checkpoint_dir):
    """Load and merge all available checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    all_results = {}
    for checkpoint in checkpoint_dir.glob("*.pkl"):
        results = pd.read_pickle(checkpoint)
        all_results.update(results)
        
    return all_results

def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model and its configuration."""
    checkpoint_path = Path(checkpoint_path)
    
    # Load model configuration
    with open(checkpoint_path / "config.json", 'r') as f:
        config = json.load(f)
    
    # Initialize models
    sentence_transformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    embed_dim = sentence_transformer.get_sentence_embedding_dimension()
    classifier = MultilabelClassifier(embed_dim, len(config['label_mapping']))
    
    # Load saved states
    model_state = torch.load(
        checkpoint_path / "model_state.pt",
        map_location=device
    )
    
    classifier.load_state_dict(model_state['classifier_state'])
    sentence_transformer.load_state_dict(model_state['encoder_state'])
    
    # Move models to device
    classifier.to(device)
    sentence_transformer.to(device)
    
    return classifier, sentence_transformer, config['label_mapping']

def process_batch(rank, world_size, texts, checkpoint_path, batch_size, output_path, save_interval=1000):
    """Process a batch of texts on a specific GPU with progress tracking and checkpointing."""
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Load model
    classifier, sentence_transformer, label_mapping = load_model(checkpoint_path, device)
    classifier = DDP(classifier, device_ids=[rank])
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, sentence_transformer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4
    )
    
    # Initialize results dictionary
    results_dict = {}
    
    # Get predictions
    classifier.eval()
    sentence_transformer.eval()
    
    total_batches = len(dataloader)
    progress_bar = tqdm(total=total_batches, desc=f'GPU {rank}', position=rank)
    
    last_save_time = time.time()
    processed_since_last_save = 0
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            indices = batch['idx'].tolist()
            outputs = classifier(embeddings)
            probs = outputs.cpu().numpy()
            
            # Store results with original indices
            for idx, (text, prob) in enumerate(zip(batch['text'], probs)):
                results_dict[indices[idx]] = {
                    'text': text,
                    'probabilities': prob
                }
            
            processed_since_last_save += len(indices)
            progress_bar.update(1)
            
            # Save checkpoint if enough samples have been processed
            if processed_since_last_save >= save_interval:
                save_checkpoint(results_dict, output_path, f"gpu{rank}")
                processed_since_last_save = 0
                last_save_time = time.time()
    
    progress_bar.close()
    cleanup()
    return results_dict, label_mapping

def load_and_filter_existing_results(existing_file: str, texts: List[str]) -> tuple[Dict, List[int]]:
    """Load existing results and identify texts that need processing."""
    print(f"Loading existing results from {existing_file}")
    
    # Initialize containers
    existing_results = {}
    reprocess_indices = []
    chunk_size = 100000  # Adjust this value based on your available memory
    
    # Process the CSV in chunks
    for chunk_idx, chunk in tqdm(enumerate(pd.read_csv(existing_file, chunksize=chunk_size))):
        start_idx = chunk_idx * chunk_size
        
        for idx, row in enumerate(chunk.itertuples()):
            global_idx = start_idx + idx
            probs = row[2:]  # Skip index and text columns
            
            # Check if the embedding is a zero vector
            if np.all(np.array(probs) == 0):
                reprocess_indices.append(global_idx)
            else:
                existing_results[global_idx] = {
                    'text': row.text,
                    'probabilities': np.array(probs)
                }
        
        # Free up memory
        del chunk
    
    print(f"Found {len(existing_results)} valid existing results")
    print(f"Found {len(reprocess_indices)} zero-vector embeddings to reprocess")
    
    # Identify completely new texts
    total_existing = (chunk_idx + 1) * chunk_size
    new_indices = list(range(total_existing, len(texts)))
    print(f"Found {len(new_indices)} new texts to process")
    
    # Combine indices that need processing
    indices_to_process = reprocess_indices + new_indices
    texts_to_process = [texts[i] for i in indices_to_process]
    
    return existing_results, texts_to_process, indices_to_process

def get_class_probabilities(texts: List[str], 
                          checkpoint_path: str, 
                          output_path: str,
                          existing_file: str = None,
                          batch_size: int = 16,
                          save_interval: int = 4000) -> pd.DataFrame:
    """
    Generate class probabilities for a list of texts using multiple GPUs with progress tracking.
    """
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")
    
    print(f"Using {world_size} GPUs")
    
    checkpoint_dir = Path(output_path).parent / "checkpoints"
    checkpoint_results = load_checkpoints(checkpoint_dir)
    
    if checkpoint_results:
        print(f"Found existing checkpoints with {len(checkpoint_results)} processed items")
        existing_results = checkpoint_results
    else:
        existing_results = {}
    
    texts_to_process = texts
    
    # Handle existing results from previous run and checkpoints
    if existing_file and Path(existing_file).exists():
        file_results, texts_to_process, indices_to_process = load_and_filter_existing_results(
            existing_file, texts
        )
        # Merge with checkpoint results
        existing_results.update(file_results)
    else:
        indices_to_process = list(range(len(texts)))
        # Filter out indices that were already processed in checkpoints
        if checkpoint_results:
            processed_indices = set(checkpoint_results.keys())
            indices_to_process = [i for i in indices_to_process if i not in processed_indices]
            texts_to_process = [texts[i] for i in indices_to_process]
    
    if not texts_to_process:
        print("All texts have been processed. Creating final DataFrame...")
        return create_results_dataframe(existing_results, texts)
    
    # Process remaining texts
    processes = []
    mp.spawn(
        process_batch,
        args=(world_size, texts_to_process, checkpoint_path, batch_size, output_path, save_interval),
        nprocs=world_size,
        join=True
    )
    
    # Load and merge all results
    checkpoint_dir = Path(output_path).parent / "checkpoints"
    new_results = load_checkpoints(checkpoint_dir)
    
    # Map the new results back to their original indices
    mapped_results = {}
    for new_idx, result in new_results.items():
        original_idx = indices_to_process[new_idx]
        mapped_results[original_idx] = result
    
    # Merge with existing results
    final_results = {**existing_results, **mapped_results}
    
    return create_results_dataframe(final_results, texts)

def create_results_dataframe(results_dict, original_texts):
    """Create a DataFrame from the results dictionary maintaining original order."""
    # Get label mapping from any result
    sample_probs = next(iter(results_dict.values()))['probabilities']
    num_labels = len(sample_probs)
    
    # Initialize empty arrays
    all_probs = np.zeros((len(original_texts), num_labels))
    all_texts = []
    
    # Fill arrays maintaining original order
    for i in range(len(original_texts)):
        result = results_dict.get(i, {'text': original_texts[i], 'probabilities': np.zeros(num_labels)})
        all_texts.append(result['text'])
        all_probs[i] = result['probabilities']
    
    # Create DataFrame
    results_df = pd.DataFrame(all_probs, columns=[f'label_{i}' for i in range(num_labels)])
    results_df.insert(0, 'text', all_texts)
    
    return results_df

def ckpt_cleanup(output_path):
    checkpoint_dir = Path(output_path).parent / "checkpoints"
    for checkpoint in checkpoint_dir.glob("*.pkl"):
        checkpoint.unlink()

def main():
    mp.set_start_method('spawn')
    
    # data_dir = '/shared/eng/pj20/firas_data/datasets/classifier_labeling_data'
    checkpoint_path = '/shared/eng/pj20/firas_data/classifiers/best_model'
    
    # # Process queries
    # queries_path = f"{data_dir}/queries_hotpot.json"
    # with open(queries_path, 'r') as f:
    #     queries = json.load(f)
    
    # output_path = f'{data_dir}/query_class_probabilities.csv'
    # print("\nProcessing queries...")
    # results = get_class_probabilities(queries, checkpoint_path, output_path)
    # results.to_csv(output_path, index=False)
    # print(f"Results saved to {output_path}")
    # # remove all checkpoints
    # ckpt_cleanup(output_path)
    
    # # Process documents
    # documents_path = f"{data_dir}/documents_hotpot.json"
    # with open(documents_path, 'r') as f:
    #     documents = json.load(f)
    
    # output_path = f'{data_dir}/document_class_probabilities.csv'
    # print("\nProcessing documents...")
    # results = get_class_probabilities(documents, checkpoint_path, output_path)
    # results.to_csv(output_path, index=False)
    # print(f"Results saved to {output_path}")
    # # remove all checkpoints
    # ckpt_cleanup(output_path)
    
    # Process Wiki
    # data_path = "/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/all_wiki_text.json"
    # data_dir = '/shared/eng/pj20/firas_data/knowledge_source/wiki_2017'
    
    # data_path = "/shared/eng/pj20/firas_data/knowledge_source/wiki_2018/all_wiki_text.json"
    # data_dir = '/shared/eng/pj20/firas_data/knowledge_source/wiki_2018'
    
    data_path = "/shared/eng/pj20/firas_data/knowledge_source/wiki_2020/all_wiki_text.json"
    # data_dir = '/shared/eng/pj20/firas_data/knowledge_source'
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'r') as f:
        wiki_text = json.load(f)
    
    existing_file = '/shared/eng/pj20/firas_data/knowledge_source/wiki_class_probabilities_2020_.csv'
    output_path = '/shared/eng/pj20/firas_data/knowledge_source/wiki_class_probabilities_2020.csv'
    print("\nProcessing Wiki text...")
    results = get_class_probabilities(
        wiki_text, 
        checkpoint_path, 
        output_path, 
        existing_file=existing_file,
        save_interval=500000
    )
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    # remove all checkpoints
    ckpt_cleanup(output_path)

if __name__ == "__main__":
    main()
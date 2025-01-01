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

def get_class_probabilities(texts: List[str], 
                          checkpoint_path: str, 
                          output_path: str,
                          batch_size: int = 16,
                          save_interval: int = 4000) -> pd.DataFrame:
    """
    Generate class probabilities for a list of texts using multiple GPUs with progress tracking.
    
    Args:
        texts: List of input texts to classify
        checkpoint_path: Path to the model checkpoint
        output_path: Path to save results and checkpoints
        batch_size: Batch size for processing per GPU
        save_interval: Number of samples to process before saving a checkpoint
    
    Returns:
        DataFrame with texts and their class probabilities
    """
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")
    
    print(f"Using {world_size} GPUs")
    
    # Check for existing checkpoints
    checkpoint_dir = Path(output_path).parent / "checkpoints"
    existing_results = load_checkpoints(checkpoint_dir)
    if existing_results is not None:
        print(f"Found existing results for {len(existing_results)} samples")
        
        # Filter out already processed texts
        processed_indices = set(existing_results.keys())
        texts_to_process = [text for i, text in enumerate(texts) if i not in processed_indices]
        print(f"Remaining samples to process: {len(texts_to_process)}")
    else:
        texts_to_process = texts
    
    if not texts_to_process:
        print("All texts have been processed. Loading results from checkpoints...")
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
    final_results = load_checkpoints(checkpoint_dir)
    if existing_results:
        final_results.update(existing_results)
    
    return create_results_dataframe(final_results, texts)

# def create_results_dataframe(results_dict, original_texts):
#     """Create a DataFrame from the results dictionary maintaining original order."""
#     # Get label mapping from any result
#     sample_probs = next(iter(results_dict.values()))['probabilities']
#     num_labels = len(sample_probs)
    
#     # Initialize empty arrays
#     all_probs = np.zeros((len(original_texts), num_labels))
#     all_texts = []
    
#     # Fill arrays maintaining original order
#     for i in range(len(original_texts)):
#         result = results_dict.get(i, {'text': original_texts[i], 'probabilities': np.zeros(num_labels)})
#         all_texts.append(result['text'])
#         all_probs[i] = result['probabilities']
    
#     # Create DataFrame
#     results_df = pd.DataFrame(all_probs, columns=[f'label_{i}' for i in range(num_labels)])
#     results_df.insert(0, 'text', all_texts)
    
#     return results_df

def ckpt_cleanup(output_path):
    checkpoint_dir = Path(output_path).parent / "checkpoints"
    for checkpoint in checkpoint_dir.glob("*.pkl"):
        checkpoint.unlink()

def create_results_dataframe(results_dict, original_texts):
    """Create a DataFrame from the results dictionary maintaining original order."""
    # Get label mapping from any result
    sample_probs = next(iter(results_dict.values()))['probabilities']
    num_labels = len(sample_probs)
    
    # Initialize empty arrays
    all_probs = np.zeros((len(original_texts), num_labels))
    all_texts = []
    
    # Process in chunks to manage memory
    chunk_size = 100000  # Adjust based on available RAM
    num_chunks = (len(original_texts) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(original_texts))
        
        # Process chunk
        for i in range(start_idx, end_idx):
            result = results_dict.get(i, {
                'text': original_texts[i], 
                'probabilities': np.zeros(num_labels)
            })
            all_texts.append(result['text'])
            all_probs[i] = result['probabilities']
    
    # Create DataFrame
    results_df = pd.DataFrame(all_probs, columns=[f'label_{i}' for i in range(num_labels)])
    results_df.insert(0, 'text', all_texts)
    
    return results_df

def save_results_parquet(results_df, output_path, chunk_size=100000):
    """Save results to parquet files in chunks."""
    output_path = Path(output_path)
    base_path = output_path.parent / output_path.stem
    base_path.mkdir(exist_ok=True)
    
    # Calculate number of chunks
    num_chunks = (len(results_df) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(results_df))
        
        # Get chunk and save to parquet
        chunk_df = results_df.iloc[start_idx:end_idx]
        chunk_path = base_path / f"chunk_{chunk_idx:05d}.parquet"
        chunk_df.to_parquet(
            chunk_path,
            compression='snappy',  # Good balance between compression and speed
            index=False
        )
        
    # Save metadata
    metadata = {
        'num_chunks': num_chunks,
        'chunk_size': chunk_size,
        'total_rows': len(results_df),
        'columns': results_df.columns.tolist()
    }
    with open(base_path / "metadata.json", 'w') as f:
        json.dump(metadata, f)

def load_results_parquet(base_path):
    """Load results from parquet files, with optional chunked reading."""
    base_path = Path(base_path)
    
    # Load metadata
    with open(base_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    def chunk_generator():
        for chunk_idx in range(metadata['num_chunks']):
            chunk_path = base_path / f"chunk_{chunk_idx:05d}.parquet"
            yield pd.read_parquet(chunk_path)
    
    return chunk_generator(), metadata

# Modified main function
def main():
    mp.set_start_method('spawn')
    
    # Process Wiki data
    data_path = "/shared/eng/pj20/firas_data/knowledge_source/wiki_2020/all_wiki_text.json"
    data_dir = '/shared/eng/pj20/firas_data/knowledge_source/wiki_2020'
    checkpoint_path = '/shared/eng/pj20/firas_data/classifiers/best_model'
    
    with open(data_path, 'r') as f:
        wiki_text = json.load(f)
    
    output_base = f'{data_dir}/wiki_class_probabilities'
    print("\nProcessing Wiki text...")
    results = get_class_probabilities(wiki_text, checkpoint_path, output_base, save_interval=500000)
    
    # Save results in chunks
    print("Saving results to parquet files...")
    save_results_parquet(results, output_base)
    print(f"Results saved to {output_base}")
    
    # Example of loading and processing results in chunks
    print("\nLoading results for verification...")
    chunk_generator, metadata = load_results_parquet(output_base)
    
    # Process chunks as needed
    for chunk_df in chunk_generator:
        # Process each chunk
        # For example, calculate mean probabilities per chunk
        mean_probs = chunk_df.iloc[:, 1:].mean()  # Skip text column
        print(f"Mean probabilities for chunk: {mean_probs}")

if __name__ == "__main__":
    main()
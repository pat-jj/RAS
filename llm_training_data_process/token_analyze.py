import torch
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import pickle

def analyze_token_lengths():
    # Load the tokenizer (using the same model as in ActionPlanner)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False)
    
    # Load training data
    print("Loading training data...")
    train_data = pickle.load(open('/shared/eng/pj20/firas_data/answerer/all_train/train.pkl', 'rb'))  # Update this path
    
    # Initialize lists to store lengths
    input_lengths = []
    
    # Process each example
    text_with_max_length = ''
    max_length = 0
    sample_text = []
    print("Analyzing token lengths...")
    for item in tqdm(train_data):
        # Tokenize input text
        input_tokens = tokenizer(item['input'], add_special_tokens=False)
        # input_tokens = tokenizer(item['label'], add_special_tokens=False)
        input_lengths.append(len(input_tokens.input_ids))
        if len(input_tokens.input_ids) > max_length:
            max_length = len(input_tokens.input_ids)
            text_with_max_length = item['input']
            # text_with_max_length = item['label']
        if len(sample_text) < 20:
            sample_text.append(item['input'])
            # sample_text.append(item['label'])
    # Calculate statistics
    input_lengths = np.array(input_lengths)
    stats = {
        'mean_length': np.mean(input_lengths),
        'max_length': np.max(input_lengths),
        'min_length': np.min(input_lengths),
        'median_length': np.median(input_lengths),
        'p90_length': np.percentile(input_lengths, 90),
        'p95_length': np.percentile(input_lengths, 95),
        'p99_length': np.percentile(input_lengths, 99),
    }
    
    # Print results
    print("\nToken Length Statistics:")
    print(f"Mean length: {stats['mean_length']:.2f}")
    print(f"Max length: {stats['max_length']}")
    print(f"Min length: {stats['min_length']}")
    print(f"Median length: {stats['median_length']}")
    print(f"90th percentile: {stats['p90_length']}")
    print(f"95th percentile: {stats['p95_length']}")
    print(f"99th percentile: {stats['p99_length']}")
    
    # Print length distribution
    print("\nLength Distribution:")
    bins = [0, 512, 1024, 2048, 3072, float('inf')]
    for i in range(len(bins)-1):
        count = np.sum((input_lengths >= bins[i]) & (input_lengths < bins[i+1]))
        percent = (count / len(input_lengths)) * 100
        print(f"{bins[i]}-{bins[i+1]}: {count} examples ({percent:.2f}%)")
        
    # print(f"Text with max length: {text_with_max_length}")
    # print(f"Sample text: {sample_text}")
    
if __name__ == "__main__":
    analyze_token_lengths()
import pickle
import json
import os
from tqdm import tqdm

def load_processed_data(output_dir):
    """Load data from pickle files"""
    data_by_split = {}
    
    for split in ['train', 'val', 'test']:
        pkl_file = os.path.join(output_dir, f'{split}.pkl')
        if os.path.exists(pkl_file):
            try:
                with open(pkl_file, 'rb') as f:
                    split_data = pickle.load(f)
                data_by_split[split] = split_data
                print(f"Loaded {split} data: {len(split_data)} samples")
            except Exception as e:
                print(f"Error loading {split} data: {e}")
                
    return data_by_split

def load_selfrag_data(path):
    """Load SelfRAG data to get instructions for filtering"""
    with open(path, 'r') as f:
        data = json.load(f)
    # Create set of all instructions for efficient lookup
    return set(item['instruction'] for item in data)

def filter_out_selfrag(processed_item, selfrag_instructions):
    """
    Returns True if item should be kept (is HotpotQA), False if item is from SelfRAG
    """
    # Get the question part from the input (after "Question: ")
    input_parts = processed_item['input'].split("Question: ")
    if len(input_parts) < 2:
        return True  # Keep if we can't determine
    
    question = input_parts[1].strip()
    # Check if this question exists in selfrag instructions
    return question not in selfrag_instructions

def main():
    # Paths
    input_dir = '/shared/eng/pj20/firas_data/action_planner/all_train'
    output_dir = '/shared/eng/pj20/firas_data/action_planner/hotpotqa_only'
    selfrag_path = '/shared/eng/pj20/firas_data/datasets/selfrag/selfrag_with_subqueries_refined.json'
    
    # Load data
    print("Loading processed data...")
    processed_data = load_processed_data(input_dir)
    
    print("Loading SelfRAG data...")
    selfrag_instructions = load_selfrag_data(selfrag_path)
    
    # Filter data
    filtered_data = {}
    for split, data in processed_data.items():
        print(f"\nFiltering {split} split...")
        filtered = []
        for item in tqdm(data):
            if filter_out_selfrag(item, selfrag_instructions):
                filtered.append(item)
        filtered_data[split] = filtered
        print(f"Kept {len(filtered)}/{len(data)} items in {split} split")
    
    # Save filtered data
    print("\nSaving filtered data...")
    os.makedirs(output_dir, exist_ok=True)
    
    for split, data in filtered_data.items():
        output_path = os.path.join(output_dir, f'{split}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} examples to {output_path}")
    
    # Save a few examples for inspection
    example_file = os.path.join(output_dir, 'examples.json')
    with open(example_file, 'w') as f:
        json.dump(filtered_data['train'][:5], f, indent=2, default=str)
    print(f"Saved 5 examples to {example_file} for inspection")

if __name__ == "__main__":
    main()
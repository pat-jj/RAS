import json
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
import random
import pickle


def get_answerer_instruction():

    return """You are a answerer given a question and retrieved graph information.
Each [SUBQ] is a subquery we generated through reasoning for the question. The retrieved graph information follows each [SUBQ] is relevant graph information we retrieved to answer the subquery.
[NO_RETRIEVAL] means the question can be answered with the question itself without any retrieval.
The main question starts with "Question: ". Please answer the question, with subqueries and retrieved graph information if they are helpful.
"""

def get_planner_instruction():
    return """You are a planner to determine if the question can be answered with current information and output the appropriate label as well as the subquery if needed.
Output [NO_RETRIEVAL] if the question can be directly answered with the question itself without any retrieval.
Output [SUBQ] with an subquery for retrieval if still needs a subquery.
Output [SUFFICIENT] if the question can be answered with the provided information.
"""

def load_planner_data(planner_data_dir):
    if os.path.exists(os.path.join(planner_data_dir, 'train.pkl')):
        with open(os.path.join(planner_data_dir, 'train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(planner_data_dir, 'val.pkl'), 'rb') as f:
            val_data = pickle.load(f)
        with open(os.path.join(planner_data_dir, 'test.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        planner_data = train_data + val_data + test_data
        return planner_data
    else:
        raise ValueError(f"Planner data not found in {planner_data_dir}, please run action_planner_data/data_process.py first.")

def clean_answer(answer):
    # Remove [Retrieval], [Relevant], [No support / Contradictory] tokens
    import re
    try:
        answer = re.sub(r'\[.*?\]', '', answer)
        # Remove <paragraph>...</paragraph> blocks
        answer = re.sub(r'<paragraph>.*?</paragraph>', '', answer)
        # Clean up any extra whitespace
        answer = ' '.join(answer.split())
    except Exception as e:
        # print(f"Error cleaning answer: {e} type of answer: {type(answer)}")
        return None
    return answer
    
def main(output_dir):
    print("Loading HotpotQA data...")
    hotpotqa_data_path = "/shared/eng/pj20/firas_data/datasets/hotpotqa/processed_hotpot.json"
    with open(hotpotqa_data_path, 'r') as f:
        hotpotqa_data = json.load(f)
    hotpotqa_pairs = {item['question']: item['answer'] for item in tqdm(hotpotqa_data)}
    
    print("Loading Selfrag data...")
    selfrag_data_path = "/shared/eng/pj20/firas_data/datasets/selfrag/selfrag_with_subqueries.json"
    with open(selfrag_data_path, 'r', encoding='utf-8') as f:
        selfrag_data = json.load(f)
        
    selfrag_pairs = {}
    no_answer_count = 0
    for item in tqdm(selfrag_data):
        if 'answer' in item:
            answer = clean_answer(item['answer'])
            if answer == None:
                no_answer_count += 1
                continue
            selfrag_pairs[item['instruction']] = answer
        else:
            no_answer_count += 1
    print(f"No answer count: {no_answer_count}")
    
    all_pairs = {**hotpotqa_pairs, **selfrag_pairs}
    
    print("Loading processed planner data...")
    planner_data_dir = "/shared/eng/pj20/firas_data/action_planner/all_train"
    planner_data = load_planner_data(planner_data_dir)
    
    planner_instruction = get_planner_instruction()
    answerer_instruction = get_answerer_instruction()
    
    print("Processing data...")
    answerer_data = []
    for item in tqdm(planner_data):
        if item['label'] == "[NO_RETRIEVAL]":
            if planner_instruction not in item['input']:
                raise ValueError(f"Planner instruction not found in {item['input']}, please run a_planner_data_process.py first.")
            question = item['input'].split("Question: ")[1]
            if question not in all_pairs:
                print(f"Question {question} not found in all_pairs, skipping...")
                continue
            answer = all_pairs[question]
            input_ = item['input'].replace(planner_instruction, answerer_instruction)
            processed_item = {
                'input': input_,
                'label': answer,
                'graphs': [],
            }
            answerer_data.append(processed_item)
            
        if item['label'] == "[SUFFICIENT]":
            if planner_instruction not in item['input']:
                raise ValueError(f"Planner instruction not found in {item['input']}, please run a_planner_data_process.py first.")
            question = item['input'].split("Question: ")[1]
            if question not in all_pairs:
                print(f"Question {question} not found in all_pairs, skipping...")
                continue
            answer = all_pairs[question]
            input_ = item['input'].replace(planner_instruction, answerer_instruction)
            processed_item = {
                'input': input_,
                'label': answer,
                'graphs': item['graphs'],
            }
            answerer_data.append(processed_item)
            
            
    random.shuffle(answerer_data)
    train_data = answerer_data[:int(len(answerer_data)*0.98)]
    val_data = answerer_data[int(len(answerer_data)*0.98):]
    
    # torch.save(train_data, os.path.join(output_dir, 'train.pt'))
    # torch.save(val_data, os.path.join(output_dir, 'val.pt'))
    
    # print(f"Saved {len(train_data)} train examples to {os.path.join(output_dir, 'train.pt')}")
    # print(f"Saved {len(val_data)} val examples to {os.path.join(output_dir, 'val.pt')}")
    
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
            
    print(f"Saved {len(train_data)} train samples to {os.path.join(output_dir, 'train.pkl')}")
    print(f"Saved {len(val_data)} val samples to {os.path.join(output_dir, 'val.pkl')}")
    
    example_file = os.path.join(output_dir, 'examples.json')
    with open(example_file, 'w') as f:
        json.dump(answerer_data[:20], f, indent=2, default=str)
    print(f"Saved 20 examples to {example_file} for inspection")
            
if __name__ == "__main__":
    main(output_dir="/shared/eng/pj20/firas_data/answerer/all_train")
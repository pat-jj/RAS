import torch
import json
from pathlib import Path

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_torch(data, filepath):
    torch.save(data, filepath)

def load_torch(filepath):
    return torch.load(filepath)


def get_subqueries_prompt(question):
    return f"""Break this multi-hop question into two sub-queries that would help find the answer step by step.

Question: {question}

Please output ONLY two sub-queries, one per line, with no additional text:
"""

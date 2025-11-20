#!/usr/bin/env python3
"""
Evaluation script for baseline results.
Supports both short-answer tasks (match metric) and long-form tasks (ROUGE/MAUVE).
"""

import argparse
import json
import copy
import string
import numpy as np
from tqdm import tqdm
from metrics import match, f1_score, accuracy
import sys
import os

# Add framework path for ROUGE/MAUVE if needed
framework_path = os.path.join(os.path.dirname(__file__), '..', 'framework')
if os.path.exists(framework_path):
    sys.path.insert(0, framework_path)
    try:
        from metrics import compute_rouge, mauve_score
        HAS_LONG_FORM_METRICS = True
    except ImportError:
        print("Warning: ROUGE/MAUVE metrics not available. Install rouge-score and mauve packages.")
        HAS_LONG_FORM_METRICS = False
else:
    HAS_LONG_FORM_METRICS = False
    print("Warning: Framework path not found. ROUGE/MAUVE metrics not available.")

from utils import load_file


def evaluate_short_answer(data, metric='match'):
    """Evaluate short-answer tasks (TriviaQA, PopQA, PubHealth, ARC-C)"""
    metric_results = []
    
    for item in tqdm(data, desc="Evaluating"):
        # Get output and handle different types (string, list, None)
        output_raw = item.get("output", "")
        if isinstance(output_raw, list):
            output = " ".join(str(x) for x in output_raw).lower() if output_raw else ""
        elif output_raw is None:
            output = ""
        else:
            output = str(output_raw).lower()
        
        # Skip invalid outputs
        if "apolo" in output or "the information" in output:
            continue
        
        golds = item.get("golds", [])
        
        if not golds:
            continue
            
        # Ensure golds is a list
        if not isinstance(golds, list):
            golds = [golds]
        
        # Convert to lowercase
        golds = [g.lower() if isinstance(g, str) else str(g).lower() for g in golds]
        
        if metric == 'match':
            result = match(output, golds)
        elif metric == 'f1':
            result = f1_score(output, golds)
        elif metric == 'accuracy':
            result = accuracy([output], [golds])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_results.append(result)
        item["metric_result"] = result
    
    if len(metric_results) == 0:
        print("Warning: No valid results to evaluate")
        return 0.0
    
    avg_score = np.mean(metric_results)
    print(f"{metric.upper()}: {avg_score:.4f} ({len(metric_results)} valid samples)")
    return avg_score


def evaluate_long_form(data):
    """Evaluate long-form tasks (ASQA, ELI5) using ROUGE and MAUVE"""
    if not HAS_LONG_FORM_METRICS:
        print("Error: ROUGE/MAUVE metrics not available")
        return None, None
    
    normalized_data = copy.deepcopy(data)
    
    # Prepare references and predictions for MAUVE
    references = []
    predictions = []
    
    for item in normalized_data:
        question = item.get("question", "").lower()
        
        # Handle different answer formats for ASQA and ELI5
        answer = ""
        if "annotations" in item and item.get("annotations"):
            # ASQA format: has annotations with long_answer
            if isinstance(item["annotations"], list) and len(item["annotations"]) > 0:
                if isinstance(item["annotations"][0], dict):
                    answer = item["annotations"][0].get("long_answer", "").lower()
                else:
                    answer = str(item["annotations"][0]).lower()
            elif isinstance(item["annotations"], dict):
                answer = item["annotations"].get("long_answer", "").lower()
        elif "answer" in item:
            # ELI5 or other format: has answer field
            answer_raw = item.get("answer", "")
            if isinstance(answer_raw, list):
                answer = " ".join(str(a) for a in answer_raw).lower()
            else:
                answer = str(answer_raw).lower()
        elif "answers" in item:
            # Alternative format
            answers_raw = item.get("answers", [])
            if isinstance(answers_raw, list) and len(answers_raw) > 0:
                answer = str(answers_raw[0]).lower()
            else:
                answer = str(answers_raw).lower()
        
        # Get output
        output_raw = item.get("output", "")
        if isinstance(output_raw, list):
            output = " ".join(str(x) for x in output_raw).lower() if output_raw else ""
        elif output_raw is None:
            output = ""
        else:
            output = str(output_raw).lower()
        
        # Format for MAUVE (first 100 words)
        ref_text = ' '.join((question + " " + answer.strip()).split()[:100]).rstrip(string.punctuation)
        pred_text = ' '.join((question + " " + output.strip()).split()[:100]).rstrip(string.punctuation)
        
        references.append(ref_text)
        predictions.append(pred_text)
    
    # Compute ROUGE
    rouge_score = None
    if HAS_LONG_FORM_METRICS:
        try:
            # Prepare data for compute_rouge - ensure all items have required fields
            rouge_data = []
            for item in normalized_data:
                if not isinstance(item, dict):
                    continue
                rouge_item = {"output": item.get("output", "")}
                # Handle ASQA annotations format
                if "annotations" in item and item.get("annotations"):
                    if isinstance(item["annotations"], list) and len(item["annotations"]) > 0:
                        rouge_item["annotations"] = item["annotations"]
                    else:
                        # Fallback: use answer if available
                        if "answer" in item:
                            answer_val = item["answer"]
                            if isinstance(answer_val, str):
                                rouge_item["annotations"] = [{"long_answer": answer_val}, {"long_answer": answer_val}]
                            else:
                                continue
                        else:
                            continue
                elif "answer" in item:
                    # ELI5 format - convert to annotations format for compute_rouge
                    answer_val = item["answer"]
                    if isinstance(answer_val, str):
                        rouge_item["answer"] = answer_val
                    elif isinstance(answer_val, list) and len(answer_val) > 0:
                        rouge_item["answer"] = str(answer_val[0])
                    else:
                        continue
                else:
                    continue
                rouge_data.append(rouge_item)
            
            if rouge_data:
                rouge_score = compute_rouge(rouge_data)
                print(f"ROUGE-L: {rouge_score:.4f}")
            else:
                print("Error: No valid data for ROUGE computation")
                rouge_score = None
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            import traceback
            print(traceback.format_exc()[:300])
            rouge_score = None
    else:
        print("ROUGE not available (install rouge-score and nltk)")
    
    # Compute MAUVE
    mauve_result = None
    if HAS_LONG_FORM_METRICS:
        try:
            mauve_result = mauve_score(predictions, references)
            print(f"MAUVE: {mauve_result:.4f}")
        except Exception as e:
            print(f"Error computing MAUVE: {e}")
            mauve_result = None
    else:
        print("MAUVE not available (install mauve package)")
    
    return rouge_score, mauve_result


def load_and_prepare_data(file_path, dataset_name):
    """Load and prepare data based on dataset type"""
    if not file_path or file_path.strip() == '':
        raise ValueError(f"Empty file path provided for dataset {dataset_name}")
    
    if file_path.endswith('.jsonl'):
        # Try to load with error handling for corrupted files
        try:
            data = load_file(file_path)
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            print("Attempting to load file line by line, skipping invalid lines...")
            # Load line by line, skipping invalid JSON
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        # Ensure item is a dict, not a string or other type
                        if isinstance(item, dict):
                            data.append(item)
                        else:
                            print(f"Warning: Skipping non-dict item on line {line_num}: {type(item)}")
                            continue
                    except json.JSONDecodeError as je:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {je}")
                        continue
            print(f"Loaded {len(data)} valid items from {file_path}")
    else:
        with open(file_path, 'r') as f:
            data_ = json.load(f)
        
        data = []
        
        # Handle different output formats
        if isinstance(data_, dict):
            if 'data' in data_:
                # Format: {"data": [...]}
                data = data_['data']
            elif 'output' in data_:
                # Format: {"output": [...], "answer": [...]}
                for i in range(len(data_['output'])):
                    item = {
                        "output": data_['output'][i].lower() if isinstance(data_['output'][i], str) else str(data_['output'][i]).lower(),
                        "golds": [d.lower() if isinstance(d, str) else str(d).lower() for d in data_['answer'][i]] if isinstance(data_['answer'][i], list) else [str(data_['answer'][i]).lower()]
                    }
                    data.append(item)
            else:
                # Assume it's a list of items
                data = data_ if isinstance(data_, list) else [data_]
        else:
            data = data_ if isinstance(data_, list) else [data_]
    
    # Post-process based on dataset
    for item in data:
        # Skip if item is not a dict (shouldn't happen, but safety check)
        if not isinstance(item, dict):
            continue
            
        # Handle ASQA format (has annotations instead of answer)
        if dataset_name == "asqa":
            # ASQA uses annotations[0].long_answer and annotations[1].long_answer
            # Ensure output exists
            if 'output' not in item:
                if 'prediction' in item:
                    item['output'] = item['prediction']
                else:
                    continue
            # Normalize output
            output_raw = item.get("output", "")
            if isinstance(output_raw, list):
                item['output'] = " ".join(str(x) for x in output_raw)
            elif output_raw is None:
                item['output'] = ""
            else:
                item['output'] = str(output_raw)
            # Ensure annotations is properly formatted for compute_rouge
            if "annotations" not in item or not item.get("annotations"):
                # Try to get from answer field as fallback
                if "answer" in item:
                    item["annotations"] = [{"long_answer": item["answer"]}, {"long_answer": item["answer"]}]
            # Keep annotations as-is for ROUGE computation
            continue  # Skip further processing for ASQA - handled in evaluate_long_form
        
        # Handle ELI5 format
        if dataset_name == "eli5":
            # ELI5 may have answer or answers field
            # Ensure output exists
            if 'output' not in item:
                if 'prediction' in item:
                    item['output'] = item['prediction']
                else:
                    continue
            # Normalize output
            output_raw = item.get("output", "")
            if isinstance(output_raw, list):
                item['output'] = " ".join(str(x) for x in output_raw)
            elif output_raw is None:
                item['output'] = ""
            else:
                item['output'] = str(output_raw)
            # Keep answer/answers as-is for long-form evaluation
            continue  # Skip further processing for ELI5 - handled in evaluate_long_form
        
        # For short-form datasets, ensure output exists
        if 'output' not in item:
            if 'prediction' in item:
                item['output'] = item['prediction']
            else:
                continue
        
        # Ensure golds/answer exists for short-form datasets
        if 'golds' not in item:
            if 'answer' in item:
                answer_val = item['answer']
                if isinstance(answer_val, list):
                    item['golds'] = answer_val
                else:
                    item['golds'] = [answer_val]
            elif 'label' in item:
                label_val = item['label']
                if isinstance(label_val, list):
                    item['golds'] = label_val
                else:
                    item['golds'] = [label_val]
            elif 'answers' in item:
                answers_val = item['answers']
                if isinstance(answers_val, list):
                    item['golds'] = answers_val
                else:
                    item['golds'] = [answers_val]
            elif 'answerKey' in item:
                item['golds'] = [item['answerKey']]
        
        # Convert to lowercase for short-form datasets
        if 'output' in item:
            output_val = item['output']
            if isinstance(output_val, list):
                item['output'] = " ".join(str(x) for x in output_val).lower()
            elif output_val is None:
                item['output'] = ""
            else:
                item['output'] = str(output_val).lower()
        
        if 'golds' in item:
            if isinstance(item['golds'], list):
                item['golds'] = [g.lower() if isinstance(g, str) else str(g).lower() for g in item['golds']]
            else:
                item['golds'] = [str(item['golds']).lower()]
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline results')
    parser.add_argument('--result_file', type=str, required=True,
                       help='Path to result file (JSON or JSONL)')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['triviaqa', 'popqa', 'pubhealth', 'arc_c', 'bio', 'asqa', 'eli5'],
                       help='Dataset name')
    parser.add_argument('--metric', type=str, default='auto',
                       choices=['auto', 'match', 'f1', 'accuracy'],
                       help='Metric to use (auto selects based on dataset)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Optional: Save evaluation results to file')
    
    args = parser.parse_args()
    
    # Auto-select metric
    if args.metric == 'auto':
        if args.dataset in ['asqa', 'eli5']:
            metric = 'long_form'  # Special case
        elif args.dataset == 'pubhealth':
            metric = 'accuracy'
        else:
            metric = 'match'
    else:
        metric = args.metric
    
    print(f"Loading results from: {args.result_file}")
    print(f"Dataset: {args.dataset}")
    print(f"Metric: {metric}")
    print("-" * 60)
    
    # Load data
    data = load_and_prepare_data(args.result_file, args.dataset)
    print(f"Loaded {len(data)} samples")
    
    # Evaluate
    if metric == 'long_form':
        rouge_score, mauve_score = evaluate_long_form(data)
        results = {
            'dataset': args.dataset,
            'rouge': rouge_score,
            'mauve': mauve_score,
            'num_samples': len(data)
        }
    else:
        score = evaluate_short_answer(data, metric=metric)
        results = {
            'dataset': args.dataset,
            'metric': metric,
            'score': float(score),
            'num_samples': len(data)
        }
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")
    
    return results


if __name__ == "__main__":
    main()


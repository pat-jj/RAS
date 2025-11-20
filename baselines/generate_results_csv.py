#!/usr/bin/env python3
"""
Generate a CSV table of all baseline results.
Scans the results directory and evaluates all result files.
"""

import os
import json
import csv
import argparse
from pathlib import Path
import sys

# Add framework to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'framework'))
from evaluate_results import load_and_prepare_data, evaluate_short_answer, evaluate_long_form

def find_result_files(results_dir):
    """Find all result JSONL/JSON files"""
    result_files = []
    for file_path in Path(results_dir).glob("*.jsonl"):
        result_files.append(str(file_path))
    for file_path in Path(results_dir).glob("*.json"):
        if "test" not in str(file_path).lower():  # Skip test files
            result_files.append(str(file_path))
    return sorted(result_files)

def parse_filename(filename):
    """Parse filename to extract model, dataset, and mode"""
    basename = os.path.basename(filename)
    name = basename.replace('.jsonl', '').replace('.json', '')
    
    # Extract dataset (first part before underscore)
    parts = name.split('_')
    if len(parts) < 2:
        return None, None, None
    
    dataset = parts[0]
    
    # Find model name
    models = ['qwen3_8b', 'llama3_8b', 'sonnet4.5', 'sonnet', 'haiku', 'opus']
    model = None
    for m in models:
        if m in name:
            model = m
            break
    
    # Determine mode
    mode = 'base'
    if 'retrieval' in name:
        if 'sure' in name:
            mode = 'sure'
        else:
            # Extract top_n
            if 'top_1' in name:
                mode = 'retrieval_top_1'
            elif 'top_5' in name:
                mode = 'retrieval_top_5'
            else:
                mode = 'retrieval'
    elif 'base' in name:
        mode = 'base'
    
    return dataset, model, mode

def evaluate_file(file_path, dataset_name):
    """Evaluate a result file and return metrics"""
    try:
        data = load_and_prepare_data(file_path, dataset_name)
        if not data:
            return None, "No data loaded"
        
        # Filter out items with None or empty outputs
        valid_data = []
        for item in data:
            if item.get('output') and item.get('output').strip():
                valid_data.append(item)
        
        if not valid_data:
            return None, "No valid data (all outputs empty)"
        
        if dataset_name in ["asqa", "eli5"]:
            # Long-form metrics
            rouge_score, mauve_result = evaluate_long_form(valid_data)
            if rouge_score is not None:
                return {"ROUGE-L": f"{rouge_score:.4f}", "MAUVE": f"{mauve_result:.4f}" if mauve_result else "N/A"}, None
            else:
                return None, "ROUGE/MAUVE not available"
        else:
            # Short-form metrics - calculate both MATCH and the dataset-specific metric
            results = {}
            
            # Always calculate MATCH for all short-form datasets
            match_score = evaluate_short_answer(valid_data, 'match')
            results['MATCH'] = f"{match_score:.4f}"
            
            # Also calculate dataset-specific metric
            if dataset_name == 'arc_c' or 'arc' in dataset_name:
                accuracy_score = evaluate_short_answer(valid_data, 'accuracy')
                results['ACCURACY'] = f"{accuracy_score:.4f}"
            elif dataset_name == 'pubhealth':
                # Already calculated MATCH above, no need to duplicate
                pass
            else:
                # TriviaQA, PopQA, Bio, 2WikiMultiHop - calculate F1
                f1_score = evaluate_short_answer(valid_data, 'f1')
                results['F1'] = f"{f1_score:.4f}"
            
            return results, None
    except Exception as e:
        import traceback
        return None, f"{str(e)}: {traceback.format_exc()[:200]}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, 
                       default='/shared/rsaas/pj20/firas_data/test_datasets/results',
                       help='Directory containing result files')
    parser.add_argument('--output_csv', type=str,
                       default='/home/pj20/server-04/FIRAS/baselines/logs/results_summary.csv',
                       help='Output CSV file path')
    args = parser.parse_args()
    
    # Find all result files
    result_files = find_result_files(args.results_dir)
    print(f"Found {len(result_files)} result files")
    
    # Collect results
    results = []
    for file_path in result_files:
        dataset, model, mode = parse_filename(file_path)
        if not dataset or not model:
            print(f"Skipping {file_path}: Could not parse filename")
            continue
        
        print(f"Evaluating: {dataset} - {model} - {mode}")
        metrics, error = evaluate_file(file_path, dataset)
        
        if metrics:
            results.append({
                'Dataset': dataset,
                'Model': model,
                'Mode': mode,
                **metrics
            })
        else:
            print(f"  Error: {error}")
            results.append({
                'Dataset': dataset,
                'Model': model,
                'Mode': mode,
                'Error': error
            })
    
    # Write CSV
    if results:
        # Get all unique metric columns
        all_metrics = set()
        for r in results:
            all_metrics.update(k for k in r.keys() if k not in ['Dataset', 'Model', 'Mode', 'Error'])
        
        fieldnames = ['Dataset', 'Model', 'Mode'] + sorted(all_metrics) + ['Error']
        
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"\nResults saved to: {args.output_csv}")
        print(f"Total results: {len(results)}")
    else:
        print("No results to save")

if __name__ == "__main__":
    main()


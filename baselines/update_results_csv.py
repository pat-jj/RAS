#!/usr/bin/env python3
"""
Update results_summary.csv with KnowTrace results.
Preserves existing results and adds/updates KnowTrace entries.
"""

import os
import csv
import sys
import argparse
from pathlib import Path

# Add framework to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'framework'))
from evaluate_results import load_and_prepare_data, evaluate_short_answer, evaluate_long_form

def parse_filename(filename):
    """Parse filename to extract model, dataset, and mode"""
    basename = os.path.basename(filename)
    name = basename.replace('.jsonl', '').replace('.json', '')
    
    # Extract dataset (first part before underscore)
    parts = name.split('_')
    if len(parts) < 2:
        return None, None, None
    
    dataset = parts[0]
    
    # Check for knowtrace model
    if 'knowtrace' in name.lower():
        # Find model (sonnet, llama3_8b, or llama2_7b)
        if 'sonnet' in name:
            model = 'sonnet'
        elif 'llama2' in name or 'llama-2' in name:
            model = 'llama2_7b'
        elif 'llama' in name or 'llama3' in name:
            model = 'llama3_8b'
        else:
            model = 'knowtrace'
        mode = 'knowtrace'
    else:
        # Existing parsing logic
        models = ['qwen3_8b', 'llama3_8b', 'sonnet4.5', 'sonnet', 'haiku', 'opus']
        model = None
        for m in models:
            if m in name:
                model = m
                break
        
        mode = 'base'
        if 'retrieval' in name:
            if 'sure' in name:
                mode = 'sure'
            else:
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
            # Short-form metrics
            results = {}
            
            # Always calculate MATCH
            match_score = evaluate_short_answer(valid_data, 'match')
            results['MATCH'] = f"{match_score:.4f}"
            
            # Also calculate dataset-specific metric
            if dataset_name == 'arc_c' or 'arc' in dataset_name:
                accuracy_score = evaluate_short_answer(valid_data, 'accuracy')
                results['ACCURACY'] = f"{accuracy_score:.4f}"
            elif dataset_name == 'pubhealth':
                pass
            else:
                # TriviaQA, PopQA, Bio, 2WikiMultiHop - calculate F1
                f1_score = evaluate_short_answer(valid_data, 'f1')
                results['F1'] = f"{f1_score:.4f}"
            
            return results, None
    except Exception as e:
        import traceback
        return None, f"{str(e)}: {traceback.format_exc()[:200]}"

def load_existing_csv(csv_path):
    """Load existing CSV and return as list of dicts"""
    if not os.path.exists(csv_path):
        return []
    
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    return rows, fieldnames

def update_csv_with_knowtrace(results_dir, csv_path):
    """Update CSV with KnowTrace results"""
    # Find KnowTrace result files
    knowtrace_files = []
    for file_path in Path(results_dir).glob("*knowtrace*.jsonl"):
        knowtrace_files.append(str(file_path))
    for file_path in Path(results_dir).glob("*knowtrace*.json"):
        knowtrace_files.append(str(file_path))
    
    if not knowtrace_files:
        print("No KnowTrace result files found")
        return
    
    print(f"Found {len(knowtrace_files)} KnowTrace result files")
    
    # Load existing CSV
    existing_rows, fieldnames = load_existing_csv(csv_path)
    
    # Create a key for existing rows: (Dataset, Model, Mode)
    existing_keys = {}
    for row in existing_rows:
        key = (row.get('Dataset', ''), row.get('Model', ''), row.get('Mode', ''))
        existing_keys[key] = row
    
    # Evaluate KnowTrace files and update
    for file_path in knowtrace_files:
        dataset, model, mode = parse_filename(file_path)
        if not dataset or not model:
            print(f"Skipping {file_path}: Could not parse filename")
            continue
        
        print(f"Evaluating: {dataset} - {model} - {mode}")
        metrics, error = evaluate_file(file_path, dataset)
        
        key = (dataset, model, mode)
        
        if metrics:
            # Create or update row
            if key in existing_keys:
                # Update existing row
                row = existing_keys[key]
                for metric, value in metrics.items():
                    row[metric] = value
                if 'Error' in row:
                    del row['Error']
                print(f"  Updated: {dataset} - {model} - {mode}")
            else:
                # Create new row
                row = {
                    'Dataset': dataset,
                    'Model': model,
                    'Mode': mode,
                    **metrics
                }
                existing_keys[key] = row
                existing_rows.append(row)
                print(f"  Added: {dataset} - {model} - {mode}")
        else:
            # Add error row
            if key not in existing_keys:
                row = {
                    'Dataset': dataset,
                    'Model': model,
                    'Mode': mode,
                    'Error': error
                }
                existing_keys[key] = row
                existing_rows.append(row)
                print(f"  Added (error): {dataset} - {model} - {mode}: {error}")
            else:
                # Update existing row with error
                existing_keys[key]['Error'] = error
                print(f"  Updated (error): {dataset} - {model} - {mode}: {error}")
    
    # Get all unique fieldnames
    all_fieldnames = set(['Dataset', 'Model', 'Mode'])
    for row in existing_rows:
        all_fieldnames.update(k for k in row.keys())
    
    # Sort fieldnames: Dataset, Model, Mode, then metrics alphabetically, then Error
    metric_fields = sorted([f for f in all_fieldnames if f not in ['Dataset', 'Model', 'Mode', 'Error']])
    fieldnames = ['Dataset', 'Model', 'Mode'] + metric_fields + ['Error']
    
    # Write updated CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            # Ensure all fields are present
            complete_row = {field: row.get(field, '') for field in fieldnames}
            writer.writerow(complete_row)
    
    print(f"\nUpdated CSV saved to: {csv_path}")
    print(f"Total rows: {len(existing_rows)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, 
                       default='/shared/rsaas/pj20/firas_data/test_datasets/results',
                       help='Directory containing result files')
    parser.add_argument('--csv_path', type=str,
                       default='/home/pj20/server-04/FIRAS/baselines/logs/results_summary.csv',
                       help='Path to results_summary.csv')
    args = parser.parse_args()
    
    update_csv_with_knowtrace(args.results_dir, args.csv_path)

if __name__ == "__main__":
    main()


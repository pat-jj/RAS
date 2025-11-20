#!/usr/bin/env python3
"""
Evaluate RAS results across all datasets and generate a summary.
"""

import os
import json
import sys
from pathlib import Path

# Add baselines path for evaluation
baselines_path = Path(__file__).parent.parent / "baselines"
framework_path = Path(__file__).parent
if baselines_path.exists():
    sys.path.insert(0, str(baselines_path))
if framework_path.exists():
    sys.path.insert(0, str(framework_path))
    
# Import evaluation functions
try:
    from evaluate_results import load_and_prepare_data, evaluate_short_answer, evaluate_long_form
    from metrics import match, f1_score, accuracy, compute_rouge, mauve_score
    HAS_METRICS = True
except ImportError as e:
    print(f"Warning: Could not import evaluation functions: {e}")
    HAS_METRICS = False

def find_ras_result_files(results_dir):
    """Find all RAS result files"""
    result_files = {}
    results_dir = Path(results_dir)
    
    # Pattern: {dataset}_test_output_{planner_model}_{answerer_model}_v3.json
    for file in results_dir.glob("*_test_output_*_v3.json"):
        # Parse dataset name
        parts = file.stem.split("_test_output_")
        if len(parts) == 2:
            dataset = parts[0]
            result_files[dataset] = str(file)
    
    return result_files

def evaluate_all_ras_results(results_dir="/shared/rsaas/pj20/firas_data/test_datasets"):
    """Evaluate all RAS results and generate summary"""
    print("=" * 60)
    print("RAS Performance Evaluation Across Datasets")
    print("=" * 60)
    print()
    
    result_files = find_ras_result_files(results_dir)
    
    if not result_files:
        print("❌ No RAS result files found!")
        return
    
    print(f"Found {len(result_files)} result files:")
    for dataset, filepath in sorted(result_files.items()):
        print(f"  - {dataset}: {filepath}")
    print()
    
    # Evaluate each dataset
    results_summary = []
    
    for dataset, filepath in sorted(result_files.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset}")
        print(f"{'='*60}")
        
        try:
            if not HAS_METRICS:
                print(f"⚠️  {dataset}: Evaluation functions not available")
                continue
                
            # Load and prepare data (pass dataset name)
            data = load_and_prepare_data(filepath, dataset)
            if not data:
                print(f"⚠️  {dataset}: No data loaded")
                continue
            
            # Determine dataset type and evaluate
            metrics = {}
            if dataset in ['asqa', 'eli5']:
                # Long-form QA
                try:
                    rouge = compute_rouge(data)
                    metrics['rouge'] = rouge
                    print(f"  ROUGE-L: {rouge:.2f}")
                except Exception as e:
                    print(f"  ROUGE error: {e}")
                
                try:
                    # Prepare for MAUVE
                    references = []
                    predictions = []
                    for item in data:
                        ref = item.get('answer', item.get('annotations', [{}])[0].get('long_answer', ''))
                        pred = item.get('output', '')
                        if isinstance(ref, list):
                            ref = ref[0] if ref else ''
                        references.append(str(ref).lower())
                        predictions.append(str(pred).lower())
                    
                    mauve = mauve_score(predictions, references)
                    metrics['mauve'] = mauve
                    print(f"  MAUVE: {mauve:.2f}")
                except Exception as e:
                    print(f"  MAUVE error: {e}")
            else:
                # Short-form QA
                match_scores = []
                f1_scores = []
                acc_scores = []
                
                for item in data:
                    output = item.get('output', '')
                    if isinstance(output, list):
                        output = ' '.join(str(x) for x in output) if output else ''
                    output = str(output).lower()
                    
                    golds = item.get('golds', item.get('answer', []))
                    if not isinstance(golds, list):
                        golds = [golds]
                    golds = [str(g).lower() for g in golds]
                    
                    if output and golds:
                        match_scores.append(match(output, golds))
                        f1_scores.append(f1_score(output, golds))
                        if dataset == 'arc_c' or dataset == 'pubhealth':
                            acc_scores.append(accuracy([output], [golds]))
                
                if match_scores:
                    metrics['match'] = sum(match_scores) / len(match_scores) * 100
                    print(f"  MATCH: {metrics['match']:.2f}")
                if f1_scores:
                    metrics['f1'] = sum(f1_scores) / len(f1_scores) * 100
                    print(f"  F1: {metrics['f1']:.2f}")
                if acc_scores:
                    metrics['accuracy'] = sum(acc_scores) / len(acc_scores)
                    print(f"  ACCURACY: {metrics['accuracy']:.2f}")
            
            if metrics:
                results_summary.append({
                    'dataset': dataset,
                    'file': filepath,
                    **metrics
                })
                print(f"✅ {dataset}: {metrics}")
            else:
                print(f"⚠️  {dataset}: No metrics computed")
        except Exception as e:
            print(f"❌ {dataset}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Dataset':<20} {'F1':<10} {'MATCH':<10} {'ACCURACY':<10} {'ROUGE-L':<10} {'MAUVE':<10}")
    print("-" * 60)
    
    for result in results_summary:
        dataset = result['dataset']
        f1 = result.get('f1', 'N/A')
        match = result.get('match', 'N/A')
        accuracy = result.get('accuracy', 'N/A')
        rouge = result.get('rouge', 'N/A')
        mauve = result.get('mauve', 'N/A')
        
        print(f"{dataset:<20} {str(f1):<10} {str(match):<10} {str(accuracy):<10} {str(rouge):<10} {str(mauve):<10}")
    
    # Save summary to JSON
    summary_file = Path(results_dir) / "ras_results_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✅ Summary saved to: {summary_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, 
                       default='/shared/rsaas/pj20/firas_data/test_datasets',
                       help='Directory containing result files')
    args = parser.parse_args()
    
    evaluate_all_ras_results(args.results_dir)


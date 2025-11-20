#!/usr/bin/env python3
"""
Helper script to update progress tracking file
"""

import json
import os
from datetime import datetime
from typing import Dict, List

PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "PROGRESS.md")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "example_results")


def parse_results_file(results_file: str) -> Dict:
    """Parse a results JSON file and extract key information"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    stats = data.get('statistics', {})
    top_examples = data.get('top_examples', [])
    args = data.get('args', {})
    
    return {
        'dataset': args.get('dataset', 'unknown'),
        'num_samples': args.get('num_samples', 0),
        'top_k': args.get('top_k', 0),
        'stats': stats,
        'top_examples': top_examples,
        'timestamp': os.path.basename(results_file).split('_')[-1].replace('.json', '')
    }


def update_progress_from_results(results_file: str):
    """Update PROGRESS.md with information from a results file"""
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return
    
    result = parse_results_file(results_file)
    
    # Read current progress file
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read()
    else:
        content = ""
    
    # Generate experiment section
    exp_section = f"""
### Experiment: {result['dataset']} - {result['timestamp']}

**Configuration:**
- Dataset: {result['dataset']}
- Number of samples: {result['num_samples']}
- Top K: {result['top_k']}

**Results:**
- Total samples: {result['stats'].get('total_samples', 0)}
- Graph wins: {result['stats'].get('graph_wins', 0)} ({result['stats'].get('graph_wins', 0)/max(result['stats'].get('total_samples', 1), 1)*100:.1f}%)
- Text wins: {result['stats'].get('text_wins', 0)} ({result['stats'].get('text_wins', 0)/max(result['stats'].get('total_samples', 1), 1)*100:.1f}%)
- Both correct: {result['stats'].get('both_correct', 0)} ({result['stats'].get('both_correct', 0)/max(result['stats'].get('total_samples', 1), 1)*100:.1f}%)
- Both wrong: {result['stats'].get('both_wrong', 0)} ({result['stats'].get('both_wrong', 0)/max(result['stats'].get('total_samples', 1), 1)*100:.1f}%)
- Average Sonnet rating: {result['stats'].get('avg_rating', 0):.2f}/10.0

**Top Examples:**
"""
    
    for i, ex in enumerate(result['top_examples'][:3], 1):
        exp_section += f"""
{i}. Example {i} (Rating: {ex.get('sonnet_rating', 0):.2f}/10.0)
   - Question: {ex.get('question', '')[:150]}...
   - Gold Answer: {ex.get('gold_answer', '')}
   - Graph Answer: {ex.get('graph_answer', '')[:150]}...
   - Text Answer: {ex.get('text_answer', '')[:150]}...
   - Complexity Score: {ex.get('complexity', {}).get('complexity_score', 0):.0f}
"""
    
    exp_section += f"""
**Correlation Analysis:**
- Average complexity (graph wins): {result['stats'].get('avg_complexity_graph_wins', 0):.0f}
- Average complexity (all): {result['stats'].get('avg_complexity_all', 0):.0f}
- Complexity ratio: {result['stats'].get('avg_complexity_graph_wins', 0)/max(result['stats'].get('avg_complexity_all', 1), 1):.2f}x

**Output File:** {os.path.basename(results_file)}

---
"""
    
    # Find where to insert (after "## Experiments Run")
    if "## Experiments Run" in content:
        # Insert after the header
        insert_pos = content.find("## Experiments Run") + len("## Experiments Run")
        # Find next section or end
        next_section = content.find("\n## ", insert_pos)
        if next_section == -1:
            next_section = len(content)
        
        new_content = content[:next_section] + exp_section + content[next_section:]
    else:
        # Add experiments section
        if "## Status Overview" in content:
            insert_pos = content.find("## Status Overview") + len("## Status Overview")
            next_section = content.find("\n## ", insert_pos)
            if next_section == -1:
                next_section = len(content)
            new_content = content[:next_section] + "\n\n## Experiments Run\n" + exp_section + content[next_section:]
        else:
            new_content = content + "\n\n## Experiments Run\n" + exp_section
    
    # Write updated content
    with open(PROGRESS_FILE, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Updated PROGRESS.md with results from {os.path.basename(results_file)}")


def list_recent_results():
    """List recent results files"""
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return []
    
    results_files = []
    for f in os.listdir(RESULTS_DIR):
        if '_results_' in f and f.endswith('.json'):
            results_files.append(os.path.join(RESULTS_DIR, f))
    
    # Sort by modification time
    results_files.sort(key=os.path.getmtime, reverse=True)
    return results_files


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Update progress tracking from results')
    parser.add_argument('--results_file', type=str, help='Path to results JSON file')
    parser.add_argument('--auto', action='store_true', help='Automatically update with most recent results file')
    
    args = parser.parse_args()
    
    if args.auto:
        results_files = list_recent_results()
        if results_files:
            print(f"Found {len(results_files)} results file(s)")
            print(f"Updating with most recent: {os.path.basename(results_files[0])}")
            update_progress_from_results(results_files[0])
        else:
            print("No results files found")
    elif args.results_file:
        update_progress_from_results(args.results_file)
    else:
        print("Usage:")
        print("  python track_progress.py --results_file <path_to_json>")
        print("  python track_progress.py --auto  # Use most recent results file")


if __name__ == "__main__":
    main()


import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_json, load_torch, save_json

class RetrievalAnalyzer:
    def __init__(self):
        self.wiki_dir = Path("/shared/eng/pj20/hotpotqa/data/processed_wiki")
        self.emb_dir = Path("/shared/eng/pj20/hotpotqa/data/embeddings")
        self.hotpot_dir = Path("/shared/eng/pj20/hotpotqa/data/processed_hotpot")
        self.results_dir = Path("/shared/eng/pj20/hotpotqa/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze_single_example(self, example, question_emb, corpus_data):
        """Analyze retrieval for a single example with both original question and subqueries"""
        results = {
            'question': example['question'],
            'supporting_titles': example['supporting_titles'],
            'subqueries': example['subqueries'],
            'retrieval_results': {}
        }
        
        # Analyze original question
        original_sims = torch.matmul(question_emb, corpus_data['embeddings'].T)
        original_ranks = self.get_ranks(original_sims, example['supporting_titles'], corpus_data['titles'])
        results['retrieval_results']['original'] = {
            'ranks': original_ranks,
            'max_similarity': float(original_sims.max()),
            'supporting_doc_similarities': [float(original_sims[i]) for i in range(len(original_sims))]
        }
        
        # Analyze each subquery
        for i, subquery in enumerate(example['subqueries']):
            subquery_emb = self.get_embedding(subquery)
            subquery_sims = torch.matmul(subquery_emb, corpus_data['embeddings'].T)
            subquery_ranks = self.get_ranks(subquery_sims, example['supporting_titles'], corpus_data['titles'])
            
            results['retrieval_results'][f'subquery_{i+1}'] = {
                'subquery': subquery,
                'ranks': subquery_ranks,
                'max_similarity': float(subquery_sims.max()),
                'supporting_doc_similarities': [float(subquery_sims[i]) for i in range(len(subquery_sims))]
            }
        
        return results

    def get_ranks(self, similarities, target_titles, all_titles):
        """Get ranks of target documents in similarity order"""
        _, indices = torch.sort(similarities, descending=True)
        ranks = {}
        for title in target_titles:
            try:
                idx = all_titles.index(title)
                rank = (indices == idx).nonzero()[0].item()
                ranks[title] = rank + 1
            except:
                ranks[title] = float('inf')
        return ranks

    def analyze_case_studies(self, results):
        """Analyze interesting cases showing retrieval deficiencies"""
        case_studies = {
            'hard_cases': [],
            'subquery_better': [],
            'subquery_worse': []
        }
        
        for example in results:
            orig_ranks = example['retrieval_results']['original']['ranks']
            subq1_ranks = example['retrieval_results']['subquery_1']['ranks']
            subq2_ranks = example['retrieval_results']['subquery_2']['ranks']
            
            # Hard cases where both original and subqueries fail
            if all(r > 100 for r in orig_ranks.values()):
                case_studies['hard_cases'].append(example)
            
            # Cases where subqueries perform better
            if (min(subq1_ranks.values()) < min(orig_ranks.values()) or 
                min(subq2_ranks.values()) < min(orig_ranks.values())):
                case_studies['subquery_better'].append(example)
            
            # Cases where subqueries perform worse
            if (min(subq1_ranks.values()) > min(orig_ranks.values()) and 
                min(subq2_ranks.values()) > min(orig_ranks.values())):
                case_studies['subquery_worse'].append(example)
        
        return case_studies

    def generate_reports(self, results, case_studies):
        """Generate detailed analysis reports"""
        report = {
            'overall_statistics': self.compute_statistics(results),
            'case_studies': {
                'hard_cases': self.format_cases(case_studies['hard_cases'][:5]),
                'subquery_better': self.format_cases(case_studies['subquery_better'][:5]),
                'subquery_worse': self.format_cases(case_studies['subquery_worse'][:5])
            }
        }
        
        save_json(report, self.results_dir / 'detailed_analysis.json')
        
        # Generate visualizations
        self.plot_rank_distributions(results)
        self.plot_similarity_distributions(results)

    def compute_statistics(self, results):
        """Compute overall retrieval statistics"""
        stats = {
            'total_examples': len(results),
            'original_question': {
                'mean_rank': np.mean([min(r['retrieval_results']['original']['ranks'].values()) for r in results]),
                'success_rate_top10': np.mean([min(r['retrieval_results']['original']['ranks'].values()) <= 10 for r in results]),
                'success_rate_top100': np.mean([min(r['retrieval_results']['original']['ranks'].values()) <= 100 for r in results])
            },
            'subqueries': {
                'mean_rank': np.mean([
                    min(
                        min(r['retrieval_results']['subquery_1']['ranks'].values()),
                        min(r['retrieval_results']['subquery_2']['ranks'].values())
                    ) for r in results
                ]),
                'success_rate_top10': np.mean([
                    min(
                        min(r['retrieval_results']['subquery_1']['ranks'].values()),
                        min(r['retrieval_results']['subquery_2']['ranks'].values())
                    ) <= 10 for r in results
                ])
            }
        }
        return stats

    def format_cases(self, cases):
        """Format case studies for readable output"""
        formatted = []
        for case in cases:
            formatted.append({
                'question': case['question'],
                'subqueries': case['subqueries'],
                'supporting_titles': case['supporting_titles'],
                'original_ranks': case['retrieval_results']['original']['ranks'],
                'subquery1_ranks': case['retrieval_results']['subquery_1']['ranks'],
                'subquery2_ranks': case['retrieval_results']['subquery_2']['ranks'],
                'analysis': self.analyze_case(case)
            })
        return formatted

    def analyze_case(self, case):
        """Provide detailed analysis for a single case"""
        orig_ranks = case['retrieval_results']['original']['ranks']
        subq1_ranks = case['retrieval_results']['subquery_1']['ranks']
        subq2_ranks = case['retrieval_results']['subquery_2']['ranks']
        
        analysis = []
        
        # Analyze semantic gap
        for title in case['supporting_titles']:
            orig_sim = case['retrieval_results']['original']['supporting_doc_similarities'][0]
            subq1_sim = case['retrieval_results']['subquery_1']['supporting_doc_similarities'][0]
            subq2_sim = case['retrieval_results']['subquery_2']['supporting_doc_similarities'][0]
            
            analysis.append(f"Document '{title}':")
            analysis.append(f"- Original question similarity: {orig_sim:.3f} (rank: {orig_ranks[title]})")
            analysis.append(f"- Subquery 1 similarity: {subq1_sim:.3f} (rank: {subq1_ranks[title]})")
            analysis.append(f"- Subquery 2 similarity: {subq2_sim:.3f} (rank: {subq2_ranks[title]})")
        
        return '\n'.join(analysis)

    def plot_rank_distributions(self, results):
        """Plot distribution of ranks for original vs subqueries"""
        orig_ranks = [min(r['retrieval_results']['original']['ranks'].values()) for r in results]
        subq_ranks = [min(
            min(r['retrieval_results']['subquery_1']['ranks'].values()),
            min(r['retrieval_results']['subquery_2']['ranks'].values())
        ) for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist([orig_ranks, subq_ranks], label=['Original', 'Subqueries'])
        plt.xlabel('Rank of First Supporting Document')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(self.results_dir / 'rank_distribution.png')
        plt.close()

def main():
    analyzer = RetrievalAnalyzer()
    
    # Load data
    examples = load_json(analyzer.hotpot_dir / 'processed_hotpot_with_subqueries.json')
    
    # Run analysis
    results = []
    for example in tqdm(examples):
        result = analyzer.analyze_single_example(example)
        results.append(result)
    
    # Generate case studies
    case_studies = analyzer.analyze_case_studies(results)
    
    # Generate reports and visualizations
    analyzer.generate_reports(results, case_studies)

if __name__ == "__main__":
    main()
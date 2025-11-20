#!/usr/bin/env python3
"""
Find convincing examples demonstrating how graph representations outperform plain text
in enhancing reasoning accuracy. This script:
1. Runs RAS with graph vs text-only representations
2. Evaluates examples with Sonnet-3.5 to rate them (0-10)
3. Analyzes correlation between complexity and graph improvements
4. Considers context rot effects
5. Selects top 2-3 examples for rebuttal
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import torch
import re

# Add framework path
framework_path = os.path.dirname(__file__)
if framework_path not in sys.path:
    sys.path.insert(0, framework_path)

from sonnet import planner_sonnet, answerer_sonnet, text_to_triples_sonnet
from utils import GraphProcessor, convert_triple_str_to_graph
from td_retriever import DenseRetriever
from claude_api import get_claude_response


def ras_with_graphs(question: str, context: List[str], graph_processor: GraphProcessor, 
                    retriever: Optional[DenseRetriever] = None, max_iteration: int = 3, 
                    max_answer_length: int = 200, debug: bool = False) -> Dict:
    """
    RAS with graph representations (normal RAS)
    """
    end_iteration_flag = False
    sub_query = question
    graphs = []
    retrieved_docs_list = []
    triple_lists = []
    subqueries = []
    inputs = []
    
    iteration = 0
    while not end_iteration_flag and iteration < max_iteration:
        if debug:
            print(f"[GRAPH] Iteration {iteration+1} starts ...")
            print(f"[GRAPH] Sub query: {sub_query}")
        
        # Stage 1: Retrieval
        if iteration == 0:
            retrieved_docs = context[:5] if len(context) > 5 else context
        else:
            if retriever is None:
                break
            try:
                retrieved_docs = retriever.retrieve(sub_query, top_k=5)
                if retrieved_docs and len(retrieved_docs) > 0:
                    if isinstance(retrieved_docs[0], tuple):
                        retrieved_docs = [item[0] for item in retrieved_docs]
            except Exception as e:
                if debug:
                    print(f"[GRAPH] Retrieval error: {e}")
                break
        
        retrieved_docs_list.append(retrieved_docs)
        
        # Stage 2: Text-to-triples-to-graph
        triples = text_to_triples_sonnet("\n".join(retrieved_docs)).replace("\n", " ")
        
        # Convert to graph structure
        graph, triples_ = convert_triple_str_to_graph(triples, graph_processor)
        if graph is None or triples_ is None:
            if debug:
                print(f"[GRAPH] Error processing triples: {triples[:200]}")
            iteration += 1
            continue
        
        graphs.append(graph)
        triple_lists.append(triples_)
        subqueries.append(sub_query)
        
        # Build planner input with graph information
        planner_input = ""
        for i in range(len(subqueries)):
            planner_input += subqueries[i] + "\n" + "Retrieved Graph Information: " + str(triple_lists[i]) + '\n'
        planner_input += "Question: " + question
        planner_input = planner_input.replace("Retrieved Graph Information:", "[PREV_GRAPH_INFO]").replace("[SUBQ]", "[PREV_SUBQ]")
        inputs.append(planner_input)
        
        if debug:
            print(f"[GRAPH] Planner input length: {len(planner_input)} chars")
        
        # Stage 3: Plan next action
        planner_output = planner_sonnet(planner_input)
        
        if debug:
            print(f"[GRAPH] Planner output: {planner_output}")
        
        if 'SUFFICIENT'.lower() in planner_output.lower() or 'NO_RETRIEVAL'.lower() in planner_output.lower():
            end_iteration_flag = True
        else:
            sub_query = planner_output.replace('[SUBQ]', '').strip()
        
        iteration += 1
    
    # Stage 4: Answering with graphs
    if len(inputs) == 0:
        question_text = "Question: " + question
        inputs.append(question_text)
    
    answerer_output = answerer_sonnet(inputs[-1], max_answer_length=max_answer_length)
    
    return {
        'answer': answerer_output,
        'graphs': graphs,
        'triple_lists': triple_lists,
        'subqueries': subqueries,
        'inputs': inputs,
        'num_iterations': iteration,
        'total_triples': sum(len(t) for t in triple_lists) if triple_lists else 0,
        'retrieved_docs_list': retrieved_docs_list
    }


def ras_with_text_only(question: str, context: List[str], 
                       retriever: Optional[DenseRetriever] = None, max_iteration: int = 3,
                       max_answer_length: int = 200, debug: bool = False) -> Dict:
    """
    RAS with text-only (no graph conversion, just raw text)
    This simulates what happens when we use plain text instead of graph structures
    """
    end_iteration_flag = False
    sub_query = question
    retrieved_docs_list = []
    text_lists = []  # Store raw text instead of triples
    subqueries = []
    inputs = []
    
    iteration = 0
    while not end_iteration_flag and iteration < max_iteration:
        if debug:
            print(f"[TEXT] Iteration {iteration+1} starts ...")
            print(f"[TEXT] Sub query: {sub_query}")
        
        # Stage 1: Retrieval
        if iteration == 0:
            retrieved_docs = context[:5] if len(context) > 5 else context
        else:
            if retriever is None:
                break
            try:
                retrieved_docs = retriever.retrieve(sub_query, top_k=5)
                if retrieved_docs and len(retrieved_docs) > 0:
                    if isinstance(retrieved_docs[0], tuple):
                        retrieved_docs = [item[0] for item in retrieved_docs]
            except Exception as e:
                if debug:
                    print(f"[TEXT] Retrieval error: {e}")
                break
        
        retrieved_docs_list.append(retrieved_docs)
        
        # Stage 2: Use raw text (no graph conversion)
        # Just concatenate the retrieved documents as plain text
        raw_text = "\n".join(retrieved_docs)
        text_lists.append(raw_text)
        subqueries.append(sub_query)
        
        # Build planner input with raw text (simulating RPG or text-only approach)
        planner_input = ""
        for i in range(len(subqueries)):
            planner_input += subqueries[i] + "\n" + "Retrieved Information: " + text_lists[i] + '\n'
        planner_input += "Question: " + question
        planner_input = planner_input.replace("Retrieved Information:", "[PREV_INFO]").replace("[SUBQ]", "[PREV_SUBQ]")
        inputs.append(planner_input)
        
        if debug:
            print(f"[TEXT] Planner input length: {len(planner_input)} chars")
        
        # Stage 3: Plan next action (using same planner but with text instead of graphs)
        planner_instruction = """You are a planner to determine if the question can be answered with current information (Subquery [PREV_SUBQ] and retrieved information [PREV_INFO]) and output the appropriate label as well as the subquery if needed.
Output [NO_RETRIEVAL] if the question can be directly answered with the question itself without any retrieval.
Output [SUBQ] with an subquery for retrieval if still needs a subquery. Do not make an similar subquery that has been made before ([PREV_SUBQ]), as it is very likely to retrieve the same information.
Output [SUFFICIENT] if the question can be answered with the provided information.
The main question starts with "Question: ".
"""
        planner_prompt = planner_instruction + "\nInput:\n" + planner_input + "\n\nOutput:"
        planner_output = get_claude_response(llm="sonnet", prompt=planner_prompt, max_tokens=200)
        
        if debug:
            print(f"[TEXT] Planner output: {planner_output}")
        
        if 'SUFFICIENT'.lower() in planner_output.lower() or 'NO_RETRIEVAL'.lower() in planner_output.lower():
            end_iteration_flag = True
        else:
            sub_query = planner_output.replace('[SUBQ]', '').strip()
        
        iteration += 1
    
    # Stage 4: Answering with text only
    if len(inputs) == 0:
        question_text = "Question: " + question
        inputs.append(question_text)
    
    # Use text-only answerer (similar format but with raw text)
    answerer_instruction = """You are a answerer given a question and retrieved information.
Each [SUBQ] is a subquery we generated through reasoning for the question. The retrieved information follows each [SUBQ] is relevant information we retrieved to answer the subquery.
The main question starts with "Question: ". Please answer the question, with subqueries and retrieved information if they are helpful (do not use them if they are not helpful).
You must answer the question, even if there's no enough information to answer the question, or you are not sure about the answer.
"""
    answerer_prompt = answerer_instruction + "\nInput:\n" + inputs[-1] + "\n\nOutput:"
    answerer_output = get_claude_response(llm="sonnet", prompt=answerer_prompt, max_tokens=max_answer_length)
    
    return {
        'answer': answerer_output,
        'text_lists': text_lists,
        'subqueries': subqueries,
        'inputs': inputs,
        'num_iterations': iteration,
        'total_text_length': sum(len(t) for t in text_lists) if text_lists else 0,
        'retrieved_docs_list': retrieved_docs_list
    }


def load_test_data(dataset: str, test_data_path: str, num_samples: int = 50):
    """Load test data for a dataset"""
    questions = []
    contexts = []
    answers = []
    
    if dataset == 'arc_c':
        data_path = os.path.join(test_data_path, dataset + "_test_processed.jsonl")
        import jsonlines
        with jsonlines.open(data_path) as reader:
            for i, item in enumerate(reader):
                if i >= num_samples:
                    break
                questions.append(item['instruction'])
                contexts.append([d['text'] for d in item['ctxs']][:5])
                answers.append(item['answerKey'])
    elif dataset == 'asqa':
        data_path = os.path.join(test_data_path, dataset + "_test_processed.json")
        with open(data_path, 'r') as f:
            data = json.load(f)['data']
        for i, item in enumerate(data[:num_samples]):
            questions.append(item['question'])
            contexts.append([d['text'] for d in item['ctxs']][:5])
            answers.append(item['answer'])
    elif dataset == '2wikimultihop':
        data_path = os.path.join(test_data_path, dataset + "_test.json")
        with open(data_path, 'r') as f:
            data = json.load(f)
        for i in range(min(num_samples, len(data['question']))):
            questions.append(data['question'][i])
            contexts.append(data['context'][i][:5])
            answers.append(data.get('answer', [''])[i] if isinstance(data.get('answer'), list) else '')
    else:
        # Default format
        data_path = os.path.join(test_data_path, dataset + "_test.json")
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            for i in range(min(num_samples, len(data.get('question', [])))):
                questions.append(data['question'][i])
                contexts.append(data['context'][i][:5] if isinstance(data['context'][i], list) else [data['context'][i]])
                answers.append(data.get('answer', [''])[i] if isinstance(data.get('answer'), list) else '')
    
    return questions, contexts, answers


def calculate_complexity(context: List[str], question: str) -> Dict:
    """Calculate task complexity metrics"""
    total_text_length = sum(len(doc) for doc in context)
    num_docs = len(context)
    avg_doc_length = total_text_length / num_docs if num_docs > 0 else 0
    
    # Count entities and relationships (rough estimate)
    question_lower = question.lower()
    num_entities = len([w for w in question.split() if w[0].isupper()])
    
    # Multi-hop indicators
    multi_hop_keywords = ['which', 'what', 'where', 'when', 'who', 'how many', 'how much', 'why']
    num_question_words = sum(1 for kw in multi_hop_keywords if kw in question_lower)
    
    # Count relationships in question
    relationship_indicators = ['relationship', 'related', 'connection', 'between', 'and', 'or']
    num_relationships = sum(1 for kw in relationship_indicators if kw in question_lower)
    
    # Estimate reasoning steps needed
    reasoning_score = num_question_words + num_relationships + (1 if num_entities > 1 else 0)
    
    return {
        'total_text_length': total_text_length,
        'num_docs': num_docs,
        'avg_doc_length': avg_doc_length,
        'num_entities': num_entities,
        'num_question_words': num_question_words,
        'num_relationships': num_relationships,
        'reasoning_score': reasoning_score,
        'complexity_score': total_text_length * num_docs * (1 + num_entities) * (1 + reasoning_score)
    }


def update_runtime_progress(dataset: str, current: int, total: int, 
                           graph_wins: int, text_wins: int, both_correct: int, 
                           both_wrong: int, start_time: float, last_activity: str = ""):
    """Update PROGRESS.md with current runtime status"""
    progress_file = os.path.join(os.path.dirname(__file__), "PROGRESS.md")
    
    if not os.path.exists(progress_file):
        return
    
    try:
        with open(progress_file, 'r') as f:
            content = f.read()
        
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        
        # Format current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update current run status section
        status_section = f"""## Current Run Status

**Status:** Running

**Last Updated:** {current_time}

**Current Experiment:**
- Dataset: {dataset}
- Sample: {current} / {total}
- Graph wins: {graph_wins}
- Text wins: {text_wins}
- Both correct: {both_correct}
- Both wrong: {both_wrong}
- Start time: {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}
- Elapsed time: {elapsed_str}

**Recent Activity:**
- {last_activity if last_activity else f"Processing sample {current}/{total}"}

---
"""
        
        # Find and replace the current run status section
        if "## Current Run Status" in content:
            # Find the section
            start_pos = content.find("## Current Run Status")
            # Find the next section
            next_section = content.find("\n## ", start_pos + 1)
            if next_section == -1:
                next_section = len(content)
            
            # Replace the section
            new_content = content[:start_pos] + status_section + content[next_section:]
        else:
            # Insert at the beginning
            if "## Status Overview" in content:
                insert_pos = content.find("## Status Overview")
                new_content = status_section + "\n" + content
            else:
                new_content = status_section + "\n" + content
        
        with open(progress_file, 'w') as f:
            f.write(new_content)
    except Exception as e:
        # Silently fail - don't interrupt the main process
        pass


def clear_runtime_progress():
    """Clear the current run status when done"""
    progress_file = os.path.join(os.path.dirname(__file__), "PROGRESS.md")
    
    if not os.path.exists(progress_file):
        return
    
    try:
        with open(progress_file, 'r') as f:
            content = f.read()
        
        # Update status to "Not running"
        if "## Current Run Status" in content:
            start_pos = content.find("## Current Run Status")
            next_section = content.find("\n## ", start_pos + 1)
            if next_section == -1:
                next_section = len(content)
            
            status_section = """## Current Run Status

**Status:** Not running

**Last Updated:** [Not started]

**Current Experiment:**
- Dataset: 
- Sample: 0 / 0
- Graph wins: 0
- Text wins: 0
- Both correct: 0
- Both wrong: 0
- Start time: 
- Elapsed time: 

**Recent Activity:**
- 

---
"""
            new_content = content[:start_pos] + status_section + content[next_section:]
            
            with open(progress_file, 'w') as f:
                f.write(new_content)
    except Exception:
        pass


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set('.,!?;:()[]{}"\'-')
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> bool:
    """Check exact match after normalization"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm


def evaluate_example_with_sonnet(comparison: Dict) -> float:
    """
    Use Sonnet-3.5 to evaluate how convincing an example is (0-10 rating)
    """
    question = comparison['question']
    gold_answer = comparison['gold_answer']
    graph_answer = comparison['graph_answer']
    text_answer = comparison['text_answer']
    graph_correct = comparison['graph_correct']
    text_correct = comparison['text_correct']
    complexity = comparison['complexity']
    graph_triples = comparison.get('graph_triples', 0)
    text_length = comparison.get('text_length', 0)
    context_length = comparison.get('context_length', 0)
    
    # Build detailed evaluation prompt
    evaluation_prompt = f"""You are evaluating an example that demonstrates how graph representations outperform plain text for reasoning tasks.

**Question:** {question}

**Gold Answer:** {gold_answer}

**Graph-based Answer (RAS with graph representations):**
{graph_answer}

**Text-only Answer (RAS with plain text):**
{text_answer}

**Correctness:**
- Graph-based answer is correct: {graph_correct}
- Text-only answer is correct: {text_correct}

**Context Information:**
- Total context length: {context_length} characters
- Number of documents: {complexity['num_docs']}
- Graph representation: {graph_triples} triples extracted
- Text representation: {text_length} characters

**Task Complexity:**
- Number of entities: {complexity['num_entities']}
- Reasoning score: {complexity['reasoning_score']}
- Complexity score: {complexity['complexity_score']:.0f}

**Evaluation Criteria:**
Rate this example from 0-10 based on how convincingly it demonstrates that graph representations enhance reasoning accuracy over plain text. Consider:

1. **Clear superiority (0-10 points):** Does the graph-based approach clearly outperform text-only? How significant is the difference?

2. **Reasoning complexity (0-10 points):** Does this example involve multi-hop reasoning, compositional reasoning, or logical chaining that benefits from graph structure?

3. **Context rot mitigation (0-10 points):** Does the graph representation help avoid information loss or confusion that occurs with long text contexts? Does it preserve relationships better?

4. **Factual grounding (0-10 points):** Does the graph structure help maintain accurate factual relationships that might be lost or confused in plain text?

5. **Demonstrative value (0-10 points):** How well does this example illustrate the advantages of graph structuring? Is it clear and understandable?

6. **Task characteristics (0-10 points):** Does this example show benefits on complex reasoning tasks (factual grounding, compositional reasoning, logical chaining)?

**Instructions:**
- Provide a single numerical rating from 0.0 to 10.0
- Consider all criteria above, but weight them based on how well they demonstrate graph advantages
- Higher scores indicate more convincing examples that clearly show graph benefits
- Lower scores indicate examples where the difference is unclear, minimal, or not demonstrative

**Output format:**
Provide ONLY a single number between 0.0 and 10.0, nothing else. For example: 8.5
"""
    
    try:
        response = get_claude_response(llm="sonnet", prompt=evaluation_prompt, max_tokens=50)
        # Extract number from response
        rating_match = re.search(r'\d+\.?\d*', response.strip())
        if rating_match:
            rating = float(rating_match.group())
            return min(10.0, max(0.0, rating))  # Clamp to 0-10
        else:
            print(f"Warning: Could not parse rating from response: {response}")
            return 0.0
    except Exception as e:
        print(f"Error evaluating example: {e}")
        return 0.0


def analyze_context_rot(graph_result: Dict, text_result: Dict) -> Dict:
    """
    Analyze context rot effects - how information degrades in long text contexts
    vs how graphs preserve structure
    """
    graph_input_length = sum(len(inp) for inp in graph_result.get('inputs', []))
    text_input_length = sum(len(inp) for inp in text_result.get('inputs', []))
    
    # Graph representation is more compact (triples vs full text)
    graph_compactness = graph_result.get('total_triples', 0)
    text_verbosity = text_result.get('total_text_length', 0)
    
    # Information density
    graph_info_density = graph_compactness / max(graph_input_length, 1)
    text_info_density = text_verbosity / max(text_input_length, 1)
    
    return {
        'graph_input_length': graph_input_length,
        'text_input_length': text_input_length,
        'graph_compactness': graph_compactness,
        'text_verbosity': text_verbosity,
        'graph_info_density': graph_info_density,
        'text_info_density': text_info_density,
        'compactness_ratio': graph_compactness / max(text_verbosity, 1),
        'input_length_ratio': graph_input_length / max(text_input_length, 1)
    }


def compare_results(graph_result: Dict, text_result: Dict, gold_answer: str, 
                   complexity: Dict, question: str, context: List[str]) -> Dict:
    """Compare graph vs text results"""
    graph_correct = exact_match(graph_result['answer'], gold_answer)
    text_correct = exact_match(text_result['answer'], gold_answer)
    
    # Calculate context efficiency
    graph_triples = graph_result.get('total_triples', 0)
    text_length = text_result.get('total_text_length', 0)
    
    # Analyze context rot
    context_rot_analysis = analyze_context_rot(graph_result, text_result)
    
    comparison = {
        'question': question,
        'gold_answer': gold_answer,
        'graph_answer': graph_result['answer'],
        'text_answer': text_result['answer'],
        'graph_correct': graph_correct,
        'text_correct': text_correct,
        'graph_improves': graph_correct and not text_correct,
        'text_improves': text_correct and not graph_correct,
        'both_correct': graph_correct and text_correct,
        'both_wrong': not graph_correct and not text_correct,
        'graph_iterations': graph_result['num_iterations'],
        'text_iterations': text_result['num_iterations'],
        'graph_triples': graph_triples,
        'text_length': text_length,
        'complexity': complexity,
        'context_length': sum(len(doc) for doc in context),
        'num_context_docs': len(context),
        'context_rot_analysis': context_rot_analysis,
        'graph_subqueries': graph_result.get('subqueries', []),
        'text_subqueries': text_result.get('subqueries', []),
        'graph_triple_lists': graph_result.get('triple_lists', []),
        'text_lists': text_result.get('text_lists', [])
    }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Find convincing examples for graph vs text comparison')
    parser.add_argument('--dataset', type=str, default='2wikimultihop',
                       choices=['triviaqa', 'popqa', 'arc_c', 'pubhealth', 'bio', 
                               'asqa', 'eli5', '2wikimultihop'],
                       help='Dataset to test')
    parser.add_argument('--test_data_path', type=str, 
                       default='/shared/rsaas/pj20/firas_data/test_datasets')
    parser.add_argument('--knowledge_path', type=str, 
                       default='/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test')
    parser.add_argument('--max_iteration', type=int, default=3)
    parser.add_argument('--max_answer_length', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./example_results')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top examples to select')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Finding Convincing Examples: Graph vs Text Representations")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Top K examples to select: {args.top_k}")
    print("=" * 80)
    
    # Load test data
    print("\nLoading test data...")
    questions, contexts, answers = load_test_data(args.dataset, args.test_data_path, args.num_samples)
    print(f"Loaded {len(questions)} questions")
    
    # Initialize components
    print("\nInitializing components...")
    graph_processor = GraphProcessor()
    
    retriever = None
    if args.dataset not in ['asqa', 'eli5']:
        print("Initializing retriever (CPU mode, 3 splits to save memory)...")
        # Force CPU mode
        retriever = DenseRetriever(
            knowledge_path=args.knowledge_path,
            num_splits=3,  # Use 3 splits instead of 5 to save memory
            debug=args.debug,
            device=torch.device('cpu'),
            faiss_gpu_ids=[]  # No GPU IDs for FAISS - forces CPU
        )
        print("Retriever initialized (CPU mode, 3 splits)")
    
    # Run comparisons
    print("\n" + "=" * 80)
    print("Running comparisons...")
    print("=" * 80)
    
    results = []
    graph_wins = 0
    text_wins = 0
    both_correct = 0
    both_wrong = 0
    
    # Track start time for progress updates
    experiment_start_time = time.time()
    
    # Initialize progress tracking
    update_runtime_progress(
        args.dataset, 0, len(questions),
        graph_wins, text_wins, both_correct, both_wrong,
        experiment_start_time, "Starting experiment..."
    )
    
    for i, (question, context, gold_answer) in enumerate(zip(questions, contexts, answers)):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(questions)}")
        print(f"Question: {question[:150]}...")
        print(f"Gold Answer: {gold_answer}")
        print(f"{'='*80}")
        
        try:
            # Calculate complexity
            complexity = calculate_complexity(context, question)
            
            # Run with graphs
            print("\n[1/2] Running RAS with graph representations...")
            start_time = time.time()
            graph_result = ras_with_graphs(
                question, context, graph_processor, retriever,
                max_iteration=args.max_iteration,
                max_answer_length=args.max_answer_length,
                debug=args.debug
            )
            graph_time = time.time() - start_time
            print(f"Graph answer: {graph_result['answer'][:200]}...")
            print(f"Time: {graph_time:.2f}s")
            
            # Run with text only
            print("\n[2/2] Running RAS with text-only...")
            start_time = time.time()
            text_result = ras_with_text_only(
                question, context, retriever,
                max_iteration=args.max_iteration,
                max_answer_length=args.max_answer_length,
                debug=args.debug
            )
            text_time = time.time() - start_time
            print(f"Text answer: {text_result['answer'][:200]}...")
            print(f"Time: {text_time:.2f}s")
            
            # Compare results
            comparison = compare_results(
                graph_result, text_result, gold_answer, complexity, question, context
            )
            comparison['graph_time'] = graph_time
            comparison['text_time'] = text_time
            results.append(comparison)
            
            # Update statistics
            if comparison['graph_improves']:
                graph_wins += 1
                print("\n✅ GRAPH WINS!")
                activity = f"Sample {i+1}: Graph wins! ✅"
            elif comparison['text_improves']:
                text_wins += 1
                print("\n✅ TEXT WINS!")
                activity = f"Sample {i+1}: Text wins"
            elif comparison['both_correct']:
                both_correct += 1
                print("\n✅ BOTH CORRECT")
                activity = f"Sample {i+1}: Both correct"
            else:
                both_wrong += 1
                print("\n❌ BOTH WRONG")
                activity = f"Sample {i+1}: Both wrong"
            
            # Update runtime progress
            update_runtime_progress(
                args.dataset, i + 1, len(questions),
                graph_wins, text_wins, both_correct, both_wrong,
                experiment_start_time, activity
            )
            
            # Save intermediate results
            if (i + 1) % 10 == 0:
                output_file = os.path.join(args.output_dir, f"{args.dataset}_intermediate.json")
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Saved intermediate results to {output_file}")
        
        except Exception as e:
            print(f"\n❌ Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Evaluate examples with Sonnet-3.5
    print("\n" + "=" * 80)
    print("Evaluating examples with Sonnet-3.5...")
    print("=" * 80)
    
    graph_win_examples = [r for r in results if r['graph_improves']]
    print(f"Found {len(graph_win_examples)} examples where graph wins")
    
    # Update progress
    update_runtime_progress(
        args.dataset, len(questions), len(questions),
        graph_wins, text_wins, both_correct, both_wrong,
        experiment_start_time, f"Evaluating {len(graph_win_examples)} examples with Sonnet-3.5..."
    )
    
    for i, example in enumerate(graph_win_examples):
        print(f"\nEvaluating example {i+1}/{len(graph_win_examples)}...")
        rating = evaluate_example_with_sonnet(example)
        example['sonnet_rating'] = rating
        print(f"Rating: {rating:.2f}/10.0")
        
        # Update progress during evaluation
        update_runtime_progress(
            args.dataset, len(questions), len(questions),
            graph_wins, text_wins, both_correct, both_wrong,
            experiment_start_time, f"Evaluated {i+1}/{len(graph_win_examples)} examples (avg rating: {sum(ex.get('sonnet_rating', 0) for ex in graph_win_examples[:i+1])/(i+1):.2f})"
        )
    
    # Sort by rating
    graph_win_examples.sort(key=lambda x: x.get('sonnet_rating', 0), reverse=True)
    
    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total samples: {len(results)}")
    print(f"Graph wins: {graph_wins} ({graph_wins/len(results)*100:.1f}%)")
    print(f"Text wins: {text_wins} ({text_wins/len(results)*100:.1f}%)")
    print(f"Both correct: {both_correct} ({both_correct/len(results)*100:.1f}%)")
    print(f"Both wrong: {both_wrong} ({both_wrong/len(results)*100:.1f}%)")
    
    # Analyze correlation with complexity
    if graph_win_examples:
        avg_complexity_graph_wins = sum(r['complexity']['complexity_score'] for r in graph_win_examples) / len(graph_win_examples)
        avg_complexity_all = sum(r['complexity']['complexity_score'] for r in results) / len(results)
        avg_rating = sum(r.get('sonnet_rating', 0) for r in graph_win_examples) / len(graph_win_examples)
        
        print(f"\nAverage complexity (graph wins): {avg_complexity_graph_wins:.0f}")
        print(f"Average complexity (all): {avg_complexity_all:.0f}")
        print(f"Complexity ratio: {avg_complexity_graph_wins/avg_complexity_all:.2f}x")
        print(f"Average Sonnet rating: {avg_rating:.2f}/10.0")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"{args.dataset}_results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'statistics': {
                'total_samples': len(results),
                'graph_wins': graph_wins,
                'text_wins': text_wins,
                'both_correct': both_correct,
                'both_wrong': both_wrong,
                'avg_complexity_graph_wins': avg_complexity_graph_wins if graph_win_examples else 0,
                'avg_complexity_all': avg_complexity_all,
                'avg_rating': avg_rating if graph_win_examples else 0
            },
            'top_examples': graph_win_examples[:args.top_k],
            'all_results': results
        }, f, indent=2)
    
    print(f"\n💾 Saved final results to {output_file}")
    
    # Update progress tracking
    try:
        from track_progress import update_progress_from_results
        update_progress_from_results(output_file)
        print(f"✅ Updated PROGRESS.md with results")
    except Exception as e:
        print(f"⚠️  Could not update progress file: {e}")
    
    # Clear runtime status
    update_runtime_progress(
        args.dataset, len(questions), len(questions),
        graph_wins, text_wins, both_correct, both_wrong,
        experiment_start_time, "✅ Experiment completed!"
    )
    time.sleep(1)  # Brief pause to ensure file is written
    clear_runtime_progress()
    
    # Print top examples
    if graph_win_examples:
        print("\n" + "=" * 80)
        print(f"TOP {args.top_k} CONVINCING EXAMPLES")
        print("=" * 80)
        for i, ex in enumerate(graph_win_examples[:args.top_k], 1):
            print(f"\n{'='*80}")
            print(f"Example {i} (Rating: {ex.get('sonnet_rating', 0):.2f}/10.0)")
            print(f"Complexity Score: {ex['complexity']['complexity_score']:.0f}")
            print(f"{'='*80}")
            print(f"\nQuestion: {ex['question']}")
            print(f"\nGold Answer: {ex['gold_answer']}")
            print(f"\nGraph-based Answer: {ex['graph_answer']}")
            print(f"\nText-only Answer: {ex['text_answer']}")
            print(f"\nContext: {ex['num_context_docs']} documents, {ex['context_length']} chars")
            print(f"Graph: {ex['graph_triples']} triples")
            print(f"Text: {ex['text_length']} chars")
            print(f"Context Rot Analysis:")
            print(f"  - Graph compactness ratio: {ex['context_rot_analysis']['compactness_ratio']:.3f}")
            print(f"  - Input length ratio: {ex['context_rot_analysis']['input_length_ratio']:.3f}")


if __name__ == "__main__":
    main()


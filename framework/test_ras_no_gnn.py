#!/usr/bin/env python3
"""
Test and evaluate RAS performance with no-GNN Qwen3-8B model.
Implements iterative planning and answering without graph tokens.
"""

import os
import torch
import json
import argparse
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import sys
import csv

# Add baselines path for metrics and utilities
baselines_path = os.path.join(os.path.dirname(__file__), '..', 'baselines')
framework_path = os.path.dirname(__file__)
if os.path.exists(baselines_path):
    sys.path.insert(0, baselines_path)
if os.path.exists(framework_path):
    sys.path.insert(0, framework_path)

# Import instructions directly (they don't require sentence_transformers)
def get_planner_instruction(model_name):
    if model_name != 'sonnet':
        return """You are a planner to determine if the question can be answered with current information and output the appropriate label as well as the subquery if needed.
Output [NO_RETRIEVAL] if the question can be directly answered with the question itself without any retrieval.
Output [SUBQ] with an subquery for retrieval if still needs a subquery.
Output [SUFFICIENT] if the question can be answered with the provided information.
"""

def get_answerer_instruction(model_name):
    if model_name != 'sonnet':
        return """You are a answerer given a question and retrieved graph information.
Each [SUBQ] is a subquery we generated through reasoning for the question. The retrieved graph information follows each [SUBQ] is relevant graph information we retrieved to answer the subquery.
[NO_RETRIEVAL] means the question can be answered with the question itself without any retrieval.
The main question starts with "Question: ". Please answer the question, with subqueries and retrieved graph information if they are helpful.
"""

# Import text_to_triples_sonnet from sonnet.py
try:
    from sonnet import text_to_triples_sonnet
except ImportError:
    # Fallback if sonnet.py not available
    def text_to_triples_sonnet(text):
        # Simple fallback - just return text as-is (not ideal but works)
        return text

# Import retriever - try framework first, then baselines
DenseRetriever = None
try:
    from td_retriever import DenseRetriever
    print("Loaded DenseRetriever from framework/td_retriever.py")
except ImportError as e:
    print(f"Could not import DenseRetriever from framework: {e}")
    try:
        # Try baselines retriever
        from baselines.baseline_retriever import BaselineRetriever
        # Create a wrapper to match DenseRetriever interface
        class DenseRetriever:
            def __init__(self, knowledge_path, debug=False, **kwargs):
                self.retriever = BaselineRetriever(
                    knowledge_path=knowledge_path,
                    dense_encoder=kwargs.get('dense_encoder', 'facebook/contriever-msmarco'),
                    num_splits=kwargs.get('num_splits', 5),
                    topk=kwargs.get('topk', 5),
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            
            def retrieve(self, query, top_k=5):
                # BaselineRetriever.retrieve expects a list of questions and returns list of dicts
                # Each dict has 'question' and 'contexts' keys
                results = self.retriever.retrieve([query])
                # Convert to expected format: list of (text, score) tuples
                if results and len(results) > 0:
                    contexts = results[0].get('contexts', [])
                    # contexts is a list of dicts with 'text' and 'score'
                    return [(ctx.get('text', ''), ctx.get('score', 0.0)) for ctx in contexts[:top_k]]
                return []
        
        print("Loaded BaselineRetriever from baselines and wrapped it")
    except ImportError as e2:
        print(f"Could not import BaselineRetriever from baselines: {e2}")
        print("⚠️  Retriever will not be available. Some datasets may not work correctly.")
        DenseRetriever = None

# Import metrics from framework/metrics.py (which has normalize_answer)
# Try to import normalize_answer first (it doesn't require nltk)
try:
    # Import normalize_answer directly - it's a standalone function
    import re
    import string
    
    def normalize_answer(s):
        """Normalize answer: remove articles, punctuation, lowercase, fix whitespace"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    # Try to import other metrics from metrics.py
    try:
        from metrics import match, f1_score, accuracy, compute_rouge, mauve_score
        HAS_FRAMEWORK_METRICS = True
    except ImportError:
        # If metrics.py can't be imported (e.g., missing nltk), use fallback implementations
        HAS_FRAMEWORK_METRICS = False
        # Define fallback functions
        def match(prediction, ground_truths):
            for gt in ground_truths:
                if normalize_answer(str(gt)) in normalize_answer(str(prediction)):
                    return 1
            return 0
        
        def f1_score(prediction, ground_truths):
            from collections import Counter
            prediction_tokens = normalize_answer(str(prediction)).split()
            max_f1 = 0
            for gt in ground_truths:
                ground_truth_tokens = normalize_answer(str(gt)).split()
                common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    continue
                precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
                recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                max_f1 = max(max_f1, f1)
            return max_f1
        
        def accuracy(preds, labels):
            match_count = 0
            for pred, label in zip(preds, labels):
                target = label[0] if isinstance(label, list) else label
                if normalize_answer(str(pred)) == normalize_answer(str(target)):
                    match_count += 1
            return 100 * (match_count / len(preds)) if preds else 0
        
        def compute_rouge(data):
            return 0.0  # Placeholder
        
        def mauve_score(predictions, references):
            return 0.0  # Placeholder
except Exception:
    # Fallback: define normalize_answer and basic metrics
    def normalize_answer(s):
        """Normalize answer: remove articles, punctuation, lowercase, fix whitespace"""
        import re
        import string
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def match(prediction, ground_truths):
        for gt in ground_truths:
            if normalize_answer(str(gt)) in normalize_answer(str(prediction)):
                return 1
        return 0
    
    def f1_score(prediction, ground_truths):
        from collections import Counter
        prediction_tokens = normalize_answer(str(prediction)).split()
        max_f1 = 0
        for gt in ground_truths:
            ground_truth_tokens = normalize_answer(str(gt)).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            max_f1 = max(max_f1, f1)
        return max_f1
    
    def accuracy(preds, labels):
        match_count = 0
        for pred, label in zip(preds, labels):
            target = label[0] if isinstance(label, list) else label
            if normalize_answer(str(pred)) == normalize_answer(str(target)):
                match_count += 1
        return 100 * (match_count / len(preds)) if preds else 0
    
    def compute_rouge(data):
        return 0.0  # Placeholder
    
    def mauve_score(predictions, references):
        return 0.0  # Placeholder
    
    HAS_FRAMEWORK_METRICS = False

# Import evaluation functions from baselines
try:
    from baselines.evaluate_results import evaluate_short_answer, evaluate_long_form
except ImportError:
    # Fallback evaluation function using normalize_answer
    def evaluate_short_answer(data, metric='match'):
        scores = []
        total = len(data)
        for idx, item in enumerate(data):
            if (idx + 1) % 100 == 0:
                # Use print instead of logger for fallback function
                print(f"Evaluating {metric}: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")
            output = str(item.get('output', ''))
            golds = item.get('golds', [])
            if not golds:
                continue
            if metric == 'match':
                scores.append(match(output, golds))
            elif metric == 'f1':
                scores.append(f1_score(output, golds))
            elif metric == 'accuracy':
                target = golds[0] if isinstance(golds, list) else golds
                scores.append(1.0 if normalize_answer(output) == normalize_answer(str(target)) else 0.0)
        result = sum(scores) / len(scores) if scores else 0.0
        print(f"Completed {metric} evaluation: {len(scores)}/{total} items, score={result:.4f}")
        return result
    
    def evaluate_long_form(data):
        # Placeholder for long-form evaluation
        return None, None
    
    # Try to import compute_rouge and mauve_score if available
    if not HAS_FRAMEWORK_METRICS:
        try:
            from baselines.metrics import compute_rouge, mauve_score
        except ImportError:
            def compute_rouge(data):
                return 0.0  # Placeholder
            
            def mauve_score(predictions, references):
                return 0.0  # Placeholder


def setup_logging():
    """Setup logging"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'test_ras_no_gnn_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path, base_model_path='Qwen/Qwen3-8B'):
    """Load model with PEFT adapter"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Load PEFT adapter - prefer latest checkpoint
    adapter_path = None
    checkpoint_step = None
    if os.path.isdir(checkpoint_path):
        # First, try to find latest checkpoint in the directory (prioritize this)
        checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith('checkpoint-')]
        if checkpoints:
            # Use the latest checkpoint (highest step number)
            latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            adapter_path = os.path.join(checkpoint_path, latest)
            try:
                checkpoint_step = int(latest.split('-')[1])
            except:
                pass
            logger.info(f"Found latest checkpoint: {adapter_path} (step {checkpoint_step})")
        elif os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json')):
            # Direct adapter path (no checkpoint subdirectory)
            adapter_path = checkpoint_path
            # Try to extract step from path
            if 'checkpoint-' in checkpoint_path:
                try:
                    checkpoint_step = int(checkpoint_path.split('checkpoint-')[1].split('/')[0])
                except:
                    pass
        elif os.path.exists(os.path.join(checkpoint_path, 'final_model')):
            adapter_path = os.path.join(checkpoint_path, 'final_model')
    
    if adapter_path and os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
        logger.info(f"Loading PEFT adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        raise FileNotFoundError(f"Adapter not found in {checkpoint_path}")
    
    model.eval()
    logger.info("Model loaded successfully!")
    
    return model, tokenizer, checkpoint_step


def generate_response(model, tokenizer, input_text, max_new_tokens=300, temperature=0.7, top_p=0.8, top_k=20):
    """Generate response for a single input - no graph tokens"""
    # Format input with Qwen3 chat format
    formatted_input = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(
        formatted_input,
        return_tensors='pt',
        truncation=True,
        max_length=2500
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant\n" in full_text:
        response = full_text.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
    else:
        # Fallback: just return what was generated after the input
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response.strip()


def clean_reasoning_text(text):
    """
    Remove reasoning text and special tokens from model output.
    This improves MATCH scores by removing tokens like <think>, <think>, etc.
    """
    import re
    
    if not text:
        return text
    
    # Remove reasoning tags (case insensitive, handle both <think> and <think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove standalone opening/closing tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?redacted_reasoning>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?reasoning>', '', text, flags=re.IGNORECASE)
    
    # Remove standalone reasoning markers (like "think think")
    text = re.sub(r'\bthink\s+think\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthinking\s*[:.]?\s*', '', text, flags=re.IGNORECASE)
    
    # Remove common reasoning prefixes at the start
    text = re.sub(r'^(let me|i need to|i should|based on|according to|looking at)\s+', '', text, flags=re.IGNORECASE)
    
    # Remove multiple newlines and extra whitespace
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = ' '.join(text.split())
    
    return text.strip()


def ras_no_gnn(model, tokenizer, question, context, retriever, max_iteration=3, max_answer_length=100, debug=False, progress_file=None):
    """
    RAS without GNN: iterative planning and answering using text only (no graph tokens)
    """
    planner_instruction = get_planner_instruction('qwen3-8b')
    answerer_instruction = get_answerer_instruction('qwen3-8b')
    end_iteration_flag = False
    
    # Stage 0: Determine if retrieval is needed
    planner_complete_input = planner_instruction + "\n" + question
    initial_planner_output = generate_response(model, tokenizer, planner_complete_input, max_new_tokens=50)
    
    if progress_file:
        with open(progress_file, 'a') as f:
            f.write(f"  Initial planning: {initial_planner_output[:150]}{'...' if len(initial_planner_output) > 150 else ''}\n")
    
    if debug:
        print(f"Initial planner output: {initial_planner_output}")
    
    if '[NO_RETRIEVAL]'.lower() in initial_planner_output.lower() or 'SUFFICIENT'.lower() in initial_planner_output.lower():
        end_iteration_flag = True
    
    # Set first sub_query to be the question itself
    sub_query = question
    
    retrieved_docs_list = []
    triple_lists = []
    subqueries = []
    inputs = []
    
    iteration = 0
    while not end_iteration_flag and iteration < max_iteration:
        if debug:
            print(f"Iteration {iteration+1} starts ...")
            print(f"Sub query: {sub_query}")
        
        # Stage 1: Retrieval
        if iteration == 0:
            if isinstance(context, list):
                retrieved_docs = context[:5] if len(context) > 5 else context
            else:
                retrieved_docs = [context] if context else []
        else:
            if retriever is None:
                # No retriever available (e.g., for asqa/eli5)
                break
            try:
                retrieved_docs = retriever.retrieve(sub_query, top_k=5)
                # Handle both formats: list of tuples or list of strings
                if retrieved_docs and len(retrieved_docs) > 0:
                    if isinstance(retrieved_docs[0], tuple):
                        retrieved_docs = [item[0] for item in retrieved_docs]
                    elif isinstance(retrieved_docs[0], dict):
                        retrieved_docs = [item.get('text', '') for item in retrieved_docs]
                    # If already list of strings, keep as is
            except Exception as e:
                if debug:
                    print(f"Retrieval error: {e}")
                break
        
        if debug:
            print(f"Retrieved docs: {retrieved_docs[:2]}")
        
        retrieved_docs_list.append(retrieved_docs)
        
        # Stage 2: Text-to-triples (as text, no graph conversion)
        triples = text_to_triples_sonnet("\n".join(retrieved_docs)).replace("\n", " ")
        
        if debug:
            print(f"Triples: {triples[:200]}")
        
        # Store information
        subqueries.append(sub_query)
        triple_lists.append(triples)
        
        # Build planner input (same format as training data)
        planner_input = ""
        for i in range(len(subqueries)):
            planner_input += subqueries[i] + "\n" + "Retrieved Graph Information: " + str(triple_lists[i]) + '\n'
        
        planner_input += "Question: " + question
        inputs.append(planner_input)
        
        if debug:
            print(f"Planner input: {planner_input[:300]}")
        
        # Stage 3: Plan next action
        planner_complete_input = planner_instruction + "\n" + planner_input
        planner_output = generate_response(model, tokenizer, planner_complete_input, max_new_tokens=50)
        
        if progress_file:
            with open(progress_file, 'a') as f:
                f.write(f"  Iter {iteration+1} planning: {planner_output[:150]}{'...' if len(planner_output) > 150 else ''}\n")
        
        if debug:
            print(f"Planner output: {planner_output}")
        
        if 'SUFFICIENT'.lower() in planner_output.lower() or 'NO_RETRIEVAL'.lower() in planner_output.lower():
            end_iteration_flag = True
            if progress_file:
                with open(progress_file, 'a') as f:
                    f.write(f"  → Planning complete (SUFFICIENT/NO_RETRIEVAL)\n")
        else:
            # Extract subquery from planner output (remove [SUBQ] tag if present)
            sub_query = planner_output.replace('[SUBQ]', '').strip()
        
        iteration += 1
    
    # Stage 4: Answering
    if len(inputs) == 0:
        question_text = "Question: " + question
        inputs.append(question_text)
    
    answerer_input = answerer_instruction + "\n" + inputs[-1]
    answerer_output = generate_response(model, tokenizer, answerer_input, max_new_tokens=max_answer_length)
    
    # Clean reasoning text from answer
    answerer_output = clean_reasoning_text(answerer_output)
    
    if debug:
        print(f"Answerer output (cleaned): {answerer_output}")
    
    return answerer_output, triple_lists, subqueries, inputs


def load_test_data(dataset, test_data_path):
    """Load test data for a dataset"""
    questions = []
    contexts = []
    answers = []
    
    if dataset == 'arc_c':
        data_path = os.path.join(test_data_path, dataset + "_test_processed.jsonl")
        try:
            from utils import load_file
        except ImportError:
            import jsonlines
            def load_file(path):
                data = []
                with jsonlines.open(path) as reader:
                    for obj in reader:
                        data.append(obj)
                return data
        data_ = load_file(data_path)
        for item in data_:
            questions.append(item['instruction'])
            contexts.append([d['text'] for d in item['ctxs']][:5])
            answers.append(item['answerKey'])
    elif dataset == 'asqa':
        data_path = os.path.join(test_data_path, dataset + "_test_processed.json")
        with open(data_path, 'r') as f:
            data = json.load(f)['data']
        for item in data:
            questions.append(item['question'])
            contexts.append([d['text'] for d in item['ctxs']][:5])
            answers.append(item['answer'])
    else:
        data_path = os.path.join(test_data_path, dataset + "_test.json")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        questions = data['question']
        contexts = data['context']
        if 'answer' in data:
            answers = data['answer']
        elif 'golds' in data:
            answers = data['golds']
        else:
            answers = [None] * len(questions)
    
    # Limit to 1000 for some datasets
    limit = 500
    if dataset in ['triviaqa', 'popqa', '2wikimultihop']:
        questions = questions[:limit]
        contexts = contexts[:limit]
        answers = answers[:limit]
    
    return questions, contexts, answers


def evaluate_dataset(model, tokenizer, dataset, test_data_path, retriever, output_path, 
                     max_iteration=3, max_answer_length=100, debug=False):
    """Evaluate on a single dataset"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading test data for {dataset}...")
    questions, contexts, answers = load_test_data(dataset, test_data_path)
    
    logger.info(f"Evaluating {len(questions)} questions...")
    
    generated_answers = []
    progress_file = output_path.replace('.json', '_progress.log')
    
    # Write initial progress
    with open(progress_file, 'w') as f:
        f.write(f"Starting evaluation for {dataset}\n")
        f.write(f"Total questions: {len(questions)}\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
    
    for i, question in enumerate(tqdm(questions, desc=f"Processing {dataset}")):
        try:
            context = contexts[i] if i < len(contexts) else []
            
            # Log question and start processing
            with open(progress_file, 'a') as f:
                f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] Question {i+1}/{len(questions)}:\n")
                f.write(f"Q: {question[:200]}{'...' if len(question) > 200 else ''}\n")
            
            answer, triple_lists, subqueries, inputs = ras_no_gnn(
                model, tokenizer, question, context, retriever,
                max_iteration=max_iteration,
                max_answer_length=max_answer_length,
                debug=debug and i < 2,  # Debug first 2 only
                progress_file=progress_file  # Pass progress file for logging
            )
            generated_answers.append(answer)
            
            # Log planning and answer
            with open(progress_file, 'a') as f:
                f.write(f"Planning (subqueries): {len(subqueries)} iterations\n")
                if subqueries:
                    for idx, subq in enumerate(subqueries):
                        f.write(f"  Iter {idx+1}: {subq[:150]}{'...' if len(subq) > 150 else ''}\n")
                f.write(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}\n")
                f.write("-" * 60 + "\n")
            
            # Log progress every 10 questions or at milestones
            if (i + 1) % 10 == 0 or (i + 1) in [1, len(questions) // 4, len(questions) // 2, len(questions) * 3 // 4]:
                progress_pct = 100 * (i + 1) / len(questions)
                with open(progress_file, 'a') as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {i+1}/{len(questions)} ({progress_pct:.1f}%)\n")
                logger.info(f"Progress: {i+1}/{len(questions)} ({progress_pct:.1f}%)")
                
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            generated_answers.append("")
            with open(progress_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Error at question {i}: {str(e)}\n")
                f.write("-" * 60 + "\n")
    
    # Save results
    results = {
        'question': questions,
        'output': generated_answers,
        'answer': answers
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Write completion to progress file
    with open(progress_file, 'a') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total processed: {len(generated_answers)}/{len(questions)}\n")
        f.write(f"Calculating metrics...\n")
    
    logger.info("Calculating metrics...")
    
    # Calculate metrics
    data_for_eval = []
    for i in range(len(questions)):
        item = {
            'output': generated_answers[i],
            'golds': answers[i] if isinstance(answers[i], list) else [answers[i]] if answers[i] else []
        }
        data_for_eval.append(item)
    
    logger.info(f"Prepared {len(data_for_eval)} items for metric calculation")
    
    metrics = {}
    if dataset in ['asqa', 'eli5']:
        # Long-form metrics
        try:
            rouge = compute_rouge(data_for_eval)
            metrics['ROUGE-L'] = rouge
            logger.info(f"ROUGE-L: {rouge:.4f}")
            with open(progress_file, 'a') as f:
                f.write(f"ROUGE-L: {rouge:.4f}\n")
        except Exception as e:
            logger.error(f"ROUGE error: {e}")
            with open(progress_file, 'a') as f:
                f.write(f"ROUGE error: {e}\n")
        
        try:
            references = []
            predictions = []
            for item in data_for_eval:
                ref = item.get('golds', [])
                pred = item.get('output', '')
                if ref:
                    references.append(str(ref[0]).lower() if isinstance(ref, list) else str(ref).lower())
                else:
                    references.append('')
                predictions.append(str(pred).lower())
            
            mauve = mauve_score(predictions, references)
            metrics['MAUVE'] = mauve
            logger.info(f"MAUVE: {mauve:.4f}")
            with open(progress_file, 'a') as f:
                f.write(f"MAUVE: {mauve:.4f}\n")
        except Exception as e:
            logger.error(f"MAUVE error: {e}")
            with open(progress_file, 'a') as f:
                f.write(f"MAUVE error: {e}\n")
    else:
        # Short-form metrics
        logger.info("Calculating MATCH score...")
        match_score = evaluate_short_answer(data_for_eval, 'match')
        metrics['MATCH'] = match_score
        logger.info(f"MATCH: {match_score:.4f}")
        with open(progress_file, 'a') as f:
            f.write(f"MATCH: {match_score:.4f}\n")
        
        if dataset == 'arc_c' or 'arc' in dataset:
            logger.info("Calculating ACCURACY score...")
            acc_score = evaluate_short_answer(data_for_eval, 'accuracy')
            metrics['ACCURACY'] = acc_score
            logger.info(f"ACCURACY: {acc_score:.4f}")
            with open(progress_file, 'a') as f:
                f.write(f"ACCURACY: {acc_score:.4f}\n")
        else:
            logger.info("Calculating F1 score...")
            f1 = evaluate_short_answer(data_for_eval, 'f1')
            metrics['F1'] = f1
            logger.info(f"F1: {f1:.4f}")
            with open(progress_file, 'a') as f:
                f.write(f"F1: {f1:.4f}\n")
    
    # Write final summary to progress file
    with open(progress_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("Final Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("=" * 60 + "\n")
    
    logger.info(f"Progress log saved to: {progress_file}")
    
    return metrics


def update_results_csv(metrics, dataset_name, model_name, mode, csv_path):
    """Update or create results_summary.csv"""
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    new_row = {
        'Dataset': dataset_name,
        'Model': model_name,
        'Mode': mode,
        'ACCURACY': f"{metrics.get('ACCURACY', 0):.4f}" if 'ACCURACY' in metrics else '',
        'F1': f"{metrics.get('F1', 0):.4f}" if 'F1' in metrics else '',
        'MATCH': f"{metrics.get('MATCH', 0):.4f}" if 'MATCH' in metrics else '',
        'MAUVE': f"{metrics.get('MAUVE', 0):.4f}" if 'MAUVE' in metrics else '',
        'ROUGE-L': f"{metrics.get('ROUGE-L', 0):.4f}" if 'ROUGE-L' in metrics else '',
        'Error': ''
    }
    
    # Check if row exists and update, otherwise append
    found = False
    for i, row in enumerate(rows):
        if row['Dataset'] == dataset_name and row['Model'] == model_name and row['Mode'] == mode:
            rows[i] = new_row
            found = True
            break
    
    if not found:
        rows.append(new_row)
    
    fieldnames = ['Dataset', 'Model', 'Mode', 'ACCURACY', 'F1', 'MATCH', 'MAUVE', 'ROUGE-L', 'Error']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logging.getLogger(__name__).info(f"Updated {csv_path} with results")


def main():
    parser = argparse.ArgumentParser(description='Test RAS with no-GNN Qwen3-8B model')
    
    parser.add_argument('--checkpoint_path', type=str,
                       default='/shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen_no_gnn',
                       help='Path to checkpoint directory')
    parser.add_argument('--base_model_path', type=str, default='Qwen/Qwen3-8B',
                       help='Path to base Qwen3-8B model')
    parser.add_argument('--dataset', type=str, nargs='+', default=['triviaqa'],
                       help='Dataset name(s) to evaluate')
    parser.add_argument('--test_data_path', type=str,
                       default='/shared/rsaas/pj20/firas_data/test_datasets',
                       help='Path to test datasets')
    parser.add_argument('--knowledge_path', type=str,
                       default='/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018',
                       help='Path to knowledge base')
    parser.add_argument('--output_dir', type=str,
                       default='/shared/rsaas/pj20/firas_data/test_datasets',
                       help='Directory to save results')
    parser.add_argument('--results_csv', type=str,
                       default='/home/pj20/server-04/FIRAS/baselines/logs/results_summary.csv',
                       help='Path to results_summary.csv')
    parser.add_argument('--max_iteration', type=int, default=3, help='Max iterations for RAS')
    parser.add_argument('--max_answer_length', type=int, default=100, help='Max answer length')
    parser.add_argument('--model_name', type=str, default='qwen3_8b_no_gnn', help='Model name for CSV')
    parser.add_argument('--mode', type=str, default='base', help='Mode for CSV')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    logger.info("Starting RAS evaluation with no-GNN Qwen3-8B")
    logger.info(f"Arguments: {args}")
    
    # Create main progress log file
    main_progress_file = os.path.join(args.output_dir, f"evaluation_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(main_progress_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RAS Evaluation Progress Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Datasets: {args.dataset}\n")
        f.write("=" * 60 + "\n\n")
    
    logger.info(f"Main progress log: {main_progress_file}")
    
    # Load model
    with open(main_progress_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model...\n")
    logger.info("Loading model...")
    model, tokenizer, checkpoint_step = load_model_and_tokenizer(args.checkpoint_path, args.base_model_path)
    if checkpoint_step:
        logger.info(f"Using checkpoint at step {checkpoint_step}")
        with open(main_progress_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Using checkpoint step {checkpoint_step}\n")
    with open(main_progress_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded successfully!\n")
    logger.info("Model loaded successfully!")
    
    # Load retriever (skip for asqa/eli5)
    datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]
    if any(d in ['asqa', 'eli5'] for d in datasets):
        logger.info("ASQA or ELI5 included, skipping retriever...")
        retriever = None
        with open(main_progress_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping retriever (ASQA/ELI5)\n")
    else:
        if DenseRetriever is None:
            logger.error("=" * 60)
            logger.error("DenseRetriever is not available!")
            logger.error("=" * 60)
            logger.error("Missing dependencies: faiss and/or sentence_transformers")
            logger.error("")
            logger.error("Options:")
            logger.error("1. Install dependencies: pip install faiss-cpu sentence-transformers")
            logger.error("2. Use 'handbook' environment which has these dependencies")
            logger.error("3. Test only datasets that don't require retrieval: asqa, eli5")
            logger.error("")
            raise ImportError("DenseRetriever is not available. Install faiss and sentence_transformers, or use 'handbook' environment.")
        
        with open(main_progress_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Loading retriever (this may take a few minutes)...\n")
        logger.info("Loading retriever (this may take a few minutes)...")
        try:
            retriever = DenseRetriever(knowledge_path=args.knowledge_path, debug=args.debug)
            with open(main_progress_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Retriever loaded successfully!\n")
            logger.info("Retriever loaded successfully!")
        except Exception as e:
            with open(main_progress_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Failed to load retriever: {e}\n")
            logger.error(f"Failed to initialize retriever: {e}")
            logger.error("This might be due to missing dependencies (sentence_transformers, faiss, etc.)")
            logger.error("Try: pip install faiss-cpu sentence-transformers")
            logger.error("Or use the 'handbook' conda environment")
            raise
    
    # Evaluate each dataset
    for dataset in datasets:
        with open(main_progress_file, 'a') as f:
            f.write(f"\n[{'='*60}]\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Starting evaluation for dataset: {dataset}\n")
            f.write(f"[{'='*60}]\n")
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating dataset: {dataset}")
        logger.info(f"{'='*60}")
        
        # Include checkpoint step in filename if available
        if checkpoint_step:
            output_path = os.path.join(args.output_dir, f"{dataset}_test_output_{args.model_name}_{args.mode}_step{checkpoint_step}.json")
        else:
            output_path = os.path.join(args.output_dir, f"{dataset}_test_output_{args.model_name}_{args.mode}.json")
        
        metrics = evaluate_dataset(
            model, tokenizer, dataset, args.test_data_path, retriever, output_path,
            max_iteration=args.max_iteration,
            max_answer_length=args.max_answer_length,
            debug=args.debug
        )
        
        # Update CSV - include checkpoint step in model name if available
        model_name_for_csv = args.model_name
        if checkpoint_step:
            model_name_for_csv = f"{args.model_name}_step{checkpoint_step}"
        update_results_csv(metrics, dataset, model_name_for_csv, args.mode, args.results_csv)
        
        with open(main_progress_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {dataset}\n")
            f.write(f"  Metrics: {metrics}\n")
        logger.info(f"Completed evaluation for {dataset}")
        logger.info(f"Metrics: {metrics}")
    
    with open(main_progress_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] All evaluations completed!\n")
        f.write("=" * 60 + "\n")
    logger.info("All evaluations completed!")
    logger.info(f"Main progress log: {main_progress_file}")


if __name__ == '__main__':
    main()


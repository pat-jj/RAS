#!/usr/bin/env python3
"""
KnowTrace baseline wrapper for the FIRAS framework.
Adapts KnowTrace to work with the baseline framework's data format.
Supports both Sonnet (via API) and LLaMA (via local model).
"""

import sys
import os
import json
import argparse
import time
from typing import List, Dict, Any
from tqdm import tqdm

# Add KnowTrace to path
knowtrace_path = os.path.join(os.path.dirname(__file__), '..', 'KnowTrace')
sys.path.insert(0, knowtrace_path)

# Import KnowTrace components
from agent import KnowTrace, LocalLLM

# Import baseline retriever
from baseline_retriever import BaselineRetriever

# Import baselines utils for file I/O - use simple versions to avoid heavy dependencies
import jsonlines

def load_file(input_fp):
    """Simple file loader to avoid heavy dependencies"""
    if input_fp.endswith(".json"):
        with open(input_fp, 'r') as f:
            return json.load(f)
    else:
        with jsonlines.open(input_fp, 'r') as jsonl_f:
            return [obj for obj in jsonl_f]

def save_file_jsonl(data, fp):
    """Simple JSONL saver to avoid heavy dependencies"""
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

# Import claude_api for Sonnet
baselines_path = os.path.dirname(__file__)
sys.path.insert(0, baselines_path)
from claude_api import get_claude_response

# Monkey patch KnowTrace to support Sonnet API
class SonnetLLM:
    """Wrapper to use Sonnet API instead of local LLM server"""
    def __init__(self, model_name="sonnet", max_new_tokens=128):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.config = {'role': 'base', 'max_new_tokens': max_new_tokens}
    
    def run(self, prompt):
        """Call Sonnet API"""
        try:
            response = get_claude_response(
                llm=self.model_name,
                prompt=prompt,
                max_tokens=self.max_new_tokens
            )
            return response
        except Exception as e:
            print(f"Error calling Sonnet API: {e}")
            return ""


def create_knowtrace_agent(question: str, answer: str, base_llm: str, step_num: int = 5, 
                           use_sonnet: bool = False, sonnet_model: str = "sonnet",
                           local_llm_port: int = 1051, retriever: BaselineRetriever = None):
    """Create a KnowTrace agent with appropriate LLM backend"""
    if use_sonnet:
        # For Sonnet, create agent and patch methods to use API
        # Use a dummy base_llm that will trigger API path
        import agent as agent_module
        
        # Patch the llm function BEFORE creating the agent
        original_llm = agent_module.llm
        
        def sonnet_llm_wrapper(prompt, model="gpt-3.5-turbo-instruct", stop=["\n"]):
            # Always use Sonnet API for gpt-3.5-turbo-instruct
            # Match original llm() function: max_tokens=350, temperature=0
            try:
                response = get_claude_response(
                    llm=sonnet_model,
                    prompt=prompt,
                    max_tokens=350,
                    temperature=0
                )
                # Don't apply stop strings - let the full response come through
                # The original OpenAI API with stop=["\n"] would stop at newline, but that might cut off answers
                # Claude API doesn't support stop strings the same way, so we get the full response
                return response
            except Exception as e:
                print(f"Error calling Sonnet API: {e}")
                import traceback
                traceback.print_exc()
                return ""
        
        # Patch the llm function in agent module
        agent_module.llm = sonnet_llm_wrapper
        
        # Now create the agent (it will use the patched llm function)
        agent = KnowTrace(question, answer, "gpt-3.5-turbo-instruct", step_num, collect_data=False)
        
        # Also replace local LLM instances with Sonnet wrapper
        sonnet_llm = SonnetLLM(model_name=sonnet_model, max_new_tokens=128)
        agent.init_llama = sonnet_llm
        agent.reason_llama = sonnet_llm
        agent.refine_llama = sonnet_llm
        agent.direct_llama = sonnet_llm
        
        # Replace retriever with baseline retriever
        if retriever is not None:
            agent.bm25_retriever = retriever
        
        return agent
    else:
        # For LLaMA, use local server (assumes server is running on specified port)
        # Update the port if needed
        agent = KnowTrace(question, answer, base_llm, step_num, collect_data=False)
        # Update ports for local LLM instances
        config = {'role': 'reasoner', 'max_new_tokens': 128, 'do_sample': True, 'temperature': 0.01, 'top_p': 0.9, 'stop_strings': ["\n\n"]}
        agent.reason_llama = LocalLLM(local_llm_port, config)
        config = {'role': 'refiner', 'max_new_tokens': 128, 'do_sample': True, 'temperature': 0.01, 'top_p': 0.9, 'stop_strings': ["\n\n"]}
        agent.refine_llama = LocalLLM(local_llm_port, config)
        config = {'role': 'base', 'max_new_tokens': 128, 'do_sample': True, 'temperature': 0.01, 'top_p': 0.9, 'stop_strings': ["\n\n"]}
        agent.init_llama = LocalLLM(local_llm_port, config)
        agent.direct_llama = LocalLLM(local_llm_port, config)
        
        # Replace retriever with baseline retriever
        if retriever is not None:
            agent.bm25_retriever = retriever
        
        return agent


def run_knowtrace_on_dataset(input_data: List[Dict], base_llm: str, step_num: int = 5,
                            use_sonnet: bool = False, sonnet_model: str = "sonnet",
                            max_items: int = None, local_llm_port: int = 1051,
                            retriever: BaselineRetriever = None):
    """Run KnowTrace on a dataset"""
    if max_items:
        input_data = input_data[:max_items]
    
    results = []
    import sys
    for idx, item in enumerate(tqdm(input_data, desc="Running KnowTrace")):
        if idx == 0:
            print(f"Processing first item...", flush=True)
            sys.stdout.flush()
        # Extract question and answer
        question = item.get("question", item.get("instruction", ""))
        
        # Get answer from various possible fields
        answer = None
        if "answer" in item:
            answer = item["answer"]
        elif "answers" in item:
            answers = item["answers"]
            answer = answers[0] if isinstance(answers, list) and answers else (answers if isinstance(answers, str) else "")
        elif "golds" in item:
            golds = item["golds"]
            answer = golds[0] if isinstance(golds, list) and golds else (golds if isinstance(golds, str) else "")
        
        if not question:
            print(f"Skipping item without question: {item}")
            continue
        
        # Ensure answer is a string
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not answer or answer == "":
            answer = "dummy"  # KnowTrace needs an answer for evaluation
        
        # Create and run agent
        try:
            import time
            import sys
            agent = create_knowtrace_agent(question, answer, base_llm, step_num, use_sonnet, sonnet_model, local_llm_port, retriever)
            agent.run()
            
            # Extract result
            result_item = item.copy()
            # Handle case where agent.answer might be "None" string or empty
            # Also check if it's the literal string "None" (case-insensitive)
            output = agent.answer if agent.answer and str(agent.answer).strip().lower() != "none" else ""
            result_item["output"] = output
            result_item["question"] = question
            if "golds" not in result_item:
                result_item["golds"] = [answer] if answer and answer != "dummy" else (item.get("golds", []))
            
            results.append(result_item)
        except Exception as e:
            import traceback
            print(f"Error processing question '{question}': {e}")
            print(traceback.format_exc())
            result_item = item.copy()
            result_item["output"] = ""
            result_item["question"] = question
            if "golds" not in result_item:
                result_item["golds"] = [answer] if answer and answer != "dummy" else (item.get("golds", []))
            results.append(result_item)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run KnowTrace baseline")
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input dataset file')
    parser.add_argument('--result_fp', type=str, required=True,
                       help='Path to save results')
    parser.add_argument('--base_llm', type=str, default="LLaMA3-8B-Instruct",
                       help='Base LLM to use (LLaMA3-8B-Instruct, Llama-2-7b-chat-hf, sonnet, sonnet3.5, etc.)')
    parser.add_argument('--step_num', type=int, default=5,
                       help='Number of inference steps')
    parser.add_argument('--sonnet_model', type=str, default="sonnet",
                       choices=["sonnet", "sonnet4.5"],
                       help='Sonnet model variant (if using Sonnet)')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to process (for testing, default: None for all items)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (for output formatting)')
    parser.add_argument('--local_llm_port', type=int, default=1051,
                       help='Port for local LLM server (if using LLaMA)')
    parser.add_argument('--knowledge_path', type=str,
                       default='/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018',
                       help='Path to knowledge base for retrieval')
    parser.add_argument('--dense_encoder', type=str, default='facebook/contriever-msmarco',
                       help='Dense encoder model for retrieval')
    parser.add_argument('--num_splits', type=int, default=5,
                       help='Number of index splits')
    
    args = parser.parse_args()
    
    # Initialize baseline retriever
    print("Initializing baseline retriever...")
    retriever = BaselineRetriever(
        knowledge_path=args.knowledge_path,
        dense_encoder=args.dense_encoder,
        num_splits=args.num_splits,
        topk=5,
        device='cpu'
    )
    print("Retriever initialized.")
    
    # Determine if using Sonnet
    use_sonnet = args.base_llm.lower() in ["sonnet", "sonnet3.5", "sonnet4.5"]
    
    # Load input data
    print(f"Loading data from: {args.input_file}")
    input_data = load_file(args.input_file)
    print(f"Loaded {len(input_data)} items")
    
    # Handle different data formats
    if isinstance(input_data, dict):
        if "data" in input_data:
            input_data = input_data["data"]
        elif "question" in input_data and isinstance(input_data["question"], list):
            # Convert dict with lists to list of dicts
            input_data_list = []
            for i in range(len(input_data["question"])):
                item = {}
                for key in input_data:
                    if isinstance(input_data[key], list) and i < len(input_data[key]):
                        item[key] = input_data[key][i]
                    else:
                        item[key] = input_data[key]
                input_data_list.append(item)
            input_data = input_data_list
    
    # Run KnowTrace
    print(f"Running KnowTrace with {args.base_llm} (steps={args.step_num})...")
    results = run_knowtrace_on_dataset(
        input_data,
        args.base_llm,
        args.step_num,
        use_sonnet,
        args.sonnet_model,
        args.max_items,
        args.local_llm_port,
        retriever
    )
    
    # Save results
    print(f"Saving results to: {args.result_fp}")
    if args.dataset in ["asqa", "eli5"]:
        # Long-form tasks use JSON format
        output = {"data": results}
        with open(args.result_fp, "w") as f:
            json.dump(output, f, indent=4)
    else:
        # Short-form tasks use JSONL format
        save_file_jsonl(results, args.result_fp)
    
    print(f"Completed! Processed {len(results)} items")


if __name__ == "__main__":
    main()


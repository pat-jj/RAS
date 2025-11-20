import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_file, TASK_INST, PROMPT_DICT, save_file_jsonl, process_arc_instruction, postprocess_answers_closed
from metrics import metric_max_over_ground_truths, exact_match_score, match, f1_score
import ast
import json
import gc
import os
from typing import List

def load_model_and_tokenizer(model_name: str, device: str = "cuda", torch_dtype=torch.bfloat16):
    """Load model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # For decoder-only models, use left padding for correct generation
    tokenizer.padding_side = "left"
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, tokenizer

def format_prompt_for_model(prompt: str, tokenizer, enable_thinking: bool = False) -> str:
    """Format prompt using tokenizer's chat template if available, otherwise return as-is
    
    Args:
        prompt: Input prompt text
        tokenizer: Tokenizer instance
        enable_thinking: For Qwen3 models, enable thinking mode (default: False)
    """
    # Try to use the tokenizer's chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        try:
            # For single-turn conversations
            messages = [{"role": "user", "content": prompt}]
            
            # For Qwen3 models, support thinking mode
            if "qwen" in tokenizer.name_or_path.lower() and "3" in tokenizer.name_or_path:
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            else:
                # For LLaMA3 and other models
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}, using raw prompt")
            return prompt
    else:
        # Fallback: use model-specific templates
        if "qwen" in tokenizer.name_or_path.lower():
            return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in tokenizer.name_or_path.lower():
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            return prompt

def generate_batch(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    enable_thinking: bool = False
):
    """Generate responses for a batch of prompts
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 for deterministic)
        enable_thinking: For Qwen3 models, enable thinking mode
    """
    # Format prompts according to model
    formatted_prompts = [format_prompt_for_model(p, tokenizer, enable_thinking) for p in prompts]
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    # Generate with appropriate parameters
    # For deterministic generation (temperature=0), use greedy decoding
    # For sampling (temperature>0), use sampling
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if temperature > 0:
        generation_kwargs.update({
            "temperature": temperature,
            "do_sample": True,
        })
    else:
        # For deterministic generation, only set do_sample=False
        # Don't include temperature, top_p, or top_k to avoid warnings
        generation_kwargs["do_sample"] = False
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )
    
    # Decode - extract only the newly generated tokens
    input_lengths = inputs.input_ids.shape[1]
    generated_texts = tokenizer.batch_decode(
        outputs[:, input_lengths:], 
        skip_special_tokens=True
    )
    
    # For Qwen3 with thinking mode, we might need to parse thinking content
    # For now, we return the full generated text
    # Users can parse thinking content if needed using the </think> token
    
    return generated_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--retrieval_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="vanilla", 
                       choices=["vanilla", "retrieval", "asqa_base", "eli5_base", "2wikimultihop"])
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--top_n', type=int, default=1)
    parser.add_argument('--result_fp', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--prompt_name', type=str, default="prompt_no_input")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path (e.g., Qwen/Qwen3-8B or meta-llama/Meta-Llama-3-8B-Instruct)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--enable_thinking', action='store_true',
                       help='For Qwen3 models, enable thinking mode (generates reasoning before answer)')
    parser.add_argument("--choices", type=str, default=None,
                       help="space-separated answer candidates")
    parser.add_argument("--instruction", type=str, default=None,
                       help="task instructions")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    
    # Load data
    input_data = load_file(args.input_file)

    # Handle different data formats
    if "asqa" in args.task:
        for id, item in enumerate(input_data):
            item["ctxs"] = item.get("docs", item.get("ctxs", []))
    
    input_data_ = []
    if "2wikimultihop" in args.input_file:
        for i in range(len(input_data['question'])):
            context_list = []
            for j in range(len(input_data['context'][i])):
                context_list.append(input_data['context'][i][j])
            input_data_.append({
                "question": input_data['question'][i], 
                "ctxs": context_list, 
                "answers": input_data['answer'][i]
            })
        input_data = input_data_
        
    if "eli5" in args.input_file:
        for i in range(len(input_data['question'])):
            input_data_.append({
                "question": input_data['question'][i], 
                "ctxs": input_data['context'][i], 
                "answers": input_data['answer'][i]
            })
        input_data = input_data_
    
    # Load retrieval results if provided
    if args.mode == "retrieval":
        if args.retrieval_file is not None:
            retrieval_data = load_file(args.retrieval_file)
            id2retrieval = {}
            for id, item in enumerate(retrieval_data):
                if "id" not in item:
                    id2retrieval[id] = item["ctxs"][:args.top_n]
                else:
                    id2retrieval[item["id"]] = item["ctxs"][:args.top_n]
            for id, item in enumerate(input_data):
                retrieval_result = id2retrieval[id if "id" not in item else item["id"]]
                # Handle different retrieval formats
                if isinstance(retrieval_result[0], list):
                    # Format: [["text", score], ...]
                    evidences = ["[{}] ".format(i+1) + ctx[0] for i, ctx in enumerate(retrieval_result)]
                elif isinstance(retrieval_result[0], dict):
                    # Format: [{"title": "...", "text": "..."}, ...]
                    evidences = ["[{}] ".format(i+1) + ctx.get("title", "") + "\n" + ctx.get("text", "") 
                                for i, ctx in enumerate(retrieval_result)]
                else:
                    # Format: ["text", ...]
                    evidences = ["[{}] ".format(i+1) + ctx for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        elif "2wikimultihop" not in args.input_file:
            print("Using top {} documents".format(args.top_n))
            for id, item in enumerate(input_data):
                retrieval_result = item["ctxs"][:args.top_n]
                if isinstance(retrieval_result[0], dict):
                    evidences = ["[{}] ".format(i+1) + ctx.get("title", "") + "\n" + ctx.get("text", "") 
                               for i, ctx in enumerate(retrieval_result)]
                else:
                    evidences = ["[{}] ".format(i+1) + str(ctx) for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        else:
            for id, item in enumerate(input_data):
                retrieval_result = item["ctxs"][:args.top_n] if len(item["ctxs"]) > args.top_n else item["ctxs"]
                evidences = ["[{}] ".format(i+1) + " ".join(ctx) if isinstance(ctx, list) else str(ctx) 
                           for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
                
    if "asqa" in args.mode:
        for id, item in enumerate(input_data):
            retrieval_result = item["ctxs"][:args.top_n]
            if isinstance(retrieval_result[0], dict):
                evidences = ["Document [{}]".format(i+1) + "(Title: {}): {}".format(
                    ctx.get("title", ""), ctx.get("text", "")) for i, ctx in enumerate(retrieval_result)]
            else:
                evidences = ["Document [{}]: {}".format(i+1, ctx) for i, ctx in enumerate(retrieval_result)]
            item["paragraph"] = "\n".join(evidences)
            
    if "eli5" in args.mode:
        for id, item in enumerate(input_data):
            evidences = ["Document [{}]: {}".format(i+1, ctx) for i, ctx in enumerate(item["ctxs"])]
            item["paragraph"] = "\n".join(evidences)
    
    # Prepare ground truth labels
    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            elif "answers" in item:
                item["golds"] = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
            elif "answer" in item:
                item["golds"] = [item["answer"]]
            elif "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            elif "answerKey" in item:
                item["golds"] = [item["answerKey"]]

        if args.task == "factscore":
            item["instruction"] = item["input"]
        else:
            if "instruction" not in item and "question" in item:
                item["instruction"] = item["question"]

        # Add task instructions
        if args.instruction is not None:
            item["instruction"] = args.instruction + "\n\n### Input:\n" + item["instruction"]
        if args.task == "fever":
            item["instruction"] = TASK_INST[args.task] + "\n\n### Input:\n" + item["instruction"]
        if args.task == "arc_c":
            item["instruction"] = process_arc_instruction(item, TASK_INST[args.task])
        if "asqa" in args.task:
            item["instruction"] = TASK_INST[args.task] + item["question"]
        if "eli5" in args.input_file:
            item["instruction"] = TASK_INST[args.task] + item["question"]
        if "2wikimultihop" in args.input_file:
            item["instruction"] = TASK_INST[args.task] + "\n" + item["question"]
    
    # Check for existing results and resume if possible
    start_idx = 0
    final_results = []
    if os.path.exists(args.result_fp):
        try:
            existing_data = load_file(args.result_fp)
            if existing_data:
                start_idx = len(existing_data)
                final_results = existing_data
                print(f"Resuming from existing results: {start_idx}/{len(input_data)} items already processed")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting from scratch.")
            start_idx = 0
            final_results = []
    
    # Process in batches
    batch_size = args.batch_size
    remaining_data = input_data[start_idx:]
    
    with tqdm(total=len(input_data), initial=start_idx) as pbar:
        for idx in range(0, len(remaining_data), batch_size):
            batch = remaining_data[idx:min(idx + batch_size, len(remaining_data))]
            
            # Format prompts
            processed_batch = [
                PROMPT_DICT[args.prompt_name].format_map(item) for item in batch
            ]
            
            # Generate responses
            try:
                preds = generate_batch(
                    model, 
                    tokenizer, 
                    processed_batch, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking
                )
            except Exception as e:
                print(f"Error generating batch: {e}")
                preds = [""] * len(batch)
            
            # Post-process and store results
            for j, (item, pred) in enumerate(zip(batch, preds)):
                item["output"] = postprocess_answers_closed(
                    pred, args.task, args.choices)
                final_results.append(item)
            
            pbar.update(len(batch))
            
            # Save results incrementally to allow resuming
            if (idx // batch_size + 1) % 10 == 0:  # Save every 10 batches
                save_file_jsonl(final_results, args.result_fp)
                print(f"Checkpoint saved: {len(final_results)}/{len(input_data)} items")
            
            # Clear cache periodically
            if idx % (batch_size * 10) == 0:
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    
    # Calculate metrics
    if "asqa" not in args.task and "eli5" not in args.task:
        for item in final_results:  # Use final_results instead of input_data
            if "output" not in item or not item.get("output"):
                continue  # Skip items without output
            if args.metric == "em":
                metric_result = metric_max_over_ground_truths(
                    exact_match_score, item["output"], item["golds"])
            elif args.metric == "accuracy":
                metric_result = 1.0 if item["golds"][0].lower() in item["output"].lower() else 0.0
            elif args.metric == "match":
                metric_result = match(item["output"], item["golds"])
            elif args.task == "factscore":
                metric_result = 0.0
            elif args.metric == "f1":
                metric_result = f1_score(item["output"], item["golds"])
            else:
                raise NotImplementedError(f"Metric {args.metric} not implemented")
            item["metric_result"] = metric_result

        metric_results = [item["metric_result"] for item in final_results if "metric_result" in item]
        if metric_results:
            print("Overall result: {0:.4f}".format(np.mean(metric_results)))
        else:
            print("Warning: No metric results calculated (all outputs may be empty)")

    # Save results
    if args.task == "factscore":
        processed_item = []
        for item in final_results:
            processed_item.append(item)
        save_file_jsonl(processed_item, args.result_fp)
    elif "asqa" in args.task or "eli5" in args.task:
        processed_item = []
        for item in final_results:
            processed_item.append(item)
        out = {"data": processed_item}
        with open(args.result_fp, "w") as f:
            json.dump(out, f, indent=4)
    else:
        save_file_jsonl(final_results, args.result_fp)  # Use final_results instead of input_data
    
    print(f"Results saved to {args.result_fp}")

if __name__ == "__main__":
    main()


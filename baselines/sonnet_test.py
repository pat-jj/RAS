import argparse
import numpy as np
from tqdm import tqdm
from utils import load_file, TASK_INST, PROMPT_DICT, save_file_jsonl, process_arc_instruction, postprocess_answers_closed
from metrics import metric_max_over_ground_truths, exact_match_score, match, f1_score
import ast
from claude_api import get_claude_response
import queue
import time
from typing import List
from threading import Lock
import threading
import random
import json

def call_model_batch(prompts: List[str], model_name: str = "sonnet", num_threads: int = 8, max_new_tokens: int = 15) -> List[str]:
    """Process prompts using optimized queue-based multithreading"""
    prompt_queue = queue.Queue()
    results = [None] * len(prompts)
    active_threads = []
    error_count = 0
    
    def worker():
        nonlocal error_count
        while True:
            try:
                idx, prompt = prompt_queue.get_nowait()
                retry_count = 0
                while retry_count < 3:  # Max 3 retries
                    try:
                        response = get_claude_response(llm=model_name, prompt=prompt, max_tokens=max_new_tokens)
                        print(prompt)
                        print(response)
                        results[idx] = response  # No lock needed as each thread writes to different index
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == 3:
                            results[idx] = ""
                            error_count += 1
                        else:
                            time.sleep(0.1 * (2 ** retry_count))  # Exponential backoff
                prompt_queue.task_done()
            except queue.Empty:
                return

    # Fill queue with all tasks at once
    for idx, prompt in enumerate(prompts):
        prompt_queue.put((idx, prompt))
    
    # Start all worker threads
    for _ in range(min(num_threads, len(prompts))):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        active_threads.append(t)
    
    # Wait for completion
    prompt_queue.join()
    
    # Clean up
    for t in active_threads:
        t.join()
        
    if error_count > 0:
        print(f"Warning: {error_count} requests failed after retries")
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--retrieval_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="vanilla")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--top_n', type=int, default=1)
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--prompt_name', type=str, default="prompt_no_input")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--choices",  type=str, default=None,
                        help="space-separated answer candidates")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions")
    args = parser.parse_args()

    input_data = load_file(args.input_file)
    # input_data = random.sample(input_data, 1000)
    # print(f"Using {len(input_data)} random samples")

    # For baseline scripts, we simply load pre-retrieved documents from `retrieval_file` option.
    if "asqa" in args.task:
        for id, item in enumerate(input_data):
            item["ctxs"] = item["docs"]
    
    input_data_ = []
    if "2wikimultihop" in args.input_file:
        for i in range(len(input_data['question'])):
            context_list = []
            for j in range(len(input_data['context'][i])):
                context_list.append(input_data['context'][i][j])
            input_data_.append({"question": input_data['question'][i], "ctxs": context_list, "answers": input_data['answer'][i]})
        # input_data = input_data_[:1000]
        
    if "eli5" in args.input_file:
        for i in range(len(input_data['question'])):
            input_data_.append({"question": input_data['question'][i], "ctxs": input_data['context'][i], "answers": input_data['answer'][i]})
        
        input_data = input_data_
    
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
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        elif "2wikimultihop" not in args.input_file:
            print("Using top {} documents".format(args.top_n))
            for id, item in enumerate(input_data):
                retrieval_result = item["ctxs"][:args.top_n]
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        else:
            for id, item in enumerate(input_data):
                retrieval_result = item["ctxs"][:args.top_n] if len(item["ctxs"]) > args.top_n else item["ctxs"]
                evidences = ["[{}] ".format(
                    i+1) + " ".join(ctx) for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
                
    if "asqa" in args.mode:
        for id, item in enumerate(input_data):
            retrieval_result = item["ctxs"][:args.top_n]
            evidences = ["Document [{}]".format(
                i+1) + "(Title: {}): {}".format(ctx["title"], ctx["text"]) for i, ctx in enumerate(retrieval_result)]
            item["paragraph"] = "\n".join(evidences)
            
    if "eli5" in args.mode:
        for id, item in enumerate(input_data):
            evidences = ["Document [{}]: {}".format(
                i+1, ctx) for i, ctx in enumerate(item["ctxs"])]
            item["paragraph"] = "\n".join(evidences)
            
    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "answer" in item:
                item["golds"] = [item["answer"]]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]

        if args.task == "factscore":
            item["instruction"] = item["input"]
        else:
            if "instruction" not in item and "question" in item:
                item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
        if args.task == "fever":
            item["instruction"] = TASK_INST[args.task] + \
                "\n\n### Input:\n" + item["instruction"]
        if args.task == "arc_c":
            item["instruction"] = process_arc_instruction(item, TASK_INST[args.task])
        if "asqa" in args.task:
            item["instruction"] = TASK_INST[args.task] + item["question"]
        if "eli5" in args.input_file:
            item["instruction"] = TASK_INST[args.task] + item["question"]
        if "2wikimultihop" in args.input_file:
            item["instruction"] = TASK_INST[args.task] + "\n" + item["question"]
                
    # Process all items in larger batches for better throughput
    batch_size = max(args.batch_size, 32)  # Use larger batches
    final_results = []
    
    with tqdm(total=len(input_data)) as pbar:
        for idx in range(0, len(input_data), batch_size):
            batch = input_data[idx:min(idx + batch_size, len(input_data))]
            processed_batch = [
                PROMPT_DICT[args.prompt_name].format_map(item) for item in batch
            ]
            
            # Process batch with optimized threading
            preds = call_model_batch(processed_batch, num_threads=8, max_new_tokens=args.max_new_tokens)
            
            # Update results
            for j, (item, pred) in enumerate(zip(batch, preds)):
                item["output"] = postprocess_answers_closed(
                    pred, args.task, args.choices)
                final_results.append(item)
            
            pbar.update(len(batch))
    
    # Calculate metrics
    if "asqa" not in args.task and "eli5" not in args.task:
        for item in input_data:
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
                metric_result = f1_score(item["output"], [item["golds"]])
                
            else:
                raise NotImplementedError
            item["metric_result"] = metric_result

        print("overall result: {0}".format(
            np.mean([item["metric_result"] for item in input_data])))

    # Handle special case for factscore task
    if args.task == "factscore":
        processed_item = []
        for item in input_data:
            processed_item.append(item)
        save_file_jsonl(processed_item, args.result_fp)
        
    elif "asqa" in args.task or "eli5" in args.task:
        processed_item = []
        for item in input_data:
            processed_item.append(item)
        out = {"data": processed_item}
        with open(args.result_fp[:-1], "w") as f:
            json.dump(out, f, indent=4)
    else:
        save_file_jsonl(input_data, args.result_fp)

if __name__ == "__main__":
    main()
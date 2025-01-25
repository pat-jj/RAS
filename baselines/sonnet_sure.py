import argparse
import numpy as np
from tqdm import tqdm
from utils import load_file, save_file_jsonl, process_arc_instruction, postprocess_answers_closed, ALCE_2_SHOT_INST, ALCE_2_SHOT_INST_BASE, ELI5_2_SHOT_INST, ELI5_2_SHOT_INST_BASE
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


PROMPT_DICT = {
    "candidate_generation": (
        "Given the following question, generate 2-3 possible answer candidates. Format your response as a JSON list of strings.\n\n"
        "Question: {instruction}\n\n"
        "Answer Candidates:"
    ),
    "conditional_summarization": (
        "Analyze the following passages to evaluate if '{candidate}' is the correct answer to the question: {instruction}\n\n"
        "Passages:\n{paragraph}\n\n"
        "Provide a concise summary explaining whether the passages support or refute this candidate being the correct answer:"
    ),
    "verification": (
        "Given the following summaries for different answer candidates to the question '{instruction}', "
        "determine which candidate is most likely correct. Analyze:\n\n"
        "{summaries}\n\n"
        "For each summary, provide:\n"
        "1. Validity (True/False)\n"
        "2. Relative informativeness score (1-10)\n\n"
        "Then select the final answer. Format response as JSON with fields: validities, scores, final_answer"
    )
}



TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false in lowercase. Do not say anything else, and only output true or false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice. Only output the letter of the answer choice, e.g. A, B, C, or D. Do not say anything else, and only output A, B, C, or D.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
             "asqa_ret": ALCE_2_SHOT_INST,
             "asqa_base": ALCE_2_SHOT_INST_BASE,
             "eli5_ret": ELI5_2_SHOT_INST,
             "eli5_base": ELI5_2_SHOT_INST_BASE,
             "2wikimultihop": "Answer the following question. Just output the answer (even if you are not sure), do not say anything else."
             }


class SuREFramework:
    def __init__(self, model_caller):
        self.model_caller = model_caller

    def generate_candidates(self, question: str) -> List[str]:
        """Step 1: Generate answer candidates"""
        prompt = PROMPT_DICT["candidate_generation"].format(instruction=question)
        response = self.model_caller([prompt])[0]
        try:
            candidates = json.loads(response)
            return candidates
        except:
            # Fallback if response isn't proper JSON
            candidates = [x.strip() for x in response.split(",")]
            return candidates[:3]

    def conditional_summarization(self, question: str, candidate: str, passages: str) -> str:
        """Step 2: Generate supporting/refuting summary for each candidate"""
        prompt = PROMPT_DICT["conditional_summarization"].format(
            candidate=candidate,
            instruction=question,
            paragraph=passages
        )
        return self.model_caller([prompt])[0]

    def verify_and_select(self, question: str, candidates: List[str], summaries: List[str]):
        """Step 3: Verify summaries and select final answer"""
        summaries_text = "\n\n".join([f"Candidate '{c}':\n{s}" 
                                    for c, s in zip(candidates, summaries)])
        prompt = PROMPT_DICT["verification"].format(
            instruction=question,
            summaries=summaries_text
        )
        response = self.model_caller([prompt])[0]
        try:
            result = json.loads(response)
            return result
        except:
            # Fallback if response isn't proper JSON
            return {"final_answer": candidates[0]}  # Simple fallback


def call_model_batch(prompts: List[str], model_name: str = "sonnet", num_threads: int = 10, max_new_tokens: int = 100) -> List[str]:
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


    sure_framework = SuREFramework(
        model_caller=lambda prompts: call_model_batch(prompts, "sonnet", 8, args.max_new_tokens)
    )
    input_data = load_file(args.input_file)
    # input_data = random.sample(input_data, 200)
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
        input_data = input_data_[:200]
        
    if "eli5" in args.input_file:
        for i in range(len(input_data['question'])):
            input_data_.append({"question": input_data['question'][i], "ctxs": input_data['context'][i], "answers": input_data['answer'][i]})
        
        input_data = input_data_
    
    
    input_data = random.sample(input_data, 200)
    
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
    results = []
    for item in tqdm(input_data):
        # Step 1: Generate candidates
        candidates = sure_framework.generate_candidates(item["instruction"])
        
        # Step 2: Conditional summarization for each candidate
        summaries = [
            sure_framework.conditional_summarization(
                item["instruction"], 
                candidate, 
                item.get("paragraph", "")
            )
            for candidate in candidates
        ]
        
        # Step 3: Verify and select final answer
        verification_result = sure_framework.verify_and_select(
            item["instruction"],
            candidates,
            summaries
        )
        
        # Store results
        item.update({
            "candidates": candidates,
            "summaries": summaries,
            "verification": verification_result,
            "output": verification_result["final_answer"]
        })
        results.append(item)
        
    
    save_file_jsonl(results, args.result_fp)
            
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
                try:
                    metric_result = f1_score(item["output"], [item["golds"]])
                except:
                    print("Error in f1 score: ", item)
                    metric_result = 0.0
                
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
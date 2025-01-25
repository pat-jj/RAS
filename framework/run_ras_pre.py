import argparse
import json
import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from models.graphllm_ans_v2 import GraphLLM as Answerer
from models.graphllm_pla_v2 import GraphLLM as Planner
from utils import GraphProcessor, get_planner_instruction, get_answerer_instruction, text_to_triples, TASK_INST, clean_document, load_file, ras_asqa_sonnet, ras_eli5_sonnet, convert_triple_str_to_graph, ALCE_2_SHOT_TRIPLES_INST
from tqdm import tqdm
from td_retriever import ThemeScopedRetriever
from sonnet import planner_sonnet, answerer_sonnet, text_to_triples_sonnet
from safetensors.torch import load_model
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import time
from functools import partial


from sonnet import *
def ras_asqa_sonnet(args, models, question, context, graph_processor, retriever):
    answerer_instruction = ALCE_2_SHOT_TRIPLES_INST
    triples = text_to_triples_sonnet("\n".join(context)).replace("\n", " ")
    # print("Triples: ", triples)

    answerer_input = answerer_instruction + question + "\nRetrieved Graph Information:" + triples + "\nAnswer:"
    # generated_answer = get_claude_response(llm="sonnet", prompt=answerer_input, max_tokens=args.max_answer_length)
    generated_answer = ""
    # print("Generated answer: ", generated_answer)
    return generated_answer, [], [triples], [question], [answerer_input]

# Data Loading
def load_data(args):
    if args.dataset == 'popqa' and args.knowledge_source != 'wiki_2020':
        raise ValueError("PopQA dataset should be tested with wiki2020 as the knowledge source!")
    
    questions = []
    contexts = []
    others = {}
    answers = []
    
    data_path = os.path.join(args.test_data_path, args.dataset + "_test.json")
    with open(data_path, 'r') as f:
        data = json.load(f)

    questions = data['question']
    contexts = data['context']
    others = {key: data[key] for key in data.keys() if key not in ['question', 'context']}
    
    q2c = {}
    for i in range(len(questions)):
        q2c[questions[i]] = contexts[i]
            
    return data, questions, q2c, others
          
           
# Model loading
def load_models(args):
    if args.planner_model != 'sonnet':
        print(f"Using {args.planner_model} as planner model, initializing model ...")
        args.llm_frozen = args.planner_frozen
        planner_model = Planner(args)
        load_model(planner_model, args.planner_checkpoint)
    else:
        planner_model = "sonnet"
        print("Using Sonnet, skipping planner model loading ...")
        
    if args.answerer_model != 'sonnet':
        print(f"Using {args.answerer_model} as answerer model, initializing model ...")
        args.llm_frozen = args.answerer_frozen
        answerer_model = Answerer(args)
        load_model(answerer_model, args.answerer_checkpoint)
    else:
        answerer_model = "sonnet"
        print("Using Sonnet, skipping answerer model loading ...")
        
    if args.text_to_triples_model == 'sonnet':
        print("Using Sonnet, skipping text-to-triples model loading ...")
        t2t_tokenizer = None
        t2t_model = "sonnet"
    else:
        print(f"Loading text-to-triples model {args.text_to_triples_model} ...")
        t2t_tokenizer = T5Tokenizer.from_pretrained(args.text_to_triples_model)
        t2t_model = T5ForConditionalGeneration.from_pretrained(args.text_to_triples_model, device_map="auto", torch_dtype=torch.bfloat16)
        
    return planner_model, answerer_model, t2t_tokenizer, t2t_model
        

def process_doc(doc):
    try:
        return text_to_triples_sonnet(doc).replace("\n", " ")
    except Exception as e:
        print(f"Error processing document: {e}")
        return ""  # Return empty string on error

# RAS
def ras(args, models, question, context, graph_processor, retriever):
    if args.dataset == "asqa" or args.dataset == "asqa_train":
        return ras_asqa_sonnet(args, models, question, context, graph_processor, retriever)
    
    if args.dataset == "eli5":
        return ras_eli5_sonnet(args, models, question, context, graph_processor, retriever)
    
    
    

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='popqa')
    parser.add_argument('--test_data_path', type=str, default='/shared/eng/pj20/firas_data/test_datasets')
    parser.add_argument('--knowledge_source', type=str, default='wiki_2017', choices=['wiki_2017', 'wiki_2018', 'wiki_2020'])
    parser.add_argument('--knowledge_path', type=str, default='')
    parser.add_argument('--dense_encoder', type=str, default='facebook/contriever-msmarco')
    parser.add_argument('--theme_encoder_path', type=str, default='/shared/eng/pj20/firas_data/classifiers/best_model')
    parser.add_argument('--theme_shifter_path', type=str, default='/shared/eng/pj20/firas_data/classifiers/best_distribution_mapper.pt')
    parser.add_argument('--text_to_triples_model', type=str, default='pat-jj/text2triple-flan-t5', choices=['pat-jj/text2triple-flan-t5', 'sonnet'])
    parser.add_argument('--planner_model', type=str, default='llama2-7b', choices=['llama2-7b', 'llama3-8b', 'sonnet'])
    parser.add_argument('--planner_checkpoint', type=str, default='')
    parser.add_argument('--answerer_model', type=str, default='llama2-7b', choices=['llama2-7b', 'llama3-8b', 'sonnet'])
    parser.add_argument('--answerer_checkpoint', type=str, default='')
    parser.add_argument('--retrieval_mode', type=str, default='theme_and_dense', choices=['theme_and_dense', 'dense_only'])
    parser.add_argument('--max_answer_length', type=int, default=100)
    parser.add_argument('--max_iteration', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    
    
    # model specific arguments
    parser.add_argument('--llm_model_path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--planner_frozen', type=str, default='False')
    parser.add_argument('--answerer_frozen', type=str, default='False')
    parser.add_argument('--finetune_method', type=str, default='lora')
    parser.add_argument('--gnn_model_name', type=str, default='gt')
    parser.add_argument('--gnn_in_dim', type=int, default=1024)
    parser.add_argument('--gnn_hidden_dim', type=int, default=1024)
    parser.add_argument('--gnn_num_layers', type=int, default=3)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--gnn_num_heads', type=int, default=8)
    parser.add_argument('--max_txt_len', type=int, default=2500)
    parser.add_argument('--max_new_tokens', type=int, default=150)  
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    return parser.parse_args()

def process_batch(questions_batch, contexts_batch):
    """Process a batch of questions and contexts in parallel"""
    triples_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Process all contexts in parallel
        triples_list = list(executor.map(
            lambda context: text_to_triples_sonnet(context).replace("\n", " "), 
            contexts_batch
        ))
    
    # Create answerer inputs for all questions
    results = []
    for question, triples in zip(questions_batch, triples_list):
        answerer_input = ALCE_2_SHOT_TRIPLES_INST + question + "\nRetrieved Graph Information:" + triples + "\nAnswer:"
        results.append({
            'question': question,
            'answer': "",  # Empty string as specified in original code
            'graphs': [],
            'triple_lists': [triples],
            'subqueries': [question],
            'inputs': [answerer_input]
        })
    return results

def main():
    args = read_args()
    print("Loading data ...")
    data, questions, q2c, others = load_data(args)
    
    print("Loading models ...")
    models = load_models(args)
    
    retriever = None

    output_path = os.path.join(args.test_data_path, args.dataset + f"_test_output_{args.planner_model}_{args.answerer_model}.json")
    
    # Load existing progress if any
    if os.path.exists(output_path):
        print(f"Load existing output ...")
        with open(output_path, 'r') as f:
            data = json.load(f)
        generated_answers = data['output']
        triple_lists = data['triple_lists']
        subqueries = data['subqueries']
        inputs = data['llm_inputs']
        processed_questions = len(generated_answers)
        print(f"Loaded {processed_questions} questions ...")
    else:
        print("No existing output, starting from scratch ...")
        generated_answers = []
        triple_lists = []
        subqueries = []
        inputs = []
        processed_questions = 0

    questions = questions[processed_questions:]
    
    # Process in batches of 100
    BATCH_SIZE = 10
    with tqdm(total=len(questions), desc="Processing questions") as pbar:
        for i in range(0, len(questions), BATCH_SIZE):
            batch_questions = questions[i:i + BATCH_SIZE]
            batch_contexts = [q2c[q] for q in batch_questions]
            
            results = process_batch(batch_questions, batch_contexts)
            
            for result in results:
                generated_answers.append(result['answer'])
                triple_lists.append(result['triple_lists'])
                subqueries.append(result['subqueries'])
                inputs.append(result['inputs'])
            
            # Save intermediate results after each batch
            data['output'] = generated_answers
            data['triple_lists'] = triple_lists
            data['subqueries'] = subqueries
            data['llm_inputs'] = inputs
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            pbar.update(len(batch_questions))

    # Final save
    data['output'] = generated_answers
    data['triple_lists'] = triple_lists
    data['subqueries'] = subqueries
    data['llm_inputs'] = inputs
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()






import argparse
import json
import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from models.graphllm_ans_v2 import GraphLLM as Answerer
from models.graphllm_pla_v2 import GraphLLM as Planner
from utils import GraphProcessor, get_planner_instruction, get_answerer_instruction, text_to_triples, TASK_INST, clean_document, load_file, ras_asqa_sonnet, ras_eli5_sonnet, convert_triple_str_to_graph
from tqdm import tqdm
from td_retriever import ThemeScopedRetriever
from sonnet import planner_sonnet, answerer_sonnet, text_to_triples_sonnet
from safetensors.torch import load_model
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import time
from functools import partial


# Data Loading
def load_data(args):
    if args.dataset == 'popqa' and args.knowledge_source != 'wiki2020':
        raise ValueError("PopQA dataset should be tested with wiki2020 as the knowledge source!")
    
    questions = []
    contexts = []
    others = {}
    answers = []
    
    if args.dataset == 'arc_c':
        data_path = os.path.join(args.test_data_path, args.dataset + "_test_processed.jsonl")
        data_ = load_file(data_path)
        for item in data_:
            questions.append(item['instruction'])
            contexts.append([d['text'] for d in item['ctxs']][:5])
            answers.append(item['answerKey'])
            
        data = {}
        data['question'] = questions
        data['context'] = contexts
        data['answer'] = answers
        data['others'] = others
        
    if args.dataset == 'asqa':
        data_path = os.path.join(args.test_data_path, args.dataset + "_test_processed.json")
        with open(data_path, 'r') as f:
            data = json.load(f)['data']
        for item in data:
            questions.append(item['question'])
            contexts.append([d['text'] for d in item['ctxs']][:5])
            answers.append(item['answer'])
            
        data = {}
        data['question'] = questions
        data['context'] = contexts
        data['answer'] = answers
        data['others'] = others
          
        
    else:
        data_path = os.path.join(args.test_data_path, args.dataset + "_test.json")
        with open(data_path, 'r') as f:
            data = json.load(f)
    
        questions = data['question']
        contexts = data['context']
        others = {key: data[key] for key in data.keys() if key not in ['question', 'context']}
    
    if args.dataset == 'triviaqa':
        questions = questions[:1000]
    
    questions_new = []
    if args.dataset == 'pubhealth':
        for question in questions:
            questions_new.append(TASK_INST["fever"] + "\n\n### Input:\n" + question)
        questions = questions_new
        
    # questions_new = []
    # if args.dataset == '2wikimultihop':
    #     for question in questions:
    #         questions_new.append(TASK_INST["2wikimultihop"] + "\nQuestion: " + question)
    #     questions = questions_new
        
    q2c = {}
    # if args.dataset != '2wikimultihop':
    for i in range(len(questions)):
        q2c[questions[i]] = contexts[i]
            
    # else:
    #     for i in range(len(questions)):
    #         context_list = []
    #         for j in range(len(contexts[i])):
    #             context_list.append(contexts[i][j])
    #         q2c[questions[i]] = context_list
    
    return data, questions, q2c, others
          
           
# Model loading
def load_models(args):
    if args.planner_model != 'sonnet':
        print(f"Using {args.planner_model} as planner model, initializing model ...")
        planner_model = Planner(args)
        load_model(planner_model, args.planner_checkpoint)
    else:
        planner_model = "sonnet"
        print("Using Sonnet, skipping planner model loading ...")
        
    if args.answerer_model != 'sonnet':
        print(f"Using {args.answerer_model} as answerer model, initializing model ...")
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
    if args.dataset == "asqa":
        return ras_asqa_sonnet(args, models, question, context, graph_processor, retriever)
    
    if args.dataset == "eli5":
        return ras_eli5_sonnet(args, models, question, context, graph_processor, retriever)
    
    planner_model, answerer_model, t2t_tokenizer, t2t_model = models
    
    planner_instruction = get_planner_instruction(args.planner_model)
    answerer_instruction = get_answerer_instruction(args.answerer_model)
    end_iteration_flag = False
    
    # Stage 0: Determine if retrieval is needed
    # start_time = time.perf_counter()
    if args.planner_model != "sonnet":
        planner_complete_input = planner_instruction + "\n" + question
        with torch.no_grad():
            planner_output = planner_model.inference({
                'input': [planner_complete_input],  # Batch size 1
                'graphs': [[]],  # Empty graphs list
                'label': ['']  # Dummy label
            })['pred'][0]
    else:
        planner_output = planner_sonnet(question)

    # planner_time = time.perf_counter() - start_time
    # print(f"Planner time: {planner_time:.2f}s")
        
    if args.debug:
        print(f"Initial planner output: {planner_output}")
    
    if '[NO_RETRIEVAL]'.lower() in planner_output.lower() or 'SUFFICIENT'.lower() in planner_output.lower():
        end_iteration_flag = True
    # else:
    #     sub_query = planner_output
    
    # we set the first sub_query to be the question itself
    sub_query = question
    
    graphs = []
    retrieved_docs_list = []
    triple_lists = []
    subqueries = []
    inputs = []
    
    

    iteration = 0
    while not end_iteration_flag and iteration < args.max_iteration:
        
        print(f"Iteration {iteration+1} starts ...")
        
        if args.debug:
            print(f"Sub query: {sub_query}")
        
        # Stage 1: Theme-based retrieval
        print(f"Retrieving information ...")
        # start_time = time.perf_counter()
        if iteration == 0:
            if args.dataset != "2wikimultihop":
                retrieved_docs = context
            else:
                retrieved_docs = context[:5]
        else:
            retrieved_docs = retriever.retrieve(sub_query, top_k=5)
            retrieved_docs = [item[0] for item in retrieved_docs]
            
        # retrieval_time = time.perf_counter() - start_time
        # print(f"Retrieval time: {retrieval_time:.2f}s")
        
        # clean_docs = [doc for doc in retrieved_docs if clean_document(doc)]
        # retrieved_docs = clean_docs[:5]
        
        if args.debug:
            print(f"Retrieved docs: {retrieved_docs}")
            
        retrieved_docs_list.append(retrieved_docs)
        
        # Stage 2: Text-to-triples-to-graph 
        print(f"Text-to-triples ...")
        # start_time = time.perf_counter()
        if t2t_model != "sonnet":
            triples = ""
            for retrieved_doc in retrieved_docs:
                triples += text_to_triples(t2t_tokenizer, t2t_model, retrieved_doc)
                
            graph = graph_processor.create_graph_from_triples(triples)
            graphs.append(graph)
        else:
            # by Sonnet
            triples = text_to_triples_sonnet("Question: " + question + "\nRetrieval:" + "\n".join(retrieved_docs)).replace("\n", " ")
            
            if answerer_model != "sonnet" or planner_model != "sonnet":
                graph, triples_ = graph_processor.create_graph_from_triples(triples)   
                
                if graph is None or triples_ is None:
                    print(f"Error processing triples: {triples}")
                    iteration += 1
                    continue
                
                graphs.append(graph)
                triples = triples_
            
        # text_to_triples_time = time.perf_counter() - start_time
        # print(f"Text-to-triples time: {text_to_triples_time:.2f}s")
        
        if args.debug:
            print(f"Triples: {triples}")
        
        # Iterative Information Gathering
        subqueries.append(sub_query)
        triple_lists.append(triples)
        
        planner_intput = ""
        for i in range(len(subqueries)):
            planner_intput += subqueries[i] + "\n" + "Retrieved Graph Information: " + str(triple_lists[i]) + '\n'
            
        planner_intput += "Question: " + question
        inputs.append(planner_intput)
        
        if planner_model == "sonnet":
            planner_intput = planner_intput.replace("Retrieved Graph Information:", "[PREV_GRAPH_INFO]").replace("[SUBQ]", "[PREV_SUBQ]")
        
        if args.debug:
            print(f"Planner input: {planner_intput}")
        
        # Stage 3: Plan next action
        print(f"Planning next action ...")
        # start_time = time.perf_counter()
        if planner_model != "sonnet":
            planner_complete_input = planner_instruction + "\n" + planner_intput
            with torch.no_grad():
                planner_output = planner_model.inference({
                    'input': [planner_complete_input],
                    'graphs': [graphs],
                    'label': [''] # dummy label
                })['pred'][0]
                
        else:
            planner_output = planner_sonnet(planner_intput)
            
        # planner_time = time.perf_counter() - start_time
        # print(f"Planner time: {planner_time:.2f}s")
            
        if args.debug:
            print(f"Planner output: {planner_output}")
            
        if 'SUFFICIENT'.lower() in planner_output.lower() or 'NO_RETRIEVAL'.lower() in planner_output.lower(): # information is sufficient
            end_iteration_flag = True
        else:
            sub_query = planner_output
        iteration += 1
    # Stage 4: Answering
    ## answer the question with the final input
    ### [NO_RETRIEVAL] case
    if len(inputs) == 0:
        question = "Question: " + question
        inputs.append(question)
        
    print(f"Answering the question ...")
    # start_time = time.perf_counter()
    
    if answerer_model != "sonnet":
        answerer_input = {
            'input': [answerer_instruction + "\n" + inputs[-1]],
            'graphs': [graphs],
            'label': [''] # dummy label
        }
    
        with torch.no_grad():
            answerer_output = answerer_model.inference(answerer_input)['pred'][0]
    else:
        answerer_output = answerer_sonnet(inputs[-1], max_answer_length=args.max_answer_length)
        
    # answerer_time = time.perf_counter() - start_time
    # print(f"Answerer time: {answerer_time:.2f}s")
        
    if args.debug:
        print(f"Answerer output: {answerer_output}")
        
    generated_answer = answerer_output
        
    return generated_answer, graphs, triple_lists, subqueries, inputs
    
    

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
    parser.add_argument('--llm_frozen', type=str, default='False')
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


def main():
    args = read_args()
    print("Loading data ...")
    data, questions, q2c, others = load_data(args)
    
    print("Loading models ...")
    models = load_models(args)
    
    print("Loading graph processor ...")
    graph_processor = GraphProcessor()
    if args.dataset == "asqa" or args.dataset == "eli5":
        print("ASQA or ELI5 task, using top 5 docs as context and no retrieval ...")
        retriever = None
    else:
        retriever = ThemeScopedRetriever(retrieval_mode=args.retrieval_mode, debug=args.debug)
    
    if os.path.exists(os.path.join(args.test_data_path, args.dataset + f"_test_output_{args.planner_model}_{args.answerer_model}.json")):
        print(f"Load existing output ...")
        with open(os.path.join(args.test_data_path, args.dataset + f"_test_output_{args.planner_model}_{args.answerer_model}.json"), 'r') as f:
            data = json.load(f)
        generated_answers = data['output']
        graphs = data['graphs']
        triple_lists = data['triple_lists']
        subqueries = data['subqueries']
        inputs = data['llm_inputs']
        processed_questions = len(generated_answers)
        print(f"Loaded {processed_questions} questions ...")
    else:
        print("No existing output, starting from scratch ...")
    
        generated_answers = []
        graphs = []
        triple_lists = []
        subqueries = []
        inputs = []
        processed_questions = 0
        
    
    cnt = 0
    questions = questions[processed_questions:]
    print("Iteratively RAS ...")
    for question in tqdm(questions):
        # try:
        generated_answer, graphs_, triple_lists_, subqueries_, inputs_ = ras(args, models, question, q2c[question], graph_processor, retriever)
        generated_answers.append(generated_answer)
        graphs.append(graphs_)
        triple_lists.append(triple_lists_)
        subqueries.append(subqueries_)
        inputs.append(inputs_)
        cnt += 1
        if cnt % 10 == 0:
            print(f"Processed {cnt} questions ...")
            data['output'] = generated_answers
            data['graphs'] = graphs
            data['triple_lists'] = triple_lists
            data['subqueries'] = subqueries
            data['llm_inputs'] = inputs
            with open(os.path.join(args.test_data_path, args.dataset + f"_test_output_{args.planner_model}_{args.answerer_model}.json"), 'w') as f:
                json.dump(data, f, indent=4)
        # except Exception as e:
        #     print(f"Error: {e}")
        #     continue
                

    with open(os.path.join(args.test_data_path, args.dataset + f"_test_output_{args.planner_model}_{args.answerer_model}.json"), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()






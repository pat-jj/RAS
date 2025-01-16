import argparse
import json
import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from utils import GraphProcessor, load_theme_classifier, load_theme_distribution_shifter, get_planner_instruction, get_answerer_instruction
from tqdm import tqdm
import numpy as np

# Data Loading
def load_data(args):
    if args.dataset == 'popqa' and args.knowledge_source != 'wiki2020':
        raise ValueError("PopQA dataset should be tested with wiki2020 as the knowledge source!")
    data_path = os.path.join(args.test_data_path, args.dataset + "_test.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    questions = data['question']
    answers = data['answer']
    others = {key: data[key] for key in data.keys() if key not in ['question', 'answer']}
    
    return questions, answers, others
    
    
# Knowledge loading
def load_knowledge(args):
    knowledge_path = os.path.join(args.knowledge_path, args.knowledge_source)
    dense_index_path = os.path.join(knowledge_path, 'embedding')
    theme_index_path = os.path.join(knowledge_path, 'theme_dist')
    
    # Load text mappings
    with open(os.path.join(dense_index_path, 'text_mapping.json'), 'r') as f:
        dense_text_mapping = json.load(f)
    with open(os.path.join(theme_index_path, 'text_mapping.json'), 'r') as f:
        theme_text_mapping = json.load(f)
    
    # Create index mappings
    theme_to_dense_idx = {}
    dense_to_theme_idx = {}
    for theme_idx, text in enumerate(theme_text_mapping):
        if text in dense_text_mapping:
            dense_idx = dense_text_mapping.index(text)
            theme_to_dense_idx[theme_idx] = dense_idx
            dense_to_theme_idx[dense_idx] = theme_idx
    
    # Load FAISS indices
    dense_faiss_index = faiss.read_index(os.path.join(dense_index_path, 'wikipedia_embeddings.faiss'))
    theme_faiss_index = faiss.read_index(os.path.join(theme_index_path, 'theme_embeddings.faiss'))
    
    return dense_text_mapping, dense_faiss_index, theme_text_mapping, theme_faiss_index, \
           theme_to_dense_idx, dense_to_theme_idx
          
           
# Model loading
def load_models(args):
    if args.planner_model != 'sonnet':
        print(f"Using {args.planner_model} as planner model, initializing model ...")
        planner_tokenizer = AutoTokenizer.from_pretrained(args.planner_model)
        planner_model = AutoModelForCausalLM.from_pretrained(args.planner_model)
    else:
        planner_tokenizer = None
        planner_model = "sonnet"
        print("Using Sonnet, skipping planner model loading ...")
        
    if args.answerer_model != 'sonnet':
        print(f"Using {args.answerer_model} as answerer model, initializing model ...")
        answerer_tokenizer = AutoTokenizer.from_pretrained(args.answerer_model)
        answerer_model = AutoModelForCausalLM.from_pretrained(args.answerer_model)
    else:
        answerer_tokenizer = None
        answerer_model = "sonnet"
        print("Using Sonnet, skipping answerer model loading ...")
        
    print("Loading Dense Encoder ...")
    dense_encoder_tokenizer = AutoTokenizer.from_pretrained(args.dense_encoder)
    dense_encoder_model = AutoModel.from_pretrained(args.dense_encoder)
    
    print("Loading Theme Encoder & Classifier ...")
    theme_classifier, theme_encoder, theme_label_mapping = load_theme_classifier(args.theme_encoder_path)
    
    print("Loading Theme Distribution Shifter ...")
    theme_shifter = load_theme_distribution_shifter(args.theme_shifter_path, input_dim=len(theme_label_mapping))
    
    print(f"Loading text-to-triples model {args.text_to_triples_model} ...")
    t2t_tokenizer = T5Tokenizer.from_pretrained(args.text_to_triples_model)
    t2t_model = T5ForConditionalGeneration.from_pretrained(args.text_to_triples_model, device_map="auto", torch_dtype=torch.bfloat16)
        
    return planner_tokenizer, planner_model, answerer_tokenizer, answerer_model, dense_encoder_tokenizer, dense_encoder_model, \
        theme_classifier, theme_encoder, theme_label_mapping, theme_shifter, t2t_tokenizer, t2t_model
        

# Theme-based retrieval and dense retrieval
def theme_scoping_and_dense_retrieval(dense_encoder_tokenizer, dense_encoder_model, \
        theme_classifier, theme_encoder, theme_label_mapping, theme_shifter, knowledge_index, \
        query, others=None, theme_top_k=50000, dense_top_k=5):
    
    dense_text_mapping, dense_faiss_index, theme_text_mapping, theme_faiss_index = knowledge_index
    query = query.split("[SUBQ]")[1].strip() if "[SUBQ]" in query else query
    
    # Create index mapping if not already created (could be done during initialization)
    theme_to_dense_idx = {}
    dense_to_theme_idx = {}
    for theme_idx, text in enumerate(theme_text_mapping):
        if text in dense_text_mapping:
            dense_idx = dense_text_mapping.index(text)
            theme_to_dense_idx[theme_idx] = dense_idx
            dense_to_theme_idx[dense_idx] = theme_idx
    
    # Stage 1: Theme-based retrieval
    query_theme_embedding = theme_encoder.encode(query, convert_to_tensor=True)
    query_theme_probs = theme_classifier(query_theme_embedding.unsqueeze(0))
    predicted_theme_distribution = theme_shifter(query_theme_probs)
    predicted_theme_distribution = predicted_theme_distribution / \
        torch.norm(predicted_theme_distribution, p=2, dim=1, keepdim=True)
    
    # Search top k with theme index
    theme_scores, theme_doc_ids = theme_faiss_index.search(predicted_theme_distribution.cpu().numpy(), k=theme_top_k)
    
    # Convert theme doc ids to dense doc ids
    dense_candidate_ids = [theme_to_dense_idx[int(idx)] for idx in theme_doc_ids[0] if int(idx) in theme_to_dense_idx]
    
    # Stage 2: Dense retrieval
    query_inputs = dense_encoder_tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_dense_embedding = dense_encoder_model(**query_inputs).last_hidden_state[:, 0].cpu().numpy()
        query_dense_embedding = query_dense_embedding / np.linalg.norm(query_dense_embedding)
    
    # Search in dense index with filtered ids
    dense_scores, dense_doc_ids = dense_faiss_index.search(query_dense_embedding, k=dense_top_k, subset_ids=dense_candidate_ids)
    
    # Get theme scores for selected documents
    selected_theme_scores = []
    for doc_id in dense_doc_ids[0]:
        # Convert dense id back to theme id
        theme_id = dense_to_theme_idx[doc_id]
        # Find position in theme results
        theme_pos = np.where(theme_doc_ids[0] == theme_id)[0][0]
        selected_theme_scores.append(theme_scores[0][theme_pos])
    
    # Combine scores
    final_scores = 0.8 * dense_scores[0] + 0.2 * np.array(selected_theme_scores)
    
    # Sort by combined scores
    sorted_indices = np.argsort(-final_scores)
    final_texts = [dense_text_mapping[dense_doc_ids[0][i]] for i in sorted_indices]
    final_scores = final_scores[sorted_indices]
    
    # # Use dense scores only
    # final_texts = [dense_text_mapping[idx] for idx in dense_doc_ids[0]]
    # final_scores = dense_scores[0]
    
    return final_texts, final_scores
        

def text_to_triples(t2t_tokenizer, t2t_model, texts):
    inputs = t2t_tokenizer(texts, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    
    input_ids = inputs['input_ids'].to(t2t_model.device)
    attention_mask = inputs['attention_mask'].to(t2t_model.device)

    with torch.no_grad():
        outputs = t2t_model.generate(input_ids=input_ids, attention_mask=attention_mask,
            max_length=512, num_beams=4, early_stopping=True, length_penalty=0.6, use_cache=True)
    
    triples = t2t_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return triples


# RAS
def ras(args, models, knowledge_index, question, others=None):
    planner_tokenizer, planner_model, answerer_tokenizer, answerer_model, dense_encoder_tokenizer, dense_encoder_model, \
        theme_classifier, theme_encoder, theme_label_mapping, theme_shifter, t2t_tokenizer, t2t_model = models
    
    graph_processor = GraphProcessor()
    planner_instruction = get_planner_instruction(args.planner_model)
    answerer_instruction = get_answerer_instruction(args.answerer_model)
    end_iteration_flag = False
    
    # Stage 0: Determine if retrieval is needed
    with torch.no_grad():
        planner_output = planner_model.inference({
            'input': [planner_instruction + "\n" + question],  # Batch size 1
            'graphs': [[]],  # Empty graphs list
            'label': ['']  # Dummy label
        })['pred'][0]
    
    if '[NO_RETRIEVAL]'.lower() in planner_output.lower():
        end_iteration_flag = True
        sub_query = question
    else:
        sub_query = planner_output
    
    graphs = []
    triple_lists = []
    subqueries = []
    inputs = []

    while not end_iteration_flag and len(graphs) < 5:
        # Stage 1: Theme-based retrieval
        retrieved_docs, retrieved_scores = theme_scoping_and_dense_retrieval(dense_encoder_tokenizer, dense_encoder_model, \
            theme_classifier, theme_encoder, theme_label_mapping, theme_shifter, knowledge_index, sub_query, others)
        
        # Stage 2: Text-to-triples-to-graph
        triples = text_to_triples(t2t_tokenizer, t2t_model, retrieved_docs)
        graph = graph_processor.create_graph_from_triples(triples)
        
        subqueries.append(sub_query)
        triple_lists.append(triples)
        graphs.append(graph)
        
        planner_intput = ""
        for i in range(len(subqueries)):
            planner_intput += subqueries[i] + "\n" + "Retrieved Graph Information: " + str(triple_lists[i]) + '\n'
            
        planner_intput += "Question: " + question
        inputs.append(planner_intput)
                
        # Stage 3: Plan next action
        with torch.no_grad():
            planner_output = planner_model.inference({
                'input': [planner_instruction + "\n" + planner_intput],
                'graphs': [graphs],
                'label': [''] # dummy label
            })['pred'][0]
            
        if '[SUFFICIENT]'.lower() in planner_output.lower(): # information is sufficient
            end_iteration_flag = True
        else:
            sub_query = planner_output
    
    # Stage 4: Answering
    ## answer the question with the final input
    answerer_input = {
        'input': [answerer_instruction + "\n" + inputs[-1]],
        'graphs': [graphs],
        'label': [''] # dummy label
    }
    
    with torch.no_grad():
        answerer_output = answerer_model.inference(answerer_input)['pred'][0]
        
    generated_answer = answerer_output
        
    return generated_answer
    
    
# Evaluation
def evaluate(args, generated_answers, true_answers):
    
    
    return {}
    
    


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='popqa')
    parser.add_argument('--test_data_path', type=str, default='/shared/eng/pj20/firas_data/datasets/test_datasets')
    parser.add_argument('--knowledge_source', type=str, default='wiki_2017', choices=['wiki_2017', 'wiki_2018', 'wiki_2020'])
    parser.add_argument('--knowledge_path', type=str, default='')
    parser.add_argument('--dense_encoder', type=str, default='facebook/contriever-msmarco')
    parser.add_argument('--theme_encoder_path', type=str, default='/shared/eng/pj20/firas_data/classifiers/best_model')
    parser.add_argument('--theme_shifter_path', type=str, default='/shared/eng/pj20/firas_data/classifiers/best_distribution_mapper.pt')
    parser.add_argument('--text_to_triples_model', type=str, default='pat-jj/text2triple-flan-t5')
    parser.add_argument('--planner_model', type=str, default='llama2-7b', choices=['llama2-7b', 'llama3-8b', 'sonnet'])
    parser.add_argument('--planner_checkpoint', type=str, default='')
    parser.add_argument('--answerer_model', type=str, default='llama2-7b', choices=['llama2-7b', 'llama3-8b', 'sonnet'])
    parser.add_argument('--answerer_checkpoint', type=str, default='')
    parser.add_argument('--retrieval_method', type=str, default='theme_and_dense', choices=['theme_and_dense', 'dense_only'])
    return parser.parse_args()


def main():
    args = read_args()
    questions, answers, others = load_data(args)
    knowledge_index = load_knowledge(args)
    models = load_models(args)
    
    print("Iteratively RAS ...")
    generated_answers = []
    for question in tqdm(questions):
        generated_answer = ras(args, models, knowledge_index, question, others)
        generated_answers.append(generated_answer)
    
    print("Evaluating ...")
    evaluation_results = evaluate(args, generated_answers, answers)
    print("Evaluation results:")
    for key in evaluation_results:
        print(f"{key}: {evaluation_results[key]}")


if __name__ == "__main__":
    main()






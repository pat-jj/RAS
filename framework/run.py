from modules.structuring import structure, load_t2t_model
from framework.modules.retrieval import theme_scoping, ThemeScopingConfig, ThemeScoping
from modules.generation import generate_subquery, generate_answer
import argparse
import json
import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from utils import load_theme_classifier

# Data Loading
def load_data(args):
    if args.dataset == 'popqa' and args.knowledge_source != 'wiki2020':
        raise ValueError("PopQA dataset should be tested with wiki2020 as the knowledge source")
    data_path = os.path.join(args.test_data_path, args.dataset + "_test.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    questions = data['question']
    answers = data['answer']
    others = {key: data[key] for key in data.keys() if key not in ['question', 'answer']}
    
    return questions, answers, others
    
    
def load_knowledge(args):
    knowledge_path = os.path.join(args.knowledge_path, args.knowledge_source)
    dense_index_path = os.path.join(knowledge_path, 'embedding')
    theme_index_path = os.path.join(knowledge_path, 'theme_dist')
    dense_text_mapping = os.path.join(dense_index_path, 'text_mapping.json')
    dense_faiss_index = os.path.join(dense_index_path, 'wikipedia_embeddings.faiss')
    if args.retrieval_method == 'theme_and_dense':
        theme_text_mapping = os.path.join(theme_index_path, 'text_mapping.json')
        theme_faiss_index = os.path.join(theme_index_path, 'theme_embeddings.faiss')
    return dense_text_mapping, dense_faiss_index, theme_text_mapping if args.retrieval_method == 'theme_and_dense' else None, theme_faiss_index if args.retrieval_method == 'theme_and_dense' else None


def init_models(args):
    if args.planner_model != 'sonnet':
        print(f"Using {args.planner_model} as planner model, initializing model ...")
        planner_tokenizer = AutoTokenizer.from_pretrained(args.planner_model)
        planner_model = AutoModelForCausalLM.from_pretrained(args.planner_model)
    else:
        print("Using Sonnet, skipping planner model initialization ...")
        
    if args.answerer_model != 'sonnet':
        print(f"Using {args.answerer_model} as answerer model, initializing model ...")
        answerer_tokenizer = AutoTokenizer.from_pretrained(args.answerer_model)
        answerer_model = AutoModelForCausalLM.from_pretrained(args.answerer_model)
    else:
        print("Using Sonnet, skipping answerer model initialization ...")
        
    print("Loading Dense Encoder ...")
    dense_encoder_tokenizer = AutoTokenizer.from_pretrained(args.dense_encoder)
    dense_encoder_model = AutoModel.from_pretrained(args.dense_encoder)
    
    print("Loading Theme Encoder ...")
    theme_classifier, theme_encoder, theme_label_mapping = load_theme_classifier(args.theme_encoder_path)
    
    print(f"Loading text-to-triples model {args.text_to_triples_model} ...")
    t2t_tokenizer = T5Tokenizer.from_pretrained(args.text_to_triples_model)
    t2t_model = T5ForConditionalGeneration.from_pretrained(args.text_to_triples_model, device_map="auto", torch_dtype=torch.bfloat16)
        
    return planner_tokenizer, planner_model, answerer_tokenizer, answerer_model, dense_encoder_tokenizer, dense_encoder_model, \
        theme_classifier, theme_encoder, theme_label_mapping, t2t_tokenizer, t2t_model
        
        
def iterative_ras(args, models, knowledge_index):
    dense_text_mapping, dense_faiss_index, theme_text_mapping, theme_faiss_index = knowledge_index
    planner_tokenizer, planner_model, answerer_tokenizer, answerer_model, dense_encoder_tokenizer, dense_encoder_model, \
        theme_classifier, theme_encoder, label_mapping, t2t_tokenizer, t2t_model = models
        
        
    
    
    
    
    


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='popqa')
    parser.add_argument('--test_data_path', type=str, default='/shared/eng/pj20/firas_data/datasets/test_datasets')
    parser.add_argument('--knowledge_source', type=str, default='wiki_2017', choices=['wiki_2017', 'wiki_2018', 'wiki_2020'])
    parser.add_argument('--knowledge_path', type=str, default='')
    parser.add_argument('--dense_encoder', type=str, default='facebook/contriever-msmarco')
    parser.add_argument('--theme_encoder_path', type=str, default='/shared/eng/pj20/firas_data/classifiers/best_model')
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
    models = init_models(args)




if __name__ == "__main__":
    main()






import json
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
import random

def load_hotpotqa_main_data(path):
    """
    [
        {
            "question": "Which magazine was started first Arthur's Magazine or First for Women?",
            "answer": "Arthur's Magazine",
            "type": "comparison",
            "supporting_docs": {
            "arthur's magazine": [
                "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into \"Godey's Lady's Book\"."
            ],
            "first for women": [
                "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies."
            ]
            },
            "sub_queries": [
            {
                "sub_query": "When was Arthur's Magazine first published?",
                "doc_topic": "arthur's magazine",
                "supporting_doc": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into \"Godey's Lady's Book\"."
            },
            {
                "sub_query": "When was First for Women magazine started?",
                "doc_topic": "first for women",
                "supporting_doc": "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies."
            }
            ]
        },
        ...
    ]
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_hotpotqa_filtered_data(path):
    """
    [
        {
            "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "answer": "Delhi",
            "type": "bridge",
            "supporting_docs": {
            "oberoi family": [
                "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group."
            ],
            "the oberoi group": [
                "The Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands."
            ]
            },
            "sub_queries": [
            {
                "sub_query": "What is the name of the hotel company associated with the Oberoi family?",
                "doc_topic": "oberoi family",
                "supporting_doc": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group."
            },
            {
                "sub_query": "Where is the head office of The Oberoi Group located?",
                "doc_topic": "the oberoi group",
                "supporting_doc": "The Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands."
            }
            ]
        },
        ...
    ]
    """
    with open(path, 'r') as f:
        data = json.load(f)
    question_set = set()
    question_to_docs = defaultdict(list)
    question_to_subqueries = defaultdict(list)
    print("Processing filtered data ...")
    for item in tqdm(data):
        try:
            if item['question'] not in question_set:
                doc_to_subquery = {}
                question_set.add(item['question'])
                for doc_topic, doc_texts in item['supporting_docs'].items():
                    for doc_text in doc_texts:
                        if doc_text != "":
                            question_to_docs[item['question']].append((doc_text))
                    
                for sub_query in item['sub_queries']:
                    if sub_query['supporting_doc'] != "":
                        doc_to_subquery[sub_query['supporting_doc']] = sub_query['sub_query']
                
                
                for doc_text in question_to_docs[item['question']]:
                    question_to_subqueries[item['question']].append((doc_to_subquery[doc_text]))
                
        except Exception as e:
            print(f"Error processing item {item['question']}: {e}")
            data.remove(item)
            continue
        
        
    return data, question_set, question_to_docs, question_to_subqueries


def load_hotpotqa_questions_can_be_directly_answered(path):
    """
    [
        {
            "question": "Which magazine was started first Arthur's Magazine or First for Women?",
            "true_answer": "Arthur's Magazine",
            "llm_answer": "Arthur's Magazine. Arthur's Magazine was started in 1983, while First for Women was started in 1989.",
            "similarity": 0.736628532409668,
            "needs_subquery": true,
            "label": "[SUBQ=YES]"
        },
        {
            "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "true_answer": "Delhi",
            "llm_answer": "Delhi",
            "similarity": 1.0,
            "needs_subquery": false,
            "label": "[SUBQ=NO] No subquery is needed."
        },
        ...
    ]
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_hotpotqa_questions_can_be_answered_with_single_retrieval(path):
    """
    [
        {
            "question": "Which magazine was started first Arthur's Magazine or First for Women?",
            "true_answer": "Arthur's Magazine",
            "llm_answer": "Arthur's Magazine",
            "similarity": 1.0,
            "can_answer_with_retrieval": true,
            "label": "[RETRIEVAL=YES] Can be answered with retrieval."
        },
        {
            "question": "New Faces of 1952 is a musical revue with songs and comedy skits, it helped jump start the career of which young performer, and American actress?",
            "true_answer": "Carol Lawrence",
            "llm_answer": "Paul Lynde, Eartha Kitt",
            "similarity": 0.2878572344779968,
            "can_answer_with_retrieval": false,
            "label": "[RETRIEVAL=NO]"
        },
        ...
    ]
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_text_to_triples(path):
    """
    [
        {
            "text": "Goertz also served as a social media producer for @midnight.",
            "generated_triple": "(S> Goertz| P> Employer| O> @midnight), (S> Goertz| P> Occupation| O> Social media producer)"
        },
        {
            "text": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
            "generated_triple": "(S> Oberoi family| P> Nationality| O> Indian), (S> Oberoi Group| P> Industry| O> Hotels), (S> Oberoi family| P> Involvement| O> Hotels), (S> Oberoi family| P> Associated with| O> Oberoi Group)"
        },
        ...
    ]
    """
    with open(path, 'r') as f:
        data = json.load(f)
        
    triples_lookup = {item['text']: [triple + ')' if not triple.endswith(')') else triple 
                                for triple in item['generated_triple'].split('), ')] 
                 for item in data}
    return triples_lookup

class GraphProcessor:
    def __init__(self, bert_model='sentence-transformers/all-roberta-large-v1'):
        """
        Initialize the data processor.
        Args:
            bert_model (str): Name of the SentenceBERT model to use for embeddings
        """
        self.sentence_model = SentenceTransformer(bert_model)
        self.embed_dim = self.sentence_model.get_sentence_embedding_dimension()
    

    def create_graph_from_triples(self, triple_strs):
        """Convert a list of triple strings into a PyG graph with predicate encodings"""
        nodes = set()
        edge_triples = []
        
        # Collect unique nodes and edges
        for triple_str in triple_strs:
            # Keep original triple string for description
            triple_str = triple_str.strip('()')
            parts = triple_str.split('|')
            
            # Extract subject, predicate, object
            subject = parts[0].replace('S>', '').strip()
            predicate = parts[1].replace('P>', '').strip()
            object_ = parts[2].replace('O>', '').strip()
            
            nodes.add(subject)
            nodes.add(object_)
            edge_triples.append((subject, predicate, object_))
        
        # Create node mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge index and collect predicates
        edge_index = []
        predicates = []
        
        for subj, pred, obj in edge_triples:
            # Add forward edge
            edge_index.append([node_to_idx[subj], node_to_idx[obj]])
            predicates.append(pred)  # Original predicate
            
            # Add reverse edge
            edge_index.append([node_to_idx[obj], node_to_idx[subj]])
            predicates.append(f"inverse_{pred}")  # Inverse predicate
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create node features (embeddings)
        node_texts = list(nodes)
        node_embeddings = self.sentence_model.encode(node_texts)
        node_features = torch.tensor(node_embeddings, dtype=torch.float)
        
        # Create edge features (only encode predicates)
        predicate_embeddings = self.sentence_model.encode(predicates)
        edge_features = torch.tensor(predicate_embeddings, dtype=torch.float)
        
        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )
        
        return graph, triple_strs  # Return original triple strings for description

    def format_document_triples(self, triples):
        """Format triples from one document keeping original format"""
        formatted = " ".join(f"({triple})" for triple in triples)  # Keep original triple format
        return formatted


def get_instruction():
    return \
# """You are a planner to determine if the question can be answered with current information.
# You will be output [NO_RETRIEVAL] if the question can be directly answered with the question itself.
# You will be output [SUBQ] with the subquery if the question needs a subquery.
# You will be output [SUFFICIENT] if the question can be answered with provided information.
# """
"""You are a planner to determine if the question can be answered with current information and output the appropriate label as well as the subquery if needed.
Output [NO_RETRIEVAL] if the question can be directly answered with the question itself without any retrieval.
Output [SUBQ] with an subquery for retrieval if still needs a subquery.
Output [SUFFICIENT] if the question can be answered with the provided information.
"""
# Output [SUBQ] with the question itself if it can be answered with retrieval by the question itself.


def main():
    # hotpotqa_main_data = load_hotpotqa_main_data('/shared/eng/pj20/firas_data/datasets/hotpotqa/hotpot_with_subqueries.json')
    hotpotqa_filtered_data, question_set, question_to_docs, question_to_subqueries = load_hotpotqa_filtered_data('/shared/eng/pj20/firas_data/datasets/hotpotqa/filtered/hotpot_filtered.json')
    q_direct_answered = load_hotpotqa_questions_can_be_directly_answered('/shared/eng/pj20/firas_data/datasets/hotpotqa/llama_subquery_data/subquery_classification.json')
    q_retrieval_answered = load_hotpotqa_questions_can_be_answered_with_single_retrieval('/shared/eng/pj20/firas_data/datasets/hotpotqa/llama_subquery_data/retrieval_classification.json')
    text_to_triples = load_text_to_triples('/shared/eng/pj20/firas_data/graph_data/hotpotqa/text_triples.json')
    
    output_dir = '/shared/eng/pj20/firas_data/action_planner/hotpot_train_1'
    
    processed_data = []
    processed_questions = set()
    
    instruction = get_instruction()
    
    graph_processor = GraphProcessor()
    
    # P( [NO_Retrieval] or [SUBQ] q_0 | Q )
    ## Case 0: The question can be answered with no retrieval
    print("Processing Case 0: The question can be answered with no retrieval ...")
    for item in tqdm(q_direct_answered):
        if not item['needs_subquery']:
            output = "[NO_RETRIEVAL]"
            processed_item = {
                'input': instruction + "\n" + item['question'],
                'label': output,
                'graphs': [],
            }
            
            processed_data.append(processed_item)
            processed_questions.add(item['question'])
             
    ## Case 1: The question can be answered with retrieval by the question itself
    # print("Processing Case 1: The question can be answered with retrieval by the question itself ...")
    # for item in tqdm(q_retrieval_answered):
    #     if item['question'] not in processed_questions:
    #         if item['can_answer_with_retrieval']:
    #             processed_item = {
    #                 'input': instruction + "\n" + item['question'],
    #                 'label': "[SUBQ]" + " " + item['question'],
    #                 'graphs': [],
    #             }
    #             processed_data.append(processed_item)
    #             processed_questions.add(item['question'])
            
    ## Case 2: The main question needs a subquery
    print("Processing Case 2: The main question needs a subquery ...")
    for item in tqdm(hotpotqa_filtered_data):
        if item['question'] not in processed_questions and question_to_subqueries[item['question']] != []:
            processed_item = {
                'input': instruction + "\n" + item['question'],
                'label': "[SUBQ]" + " " + question_to_subqueries[item['question']][0],
                'graphs': [],
            }
            processed_data.append(processed_item)
            processed_questions.add(item['question'])
            
    # P([Sufficient] or [SUBQ q_(i+1)] | [KG] [SUBQ] q_0 + [Text(g_0)] … + [SUBQ] q_i + [Text(g_i)] + Q )
    print("Processing main data ...")
    skipped_questions = set()
    for item in tqdm(hotpotqa_filtered_data):
        question = item['question']
        docs = question_to_docs[question]
        subqueries = question_to_subqueries[question]
        
        if len(docs) == 0 or len(subqueries) == 0:
            skipped_questions.add(question)
            continue
        
        try:
            assert len(docs) == len(subqueries)
        except Exception as e:
            print(f"Error processing item {item['question']}: {e}")
        
        try:
            for i in range(len(docs)):
                input_ = ""
                for j in range(i):
                    input_ += "[SUBQ] " + subqueries[j] + "\n" + "Retrieved Graph Information: " + str(text_to_triples[docs[j]]) + "\n"
                input_ += "[SUBQ] " + subqueries[i] + "\n" + "Retrieved Graph Information: " + str(text_to_triples[docs[i]]) + "\n" + "Question: " + question
                
                graphs = []
                for j in range(i):
                    graph, triples = graph_processor.create_graph_from_triples(text_to_triples[docs[j]])
                    graphs.append(graph)
                graph, triples = graph_processor.create_graph_from_triples(text_to_triples[docs[i]])
                graphs.append(graph)
            
                processed_item = {
                    'input': instruction + "\n" + input_,
                    'label': "[SUFFICIENT]" if i == len(docs) - 1 else "[SUBQ]" + " " + subqueries[i+1],
                    'graphs': graphs,
                }
                processed_data.append(processed_item)
                
        except Exception as e:
            skipped_questions.add(question)
            print(f"Error processing item {item['question']}: {e}")
            continue
            
    print("Processed data length: ", len(processed_data))
    print("Skipped questions: ", len(skipped_questions))
    # Save processed data
    print(f"Saving {len(processed_data)} processed examples...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly split into train, val, test
    random.shuffle(processed_data)
    train_data = processed_data[:int(len(processed_data)*0.9)]
    val_data = processed_data[int(len(processed_data)*0.9):int(len(processed_data)*0.95)]
    test_data = processed_data[int(len(processed_data)*0.95):]
    
    torch.save(train_data, os.path.join(output_dir, 'train.pt'))
    torch.save(val_data, os.path.join(output_dir, 'val.pt'))
    torch.save(test_data, os.path.join(output_dir, 'test.pt'))
    
    print(f"Saved {len(train_data)} train examples to {os.path.join(output_dir, 'train.pt')}")
    print(f"Saved {len(val_data)} val examples to {os.path.join(output_dir, 'val.pt')}")
    print(f"Saved {len(test_data)} test examples to {os.path.join(output_dir, 'test.pt')}")
    
    # Save a few examples for inspection
    example_file = os.path.join(output_dir, 'examples.json')
    with open(example_file, 'w') as f:
        json.dump(processed_data[:5], f, indent=2, default=str)
    print(f"Saved 5 examples to {example_file} for inspection")
    
    
if __name__ == "__main__":
    main()
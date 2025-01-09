import json
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os

class HotpotQAProcessor:
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

    def process_data(self, hotpot_path, triples_path, output_dir):
        """
        Process HotpotQA data and corresponding triples into format for GraphLLM
        Args:
            hotpot_path (str): Path to HotpotQA JSON file
            triples_path (str): Path to triples JSON file
            output_dir (str): Directory to save processed data
        """
        # Load data
        print("Loading data files...")
        with open(hotpot_path, 'r') as f:
            hotpot_data = json.load(f)
        
        with open(triples_path, 'r') as f:
            triples_data = json.load(f)
        
        # Create triples lookup
        print("Creating triples lookup...")
        triples_lookup = {item['text']: item['generated_triple'].split('), ') 
                         for item in triples_data}
        
        processed_data = []
        print("Processing examples...")
        for item in tqdm(hotpot_data):
            try:
                graphs = []
                doc_triples = []
                
                # Process each supporting document
                for title, texts in item['supporting_docs'].items():
                    for text in texts:
                        if text in triples_lookup:
                            # Each triple already includes the full format (S> ... | P> ... | O> ...)
                            triples = triples_lookup[text]
                            # Create graph for this document
                            graph, orig_triples = self.create_graph_from_triples(triples)
                            graphs.append(graph)
                            
                            # Format triples for this document
                            doc_triples.append(self.format_document_triples(orig_triples))
                
                if not graphs:  # Skip if no graphs were created
                    continue
                    
                processed_item = {
                    'id': len(processed_data),
                    'question': item['question'],
                    'graphs': graphs,
                    'desc': "Retrieved Graph Information:\n" + "\n\n".join(doc_triples).replace(") ((", "), ("),  # Join all documents' triples
                    'label': item['answer']
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"Error processing item {item['question']}: {e}")
                continue
        
        # Save processed data
        print(f"Saving {len(processed_data)} processed examples...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as PyTorch files
        for split in ['train', 'val', 'test']:
            if split == 'train':
                data = processed_data[:int(len(processed_data)*0.8)]
            elif split == 'val':
                data = processed_data[int(len(processed_data)*0.8):int(len(processed_data)*0.9)]
            else:
                data = processed_data[int(len(processed_data)*0.9):]
                
            torch.save(data, os.path.join(output_dir, f'{split}.pt'))
            print(f"Saved {len(data)} examples to {split}.pt")
        
        # Save a few examples for inspection
        example_file = os.path.join(output_dir, 'examples.json')
        with open(example_file, 'w') as f:
            json.dump(processed_data[:5], f, indent=2, default=str)
        print(f"Saved 5 examples to {example_file} for inspection")
        
        return processed_data

# Usage example:
if __name__ == "__main__":
    processor = HotpotQAProcessor()
    processed_data = processor.process_data(
        # hotpot_path="/shared/eng/pj20/firas_data/datasets/hotpotqa/processed_hotpot.json",
        hotpot_path="/shared/eng/pj20/firas_data/datasets/hotpotqa/filtered/hotpot_filtered.json",
        triples_path="/shared/eng/pj20/firas_data/graph_data/hotpotqa/text_triples.json",
        output_dir="/shared/eng/pj20/firas_data/inference_model/hotpotqa_train"
    )
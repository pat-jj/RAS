import json
from sentence_transformers import SentenceTransformer
import torch
import re
from tqdm import tqdm

def extract_triples(triple_str):
    """Extract subject, predicate, object from triple string."""
    pattern = r'\(S> (.*?)\| P> (.*?)\| O> (.*?)\)'
    triples = []
    matches = re.finditer(pattern, triple_str)
    for match in matches:
        triples.append({
            'subject': match.group(1).strip(),
            'predicate': match.group(2).strip(),
            'object': match.group(3).strip()
        })
    return triples

def process_doc_triples(doc2triples_path):
    """Process doc2triples.json and extract unique nodes and edges."""
    with open(doc2triples_path, 'r') as f:
        doc_triples = json.load(f)
    
    nodes = set()
    edges = set()
    
    print(f"Processing {len(doc_triples)} documents")
    for item in tqdm(doc_triples):
        triples = extract_triples(item['generated_triple'])
        for triple in triples:
            nodes.add(triple['subject'])
            nodes.add(triple['object'])
            edges.add(triple['predicate'])
    
    return list(nodes), list(edges)

def generate_embeddings(texts, model_name='sentence-transformers/all-roberta-large-v1'):
    """Generate embeddings for a list of texts using RoBERTa."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def main():
    # Process triples and get unique nodes and edges
    print("Processing triples")
    nodes, edges = process_doc_triples('/shared/eng/pj20/firas_data/datasets/hotpotqa/doc2triples.json')
    
    # Generate embeddings
    print(f"Generating embeddings for {len(nodes)} nodes")
    node_embeddings = generate_embeddings(nodes)
    print(f"Generating embeddings for {len(edges)} edges")
    edge_embeddings = generate_embeddings(edges)
    
    # Save embeddings (dim: 1024)
    torch.save({
        'nodes': nodes,
        'edges': edges,
        'node_embeddings': node_embeddings,
        'edge_embeddings': edge_embeddings
    }, '/shared/eng/pj20/firas_data/datasets/hotpotqa/graph_embeddings.pt')

if __name__ == "__main__":
    main()
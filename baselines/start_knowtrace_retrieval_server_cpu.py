#!/usr/bin/env python3
"""
Start KnowTrace retrieval server on CPU
"""

import sys
import os
import argparse

# Add KnowTrace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'KnowTrace'))

# Force CPU usage before importing torch-dependent modules
import torch
# Monkey patch to force CPU
original_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False

# Patch .cuda() method to return CPU model
original_cuda = torch.nn.Module.cuda
def cpu_cuda(self, device=None):
    """Force .cuda() to return CPU model"""
    return self.cpu()
torch.nn.Module.cuda = cpu_cuda

from retriever.retrieval_server import Retriever, app
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Start KnowTrace retrieval server on CPU")
    parser.add_argument('--corpus_path', type=str, 
                       default='/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018/wiki_corpus.json',
                       help='Path to corpus JSON file')
    parser.add_argument('--retrieval_method', type=str, default='contriever',
                       choices=['bm25', 'contriever', 'dpr'],
                       help='Retrieval method')
    parser.add_argument('--port', type=int, default=8001,
                       help='Port for retrieval server')
    parser.add_argument('--data_name', type=str, default='firas',
                       help='Data name for indexing')
    parser.add_argument('--topk', type=int, default=5,
                       help='Number of top results to return')
    
    args = parser.parse_args()
    
    # Check if corpus exists
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus file not found at {args.corpus_path}")
        print("Please check the path or create the corpus file first")
        sys.exit(1)
    
    print(f"Starting KnowTrace retrieval server on CPU...")
    print(f"Corpus: {args.corpus_path}")
    print(f"Method: {args.retrieval_method}")
    print(f"Port: {args.port}")
    
    # Create retriever
    retriever = Retriever(
        data_name=args.data_name,
        topk=args.topk,
        corpus_path=args.corpus_path,
        retrieval_method=args.retrieval_method,
        index_init=False
    )
    
    # Force CPU for contriever/dpr if used
    # Note: The retriever is created in Retriever.__init__, so we need to patch it after creation
    # For contriever, the model is loaded with .cuda() in the original code
    # We'll need to move it to CPU after creation
    if args.retrieval_method in ['contriever', 'dpr']:
        # The model is stored in retriever.model
        # For contriever, it's a FIFS object with query_encoder and doc_encoder
        if hasattr(retriever, 'model'):
            model_obj = retriever.model
            # Check if it's a FIFS (FlatIPFaissSearch) object
            if hasattr(model_obj, 'query_encoder'):
                if hasattr(model_obj.query_encoder, 'cuda'):
                    model_obj.query_encoder = model_obj.query_encoder.cpu()
                if hasattr(model_obj.query_encoder, 'eval'):
                    model_obj.query_encoder.eval()
            if hasattr(model_obj, 'doc_encoder'):
                if hasattr(model_obj.doc_encoder, 'cuda'):
                    model_obj.doc_encoder = model_obj.doc_encoder.cpu()
                if hasattr(model_obj.doc_encoder, 'eval'):
                    model_obj.doc_encoder.eval()
            # For DPR/SentenceBERT - check for q_model and doc_model
            if hasattr(model_obj, 'q_model'):
                if hasattr(model_obj.q_model, 'cuda'):
                    model_obj.q_model = model_obj.q_model.cpu()
                if hasattr(model_obj.q_model, 'eval'):
                    model_obj.q_model.eval()
            if hasattr(model_obj, 'doc_model'):
                if hasattr(model_obj.doc_model, 'cuda'):
                    model_obj.doc_model = model_obj.doc_model.cpu()
                if hasattr(model_obj.doc_model, 'eval'):
                    model_obj.doc_model.eval()
    
    # Set global retriever for the FastAPI app
    import retriever.retrieval_server as rs
    rs.retriever = retriever
    
    print(f"Retrieval server ready on port {args.port}")
    print("Press Ctrl+C to stop")
    
    # Run server
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)

if __name__ == "__main__":
    main()


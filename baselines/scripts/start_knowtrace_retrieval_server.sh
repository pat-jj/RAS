#!/bin/bash

# Script to start KnowTrace retrieval server on CPU

# Configuration
CORPUS_PATH="/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018/wiki_corpus.json"
RETRIEVAL_METHOD="contriever"  # or "bm25" or "dpr"
PORT=8001
DATA_NAME="firas"

# Check if corpus file exists
if [ ! -f "$CORPUS_PATH" ]; then
    echo "Error: Corpus file not found at $CORPUS_PATH"
    echo "Please check the path or create the corpus file first"
    exit 1
fi

echo "Starting KnowTrace retrieval server..."
echo "Corpus: $CORPUS_PATH"
echo "Method: $RETRIEVAL_METHOD"
echo "Port: $PORT"

# Change to KnowTrace directory
cd /home/pj20/server-04/FIRAS/KnowTrace

# Start the retrieval server
# For CPU usage, we'll modify the retrieval_server.py to use CPU
python -c "
import sys
import os
sys.path.insert(0, '/home/pj20/server-04/FIRAS/KnowTrace')

# Patch to force CPU usage
import torch
torch.cuda.is_available = lambda: False

from retriever.retrieval_server import Retriever, app
import uvicorn

# Create retriever with CPU
retriever = Retriever(
    data_name='$DATA_NAME',
    topk=5,
    corpus_path='$CORPUS_PATH',
    retrieval_method='$RETRIEVAL_METHOD',
    index_init=False
)

# Force CPU for contriever if used
if '$RETRIEVAL_METHOD' == 'contriever' and hasattr(retriever, 'model'):
    if hasattr(retriever.model, 'query_encoder'):
        retriever.model.query_encoder = retriever.model.query_encoder.cpu()
    if hasattr(retriever.model, 'doc_encoder'):
        retriever.model.doc_encoder = retriever.model.doc_encoder.cpu()

# Set global retriever
import retriever.retrieval_server as rs
rs.retriever = retriever

print('Retrieval server ready on port $PORT')
uvicorn.run(app, host='0.0.0.0', port=$PORT, workers=1)
" 2>&1


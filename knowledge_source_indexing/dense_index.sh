#!/bin/bash

# Exit on error
set -e

# Set visible GPU devices
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7

# Check for required Python packages
echo "Checking required packages..."
python -c "
import sys
try:
    import faiss
    import torch
    import transformers
    print('✓ All required packages are installed')
except ImportError as e:
    print(f'✗ Missing package: {str(e)}')
    print('\nPlease install missing packages using either:')
    print('conda install -c pytorch faiss-gpu')
    print('or')
    print('pip install --user faiss-gpu')
    sys.exit(1)
"

# Define Wikipedia versions and their paths
declare -A WIKI_VERSIONS=(
    ["2017"]="/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/all_wiki_text.json:/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding"
    ["2018"]="/shared/eng/pj20/firas_data/knowledge_source/wiki_2018/all_wiki_text.json:/shared/eng/pj20/firas_data/knowledge_source/wiki_2018/embedding"
    ["2020"]="/shared/eng/pj20/firas_data/knowledge_source/wiki_2020/all_wiki_text.json:/shared/eng/pj20/firas_data/knowledge_source/wiki_2020/embedding"
)

# Calculate number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
BATCH_SIZE=256

# Function to check if process completed successfully
check_success() {
    if [ $? -eq 0 ]; then
        echo "✓ Processing completed successfully for Wikipedia $1"
        return 0
    else
        echo "✗ Error processing Wikipedia $1"
        return 1
    fi
}

# Function to process one Wikipedia version
process_wiki_version() {
    local version=$1
    local paths=${WIKI_VERSIONS[$version]}
    local input_file=$(echo $paths | cut -d: -f1)
    local output_dir=$(echo $paths | cut -d: -f2)
    
    echo "========================================="
    echo "Processing Wikipedia $version"
    echo "Input: $input_file"
    echo "Output: $output_dir"
    echo "Using $NUM_GPUS GPUs with batch size $BATCH_SIZE per GPU"
    echo "========================================="
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Clear CUDA cache
    python -c "import torch; torch.cuda.empty_cache()"
    
    # Run the distributed training
    MASTER_PORT=$(shuf -i 29500-29999 -n 1)
    echo "Starting distributed training with $NUM_GPUS GPUs on port $MASTER_PORT..."
    
    PYTHONPATH="$PYTHONPATH:$(pwd)" torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=localhost \
        --master_port=$MASTER_PORT \
        --node_rank=0 \
        --nnodes=1 \
        dense_indexing.py \
        --input_file "$input_file" \
        --output_dir "$output_dir" \
        --batch_size $BATCH_SIZE
    
    return $?
}

# Process each Wikipedia version sequentially
for version in "2017" "2018" "2020"; do
    echo "Starting processing of Wikipedia $version"
    
    # Check if output already exists
    output_dir=$(echo ${WIKI_VERSIONS[$version]} | cut -d: -f2)
    if [ -f "$output_dir/wikipedia_embeddings.faiss" ]; then
        echo "⚠️  Wikipedia $version already processed (found existing FAISS index). Skipping..."
        continue
    fi
    
    # Process this version
    process_wiki_version $version
    
    if ! check_success $version; then
        echo "Stopping due to error in Wikipedia $version processing"
        exit 1
    fi
    
    # Wait for a bit between versions to allow memory to clear naturally
    echo "Waiting 60 seconds before starting next version..."
    sleep 60
done

echo "All Wikipedia versions processed successfully!"
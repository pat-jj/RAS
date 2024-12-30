#!/bin/bash

# Set visible GPU devices
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7

# Input and output paths
INPUT_FILE="/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/all_wiki_text.json"
OUTPUT_DIR="/shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding"

# INPUT_FILE="/shared/eng/pj20/firas_data/knowledge_source/wiki_2018/all_wiki_text.json"
# OUTPUT_DIR="/shared/eng/pj20/firas_data/knowledge_source/wiki_2018/embedding"

# INPUT_FILE="/shared/eng/pj20/firas_data/knowledge_source/wiki_2020/all_wiki_text.json"
# OUTPUT_DIR="/shared/eng/pj20/firas_data/knowledge_source/wiki_2020/embedding"

# Calculate number of GPUs from CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Set batch size per GPU (adjust based on your GPU memory)
# Using a moderate batch size that should work with most GPUs
BATCH_SIZE=32

echo "Starting Contriever encoder with $NUM_GPUS GPUs"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"

# Run the distributed training script
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$(shuf -i 29500-29999 -n 1) \
    dense_indexing.py \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE
#!/bin/bash
# Script to run graph vs text comparison experiments

source /home/pj20/miniconda3/etc/profile.d/conda.sh
conda activate handbook

cd /home/pj20/server-04/FIRAS/framework

# Set to use CPU for retriever
export CUDA_VISIBLE_DEVICES=""

# Run comparison on different datasets
# Start with 2wikimultihop (multi-hop reasoning - should show graph benefits)
echo "Running comparison on 2wikimultihop (multi-hop reasoning)..."
python compare_graph_vs_text.py \
    --dataset 2wikimultihop \
    --num_samples 15 \
    --max_iteration 3 \
    --max_answer_length 100 \
    --use_cpu_retriever \
    --output_dir ./comparison_results \
    --debug

echo ""
echo "=========================================="
echo "Comparison completed!"
echo "Check results in ./comparison_results/"
echo "=========================================="


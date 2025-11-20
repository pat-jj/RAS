#!/bin/bash
# Simple example script to test Qwen3-8B inference

cd /home/pj20/server-04/FIRAS/framework

source /home/pj20/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# Test with a simple question
echo "Testing with simple question..."
python test_qwen_inference.py \
    --checkpoint /shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen/checkpoint_10_of_20_qwen.safetensors \
    --question "What is the capital of France?" \
    --max_iterations 3

echo ""
echo "Testing with a more complex question..."
python test_qwen_inference.py \
    --checkpoint /shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen/checkpoint_10_of_20_qwen.safetensors \
    --question "Who wrote the novel '1984' and what is it about?" \
    --max_iterations 3

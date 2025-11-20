#!/bin/bash

# Example script to run local LLM baselines (Qwen3-8B-Instruct and LLaMA3-8B-Instruct)
# Both with and without retrieval

# ============================================
# Qwen3-8B-Instruct Baselines
# ============================================

# Qwen3-8B-Instruct WITHOUT retrieval
echo "Running Qwen3-8B-Instruct without retrieval on TriviaQA..."
python local_llm_test.py \
    --input_file /shared/rsaas/pj20/firas_data/test_datasets/original/triviaqa.jsonl \
    --mode vanilla \
    --task qa \
    --prompt_name prompt_no_input \
    --max_new_tokens 100 \
    --metric match \
    --result_fp /shared/rsaas/pj20/firas_data/test_datasets/results/triviaqa_qwen3_8b_base.jsonl \
    --model_name Qwen/Qwen3-8B \
    --batch_size 4 \
    --device cuda

# Qwen3-8B-Instruct WITH retrieval
echo "Running Qwen3-8B-Instruct with retrieval on TriviaQA..."
python local_llm_test.py \
    --input_file /shared/rsaas/pj20/firas_data/test_datasets/original/triviaqa.jsonl \
    --retrieval_file /shared/rsaas/pj20/firas_data/test_datasets/retrieval/triviaqa_dense_ret.jsonl \
    --mode retrieval \
    --top_n 5 \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 100 \
    --metric match \
    --result_fp /shared/rsaas/pj20/firas_data/test_datasets/results/triviaqa_qwen3_8b_retrieval_top_5.jsonl \
    --model_name Qwen/Qwen3-8B \
    --batch_size 4 \
    --device cuda

# ============================================
# LLaMA3-8B-Instruct Baselines
# ============================================

# LLaMA3-8B-Instruct WITHOUT retrieval
echo "Running LLaMA3-8B-Instruct without retrieval on TriviaQA..."
python local_llm_test.py \
    --input_file /shared/rsaas/pj20/firas_data/test_datasets/original/triviaqa.jsonl \
    --mode vanilla \
    --task qa \
    --prompt_name prompt_no_input \
    --max_new_tokens 100 \
    --metric match \
    --result_fp /shared/rsaas/pj20/firas_data/test_datasets/results/triviaqa_llama3_8b_base.jsonl \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 4 \
    --device cuda

# LLaMA3-8B-Instruct WITH retrieval
echo "Running LLaMA3-8B-Instruct with retrieval on TriviaQA..."
python local_llm_test.py \
    --input_file /shared/rsaas/pj20/firas_data/test_datasets/original/triviaqa.jsonl \
    --retrieval_file /shared/rsaas/pj20/firas_data/test_datasets/retrieval/triviaqa_dense_ret.jsonl \
    --mode retrieval \
    --top_n 5 \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 100 \
    --metric match \
    --result_fp /shared/rsaas/pj20/firas_data/test_datasets/results/triviaqa_llama3_8b_retrieval_top_5.jsonl \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 4 \
    --device cuda

# ============================================
# More Examples
# ============================================

# Qwen3-8B-Instruct on PubHealth (Fact Verification)
# echo "Running Qwen3-8B-Instruct on PubHealth..."
# python local_llm_test.py \
#     --input_file /shared/rsaas/pj20/firas_data/test_datasets/original/pubhealth.jsonl \
#     --mode retrieval \
#     --top_n 1 \
#     --task fever \
#     --prompt_name prompt_no_input_retrieval \
#     --max_new_tokens 50 \
#     --metric match \
#     --result_fp /shared/rsaas/pj20/firas_data/test_datasets/results/pubhealth_qwen3_8b_retrieval_top_1.jsonl \
#     --model_name Qwen/Qwen3-8B \
#     --batch_size 4 \
#     --device cuda

# LLaMA3-8B-Instruct on ARC-C (Multiple Choice)
# echo "Running LLaMA3-8B-Instruct on ARC-C..."
# python local_llm_test.py \
#     --input_file /shared/rsaas/pj20/firas_data/test_datasets/original/arc_c.jsonl \
#     --mode vanilla \
#     --task arc_c \
#     --prompt_name prompt_no_input \
#     --max_new_tokens 10 \
#     --metric match \
#     --result_fp /shared/rsaas/pj20/firas_data/test_datasets/results/arc_c_llama3_8b_base.jsonl \
#     --model_name meta-llama/Meta-Llama-3-8B-Instruct \
#     --batch_size 4 \
#     --device cuda

# Qwen3-8B-Instruct on ELI5 (Long-form QA)
# echo "Running Qwen3-8B-Instruct on ELI5..."
# python local_llm_test.py \
#     --input_file /shared/rsaas/pj20/firas_data/test_datasets/eli5_test.json \
#     --retrieval_file /shared/rsaas/pj20/firas_data/test_datasets/retrieval/eli5_dense_ret.jsonl \
#     --mode eli5_ret \
#     --top_n 5 \
#     --task eli5_ret \
#     --prompt_name eli5_ret \
#     --max_new_tokens 300 \
#     --metric match \
#     --result_fp /shared/rsaas/pj20/firas_data/test_datasets/results/eli5_qwen3_8b_retrieval_top_5.jsonl \
#     --model_name Qwen/Qwen3-8B \
#     --batch_size 2 \
#     --device cuda


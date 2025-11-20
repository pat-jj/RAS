#!/bin/bash

# Test script to run a small subset of experiments first
# This tests one model on one dataset to verify everything works

DATA_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
RESULTS_DIR="${DATA_DIR}/results"
RETRIEVAL_DIR="${DATA_DIR}/retrieval"
ORIGINAL_DIR="${DATA_DIR}/original"

# Test with Qwen3 on TriviaQA (first 10 samples)
echo "Testing Qwen3-8B on TriviaQA (small subset)..."

# First, create a small test file
TEST_FILE="${ORIGINAL_DIR}/triviaqa_test_small.jsonl"
head -10 "${ORIGINAL_DIR}/triviaqa.jsonl" > "$TEST_FILE"

# Run without retrieval
python local_llm_test.py \
    --input_file "$TEST_FILE" \
    --mode vanilla \
    --task qa \
    --prompt_name prompt_no_input \
    --max_new_tokens 100 \
    --metric match \
    --result_fp "${RESULTS_DIR}/test_triviaqa_qwen3_8b_base.jsonl" \
    --model_name Qwen/Qwen3-8B \
    --batch_size 2 \
    --device cuda

echo "Test completed. Check results at: ${RESULTS_DIR}/test_triviaqa_qwen3_8b_base.jsonl"


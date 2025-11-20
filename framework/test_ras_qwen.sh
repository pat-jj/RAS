#!/bin/bash
# Script to test RAS performance with Qwen3-8B planner

source /home/pj20/miniconda3/etc/profile.d/conda.sh
export WANDB_API_KEY="fcb8fe58088a42fc17b32841ed54c2785d670a66"

cd /home/pj20/server-04/FIRAS/framework
conda activate handbook

# Find latest Qwen checkpoint
CHECKPOINT_DIR="/shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen"
LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/checkpoint_*_qwen.safetensors 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No Qwen checkpoint found!"
    exit 1
fi

echo "✅ Using Qwen checkpoint: $LATEST_CHECKPOINT"
ls -lh "$LATEST_CHECKPOINT"
echo ""

# Datasets to test (start with a few for testing)
DATASETS=("triviaqa" "popqa" "2wikimultihop")

# Knowledge sources mapping
declare -A KNOWLEDGE_SOURCES=(
    ["triviaqa"]="wiki_2018"
    ["popqa"]="wiki_2020"
    ["2wikimultihop"]="wiki_2018"
)

# Knowledge paths
declare -A KNOWLEDGE_PATHS=(
    ["wiki_2018"]="/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018"
    ["wiki_2020"]="/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2020"
)

# Max answer lengths
declare -A MAX_ANSWER_LENGTHS=(
    ["triviaqa"]="100"
    ["popqa"]="100"
    ["2wikimultihop"]="100"
)

# Test each dataset
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Testing dataset: $dataset with Qwen3-8B"
    echo "=========================================="
    
    knowledge_source=${KNOWLEDGE_SOURCES[$dataset]}
    knowledge_path=${KNOWLEDGE_PATHS[$knowledge_source]}
    max_answer_length=${MAX_ANSWER_LENGTHS[$dataset]}
    
    python run_ras.py \
        --dataset "$dataset" \
        --test_data_path /shared/rsaas/pj20/firas_data/test_datasets \
        --knowledge_source "$knowledge_source" \
        --knowledge_path "$knowledge_path" \
        --text_to_triples_model sonnet \
        --planner_model qwen3-8b \
        --planner_frozen False \
        --planner_checkpoint "$LATEST_CHECKPOINT" \
        --llm_model_path Qwen/Qwen3-8B \
        --answerer_model qwen3-8b \
        --answerer_frozen False \
        --answerer_checkpoint "$LATEST_CHECKPOINT" \
        --max_answer_length "$max_answer_length" \
        --max_new_tokens "$max_answer_length" \
        --max_iteration 3
    
    echo ""
    echo "✅ Completed $dataset"
    
    # Evaluate results
    OUTPUT_FILE="/shared/rsaas/pj20/firas_data/test_datasets/${dataset}_test_output_qwen3_8b_qwen3_8b_v3.json"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "📊 Evaluating $dataset..."
        cd /home/pj20/server-04/FIRAS/baselines
        python evaluate_results.py "$OUTPUT_FILE" --dataset "$dataset" 2>&1 | tail -10
        cd /home/pj20/server-04/FIRAS/framework
    fi
    echo ""
done

echo "=========================================="
echo "✅ All datasets tested with Qwen3-8B!"
echo "=========================================="


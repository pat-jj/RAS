#!/bin/bash
# Script to test RAS performance across all datasets

source /home/pj20/miniconda3/etc/profile.d/conda.sh
export WANDB_API_KEY="fcb8fe58088a42fc17b32841ed54c2785d670a66"

cd /home/pj20/server-04/FIRAS/framework
conda activate handbook

# Find latest checkpoint - prefer checkpoint_20_of_20 (final checkpoint)
CHECKPOINT_DIR="/shared/rsaas/pj20/firas_data/multitask/checkpoints_20"
LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/checkpoint_20_of_20*.safetensors 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    # Fallback to latest_checkpoint
    LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/latest_checkpoint.safetensors 2>/dev/null | head -1)
fi

if [ -z "$LATEST_CHECKPOINT" ]; then
    # Fallback to other checkpoint locations
    LATEST_CHECKPOINT=$(ls -t /shared/rsaas/pj20/firas_data/multitask/checkpoints*/latest_checkpoint.safetensors 2>/dev/null | head -1)
fi

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoint found!"
    exit 1
fi

echo "✅ Using checkpoint: $LATEST_CHECKPOINT"
ls -lh "$LATEST_CHECKPOINT"
echo ""

# Datasets to test
DATASETS=("triviaqa" "popqa" "pubhealth" "arc_c" "bio" "asqa" "eli5" "2wikimultihop")

# Knowledge sources mapping
declare -A KNOWLEDGE_SOURCES=(
    ["triviaqa"]="wiki_2018"
    ["popqa"]="wiki_2020"
    ["pubhealth"]="wiki_2018"
    ["arc_c"]="wiki_2018"
    ["bio"]="wiki_2018"
    ["asqa"]="wiki_2018"
    ["eli5"]="wiki_2018"
    ["2wikimultihop"]="wiki_2018"
)

# Knowledge paths
declare -A KNOWLEDGE_PATHS=(
    ["wiki_2018"]="/shared/eng/pj20/firas_data/knowledge_source/wiki_2018"
    ["wiki_2020"]="/shared/eng/pj20/firas_data/knowledge_source/wiki_2020"
)

# Max answer lengths
declare -A MAX_ANSWER_LENGTHS=(
    ["triviaqa"]="100"
    ["popqa"]="100"
    ["pubhealth"]="50"
    ["arc_c"]="50"
    ["bio"]="100"
    ["asqa"]="300"
    ["eli5"]="300"
    ["2wikimultihop"]="100"
)

# Test each dataset
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Testing dataset: $dataset"
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
        --planner_model llama2-7b \
        --planner_frozen False \
        --planner_checkpoint "$LATEST_CHECKPOINT" \
        --answerer_model sonnet \
        --max_answer_length "$max_answer_length" \
        --max_iteration 3
    
    echo ""
    echo "✅ Completed $dataset"
    
    # Evaluate results
    OUTPUT_FILE="/shared/rsaas/pj20/firas_data/test_datasets/${dataset}_test_output_llama2-7b_sonnet_v3.json"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "📊 Evaluating $dataset..."
        cd /home/pj20/server-04/FIRAS/baselines
        python evaluate_results.py "$OUTPUT_FILE" 2>&1 | tail -5
        cd /home/pj20/server-04/FIRAS/framework
    fi
    echo ""
done

echo "=========================================="
echo "✅ All datasets tested!"
echo "=========================================="
echo ""
echo "📊 Generating summary..."
cd /home/pj20/server-04/FIRAS/baselines
python generate_results_csv.py 2>&1 | tail -20


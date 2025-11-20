#!/bin/bash
# Script to run RAS evaluation with no-GNN Qwen3-8B model

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source /home/pj20/miniconda3/etc/profile.d/conda.sh
conda activate trl

cd /home/pj20/server-04/FIRAS/framework

# Checkpoint path (will auto-detect latest checkpoint)
CHECKPOINT_PATH="/shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen_no_gnn"

# Test datasets - you can modify this list
# Available: triviaqa, popqa, arc_c, pubhealth, 2wikimultihop, bio, asqa, eli5
DATASETS=("triviaqa" "popqa" "arc_c" "pubhealth" "2wikimultihop" "bio")

# Knowledge base path (adjust based on dataset)
KNOWLEDGE_PATH="/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018"

# Output CSV
RESULTS_CSV="/home/pj20/server-04/FIRAS/baselines/logs/results_summary.csv"

echo "Starting RAS evaluation with no-GNN Qwen3-8B..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Datasets: ${DATASETS[@]}"

# Output directory for results
OUTPUT_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
MODEL_NAME="qwen3_8b_no_gnn"
MODE="base"

# Run evaluation for each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating dataset: $dataset"
    echo "=========================================="
    
    # Check for existing results for the CURRENT checkpoint
    # First, detect which checkpoint will be used
    CHECKPOINT_STEP=$(python3 << 'PYEOF'
import os
checkpoint_path = "$CHECKPOINT_PATH"
if os.path.isdir(checkpoint_path):
    checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith('checkpoint-')]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
        try:
            step = int(latest.split('-')[1])
            print(step)
        except:
            pass
PYEOF
)
    
    # Build expected output filename for current checkpoint
    if [ -n "$CHECKPOINT_STEP" ]; then
        EXPECTED_FILE="${OUTPUT_DIR}/${dataset}_test_output_${MODEL_NAME}_${MODE}_step${CHECKPOINT_STEP}.json"
    else
        EXPECTED_FILE="${OUTPUT_DIR}/${dataset}_test_output_${MODEL_NAME}_${MODE}.json"
    fi
    
    # Check if results for THIS checkpoint already exist
    if [ -f "$EXPECTED_FILE" ]; then
        # Check if file has valid content
        if python3 -c "import json; data=json.load(open('$EXPECTED_FILE')); print('valid' if 'output' in data and len(data.get('output', [])) > 0 else 'invalid')" 2>/dev/null | grep -q "valid"; then
            echo "✅ Results already exist for $dataset (checkpoint step ${CHECKPOINT_STEP:-'unknown'}): $EXPECTED_FILE"
            echo "   Skipping evaluation..."
            continue
        else
            echo "⚠️  Results file exists but appears incomplete, re-evaluating..."
        fi
    else
        echo "📝 No existing results for checkpoint step ${CHECKPOINT_STEP:-'unknown'}, will evaluate..."
    fi
    
    # Adjust knowledge path based on dataset
    if [ "$dataset" == "popqa" ]; then
        KNOWLEDGE_PATH="/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2020"
    else
        KNOWLEDGE_PATH="/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018"
    fi
    
    echo "Running evaluation for $dataset..."
    python test_ras_no_gnn.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --dataset "$dataset" \
        --knowledge_path "$KNOWLEDGE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --results_csv "$RESULTS_CSV" \
        --max_iteration 3 \
        --max_answer_length 100 \
        --model_name "$MODEL_NAME" \
        --mode "$MODE"
    
    if [ $? -ne 0 ]; then
        echo "❌ Error evaluating $dataset"
        exit 1
    fi
    
    echo "✅ Completed evaluation for $dataset"
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: $RESULTS_CSV"
echo "=========================================="


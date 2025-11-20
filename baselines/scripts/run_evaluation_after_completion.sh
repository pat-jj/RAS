#!/bin/bash
# Script to run evaluation on all completed results after experiments finish

# Base paths
DATA_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
RESULTS_DIR="${DATA_DIR}/results"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate firas_baselines

cd /home/pj20/server-04/FIRAS/baselines

echo "=========================================="
echo "Running Evaluation on All Completed Results"
echo "=========================================="
echo ""

# Wait for processes to complete
echo "Waiting for experiments to complete..."
while ps aux | grep -q "local_llm_test.py" | grep -v grep; do
    sleep 60
    echo "$(date): Still waiting for processes to complete..."
done

echo "All processes completed. Starting evaluation..."
echo ""

# Datasets and their metrics
declare -A DATASET_METRICS
DATASET_METRICS["triviaqa"]="f1"
DATASET_METRICS["popqa"]="f1"
DATASET_METRICS["pubhealth"]="match"
DATASET_METRICS["arc_c"]="match"
DATASET_METRICS["bio"]="factscore"
DATASET_METRICS["asqa"]="rouge"
DATASET_METRICS["eli5"]="rouge"

# Models
MODELS=("qwen3_8b" "llama3_8b")

# Modes
MODES=("base" "retrieval_top_5")

# Evaluate all result files
for model in "${MODELS[@]}"; do
    for dataset in "${!DATASET_METRICS[@]}"; do
        for mode in "${MODES[@]}"; do
            # Skip arc_c retrieval
            if [ "$dataset" = "arc_c" ] && [ "$mode" = "retrieval_top_5" ]; then
                continue
            fi
            
            result_file="${RESULTS_DIR}/${dataset}_${model}_${mode}.jsonl"
            
            if [ -f "$result_file" ]; then
                echo "=========================================="
                echo "Evaluating: ${dataset}_${model}_${mode}"
                echo "=========================================="
                
                metric="${DATASET_METRICS[$dataset]}"
                if [ "$metric" = "rouge" ]; then
                    metric="auto"  # Use auto for long-form tasks
                fi
                
                python evaluate_results.py \
                    --result_file "$result_file" \
                    --dataset "$dataset" \
                    --metric "$metric"
                
                echo ""
            else
                echo "Skipping: ${result_file} (not found)"
            fi
        done
    done
done

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="


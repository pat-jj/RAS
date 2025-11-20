#!/bin/bash
DATA_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
RESULTS_DIR="${DATA_DIR}/results"
ORIGINAL_DIR="${DATA_DIR}/original"
KNOWTRACE_DIR="/home/pj20/server-04/FIRAS/baselines"
DATASETS=("triviaqa" "popqa" "pubhealth" "arc_c" "bio" "asqa" "eli5")

run_knowtrace() {
    local model=$1
    local model_name=$2
    local dataset=$3
    local step_num=${4:-5}
    
    local input_file=""
    if [ "$dataset" = "asqa" ]; then
        input_file="${ORIGINAL_DIR}/${dataset}.json"
    elif [ "$dataset" = "eli5" ]; then
        input_file="${DATA_DIR}/${dataset}_test.json"
    else
        input_file="${ORIGINAL_DIR}/${dataset}.jsonl"
    fi
    
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file not found: $input_file"
        return 1
    fi
    
    local output_file="${RESULTS_DIR}/${dataset}_${model}_knowtrace.jsonl"
    
    local cmd="python knowtrace_test.py \
        --input_file \"${input_file}\" \
        --result_fp \"${output_file}\" \
        --base_llm ${model_name} \
        --step_num ${step_num} \
        --dataset ${dataset} \
        --local_llm_port 1051 \
        --max_items 800 \
        --num_splits 1"
    
    echo "Running: KnowTrace with ${model} on ${dataset}"
    echo "Command: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: KnowTrace with ${model} on ${dataset}"
        echo ""
        
        echo "Evaluating results..."
        python evaluate_results.py \
            --result_file "${output_file}" \
            --dataset ${dataset} \
            --metric auto
        echo ""
    else
        echo "✗ Failed: KnowTrace with ${model} on ${dataset}"
        echo ""
    fi
}

cd ${KNOWTRACE_DIR}
echo '=========================================='
echo 'Running KnowTrace experiments for LLaMA'
echo '=========================================='
echo ''

for dataset in "${DATASETS[@]}"; do
    echo "--- KnowTrace (LLaMA) on ${dataset} ---"
    run_knowtrace "llama3_8b" "LLaMA3-8B-Instruct" "${dataset}" 5
done

echo ''
echo '=========================================='
echo 'LLaMA experiments completed!'
echo '=========================================='

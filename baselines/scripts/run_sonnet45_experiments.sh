#!/bin/bash
# Script to run Sonnet 4.5 baseline experiments (vanilla, retrieval, and SuRE)

# Base paths
DATA_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
RESULTS_DIR="${DATA_DIR}/results"
ORIGINAL_DIR="${DATA_DIR}/original"

# Model name
MODEL="sonnet4.5"

# Datasets to run (excluding 2wikimultihopqa)
DATASETS=("triviaqa" "popqa" "pubhealth" "arc_c" "bio" "asqa" "eli5")

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate firas_baselines

cd /home/pj20/server-04/FIRAS/baselines

# Set up log file
LOG_DIR="/home/pj20/server-04/FIRAS/baselines/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/${MODEL}_all.log"

echo "=========================================="
echo "Running Sonnet 4.5 Baseline Experiments"
echo "Log file: ${LOG_FILE}"
echo "=========================================="
echo ""

# Function to run vanilla baseline
run_vanilla() {
    local dataset=$1
    local input_file=""
    local task=""
    local prompt_name=""
    local max_tokens=100
    local metric="match"
    
    # Determine input file
    if [ "$dataset" = "asqa" ]; then
        input_file="${ORIGINAL_DIR}/${dataset}.json"
    elif [ "$dataset" = "eli5" ]; then
        input_file="${DATA_DIR}/${dataset}_test.json"
    else
        input_file="${ORIGINAL_DIR}/${dataset}.jsonl"
    fi
    
    # Determine task and prompt
    case $dataset in
        "triviaqa"|"popqa")
            task="qa"
            prompt_name="prompt_no_input"
            ;;
        "pubhealth")
            task="fever"
            prompt_name="prompt_no_input"
            max_tokens=50
            metric="match"
            ;;
        "arc_c")
            task="arc_c"
            prompt_name="prompt_no_input"
            max_tokens=10
            metric="match"
            ;;
        "bio")
            task="factscore"
            prompt_name="prompt_no_input"
            max_tokens=300
            metric="factscore"
            ;;
        "asqa")
            task="asqa_base"
            prompt_name="asqa_base"
            max_tokens=300
            metric="rouge"
            ;;
        "eli5")
            task="eli5_base"
            prompt_name="eli5_base"
            max_tokens=300
            metric="rouge"
            ;;
    esac
    
    local output_file="${RESULTS_DIR}/${dataset}_${MODEL}_base.jsonl"
    
    echo "=========================================="
    echo "Running: ${MODEL} on ${dataset} (vanilla)"
    echo "=========================================="
    echo "$(date): Starting ${MODEL} on ${dataset} (vanilla)" >> ${LOG_FILE}
    
    python sonnet_test.py \
        --input_file "${input_file}" \
        --mode vanilla \
        --task ${task} \
        --prompt_name ${prompt_name} \
        --max_new_tokens ${max_tokens} \
        --metric ${metric} \
        --result_fp "${output_file}" \
        --model_name ${MODEL} \
        --batch_size 8 \
        --num_threads 8 2>&1 | tee -a ${LOG_FILE}
    
    if [ $? -eq 0 ] && [ -f "${output_file}" ]; then
        echo "✓ Completed: ${MODEL} on ${dataset} (vanilla)"
        echo "$(date): ✓ Completed ${MODEL} on ${dataset} (vanilla)" >> ${LOG_FILE}
    else
        echo "✗ Failed: ${MODEL} on ${dataset} (vanilla)"
        echo "$(date): ✗ Failed ${MODEL} on ${dataset} (vanilla)" >> ${LOG_FILE}
    fi
    echo ""
}

# Function to run retrieval baseline
run_retrieval() {
    local dataset=$1
    local input_file=""
    local task=""
    local prompt_name=""
    local max_tokens=100
    local metric="match"
    local top_n=5
    
    # Skip retrieval for arc_c
    if [ "$dataset" = "arc_c" ]; then
        return
    fi
    
    # Determine input file
    if [ "$dataset" = "asqa" ]; then
        input_file="${ORIGINAL_DIR}/${dataset}.json"
    elif [ "$dataset" = "eli5" ]; then
        input_file="${DATA_DIR}/${dataset}_test.json"
    else
        input_file="${ORIGINAL_DIR}/${dataset}.jsonl"
    fi
    
    # Determine task and prompt
    case $dataset in
        "triviaqa"|"popqa")
            task="qa"
            prompt_name="prompt_no_input_retrieval"
            ;;
        "pubhealth")
            task="fever"
            prompt_name="prompt_no_input_retrieval"
            max_tokens=50
            metric="match"
            top_n=1
            ;;
        "bio")
            task="factscore"
            prompt_name="prompt_no_input_retrieval"
            max_tokens=300
            metric="factscore"
            ;;
        "asqa")
            task="asqa_ret"
            prompt_name="asqa_ret"
            max_tokens=300
            metric="rouge"
            ;;
        "eli5")
            task="eli5_ret"
            prompt_name="eli5_ret"
            max_tokens=300
            metric="rouge"
            ;;
    esac
    
    local output_file="${RESULTS_DIR}/${dataset}_${MODEL}_retrieval_top_${top_n}.jsonl"
    
    echo "=========================================="
    echo "Running: ${MODEL} on ${dataset} (retrieval)"
    echo "=========================================="
    echo "$(date): Starting ${MODEL} on ${dataset} (retrieval)" >> ${LOG_FILE}
    
    python sonnet_test.py \
        --input_file "${input_file}" \
        --mode retrieval \
        --task ${task} \
        --prompt_name ${prompt_name} \
        --max_new_tokens ${max_tokens} \
        --metric ${metric} \
        --result_fp "${output_file}" \
        --model_name ${MODEL} \
        --top_n ${top_n} \
        --batch_size 8 \
        --num_threads 8 2>&1 | tee -a ${LOG_FILE}
    
    if [ $? -eq 0 ] && [ -f "${output_file}" ]; then
        echo "✓ Completed: ${MODEL} on ${dataset} (retrieval)"
        echo "$(date): ✓ Completed ${MODEL} on ${dataset} (retrieval)" >> ${LOG_FILE}
    else
        echo "✗ Failed: ${MODEL} on ${dataset} (retrieval)"
        echo "$(date): ✗ Failed ${MODEL} on ${dataset} (retrieval)" >> ${LOG_FILE}
    fi
    echo ""
}

# Function to run SuRE baseline (for ELI5 and ASQA)
run_sure() {
    local dataset=$1
    local input_file=""
    local task=""
    local prompt_name=""
    local max_tokens=300
    local metric="rouge"
    local top_n=5
    
    # SuRE is only for long-form tasks
    if [ "$dataset" != "asqa" ] && [ "$dataset" != "eli5" ]; then
        return
    fi
    
    # Determine input file
    if [ "$dataset" = "asqa" ]; then
        input_file="${ORIGINAL_DIR}/${dataset}.json"
        task="asqa_ret"
        prompt_name="asqa_ret"
    elif [ "$dataset" = "eli5" ]; then
        input_file="${DATA_DIR}/${dataset}_test.json"
        task="eli5_ret"
        prompt_name="eli5_ret"
    fi
    
    local output_file="${RESULTS_DIR}/${dataset}_${MODEL}_retrieval_top_${top_n}_sure.jsonl"
    
    echo "=========================================="
    echo "Running: ${MODEL} on ${dataset} (SuRE)"
    echo "=========================================="
    echo "$(date): Starting ${MODEL} on ${dataset} (SuRE)" >> ${LOG_FILE}
    
    python sonnet_sure.py \
        --input_file "${input_file}" \
        --mode ${task} \
        --task ${task} \
        --prompt_name ${prompt_name} \
        --max_new_tokens ${max_tokens} \
        --metric ${metric} \
        --result_fp "${output_file}" \
        --model_name ${MODEL} \
        --top_n ${top_n} \
        --batch_size 8 \
        --num_threads 8 2>&1 | tee -a ${LOG_FILE}
    
    if [ $? -eq 0 ] && [ -f "${output_file}" ]; then
        echo "✓ Completed: ${MODEL} on ${dataset} (SuRE)"
        echo "$(date): ✓ Completed ${MODEL} on ${dataset} (SuRE)" >> ${LOG_FILE}
    else
        echo "✗ Failed: ${MODEL} on ${dataset} (SuRE)"
        echo "$(date): ✗ Failed ${MODEL} on ${dataset} (SuRE)" >> ${LOG_FILE}
    fi
    echo ""
}

# Run all experiments
for dataset in "${DATASETS[@]}"; do
    # Vanilla baseline
    run_vanilla "$dataset"
    
    # Retrieval baseline
    run_retrieval "$dataset"
    
    # SuRE baseline (only for asqa and eli5)
    run_sure "$dataset"
done

echo "=========================================="
echo "All Sonnet 4.5 experiments completed!"
echo "=========================================="


#!/bin/bash
# Script to run a single experiment configuration

# Base paths
DATA_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
RESULTS_DIR="${DATA_DIR}/results"
ORIGINAL_DIR="${DATA_DIR}/original"

# Models
QWEN_MODEL="Qwen/Qwen3-8B"
LLAMA_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Get arguments
MODE=$1  # "vanilla" or "retrieval"
MODEL=$2  # "qwen3_8b" or "llama3_8b"

if [ "$MODEL" = "qwen3_8b" ]; then
    MODEL_NAME="$QWEN_MODEL"
else
    MODEL_NAME="$LLAMA_MODEL"
fi

# Datasets to run (excluding 2wikimultihopqa)
DATASETS=("triviaqa" "popqa" "pubhealth" "arc_c" "bio" "asqa" "eli5")

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate firas_baselines

cd /home/pj20/server-04/FIRAS/baselines

# Set up log file
LOG_DIR="/home/pj20/server-04/FIRAS/baselines/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/${MODEL}_${MODE}.log"

echo "=========================================="
echo "Running ${MODEL} experiments (${MODE} mode)"
echo "Log file: ${LOG_FILE}"
echo "=========================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    # Skip retrieval for arc_c
    if [ "$MODE" = "retrieval" ] && [ "$dataset" = "arc_c" ]; then
        echo "Skipping retrieval for ${dataset} (multiple choice task)"
        continue
    fi
    
    # Determine input file
    input_file=""
    if [ "$dataset" = "asqa" ]; then
        input_file="${ORIGINAL_DIR}/${dataset}.json"
    elif [ "$dataset" = "eli5" ]; then
        input_file="${DATA_DIR}/${dataset}_test.json"
    else
        input_file="${ORIGINAL_DIR}/${dataset}.jsonl"
    fi
    
    # Determine task and prompt
    task=""
    prompt_name=""
    max_tokens=100
    metric="match"
    
    case $dataset in
        "triviaqa"|"popqa")
            task="qa"
            if [ "$MODE" = "retrieval" ]; then
                prompt_name="prompt_no_input_retrieval"
            else
                prompt_name="prompt_no_input"
            fi
            ;;
        "pubhealth")
            task="fever"
            if [ "$MODE" = "retrieval" ]; then
                prompt_name="prompt_no_input_retrieval"
            else
                prompt_name="prompt_no_input"
            fi
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
            if [ "$MODE" = "retrieval" ]; then
                prompt_name="prompt_no_input_retrieval"
            else
                prompt_name="prompt_no_input"
            fi
            max_tokens=300
            metric="factscore"
            ;;
        "asqa")
            task="asqa_base"
            if [ "$MODE" = "retrieval" ]; then
                task="asqa_ret"
                prompt_name="asqa_ret"
            else
                prompt_name="asqa_base"
            fi
            max_tokens=300
            metric="rouge"
            ;;
        "eli5")
            task="eli5_base"
            if [ "$MODE" = "retrieval" ]; then
                task="eli5_ret"
                prompt_name="eli5_ret"
            else
                prompt_name="eli5_base"
            fi
            max_tokens=300
            metric="rouge"
            ;;
    esac
    
    # Determine output file
    mode_suffix="base"
    if [ "$MODE" = "retrieval" ]; then
        mode_suffix="retrieval_top_5"
    fi
    output_file="${RESULTS_DIR}/${dataset}_${MODEL}_${mode_suffix}.jsonl"
    
    # Select GPU (rotate between 5, 6, 7)
    gpu_id=$((RANDOM % 3 + 5))
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    # Build command
    cmd="python local_llm_test.py \
        --input_file \"${input_file}\" \
        --mode ${MODE} \
        --task ${task} \
        --prompt_name ${prompt_name} \
        --max_new_tokens ${max_tokens} \
        --metric ${metric} \
        --result_fp \"${output_file}\" \
        --model_name ${MODEL_NAME} \
        --batch_size 4 \
        --device cuda"
    
    if [ "$MODE" = "retrieval" ]; then
        cmd="${cmd} --top_n 5"
    fi
    
    echo "=========================================="
    echo "Running: ${MODEL} on ${dataset} (${MODE})"
    echo "GPU: ${gpu_id}"
    echo "=========================================="
    echo "$(date): Starting ${MODEL} on ${dataset} (${MODE})" >> ${LOG_FILE}
    eval $cmd 2>&1 | tee -a ${LOG_FILE}
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${MODEL} on ${dataset} (${MODE})"
        echo "$(date): ✓ Completed ${MODEL} on ${dataset} (${MODE})" >> ${LOG_FILE}
        
        # Evaluate results (only if output file exists and is not empty)
        if [ -n "${output_file}" ] && [ -f "${output_file}" ]; then
            echo "Evaluating results from: ${output_file}"
            python evaluate_results.py \
                --result_file "${output_file}" \
                --dataset ${dataset} \
                --metric auto 2>&1 | tee -a ${LOG_FILE}
        else
            echo "Warning: Output file not found or empty path: '${output_file}', skipping evaluation" | tee -a ${LOG_FILE}
        fi
    else
        echo "✗ Failed: ${MODEL} on ${dataset} (${MODE})"
        echo "$(date): ✗ Failed ${MODEL} on ${dataset} (${MODE})" >> ${LOG_FILE}
    fi
    echo ""
done

echo "=========================================="
echo "All ${MODE} experiments completed for ${MODEL}!"
echo "=========================================="


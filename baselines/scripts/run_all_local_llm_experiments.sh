#!/bin/bash

# Comprehensive script to run Qwen3-8B and LLaMA3-8B-Instruct baselines
# on all datasets (except 2wikimultihopqa) with and without retrieval

# Base paths
DATA_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
RESULTS_DIR="${DATA_DIR}/results"
RETRIEVAL_DIR="${DATA_DIR}/retrieval"
ORIGINAL_DIR="${DATA_DIR}/original"

# Models
QWEN_MODEL="Qwen/Qwen3-8B"
LLAMA_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Datasets to run (excluding 2wikimultihopqa)
DATASETS=("triviaqa" "popqa" "pubhealth" "arc_c" "bio" "asqa" "eli5")

# Create directories if they don't exist
mkdir -p ${RESULTS_DIR}
mkdir -p ${RETRIEVAL_DIR}

echo "=========================================="
echo "Running Local LLM Baseline Experiments"
echo "=========================================="
echo ""

# Function to run retrieval first
run_retrieval() {
    local dataset=$1
    local input_file="${ORIGINAL_DIR}/${dataset}.jsonl"
    local output_file="${RETRIEVAL_DIR}/${dataset}_dense_ret.jsonl"
    
    # Check if retrieval file already exists
    if [ -f "$output_file" ]; then
        echo "Retrieval file already exists: $output_file"
        return 0
    fi
    
    # Handle different file formats
    if [ "$dataset" = "asqa" ]; then
        input_file="${ORIGINAL_DIR}/${dataset}.json"
    elif [ "$dataset" = "eli5" ]; then
        input_file="${DATA_DIR}/${dataset}_test.json"
    fi
    
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file not found: $input_file"
        return 1
    fi
    
    echo "Running retrieval for ${dataset}..."
    python dense_ret.py \
        --query_file "$input_file" \
        --output_file "$output_file" \
        --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
        --num_splits 5 \
        --batch_size 32
    
    if [ $? -ne 0 ]; then
        echo "Error: Retrieval failed for ${dataset}"
        return 1
    fi
    echo "Retrieval completed for ${dataset}"
    echo ""
}

# Function to run baseline experiment
run_baseline() {
    local model=$1
    local model_name=$2
    local dataset=$3
    local mode=$4  # "vanilla" or "retrieval"
    local top_n=${5:-5}
    
    # Determine input file
    local input_file=""
    if [ "$dataset" = "asqa" ]; then
        input_file="${ORIGINAL_DIR}/${dataset}.json"
    elif [ "$dataset" = "eli5" ]; then
        input_file="${DATA_DIR}/${dataset}_test.json"
    else
        input_file="${ORIGINAL_DIR}/${dataset}.jsonl"
    fi
    
    # Determine task and prompt
    local task=""
    local prompt_name=""
    local max_tokens=100
    local metric="match"
    
    case $dataset in
        "triviaqa"|"popqa")
            task="qa"
            if [ "$mode" = "retrieval" ]; then
                prompt_name="prompt_no_input_retrieval"
            else
                prompt_name="prompt_no_input"
            fi
            ;;
        "pubhealth")
            task="fever"
            if [ "$mode" = "retrieval" ]; then
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
            if [ "$mode" = "retrieval" ]; then
                prompt_name="prompt_no_input_retrieval"
            else
                prompt_name="prompt_no_input"
            fi
            max_tokens=300
            metric="factscore"
            ;;
        "asqa")
            task="asqa_base"
            if [ "$mode" = "retrieval" ]; then
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
            if [ "$mode" = "retrieval" ]; then
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
    local mode_suffix="base"
    if [ "$mode" = "retrieval" ]; then
        mode_suffix="retrieval_top_${top_n}"
    fi
    local output_file="${RESULTS_DIR}/${dataset}_${model}_${mode_suffix}.jsonl"
    
    # Build command with GPU selection (use available GPUs 5, 6, 7)
    # Rotate through available GPUs to distribute load
    local gpu_id=$((RANDOM % 3 + 5))  # Random between 5, 6, 7
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    local cmd="python local_llm_test.py \
        --input_file \"${input_file}\" \
        --mode ${mode} \
        --task ${task} \
        --prompt_name ${prompt_name} \
        --max_new_tokens ${max_tokens} \
        --metric ${metric} \
        --result_fp \"${output_file}\" \
        --model_name ${model_name} \
        --batch_size 4 \
        --device cuda"
    
    if [ "$mode" = "retrieval" ]; then
        # Use existing ctxs from dataset files (no need for separate retrieval file)
        cmd="${cmd} --top_n ${top_n}"
        echo "Note: Using existing retrieval results (ctxs) from dataset file"
    fi
    
    echo "Running: ${model} on ${dataset} (${mode})"
    echo "Command: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${model} on ${dataset} (${mode})"
        echo ""
        
        # Evaluate results
        echo "Evaluating results..."
        python evaluate_results.py \
            --result_file "${output_file}" \
            --dataset ${dataset} \
            --metric auto
        echo ""
    else
        echo "✗ Failed: ${model} on ${dataset} (${mode})"
        echo ""
    fi
}

# Main execution
echo "Note: Using existing retrieval results (ctxs) from dataset files"
echo "Skipping retrieval step - datasets already contain top-5 retrieval results"
echo ""

echo "=========================================="
echo "Running baseline experiments"
echo "=========================================="
echo ""

# Run experiments for each model and dataset
for model in "qwen3_8b" "llama3_8b"; do
    if [ "$model" = "qwen3_8b" ]; then
        model_name="$QWEN_MODEL"
    else
        model_name="$LLAMA_MODEL"
    fi
    
    echo "=========================================="
    echo "Running experiments for ${model}"
    echo "=========================================="
    echo ""
    
    for dataset in "${DATASETS[@]}"; do
        # Run without retrieval
        echo "--- ${model} on ${dataset} (without retrieval) ---"
        run_baseline "$model" "$model_name" "$dataset" "vanilla"
        
        # Run with retrieval
        if [ "$dataset" = "arc_c" ]; then
            echo "Skipping retrieval for ${dataset} (multiple choice task)"
        else
            echo "--- ${model} on ${dataset} (with retrieval) ---"
            # For ASQA and ELI5, they use their own retrieval mode
            if [ "$dataset" = "asqa" ] || [ "$dataset" = "eli5" ]; then
                # These datasets use special modes: asqa_base/asqa_ret and eli5_base/eli5_ret
                run_baseline "$model" "$model_name" "$dataset" "retrieval" 5
            else
                run_baseline "$model" "$model_name" "$dataset" "retrieval" 5
            fi
        fi
    done
    
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: ${RESULTS_DIR}"
echo ""


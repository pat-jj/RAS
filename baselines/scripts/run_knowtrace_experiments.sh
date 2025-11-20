#!/bin/bash

# Script to run KnowTrace experiments on all datasets with both Sonnet and LLaMA
# Runs in screen sessions for long-running experiments

# Base paths
DATA_DIR="/shared/rsaas/pj20/firas_data/test_datasets"
RESULTS_DIR="${DATA_DIR}/results"
ORIGINAL_DIR="${DATA_DIR}/original"
KNOWTRACE_DIR="/home/pj20/server-04/FIRAS/baselines"
CORPUS_PATH="/shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018/all_wiki_text_cleaned.json"

# Datasets to run (excluding 2wikimultihopqa)
DATASETS=("triviaqa" "popqa" "pubhealth" "arc_c" "bio" "asqa" "eli5")

# Create directories if they don't exist
mkdir -p ${RESULTS_DIR}
mkdir -p ${KNOWTRACE_DIR}/logs

echo "=========================================="
echo "Running KnowTrace Experiments"
echo "=========================================="
echo ""

echo "Using baseline retriever (no separate server needed)"
echo ""

# Function to run KnowTrace experiment
run_knowtrace() {
    local model=$1
    local model_name=$2
    local dataset=$3
    local step_num=${4:-5}
    
    # Determine input file
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
    
    # Determine output file
    local output_file="${RESULTS_DIR}/${dataset}_${model}_knowtrace.jsonl"
    
    # Build command
    local cmd="python knowtrace_test.py \
        --input_file \"${input_file}\" \
        --result_fp \"${output_file}\" \
        --base_llm ${model_name} \
        --step_num ${step_num} \
        --dataset ${dataset} \
        --max_items 800 \
        --num_splits 5"
    
    if [ "$model_name" = "sonnet" ] || [ "$model_name" = "sonnet3.5" ]; then
        cmd="${cmd} --sonnet_model sonnet"
    else
        cmd="${cmd} --local_llm_port 1051"
    fi
    
    echo "Running: KnowTrace with ${model} on ${dataset}"
    echo "Command: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: KnowTrace with ${model} on ${dataset}"
        echo ""
        
        # Evaluate results
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

# Main execution - run in screen sessions
echo "=========================================="
echo "Starting KnowTrace experiments in screen sessions"
echo "=========================================="
echo ""

# Create a temporary script for Sonnet experiments
SONNET_SCRIPT="${KNOWTRACE_DIR}/run_knowtrace_sonnet.sh"
cat > ${SONNET_SCRIPT} << 'EOFSCRIPT'
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
        --sonnet_model sonnet \
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
echo 'Running KnowTrace experiments for Sonnet'
echo '=========================================='
echo ''

for dataset in "${DATASETS[@]}"; do
    echo "--- KnowTrace (Sonnet) on ${dataset} ---"
    run_knowtrace "sonnet" "sonnet" "${dataset}" 5
done

echo ''
echo '=========================================='
echo 'Sonnet experiments completed!'
echo '=========================================='
EOFSCRIPT
chmod +x ${SONNET_SCRIPT}

# Create a temporary script for LLaMA experiments
LLAMA_SCRIPT="${KNOWTRACE_DIR}/run_knowtrace_llama.sh"
cat > ${LLAMA_SCRIPT} << 'EOFSCRIPT'
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
EOFSCRIPT
chmod +x ${LLAMA_SCRIPT}

# Run experiments for Sonnet in screen
echo "Starting Sonnet experiments in screen session 'knowtrace_sonnet'..."
screen -dmS knowtrace_sonnet bash -c "${SONNET_SCRIPT} 2>&1 | tee ${KNOWTRACE_DIR}/logs/knowtrace_sonnet.log"

# Create a temporary script for LLaMA-2-7B experiments
LLAMA2_SCRIPT="${KNOWTRACE_DIR}/run_knowtrace_llama2.sh"
cat > ${LLAMA2_SCRIPT} << 'EOFSCRIPT'
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
    local llm_port=${5:-1052}
    
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
        --local_llm_port ${llm_port} \
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
echo 'Running KnowTrace experiments for LLaMA-2-7B'
echo '=========================================='
echo ''

for dataset in "${DATASETS[@]}"; do
    echo "--- KnowTrace (LLaMA-2-7B) on ${dataset} ---"
    run_knowtrace "llama2_7b" "Llama-2-7b-chat-hf" "${dataset}" 5 1052
done

echo ''
echo '=========================================='
echo 'LLaMA-2-7B experiments completed!'
echo '=========================================='
EOFSCRIPT
chmod +x ${LLAMA2_SCRIPT}

# Run experiments for LLaMA in screen
echo "Starting LLaMA3-8B experiments in screen session 'knowtrace_llama'..."
screen -dmS knowtrace_llama bash -c "${LLAMA_SCRIPT} 2>&1 | tee ${KNOWTRACE_DIR}/logs/knowtrace_llama.log"

# Run experiments for LLaMA-2-7B in screen
echo "Starting LLaMA-2-7B experiments in screen session 'knowtrace_llama2'..."
screen -dmS knowtrace_llama2 bash -c "${LLAMA2_SCRIPT} 2>&1 | tee ${KNOWTRACE_DIR}/logs/knowtrace_llama2.log"

echo ""
echo "=========================================="
echo "KnowTrace experiments started in screen sessions!"
echo "=========================================="
echo ""
echo "Screen sessions:"
echo "  - knowtrace_sonnet: Sonnet experiments"
echo "  - knowtrace_llama: LLaMA3-8B experiments"
echo "  - knowtrace_llama2: LLaMA-2-7B experiments"
echo ""
echo "To attach to a session:"
echo "  screen -r knowtrace_sonnet"
echo "  screen -r knowtrace_llama"
echo "  screen -r knowtrace_llama2"
echo ""
echo "To list all sessions:"
echo "  screen -ls"
echo ""
echo "Logs are being saved to: ${KNOWTRACE_DIR}/logs/"
echo ""
echo "Note: LLaMA-2-7B experiments use port 1052 for the local LLM server."
echo "      Make sure the LLaMA-2-7B server is running on port 1052."
echo ""


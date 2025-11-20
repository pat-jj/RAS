# Baselines

This directory contains scripts for running baseline experiments with dense retrieval and various LLM models (Claude Sonnet, Sonnet 4.5, Qwen3-8B, LLaMA3-8B-Instruct).

## Overview

The baselines consist of two main components:
1. **Dense Retrieval** (`dense_ret.py`) - Retrieves relevant documents from a knowledge base
2. **LLM Baselines** - Generate answers using retrieved documents:
   - **Claude Sonnet** (`sonnet_test.py`, `sonnet_sure.py`) - API-based baselines
   - **Sonnet 4.5** (`sonnet_test.py`, `sonnet_sure.py`) - Latest Claude Sonnet model
   - **Local LLMs** (`local_llm_test.py`) - Qwen3-8B and LLaMA3-8B-Instruct

## Setup

### 1. API Configuration

Before running baselines, you need to configure API access:

1. Copy the example API file:
   ```bash
   cp claude_api_example.py claude_api.py
   ```

2. Edit `claude_api.py` and fill in your API key information.

### 2. Dependencies

Make sure you have installed all required dependencies (see main README.md).

## Usage

### Step 1: Run Dense Retrieval

First, retrieve relevant documents for your queries using dense retrieval:

```bash
python dense_ret.py \
    --query_file /path/to/your/queries.jsonl \
    --output_file /path/to/output/retrieval_results.jsonl \
    --knowledge_path /path/to/knowledge_base \
    --dense_encoder facebook/contriever-msmarco \
    --num_splits 5 \
    --batch_size 32
```

**Arguments:**
- `--query_file`: Path to input file containing queries (JSONL format)
- `--output_file`: Path to save retrieval results
- `--knowledge_path`: Path to knowledge base indices (should contain `embedding/` subdirectory with FAISS indices)
- `--dense_encoder`: Dense encoder model (default: `facebook/contriever-msmarco`)
- `--num_splits`: Number of index splits (default: 5)
- `--batch_size`: Batch size for processing queries (default: 32)

**Input Format:**
The query file should be in JSONL format with each line containing:
```json
{"instruction": "Your question here"}
```
or
```json
{"question": "Your question here"}
```

**Output Format:**
The output file will contain the same structure with an additional `ctxs` field:
```json
{"instruction": "Your question here", "ctxs": [["retrieved_doc_1", score_1], ["retrieved_doc_2", score_2], ...]}
```

### Step 2: Run LLM Baselines

#### Option A: Standard Baseline (`sonnet_test.py`)

Run standard baseline with Sonnet:

```bash
python sonnet_test.py \
    --input_file /path/to/test_data.jsonl \
    --retrieval_file /path/to/retrieval_results.jsonl \
    --mode retrieval \
    --top_n 5 \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 100 \
    --metric match \
    --result_fp /path/to/output/results.jsonl
```

**Arguments:**
- `--input_file`: Path to test dataset file
- `--retrieval_file`: Path to retrieval results from Step 1 (optional, if not provided, uses contexts from input_file)
- `--mode`: Mode of operation (`vanilla`, `retrieval`, `asqa_base`, `eli5_base`, `2wikimultihop`, etc.)
- `--top_n`: Number of retrieved documents to use (default: 1)
- `--task`: Task type (`qa`, `fever`, `arc_c`, `factscore`, `asqa_base`, `eli5_ret`, `2wikimultihop`, etc.)
- `--prompt_name`: Prompt template name (`prompt_no_input`, `prompt_no_input_retrieval`, `asqa_base`, `eli5_base`, etc.)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 15)
- `--metric`: Evaluation metric (`match`, `f1`, `factscore`, etc.)
- `--result_fp`: Path to save results
- `--batch_size`: Batch size for API calls (default: 8)

#### Option B: SuRE Framework Baseline (`sonnet_sure.py`)

Run baseline with SuRE (Summarize, Retrieve, Evaluate) framework:

```bash
python sonnet_sure.py \
    --input_file /path/to/test_data.jsonl \
    --retrieval_file /path/to/retrieval_results.jsonl \
    --mode eli5_ret \
    --top_n 5 \
    --task eli5_ret \
    --prompt_name eli5_ret \
    --max_new_tokens 300 \
    --metric match \
    --result_fp /path/to/output/results.jsonl
```

**Arguments:** Same as `sonnet_test.py`

## Supported Tasks

The baselines support the following tasks:

- **QA Tasks**: `qa`, `triviaqa`, `popqa`
- **Fact Verification**: `fever`, `pubhealth`
- **Multiple Choice**: `arc_c`, `arc_easy`, `obqa`
- **Long-form QA**: `eli5`, `asqa`
- **Multi-hop QA**: `2wikimultihop`
- **Factual Consistency**: `factscore`, `bio`

## Examples

### Example 1: TriviaQA with Retrieval

```bash
# Step 1: Retrieve documents
python dense_ret.py \
    --query_file <path_to_test_datasets>/original/triviaqa.jsonl \
    --output_file <path_to_test_datasets>/retrieval/triviaqa_dense_ret.jsonl \
    --knowledge_path <path_to_knowledge_base> \
    --num_splits 5

# Step 2: Run baseline
python sonnet_test.py \
    --input_file <path_to_test_datasets>/original/triviaqa.jsonl \
    --retrieval_file <path_to_test_datasets>/retrieval/triviaqa_dense_ret.jsonl \
    --mode retrieval \
    --top_n 5 \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 100 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/triviaqa_sonnet_retrieval_top_5.jsonl
```

### Example 2: ELI5 with SuRE Framework

```bash
# Step 1: Retrieve documents
python dense_ret.py \
    --query_file <path_to_test_datasets>/eli5_test.json \
    --output_file <path_to_test_datasets>/retrieval/eli5_dense_ret.jsonl \
    --knowledge_path <path_to_knowledge_base> \
    --num_splits 5

# Step 2: Run SuRE baseline
python sonnet_sure.py \
    --input_file <path_to_test_datasets>/eli5_test.json \
    --retrieval_file <path_to_test_datasets>/retrieval/eli5_dense_ret.jsonl \
    --mode eli5_ret \
    --top_n 5 \
    --task eli5_ret \
    --prompt_name eli5_ret \
    --max_new_tokens 300 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/eli5_sonnet_retrieval_top_5_sure.jsonl
```

### Example 3: ARC-C (Multiple Choice)

```bash
python sonnet_test.py \
    --input_file <path_to_test_datasets>/original/arc_c.jsonl \
    --mode vanilla \
    --task arc_c \
    --prompt_name prompt_no_input \
    --max_new_tokens 10 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/arc_c_sonnet.jsonl
```

### Example 4: PubHealth (Fact Verification)

```bash
python sonnet_test.py \
    --input_file <path_to_test_datasets>/original/pubhealth.jsonl \
    --mode retrieval \
    --top_n 1 \
    --task fever \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 50 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/pubhealth_sonnet_retrieval_top_1.jsonl
```

## File Formats

### Input Query File (JSONL)
```json
{"instruction": "What is the capital of France?"}
{"question": "Who wrote Romeo and Juliet?"}
```

### Retrieval Output File (JSONL)
```json
{"instruction": "What is the capital of France?", "ctxs": [["Paris is the capital...", 0.95], ["France is a country...", 0.89]]}
```

### Test Data File
Format depends on the dataset:
- **JSONL**: `{"question": "...", "answers": ["..."]}`
- **JSON**: `{"question": [...], "answer": [...]}`

## Local LLM Baselines (Qwen3-8B-Instruct, LLaMA3-8B-Instruct)

We also support running baselines with local LLM models using `local_llm_test.py`. This script supports both Qwen3-8B-Instruct and LLaMA3-8B-Instruct models, with and without retrieval.

### Basic Usage

```bash
python local_llm_test.py \
    --input_file /path/to/test_data.jsonl \
    --retrieval_file /path/to/retrieval_results.jsonl \
    --mode retrieval \
    --top_n 5 \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 100 \
    --metric match \
    --result_fp /path/to/output/results.jsonl \
    --model_name Qwen/Qwen3-8B \
    --batch_size 4 \
    --device cuda
```

**Key Arguments:**
- `--model_name`: Model name or path (e.g., `Qwen/Qwen3-8B` or `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--batch_size`: Batch size for inference (default: 4, adjust based on GPU memory)
- `--temperature`: Sampling temperature (default: 0.0 for deterministic)
- `--enable_thinking`: (Qwen3 only) Enable thinking mode to generate reasoning before the answer

### Examples

#### Qwen3-8B-Instruct WITHOUT Retrieval
```bash
python local_llm_test.py \
    --input_file <path_to_test_datasets>/original/triviaqa.jsonl \
    --mode vanilla \
    --task qa \
    --prompt_name prompt_no_input \
    --max_new_tokens 100 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/triviaqa_qwen3_8b_base.jsonl \
    --model_name Qwen/Qwen3-8B \
    --batch_size 4 \
    --device cuda
```

#### Qwen3-8B-Instruct WITH Retrieval
```bash
python local_llm_test.py \
    --input_file <path_to_test_datasets>/original/triviaqa.jsonl \
    --retrieval_file <path_to_test_datasets>/retrieval/triviaqa_dense_ret.jsonl \
    --mode retrieval \
    --top_n 5 \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 100 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/triviaqa_qwen3_8b_retrieval_top_5.jsonl \
    --model_name Qwen/Qwen3-8B \
    --batch_size 4 \
    --device cuda
```

#### LLaMA3-8B-Instruct WITHOUT Retrieval
```bash
python local_llm_test.py \
    --input_file <path_to_test_datasets>/original/triviaqa.jsonl \
    --mode vanilla \
    --task qa \
    --prompt_name prompt_no_input \
    --max_new_tokens 100 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/triviaqa_llama3_8b_base.jsonl \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 4 \
    --device cuda
```

#### LLaMA3-8B-Instruct WITH Retrieval
```bash
python local_llm_test.py \
    --input_file <path_to_test_datasets>/original/triviaqa.jsonl \
    --retrieval_file <path_to_test_datasets>/retrieval/triviaqa_dense_ret.jsonl \
    --mode retrieval \
    --top_n 5 \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --max_new_tokens 100 \
    --metric match \
    --result_fp <path_to_test_datasets>/results/triviaqa_llama3_8b_retrieval_top_5.jsonl \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 4 \
    --device cuda
```

### Running All Baselines

#### Local LLM Baselines (Qwen3-8B, LLaMA3-8B-Instruct)

You can use the provided shell script to run all local LLM baselines:

```bash
bash run_local_llm.sh
```

Or run individual modes:

```bash
# Run vanilla (w/o retrieval) experiments
bash run_single_experiment.sh vanilla qwen3_8b

# Run retrieval (w/ retrieval) experiments
bash run_single_experiment.sh retrieval qwen3_8b
```

#### Sonnet 4.5 Baselines

Run all Sonnet 4.5 baselines (vanilla, retrieval, and SuRE):

```bash
bash run_sonnet45_experiments.sh
```

This will run:
- Vanilla baseline (w/o retrieval) on all datasets
- Retrieval baseline (w/ retrieval) on all datasets
- SuRE baseline on ASQA and ELI5 (long-form tasks)

**Model Configuration:**
- Model name: `sonnet4.5` (maps to `anthropic.claude-sonnet-4-5-20250929-v1:0`)
- Configured in `claude_api.py` in the `BEDROCK_MODEL_NAME_MAP`

**Note**: Make sure to:
1. Have sufficient GPU memory (8B models typically need ~16GB VRAM)
2. Adjust `--batch_size` based on your GPU memory
3. **Model Access**:
   - **Qwen3-8B**: Available at [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) - no special access required
   - **LLaMA3-8B-Instruct**: Available at [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) - requires Meta's approval and HuggingFace authentication. You need to:
     - Request access on the HuggingFace model page
     - Accept Meta's Llama 3 Community License Agreement
     - Authenticate with HuggingFace: `huggingface-cli login`

## Result File Locations

All baseline results are stored in:
```
<results_dir>/results/
```

**File Naming Convention:**
- Format: `<dataset>_<model>_<mode>.jsonl`
- Examples:
  - `triviaqa_qwen3_8b_base.jsonl` - Qwen3-8B vanilla baseline
  - `triviaqa_qwen3_8b_retrieval_top_5.jsonl` - Qwen3-8B with retrieval
  - `triviaqa_sonnet4.5_base.jsonl` - Sonnet 4.5 vanilla baseline
  - `triviaqa_sonnet4.5_retrieval_top_5.jsonl` - Sonnet 4.5 with retrieval
  - `asqa_sonnet4.5_retrieval_top_5_sure.jsonl` - Sonnet 4.5 SuRE baseline
  - `eli5_llama3_8b_base.jsonl` - LLaMA3-8B vanilla baseline

**Models:**
- `qwen3_8b` - Qwen3-8B-Instruct
- `llama3_8b` - LLaMA3-8B-Instruct
- `sonnet4.5` - Claude Sonnet 4.5
- `sonnet` - Claude Sonnet (legacy)

**Modes:**
- `base` - Vanilla baseline (no retrieval)
- `retrieval_top_N` - With retrieval (N documents)
- `retrieval_top_N_sure` - SuRE framework baseline

## Notes

1. **Memory Management**: 
   - The dense retrieval script loads indices per split to manage memory efficiently.
   - For local LLM baselines, adjust `--batch_size` based on your GPU memory (smaller batch size = less memory usage).

2. **API Rate Limiting**: The Sonnet scripts include retry logic and batch processing to handle API rate limits.

3. **Sonnet 4.5**: The latest Claude Sonnet model. Use `--model_name sonnet4.5` in `sonnet_test.py` or `sonnet_sure.py` to use this model.

3. **Evaluation Metrics**: 
   - `match`: Exact match or substring match
   - `f1`: F1 score between predicted and ground truth
   - `factscore`: FactScore metric for factual consistency

4. **Knowledge Base**: Make sure your knowledge base path contains:
   ```
   knowledge_path/
   └── embedding/
       ├── text_mapping_0.json
       ├── text_mapping_1.json
       ├── ...
       ├── wikipedia_embeddings_0.faiss
       ├── wikipedia_embeddings_1.faiss
       └── ...
   ```

5. **GPU Usage**: 
   - The dense retrieval script will use GPU if available. You can control GPU usage through CUDA_VISIBLE_DEVICES environment variable.
   - Local LLM baselines require GPU for reasonable inference speed. CPU inference is possible but very slow.

## Monitoring Running Experiments

When running experiments in the background, you can monitor them using:

### Check Running Processes
```bash
# Count active processes (should be 2: vanilla and retrieval)
ps aux | grep "local_llm_test.py" | grep -v grep | wc -l

# See what's running
ps aux | grep "local_llm_test.py" | grep -v grep
```

### Monitor via Log Files (Recommended)
```bash
# Tail vanilla experiments log (w/o retrieval)
tail -f logs/qwen3_8b_vanilla.log

# Tail retrieval experiments log (w/ retrieval)
tail -f logs/qwen3_8b_retrieval.log

# Tail both logs simultaneously (in separate terminals)
# Terminal 1:
tail -f logs/qwen3_8b_vanilla.log

# Terminal 2:
tail -f logs/qwen3_8b_retrieval.log

# Or use multitail to see both in one terminal (if installed)
multitail logs/qwen3_8b_vanilla.log logs/qwen3_8b_retrieval.log
```

### Check Progress via Result Files
```bash
# List result files as they're created (adjust path to your results directory)
ls -lht <results_dir>/*.jsonl | head -10

# Count completed experiments
ls -1 <results_dir>/*qwen3*.jsonl 2>/dev/null | wc -l
```

### Check GPU Usage
```bash
nvidia-smi
# Or watch continuously
watch -n 1 nvidia-smi
```

### Running Experiments in Background

To run experiments that persist after disconnection, use `nohup` or `screen`:

```bash
# Using nohup (output goes to log file)
nohup bash run_single_experiment.sh vanilla qwen3_8b > vanilla.log 2>&1 &
nohup bash run_single_experiment.sh retrieval qwen3_8b > retrieval.log 2>&1 &

# Using screen (can reconnect later)
screen -dmS firas_vanilla bash -c "bash run_single_experiment.sh vanilla qwen3_8b"
screen -dmS firas_retrieval bash -c "bash run_single_experiment.sh retrieval qwen3_8b"

# Reconnect to screen sessions
screen -r firas_vanilla  # Press Ctrl+A then D to detach
screen -r firas_retrieval
```

## Troubleshooting

- **Out of Memory**: Reduce `--batch_size` or `--num_splits`
- **API Errors**: Check your `claude_api.py` configuration and API key
- **File Not Found**: Verify all file paths are correct and files exist
- **Index Errors**: Ensure knowledge base indices are properly created and split
- **Screen Sessions Not Found**: If screen sessions don't exist, processes may be running via nohup. Check with `ps aux | grep local_llm_test`


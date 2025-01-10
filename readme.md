# Efficient and Effective Knowledge Serving to LLMs with Iterative Retrieval-And-Structure

#### Environment Setup

```bash
# Create new env
conda create -n ras python=3.10 -y

# Activate it
conda activate ras

# Install PyTorch first, separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install torch-geometric

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html #(depends on your cuda version)

pip install transformers wandb tqdm peft accelerate bitsandbytes sentencepiece


```


#### Run evaluation
For long-form evaluations, set up external libraries or repositories to run evaluations.

- `factscore==v0.1.5` (bio)
Please follow the instructions at the [FactScore](https://github.com/shmsw25/FActScore) official repository to set up your environment.
```
python -m factscore.factscorer --data_path YOUR_OUTPUT_FILE  --model_name retrieval+ChatGPT --cache_dir YOUR_CACHE_DIR --openai_key YOUR_OPEN_AI_KEY --verbose
```

- [ALCE/ASQA](https://github.com/princeton-nlp/ALCE)

ALCE provides a comprehensive evaluation using multiple different metrics for long-form QA. For your first evaluation, install the ALCE repo and download the data.
```
git clone https://github.com/princeton-nlp/ALCE.git
python3 -m alce_env
cd ALCE
bash download_data.sh
```

For ASQA, you can run evaluations as follows. Note that ASQA evaluations require T5-XXL (11B)-based NLI module.
```
python eval.py --f YOUR_OUTPUT_FILE --citations --qa --mauve
```


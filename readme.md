# Efficient and Effective Knowledge Serving to LLMs with Iterative Retrieval-And-Structure

## Environment Setup

```bash
# Create new env
conda create -n ras python=3.10 -y

# Activate it
conda activate ras

# Install PyTorch first, separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install torch-geometric

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

pip install transformers wandb tqdm peft accelerate bitsandbytes sentencepiece


```


#!/bin/bash
# Setup script for SFT training environment with DeepSpeed support

ENV_NAME="sft_training"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Setting up conda environment: $ENV_NAME"
echo "=========================================="

# Create conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA support..."
# Install PyTorch via pip (more reliable than conda for CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing core dependencies..."
pip install transformers>=4.51.0
pip install accelerate
pip install peft
pip install trl
pip install datasets
pip install wandb
pip install safetensors
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html || pip install torch-scatter
pip install torch-geometric  # Needed to unpickle data files

echo "Installing DeepSpeed..."
# Try to set CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Setting CUDA_HOME to /usr/local/cuda"
    elif [ -d "/usr/local/cuda-11.8" ]; then
        export CUDA_HOME=/usr/local/cuda-11.8
        echo "Setting CUDA_HOME to /usr/local/cuda-11.8"
    fi
fi

# Install DeepSpeed (it will work even without CUDA_HOME for basic usage)
pip install deepspeed

# Note: DeepSpeed may show warnings about CUDA_HOME, but it will still work
# for basic ZeRO optimization without compiling custom ops

echo "Installing other dependencies..."
pip install tqdm
pip install numpy
pip install scipy

echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify DeepSpeed installation:"
echo "  python -c 'import deepspeed; print(deepspeed.__version__)'"
echo ""


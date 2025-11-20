#!/bin/bash
# Script to start experiments in a screen session for persistence

cd /home/pj20/server-04/FIRAS/baselines

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate firas_baselines

# Run experiments
bash run_all_local_llm_experiments.sh


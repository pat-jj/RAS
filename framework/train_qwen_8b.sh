#!/bin/bash
# Example training command for Qwen3-8B GraphLLM

source /home/pj20/miniconda3/etc/profile.d/conda.sh
export WANDB_API_KEY="fcb8fe58088a42fc17b32841ed54c2785d670a66"

cd /home/pj20/server-04/FIRAS/framework
conda activate handbook

# Basic training with defaults (LoRA fine-tuning)
python train_qwen_8b.py \
    --finetune_method lora \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --epochs 1 \
    --llm_frozen False

# Or with custom paths and more control:
# python train_qwen_8b.py \
#     --finetune_method lora \
#     --batch_size 2 \
#     --grad_accum_steps 8 \
#     --data_dir /shared/rsaas/pj20/firas_data/multitask \
#     --output_dir /shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen \
#     --epochs 1 \
#     --llm_frozen False \
#     --lr 1e-5 \
#     --warmup_ratio 0.15

# Debug mode (uses validation data, smaller batch, 1 epoch):
# python train_qwen_8b.py --debug

# Resume from checkpoint:
# python train_qwen_8b.py \
#     --resume_from_checkpoint /shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen/checkpoint_10_of_20_qwen.safetensors \
#     --finetune_method lora \
#     --batch_size 2 \
#     --grad_accum_steps 8

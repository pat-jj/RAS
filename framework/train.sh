python train.py \
    --finetune_method lora \
    --batch_size 2 \
    --grad_accum_steps 2 \
    --data_dir YOUR_DATA_DIR \
    --output_dir YOUR_OUTPUT_DIR \
    --epochs 1 \
    --llm_frozen False


# python train.py \
#     --finetune_method lora \
#     --batch_size 2 \
#     --grad_accum_steps 2 \
#     --data_dir /shared/eng/pj20/firas_data/multitask \
#     --output_dir /shared/eng/pj20/firas_data/multitask/checkpoints_20 \
#     --epochs 1 \
#     --llm_frozen False
    # --resume_from_checkpoint /shared/eng/pj20/firas_data/multitask/checkpoints_20/checkpoint_3_of_20.safetensors


# python train_8b.py \
#     --finetune_method lora \
#     --batch_size 2 \
#     --grad_accum_steps 2 \
#     --data_dir /shared/eng/pj20/firas_data/multitask \
#     --output_dir /shared/eng/pj20/firas_data/multitask/checkpoints_20 \
#     --epochs 1 \
#     --llm_frozen False \
#     --llm_model_path meta-llama/Meta-Llama-3-8B
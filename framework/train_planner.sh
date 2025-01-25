python train_planner.py \
    --finetune_method lora \
    --batch_size 2 \
    --grad_accum_steps 2 \
    --data_dir /shared/eng/pj20/firas_data/multitask \
    --output_dir /shared/eng/pj20/firas_data/multitask/checkpoints_20 \
    --hf_repo_id pat-jj/ras_planner \
    --hf_token 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 \
    --epochs 1 \
    --llm_frozen False
python train_planner.py \
    --finetune_method lora \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --data_dir /shared/eng/pj20/firas_data/action_planner/all_train \
    --output_dir /shared/eng/pj20/firas_data/action_planner/all_train/checkpoints_p_tune\
    --hf_repo_id pat-jj/ras_planner \
    --hf_token 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 \
    --epochs 3 \
    --llm_frozen True
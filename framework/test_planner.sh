python test_planner.py \
    --checkpoint_path /shared/eng/pj20/firas_data/action_planner/all_train/checkpoints_lora/latest_checkpoint_epoch_lora_False.safetensors \
    --test_data_path /shared/eng/pj20/firas_data/action_planner/all_train/test.pkl \
    --output_path results.json \
    --finetune_method lora \
    --llm_frozen False
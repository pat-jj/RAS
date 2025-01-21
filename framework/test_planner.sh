python test_planner.py \
    --checkpoint_path /shared/eng/pj20/firas_data/action_planner/hotpotqa_only/ptune/latest_checkpoint_epoch_lora_True.safetensors \
    --test_data_path /shared/eng/pj20/firas_data/action_planner/all_train/test.pkl \
    --output_path results.json \
    --finetune_method lora \
    --llm_frozen True
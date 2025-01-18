python test_answerer.py \
    --checkpoint_path /shared/eng/pj20/firas_data/answerer/all_train/checkpoints_lora_v3/latest_checkpoint.safetensors \
    --test_data_path /shared/eng/pj20/firas_data/answerer/all_train/val_v2.pkl \
    --output_path test_results.json \
    --finetune_method lora \
    --llm_frozen False
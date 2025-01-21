python test_answerer.py \
    --checkpoint_path /shared/eng/pj20/firas_data/answerer/all_train/checkpoints_lora_v3/latest_checkpoint.safetensors \
    --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/eli5_test_output_graphllm_graphllm_answerer_data.pkl \
    --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_eli5_answerer_test_results_all_graph.json \
    --finetune_method lora \
    --llm_frozen False \
    --batch_size 1
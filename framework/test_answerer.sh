# python test_answerer.py \
#     --checkpoint_path /shared/eng/pj20/firas_data/multitask/checkpoints/latest_checkpoint.safetensors \
#     --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/2wikimultihop_test_output_graphllm_graphllm_answerer_data.pkl \
#     --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_2wikimultihop_answerer_test_results_all_graph.json \
#     --finetune_method lora \
#     --llm_frozen False \
#     --batch_size 1

# python test_answerer.py \
#     --checkpoint_path /shared/eng/pj20/firas_data/multitask/checkpoints/latest_checkpoint.safetensors \
#     --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/asqa_test_output_sonnet_sonnet_answerer_data.pkl \
#     --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_asqa_answerer_test_results_all_graph.json \
#     --finetune_method lora \
#     --llm_frozen False \
#     --batch_size 1

# /shared/eng/pj20/firas_data/multitask/ckpt_ptune/latest_checkpoint_epoch_lora_True.safetensors
python test_answerer.py \
    --checkpoint_path /shared/eng/pj20/firas_data/multitask/checkpoints/latest_checkpoint.safetensors \
    --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/triviaqa_test_output_llama2-7b_sonnet_answerer_data_no_plan.pkl \
    --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_triviaqa_answerer_test_results_all_graph_no_plan.json \
    --finetune_method lora \
    --llm_frozen False \
    --batch_size 1

# python test_answerer.py \
#     --checkpoint_path /shared/eng/pj20/firas_data/multitask/ckpt_ptune/latest_checkpoint_epoch_lora_True.safetensors \
#     --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/arc_c_test_output_sonnet_sonnet_answerer_data.pkl \
#     --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_arc_c_answerer_test_results_all_graph_ptuned.json \
#     --finetune_method lora \
#     --llm_frozen True \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=1,2,4
# python test_answerer_no_graph.py \
#     --checkpoint_path /shared/eng/pj20/firas_data/multitask/ckpt_no_graph/latest_checkpoint_epoch_lora_False_nograph.safetensors \
#     --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/triviaqa_test_output_llama2-7b_sonnet_answerer_data.pkl \
#     --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_triviaqa_answerer_test_results_no_graph.json \
#     --finetune_method lora \
#     --llm_frozen False \
#     --batch_size 1

# python test_answerer_no_graph.py \
#     --checkpoint_path /shared/eng/pj20/firas_data/multitask/ckpt_no_graph/latest_checkpoint_epoch_lora_False_nograph.safetensors \
#     --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/2wikimultihop_test_output_llama2-7b_sonnet_answerer_data.pkl \
#     --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_2wikimultihop_answerer_test_results_no_graph.json \
#     --finetune_method lora \
#     --llm_frozen False \
#     --batch_size 1

# python test_answerer_no_graph.py \
#     --checkpoint_path /shared/eng/pj20/firas_data/multitask/ckpt_no_graph/latest_checkpoint_epoch_lora_False_nograph.safetensors \
#     --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/pubhealth_test_output_graphllm_graphllm_answerer_data.pkl \
#     --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_pubhealth_answerer_test_results_no_graph.json \
#     --finetune_method lora \
#     --llm_frozen False \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=1
# python test_answerer_no_graph.py \
#     --checkpoint_path /shared/eng/pj20/firas_data/multitask/ckpt_no_graph/latest_checkpoint_epoch_lora_False_nograph.safetensors \
#     --test_data_path /shared/eng/pj20/firas_data/test_datasets/answerer/asqa_test_output_sonnet_sonnet_answerer_data.pkl \
#     --output_path /shared/eng/pj20/firas_data/test_datasets/results/graphllm_asqa_answerer_test_results_no_graph.json \
#     --finetune_method lora \
#     --llm_frozen False \
#     --batch_size 1
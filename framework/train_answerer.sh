# export CUDA_VISIBLE_DEVICES=0,1
# python train.py \
#     --finetune_method full \
#     --batch_size 8 \
#     --grad_accum_steps 4 \
#     --output_dir /shared/eng/pj20/firas_data/inference_model/hotpotqa_train/checkpoints_full \
#     --epochs 3

# CUDA_VISIBLE_DEVICES=2,3,4,5
# python train_answerer.py \
#     --finetune_method lora \
#     --batch_size 4 \
#     --grad_accum_steps 4 \
#     --data_dir /shared/eng/pj20/firas_data/answerer/all_train \
#     --output_dir /shared/eng/pj20/firas_data/answerer/all_train/checkpoints_p_tune \
#     --hf_repo_id pat-jj/ras_answerer \
#     --hf_token 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 \
#     --epochs 10 \
#     --llm_frozen True

python train_answerer.py \
    --finetune_method lora \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --data_dir /shared/eng/pj20/firas_data/answerer/all_train \
    --output_dir /shared/eng/pj20/firas_data/answerer/all_train/checkpoints_lora_v3 \
    --hf_repo_id pat-jj/ras_answerer \
    --hf_token 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 \
    --epochs 6 
# export CUDA_VISIBLE_DEVICES=0,1
# python train.py \
#     --finetune_method full \
#     --batch_size 8 \
#     --grad_accum_steps 4 \
#     --output_dir /shared/eng/pj20/firas_data/inference_model/hotpotqa_train/checkpoints_full \
#     --epochs 3

export CUDA_VISIBLE_DEVICES=3,4,5,6,7
python train_answerer.py \
    --finetune_method lora \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --output_dir /shared/eng/pj20/firas_data/inference_model/hotpotqa_train/checkpoints_lora \
    --epochs 3
# export CUDA_VISIBLE_DEVICES=0,1
# python train_planner.py \
#     --finetune_method full \
#     --batch_size 8 \
#     --grad_accum_steps 4 \
#     --output_dir /shared/eng/pj20/firas_data/action_planner/hotpot_train/checkpoints_full \
#     --epochs 3

export CUDA_VISIBLE_DEVICES=3,4,5,6,7
python train_planner.py \
    --finetune_method lora \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --output_dir /shared/eng/pj20/firas_data/action_planner/hotpot_train/checkpoints_lora \
    --epochs 3
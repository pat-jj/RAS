# export CUDA_VISIBLE_DEVICES=0,1
# python train_planner.py \
#     --finetune_method full \
#     --batch_size 8 \
#     --grad_accum_steps 4 \
#     --output_dir /shared/eng/pj20/firas_data/action_planner/hotpot_train/checkpoints_full \
#     --epochs 3

python train_planner.py \
    --finetune_method full \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --data_dir /shared/eng/pj20/firas_data/action_planner/all_train \
    --output_dir /shared/eng/pj20/firas_data/action_planner/all_train/checkpoints_full \
    --epochs 3
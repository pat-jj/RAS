export CUDA_VISIBLE_DEVICES=2,3,4,5

# python run_ras.py \
#     --dataset pubhealth \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 50 \
#     --debug


# python run_ras.py \
#     --dataset popqa \
#     --knowledge_source wiki_2020 \
#     --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2020 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 100
    # --debug


# python run_ras.py \
#     --dataset 2wikimultihop \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only

# export CUDA_VISIBLE_DEVICES=2,3,4,5

# python run_ras.py \
#     --dataset triviaqa \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5
# python run_ras.py \
#     --dataset arc_c \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 50 

# python run_ras.py \
#     --dataset asqa \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 300

# python run_ras.py \
#     --dataset eli5 \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 300

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python run_ras.py \
    --dataset 2wikimultihop popqa\
    --knowledge_source wiki_2020 \
    --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2020 \
    --text_to_triples_model sonnet \
    --planner_model llama2-7b \
    --planner_frozen False \
    --planner_checkpoint /shared/rsaas/pj20/firas_data/multitask/checkpoints/latest_checkpoint.safetensors \
    --answerer_model sonnet \
    --retrieval_mode dense_only \
    --max_answer_length 100 

python run_ras.py \
    --dataset triviaqa \
    --knowledge_source wiki_2018 \
    --knowledge_path /shared/rsaas/pj20/firas_data/knowledge_source/wiki_2018 \
    --text_to_triples_model sonnet \
    --planner_model llama2-7b \
    --planner_frozen False \
    --planner_checkpoint /shared/rsaas/pj20/firas_data/multitask/checkpoints/latest_checkpoint.safetensors \
    --answerer_model sonnet \
    --retrieval_mode dense_only \
    --max_answer_length 100 
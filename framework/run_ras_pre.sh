export CUDA_VISIBLE_DEVICES=2,3,4,5

# python run_ras.py \
#     --dataset pubhealth \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 50 \
#     --debug


# python run_ras.py \
#     --dataset popqa \
#     --knowledge_source wiki_2020 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2020 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 100
    # --debug


# python run_ras.py \
#     --dataset 2wikimultihop \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only

# export CUDA_VISIBLE_DEVICES=2,3,4,5

# python run_ras.py \
#     --dataset triviaqa \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5
# python run_ras.py \
#     --dataset arc_c \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 50 

python run_ras_pre.py \
    --dataset asqa_train \
    --knowledge_source wiki_2018 \
    --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
    --text_to_triples_model sonnet \
    --planner_model sonnet \
    --answerer_model sonnet \
    --retrieval_mode dense_only \
    --max_answer_length 300

# python run_ras.py \
#     --dataset eli5 \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 300


# export CUDA_VISIBLE_DEVICES=2,3,4,5,6
# python run_ras.py \
#     --dataset popqa \
#     --knowledge_source wiki_2020 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2020 \
#     --text_to_triples_model sonnet \
#     --planner_model llama2-7b \
#     --planner_frozen True \
#     --planner_checkpoint /shared/eng/pj20/firas_data/action_planner/hotpotqa_only/ptune/latest_checkpoint_epoch_lora_True.safetensors \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 100 

# python run_ras.py \
#     --dataset triviaqa \
#     --knowledge_source wiki_2018 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
#     --text_to_triples_model sonnet \
#     --planner_model llama2-7b \
#     --planner_frozen True \
#     --planner_checkpoint /shared/eng/pj20/firas_data/action_planner/hotpotqa_only/ptune/latest_checkpoint_epoch_lora_True.safetensors \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only \
#     --max_answer_length 100 
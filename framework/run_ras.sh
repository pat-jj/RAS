export CUDA_VISIBLE_DEVICES=2,3,4,5

python run_ras.py \
    --dataset pubhealth \
    --knowledge_source wiki_2018 \
    --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
    --text_to_triples_model sonnet \
    --planner_model sonnet \
    --answerer_model sonnet \
    --retrieval_mode dense_only \
    --max_answer_length 50 \
    --debug


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
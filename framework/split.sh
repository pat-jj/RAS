# python split_indices.py --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2020 --num_splits 5
python split_dense.py --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 --num_splits 5
# python split_dense.py --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2020 --num_splits 5

# python run_ras.py \
#     --dataset popqa \
#     --knowledge_source wiki_2020 \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2020 \
#     --text_to_triples_model sonnet \
#     --planner_model sonnet \
#     --answerer_model sonnet \
#     --retrieval_mode dense_only
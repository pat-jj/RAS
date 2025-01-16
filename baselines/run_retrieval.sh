# python theme_dense_ret.py \
#     --query_file /shared/eng/pj20/firas_data/test_datasets/original/popqa.jsonl \
#     --output_file /shared/eng/pj20/firas_data/test_datasets/retrieval/popqa_td_ret.jsonl \
#     --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2020 \
#     --theme_encoder_path /shared/eng/pj20/firas_data/classifiers/best_model \
#     --theme_shifter_path /shared/eng/pj20/firas_data/classifiers/best_distribution_mapper.pt

python theme_dense_ret_gpu_v1.py \
    --query_file /shared/eng/pj20/firas_data/test_datasets/original/popqa.jsonl \
    --output_file /shared/eng/pj20/firas_data/test_datasets/retrieval/popqa_td_ret_gpu_2018.jsonl \
    --knowledge_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2018 \
    --theme_encoder_path /shared/eng/pj20/firas_data/classifiers/best_model \
    --theme_shifter_path /shared/eng/pj20/firas_data/classifiers/best_distribution_mapper.pt \
    --num_workers 15
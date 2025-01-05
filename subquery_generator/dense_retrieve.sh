export CUDA_VISIBLE_DEVICES=1

python encode_queries.py \
    --hotpot_path /shared/eng/pj20/firas_data/datasets/hotpotqa/hotpot_with_subqueries.json \
    --output_path /shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval/queries.pt \

export CUDA_VISIBLE_DEVICES=0,1,2,5,7

python hotpot_retrieve_doc.py \
    --output_dir /shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval \
    --faiss_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding/wikipedia_embeddings.faiss \
    --text_mapping_path /shared/eng/pj20/firas_data/knowledge_source/wiki_2017/embedding/text_mapping.json \
    --queries_embeddings_path /shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval/queries.pt \
    --num_gpus 5
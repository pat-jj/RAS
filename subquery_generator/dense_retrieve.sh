export CUDA_VISIBLE_DEVICES=2,3,6,7

python hotpot_retrieve_doc.py \
    --hotpot_path /shared/eng/pj20/firas_data/datasets/hotpotqa/hotpot_with_subqueries.json \
    --output_dir /shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval \
    --batch_size 32 \
    --num_gpus 4
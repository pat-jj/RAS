# python sonnet_test.py \
# --input_file /shared/eng/pj20/firas_data/test_datasets/original/triviaqa.jsonl \
# --max_new_tokens 100 --metric match \
# --result_fp /shared/eng/pj20/firas_data/test_datasets/results/triviaqa_sonnet.jsonl \
# --task qa \
# --prompt_name prompt_no_input

# python sonnet_test.py \
# --input_file /shared/eng/pj20/firas_data/test_datasets/original/triviaqa.jsonl \
# --max_new_tokens 100 --metric match \
# --result_fp /shared/eng/pj20/firas_data/test_datasets/results/triviaqa_sonnet_retrieval_top_5.jsonl \
# --task qa \
# --prompt_name prompt_no_input_retrieval \
# --mode retrieval \
# --top_n 5


# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/original/pubhealth.jsonl \
#     --max_new_tokens 50 --metric match \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/pubhealth_sonnet.jsonl \
#     --task fever \
#     --prompt_name prompt_no_input


# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/original/pubhealth.jsonl \
#     --max_new_tokens 50 --metric match \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/pubhealth_sonnet_retrieval_top_1.jsonl \
#     --task fever \
#     --prompt_name prompt_no_input_retrieval \
#     --mode retrieval \
#     --top_n 5


# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/original/arc_c.jsonl \
#     --max_new_tokens 10 --metric match \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/arc_c_sonnet.jsonl \
#     --task arc_c \
#     --prompt_name prompt_no_input

# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/original/arc_c.jsonl \
#     --max_new_tokens 10 --metric match \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/arc_c_sonnet_retrieval_top_5.jsonl \
#     --task arc_c \
#     --prompt_name prompt_no_input_retrieval \
#     --mode retrieval \
#     --top_n 5

# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/original/bio.jsonl \
#     --max_new_tokens 300 --metric match \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/bio_sonnet_retrieval_top_5.jsonl \
#     --task factscore \
#     --prompt_name prompt_no_input_retrieval \
#     --mode retrieval \
#     --top_n 5


# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/original/asqa.json \
#     --max_new_tokens 300 --metric match \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/asqa_sonnet_retrieval_top_5_base.jsonl \
#     --task asqa_base \
#     --prompt_name asqa_base \
#     --mode asqa_base \
#     --top_n 5


# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/2wikimultihop_test.json \
#     --max_new_tokens 300 --metric f1 \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/2wikimultihop_sonnet_base.jsonl \
#     --task qa \
#     --prompt_name prompt_no_input

python sonnet_test.py \
    --input_file /shared/eng/pj20/firas_data/test_datasets/2wikimultihop_test.json \
    --max_new_tokens 300 --metric f1 \
    --result_fp /shared/eng/pj20/firas_data/test_datasets/results/2wikimultihop_sonnet_retrieval_top_1.jsonl \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --mode retrieval \
    --top_n 1

python sonnet_test.py \
    --input_file /shared/eng/pj20/firas_data/test_datasets/2wikimultihop_test.json \
    --max_new_tokens 300 --metric f1 \
    --result_fp /shared/eng/pj20/firas_data/test_datasets/results/2wikimultihop_sonnet_retrieval_top_5.jsonl \
    --task qa \
    --prompt_name prompt_no_input_retrieval \
    --mode retrieval \
    --top_n 5


# python sonnet_test.py \
#     --input_file /shared/eng/pj20/firas_data/test_datasets/original/bio.jsonl \
#     --max_new_tokens 300 --metric factscore \
#     --result_fp /shared/eng/pj20/firas_data/test_datasets/results/bio_sonnet_base.jsonl \
#     --task factscore \
#     --prompt_name prompt_no_input
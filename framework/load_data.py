# This is the data loading file for the framework
from datasets import load_dataset
import json 
import jsonlines


NDOCS =5

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


##TODO
def load_hotpot_qa(path):
    pass


def load_adv_hotpot_qa(path):
    pass



'''
python run_long_form_static.py --model_name selfrag/selfrag_llama2_7b  --ndocs 5 --max_new_tokens 300 --threshold 0.2  --use_grounding --use_utility --use_seqscore  --task asqa --input_file eval_data/asqa_eval_gtr_top100.json   --output_file ./out --max_depth 7 --mode always_retrieve 
  
'''
##948 qa samples, long answer question, only used as test data in self-RAG
def load_asqa(path):
    input_data = json.load(open(path))
    ndocs = 5

    test_out_dict = {"question": [], "answer": [], "docs": []}
    for instance_idx, item in enumerate(input_data):
        prompt = item["question"]
        ctxs = item["docs"][:ndocs]  # top 5 relevant docs of the question
        answer = item["answer"]

        test_out_dict["question"].append(prompt)
        test_out_dict["answer"].append(answer)
        test_out_dict["docs"].append(ctxs)

    return test_out_dict



##https://github.com/facebookresearch/ELI5/tree/main
def load_eli5():
    eli5 = load_dataset("defunct-datasets/eli5",trust_remote_code=True)
    eli5_test = eli5['test']
    import pdb; pdb.set_trace()

    pass



##1399 sample, test data for selg-rag. short answer question
def load_pop_qa(path):
    input_data = load_jsonlines(path)
    ndocs = NDOCS

    test_out_dict = {"question": [], "answer": [], "context": [], "s_wiki_title": [], "prop": []}
    for instance_idx, item in enumerate(input_data):
        prompt = item["question"]
        ctxs = item["ctxs"][:ndocs]  # top 5 relevant docs of the question
        answer = item["answers"]
        s_wiki_title = item["s_wiki_title"]
        prop= item["prop"]

        test_out_dict["question"].append(prompt)
        test_out_dict["answer"].append(answer)
        test_out_dict["context"].append(ctxs)
        test_out_dict["s_wiki_title"].append(s_wiki_title)
        test_out_dict["prop"].append(prop)

    return test_out_dict


#test data in self-rag, yes/no question have 987 samples
def load_pubhealth(path):
    input_data = load_jsonlines(path)
    ndocs = NDOCS

    test_out_dict = {"question": [], "answer": [], "context": [], "claim": [], "label": []}
    for instance_idx, item in enumerate(input_data):
        prompt = item["question"]
        ctxs = item["ctxs"][:ndocs]  # top 5 relevant docs of the question
        answer = item["answers"]
        claim = item["claim"]
        label= item["label"]

        test_out_dict["question"].append(prompt)
        test_out_dict["answer"].append(answer)
        test_out_dict["context"].append(ctxs)
        test_out_dict["claim"].append(claim)
        test_out_dict["label"].append(label)

    return test_out_dict



###2wiki multi-hop data, 12576 sample, test data.
# https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file
def load_2wiki_multi_hop(path):
    ##read the json file
    test_out_dict = {"question": [], "answer": [], "context": []}
    with open(path, 'r') as f:
        input_data = json.load(f)
        
    for instance_idx, item in enumerate(input_data):
        prompt = item["question"]
        ctxs = item["context"]  # top 5 relevant docs of the question
        answer = item["answer"]

        test_out_dict["question"].append(prompt)
        test_out_dict["answer"].append(answer)
        test_out_dict["context"].append(ctxs)

    return test_out_dict



#https://github.com/KGQA/QALD-10
def load_qald10(path):
    pass




#https://github.com/sheffieldnlp/naacl2018-fever/blob/master/scripts/download-data.sh
#https://github.com/sheffieldnlp/naacl2018-fever/blob/master/scripts/download-paper.sh
# training data, 
def load_fever(path):  
    pass
    # input_data = load_jsonlines(path)
    # ndocs = NDOCS

    # train_out_dict = {"question": [], "answer": [], "context": [], "claim": [], "label": []}
    # for instance_idx, item in enumerate(input_data):
    #     ctxs = item["evidence"][:ndocs]  # top 5 relevant docs of the question
    #     claim = item["claim"]
    #     label= item["label"]

    #     train_out_dict["question"].append(prompt)
    #     train_out_dict["answer"].append(answer)
    #     train_out_dict["context"].append(ctxs)
    #     train_out_dict["claim"].append(claim)
    #     train_out_dict["label"].append(label)
    # return train_out_dict




# load_asqa('../eval_data/asqa_eval_gtr_top100.json')
# load_pubhealth('../eval_data/health_claims_processed.jsonl')
# load_pop_qa('../eval_data/popqa_longtail_w_gs.jsonl')
# load_2wiki_multi_hop('../eval_data/2wikimultihop/test.json')
load_eli5()
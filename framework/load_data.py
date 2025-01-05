# This is the data loading file for the framework
from datasets import load_dataset
import json 
import jsonlines


NDOCS =5

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst



################################################
#     download the data from self-rag repo     #
################################################

#python run_long_form_static.py --model_name selfrag/selfrag_llama2_7b \
#  --ndocs 5 --max_new_tokens 300 --threshold 0.2  --use_grounding --use_utility \
# --use_seqscore  --task asqa --input_file eval_data/asqa_eval_gtr_top100.json \
#   --output_file ./out --max_depth 7 --mode always_retrieve 
  
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

    ###save it as json file
    with open('./out_file/asqa_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]) or len(test_out_dict["question"])!=len(test_out_dict["docs"]):
        print("asaq Error: the length of the question, answer, and context is not the same.")
    return True


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

    ###save it as json file
    with open('./out_file/popqa_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]) or len(test_out_dict["question"])!=len(test_out_dict["ctxs"]):
        print("popqa Error: the length of the question, answer, and context is not the same.")

    return True


#test data in self-rag, yes/no question, have 987 samples
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

    ###save it as json file
    with open('./out_file/pubhealth_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]) or len(test_out_dict["question"])!=len(test_out_dict["ctxs"]):
        print("pubhealth Error: the length of the question, answer, and context is not the same.")

    return True


###2wiki multi-hop data, 12576 sample, test data for self-rag.
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


    ###save it as json file
    with open('./out_file/2wikimultihop_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]) or len(test_out_dict["question"])!=len(test_out_dict["context"]):
        print("2wikimultihop Error: the length of the question, answer, and context is not the same.")

    return True




##############################################################################
#    download from ToG repo, https://github.com/IDEA-FinAI/ToG-2/tree/main   #
##############################################################################
#https://github.com/IDEA-FinAI/ToG-2/blob/72e8b737d95e54fe742b652fd78d269865f840b7/eval/utils.py
# test data for ToG2
def load_fever(path):  
    with open(path, 'r') as f:
        input_data = json.load(f)
    test_out_dict = {"question": [], "answer": [], "possible_answer_dict":[]}
    
    possible_answer_dict = {
            'REFUTES': ['refutes', 'refute', 'false', 'incorrect', 'not accurate', 'not true', 'not correct',
                        'does not make sense', 'not entirely accurate', 'incomplete'],
            'SUPPORTS': ['supports', 'support', 'true', 'correct'],
            'NOT ENOUGH INFO': ['not enough information', 'not enough info']
        }
    
    
    for instance_idx, item in enumerate(input_data):
        answer = item['label']
        question = item['claim']
        
        alt_ans = possible_answer_dict[answer]
        test_out_dict["question"].append(question)
        test_out_dict["answer"].append(answer)
        test_out_dict["possible_answer_dict"].append(alt_ans)

    ###save it as json file
    with open('./out_file/fever_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]) or len(test_out_dict["question"])!=len(test_out_dict["possible_answer_dict"]):
        print("fever Error: the length of the question, answer, and context is not the same.")
    
    return True
   

# 333 QA pairs, multiple answer, test data for ToG2
def load_qald10(path):
    test_out_dict = {"question": [], "answer": []}
    with open(path, 'r') as f:
        input_data = json.load(f)
        
    for instance_idx, item in enumerate(input_data):
        prompt = item["question"]
        answer = item["answer"].values()

        test_out_dict["question"].append(prompt)
        test_out_dict["answer"].append(answer)

    ###save it as json file
    with open('./out_file/qald10_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]):
        print("qald10 Error: the length of the question and answer is not the same.")
 
    return True


# 308 QA pairs, short answer, test data for ToG2
def load_adv_hotpot_qa(path):
    test_out_dict = {"question": [], "answer": []}
    with open(path, 'r') as f:
        input_data = json.load(f)
        
    for instance_idx, item in enumerate(input_data):
        prompt = item["question"]
        answer = item["answer"]

        test_out_dict["question"].append(prompt)
        test_out_dict["answer"].append(answer)

    ###save it as json file
    with open('./out_file/hotpotadv_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]):
        print("hotpotadv Error: the length of the question and answer is not the same.")

    return True



##  TODO
def load_hotpot_qa(path):
    pass







######################################################################################
#   test data for RPG, https://github.com/haruhi-sudo/RPG, no data source provided   #
######################################################################################
## download from this link: https://github.com/HCY123902/atg-w-fg-rw/tree/main/tasks/qa_feedback/data
# inputs are ambiguous questions with multiple interpretations, same as asaq
def load_eli5(path):
    test_out_dict = {"question": [], "answer": [], "context": []}
    with open(path, 'r') as f:
        input_data = json.load(f)
        
    for instance_idx, item in enumerate(input_data):
        prompt = item["question"]
        ctxs = item["docs"]["text"]  # top 5 relevant docs of the question
        import pdb; pdb.set_trace()
        answer = item["answer"]

        test_out_dict["question"].append(prompt)
        test_out_dict["answer"].append(answer)
        test_out_dict["context"].append(ctxs)

    ###save it as json file
    with open('./out_file/eli5_test.json', 'w') as f:
        json.dump(test_out_dict, f)

    if len(test_out_dict["question"])!=len(test_out_dict["answer"]) or len(test_out_dict["question"])!=len(test_out_dict["context"]):
        print("eli5 Error: the length of the question, answer, and context is not the same.")

    return True

if __name__ == '__main__':
    # out = load_asqa('../eval_data/asqa_eval_gtr_top100.json')
    # out = load_pubhealth('../eval_data/health_claims_processed.jsonl')
    # out = load_pop_qa('../eval_data/popqa_longtail_w_gs.jsonl')
    # out = load_2wiki_multi_hop('../eval_data/2wikimultihop/test.json')

    # out = load_qald10('../eval_data/data/qald_10-en.json')
    # out = load_adv_hotpot_qa('../eval_data/data/hotpotadv_dev.json')
    # out = load_fever('../eval_data/data/fever_1000.json')

    out = load_eli5('../eval_data/eli5_test.json')
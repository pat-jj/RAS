import torch
import torch.nn.parallel
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data



def get_planner_instruction(model_name):
    if model_name != 'sonnet':
        return """You are a planner to determine if the question can be answered with current information and output the appropriate label as well as the subquery if needed.
Output [NO_RETRIEVAL] if the question can be directly answered with the question itself without any retrieval.
Output [SUBQ] with an subquery for retrieval if still needs a subquery.
Output [SUFFICIENT] if the question can be answered with the provided information.
"""


def get_answerer_instruction(model_name):
    if model_name != 'sonnet':
        return """You are a answerer given a question and retrieved graph information.
Each [SUBQ] is a subquery we generated through reasoning for the question. The retrieved graph information follows each [SUBQ] is relevant graph information we retrieved to answer the subquery.
[NO_RETRIEVAL] means the question can be answered with the question itself without any retrieval.
The main question starts with "Question: ". Please answer the question, with subqueries and retrieved graph information if they are helpful.
"""


class MultilabelClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_labels)
        )
        
    def forward(self, embeddings):
        return torch.sigmoid(self.classifier(embeddings))

class DistributionMapper(nn.Module):
    def __init__(self, input_dim):
        super(DistributionMapper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        logits = self.model(x)
        # Add small epsilon before softmax for numerical stability
        return torch.softmax(logits + 1e-15, dim=1)
    
    
def load_theme_classifier(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    
    # Load model configuration
    with open(checkpoint_path / "config.json", 'r') as f:
        config = json.load(f)
    
    # Initialize models
    sentence_transformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    embed_dim = sentence_transformer.get_sentence_embedding_dimension()
    classifier = MultilabelClassifier(embed_dim, len(config['label_mapping']))
    
    # Load saved states to CPU first
    model_state = torch.load(
        checkpoint_path / "model_state.pt",
        map_location='cpu'    # Load to CPU regardless of where it was saved
    )
    
    classifier.load_state_dict(model_state['classifier_state'])
    sentence_transformer.load_state_dict(model_state['encoder_state'])
    
    # Models can be moved to the desired device later when needed
    # classifier.to(device)
    # sentence_transformer.to(device)
    
    return classifier, sentence_transformer, config['label_mapping']


def load_theme_distribution_shifter(checkpoint_path: str, input_dim: int):
    """Load the theme distribution shifter model from checkpoint"""
    distribution_mapper = DistributionMapper(input_dim)
    
    # Load saved states to CPU first
    model_state = torch.load(
        checkpoint_path,
        map_location='cpu'    # Load to CPU regardless of where it was saved
    )
    
    distribution_mapper.load_state_dict(model_state)
    
    return distribution_mapper



class GraphProcessor:
    def __init__(self, bert_model='sentence-transformers/all-roberta-large-v1'):
        """
        Initialize the data processor.
        Args:
            bert_model (str): Name of the SentenceBERT model to use for embeddings
        """
        self.sentence_model = SentenceTransformer(bert_model)
        self.embed_dim = self.sentence_model.get_sentence_embedding_dimension()
    

    def create_graph_from_triples(self, triple_strs):
        """Convert a list of triple strings into a PyG graph with predicate encodings"""
        nodes = set()
        edge_triples = []
        
        # Collect unique nodes and edges
        for triple_str in triple_strs:
            # Keep original triple string for description
            triple_str = triple_str.strip('()')
            parts = triple_str.split('|')
            
            # Extract subject, predicate, object
            subject = parts[0].replace('S>', '').strip()
            predicate = parts[1].replace('P>', '').strip()
            object_ = parts[2].replace('O>', '').strip()
            
            nodes.add(subject)
            nodes.add(object_)
            edge_triples.append((subject, predicate, object_))
        
        # Create node mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge index and collect predicates
        edge_index = []
        predicates = []
        
        for subj, pred, obj in edge_triples:
            # Add forward edge
            edge_index.append([node_to_idx[subj], node_to_idx[obj]])
            predicates.append(pred)  # Original predicate
            
            # Add reverse edge
            edge_index.append([node_to_idx[obj], node_to_idx[subj]])
            predicates.append(f"inverse_{pred}")  # Inverse predicate
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create node features (embeddings)
        node_texts = list(nodes)
        node_embeddings = self.sentence_model.encode(node_texts)
        node_features = torch.tensor(node_embeddings, dtype=torch.float)
        
        # Create edge features (only encode predicates)
        predicate_embeddings = self.sentence_model.encode(predicates)
        edge_features = torch.tensor(predicate_embeddings, dtype=torch.float)
        
        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )
        
        return graph
    


def text_to_triples(t2t_tokenizer, t2t_model, texts):
    inputs = t2t_tokenizer(texts, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    
    input_ids = inputs['input_ids'].to(t2t_model.device)
    attention_mask = inputs['attention_mask'].to(t2t_model.device)

    with torch.no_grad():
        outputs = t2t_model.generate(input_ids=input_ids, attention_mask=attention_mask,
            max_length=512, num_beams=4, early_stopping=True, length_penalty=0.6, use_cache=True)
    
    triples = t2t_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return triples


def clean_document(doc):
    """Filter out invalid or dirty documents"""
    # Remove documents that are too short
    if len(doc.strip()) <3:
        return False
        
    # Remove documents that are mostly special characters or punctuation
    special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in doc) / len(doc)
    if special_char_ratio > 0.3:  # if more than 30% are special characters
        return False
        
    # Remove documents that contain emojis or other unicode symbols
    if any(ord(c) > 127 for c in doc):
        return False
        
    # Remove documents that are just punctuation or dots
    if doc.strip() in ['.', '...', '. . .']:
        return False
        
    return True

## Self-RAG's utils

import jsonlines
import json
import copy
import re

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
}

# in ALCE's official github repo (https://github.com/princeton-nlp/ALCE/blob/main/prompts/asqa_default.json)
ALCE_2_SHOT_INST = """Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.

Question: Which is the most rainy place on earth?

Document [1](Title: Cherrapunji): Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in
Document [2](Title: Cherrapunji): Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw "hima" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall
Document [3](Title: Mawsynram): Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the "Guinness Book of World Records", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′
Document [4](Title: Earth rainfall climatology): Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of
Document [5](Title: Going to Extremes): in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where "survival"

Answer: Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861 [1].


Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.

Question: When did the us break away from england?

Document [1](Title: United States withdrawal from Saudi Arabia): United States withdrawal from Saudi Arabia Beginning during Operation Desert Shield in August 1990, while preparing for the Gulf War, the United States sent a large troop contingent to Saudi Arabia. After the war, remnant troops, primarily U.S. Air Force personnel, augmented by a smaller number of coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The United Kingdom and France also maintained a small contingent of Royal Air Force and French Air Force
Document [2](Title: Decolonization of the Americas): and France has fully "integrated" most of its former colonies as fully constituent "departments" of France. The United States of America declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, the date when the Declaration of Independence was officially adopted by Congress), in so doing becoming the first independent, foreign-recognized nation in the Americas and the first European colonial entity to break from its mother country. Britain formally acknowledged American independence in 1783 after its defeat in the American Revolutionary War. Although initially occupying only the land east of the Mississippi
Document [3](Title: American Revolution): second British army at Yorktown in the fall of 1781, effectively ending the war. The Treaty of Paris was signed September 3, 1783, formally ending the conflict and confirming the new nation's complete separation from the British Empire. The United States took possession of nearly all the territory east of the Mississippi River and south of the Great Lakes, with the British retaining control of Canada and Spain taking Florida. Among the significant results of the revolution was the creation of the United States Constitution, establishing a relatively strong federal national government that included an executive, a national judiciary, and
Document [4](Title: Decolonization): accelerate decolonialization and bring an end to the colonial empires of its Western allies, most importantly during the 1956 Suez Crisis, but American military bases were established around the world and direct and indirect interventions continued in Korea, Indochina, Latin America ("inter alia", the 1965 occupation of the Dominican Republic), Africa, and the Middle East to oppose Communist invasions and insurgencies. Since the dissolution of the Soviet Union, the United States has been far less active in the Americas, but invaded Afghanistan and Iraq following the September 11 attacks in 2001, establishing army and air bases in Central Asia. Before
Document [5](Title: Decolonization): the responsibility of the United Kingdom (with a copy of the new constitution annexed), and finally, if approved, issuance of an Order of Council fixing the exact date of independence. After World War I, several former German and Ottoman territories in the Middle East, Africa, and the Pacific were governed by the UK as League of Nations mandates. Some were administered directly by the UK, and others by British dominions – Nauru and the Territory of New Guinea by Australia, South West Africa by the Union of South Africa, and Western Samoa by New Zealand. Egypt became independent in 1922,

Answer: The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress) [2]. The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire [3].


"""


TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false in lowercase. Do not say anything else, and only output true or false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice. Only output the letter of the answer choice, e.g. A, B, C, or D. Do not say anything else, and only output A, B, C, or D.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
             "asqa_original": ALCE_2_SHOT_INST + "\n" + "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."
             
             }

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                          "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                       "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def preprocess_input(input_data, task):
    if task == "factscore":
        for item in input_data:
            item["instruction"] = item["input"]
            item["output"] = [item["output"]
                              ] if "output" in item else [item["topic"]]
        return input_data

    elif task == "qa":
        for item in input_data:
            if "instruction" not in item:
                item["instruction"] = item["question"]
            if "answers" not in item and "output" in item:
                item["answers"] = "output"
        return input_data

    elif task in ["asqa", "eli5"]:
        processed_input_data = []
        for instance_idx, item in enumerate(input_data["data"]):
            prompt = item["question"]
            instructions = TASK_INST[task]
            prompt = instructions + "## Input:\n\n" + prompt
            entry = copy.deepcopy(item)
            entry["instruction"] = prompt
            processed_input_data.append(entry)
        return processed_input_data


def postprocess_output(input_instance, prediction, task, intermediate_results=None):
    if task == "factscore":
        return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

    elif task == "qa":
        input_instance["pred"] = prediction
        return input_instance

    elif task in ["asqa", "eli5"]:
        # ALCE datasets require additional postprocessing to compute citation accuracy.
        final_output = ""
        docs = []
        if "splitted_sentences" not in intermediate_results:
            input_instance["output"] = postprocess(prediction)

        else:
            for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = postprocess(sent)
                final_output += postprocessed_result[:-
                                                     1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if final_output[-1] == " ":
                final_output = final_output[:-1]
            input_instance["output"] = final_output
        input_instance["docs"] = docs
        return input_instance

def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key in ["A", "B", "C", "D"]:
            answer_labels[answer_key] = text

    if "D" not in answer_labels:
        answer_labels["D"] = ""
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
    return processed_instruction


def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output
    
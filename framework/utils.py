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
        
        return graph, triple_strs 
    


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
ALCE_2_SHOT_INST_BASE ="""Instruction: Write an ACCURATE, ENGAGING, and CONCISE answer for the given question. Use an unbiased and journalistic tone. The answer should be a single concise paragraph without any listing or line breaks. Only output the answer, do not say anything else.

Question: Which is the most rainy place on earth?
Answer: Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm, although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861.

Question: When did the us break away from england?
Answer: The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress). The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire.

Question: """


ALCE_2_SHOT_INST = """You are a helpful assistant that answers the following questions with proper citations.

Instruction: Write an ACCURATE, ENGAGING, and CONCISE answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite AT LEAST ONE document and AT MOST THREE documents in EACH sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. The answer should be a single concise paragraph without any listing or line breaks. Only output the answer, do not say anything else.


Example 1:
Question: Which is the most rainy place on earth?

Document [1](Title: Cherrapunji): Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in
Document [2](Title: Cherrapunji): Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw "hima" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall
Document [3](Title: Mawsynram): Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the "Guinness Book of World Records", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′
Document [4](Title: Earth rainfall climatology): Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of
Document [5](Title: Going to Extremes): in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where "survival"

Answer: Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861 [1].


Example 2:
Question: When did the us break away from england?

Document [1](Title: United States withdrawal from Saudi Arabia): United States withdrawal from Saudi Arabia Beginning during Operation Desert Shield in August 1990, while preparing for the Gulf War, the United States sent a large troop contingent to Saudi Arabia. After the war, remnant troops, primarily U.S. Air Force personnel, augmented by a smaller number of coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The United Kingdom and France also maintained a small contingent of Royal Air Force and French Air Force
Document [2](Title: Decolonization of the Americas): and France has fully "integrated" most of its former colonies as fully constituent "departments" of France. The United States of America declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, the date when the Declaration of Independence was officially adopted by Congress), in so doing becoming the first independent, foreign-recognized nation in the Americas and the first European colonial entity to break from its mother country. Britain formally acknowledged American independence in 1783 after its defeat in the American Revolutionary War. Although initially occupying only the land east of the Mississippi
Document [3](Title: American Revolution): second British army at Yorktown in the fall of 1781, effectively ending the war. The Treaty of Paris was signed September 3, 1783, formally ending the conflict and confirming the new nation's complete separation from the British Empire. The United States took possession of nearly all the territory east of the Mississippi River and south of the Great Lakes, with the British retaining control of Canada and Spain taking Florida. Among the significant results of the revolution was the creation of the United States Constitution, establishing a relatively strong federal national government that included an executive, a national judiciary, and
Document [4](Title: Decolonization): accelerate decolonialization and bring an end to the colonial empires of its Western allies, most importantly during the 1956 Suez Crisis, but American military bases were established around the world and direct and indirect interventions continued in Korea, Indochina, Latin America ("inter alia", the 1965 occupation of the Dominican Republic), Africa, and the Middle East to oppose Communist invasions and insurgencies. Since the dissolution of the Soviet Union, the United States has been far less active in the Americas, but invaded Afghanistan and Iraq following the September 11 attacks in 2001, establishing army and air bases in Central Asia. Before
Document [5](Title: Decolonization): the responsibility of the United Kingdom (with a copy of the new constitution annexed), and finally, if approved, issuance of an Order of Council fixing the exact date of independence. After World War I, several former German and Ottoman territories in the Middle East, Africa, and the Pacific were governed by the UK as League of Nations mandates. Some were administered directly by the UK, and others by British dominions – Nauru and the Territory of New Guinea by Australia, South West Africa by the Union of South Africa, and Western Samoa by New Zealand. Egypt became independent in 1922,

Answer: The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress) [2]. The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire [3].


Your turn:
Question: """


ELI5_2_SHOT_INST_BASE = """Instruction: Write an ACCURATE, ENGAGING, and CONCISE answer for the given question. Use an unbiased and journalistic tone. Follow the writing style and length of the answers shown in the examples. The answer should be a single concise paragraph without any listing or line breaks. Only output the answer, do not say anything else.

Question: Why did New York City try to ban food donations to the poor?
Answer: New York City, under Mayor Michael Bloomberg's administration, banned citizens from donating food directly to homeless shelters because the city could not assess the salt, fat, and fiber content. Bloomberg's administration was heavily criticized for losing their common sense by becoming too focused on what people eat.

Question: What's the difference between Shia vs. Sunni Islam?
Answer: The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali. Nowadays, Sunni and Shia are the major branches of Islam.

Question: """



ELI5_2_SHOT_INST = """You are a helpful assistant that answers the following questions with proper citations.

Instruction: Write an ACCURATE, ENGAGING, and CONCISE answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite AT LEAST ONE document and AT MOST THREE documents in EACH sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Follow the writing style (one paragraph of one or more sentences) of the answers shown in the examples. Follow the writing style and length of the answers shown in the examples. The answer should be a single concise paragraph without any listing or line breaks. Only output the answer, do not say anything else.


Example 1:
Question: Why did New York City try to ban food donations to the poor?

Document [1]: believe that they are \u201chelping\u201d the homeless by passing such laws. In New York City, Mayor Bloomberg has banned citizens from donating food directly to homeless shelters and he is actually convinced that it was the right thing to do for the homeless\u2026 Mayor Michael Bloomberg\u2019s food police have struck again! Outlawed are food donations to homeless shelters because the city can\u2019t assess their salt, fat and fiber content, reports CBS 2\u2019s Marcia Kramer. Glenn Richter arrived at a West Side synagogue on Monday to collect surplus bagels \u2014 fresh nutritious bagels \u2014 to donate to the poor..
Document [2]: muck: Bloomberg Bans Food Donations in New York City Food Might Be Salty or Too High in Calories, City Explains Washington, D.C. \u2013 New York Mayor Michael Bloomberg\u2019s administration is now banning all food being offered to the city\u2019s homeless shelters. New York City\u2019s bureaucrats have become so singularly focused on what people eat, says the National Center for Public Policy Research, that they\u2019ve lost their common sense. \u201cSo much for serving the homeless: The Bloomberg administration is now taking the term \u2018food police\u2019 to new depths, blocking food donations to all government-run facilities that serve the
Document [3]: New York City bans food donations - WND Front Page Health U.S. New York City bans food donations Inability to control 'nutritional content' cited as reason New York City homeless shelters have Mayor Michael Bloomberg to thank for a halt in food donations, for which hungry families are waiting, according to one public policy advocate. \"The Bloomberg administration is now taking the term 'food police' to new depths, blocking food donations to all government-run facilities that serve the city's homeless,\" says Jeff Stier, a National Center for Public Policy Research senior fellow. Currently, no food can be given to government-run, New York City facilities, despite hungry crowds perfectly
Document [4]: New York City bans food donations - WND Services didn't return WND calls. Stier told WND that he specifically was told by Diamond that the policy was tied to the nutritional guidelines set by the mayor. \"They can say that this ban on donations is a long-standing policy, but they can\u2019t document it,\" Stier told WND. \"I've also been told that there are numerous food shelves that have been accepting food donations, not just one.\" Stier is a member of a New York Synagogue that has donated food for over a decade. He is outraged that the DHS' response to his demand to know why the practice can
Document [5]: New York City bans food donations - WND ban on donated food. In fact, it thrives because of food donations. New York City Rescue Mission has been providing food, clothing, shelter and spiritual hope for needy New Yorkers since 1872. \"We feed over 500 people a day, all through donations,\" said James Varnhagen, NYCRM director. \"Boxed food, canned food, prepared food, we take any food,\" he told WND. \"We couldn't survive without donations,\" he said.

Answer: New York City, under Mayor Michael Bloomberg's administration, banned citizens from donating food directly to homeless shelters because the city could not assess the salt, fat, and fiber content [1][2][3]. Bloomberg's administration was heavily criticized for losing their common sense by becoming too focused on what people eat [2].


Example 2:
Question: What's the difference between Shia vs. Sunni Islam?

Document [1]: centuries-long strained relationship between Sunnis and Shias. As a scholar of Islam and a public educator, I often field questions about Sunnis, Shias and the sects of Islam. What exactly is the Shia-Sunni divide? And what is its history? History of divide Both Sunnis and Shias \u2013 drawing their faith and practice from the Qur\u2019an and the life of the Prophet Muhammad \u2013 agree on most of the fundamentals of Islam. The differences are related more to historical events, ideological heritage and issues of leadership. The first and central difference emerged after the death of Prophet Muhammad in A.D. 632.
Document [2]: What\u2019s the difference between Sunni and Shia Islam? Sunni and Shia identities (the 2 main branches of Islam) first formed around a dispute over leadership succession after the death of the Prophet Muhammad in 632 A.D. Sunni is the larger branch (estimated 85-90% of total world Muslim population) and it's adherents are referred to as \"people of the tradition of Muhammad\", while Shia are \"followers\" of Muhammad's son-in-law and cousin Ali. Sunnis rely heavily on the practice of the Prophet Muhammad and his teachings, the Shia view their ayatollahs as reflections of God on earth. What challenges does the anti-IS
Document [3]: of Muhammad, the last prophet of God. A follower of Islam is known as a Muslim. Many Muslims believe that their sole purpose is to worship and serve God, for which they have established five pillars of Islam that guides a Muslim on almost every aspect of life and society. Due to differences, Muslims have been divided into two primary sects: The Sunnis and the Shias. These two sects have many similarities and both consider themselves are Muslims, following the will of God. However, they are also different from each other in certain aspects. Both the Sunnis and the Shias
Document [4]: What is the difference between Shia and Sunni Islam? - Islam Stack Exchange between Mutah marriage and Misyar marriage? What theological and historical factors distinguish Ibadi Islam from either Shia or Sunni schools? What are the principle/fundamental differences between Sunni and Shia? Nikah between a Sunni girl and Shia boy What is the difference between \u201cMubtalat-of-Wudu\u201d of Shia and Sunni? How can the Hadith be reliable when Sunnis and Shia follow different points of reference? Rejection of Mutawatir Hadith in Sunni Islam and Shia Islam
Document [5]: What is the difference between Sunni and Shia Islam? | Patrick Syder Travel What is the difference between Sunni and Shia Islam? This Channel 4 link answers some of the key questions about the difference between Sunni and Shia Islam and alarmingly, the politics on what is happening and why, in Syria\u2026\u2026. http://www.channel4.com/news/sunni-shia-islam-muslim-syria-middle-east-key-questions \u2190 Ethiopia Appeal \u2013 Help sponsor a nurse to train and to help others G\u00f6bekli Tepe, Turkey: a new wonder of the ancient world by Jeremy Seal (Telegraph Travel Section 23/04/2013) \u2192

Answer: The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership [1]. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. [1][2]. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali [2]. Nowadays, Sunni and Shia are the major branches of Islam [3].


Your turn:
Question: """


ALCE_2_SHOT_TRIPLES_INST = """Instruction: Write an ACCURATE, ENGAGING, and CONCISE answer for the given question using the retrieved graph information (some of which might be irrelevant). Use an unbiased and journalistic tone. Follow the writing style and length of the answers shown in the examples. The answer should be a single concise paragraph without any listing or line breaks. Only output the answer, do not say anything else.

Examples:

Question: Which is the most rainy place on earth?
Retrieved Graph Information: (S> Cherrapunji| P> Alternative name| O> Sohra), (S> Cherrapunji| P> Located in| O> East Khasi Hills district), (S> Cherrapunji| P> Located in state| O> Meghalaya), (S> Cherrapunji| P> Located in country| O> India), (S> Cherrapunji| P> Traditional capital of| O> Nongkhlaw hima), (S> Cherrapunji| P> Previously credited as| O> Wettest place on Earth), (S> Cherrapunji| P> Holds record for| O> Most rainfall in a calendar month), (S> Cherrapunji| P> Record month| O> July 1861), (S> Cherrapunji| P> Holds record for| O> Most rain in a year), (S> Cherrapunji| P> Record year| O> August 1860 to July 1861), (S> Mawsynram| P> Located in| O> East Khasi Hills district), (S> Mawsynram| P> Located in state| O> Meghalaya), (S> Mawsynram| P> Located in country| O> India), (S> Mawsynram| P> Distance from Shillong| O> 65 kilometres), (S> Mawsynram| P> Currently considered| O> Wettest place on Earth), (S> Mawsynram| P> Average annual rainfall| O> 11,872 mm), (S> Mawsynram| P> Rainfall record| O> 26,000 mm), (S> Mawsynram| P> Rainfall record year| O> 1985), (S> Lloró| P> Located in country| O> Colombia), (S> Lloró| P> Average yearly rainfall| O> 12,717 mm), (S> Lloró| P> Rainfall measurement period| O> 1952 to 1989), (S> López de Micay| P> Located in country| O> Colombia), (S> López de Micay| P> Annual rainfall| O> 12,892 mm), (S> López de Micay| P> Rainfall measurement period| O> 1960 to 2012), (S> Big Bog| P> Located on| O> Maui island), (S> Big Bog| P> Average annual rainfall| O> 404 inches), (S> Big Bog| P> Considered| O> Wettest location in US and Oceania)
Answer: Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 . However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm , although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861.


Question: When did the us break away from england?
Retrieved Graph Information: (S> United States of America| P> Declared independence from| O> Great Britain), (S> United States of America| P> Declaration date| O> July 2, 1776), (S> Declaration of Independence| P> Adoption date| O> July 4, 1776), (S> Britain| P> Acknowledged independence of| O> United States of America), (S> Britain| P> Acknowledgment year| O> 1783), (S> Treaty of Paris| P> Signing date| O> September 3, 1783), (S> Treaty of Paris| P> Confirmed| O> United States separation from British Empire), (S> United States| P> Took possession of| O> Territory east of Mississippi River), (S> United States| P> Took possession of| O> Territory south of Great Lakes), (S> Britain| P> Retained control of| O> Canada), (S> Spain| P> Took control of| O> Florida)
Answer: The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress) . The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire.


Your turn:
Question: """


from sonnet import *
def ras_asqa_sonnet(args, models, question, context, graph_processor, retriever):
    answerer_instruction = ALCE_2_SHOT_TRIPLES_INST
    triples = text_to_triples_sonnet("Question: " + question + "\nRetrieval:" + "\n".join(context)).replace("\n", " ")
    print("Triples: ", triples)

    answerer_input = answerer_instruction + question + "\nRetrieved Graph Information:" + triples + "\nAnswer:"
    generated_answer = get_claude_response(llm="sonnet", prompt=answerer_input, max_tokens=args.max_answer_length)
    print("Generated answer: ", generated_answer)
    return generated_answer, [], [triples], [question], [answerer_input]


ELI5_2_SHOT_TRIPLES_INST = """Instruction: Write an ACCURATE, ENGAGING, and CONCISE answer for the given question using the retrieved graph information (some of which might be irrelevant). Use an unbiased and journalistic tone. Follow the writing style and length of the answers shown in the examples. The answer should be a single concise paragraph without any listing or line breaks. Only output the answer, do not say anything else.

Question: Why did New York City try to ban food donations to the poor?
Retrieved Graph Information: (S> New York City| P> Attempted to ban| O> Food donations to the poor), (S> Mayor Bloomberg| P> Banned| O> Citizens from donating food directly to homeless shelters), (S> New York City| P> Cannot assess| O> Salt, fat and fiber content of donated food), (S> Glenn Richter| P> Attempted to donate| O> Surplus bagels to the poor), (S> Bloomberg administration| P> Blocked| O> Food donations to government-run facilities), (S> New York City| P> Cited reason for ban| O> Inability to control nutritional content), (S> Jeff Stier| P> Criticized| O> Bloomberg administration's food donation ban), (S> New York City Rescue Mission| P> Relies on| O> Food donations), (S> New York City Rescue Mission| P> Feeds| O> Over 500 people a day through donations), (S> James Varnhagen| P> Is director of| O> New York City Rescue Mission)
Answer: New York City, under Mayor Michael Bloomberg's administration, banned citizens from donating food directly to homeless shelters because the city could not assess the salt, fat, and fiber content. Bloomberg's administration was heavily criticized for losing their common sense by becoming too focused on what people eat.

Question: What's the difference between Shia vs. Sunni Islam?
Retrieved Graph Information: (S> Sunni Islam| P> Percentage of world Muslim population| O> 85-90%), (S> Sunni Muslims| P> Referred to as| O> People of the tradition of Muhammad), (S> Shia Muslims| P> Referred to as| O> Followers of Muhammad's son-in-law and cousin Ali), (S> Sunni-Shia divide| P> Originated from| O> Dispute over leadership succession), (S> Sunni-Shia divide| P> Began after| O> Death of Prophet Muhammad in 632 A.D.), (S> Sunni Muslims| P> Rely heavily on| O> Practice and teachings of Prophet Muhammad), (S> Shia Muslims| P> View as reflections of God on earth| O> Ayatollahs), (S> Islam| P> Has| O> Five pillars), (S> Five pillars of Islam| P> Guide Muslims on| O> Almost every aspect of life and society), (S> Sunni and Shia| P> Are| O> Two primary sects of Islam), (S> Sunni and Shia| P> Share| O> Many similarities), (S> Sunni and Shia| P> Differ in| O> Certain aspects)
Answer: The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali. Nowadays, Sunni and Shia are the major branches of Islam.

Question: """

def ras_eli5_sonnet(args, models, question, context, graph_processor, retriever):
    answerer_instruction = ELI5_2_SHOT_TRIPLES_INST
    triples = text_to_triples_sonnet("Question: " + question + "\nRetrieval:" + "\n".join(context)).replace("\n", " ")
    print("Triples: ", triples)

    answerer_input = answerer_instruction + question + "\nRetrieved Graph Information:" + triples + "\nAnswer:"
    generated_answer = get_claude_response(llm="sonnet", prompt=answerer_input, max_tokens=args.max_answer_length)
    print("Generated answer: ", generated_answer)
    return generated_answer, [], [triples], [question], [answerer_input]


TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false in lowercase. Do not say anything else, and only output true or false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice. Only output the letter of the answer choice, e.g. A, B, C, or D. Do not say anything else, and only output A, B, C, or D.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
             "asqa_original": ALCE_2_SHOT_INST + "\n" + "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
             "2wikimultihop": "Answer the following question. Just output the answer (even if you are not sure), do not say anything else."

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
    
    
def convert_triple_str_to_list(triple_str):
        
    triples = [triple + ')' if not triple.endswith(')') else triple 
                                for triple in triple_str.split('), ')] 
    return triples

def convert_triple_str_to_graph(triple_str, graph_processor):
    triples = convert_triple_str_to_list(triple_str)
    try:
        graph, triples = graph_processor.create_graph_from_triples(triples)
    except Exception as e:
        print(f"Error processing triples: {triples}")
        print(f"Error: {e}")
        return None, None
    return graph, triples

    
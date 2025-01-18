import numpy as np
import string
import re
from collections import Counter
import re
import string
import re
from nltk import sent_tokenize
from rouge_score import rouge_scorer, scoring
import mauve

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if normalize_answer(pred) == normalize_answer(target):
            match_count += 1

    return 100 * (match_count / len(preds))


# def f1(decoded_preds, decoded_labels):
#     f1_all = []
#     for prediction, answers in zip(decoded_preds, decoded_labels):
#         if type(answers) == list:
#             if len(answers) == 0:
#                 return 0
#             f1_all.append(np.max([qa_f1_score(prediction, gt)
#                           for gt in answers]))
#         else:
#             f1_all.append(qa_f1_score(prediction, answers))
#     return 100 * np.mean(f1_all)

def f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truths):
    return max([f1(prediction, gt) for gt in ground_truths])


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def find_entity_tags(sentence):
    entity_regex = r'(.+?)(?=\s<|$)'
    tag_regex = r'<(.+?)>'
    entity_names = re.findall(entity_regex, sentence)
    tags = re.findall(tag_regex, sentence)

    results = {}
    for entity, tag in zip(entity_names, tags):
        if "<" in entity:
            results[entity.split("> ")[1]] = tag
        else:
            results[entity] = tag
    return results

def match(prediction, ground_truth):
    for gt in ground_truth:
        if normalize_answer(gt) in normalize_answer(prediction):
            return 1
    return 0


# import copy
# import json
# from metrics import *
# from utils import *
# from sentence_transformers import SentenceTransformer
# import torch.nn.functional as F

# # Load the sentence transformer model
# model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

# def cosine_match(output, golds, threshold=0.65):
    
#     # first filter the matched
#     if match(output, golds):
#         return 1
#     output = normalize_answer(output)
#     golds = [normalize_answer(g) for g in golds]
#     # Get embeddings for output and all gold answers
#     output_emb = model.encode(output, convert_to_tensor=True)
#     gold_embs = model.encode(golds, convert_to_tensor=True)
    
#     # Calculate cosine similarities
#     similarities = F.cosine_similarity(output_emb.unsqueeze(0), gold_embs)
    
#     # Check if any similarity exceeds threshold
#     max_similarity = similarities.max().item()
#     return 1 if max_similarity >= threshold else 0



def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(hypotheses,
                        references1,
                        references2=[],
                        metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def mauve_score(predictions, references):
    mauve_results = mauve.compute_mauve(
        p_text=references,
        q_text=predictions,
        device_id=4,
        max_text_length=512,
        verbose=True,
        batch_size=32,
        featurize_model_name="gpt2-large"
    )
    return mauve_results.mauve * 100
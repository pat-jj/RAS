import json
from pathlib import Path
from collections import defaultdict
from utils import load_json, save_json
from tqdm import tqdm


def get_doc_content(wiki_dir, title, batch_idx):
    """Read document content from wiki batch file"""
    batch_file = wiki_dir / f'wiki_batch_{batch_idx}.json'
    try:
        with open(batch_file, 'r') as f:
            batch_docs = json.load(f)
            # Find the title case-insensitively
            for doc_title, doc in batch_docs.items():
                if doc_title.lower() == title.lower():
                    return doc['text']
    except Exception as e:
        print(f"Error reading {title} from batch {batch_idx}: {e}")
    return None


def process_hotpot():
    hotpot_path = "/shared/eng/pj20/hotpotqa/hotpot_train_v1.1.json"
    wiki_dir = Path("/shared/eng/pj20/hotpotqa/data/processed_wiki")
    output_dir = Path("/shared/eng/pj20/hotpotqa/data/processed_hotpot")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load wiki index
    title_to_batch = load_json(wiki_dir / 'title_to_batch.json')
    title_to_batch = {title.lower(): batch_idx for title, batch_idx in title_to_batch.items()}
    
    # Load HotpotQA data
    with open(hotpot_path, 'r') as f:
        hotpot_data = json.load(f)
    
    # Process examples
    processed_examples = []
    for item in tqdm(hotpot_data):
        # Group supporting facts by document
        doc_facts = defaultdict(list)
        for title, sent_idx in item['supporting_facts']:
            doc_facts[title.lower()].append(sent_idx)
        
        # Check if all supporting docs are in corpus
        if all(title in title_to_batch for title in doc_facts.keys()):
            # Get content for each supporting document
            supporting_docs = {}
            for title in doc_facts:
                batch_idx = title_to_batch[title]
                doc_content = get_doc_content(wiki_dir, title, batch_idx)
                if doc_content:
                    # print(f"Found content for document: {title}")
                    # print(f"Supporting sentences: {doc_facts[title]}")
                    # print(f"Content length: {len(doc_content)}")
                    supporting_sentences = [doc_content[idx+1] for idx in doc_facts[title] if idx+1 < len(doc_content)]
                    supporting_docs[title] = supporting_sentences
                    # {
                    #     'sentences': supporting_sentences,
                    #     'sentence_indices': doc_facts[title],
                    #     'full_text': doc_content
                    # }
                else:
                    # print(f"Could not find content for document: {title}")
                    continue
            
            if len(supporting_docs) == len(doc_facts):
                processed_examples.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'type': item['type'],
                    'supporting_docs': supporting_docs
                })
    
    print(f"Processed {len(processed_examples)} examples out of {len(hotpot_data)}")
    save_json(processed_examples, output_dir / 'processed_hotpot.json')

if __name__ == "__main__":
    process_hotpot()
import bz2
import json
import glob
import os
import re
from tqdm import tqdm
from pathlib import Path
from utils import save_json, load_json

def strip_html_tags(text):
    """Strip HTML tags but keep the entity names between <a href=...> tags"""
    # This pattern matches <a href="...">entity</a>
    pattern = r'<a href="[^"]*">([^<]+)</a>'
    return re.sub(pattern, r'\1', text)

def read_wiki_file(filepath):
    docs = []
    with bz2.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                # Process text properly following their format
                # Each paragraph is a list of sentences
                text = []
                for paragraph in doc['text']:
                    # Join sentences in paragraph without separators and strip HTML tags
                    paragraph_text = ''.join(paragraph)
                    cleaned_text = strip_html_tags(paragraph_text)
                    text.append(cleaned_text)
                
                docs.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'text': text,  # Keep paragraph structure
                    'url': doc['url']
                })
            except:
                continue
    return docs

def get_processed_files(output_dir):
    """Get list of already processed batch files"""
    processed = set()
    for filepath in output_dir.glob('wiki_batch_*.json'):
        try:
            batch_num = int(filepath.stem.split('_')[-1])
            processed.add(batch_num)
        except:
            continue
    return processed

def process_wiki_corpus():
    base_path = "/shared/eng/pj20/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed"
    # Check all folders from AA to ZZ
    folders = []
    for i in range(26):
        for j in range(26):
            folder = chr(65 + i) + chr(65 + j)
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                folders.append(folder_path)
    
    print(f"Found {len(folders)} folders")
    
    output_dir = Path("/shared/eng/pj20/hotpotqa/data/processed_wiki")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing indices if they exist
    title_to_doc = {}
    id_to_doc = {}
    if (output_dir / 'title_to_batch.json').exists():
        title_to_doc = load_json(output_dir / 'title_to_batch.json')
        id_to_doc = load_json(output_dir / 'id_to_batch.json')

    # Get already processed batch numbers
    processed_batches = get_processed_files(output_dir)
    print(f"Found {len(processed_batches)} already processed batches")
    
    # Process all files
    all_files = []
    for folder in glob.glob(os.path.join(base_path, '*/')):
        all_files.extend(glob.glob(os.path.join(folder, '*.bz2')))
    all_files.sort()  # Ensure consistent ordering
    
    for i, filepath in enumerate(tqdm(all_files)):
        # Skip if this batch was already processed
        if i in processed_batches:
            continue
            
        docs = read_wiki_file(filepath)
        
        # Save batch of documents
        batch_docs = {doc['title']: doc for doc in docs}
        save_json(batch_docs, output_dir / f'wiki_batch_{i}.json')
        
        # Update indices
        title_to_doc.update({doc['title']: i for doc in docs})  # Store batch index instead of full doc
        id_to_doc.update({doc['id']: i for doc in docs})
        
        # Periodically save indices
        if i % 100 == 0:
            save_json(title_to_doc, output_dir / 'title_to_batch.json')
            save_json(id_to_doc, output_dir / 'id_to_batch.json')
    
    # Final save of indices
    save_json(title_to_doc, output_dir / 'title_to_batch.json')
    save_json(id_to_doc, output_dir / 'id_to_batch.json')
    
    # Save completion marker
    save_json({'completed': True, 'total_batches': len(all_files)}, 
              output_dir / 'processing_complete.json')

if __name__ == "__main__":
    process_wiki_corpus()
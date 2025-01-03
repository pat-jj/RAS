### This script identifies questions that don't need subqueries but can be answered using retrieval based on the main query

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import json
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import os

class RetrievalClassifier:
    def __init__(self, llm_name="meta-llama/Llama-2-7b-chat-hf", 
                 sent_bert_name='sentence-transformers/all-MiniLM-L6-v2',
                 similarity_threshold=0.8,
                 cache_dir="/shared/eng/pj20/hf_cache"):
        # Initialize LLaMA tokenizer with proper settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Initialize model with proper settings
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Initialize SentenceBERT
        self.sent_bert = SentenceTransformer(sent_bert_name, cache_folder=cache_dir)
        if torch.cuda.is_available():
            self.sent_bert.to('cuda:0')
        self.similarity_threshold = similarity_threshold

    def get_llm_answer_with_retrieval(self, question, retrieved_docs):
        """Get answer from LLM using retrieved documents as context."""
        # Sort retrieved docs by score and get the top ones
        sorted_docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)
        
        # Create the base prompt without context
        base_prompt = f"Based on the context above, please provide a direct, short answer (use one or two words if possible) to this question: {question}"
        
        # Calculate base prompt tokens
        base_tokens = len(self.tokenizer.encode(base_prompt))
        
        # Target max tokens (Llama-2 has 4096 token context window)
        max_tokens = 4000  # Leave some room for generation
        available_tokens = max_tokens - base_tokens
        
        # Accumulate context while tracking tokens
        context_pieces = []
        current_tokens = 0
        
        for doc in sorted_docs:
            doc_text = doc["text"]
            doc_tokens = len(self.tokenizer.encode(doc_text))
            
            if current_tokens + doc_tokens <= available_tokens:
                context_pieces.append(doc_text)
                current_tokens += doc_tokens
            else:
                break
                
        # Join the context pieces that fit within the window
        context = " ".join(context_pieces)
        
        # Create prompt with context
        messages = [
            {"role": "user", "content": f"""Here is some context to help answer a question:
            
            {context}
            
            {base_prompt}"""}
        ]
        
        # Use the chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        )
        
        # Move to same device as model
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                num_return_sequences=1
            )
            
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return self.clean_answer(response)

    def clean_answer(self, answer):
        """Clean the answer while preserving the actual content."""
        # Remove any system prompts or user messages if they got included
        answer = answer.split("[/INST]")[1].strip()
        
        # Remove extra whitespace and newlines
        answer = ' '.join(answer.split())
        
        return answer.strip()

    def compute_similarity(self, text1, text2):
        """Compute semantic similarity between two texts."""
        emb1 = self.sent_bert.encode(text1, convert_to_tensor=True).to('cuda:0')
        emb2 = self.sent_bert.encode(text2, convert_to_tensor=True).to('cuda:0')
        
        sim = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return sim.item()

def generate_retrieval_examples(hotpot_path, retrieval_path, output_path_1, output_path_2):
    """Generate examples of questions that can be answered correctly with retrieval."""
    # Initialize classifier
    classifier = RetrievalClassifier()
    
    # Load HotpotQA data
    with open(hotpot_path, 'r') as f:
        hotpot_data = json.load(f)
        
    # Load retrieval results
    with open(retrieval_path, 'r') as f:
        retrieval_data = json.load(f)
    
    # Create question to retrieval mapping
    retrieval_map = {item['question']: item['wiki_retrieved_docs'] for item in retrieval_data}
    
    llm_answers = []
    processed_questions_answers = {}
    if os.path.exists(output_path_1):
        with open(output_path_1, 'r') as f:
            llm_answers = json.load(f)
        processed_questions_answers = {
            item['question']: item['llm_answer'] for item in llm_answers
        }
        print(f"Loaded {len(processed_questions_answers)} LLM answers")
    
    # First, generate all LLM answers with retrieval
    print("Generating LLM answers with retrieval...")
    
    rest_of_hotpot_data = [item for item in hotpot_data if item['question'] not in processed_questions_answers.keys()]
    print(f"Generating LLM answers for {len(rest_of_hotpot_data)} questions")

    for item in tqdm(rest_of_hotpot_data):
        if item['question'] not in retrieval_map:
            print(f"Warning: No retrieval results found for question: {item['question']}")
            continue
            
        llm_answer = classifier.get_llm_answer_with_retrieval(
            item['question'], 
            retrieval_map[item['question']]
        )
                
        processed_item = {
            'question': item['question'],
            'true_answer': item['answer'],
            'llm_answer': llm_answer,
        }
        llm_answers.append(processed_item)
        
        if len(llm_answers) % 2000 == 0:
            print(f"Generated {len(llm_answers)} LLM answers")
            with open(output_path_1, 'w') as f:
                json.dump(llm_answers, f, indent=2)
                
    with open(output_path_1, 'w') as f:
        json.dump(llm_answers, f, indent=2)
    
    processed_questions_answers = {
        item['question']: item['llm_answer'] for item in llm_answers
    }
    
    # Then, compute all similarities
    print("Computing similarities...")
    processed_data = []
    for item in tqdm(hotpot_data):
        if item['question'] not in processed_questions_answers:
            continue
            
        llm_answer = processed_questions_answers[item['question']]
        similarity = classifier.compute_similarity(llm_answer, item['answer'])
        needs_retrieval = similarity >= classifier.similarity_threshold
        
        processed_item = {
            'question': item['question'],
            'true_answer': item['answer'],
            'llm_answer': llm_answer,
            'similarity': similarity,
            'can_answer_with_retrieval': needs_retrieval,
            'label': '[RETRIEVAL=YES] Can be answered with retrieval.' if needs_retrieval else '[RETRIEVAL=NO]',
        }
        processed_data.append(processed_item)
        
        # Save intermediate results
        if len(processed_data) % 10000 == 0:
            print(f"Processed {len(processed_data)} questions")
            with open(output_path_2, 'w') as f:
                json.dump(processed_data, f, indent=2)
    
    # Save final results
    with open(output_path_2, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Print statistics
    total = len(processed_data)
    retrieval_success = sum(1 for item in processed_data if item['can_answer_with_retrieval'])
    print(f"\nStatistics:")
    print(f"Total questions: {total}")
    print(f"Questions answerable with retrieval: {retrieval_success} ({retrieval_success/total*100:.2f}%)")
    print(f"Questions not answerable with retrieval: {total-retrieval_success} ({(total-retrieval_success)/total*100:.2f}%)")

if __name__ == "__main__":
    generate_retrieval_examples(
        '/shared/eng/pj20/firas_data/datasets/hotpotqa/hotpot_with_subqueries.json',
        '/shared/eng/pj20/firas_data/datasets/hotpotqa/wiki_retrieval/wiki_retrieval_results.json',
        '/shared/eng/pj20/firas_data/datasets/hotpotqa/llama_subquery_data/llama_retrieval_answers.json',
        '/shared/eng/pj20/firas_data/datasets/hotpotqa/llama_subquery_data/retrieval_classification.json'
    )
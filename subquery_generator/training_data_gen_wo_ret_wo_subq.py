### This script identifies questions that don't need both subqueries and re
### The model can directly answer the question

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import json
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import os

class NoSubqueryClassifier:
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

    def get_llm_answer(self, question):
        """Get direct answer from LLM without any context."""
        messages = [
            {"role": "user", "content": f"Please provide a direct, short answer (use one or two words if possible) to this question: {question}"}
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
        
        # Clean up the response
        answer = self.clean_answer(response)
        
        return answer

    def clean_answer(self, answer):
        """Clean the answer while preserving the actual content."""
        # Remove any system prompts or user messages if they got included
        answer = answer.split("[/INST]")[1].strip()
            
        # Remove extra whitespace and newlines
        answer = ' '.join(answer.split())
        
        return answer.strip()


    def compute_similarity(self, text1, text2):
        """Compute semantic similarity between two texts."""
        # Get embeddings and ensure they're on GPU
        emb1 = self.sent_bert.encode(text1, convert_to_tensor=True).to('cuda:0')
        emb2 = self.sent_bert.encode(text2, convert_to_tensor=True).to('cuda:0')
        
        # Compute cosine similarity
        sim = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return sim.item()

    # def needs_subquery(self, question, true_answer):
    #     """Determine if a question needs subqueries by comparing LLM answer with ground truth."""
    #     llm_answer = self.get_llm_answer(question)
    #     similarity = self.compute_similarity(llm_answer, true_answer)
        
    #     return similarity < self.similarity_threshold, llm_answer, similarity

def generate_no_subquery_examples(hotpot_path, output_path_1, output_path_2):
    """Generate examples of questions that don't need subqueries based on semantic similarity."""
    # Initialize classifier
    classifier = NoSubqueryClassifier()
    
    # Load HotpotQA data
    with open(hotpot_path, 'r') as f:
        hotpot_data = json.load(f)
        
    llm_answers = []
    if os.path.exists(output_path_1):
        with open(output_path_1, 'r') as f:
            llm_answers = json.load(f)
        processed_questions_answers = {
            item['question']: item['llm_answer'] for item in llm_answers
        }
        print(f"Loaded {len(processed_questions_answers)} LLM answers")
    
    # First, generate all LLM answers
    print("Generating LLM answers...")
    
    rest_of_hotpot_data = [item for item in hotpot_data if item['question'] not in processed_questions_answers.keys()]
    print(f"Generating LLM answers for {len(rest_of_hotpot_data)} questions")

    for item in tqdm(rest_of_hotpot_data):
        llm_answer = classifier.get_llm_answer(item['question'])
                
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
        llm_answer = processed_questions_answers[item['question']]
        similarity = classifier.compute_similarity(llm_answer, item['answer'])
        needs_subq = similarity < classifier.similarity_threshold
        
        processed_item = {
            'question': item['question'],
            'true_answer': item['answer'],
            'llm_answer': llm_answer,
            'similarity': similarity,
            'needs_subquery': needs_subq,
            'label': '[SUBQ=NO] No subquery is needed.' if not needs_subq else '[SUBQ=YES]',
        }
        processed_data.append(processed_item)
        
        # Save intermediate results every 1000 items
        if len(processed_data) % 10000 == 0:
            print(f"Processed {len(processed_data)} questions")
            with open(output_path_2, 'w') as f:
                json.dump(processed_data, f, indent=2)
    
    # Save final results
    with open(output_path_2, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Print statistics
    total = len(processed_data)
    no_subq = sum(1 for item in processed_data if not item['needs_subquery'])
    print(f"\nStatistics:")
    print(f"Total questions: {total}")
    print(f"Questions not needing subquery: {no_subq} ({no_subq/total*100:.2f}%)")
    print(f"Questions needing subquery: {total-no_subq} ({(total-no_subq)/total*100:.2f}%)")

if __name__ == "__main__":
    generate_no_subquery_examples(
        '/shared/eng/pj20/firas_data/datasets/hotpotqa/hotpot_with_subqueries.json',
        '/shared/eng/pj20/firas_data/datasets/hotpotqa/llama_subquery_data/llama_answers.json',
        '/shared/eng/pj20/firas_data/datasets/hotpotqa/llama_subquery_data/subquery_classification.json'
    )

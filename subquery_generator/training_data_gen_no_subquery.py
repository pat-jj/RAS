from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import json
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

class NoSubqueryClassifier:
    def __init__(self, llm_name="NousResearch/Llama-2-7b-hf", 
                 sent_bert_name='sentence-transformers/all-MiniLM-L6-v2',
                 similarity_threshold=0.8,
                 cache_dir='/shared/eng/pj20/hf_cache'):
        # Initialize LLaMA
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name, 
            cache_dir=cache_dir,
            device_map="auto",  # Automatically distribute across available GPUs
            torch_dtype=torch.float16  # Use half precision to save memory
        )
        self.model.eval()
        
        # Initialize SentenceBERT and move to first GPU
        self.sent_bert = SentenceTransformer(sent_bert_name, cache_folder=cache_dir)
        self.sent_bert.to('cuda:0')
        self.similarity_threshold = similarity_threshold

    def get_llm_answer(self, question, max_length=128):
        """Get direct answer from LLM without any context."""
        # Llama 2 chat format
        prompt = f"""[INST] Question: {question}
Please provide a direct, concise answer. [/INST]"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        
        # Move inputs to GPU
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=32,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer part after the instruction
        answer = response.split("[/INST]")[-1].strip()
        return answer

    def compute_similarity(self, text1, text2):
        """Compute semantic similarity between two texts."""
        # Get embeddings and ensure they're on GPU
        emb1 = self.sent_bert.encode(text1, convert_to_tensor=True).to('cuda:0')
        emb2 = self.sent_bert.encode(text2, convert_to_tensor=True).to('cuda:0')
        
        # Compute cosine similarity
        sim = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return sim.item()

    def needs_subquery(self, question, true_answer):
        """Determine if a question needs subqueries by comparing LLM answer with ground truth."""
        llm_answer = self.get_llm_answer(question)
        similarity = self.compute_similarity(llm_answer, true_answer)
        
        return similarity < self.similarity_threshold, llm_answer, similarity

def generate_no_subquery_examples(hotpot_path, output_path_1, output_path_2):
    """Generate examples of questions that don't need subqueries based on semantic similarity."""
    # Initialize classifier
    classifier = NoSubqueryClassifier()
    
    # Load HotpotQA data
    with open(hotpot_path, 'r') as f:
        hotpot_data = json.load(f)
    
    # First, generate all LLM answers
    print("Generating LLM answers...")
    llm_answers = []
    answered_data = []
    for item in tqdm(hotpot_data):
        llm_answer = classifier.get_llm_answer(item['question'])
        llm_answers.append(llm_answer)
        
        processed_item = {
            'question': item['question'],
            'true_answer': item['answer'],
            'llm_answer': llm_answer,
        }
        answered_data.append(processed_item)
        
        if len(answered_data) % 1000 == 0:
            print(f"Generated {len(answered_data)} LLM answers")
            with open(output_path_1, 'w') as f:
                json.dump(answered_data, f, indent=2)
    
    # Then, compute all similarities
    print("Computing similarities...")
    processed_data = []
    for item, llm_answer in tqdm(zip(hotpot_data, llm_answers)):
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
        if len(processed_data) % 1000 == 0:
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

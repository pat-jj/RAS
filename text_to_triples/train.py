import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
from huggingface_hub import HfFolder
import json
import wandb
from tqdm import tqdm
import evaluate
import numpy as np
import os
import shutil

class TripletDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append({
                    'text': item['text'],
                    'triplet': item['triplet']
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_encodings = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer(
            item['triplet'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'labels': target_encodings['input_ids'].squeeze(),
            'text': item['text'],
            'triplet': item['triplet']
        }

def evaluate_model(model, val_loader, tokenizer, rouge):
    model.eval()
    total_val_loss = 0
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_val_loss += loss.item()
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=1,
                early_stopping=True
            )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_refs = batch['triplet']
            
            predictions.extend(decoded_preds)
            references.extend(decoded_refs)
    
    avg_val_loss = total_val_loss / len(val_loader)
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    return avg_val_loss, rouge_scores

def train():
    # Initialize wandb and model as before
    wandb.init(project="text2triplet", name="flan-t5-large")
    
    # Hugging Face repository name
    hf_model_name = "pat-jj/text2triple-flan-t5"  # Change this to your desired repository name
    
    # Check if logged into HuggingFace
    if not HfFolder.get_token():
        print("Please log in to Hugging Face first using `huggingface-cli login`")
        return
    
    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    
    train_path = "/shared/eng/pj20/firas_data/datasets/WikiOFGraph/WikiOFGraph_train.jsonl"
    val_path = "/shared/eng/pj20/firas_data/datasets/WikiOFGraph/WikiOFGraph-test-small.jsonl"
    
    save_path = "/shared/eng/pj20/firas_data/text2triple_model"
    os.makedirs(save_path, exist_ok=True)
    
    train_dataset = TripletDataset(train_path, tokenizer)
    val_dataset = TripletDataset(val_path, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    best_val_loss = float('inf')
    rouge = evaluate.load('rouge')
    global_step = 0
    validation_step = 7000
    
    # Create model card content
    model_card = f"""
# Text2Triple Flan-T5 Model

This model converts natural language text into structured triplets. It was fine-tuned on the WikiOFGraph dataset using Flan-T5-Large as the base model.

## Model Details
- Base Model: google/flan-t5-large
- Task: Text to Triple Generation
- Training Data: WikiOFGraph dataset
- Input: Natural language text
- Output: Structured triplets in the format (<S> subject| <P> predicate| <O> object)

## Training Metrics
Best Validation Loss: {best_val_loss}
    """
    
    model.train()
    total_train_loss = 0
    
    progress_bar = tqdm(total=num_training_steps, desc="Training")
    
    for epoch in range(1):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            
            wandb.log({"train_loss": loss.item(), "global_step": global_step})
            
            # Validate every validation_step steps
            # Validate every validation_step steps
            if global_step % validation_step == 0:
                avg_train_loss = total_train_loss / validation_step
                total_train_loss = 0
                
                print(f"\nStep {global_step}: Running validation...")
                avg_val_loss, rouge_scores = evaluate_model(model, val_loader, tokenizer, rouge)
                
                # Log metrics
                wandb.log({
                    "global_step": global_step,
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                    "rouge1": rouge_scores['rouge1'],
                    "rouge2": rouge_scores['rouge2'],
                    "rougeL": rouge_scores['rougeL']
                })
                
                # Save if better performance
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_save_path = os.path.join(save_path, "best_model")
                    
                    # Remove old model if it exists
                    if os.path.exists(model_save_path):
                        shutil.rmtree(model_save_path)
                    
                    # Save new best model locally
                    model.save_pretrained(model_save_path)
                    tokenizer.save_pretrained(model_save_path)
                    
                    # Update model card with current metrics
                    updated_model_card = f"""
# Text2Triple Flan-T5 Model

This model converts natural language text into structured triplets. It was fine-tuned on the WikiOFGraph dataset using Flan-T5-Large as the base model.

## Model Details
- Base Model: google/flan-t5-large
- Task: Text to Triple Generation
- Training Data: WikiOFGraph dataset
- Input: Natural language text
- Output: Structured triplets in the format (<S> subject| <P> predicate| <O> object)

## Training Metrics
- Best Validation Loss: {best_val_loss:.4f}
- ROUGE-1: {rouge_scores['rouge1']:.4f}
- ROUGE-2: {rouge_scores['rouge2']:.4f}
- ROUGE-L: {rouge_scores['rougeL']:.4f}
- Training Step: {global_step}
                    """
                    
                    # Save model card
                    with open(os.path.join(model_save_path, "README.md"), "w") as f:
                        f.write(updated_model_card)
                    
                    # Push to Hugging Face Hub
                    try:
                        model.push_to_hub(hf_model_name, 
                                        commit_message=f"Step {global_step} - Val Loss: {avg_val_loss:.4f}")
                        tokenizer.push_to_hub(hf_model_name, 
                                            commit_message=f"Step {global_step} - Val Loss: {avg_val_loss:.4f}")
                        
                        print(f"\nStep {global_step}:")
                        print(f"New best model saved and pushed to HuggingFace Hub!")
                        print(f"Val Loss: {avg_val_loss:.4f}")
                        print(f"Average training loss: {avg_train_loss:.4f}")
                        print(f"ROUGE scores: {rouge_scores}")
                        print(f"Model available at: https://huggingface.co/{hf_model_name}")
                    except Exception as e:
                        print(f"Failed to push to HuggingFace Hub: {str(e)}")
                        print("Continuing training...")
                
                # Set back to training mode
                model.train()
            
            if global_step >= num_training_steps:
                break
        
        if global_step >= num_training_steps:
            break
    
    progress_bar.close()

if __name__ == "__main__":
    train()
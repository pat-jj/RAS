import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
import evaluate
import wandb
import numpy as np
import os
if "HOME" not in os.environ:
    os.environ["HOME"] = os.path.expanduser("~")

from accelerate import PartialState
device_string = PartialState().process_index

# Initialize wandb
wandb.init(project="text2triplet", name="llama-3.2-3b-finetune", mode="online")

# Constants
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "/shared/eng/pj20/firas_data/text2triple_llama_model"
HF_TOKEN = "HF_TOKEN"  # Replace with actual token

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={'':device_string},
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set up chat template
tokenizer.chat_template = tokenizer.chat_template or "llama-3.1"

# Load datasets
train_data = load_dataset('json', data_files='/shared/eng/pj20/firas_data/datasets/WikiOFGraph/hotpotqa_train.jsonl')
valid_data = load_dataset('json', data_files='/shared/eng/pj20/firas_data/datasets/WikiOFGraph/hotpotqa_valid.jsonl')

def formatting_prompts_func(examples):
    messages = [{
        "role": "user",
        "content": f"Convert the following text to triples:\n\nText: {examples['input']}"
    }, {
        "role": "assistant",
        "content": examples['output']
    }]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Process datasets
train_data = train_data['train'].map(formatting_prompts_func)
valid_data = valid_data['train'].map(formatting_prompts_func)

# Print example to verify formatting
print("Sample processed example:")
print(train_data[0]['text'])

# Initialize metrics
rouge = evaluate.load('rouge')

def compute_metrics(pred):
    predictions = pred.predictions
    labels = pred.label_ids
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Extract just the triples part
    decoded_preds = [pred.split("Triples:")[-1].strip() if "Triples:" in pred else pred for pred in decoded_preds]
    decoded_labels = [label.split("Triples:")[-1].strip() if "Triples:" in label else label for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract scores
    result = {key: value * 100 for key, value in result.items()}
    
    # Log metrics to wandb
    wandb.log({
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL']
    })
    
    return {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL']
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=3,
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_steps=1000,
    logging_steps=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    torch_compile=False,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    metric_for_best_model="rougeL",
    # load_best_model_at_end=True,
    greater_is_better=True,
    report_to="wandb",
    push_to_hub=True,
    hub_model_id="pat-jj/text2graph-llama-3.2-3b",
    hub_token=HF_TOKEN,
    # eval_on_start=True,
    ddp_timeout=36000,  # Increase timeout to 10 hours
)

# Initialize trainer directly (no PEFT/LoRA configuration needed)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    dataset_text_field="text",
    compute_metrics=compute_metrics,
    max_seq_length=MAX_SEQ_LENGTH,
)

# Verify a batch
batch = next(iter(trainer.get_train_dataloader()))
print("\nSample batch shape:", batch['input_ids'].shape)

# Train the model
try:
    trainer.train()
except Exception as e:
    print(f"Error training the model: {e}")
    trainer.save_model()
    wandb.finish()
    exit()
    
# Save the final model locally
trainer.save_model()

# End wandb run
wandb.finish()
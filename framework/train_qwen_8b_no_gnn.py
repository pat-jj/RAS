import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import wandb
from datetime import datetime
import pickle


def setup_logging():
    """Setup logging"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'training_qwen_no_gnn_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_wandb(args):
    """Initialize wandb"""
    os.environ['WANDB_START_METHOD'] = 'thread'
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    os.environ['WANDB_RUN_ID'] = run_id

    wandb.init(
        project="qwen-no-gnn-sft",
        config=vars(args),
        settings=wandb.Settings(start_method="thread")
    )
    return True


def load_and_format_data(data_path):
    """Load data and format as messages for SFT - exactly like your example"""
    data = pickle.load(open(data_path, 'rb'))
    
    # Format as messages (chat format) - exactly like your example
    messages_list = []
    for item in data:
        input_text = item['input']
        label_text = item['label']
        
        # Format as Qwen3 chat messages - exactly like your example
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": label_text}
        ]
        messages_list.append(messages)
    
    return messages_list


def main():
    parser = argparse.ArgumentParser(description='Train Qwen3-8B without GNN using TRL SFTTrainer')
    
    # Model arguments
    parser.add_argument('--llm_model_path', type=str, default='Qwen/Qwen3-8B', help='Path to Qwen3-8B model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/shared/rsaas/pj20/firas_data/multitask')
    parser.add_argument('--output_dir', type=str, default='/shared/rsaas/pj20/firas_data/multitask/checkpoints_qwen_no_gnn')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=2800, help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=5000, help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluate every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3, help='Keep only last N checkpoints')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples (None = use all)')
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    logger.info("Starting Qwen3-8B training without GNN")
    logger.info(f"Arguments: {args}")
    
    wandb_enabled = setup_wandb(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and format datasets - exactly like your example
    logger.info("Loading datasets...")
    train_messages = load_and_format_data(os.path.join(args.data_dir, 'combined_train_v3.pkl'))
    val_messages = load_and_format_data(os.path.join(args.data_dir, 'combined_val_v2.pkl'))
    
    # Limit training data if specified (for LoRA, we can use full set)
    if args.max_train_samples is not None and len(train_messages) > args.max_train_samples:
        logger.info(f"Limiting training data from {len(train_messages)} to {args.max_train_samples} samples")
        train_messages = train_messages[:args.max_train_samples]
    
    logger.info(f"Train dataset size: {len(train_messages)}")
    logger.info(f"Val dataset size: {len(val_messages)}")
    
    # Create datasets - exactly like your example
    train_dataset = Dataset.from_dict({
        "messages": train_messages
    })
    val_dataset = Dataset.from_dict({
        "messages": val_messages
    })
    
    # Load model and tokenizer - exactly like your example
    logger.info(f"Loading model from {args.llm_model_path}...")
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, trust_remote_code=True, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA config - use SFTTrainer's built-in support
    logger.info("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Training config - exactly like your example
    training_args = SFTConfig(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        save_strategy='steps',  # Save by steps, not epoch
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
    )
    
    # Initialize trainer - exactly like your example, but with LoRA
    logger.info("Initializing SFTTrainer with LoRA...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # Like your example - no eval dataset
        processing_class=tokenizer,
        peft_config=peft_config,  # Add LoRA config
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")
    if wandb_enabled:
        wandb.finish()


if __name__ == '__main__':
    main()

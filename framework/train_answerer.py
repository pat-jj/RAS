import os
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import argparse
from models.graph_llm import GraphLLM
from torch.optim import AdamW
import wandb
from tqdm import tqdm
import datetime
import math
from torch.optim.lr_scheduler import LambdaLR

class HotpotQADataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def setup_logging(args):
    # Create logs directory
    os.makedirs('llm_tune/logs', exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('llm_tune/logs', f'training_{timestamp}.log')
    
    # Setup logging configuration
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
    run_id = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
    os.environ['WANDB_RUN_ID'] = run_id

    wandb.init(
        project="graphllm-hotpotqa",
        config=args,
        settings=wandb.Settings(start_method="thread")
    )
    return True

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer and 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer."""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, optimizer, scheduler, epoch, args):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    running_loss = []  # For loss smoothing
    
    for batch_idx, batch in enumerate(progress_bar):
        # Forward pass
        loss = model(batch)
        loss = loss / args.grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Track gradient norms
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Gradient accumulation and optimization step
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        # Loss tracking and logging
        total_loss += loss.item() * args.grad_accum_steps
        running_loss.append(loss.item() * args.grad_accum_steps)
        if len(running_loss) > 100:  # Keep last 100 losses for smoothing
            running_loss.pop(0)
        smoothed_loss = sum(running_loss) / len(running_loss)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': smoothed_loss,
            'lr': scheduler.get_last_lr()[0],
            'grad_norm': grad_norm
        })
        
        # Log to wandb
        if batch_idx % args.log_interval == 0:
            wandb.log({
                'train_loss': loss.item() * args.grad_accum_steps,
                'smoothed_loss': smoothed_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'gradient_norm': grad_norm,
                'epoch': epoch,
                'step': batch_idx + epoch * len(train_loader)
            })
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            loss = model(batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--llm_model_path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--llm_frozen', type=str, default='False')
    parser.add_argument('--finetune_method', type=str, default='lora', choices=['full', 'lora'],
                      help='Finetuning method: full for full-parameter, lora for LoRA')
    parser.add_argument('--gnn_model_name', type=str, default='gt')
    parser.add_argument('--gnn_in_dim', type=int, default=1024)
    parser.add_argument('--gnn_hidden_dim', type=int, default=1024)
    parser.add_argument('--gnn_num_layers', type=int, default=3)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--gnn_num_heads', type=int, default=8)
    
    # Training arguments
    parser.add_argument('--data_dir', type=str, default='/shared/eng/pj20/firas_data/inference_model/hotpotqa_train')
    parser.add_argument('--output_dir', type=str, default='/shared/eng/pj20/firas_data/inference_model/hotpotqa_train/checkpoints')
    parser.add_argument('--max_txt_len', type=int, default=1536)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)  # Lower base learning rate
    parser.add_argument('--warmup_ratio', type=float, default=0.15)  # Longer warmup
    parser.add_argument('--grad_accum_steps', type=int, default=4)  # More accumulation steps
    parser.add_argument('--grad_clip', type=float, default=0.5)  # Lower clip threshold
    parser.add_argument('--weight_decay', type=float, default=0.01)  # Add weight decay
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)
    
    # Add LoRA specific arguments
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Adjust batch size based on finetuning method
    if args.finetune_method == 'full':
        # Reduce batch size and accumulate gradients for full finetuning
        args.batch_size = max(1, args.batch_size // 4)  # Reduce batch size
        
    # Setup logging
    logger = setup_logging(args)
    
    # Initialize wandb
    wandb_enabled = setup_wandb(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Loading datasets...")
    
    # Load datasets
    train_dataset = HotpotQADataset(os.path.join(args.data_dir, 'train.pt'))
    val_dataset = HotpotQADataset(os.path.join(args.data_dir, 'val.pt'))
    
    logger.info("Initializing model...")
    model = GraphLLM(args)
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'id': [item['id'] for item in x],
            'question': [item['question'] for item in x],
            'graphs': [item['graphs'] for item in x],
            'desc': [item['desc'] for item in x],
            'label': [item['label'] for item in x]
        }
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'id': [item['id'] for item in x],
            'question': [item['question'] for item in x],
            'graphs': [item['graphs'] for item in x],
            'desc': [item['desc'] for item in x],
            'label': [item['label'] for item in x]
        }
    )
    
    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), 
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     betas=(0.9, 0.95))  # Modified betas
    
    # Use cosine schedule instead of linear
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, args)
        val_loss = evaluate(model, val_loader)
        
        logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f}')
        logger.info(f'Epoch {epoch} - Validation Loss: {val_loss:.4f}')
        
        if wandb_enabled:
            wandb.log({
                'epoch': epoch,
                'train_loss_epoch': train_loss,
                'val_loss_epoch': val_loss
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, f'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, model_path)
            logger.info(f'Saved best model to {model_path}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, model_path)
    
    if wandb_enabled:
        wandb.finish()

if __name__ == '__main__':
    main()
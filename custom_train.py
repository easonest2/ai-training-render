#!/usr/bin/env python3
"""
Custom Training Script for MicroGPT Pro
This script adds new training content to your existing model without losing progress.
It automatically resumes from the latest checkpoint and continues training.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from datetime import datetime
import time

from model import create_model, GPTConfig
from tokenizer_byte import tokenizer, encode, decode
from config import MODEL_CONFIG, GENERATION_CONFIG

class TextDataset(Dataset):
    """Dataset for training text data"""
    
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk of text
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def prepare_training_data(data_path, block_size=1024):
    """Prepare training data from text files"""
    print(f"Loading training data from {data_path}...")
    
    all_text = ""
    
    if os.path.isfile(data_path):
        # Single file
        with open(data_path, 'r', encoding='utf-8') as f:
            all_text = f.read()
    elif os.path.isdir(data_path):
        # Directory of files
        for filename in os.listdir(data_path):
            if filename.endswith(('.txt', '.md', '.py', '.js', '.html', '.css')):
                filepath = os.path.join(data_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        all_text += f.read() + "\n"
                except Exception as e:
                    print(f"Warning: Could not read {filename}: {e}")
    
    if not all_text.strip():
        raise ValueError("No training data found!")
    
    print(f"Loaded {len(all_text)} characters of text")
    
    # Tokenize the text
    tokens = encode(all_text, add_eos=False)
    print(f"Tokenized into {len(tokens)} tokens")
    
    return tokens

def create_training_config():
    """Create training configuration"""
    return {
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'warmup_steps': 100,
        'max_epochs': 5,  # Shorter for incremental training
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'save_every': 25,      # Save every 25 iterations
        'eval_every': 15,      # Evaluate every 15 iterations
        'eval_iters': 50,
        'log_every': 5,        # Log every 5 iterations
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'block_size': MODEL_CONFIG['block_size'],
        'vocab_size': MODEL_CONFIG['vocab_size'],
        'n_layer': MODEL_CONFIG['n_layer'],
        'n_head': MODEL_CONFIG['n_head'],
        'n_embd': MODEL_CONFIG['n_embd'],
        'dropout': MODEL_CONFIG['dropout']
    }

def load_checkpoint(checkpoint_path, config):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    
    # Create model with same config
    model = create_model(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    
    # Setup optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=(0.9, 0.95),
        device_type=config['device']
    )
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get training state
    start_iter = checkpoint.get('iter_num', 0)
    loss_history = checkpoint.get('loss_history', [])
    
    print(f"✓ Checkpoint loaded successfully!")
    print(f"  - Starting from iteration: {start_iter}")
    print(f"  - Previous loss history: {len(loss_history)} entries")
    
    return model, optimizer, start_iter, loss_history

def save_checkpoint(model, optimizer, iter_num, loss_history, config, filename):
    """Save training checkpoint"""
    checkpoint = {
        'iter_num': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join('artifacts', filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

def train_model_incremental(model, train_loader, config, start_iter=0, loss_history=None):
    """Incremental training loop that continues from checkpoint"""
    print(f"🔄 Starting incremental training on {config['device']}...")
    print(f"📈 Continuing from iteration {start_iter}")
    
    # Move model to device
    model.to(config['device'])
    model.train()
    
    # Initialize loss history if not provided
    if loss_history is None:
        loss_history = []
    
    # Setup optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=(0.9, 0.95),
        device_type=config['device']
    )
    
    # Learning rate scheduler
    def get_lr(it):
        if it < config['warmup_steps']:
            return config['learning_rate'] * it / config['warmup_steps']
        else:
            decay_factor = 0.1 ** (it / (config['max_epochs'] * len(train_loader)))
            return config['learning_rate'] * decay_factor
    
    # Training loop
    iter_num = start_iter
    running_loss = 0.0
    last_save_time = time.time()
    
    print(f"🎯 Training for {config['max_epochs']} epochs...")
    
    for epoch in range(config['max_epochs']):
        print(f"\n📚 Epoch {epoch + 1}/{config['max_epochs']}")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(config['device']), y.to(config['device'])
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                # Update learning rate
                lr = get_lr(iter_num)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Update iteration counter
                iter_num += 1
                
                # Logging
                running_loss += loss.item() * config['gradient_accumulation_steps']
                
                if iter_num % config['log_every'] == 0:
                    avg_loss = running_loss / config['log_every']
                    print(f"  Iter {iter_num:4d} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
                    loss_history.append(avg_loss)
                    running_loss = 0.0
                
                # Evaluation
                if iter_num % config['eval_every'] == 0:
                    model.eval()
                    eval_loss = 0.0
                    with torch.no_grad():
                        for _ in range(config['eval_iters']):
                            x_eval, y_eval = next(iter(train_loader))
                            x_eval, y_eval = x_eval.to(config['device']), y_eval.to(config['device'])
                            _, loss_eval = model(x_eval, y_eval)
                            eval_loss += loss_eval.item()
                    eval_loss /= config['eval_iters']
                    print(f"  📊 Evaluation | Loss: {eval_loss:.4f}")
                    model.train()
                
                # Save checkpoint
                if iter_num % config['save_every'] == 0:
                    save_checkpoint(model, optimizer, iter_num, loss_history, config, 'checkpoint_latest.pt')
                
                # Time-based checkpoint (every 10 minutes)
                current_time = time.time()
                if current_time - last_save_time > 600:  # 10 minutes
                    save_checkpoint(model, optimizer, iter_num, loss_history, config, f'checkpoint_time_{int(current_time)}.pt')
                    last_save_time = current_time
    
    # Save session completion checkpoint (always save at end of training)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_checkpoint(model, optimizer, iter_num, loss_history, config, f'checkpoint_session_{session_timestamp}.pt')
    print(f"💾 Session checkpoint saved: checkpoint_session_{session_timestamp}.pt")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, iter_num, loss_history, config, 'checkpoint_final.pt')
    
    # Save final model
    final_path = os.path.join('artifacts', 'model_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"🎉 Final model saved: {final_path}")
    
    return model

def create_custom_prompts_file():
    """Create a file with custom training prompts"""
    print("\n📝 Creating custom training prompts file...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"custom_prompts_{timestamp}.txt"
    
    print(f"Enter your training prompts (one per line)")
    print(f"Type 'DONE' on a new line when finished")
    print(f"File will be saved as: {filename}")
    print("-" * 50)
    
    prompts = []
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.upper() == 'DONE':
            break
        if prompt:
            prompts.append(prompt)
    
    if not prompts:
        print("No prompts entered. Exiting.")
        return None
    
    # Save prompts to file
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(prompt + "\n")
    
    print(f"✅ Saved {len(prompts)} prompts to {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description='Custom Training for MicroGPT Pro - Add New Content')
    parser.add_argument('--custom_prompts', action='store_true', 
                       help='Create and use custom training prompts')
    parser.add_argument('--data_file', type=str, 
                       help='Specific text file to add to training')
    parser.add_argument('--data_dir', type=str, default='training_data/',
                       help='Training data directory (default: training_data/)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_latest.pt',
                       help='Checkpoint to resume from (default: checkpoint_latest.pt)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    
    args = parser.parse_args()
    
    print("🚀 MicroGPT Pro - Custom Training (Add New Content)")
    print("=" * 60)
    
    # Create training configuration
    config = create_training_config()
    config['max_epochs'] = args.epochs
    
    print(f"📋 Training Configuration:")
    print(f"  - Model: {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} embeddings")
    print(f"  - Training: {config['max_epochs']} epochs, batch size {config['batch_size']}")
    print(f"  - Device: {config['device']}")
    print(f"  - Checkpoint: {args.checkpoint}")
    
    # Determine data source
    if args.custom_prompts:
        data_source = create_custom_prompts_file()
        if not data_source:
            return
    elif args.data_file:
        data_source = args.data_file
        if not os.path.exists(data_source):
            print(f"❌ Error: Data file not found: {data_source}")
            return
    else:
        data_source = args.data_dir
        if not os.path.exists(data_source):
            print(f"❌ Error: Training data directory not found: {data_source}")
            return
    
    print(f"\n📁 Using training data from: {data_source}")
    
    # Prepare training data
    try:
        training_tokens = prepare_training_data(data_source, config['block_size'])
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
        return
    
    # Create dataset and dataloader
    dataset = TextDataset(training_tokens, config['block_size'])
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    print(f"📊 Created dataset with {len(dataset)} samples")
    
    # Load or create model
    checkpoint_path = os.path.join('artifacts', args.checkpoint)
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        model, optimizer, start_iter, loss_history = load_checkpoint(checkpoint_path, config)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Creating new model (this will start from scratch)")
        model = create_model(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd']
        )
        start_iter = 0
        loss_history = []
    
    print(f"🤖 Model has {model.get_num_params():,} parameters")
    
    # Train the model incrementally
    try:
        print(f"\n🎯 Starting incremental training...")
        print(f"📈 This will ADD to your existing training progress!")
        
        trained_model = train_model_incremental(model, train_loader, config, start_iter, loss_history)
        
        print(f"\n🎉 Incremental training completed successfully!")
        print(f"✅ Your model has been updated with new content!")
        print(f"💾 Final model saved in 'artifacts/model_final.pt'")
        print(f"🔄 You can continue training anytime by running this script again!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

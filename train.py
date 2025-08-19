#!/usr/bin/env python3
"""
Training script for MicroGPT Pro.
This script trains the MicroGPT model on your custom dataset.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np

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
        'max_epochs': 10,
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'save_every': 50,      # Save every ~50 iterations (about 10 minutes)
        'eval_every': 25,      # Evaluate every ~25 iterations (about 5 minutes)
        'eval_iters': 100,
        'log_every': 10,       # Log progress every 10 iterations
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'block_size': MODEL_CONFIG['block_size'],
        'vocab_size': MODEL_CONFIG['vocab_size'],
        'n_layer': MODEL_CONFIG['n_layer'],
        'n_head': MODEL_CONFIG['n_head'],
        'n_embd': MODEL_CONFIG['n_embd'],
        'dropout': MODEL_CONFIG['dropout']
    }

def train_model(model, train_loader, config):
    """Main training loop"""
    print(f"Starting training on {config['device']}...")
    
    # Move model to device
    model.to(config['device'])
    model.train()
    
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
        return config['learning_rate'] * 0.1 ** (it / (config['max_epochs'] * len(train_loader)))
    
    # Training loop
    iter_num = 0
    best_loss = float('inf')
    lr = config['learning_rate']  # Initialize lr variable
    last_save_time = datetime.now()  # Track time for regular saves
    
    for epoch in range(config['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['max_epochs']}")
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(config['device']), y.to(config['device'])
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update learning rate
                lr = get_lr(iter_num)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                optimizer.step()
                optimizer.zero_grad()
                iter_num += 1
                
                # Logging - only show lr if it's been updated
                if iter_num % config['log_every'] == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{lr:.2e}"
                    })
                
                # Evaluation
                if iter_num % config['eval_every'] == 0:
                    eval_loss = evaluate_model(model, train_loader, config)
                    print(f"\nIteration {iter_num}: Train loss = {loss.item():.4f}, Eval loss = {eval_loss:.4f}")
                    
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        save_checkpoint(model, optimizer, iter_num, eval_loss, config)
                        print(f"🏆 New best model saved! Loss: {eval_loss:.4f}")
                
                # Save checkpoint every 10 minutes OR every save_every iterations
                current_time = datetime.now()
                time_since_last_save = (current_time - last_save_time).total_seconds() / 60  # minutes
                
                if iter_num % config['save_every'] == 0 or time_since_last_save >= 10:
                    save_checkpoint(model, optimizer, iter_num, loss.item(), config)
                    last_save_time = current_time
                    print(f"💾 Checkpoint saved at iteration {iter_num} (every 10 minutes)")
    
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    return model

def evaluate_model(model, data_loader, config):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    eval_iters = min(config['eval_iters'], len(data_loader))
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= eval_iters:
                break
            x, y = x.to(config['device']), y.to(config['device'])
            logits, loss = model(x, y)
            total_loss += loss.item()
    
    model.train()
    return total_loss / eval_iters

def save_checkpoint(model, optimizer, iteration, loss, config):
    """Save model checkpoint"""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = f"checkpoints/model_iter_{iteration}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save as final model
    final_path = "artifacts/model_final.pt"
    torch.save(checkpoint, final_path)
    print(f"Final model saved: {final_path}")

def generate_sample_text(model, config, prompt="Hello, how are you?"):
    """Generate sample text to test the trained model"""
    print(f"\nGenerating sample text with prompt: '{prompt}'")
    
    model.eval()
    with torch.no_grad():
        # Encode prompt
        input_tokens = encode(prompt, add_eos=False)
        input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=config['device'])
        
        # Generate
        generated = model.generate(
            input_tensor,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )
        
        # Decode
        response_tokens = generated[0][len(input_tokens):].tolist()
        response = decode(response_tokens, remove_eos=True)
        
        print(f"Generated: {response}")
    
    model.train()

def main():
    parser = argparse.ArgumentParser(description='Train MicroGPT Pro')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to training data file or directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--block-size', type=int, default=1024,
                       help='Sequence length for training')
    
    args = parser.parse_args()
    
    # Create training configuration
    config = create_training_config()
    config.update({
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'block_size': args.block_size
    })
    
    print("MicroGPT Pro Training Configuration:")
    print(f"Model: {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} embeddings")
    print(f"Training: {config['max_epochs']} epochs, batch size {config['batch_size']}, lr {config['learning_rate']}")
    print(f"Device: {config['device']}")
    
    # Prepare training data
    try:
        training_tokens = prepare_training_data(args.data, config['block_size'])
    except Exception as e:
        print(f"Error preparing training data: {e}")
        return
    
    # Create dataset and dataloader
    dataset = TextDataset(training_tokens, config['block_size'])
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Create model
    model = create_model(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd']
    )
    
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Train the model
    try:
        trained_model = train_model(model, train_loader, config)
        
        # Generate sample text
        generate_sample_text(trained_model, config)
        
        print("\n🎉 Training completed successfully!")
        print("Your trained model is saved in 'artifacts/model_final.pt'")
        print("You can now use it with your MicroGPT Pro server!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

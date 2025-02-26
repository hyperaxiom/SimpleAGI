# train_agi.py
# Main training script for our simple AGI project

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import argparse
import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import our modules
from simple_agi import SimpleAGI, AGIConfig, Benchmarks
from data_utils import SimpleTokenizer, load_text_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Simple AGI model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints and logs")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=50000, help="Size of vocabulary")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--eval_interval", type=int, default=500, help="Interval between evaluations")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval between saving checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    
    return parser.parse_args()

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def get_lr_scheduler(optimizer, warmup_steps, max_steps):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model, optimizer, scheduler, step, tokenizer, config, output_dir):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
    
    # Save optimizer and scheduler
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step
    }, os.path.join(checkpoint_dir, "optimizer.pt"))
    
    # Save tokenizer
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))
    
    # Save config
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f)
    
    print(f"Checkpoint saved at step {step} to {checkpoint_dir}")

def load_checkpoint(model, optimizer, scheduler, tokenizer, resume_path, device):
    """Load model from checkpoint"""
    print(f"Resuming from checkpoint: {resume_path}")
    
    # Load model
    model.load_state_dict(torch.load(os.path.join(resume_path, "model.pt"), map_location=device))
    
    # Load optimizer and scheduler
    checkpoint = torch.load(os.path.join(resume_path, "optimizer.pt"), map_location=device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    step = checkpoint["step"]
    
    # Load tokenizer if exists
    if os.path.exists(os.path.join(resume_path, "tokenizer.json")):
        tokenizer = SimpleTokenizer.load(os.path.join(resume_path, "tokenizer.json"))
    
    return model, optimizer, scheduler, tokenizer, step

def plot_metrics(metrics_file, output_dir):
    """Plot training metrics"""
    if not os.path.exists(metrics_file):
        print(f"No metrics file found at {metrics_file}")
        return
    
    # Load metrics
    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            metrics.append(json.loads(line))
    
    if not metrics:
        print("No metrics to plot")
        return
    
    # Extract data
    steps = [m.get("step", i) for i, m in enumerate(metrics)]
    train_losses = [m.get("train_loss", None) for m in metrics]
    eval_losses = [m.get("eval_loss", None) for m in metrics]
    perplexities = [m.get("eval_perplexity", None) for m in metrics]
    
    # Filter out None values
    train_steps = [s for s, l in zip(steps, train_losses) if l is not None]
    train_losses = [l for l in train_losses if l is not None]
    
    eval_steps = [s for s, l in zip(steps, eval_losses) if l is not None]
    eval_losses = [l for l in eval_losses if l is not None]
    
    ppl_steps = [s for s, p in zip(steps, perplexities) if p is not None]
    perplexities = [p for p in perplexities if p is not None]
    
    # Create output directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training loss
    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(plots_dir, "train_loss.png"))
        plt.close()
    
    # Plot evaluation loss
    if eval_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_steps, eval_losses)
        plt.title("Evaluation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(plots_dir, "eval_loss.png"))
        plt.close()
    
    # Plot perplexity
    if perplexities:
        plt.figure(figsize=(10, 6))
        plt.plot(ppl_steps, perplexities)
        plt.title("Perplexity")
        plt.xlabel("Step")
        plt.ylabel("Perplexity")
        plt.savefig(os.path.join(plots_dir, "perplexity.png"))
        plt.close()
    
    print(f"Metrics plots saved to {plots_dir}")

def log_metrics(metrics, output_dir):
    """Log metrics to file"""
    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    # Add timestamp
    metrics["timestamp"] = time.time()
    
    # Write to file
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")

def train_agi(args):
    """Main training function"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create config
    config = AGIConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        log_dir=os.path.join(args.output_dir, "logs")
    )
    
    # Find data files
    data_files = []
    if os.path.exists(args.data_dir):
        for root, _, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith(".txt"):
                    data_files.append(os.path.join(root, file))
    
    if not data_files:
        # Create a small sample data file for demonstration
        os.makedirs(args.data_dir, exist_ok=True)
        sample_file = os.path.join(args.data_dir, "sample.txt")
        with open(sample_file, "w") as f:
            f.write("This is a sample text for training our AGI model.\n")
            f.write("Artificial general intelligence is the hypothetical ability of an intelligent agent to understand or learn any intellectual task that a human being can.\n")
            f.write("Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.\n")
        data_files.append(sample_file)
        print(f"Created sample data file: {sample_file}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Create model
    model = SimpleAGI(config)
    model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.max_steps)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        model, optimizer, scheduler, tokenizer, start_step = load_checkpoint(
            model, optimizer, scheduler, tokenizer, args.resume_from, device
        )
    
    # Load data
    data_loader, tokenizer = load_text_data(
        data_files,
        tokenizer=tokenizer if tokenizer.fitted else None,
        vocab_size=config.vocab_size,
        max_length=config.context_length,
        batch_size=args.batch_size
    )
    
    # Split data for training and evaluation
    train_size = int(0.9 * len(data_loader.dataset))
    eval_size = len(data_loader.dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        data_loader.dataset, [train_size, eval_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_loader.collate_fn
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_loader.collate_fn
    )
    
    # Training loop
    print(f"Starting training from step {start_step}")
    model.train()
    total_loss = 0
    start_time = time.time()
    best_eval_loss = float("inf")
    
    for step in range(start_step, config.max_steps):
        # Get batch
        try:
            batch = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        logits, loss = model(input_ids, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Log progress
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step+1}/{config.max_steps}, Loss: {loss.item():.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                  f"Time: {elapsed:.2f}s, "
                  f"Tokens per second: {input_ids.numel() * 10 / elapsed:.2f}")
            
            # Log metrics
            log_metrics({
                "step": step + 1,
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_time": elapsed,
                "tokens_per_second": input_ids.numel() * 10 / elapsed
            }, args.output_dir)
            
            start_time = time.time()
            total_loss = 0
        
        # Evaluate model
        if (step + 1) % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device)
            
            # Log eval metrics
            perplexity = torch.exp(torch.tensor(eval_loss)).item()
            log_metrics({
                "step": step + 1,
                "eval_loss": eval_loss,
                "eval_perplexity": perplexity
            }, args.output_dir)
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(
                    model, optimizer, scheduler, step + 1, tokenizer, config,
                    os.path.join(args.output_dir, "best_model")
                )
            
            model.train()
        
        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, step + 1, tokenizer, config,
                args.output_dir
            )
            
            # Plot metrics
            plot_metrics(
                os.path.join(args.output_dir, "metrics.jsonl"),
                args.output_dir
            )
            
            # Run benchmarks
            if step + 1 >= config.max_steps // 2:  # Only benchmark in the second half of training
                print("Running benchmarks...")
                Benchmarks.run_all_benchmarks(model, tokenizer, device)
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config.max_steps, tokenizer, config,
        os.path.join(args.output_dir, "final_model")
    )
    
    # Final evaluation
    eval_loss = evaluate(model, eval_loader, device)
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    print(f"Final evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    # Run final benchmarks
    print("Running final benchmarks...")
    overall_score = Benchmarks.run_all_benchmarks(model, tokenizer, device)
    
    # Plot final metrics
    plot_metrics(
        os.path.join(args.output_dir, "metrics.jsonl"),
        args.output_dir
    )
    
    print(f"Training completed! Final benchmark score: {overall_score:.4f}")
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model')}")

def evaluate(model, data_loader, device):
    """Evaluation function"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits, loss = model(input_ids, labels)
            
            # Update total loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return avg_loss

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, device="cpu"):
    """Generate text using the trained model"""
    model.eval()
    
    # Tokenize prompt
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate text
    output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature, top_k=top_k)
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text

if __name__ == "__main__":
    args = parse_args()
    train_agi(args)
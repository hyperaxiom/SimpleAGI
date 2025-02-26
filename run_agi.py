# run_agi.py
# Simple script to run and interact with our AGI model

import argparse
import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import our modules
from simple_agi import SimpleAGI, AGIConfig, Benchmarks
from data_utils import SimpleTokenizer
from train_agi import generate_text

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Simple AGI model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for text generation")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k for text generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length for generated text")
    
    return parser.parse_args()

def load_model(model_path, device):
    """Load the trained model and tokenizer"""
    # Load config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = AGIConfig(**config_dict)
    else:
        print(f"No config found at {config_path}, using default config")
        config = AGIConfig()
    
    # Create model
    model = SimpleAGI(config)
    
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(os.path.join(model_path, "tokenizer.json"))
    
    return model, tokenizer, config

def interactive_mode(model, tokenizer, device, temperature=0.8, top_k=40, max_length=200):
    """Run the model in interactive mode"""
    print("\n" + "="*50)
    print("Interactive AGI Mode")
    print("Type 'exit' to quit")
    print("="*50 + "\n")
    
    while True:
        # Get input
        prompt = input("\nYou: ")
        
        if prompt.lower() in ["exit", "quit", "q"]:
            break
        
        # Generate response
        print("\nAGI: ", end="")
        
        # Generate text
        start_time = datetime.now()
        generated_text = generate_text(
            model, tokenizer, prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            device=device
        )
        end_time = datetime.now()
        
        # Calculate generation time
        generation_time = (end_time - start_time).total_seconds()
        
        # Print response
        print(generated_text)
        print(f"\n[Generated {len(generated_text.split())} words in {generation_time:.2f} seconds]")

def run_benchmarks(model, tokenizer, device):
    """Run all benchmarks"""
    print("\n" + "="*50)
    print("Running AGI Benchmarks")
    print("="*50 + "\n")
    
    overall_score = Benchmarks.run_all_benchmarks(model, tokenizer, device)
    
    print(f"\nOverall benchmark score: {overall_score:.4f}")

def main():
    args = parse_args()
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer, config = load_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run benchmarks if requested
    if args.benchmark:
        run_benchmarks(model, tokenizer, device)
    
    # Run interactive mode if requested
    if args.interactive:
        interactive_mode(
            model, tokenizer, device,
            temperature=args.temperature,
            top_k=args.top_k,
            max_length=args.max_length
        )
    
    # If neither benchmark nor interactive, run a simple demo
    if not args.benchmark and not args.interactive:
        prompts = [
            "Artificial general intelligence is",
            "The future of humanity depends on",
            "The most important skill for an AI to learn is",
            "The relationship between humans and machines will"
        ]
        
        print("\n" + "="*50)
        print("AGI Demo Generation")
        print("="*50 + "\n")
        
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            print("-" * 30)
            
            generated_text = generate_text(
                model, tokenizer, prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            
            print(f"Generated: {generated_text}")
            print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
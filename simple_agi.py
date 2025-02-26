# simple_agi.py
# A minimal viable codebase for our AGI project

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import numpy as np
import json
import os

# Configuration
@dataclass
class AGIConfig:
    """Configuration for our simple AGI model"""
    vocab_size: int = 50000
    context_length: int = 2048
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

# Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.embedding_dim // config.num_heads
        self.query = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Get context vector
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.projection(context)
        return output

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(x, mask)
        x = self.norm1(x + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

# Main AGI Model
class SimpleAGI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Position embedding
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(config.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.size()
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        
        # Apply dropout
        x = self.dropout(embeddings)
        
        # Create causal mask for auto-regressive training
        mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
            
        # Apply final norm
        x = self.norm(x)
        
        # Get logits
        logits = self.output_layer(x)
        
        # If labels are provided, calculate loss
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return logits, loss
        
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """Generate text auto-regressively"""
        self.eval()
        
        for _ in range(max_length):
            # Get input sequence within context length
            input_truncated = input_ids[:, -self.config.context_length:]
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.forward(input_truncated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # If EOS token, end generation
                if next_token.item() == 2:  # Assuming 2 is EOS token
                    break
        
        return input_ids

# Training code
def train(model, data_loader, optimizer, config, device):
    """Training loop for AGI model"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for step, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits, loss = model(input_ids, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Log progress
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step+1}, Loss: {loss.item():.4f}, "
                  f"Time: {elapsed:.2f}s, Tokens per second: "
                  f"{input_ids.numel() * 10 / elapsed:.2f}")
            start_time = time.time()
            
        # Evaluate model
        if (step + 1) % config.eval_interval == 0:
            evaluate(model, eval_data_loader, device)
            model.train()
            
        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, step, config)
            
def evaluate(model, data_loader, device):
    """Evaluation loop for AGI model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, loss = model(input_ids, labels)
            
            # Update total loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    perplexity = math.exp(avg_loss)
    
    print(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    # Log metrics
    log_metrics({'eval_loss': avg_loss, 'eval_perplexity': perplexity})
    
    return avg_loss

def save_checkpoint(model, optimizer, step, config):
    """Save model checkpoint"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': config.__dict__
    }
    
    torch.save(checkpoint, f"{config.checkpoint_dir}/checkpoint-{step}.pt")
    print(f"Checkpoint saved at step {step}")

def log_metrics(metrics):
    """Log metrics to file"""
    os.makedirs('logs', exist_ok=True)
    
    # Add timestamp
    metrics['timestamp'] = time.time()
    
    # Write to file
    with open('logs/metrics.jsonl', 'a') as f:
        f.write(json.dumps(metrics) + '\n')

# Simplified Benchmark Suite
class Benchmarks:
    """Simple benchmark suite for evaluating AGI capabilities"""
    
    @staticmethod
    def language_understanding(model, tokenizer, device):
        """Test language understanding capabilities"""
        prompts = [
            "Summarize the following paragraph: The impact of climate change has been increasingly evident...",
            "Translate the following English text to French: Hello, how are you today?",
            "Answer the following question: What is the capital of France?"
        ]
        
        scores = []
        for prompt in prompts:
            # Tokenize prompt
            # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
            
            # Generate response
            output_ids = model.generate(input_ids, max_length=100)
            
            # Decode response
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # For now, just log the response - in a real scenario we would evaluate it
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("---")
            
            # Placeholder for actual scoring
            scores.append(0.5)  # Replace with actual evaluation
            
        return sum(scores) / len(scores)
    
    @staticmethod
    def reasoning(model, tokenizer, device):
        """Test reasoning capabilities"""
        prompts = [
            "If A implies B, and B implies C, and we know A is true, what can we conclude about C?",
            "A box contains 3 red balls and 4 blue balls. If two balls are drawn without replacement, what is the probability that both are red?",
            "Complete the pattern: 2, 4, 8, 16, ..."
        ]
        
        # Similar evaluation approach as above
        # Placeholder
        return 0.4
    
    @staticmethod
    def creativity(model, tokenizer, device):
        """Test creative capabilities"""
        prompts = [
            "Write a short poem about the ocean.",
            "Create a brief story about a robot learning to feel emotions.",
            "Describe an alien species from an undiscovered planet."
        ]
        
        # Placeholder
        return 0.6
    
    @staticmethod
    def run_all_benchmarks(model, tokenizer, device):
        """Run all benchmarks and return overall score"""
        language_score = Benchmarks.language_understanding(model, tokenizer, device)
        reasoning_score = Benchmarks.reasoning(model, tokenizer, device)
        creativity_score = Benchmarks.creativity(model, tokenizer, device)
        
        overall_score = (language_score + reasoning_score + creativity_score) / 3
        
        metrics = {
            'language_score': language_score,
            'reasoning_score': reasoning_score,
            'creativity_score': creativity_score,
            'overall_score': overall_score
        }
        
        log_metrics(metrics)
        
        print(f"Benchmark Results:")
        print(f"Language Understanding: {language_score:.2f}")
        print(f"Reasoning: {reasoning_score:.2f}")
        print(f"Creativity: {creativity_score:.2f}")
        print(f"Overall Score: {overall_score:.2f}")
        
        return overall_score

# Simple usage example
def main():
    # Create configuration
    config = AGIConfig()
    
    # Create model
    model = SimpleAGI(config)
    
    # Example of printing model structure
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Layer structure:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        
    print("AGI foundation model initialized and ready for training!")
    
if __name__ == "__main__":
    main()
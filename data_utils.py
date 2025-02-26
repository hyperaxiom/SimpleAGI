# data_utils.py
# Simple data loading and tokenization utilities for our AGI project

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Optional, Union
import random
from collections import Counter
import re

class SimpleTokenizer:
    """A basic tokenizer for our AGI project"""
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.token_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_counts = Counter()
        self.fitted = False
        
    def fit(self, texts):
        """Build vocabulary from texts"""
        # Simple word-level tokenization
        for text in texts:
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            self.word_counts.update(words)
            
        # Take most common words
        most_common = self.word_counts.most_common(self.vocab_size - 4)  # -4 for special tokens
        
        # Add to vocabulary
        for i, (word, _) in enumerate(most_common):
            self.token_to_id[word] = i + 4  # +4 for special tokens
            self.id_to_token[i + 4] = word
            
        self.fitted = True
        print(f"Vocabulary built with {len(self.token_to_id)} tokens")
        
    def encode(self, text, add_special_tokens=True):
        """Convert text to token IDs"""
        if not self.fitted:
            raise ValueError("Tokenizer needs to be fitted before encoding")
            
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        ids = []
        
        if add_special_tokens:
            ids.append(self.token_to_id["<BOS>"])
            
        for word in words:
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            else:
                ids.append(self.token_to_id["<UNK>"])
                
        if add_special_tokens:
            ids.append(self.token_to_id["<EOS>"])
            
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """Convert token IDs back to text"""
        tokens = []
        
        for id in ids:
            token = self.id_to_token.get(id, "<UNK>")
            if skip_special_tokens and token in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
                continue
            tokens.append(token)
            
        return " ".join(tokens)
    
    def save(self, path):
        """Save tokenizer vocabulary to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "vocab_size": self.vocab_size,
                "fitted": self.fitted
            }, f)
            
    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(id): token for id, token in data["token_to_id"].items()}
        tokenizer.fitted = data["fitted"]
        
        return tokenizer
    
    def batch_encode(self, texts, max_length=None, padding=True, truncation=True):
        """Encode a batch of texts"""
        batch_ids = []
        
        for text in texts:
            ids = self.encode(text)
            
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
                
            batch_ids.append(ids)
            
        if padding and max_length:
            # Pad sequences to max_length
            batch_ids = [ids + [self.token_to_id["<PAD>"]] * (max_length - len(ids)) 
                         for ids in batch_ids]
            
        return batch_ids
    
class TextDataset(Dataset):
    """Simple dataset for text data"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Encode all texts
        self.encoded_texts = []
        for text in texts:
            encoded = self.tokenizer.encode(text)
            if len(encoded) > 1:  # Ensure there's at least one token besides special tokens
                self.encoded_texts.append(encoded)
        
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        encoded_text = self.encoded_texts[idx]
        
        # Truncate if necessary
        if len(encoded_text) > self.max_length:
            start_idx = random.randint(0, len(encoded_text) - self.max_length)
            encoded_text = encoded_text[start_idx:start_idx + self.max_length]
            
        # Create input_ids and labels (shifted right for next token prediction)
        input_ids = torch.tensor(encoded_text)
        labels = input_ids.clone()
        
        return {"input_ids": input_ids, "labels": labels}
    
def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Find max length in batch
    max_length = max(len(ids) for ids in input_ids)
    
    # Pad sequences
    input_ids_padded = torch.stack([
        torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)]) 
        for ids in input_ids
    ])
    
    labels_padded = torch.stack([
        torch.cat([lbs, torch.zeros(max_length - len(lbs), dtype=torch.long)]) 
        for lbs in labels
    ])
    
    return {"input_ids": input_ids_padded, "labels": labels_padded}

def load_text_data(file_paths, tokenizer=None, vocab_size=50000, max_length=512, batch_size=16):
    """Load text data from files and prepare for training"""
    all_texts = []
    
    # Load texts
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            all_texts.extend([text.strip() for text in texts if text.strip()])
            
    print(f"Loaded {len(all_texts)} text segments")
    
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        tokenizer.fit(all_texts)
        
    # Create dataset
    dataset = TextDataset(all_texts, tokenizer, max_length=max_length)
    
    # Create data loader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return data_loader, tokenizer

def prepare_benchmark_data():
    """Prepare data for benchmarking AGI capabilities"""
    # This is a placeholder. In a real system, we would load actual benchmark datasets
    
    # Example benchmark data
    benchmark_data = {
        "language_understanding": {
            "questions": [
                "What is the capital of France?",
                "Who wrote the novel 'Pride and Prejudice'?",
                "Explain the process of photosynthesis briefly."
            ],
            "references": [
                "Paris",
                "Jane Austen",
                "Photosynthesis is the process where plants convert sunlight into energy, using carbon dioxide and water to produce glucose and oxygen."
            ]
        },
        "reasoning": {
            "problems": [
                "If Sarah has 5 apples and gives 2 to John, how many does she have left?",
                "All mammals are warm-blooded. A whale is a mammal. Is a whale warm-blooded?",
                "Continue the sequence: 2, 4, 8, 16, ..."
            ],
            "solutions": [
                "3",
                "Yes",
                "32"
            ]
        },
        "creativity": {
            "prompts": [
                "Write a short poem about the moon.",
                "Describe a world where humans can fly.",
                "Create a new animal by combining features of existing animals."
            ]
        }
    }
    
    return benchmark_data

# Example usage
def main():
    # Example text data
    sample_texts = [
        "Artificial general intelligence is the ability of an intelligent agent to understand or learn any intellectual task that a human being can.",
        "Machine learning is a field of computer science that uses statistical techniques to give computer systems the ability to learn from data.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."
    ]
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.fit(sample_texts)
    
    # Encode text
    encoded = tokenizer.encode(sample_texts[0])
    print(f"Encoded: {encoded}")
    
    # Decode back
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Create dataset
    dataset = TextDataset(sample_texts, tokenizer, max_length=20)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Check batch
    for batch in dataloader:
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        break
        
if __name__ == "__main__":
    main()
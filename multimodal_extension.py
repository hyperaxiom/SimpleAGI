# multimodal_extension.py
# Simple extension to add vision capabilities to our AGI system

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from simple_agi import SimpleAGI, AGIConfig

class VisionEncoder(nn.Module):
    """Vision encoder component for multimodal AGI"""
    
    def __init__(self, embedding_dim=768, pretrained=True):
        super().__init__()
        
        # Use a pretrained ResNet as the vision backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Project visual features to embedding dimension
        self.projection = nn.Linear(2048, embedding_dim)
        
        # Initialize normalization transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def forward(self, images):
        """
        Process a batch of images
        images: tensor of shape [batch_size, 3, 224, 224]
        """
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(images)
        
        # Reshape features: [batch_size, 2048, 1, 1] -> [batch_size, 2048]
        features = features.squeeze(-1).squeeze(-1)
        
        # Project to embedding dimension
        embeddings = self.projection(features)
        
        return embeddings
    
    def preprocess_image(self, image):
        """Preprocess a PIL Image or file-like object"""
        if isinstance(image, bytes) or isinstance(image, io.IOBase):
            image = Image.open(image).convert('RGB')
            
        return self.transform(image).unsqueeze(0)  # Add batch dimension

class MultimodalAGI(nn.Module):
    """Multimodal AGI model with vision and language capabilities"""
    
    def __init__(self, config, vision_embedding_dim=768):
        super().__init__()
        
        self.config = config
        
        # Create language model
        self.language_model = SimpleAGI(config)
        
        # Create vision encoder
        self.vision_encoder = VisionEncoder(embedding_dim=vision_embedding_dim)
        
        # Create cross-modal adapter
        self.vision_to_text_adapter = nn.Sequential(
            nn.Linear(vision_embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        
        # Token for image representation in text
        self.image_token_embedding = nn.Parameter(torch.randn(1, config.embedding_dim))
        
    def encode_images(self, images):
        """Encode a batch of images"""
        # Process images with vision encoder
        image_features = self.vision_encoder(images)
        
        # Project to text embedding space
        image_embeddings = self.vision_to_text_adapter(image_features)
        
        return image_embeddings
        
    def forward(self, input_ids=None, images=None, labels=None):
        """
        Forward pass of the multimodal model
        input_ids: tensor of token IDs [batch_size, seq_len]
        images: tensor of images [batch_size, 3, 224, 224] or None
        labels: tensor of target token IDs [batch_size, seq_len] or None
        """
        batch_size = input_ids.size(0) if input_ids is not None else images.size(0)
        
        # Get token embeddings from language model
        token_embeddings = self.language_model.token_embedding(input_ids)
        
        # Add position embeddings
        positions = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, input_ids.size(1))
        position_embeddings = self.language_model.position_embedding(positions)
        
        embeddings = token_embeddings + position_embeddings
        
        # Process images if provided
        if images is not None:
            # Get image embeddings
            image_embeddings = self.encode_images(images)
            
            # Replace the <image> token embeddings with actual image embeddings
            # Assuming <image> token has a specific ID, e.g., 5
            image_token_id = 5  # This should be defined in your tokenizer
            image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
            
            # If there are image tokens in the input
            if image_token_positions[0].size(0) > 0:
                for i, pos in enumerate(zip(image_token_positions[0], image_token_positions[1])):
                    batch_idx, seq_idx = pos
                    if i < image_embeddings.size(0):  # Make sure we have enough image embeddings
                        embeddings[batch_idx, seq_idx] = image_embeddings[i % image_embeddings.size(0)]
        
        # Apply dropout
        x = self.language_model.dropout(embeddings)
        
        # Create causal mask for auto-regressive training
        seq_len = input_ids.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.language_model.transformer_blocks:
            x = block(x, mask)
            
        # Apply final norm
        x = self.language_model.norm(x)
        
        # Get logits
        logits = self.language_model.output_layer(x)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return logits, loss
    
    def generate_with_image(self, input_ids, image, max_length=100, temperature=1.0, top_k=50):
        """Generate text conditioned on an image"""
        self.eval()
        
        # Preprocess image
        if image is not None:
            processed_image = self.vision_encoder.preprocess_image(image).to(input_ids.device)
            image_embedding = self.encode_images(processed_image)
        else:
            image_embedding = None
            
        # Generate text auto-regressively
        for _ in range(max_length):
            # Get input sequence within context length
            input_truncated = input_ids[:, -self.config.context_length:]
            
            # Forward pass
            with torch.no_grad():
                # Create token embeddings
                token_embeddings = self.language_model.token_embedding(input_truncated)
                
                # Add position embeddings
                positions = torch.arange(0, input_truncated.size(1), dtype=torch.long, device=input_truncated.device)
                positions = positions.unsqueeze(0).expand(input_truncated.size(0), input_truncated.size(1))
                position_embeddings = self.language_model.position_embedding(positions)
                
                embeddings = token_embeddings + position_embeddings
                
                # Inject image embedding if needed
                if image_embedding is not None:
                    # Locate image tokens
                    image_token_id = 5  # This should be defined in your tokenizer
                    image_token_positions = (input_truncated == image_token_id).nonzero(as_tuple=True)
                    
                    # If there are image tokens in the input
                    if image_token_positions[0].size(0) > 0:
                        for i, pos in enumerate(zip(image_token_positions[0], image_token_positions[1])):
                            batch_idx, seq_idx = pos
                            embeddings[batch_idx, seq_idx] = image_embedding[0]  # Use the first image embedding
                
                # Apply dropout
                x = self.language_model.dropout(embeddings)
                
                # Create causal mask
                seq_len = input_truncated.size(1)
                mask = torch.tril(torch.ones((seq_len, seq_len), device=input_truncated.device)).unsqueeze(0)
                
                # Apply transformer blocks
                for block in self.language_model.transformer_blocks:
                    x = block(x, mask)
                    
                # Apply final norm
                x = self.language_model.norm(x)
                
                # Get logits for next token prediction
                logits = self.language_model.output_layer(x)
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

class MultimodalBenchmarks:
    """Benchmark suite for evaluating multimodal AGI capabilities"""
    
    @staticmethod
    def image_captioning(model, tokenizer, image_paths, device):
        """Test image captioning capabilities"""
        scores = []
        
        for image_path in image_paths:
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Prepare prompt with image token
                prompt = "Describe this image: <image>"
                input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
                
                # Generate caption
                output_ids = model.generate_with_image(input_ids, image, max_length=50)
                
                # Decode caption
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                print(f"Image: {image_path}")
                print(f"Caption: {caption}")
                print("---")
                
                # Placeholder for actual scoring
                scores.append(0.7)  # Replace with actual evaluation
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def visual_question_answering(model, tokenizer, image_paths, questions, device):
        """Test visual question answering capabilities"""
        scores = []
        
        for image_path, question in zip(image_paths, questions):
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Prepare prompt with image token
                prompt = f"Answer this question about the image: {question} <image>"
                input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
                
                # Generate answer
                output_ids = model.generate_with_image(input_ids, image, max_length=50)
                
                # Decode answer
                answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                print(f"Image: {image_path}")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print("---")
                
                # Placeholder for actual scoring
                scores.append(0.6)  # Replace with actual evaluation
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def run_all_benchmarks(model, tokenizer, image_dir, device):
        """Run all multimodal benchmarks"""
        # Find image files
        image_paths = []
        if os.path.exists(image_dir):
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print("No images found for benchmarking.")
            return 0.0
        
        # Sample questions for VQA
        questions = [
            "What color is the main object?",
            "How many people are in the image?",
            "What is happening in this scene?",
            "What is the weather like in this image?",
            "Is this indoors or outdoors?"
        ]
        
        # Run benchmarks
        caption_score = MultimodalBenchmarks.image_captioning(model, tokenizer, image_paths[:5], device)
        
        # Prepare question-image pairs
        vqa_images = image_paths[:min(5, len(image_paths))]
        vqa_questions = questions[:len(vqa_images)]
        vqa_score = MultimodalBenchmarks.visual_question_answering(model, tokenizer, vqa_images, vqa_questions, device)
        
        # Calculate overall score
        overall_score = (caption_score + vqa_score) / 2
        
        print(f"Multimodal Benchmark Results:")
        print(f"Image Captioning: {caption_score:.2f}")
        print(f"Visual Question Answering: {vqa_score:.2f}")
        print(f"Overall Multimodal Score: {overall_score:.2f}")
        
        return overall_score

def convert_language_to_multimodal(language_model_path, output_path):
    """Convert a trained language model to a multimodal model"""
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load language model and tokenizer
    config_path = os.path.join(language_model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = AGIConfig(**config_dict)
    else:
        print(f"No config found at {config_path}, using default config")
        config = AGIConfig()
    
    # Create language model
    language_model = SimpleAGI(config)
    
    # Load model weights
    language_model.load_state_dict(torch.load(os.path.join(language_model_path, "model.pt"), map_location=device))
    
    # Create multimodal model
    multimodal_model = MultimodalAGI(config)
    
    # Copy language model weights
    multimodal_model.language_model = language_model
    
    # Save multimodal model
    os.makedirs(output_path, exist_ok=True)
    torch.save(multimodal_model.state_dict(), os.path.join(output_path, "model.pt"))
    
    # Copy tokenizer
    import shutil
    tokenizer_path = os.path.join(language_model_path, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, os.path.join(output_path, "tokenizer.json"))
    
    # Save config
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config.__dict__, f)
    
    print(f"Multimodal model created and saved to {output_path}")
    return multimodal_model

def update_tokenizer_for_images(tokenizer_path, output_path):
    """Update tokenizer to include image token"""
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    
    # Add image token if not already present
    if "<image>" not in tokenizer.token_to_id:
        image_token_id = len(tokenizer.token_to_id)
        tokenizer.token_to_id["<image>"] = image_token_id
        tokenizer.id_to_token[image_token_id] = "<image>"
        
        # Save updated tokenizer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tokenizer.save(output_path)
        
        print(f"Tokenizer updated with image token and saved to {output_path}")
    else:
        print("Tokenizer already has image token")
        
    return tokenizer

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Multimodal AGI utilities")
    parser.add_argument("--language_model_path", type=str, help="Path to language model checkpoint")
    parser.add_argument("--output_path", type=str, help="Path to save multimodal model")
    parser.add_argument("--update_tokenizer", action="store_true", help="Update tokenizer with image token")
    
    args = parser.parse_args()
    
    if args.update_tokenizer and args.language_model_path:
        tokenizer_path = os.path.join(args.language_model_path, "tokenizer.json")
        output_path = os.path.join(args.output_path, "tokenizer.json") if args.output_path else tokenizer_path
        update_tokenizer_for_images(tokenizer_path, output_path)
        
    if args.language_model_path and args.output_path:
        convert_language_to_multimodal(args.language_model_path, args.output_path)
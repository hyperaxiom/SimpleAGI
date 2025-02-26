# Simple AGI Project

This repository contains a minimal viable implementation of an Artificial General Intelligence (AGI) project. Following the philosophy that "the simplest approach might be the smartest," this codebase provides a foundation for developing and exploring AGI capabilities.

## Project Structure

```
simple-agi/
├── simple_agi.py       # Core AGI model implementation
├── data_utils.py       # Data loading and tokenization utilities
├── train_agi.py        # Training script
├── run_agi.py          # Script to run and interact with trained models
├── multimodal_extension.py  # Extension for multimodal capabilities
└── data/               # Directory for training data
    └── sample.txt      # Sample training data
```

## Core Components

1. **SimpleAGI** - A transformer-based language model that serves as the foundation for AGI
2. **Tokenizer** - A simple word-level tokenizer for processing text
3. **Training Pipeline** - Tools for training and evaluating the model
4. **Benchmarks** - Simple benchmarks to evaluate AGI capabilities
5. **Multimodal Extension** - Optional extension to add vision capabilities

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/simple-agi.git
cd simple-agi
pip install -r requirements.txt
```

### Training

To train the model from scratch:

```bash
python train_agi.py --data_dir data/ --output_dir output/
```

You can customize training with additional arguments:

```bash
python train_agi.py --data_dir data/ --output_dir output/ --embedding_dim 512 --num_layers 8 --batch_size 32 --learning_rate 5e-4
```

### Running the Model

To interact with a trained model:

```bash
python run_agi.py --model_path output/final_model --interactive
```

To run benchmarks on a trained model:

```bash
python run_agi.py --model_path output/final_model --benchmark
```

### Multimodal Extension

To convert a trained language model to a multimodal model:

```bash
python multimodal_extension.py --language_model_path output/final_model --output_path output/multimodal_model --update_tokenizer
```

## Model Architecture

The core AGI model is based on a transformer architecture:

- Embedding layer for tokens and positions
- Multiple transformer blocks with self-attention and feed-forward layers
- Layer normalization and residual connections
- Output layer for next token prediction

The multimodal extension adds:

- Vision encoder based on ResNet50
- Cross-modal adapter to connect vision and language
- Special token handling for image inputs

## Benchmarks

The model is evaluated on the following capabilities:

1. **Language Understanding** - Comprehension, summarization, translation
2. **Reasoning** - Logical reasoning, mathematical problem-solving, pattern recognition
3. **Creativity** - Creative writing, storytelling, idea generation
4. **Multimodal Understanding** (with extension) - Image captioning, visual question answering

## Iterative Development Philosophy

This project follows an iterative development approach:

1. Start with a simple, functional foundation
2. Benchmark and identify limitations
3. Make targeted improvements
4. Repeat

We believe that AGI will emerge from an iterative process of scaling and refining capabilities, rather than from a single revolutionary architecture.

## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues to discuss improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

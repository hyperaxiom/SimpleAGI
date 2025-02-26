python -c "
from datasets import load_dataset
import os

# Create directory
os.makedirs('data_wiki', exist_ok=True)

# Load the standard wikitext-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Save to file
with open('data_wiki/wikitext.txt', 'w', encoding='utf-8') as f:
    for item in dataset:
        f.write(item['text'] + '\n')

print('Dataset saved to data_wiki/wikitext.txt')
"

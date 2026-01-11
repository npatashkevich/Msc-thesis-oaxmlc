import os
import json
import pandas as pd
import numpy as np
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tqdm import tqdm
import argparse
import re

# --- Constants for Data Filtering ---
# Broad concepts and root categories excluded to focus on specific labels
EXCLUDED_CONCEPTS = [
    "C154945302", "C11413529", "C199360897", "C111919701", 
    "C76155785", "C31258907", "C556297831", "C41008148"
]

def clean_text(text):
    """
    Standardizes text by removing HTML tags, lowercasing, 
    and removing extra whitespaces.
    """
    if not isinstance(text, str): return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)  
    # Normalize whitespaces and lowercase
    return " ".join(text.split()).lower()

def hierarchy_pruning(df, min_df=50):
    """
    Recursively removes leaf labels that appear fewer than min_df times.
    Documents that end up with no labels are also removed.
    """
    print(f"Starting hierarchy pruning (min_df={min_df})...")
    while True:
        # Calculate label frequencies
        label_freq = df['concepts'].explode().value_counts()
        # Identify labels to drop
        to_drop = set(label_freq[label_freq < min_df].index)
        
        if not to_drop: 
            break
            
        # Filter labels and remove documents with empty label sets
        df['concepts'] = df['concepts'].apply(lambda x: [l for l in x if l not in to_drop])
        df = df[df['concepts'].map(len) > 0]
    return df

def main(args):
    # --- Step 1: Data Loading & Cleaning ---
    print("Step 1: Loading and cleaning raw data...")
    df = pd.read_json(args.input, lines=True)
    
    # Combine title and abstract for the input text
    df['text'] = (df['title'] + " " + df.get('abstract', '')).apply(clean_text)
    
    # Remove duplicates based on text content
    df = df.drop_duplicates(subset=['text'])
    
    # Exclude broad/root concepts
    df['concepts'] = df['concepts'].apply(lambda x: [l for l in x if l not in EXCLUDED_CONCEPTS])
    df = df[df['concepts'].map(len) > 0]
    
    # Perform recursive pruning as described in the methodology
    df = hierarchy_pruning(df)

    # --- Step 2: Subset Creation & Train/Test Splitting ---
    print(f"Step 2: Creating {args.subset} subset and performing splits...")
    if args.subset == '120k':
        df = df.sample(n=120000, random_state=42)
    elif args.subset == '1m':
        df = df.sample(n=1000000, random_state=42)

    # Split into Train (80%), Validation (10%), and Test (10%)
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # --- Step 3: SentencePiece Tokenization & Artifact Export ---
    print("Step 3: Training SentencePiece and exporting preprocessed data...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save raw training text for SentencePiece trainer
    tmp_train_text = os.path.join(args.out_dir, "train_texts_raw.txt")
    train['text'].to_csv(tmp_train_text, index=False, header=False)

    # Train BPE model with a 10k vocabulary size
    spm.SentencePieceTrainer.train(
        input=tmp_train_text, model_prefix='spm_10k_bpe',
        vocab_size=10000, model_type='bpe'
    )
    sp = spm.SentencePieceProcessor(model_file='spm_10k_bpe.model')

    # Generate label mapping (label2id)
    all_labels = sorted(list(set(df['concepts'].explode())))
    label2id = {l: i for i, l in enumerate(all_labels)}
    with open(os.path.join(args.out_dir, "label2id.json"), 'w') as f:
        json.dump(label2id, f)

    # Internal helper to tokenize and save as .npy for AttentionXML/MATCH baselines
    def tokenize_and_save(data, name):
        ids = [sp.encode(t) for t in data['text']]
        np.save(os.path.join(args.out_dir, f"{name}_texts.npy"), np.array(ids, dtype=object))
        print(f"Saved: {name}_texts.npy")

    tokenize_and_save(train, "train")
    tokenize_and_save(test, "test")

    # Initialize word embeddings using Xavier normal distribution
    # This matches the initialization used in the benchmarked models
    vocab_size = 10000
    emb_dim = 300
    emb_init = np.random.randn(vocab_size, emb_dim) * np.sqrt(2 / (vocab_size + emb_dim))
    np.save(os.path.join(args.out_dir, "emb_init.npy"), emb_init)
    
    print(f"Preprocessing complete. All artifacts saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for OAXMLC dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw documents.json file")
    parser.add_argument("--subset", type=str, choices=['1m', '120k'], default='120k', help="Select subset size")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory for processed files")
    args = parser.parse_args()
    main(args)
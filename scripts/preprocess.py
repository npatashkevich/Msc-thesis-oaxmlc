"""
OAXMLC Preprocessing Pipeline
Matches the methodology of the MSc Thesis (UniFR 2025).
Handles text cleaning, recursive label pruning (min_df=50), and BPE tokenization.
"""

import os
import json
import pandas as pd
import numpy as np
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import argparse
import re

EXCLUDED_CONCEPTS = [
    "C154945302", "C11413529", "C199360897", "C111919701", 
    "C76155785", "C31258907", "C556297831", "C41008148"
]

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', '', text)  
    return " ".join(text.split()).lower()

def hierarchy_pruning(df, min_df=50):
    print(f"Starting hierarchy pruning (min_df={min_df})...")
    while True:
        label_freq = df['concepts'].explode().value_counts()
        to_drop = set(label_freq[label_freq < min_df].index)
        if not to_drop: break
        df['concepts'] = df['concepts'].apply(lambda x: [l for l in x if l not in to_drop])
        df = df[df['concepts'].map(len) > 0]
    return df

def main(args):
    print("Step 1: Loading and cleaning raw data...")
    df = pd.read_json(args.input, lines=True)
    df['text'] = (df['title'] + " " + df.get('abstract', '')).apply(clean_text)
    df = df.drop_duplicates(subset=['text'])
    df['concepts'] = df['concepts'].apply(lambda x: [l for l in x if l not in EXCLUDED_CONCEPTS])
    df = df[df['concepts'].map(len) > 0]
    df = hierarchy_pruning(df)

    print(f"Step 2: Creating {args.subset} subset and performing splits...")
    if args.subset == '120k':
        df = df.sample(n=120000, random_state=42)
    elif args.subset == '1m':
        df = df.sample(n=1000000, random_state=42)

    # 80/10/10 Split
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    print("Step 3: Artifact export...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Train SentencePiece
    tmp_text = os.path.join(args.out_dir, "tmp_spm.txt")
    train['text'].to_csv(tmp_text, index=False, header=False)
    spm.SentencePieceTrainer.train(input=tmp_text, model_prefix='spm_10k_bpe', vocab_size=10000, model_type='bpe')
    sp = spm.SentencePieceProcessor(model_file='spm_10k_bpe.model')

    def tokenize_and_save(data, name):
        ids = [sp.encode(t) for t in data['text']]
        np.save(os.path.join(args.out_dir, f"{name}_texts.npy"), np.array(ids, dtype=object))

    tokenize_and_save(train, "train")
    tokenize_and_save(val, "val")
    tokenize_and_save(test, "test")

    # Save Label Mapping
    all_labels = sorted(list(set(df['concepts'].explode())))
    label2id = {l: i for i, l in enumerate(all_labels)}
    with open(os.path.join(args.out_dir, "label2id.json"), 'w') as f:
        json.dump(label2id, f)

    print(f"Done. Artifacts saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--subset", type=str, choices=['1m', '120k'], default='120k')
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    main(args)

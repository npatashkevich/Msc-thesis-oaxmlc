import numpy as np
import argparse

def calculate_propensity(train_labels_path, out_path, num_labels=7499):
    """
    Computes Inverse Propensity weights to account for the power-law distribution 
    of labels in the OpenAlex dataset. 
    Following the approach by Bhatia et al. (2016).
    """
    print(f"Loading training labels from: {train_labels_path}")
    train_labels = np.load(train_labels_path, allow_pickle=True)
    
    # Initialize counts for each label
    counts = np.zeros(num_labels, dtype=np.int64)
    for row in train_labels:
        if row is None: continue
        if np.isscalar(row): row = [int(row)]
        for label_id in row:
            if 0 <= label_id < num_labels:
                counts[label_id] += 1

    # Standard propensity parameters (A, B, C)
    A, B, C = 0.55, 1.5, 0.5
    inv_prop = 1.0 + A * np.power(counts + B, -C)
    
    np.save(out_path, inv_prop)
    print(f"Success: Inverse propensity weights saved to {out_path}")

if __name__ == "__main__":
    calculate_propensity("data/train_labels.npy", "data/inv_prop.npy")
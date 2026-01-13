import os
import numpy as np
from scipy.sparse import csr_matrix, load_npz

def dcg_at_k(rel, k):
    """
    Computes Discounted Cumulative Gain (DCG) at rank K.
    """
    rel = np.asarray(rel, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float((rel * discounts).sum())

def idcg_at_k_weights(true_weights, k):
    """
    Computes Ideal DCG for propensity-scored metrics by ranking 
    the rarest labels (highest weights) first.
    """
    if len(true_weights) == 0:
        return 0.0
    w = np.sort(np.asarray(true_weights, dtype=np.float64))[::-1]
    w = w[: min(k, w.size)]
    discounts = 1.0 / np.log2(np.arange(2, w.size + 2))
    return float((w * discounts).sum())

def topk_from_csr_row(row_data, row_indices, k):
    """
    Extracts top-K indices from a sparse CSR matrix row based on scores.
    """
    if row_data.size == 0:
        return []
    if row_data.size <= k:
        order = np.argsort(-row_data)
        return row_indices[order].tolist()
    # Use partition for efficiency on large label sets
    part = np.argpartition(-row_data, k - 1)[:k]
    order_in_part = np.argsort(-row_data[part])
    return row_indices[part[order_in_part]].tolist()

def run_evaluation(preds_path, labels_path, inv_prop_path, k_list=[1, 3, 5]):
    """
    Comprehensive evaluation suite for OpenAlex XMLC.
    Metrics: P@K, nDCG@K, PS-Precision@K, PS-nDCG@K, and Label Coverage.
    """
    print(f"Loading predictions: {preds_path}")
    preds = load_npz(preds_path).tocsr()
    num_samples, num_labels = preds.shape

    print("Loading ground-truth and propensity weights...")
    y_true_np = np.load(labels_path, allow_pickle=True)
    inv_prop = np.load(inv_prop_path).ravel()

    # Accumulators for metrics
    prec_sums = {k: 0.0 for k in k_list}
    ndcg_sums = {k: 0.0 for k in k_list}
    psp_sums = {k: 0.0 for k in k_list}
    psndcg_sums = {k: 0.0 for k in k_list}
    counts = {k: 0 for k in k_list}
    
    # Track predicted labels for coverage metrics
    all_predicted_labels = set()
    
    discounts = {k: 1.0 / np.log2(np.arange(2, k + 2)) for k in k_list}
    p_indptr, p_indices, p_data = preds.indptr, preds.indices, preds.data

    for i in range(num_samples):
        # 1. Prepare ground truth
        y_true = y_true_np[i]
        if np.isscalar(y_true): y_true = [y_true]
        y_true_set = set(y_true)
        n_true = len(y_true_set)
        if n_true == 0: continue

        # 2. Extract top predictions
        row_idx = p_indices[p_indptr[i]:p_indptr[i+1]]
        row_val = p_data[p_indptr[i]:p_indptr[i+1]]
        if row_val.size == 0: continue
        
        ranked_pred = topk_from_csr_row(row_val, row_idx, k=max(k_list))
        all_predicted_labels.update(ranked_pred[:5]) # Store for Coverage@5

        # 3. Calculate metrics for each K
        true_w = [inv_prop[l] for l in y_true_set]

        for k in k_list:
            if n_true >= k:
                counts[k] += 1
                pred_k = ranked_pred[:k]
                
                # Standard Metrics
                hits_binary = np.array([1.0 if lab in y_true_set else 0.0 for lab in pred_k])
                prec_sums[k] += hits_binary.sum() / k
                
                # Standard nDCG
                idcg = dcg_at_k(np.ones(min(k, n_true)), k)
                ndcg_sums[k] += dcg_at_k(hits_binary, k) / idcg if idcg > 0 else 0.0

                # Propensity-Scored Metrics (Tail-aware)
                w_pred = inv_prop[pred_k]
                
                # PS-Precision: weighted hits normalized by sum of predicted weights
                num_psp = float((w_pred * hits_binary).sum())
                den_psp = float(w_pred.sum()) if w_pred.size else 1.0
                psp_sums[k] += (num_psp / den_psp)

                # PS-nDCG
                dcg_w = float((w_pred * hits_binary * discounts[k]).sum())
                idcg_ps = idcg_at_k_weights(true_w, k)
                psndcg_sums[k] += (dcg_w / idcg_ps) if idcg_ps > 0 else 0.0

    # 4. Coverage Metrics
    coverage_at5 = len(all_predicted_labels) / float(num_labels)
    covered_idx = np.array(list(all_predicted_labels), dtype=np.int64)
    bn_coverage_at5 = inv_prop[covered_idx].sum() / inv_prop.sum()

    # 5. Output Results
    print("\n" + "="*30)
    print("📊 EVALUATION RESULTS")
    print("="*30)
    
    print("\n📈 Standard Ranking:")
    for k in k_list:
        p_k = prec_sums[k] / max(1, counts[k])
        n_k = ndcg_sums[k] / max(1, counts[k])
        print(f"  P@{k}: {p_k:.6f} | nDCG@{k}: {n_k:.6f}")

    print("\n🔧 Propensity-Scored (Tail-Aware):")
    for k in k_list:
        psp_k = psp_sums[k] / max(1, counts[k])
        psn_k = psndcg_sums[k] / max(1, counts[k])
        print(f"  PSP@{k}: {psp_k:.6f} | PS-nDCG@{k}: {psn_k:.6f}")

    print("\n📚 Model Coverage (@5):")
    print(f"  Label Coverage: {coverage_at5:.6f}")
    print(f"  BN-Coverage:    {bn_coverage_at5:.6f}")

if __name__ == "__main__":
    # Update these paths for your environment
    run_evaluation(
        preds_path="results/CascadeXML/bert_preds.npz",
        labels_path="data/processed/test_labels.npy",
        inv_prop_path="data/processed/inv_prop.npy"
    )
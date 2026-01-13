# Msc-thesis-oaxmlc

# Evaluating Advanced Models in Extreme Multi-Label Classification on OpenAlex

This repository contains code and experiments accompanying the MSc thesis:

"Evaluating Advanced Models in Extreme Multi-Label Classification on OpenAlex"
University of Fribourg, 2025

## 1. Research Goal
We evaluate specialized XMLC models (AttentionXML, MATCH, CascadeXML)
against modern LLMs (LLaMA, Mistral, Qwen) on OpenAlex-scale data,
focusing on accuracy, tail-label coverage, and efficiency.

## 2. Dataset
- Source: OAXMLC (OpenAlex snapshot Jan 20, 2025)
- Subsets:
  - OA1M
  - OA120K
- Labels: OpenAlex concepts (7,498 labels after pruning)

See Chapter 3 of the thesis.

## 3. Models
- AttentionXML
- MATCH
- CascadeXML
- LLMs (zero-shot and LoRA-tuned)

See Chapter 4.

## 4. Evaluation
Metrics:
- Precision@k, Recall@k, nDCG@k
- Propensity-scored metrics
- Tail-label coverage
- Runtime & memory

See Chapter 5.

## 5. Results
Results and plots used in the thesis are available in `results/`.

## 6. Reproducibility
Instructions to reproduce experiments.

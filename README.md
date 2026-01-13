# Evaluating Advanced Models in Extreme Multi-Label Classification on OpenAlex

This repository contains the code and experimental framework for the Master's Thesis: 
**"Evaluating Advanced Models in Extreme Multi-Label Classification on OpenAlex"** completed at the **University of Fribourg (2025)**.

## 📌 Project Overview
Extreme Multi-Label Classification (XMLC) is essential for academic search engines and scientific knowledge graphs where documents are assigned multiple labels from very large, imbalanced sets[cite: 15]. 

[cite_start]This research provides a comprehensive benchmark of specialized XMLC architectures against modern Large Language Models (LLMs) on a massive scholarly dataset[cite: 17, 18, 68].

### Key Contributions:
* [cite_start]**Specialized XMLC:** Evaluation of **AttentionXML**, **MATCH**, and **CascadeXML**[cite: 68, 75].
* [cite_start]**LLM Adaptation:** Benchmarking **Llama 3.1 8B**, **Mistral v0.3 7B**, and **Qwen 2.5 7B** using Parameter-Efficient Fine-Tuning (**LoRA**)[cite: 18, 76, 653].
* [cite_start]**Tail-Aware Analysis:** Detailed evaluation of performance on rare labels using propensity-scored metrics[cite: 20, 71, 126].

## 📊 Dataset: OAXMLC
[cite_start]We utilized the **OAXMLC** dataset, curated from an OpenAlex snapshot (Jan 20, 2025)[cite: 19, 434].
* [cite_start]**Domain:** Computer Science publications[cite: 435].
* [cite_start]**Label Space:** 7,498 unique concept labels after hierarchy-aware pruning[cite: 571, 621].
* [cite_start]**Scale:** Two reproducible subsets: **OA1M** (1M docs) and **OA120K** (label-weighted)[cite: 73, 106].

## 🚀 Model Performance Summary
[cite_start]Specialized XMLC models remain the superior choice for production-scale indexing due to their balance of accuracy and efficiency[cite: 21, 1075].

### Core Metrics (100k Test Split)
| Model | P@5 | nDCG@5 | PS-nDCG@5 (Tail-aware) | Inference Time |
| :--- | :---: | :---: | :---: | :--- |
| **AttentionXML** | **0.901** | **0.924** | 0.728 | ~10 min |
| **MATCH** | 0.740 | 0.889 | 0.609 | ~5 min |
| **CascadeXML** | 0.638 | 0.695 | **0.732** | **< 3 min** |

> [cite_start]**Key Insight:** **AttentionXML** dominates in ranking depth [cite: 1214][cite_start], while **CascadeXML** offers the best efficiency and high recall for rare "tail" labels[cite: 1001, 1216]. [cite_start]LLMs, even with LoRA, were significantly costlier to serve and underperformed in strict ranking tasks[cite: 86, 1223].

## 🛠 Technical Implementation
* [cite_start]**Environment:** PyTorch 2.2, Transformers 4.41, vLLM 0.4.2[cite: 893, 894, 897].
* [cite_start]**Training:** LoRA adaptation with rank $r = 16$ and $\alpha = 16$[cite: 937].
* [cite_start]**Hardware:** NVIDIA **L40S** and **Tesla V100** GPUs[cite: 885, 886].
* [cite_start]**Serving:** Optimized inference via **vLLM** with constrained decoding[cite: 77, 934, 943].

## 📂 Repository Structure
* [cite_start]`data/`: Preprocessing scripts for OpenAlex JSONL files[cite: 554, 900].
* [cite_start]`models/`: Implementation details for XMLC models and LLM prompt templates[cite: 671, 930].
* [cite_start]`evaluation/`: Definitions of Precision@k, nDCG@k, and Propensity-Scored variants [cite: 773-806].
* [cite_start]`results/`: Detailed logs and metrics for each model run[cite: 950, 1007, 1081].

## 🎓 Thesis Information
* [cite_start]**Author:** Natallia Patashkevich [cite: 4]
* [cite_start]**Supervisor:** Prof. Dr. Philippe Cudré-Mauroux [cite: 6]
* [cite_start]**Institution:** Exascale Infolab, University of Fribourg (Switzerland) [cite: 9]
* [cite_start]**Date:** August 2025 [cite: 5]

---
[cite_start]*For more details on methodology, please refer to the full thesis (Chapter 8: Discussion and Recommendations)[cite: 1083, 1154].*

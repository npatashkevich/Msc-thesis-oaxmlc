# Evaluating Advanced Models in Extreme Multi-Label Classification on OpenAlex

[cite_start]This repository contains the code and experimental framework for my Master's Thesis at the **University of Fribourg (2025)**[cite: 3, 8].

## 📌 Project Overview
[cite_start]Extreme Multi-Label Classification (XMLC) is a critical task for academic search engines and scientific knowledge graphs[cite: 15]. [cite_start]This research provides a comprehensive benchmark of specialized XMLC architectures against modern Large Language Models (LLMs) on a massive scholarly dataset[cite: 17, 68].

**Key Contributions:**
* [cite_start]Evaluated specialized models: **AttentionXML**, **MATCH**, and **CascadeXML**[cite: 68, 75].
* [cite_start]Adapted and benchmarked LLMs: **Llama 3.1 8B**, **Mistral v0.3 7B**, and **Qwen 2.5 7B** using Parameter-Efficient Fine-Tuning (**LoRA**)[cite: 18, 76, 653].
* [cite_start]Conducted a tail-aware analysis using propensity-scored metrics to evaluate performance on rare labels[cite: 20, 71, 126].

## 📊 Dataset: OAXMLC
[cite_start]We utilized the **OAXMLC** dataset, a curated corpus derived from the **OpenAlex** snapshot (January 20, 2025)[cite: 19, 434, 435].
* [cite_start]**Domain:** Computer Science publications[cite: 435].
* [cite_start]**Label Space:** 7,498 unique concept labels after hierarchy-aware pruning[cite: 571, 621].
* [cite_start]**Scale:** Two reproducible subsets: **OA1M** (1,000,000 documents) and **OA120K** (label-weighted subset)[cite: 73, 106].
* [cite_start]**Features:** Only titles and abstracts were used for textual representation[cite: 91, 473].

## 🚀 Model Performance Summary
[cite_start]Our results demonstrate that specialized XMLC models remain the superior choice for production-scale academic indexing[cite: 21, 1075].

### Core Metrics (100k Test Split)
| Model | P@5 | nDCG@5 | PS-nDCG@5 (Tail-aware) | Inference Time |
| :--- | :--- | :--- | :--- | :--- |
| **AttentionXML** | **0.901** | **0.924** | 0.728 | ~10 min |
| **MATCH** | 0.740 | 0.889 | 0.609 | ~5 min |
| **CascadeXML** | 0.638 | 0.695 | **0.732** | **< 3 min** |

[cite_start]*Note: Results extracted from the comprehensive evaluation in Chapter 7[cite: 1008, 1010, 1020].*

### Key Insights:
* [cite_start]**AttentionXML** achieved the strongest ranking depth and macro-F1 score[cite: 1214, 1218].
* [cite_start]**MATCH** provided the broadest label exploration with a $BN\_coverage@5$ of 0.827[cite: 1004, 1218].
* [cite_start]**CascadeXML** offered the highest efficiency and superior performance on rare labels at $k=1$[cite: 1001, 1220, 1216].
* [cite_start]**LLMs** (even with LoRA) significantly underperformed specialized models and were costlier to serve[cite: 86, 1223, 1076].

## 🛠 Technical Implementation
* [cite_start]**Environment:** PyTorch 2.2, Transformers 4.41, vLLM 0.4.2[cite: 893, 894, 897].
* [cite_start]**Training:** LoRA adaptation with rank $r=16$ and $\alpha=16$[cite: 937].
* [cite_start]**Hardware:** Experiments were conducted on a cluster using NVIDIA **L40S** and **Tesla V100** GPUs[cite: 885, 886].
* [cite_start]**Inference:** Optimized serving via **vLLM** with constrained decoding to ensure valid concept outputs[cite: 77, 934, 943].

## 📂 Repository Structure
* [cite_start]`data/`: Preprocessing scripts for OpenAlex JSONL files[cite: 554, 900].
* [cite_start]`models/`: Implementation details for specialized architectures and LLM prompt templates[cite: 671, 930].
* [cite_start]`evaluation/`: Formal definitions of Precision@k, nDCG@k, and Propensity-Scored variants [cite: 773-806].
* [cite_start]`results/`: Detailed logs and metrics for each model run[cite: 950, 1007, 1081].

## 🎓 Thesis Information
* [cite_start]**Author:** Natallia Patashkevich[cite: 4].
* [cite_start]**Supervisor:** Prof. Dr. Philippe Cudré-Mauroux[cite: 6].
* [cite_start]**Institution:** Exascale Infolab, University of Fribourg (Switzerland)[cite: 9].
* [cite_start]**Date:** August 2025[cite: 5].

---
[cite_start]*For more details on the methodology and qualitative error analysis, please refer to the full thesis document (Chapter 8: Discussion and Recommendations)[cite: 1083, 1154].*

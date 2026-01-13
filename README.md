# Evaluating Advanced Models in XMLC on OpenAlex

This repository contains the experimental framework and results for my Master's Thesis at the **University of Fribourg (2025)**

## 📌 Research Summary
The project benchmarks specialized Extreme Multi-Label Classification (XMLC) architectures against Large Language Models (LLMs) using the **OpenAlex** scholarly graph

**The Challenge:** Assigning relevant scientific concepts from a massive 7,498-label vocabulary to research papers based solely on their titles and abstracts

## 🚀 Key Results
Specialized XMLC models (AttentionXML, MATCH, CascadeXML) consistently outperform general-purpose LLMs in both ranking accuracy and computational efficiency for production-scale indexing

## 🛠 Technical Stack
* **Models:** AttentionXML, MATCH, CascadeXML, Llama 3.1 8B, Qwen 2.5 7B, Mistral v0.3 7B
* **Libraries:** PyTorch, PEFT (LoRA), vLLM for optimized inference
* **Hardware:** NVIDIA L40S and Tesla V100 GPUs

## 📂 Repository Structure
* `/data`: Preprocessing and cleaning pipelines for OpenAlex data
* `/models`: Implementation details and prompt templates
* `/results`: Comprehensive evaluation logs and performance plots

---
**Author:** Natallia Patashkevich  
**Supervisor:** Prof. Dr. Philippe Cudré-Mauroux  
**Laboratory:** Exascale Infolab, University of Fribourg (Switzerland)

<div align="center">

# FINE-TUNING LANGUAGE MODELS

![last commit](https://img.shields.io/github/last-commit/Abeshith/FineTuning_LanguageModels?color=blue&label=last%20commit&style=flat)
![Python](https://img.shields.io/badge/python-84.2%25-blue&style=flat)
![Languages](https://img.shields.io/badge/languages-4-lightgrey&style=flat)

## Built with the tools and technologies:

![Python](https://img.shields.io/badge/Python-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-yellow?style=flat&logo=huggingface&logoColor=white)
![UnSloth](https://img.shields.io/badge/UnSloth-green?style=flat&logo=lightning&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-orange?style=flat&logo=jupyter&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-green?style=flat&logo=nvidia&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-purple?style=flat&logo=matrix&logoColor=white)
![PEFT](https://img.shields.io/badge/PEFT-blue?style=flat&logo=parameter&logoColor=white)

![TRL](https://img.shields.io/badge/TRL-red?style=flat&logo=reinforcement&logoColor=white)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-lightblue?style=flat&logo=memory&logoColor=white)
![Datasets](https://img.shields.io/badge/🤗%20Datasets-yellow?style=flat&logo=huggingface&logoColor=white)
![Accelerate](https://img.shields.io/badge/🤗%20Accelerate-yellow?style=flat&logo=huggingface&logoColor=white)
![WandB](https://img.shields.io/badge/WandB-black?style=flat&logo=weightsandbiases&logoColor=white)
![DPO](https://img.shields.io/badge/DPO-purple?style=flat&logo=optimization&logoColor=white)
![RLHF](https://img.shields.io/badge/RLHF-red?style=flat&logo=feedback&logoColor=white)
![Quantization](https://img.shields.io/badge/Quantization-lightgrey?style=flat&logo=compress&logoColor=white)

![Fine-Tuning Language Models](https://github.com/Abeshith/FineTuning_LanguageModels/blob/main/banner.png)

</div>


## What is Fine-Tuning?

Fine-tuning is the process of adapting a pre-trained language model to specific tasks or domains by training it on task-specific data. Instead of training from scratch, fine-tuning leverages the knowledge already learned by large models during pre-training, making it computationally efficient and effective.

## Why Fine-Tuning?

1. Training large language models from scratch requires enormous computational resources (millions of dollars). Fine-tuning reduces this cost by 99%.

2. Pre-trained models already understand language patterns. Fine-tuning requires only thousands of examples instead of billions.

3. Fine-tuning takes hours or days instead of months required for pre-training.

4. Adapts general-purpose models to excel at specific tasks like sentiment analysis, code generation, or medical diagnosis.

5. Enables models to understand domain-specific terminology and patterns (legal, medical, technical).

## 🧠 Understanding the Fine-Tuning Process

### 🔄 Full Fine-Tuning
Updates all model parameters during training. Requires high computational resources but achieves best task-specific performance.

### ⚡ Parameter-Efficient Fine-Tuning (PEFT)
Updates only a small subset of parameters with 90% reduction in computational cost while matching full fine-tuning performance.

## 📊 Quantization: Memory Optimization

Quantization reduces model weight precision from 32-bit to lower precision (8-bit, 4-bit), dramatically reducing memory usage while maintaining performance.

### 🎯 Precision Formats
- **FP32**: 4 bytes per parameter (baseline)
- **INT8**: 1 byte per parameter (75% memory reduction)  
- **INT4**: 0.5 bytes per parameter (87.5% memory reduction)

### 📈 Example: 7B Parameter Model
```
FP32: 7B × 4 bytes = 28 GB
INT8: 7B × 1 byte = 7 GB (75% reduction)
INT4: 7B × 0.5 bytes = 3.5 GB (87.5% reduction)
```

## 🎯 LoRA: Low-Rank Adaptation

LoRA decomposes weight updates into low-rank matrices, reducing trainable parameters by 99% while maintaining performance.

### 📐 Mathematical Foundation
```
W' = W + ΔW
ΔW = A × B
```
- **W**: Original pre-trained weights (frozen)
- **A**: Low-rank matrix (d × r)  
- **B**: Low-rank matrix (r × k)
- **r**: Rank (much smaller than d or k)

### 🔢 Rank Selection Guidelines
- **r = 8-16**: Standard choice, good balance
- **r = 32-64**: Complex adaptations
- **r = 128+**: Approaching full fine-tuning

### 📊 Parameter Reduction Example
4096 × 4096 layer with r=16:
```
Original: 4096 × 4096 = 16,777,216 parameters
LoRA: (4096 × 16) + (16 × 4096) = 131,072 parameters
Reduction: 99.2% fewer parameters
```

## 🔧 Adapters: Modular Fine-Tuning

Small neural network modules inserted between transformer layers for task-specific adaptation.

### 🏗️ Architecture
```
Input → Layer Norm → Adapter → Residual Connection → Output
```

### ⚖️ LoRA vs Adapters
| Aspect | LoRA | Adapters |
|--------|------|----------|
| Parameter Count | 0.1-1% | 2-4% |
| Training Speed | Faster | Moderate |
| Modularity | Limited | High |

## 📁 Repository Structure

This repository contains three specialized folders:

### 📁 [FineTuning LanguageModels/](./FineTuning%20LanguageModels/)
Foundational fine-tuning techniques and inference examples for beginners.

### 📁 [FineTuning LargeLanguageModels/](./FineTuning%20LargeLanguageModels/)
Advanced model-specific fine-tuning using UnSloth, LoRA, and quantization techniques.

### 📁 [RLHF Optimization/](./RLHF%20Optimization/)
Cutting-edge human preference alignment using Direct Preference Optimization (DPO).

## 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/Abeshith/FineTuning_LanguageModels.git
cd FineTuning_LanguageModels

# Install dependencies
pip install transformers datasets torch torchvision
pip install unsloth peft bitsandbytes trl accelerate
```



# Vision Model Fine-Tuning 🔍

Fine-tuning vision-language models for mathematical formula recognition and LaTeX generation.

## 📋 Contents

### `Unsloth_QWEN2_VisionModel_FIneTuning.ipynb`
**Qwen2-VL Fine-Tuning for LaTeX OCR**

Fine-tune the Qwen2-VL-7B model to convert mathematical formula images into LaTeX code using Unsloth optimization.

#### 🎯 **Purpose**
- Convert mathematical images to LaTeX representations
- Demonstrate vision-language model fine-tuning
- Optimize multimodal model performance

#### 🔧 **Key Technologies**
- **Model**: Qwen2-VL-7B-Instruct
- **Framework**: Unsloth (2x faster training)
- **Dataset**: LaTeX OCR mathematical formulas
- **Optimization**: LoRA + 4-bit quantization

#### 📊 **Configuration**
```python
# LoRA Parameters
r = 16, lora_alpha = 16, lora_dropout = 0

# Training Setup  
batch_size = 2, learning_rate = 2e-4, max_steps = 30
```

#### 🛠️ **Requirements**
- **GPU**: 8GB+ VRAM
- **Libraries**: unsloth, transformers, torch, datasets

#### 📝 **Dataset Format**
Images of mathematical formulas paired with their LaTeX representations in conversation format.

## 🎓 **Skill Level**: Advanced
Vision-language models, LoRA fine-tuning, mathematical notation

## 🚀 **Getting Started**
1. Install dependencies with pip commands
2. Load Qwen2-VL model with 4-bit quantization  
3. Configure LoRA parameters
4. Prepare vision-text dataset
5. Train with UnslothVisionDataCollator

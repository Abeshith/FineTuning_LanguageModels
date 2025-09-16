# 🧠 Memory Usage Benchmarks & Optimization Guide

## 📋 Complete Memory Comparison Table

### **Training Memory Requirements**

| Model | Parameters | Full FT | LoRA | LoRA + 4bit | LoRA + 8bit | Min GPU |
|-------|------------|---------|------|-------------|-------------|---------|
| **Phi-3 Mini** | 3.8B | 46GB | 8GB | **4GB** ✅ | 6GB | T4 (15GB) |
| **Llama-3-8B** | 8B | 96GB | 16GB | **6GB** ✅ | 12GB | T4 (15GB) |
| **Qwen2.5-7B** | 7.6B | 91GB | 15GB | **6GB** ✅ | 11GB | T4 (15GB) |
| **Qwen2-VL-7B** | 7B | 84GB | 14GB | **8GB** ✅ | 12GB | T4 (15GB) |
| **GPT-OSS-20B** | 20B | 240GB | 40GB | **14GB** ✅ | 28GB | V100 (32GB) |
| **Llama-3-70B** | 70B | 840GB | 140GB | **42GB** | 105GB | A100 (80GB) |

> **✅ Green = Works on Google Colab**  
> **🟡 Yellow = Needs paid Colab Pro**  
> **🔴 Red = Needs enterprise hardware**

### **Inference Memory Requirements**

| Model | FP16 | 8-bit | 4-bit | Quantized Size |
|-------|------|-------|-------|----------------|
| **Phi-3 Mini** | 7.6GB | 3.8GB | **2.1GB** | 📱 Mobile-ready |
| **Llama-3-8B** | 16GB | 8GB | **4.5GB** | 💻 Desktop-ready |  
| **Qwen2.5-7B** | 15GB | 7.5GB | **4.2GB** | 💻 Desktop-ready |
| **Qwen2-VL-7B** | 14GB | 7GB | **4.8GB** | 💻 Desktop-ready |
| **GPT-OSS-20B** | 40GB | 20GB | **11GB** | 🖥️ Workstation |

## ⚙️ Memory Optimization Techniques

### **🔹 Quantization Comparison**

```python
# Memory usage example for Llama-3-8B
model_configs = {
    "fp16": {"memory": "16GB", "quality": "100%", "speed": "1x"},
    "8-bit": {"memory": "8GB", "quality": "99%", "speed": "0.8x"},
    "4-bit": {"memory": "4.5GB", "quality": "95%", "speed": "0.6x"},
}
```

### **🔸 LoRA Rank Impact**

| LoRA Rank | Memory Usage | Training Speed | Model Quality |
|-----------|--------------|----------------|---------------|
| **r=8** | Lowest | Fastest | Good ⭐⭐⭐ |
| **r=16** | Low | Fast | Better ⭐⭐⭐⭐ |
| **r=32** | Medium | Medium | Best ⭐⭐⭐⭐⭐ |
| **r=64** | High | Slow | Diminishing returns |

### **🔺 Batch Size vs Memory**

```python
# Memory scaling with batch size (Llama-3-8B + LoRA + 4bit)
batch_configs = {
    1: "6GB",   # Minimum viable
    2: "8GB",   # Recommended for T4
    4: "12GB",  # Optimal for V100
    8: "20GB",  # Enterprise grade
}
```

## 🛠️ Optimization Strategies

### **Strategy 1: Gradient Accumulation**
```python
# Instead of large batch size, accumulate gradients
SFTConfig(
    per_device_train_batch_size=1,  # Low memory
    gradient_accumulation_steps=8,  # Effective batch = 8
)
# Result: Same performance, 75% less memory
```

### **Strategy 2: Gradient Checkpointing**
```python
# Trade computation for memory
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",  # 40% memory reduction
)
```

### **Strategy 3: Mixed Precision Training**
```python
SFTConfig(
    fp16=True,  # 50% memory reduction
    dataloader_pin_memory=False,  # Additional 5-10% savings
)
```

## 📈 Performance Benchmarks

### **Training Speed Comparison**

| Technique | Time (100 steps) | Memory | Quality |
|-----------|------------------|---------|---------|
| **Full Fine-tuning** | 45 min | 96GB | 100% |
| **LoRA** | 15 min | 16GB | 95% |
| **LoRA + 4bit** | 18 min | 6GB | 92% |
| **LoRA + Unsloth** | **8 min** | **6GB** | **95%** |

> **Unsloth provides 2x speedup with same memory usage**

### **Model Quality vs Efficiency**

```python
# Quality benchmarks on instruction-following (MMLU score)
efficiency_comparison = {
    "Phi-3 Mini + LoRA": {"score": 68.1, "memory": "4GB", "cost": "$"},
    "Llama-3-8B + LoRA": {"score": 79.2, "memory": "6GB", "cost": "$$"},
    "Qwen2.5-7B + LoRA": {"score": 81.5, "memory": "6GB", "cost": "$$"},
    "Qwen2-VL-7B + LoRA": {"score": 77.8, "memory": "8GB", "cost": "$$"},
    "GPT-OSS-20B + LoRA": {"score": 85.3, "memory": "14GB", "cost": "$$$"},
}
```

## 💡 Hardware Recommendations

### **🎯 Budget Setup (<$10/month)**
- **Platform**: Google Colab T4
- **Models**: Phi-3, Llama-3-8B, Qwen2.5-7B
- **Technique**: LoRA + 4-bit quantization
- **Memory**: 4-8GB usage

### **💼 Professional Setup ($50-100/month)**  
- **Platform**: Runpod RTX 4090 / A6000
- **Models**: Any 7B-20B model
- **Technique**: LoRA + 8-bit or full precision
- **Memory**: 24-48GB available

### **🏢 Enterprise Setup ($500+/month)**
- **Platform**: AWS p4d.24xlarge  
- **Models**: 70B+ models
- **Technique**: Full fine-tuning or large LoRA
- **Memory**: 320GB+ available

## 🔧 Memory Monitoring Tools

### **Real-time Memory Tracking**
```python
import torch

def print_gpu_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

# Use during training
print_gpu_usage()  # Check at different stages
```

### **Memory Optimization Checker**
```python
def optimize_memory_settings(model_size_b, available_memory_gb):
    """Recommend optimal settings based on available memory"""
    
    if model_size_b <= 4 and available_memory_gb >= 15:
        return {"quantization": "4-bit", "lora_rank": 16, "batch_size": 2}
    elif model_size_b <= 8 and available_memory_gb >= 15:  
        return {"quantization": "4-bit", "lora_rank": 16, "batch_size": 1}
    elif model_size_b <= 20 and available_memory_gb >= 32:
        return {"quantization": "4-bit", "lora_rank": 32, "batch_size": 1}
    else:
        return {"error": "Insufficient memory for this model"}
```

## ⚡ Quick Memory Fixes

### **❌ CUDA Out of Memory**
Try these in order:
```python
# 1. Reduce batch size
per_device_train_batch_size=1

# 2. Enable gradient checkpointing
use_gradient_checkpointing="unsloth"

# 3. Lower LoRA rank
r=8  # instead of r=16

# 4. Use 4-bit quantization
load_in_4bit=True

# 5. Clear cache
torch.cuda.empty_cache()
```

### **❌ Slow Training**
Optimization checklist:
```python
# 1. Use Unsloth
FastLanguageModel.from_pretrained()

# 2. Enable mixed precision
fp16=True

# 3. Optimize batch size
# Find sweet spot for your GPU

# 4. Pin memory (if enough RAM)
dataloader_pin_memory=True
```

## 📊 Real-World Examples

### **Google Colab T4 (15GB)**
```python
# Optimal configuration for T4
model_config = {
    "model": "unsloth/llama-3-8b-bnb-4bit",
    "load_in_4bit": True,
    "lora_rank": 16,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "expected_memory": "8GB"
}
```

### **RTX 4090 (24GB)**
```python
# High-performance setup
model_config = {
    "model": "unsloth/qwen2.5-7b-instruct",
    "load_in_4bit": False,  # Can use 8-bit
    "lora_rank": 32,
    "batch_size": 4,
    "gradient_accumulation": 2,
    "expected_memory": "18GB"
}
```

### **A100 (80GB)**
```python
# Enterprise configuration
model_config = {
    "model": "meta-llama/Llama-3-70b-hf",
    "load_in_4bit": True,  # Still recommend for 70B
    "lora_rank": 64,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "expected_memory": "45GB"
}
```

---

**Need specific optimization help?** → [Troubleshooting Guide](./troubleshooting.md)
# 🛠️ Complete Troubleshooting Guide

## 🚨 Memory Issues

### **❌ CUDA Out of Memory**

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**🔧 Solutions (Try in order):**

**1. Reduce Batch Size**
```python
SFTConfig(
    per_device_train_batch_size=1,  # Start with 1
    gradient_accumulation_steps=4,  # Maintain effective batch size
)
```

**2. Enable 4-bit Quantization**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="your_model",
    load_in_4bit=True,  # 75% memory reduction
)
```

**3. Lower LoRA Rank**
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # Down from 16 or 32
)
```

**4. Enable Gradient Checkpointing**
```python
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",  # 40% memory savings
)
```

**5. Clear GPU Cache**
```python
import torch
torch.cuda.empty_cache()
import gc; gc.collect()
```

### **❌ Slow Memory Leak During Training**

**Symptoms**: Memory usage gradually increases

**🔧 Solutions:**
```python
# Add to training config
SFTConfig(
    dataloader_pin_memory=False,  # Reduce memory pressure
    dataloader_num_workers=0,     # Avoid multiprocessing issues
    save_steps=50,                # Regular cleanup
)

# Manual cleanup every N steps
if step % 50 == 0:
    torch.cuda.empty_cache()
```

## ⚠️ Installation Issues

### **❌ Unsloth Installation Fails**

**Error Message:**
```
ERROR: Could not build wheels for unsloth
```

**🔧 Solutions:**

**1. Use Colab-Specific Installation**
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**2. Install Dependencies First**
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers accelerate peft bitsandbytes
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**3. For Local Installation**
```bash
# Check CUDA version first
nvcc --version

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

### **❌ Version Conflicts**

**Error Message:**
```
AttributeError: 'TrainingArguments' object has no attribute 'padding_value'
```

**🔧 Solution:**
```python
# Use exact compatible versions
!pip install transformers==4.45.2
!pip install trl==0.11.4
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## 🏃‍♂️ Performance Issues

### **❌ Very Slow Training**

**Symptoms**: Training takes much longer than expected

**🔧 Optimizations:**

**1. Use Unsloth Optimizations**
```python
from unsloth import FastLanguageModel  # 2x speedup
model, tokenizer = FastLanguageModel.from_pretrained(...)
```

**2. Enable Mixed Precision**
```python
SFTConfig(
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
)
```

**3. Optimize Batch Size**
```python
# Find optimal batch size for your GPU
batch_sizes_to_try = [1, 2, 4, 8]
for batch_size in batch_sizes_to_try:
    try:
        # Test training step
        break
    except RuntimeError:
        continue
```

**4. Disable Unnecessary Features**
```python
SFTConfig(
    logging_steps=50,            # Less frequent logging
    save_strategy="no",          # No checkpointing during training
    evaluation_strategy="no",    # No evaluation during training
)
```

### **❌ Training Diverges (Loss Increases)**

**Symptoms**: Training loss keeps increasing

**🔧 Solutions:**

**1. Lower Learning Rate**
```python
SFTConfig(
    learning_rate=1e-5,  # Down from 2e-4
)
```

**2. Add Gradient Clipping**  
```python
SFTConfig(
    max_grad_norm=1.0,  # Prevent gradient explosion
)
```

**3. Check Data Quality**
```python
# Inspect your dataset
for example in dataset.take(5):
    print(f"Input length: {len(example['text'])}")
    print(f"Sample: {example['text'][:200]}...")
```

## 🗣️ Model Output Issues

### **❌ Model Generates Nonsense**

**Symptoms**: Gibberish or incoherent responses

**🔧 Debugging Steps:**

**1. Check Tokenizer Setup**
```python
# Ensure proper tokenizer configuration
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**2. Verify Data Formatting**
```python
# Check if chat template is applied correctly
sample = dataset[0]
print("Formatted text:")
print(sample['text'])
```

**3. Test Base Model First**
```python
# Before fine-tuning, test the base model
base_model, tokenizer = FastLanguageModel.from_pretrained("base_model")
FastLanguageModel.for_inference(base_model)

# Generate and check output quality
```

**4. Reduce Learning Rate**
```python
SFTConfig(
    learning_rate=5e-5,  # Much lower
    warmup_ratio=0.1,    # Gradual warmup
)
```

### **❌ Model Doesn't Follow Instructions**

**Symptoms**: Model ignores prompts or gives generic responses

**🔧 Solutions:**

**1. Use DPO Training**
```python
# After SFT, add DPO training for better instruction following
from trl import DPOTrainer

# Add preference data and DPO training
```

**2. Improve Dataset Quality**
```python
# Ensure clear instruction-response format
good_format = {
    "instruction": "Explain photosynthesis",
    "output": "Photosynthesis is the process..."
}
```

**3. Use Better Chat Template**
```python
# Apply proper chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": response}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

## 📊 Data Issues

### **❌ Dataset Loading Fails**

**Error Message:**
```
DatasetGenerationError: An error occurred while generating the dataset
```

**🔧 Solutions:**

**1. Check Dataset Format**
```python
# Verify dataset exists and is accessible
from datasets import load_dataset
dataset = load_dataset("dataset_name", split="train")
print(dataset.features)
```

**2. Handle Missing Columns**
```python
# Check required columns exist
required_columns = ['instruction', 'output']
available_columns = dataset.column_names
missing = [col for col in required_columns if col not in available_columns]
if missing:
    print(f"Missing columns: {missing}")
```

**3. Use Local Dataset**
```python
# If online loading fails, download manually
dataset = load_dataset("json", data_files="your_data.jsonl")
```

### **❌ Poor Training Results**

**Symptoms**: Model performs badly after training

**🔧 Diagnostic Steps:**

**1. Check Dataset Size**
```python
print(f"Dataset size: {len(dataset)}")
# Need at least 100-200 examples for meaningful results
```

**2. Inspect Data Quality**
```python
# Look for patterns in your data
lengths = [len(example['text']) for example in dataset]
print(f"Average length: {sum(lengths)/len(lengths)}")
print(f"Max length: {max(lengths)}")

# Check for duplicates
texts = [example['text'] for example in dataset]
unique_texts = set(texts)
print(f"Unique examples: {len(unique_texts)} / {len(texts)}")
```

**3. Validate Formatting**
```python
# Check if examples follow expected format
for i, example in enumerate(dataset.take(3)):
    print(f"Example {i}:")
    print(example['text'])
    print("-" * 50)
```

## 🔧 Environment Issues

### **❌ Colab Disconnections**

**Symptoms**: Training interrupted by Colab timeouts

**🔧 Prevention:**

**1. Enable Automatic Saving**
```python
SFTConfig(
    save_strategy="steps",
    save_steps=100,
    output_dir="checkpoints",
)
```

**2. Use Shorter Training Runs**
```python
SFTConfig(
    max_steps=200,  # Instead of epochs
)
```

**3. Activity Script**
```javascript
// Run in browser console to prevent idle timeout
function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
```

### **❌ Import Errors**

**Error Message:**
```
ModuleNotFoundError: No module named 'unsloth'
```

**🔧 Solutions:**

**1. Restart Runtime**
```
# After installation, restart runtime
Runtime > Restart Runtime
```

**2. Check Installation**  
```python
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
if 'unsloth' in result.stdout:
    print("✅ Unsloth installed")
else:
    print("❌ Unsloth not found")
```

**3. Reinstall Clean**
```python
!pip uninstall unsloth -y
!pip cache purge
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## 🧪 Testing & Validation

### **🔍 Quick Model Test**

```python
def quick_model_test(model, tokenizer):
    """Test model with simple prompts"""
    
    test_prompts = [
        "Hello, how are you?",
        "Explain machine learning in one sentence.",
        "Write a haiku about programming.",
    ]
    
    FastLanguageModel.for_inference(model)
    
    for prompt in test_prompts:
        inputs = tokenizer([prompt], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Q: {prompt}")
        print(f"A: {response[len(prompt):]}")
        print("-" * 40)
```

### **📊 Performance Benchmarking**

```python
import time

def benchmark_model(model, tokenizer, num_tests=10):
    """Benchmark inference speed"""
    
    prompt = "Explain artificial intelligence."
    inputs = tokenizer([prompt], return_tensors="pt")
    
    # Warmup
    model.generate(**inputs, max_new_tokens=50)
    
    # Benchmark
    times = []
    for _ in range(num_tests):
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=50)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.2f}s")
    print(f"Tokens per second: {50/avg_time:.1f}")
```

## 🔍 Vision Model Specific Issues

### **❌ Vision Model Loading Fails**

**Error Message:**
```
AttributeError: 'FastVisionModel' object has no attribute 'from_pretrained'
```

**🔧 Solutions:**

**1. Use Correct Import**
```python
from unsloth import FastVisionModel  # Not FastLanguageModel
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit=True,
)
```

**2. Check Image Format**
```python
# Ensure images are in correct format
from PIL import Image
image = Image.open("path/to/image.jpg")
# Image should be PIL.Image object
```

### **❌ Vision Training Memory Issues**

**Symptoms**: Higher memory usage than text-only models

**🔧 Solutions:**
```python
# Use more aggressive memory optimization
model = FastVisionModel.get_peft_model(
    model,
    r=8,  # Lower rank for vision models
    finetune_vision_layers=True,
    finetune_language_layers=False,  # Only vision layers
)

# Smaller batch sizes
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
```

## 📞 Getting Help

### **🔍 Before Asking for Help**

1. **Check Error Messages**: Copy the full error traceback
2. **Verify Installation**: Ensure all packages are correctly installed  
3. **Test Minimal Example**: Try the simplest possible case first
4. **Check Resources**: Monitor GPU memory and usage

### **📝 How to Report Issues**

Include this information when asking for help:

```python
# System information
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Package versions
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
for package in ['unsloth', 'transformers', 'trl', 'peft']:
    if package in result.stdout:
        lines = [line for line in result.stdout.split('\n') if package in line]
        print(lines[0] if lines else f"{package}: not found")
```

### **🆘 Support Channels**

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Abeshith/FineTuning_LanguageModels/issues)
- 💬 **General Questions**: [GitHub Discussions](https://github.com/Abeshith/FineTuning_LanguageModels/discussions)
- 📚 **Documentation**: Check the [docs/](../) folder
- 💭 **Community**: Join our Discord (link in repo)

### **🎯 Quick Fixes Reference**

| Problem | Quick Fix |
|---------|-----------|
| CUDA OOM | `per_device_train_batch_size=1` |
| Slow training | Use Unsloth + `fp16=True` |
| Bad outputs | Check tokenizer + reduce learning rate |
| Import errors | Restart runtime after installation |
| Version conflicts | Use exact version pins |
| Memory leak | `dataloader_pin_memory=False` |
| Training divergence | `learning_rate=1e-5` + `max_grad_norm=1.0` |

---

**Still stuck?** → [Open an issue](https://github.com/Abeshith/FineTuning_LanguageModels/issues/new) with the system information above and detailed error description.
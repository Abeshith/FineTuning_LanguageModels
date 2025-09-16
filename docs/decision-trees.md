# 🌳 Decision Trees: When to Use What Technique

## 🎯 Main Decision Tree

```
I want to fine-tune an LLM
├── What's your experience level?
│   ├── Beginner → Start with Basic LoRA
│   ├── Intermediate → What's your goal?
│   │   ├── Better conversations → Try DPO Training
│   │   ├── Domain expertise → Use task-specific datasets
│   │   ├── Multiple languages → Use Qwen2.5
│   │   └── Vision + text → Use Qwen2-VL
│   └── Advanced → What's your use case?
│       ├── Research → Full fine-tuning or high-rank LoRA
│       ├── Production → DPO + evaluation pipeline
│       └── Experimentation → Quick LoRA experiments
```

## 🔍 Model Selection Decision Tree

### **Step 1: What's Your Hardware?**

**📱 Google Colab T4 (15GB) or similar**
- ✅ Phi-3 Mini (3.8B) - Best efficiency
- ✅ Llama-3-8B - Best general performance
- ✅ Qwen2.5-7B - Best multilingual
- ✅ Qwen2-VL-7B - Vision-language tasks  
- ❌ Models >20B - Won't fit

**💻 Mid-range GPU (24GB)**
- ✅ All 7B-8B models
- ✅ GPT-OSS-20B - Advanced reasoning
- ⚠️ 30B models (tight fit)
- ❌ 70B+ models

**🖥️ High-end GPU (40GB+)**
- ✅ All models up to 70B
- ✅ Full precision training
- ✅ Large batch sizes

### **Step 2: What's Your Task?**

**📝 Text Generation & Creative Writing**
- Best: Llama-3-8B-Instruct
- Alternative: GPT-OSS-20B
- Budget: Phi-3 Mini

**💻 Code Generation & Programming**
- Best: Phi-3 Mini (specialized for coding)
- Alternative: CodeLlama-7B
- Advanced: GPT-OSS-20B

**🌍 Multilingual Tasks**
- Best: Qwen2.5-7B (29+ languages)
- Alternative: Llama-3-8B
- Specialized: mT5 variants

**🧮 Mathematical Reasoning**
- Best: Qwen2.5-7B (math-optimized)
- Alternative: GPT-OSS-20B
- Specialized: MathCodeT5

**🖼️ Vision-Language Tasks**
- Best: Qwen2-VL-7B (image + text)
- Alternative: LLaVA variants
- Specialized: GPT-4V fine-tuned

**💬 Conversational AI**
- Best: Llama-3-8B-Instruct + DPO
- Alternative: Qwen2.5-7B
- Budget: Phi-3 Mini

## ⚙️ Technique Selection Decision Tree

### **Fine-Tuning Method Decision**

```python
def choose_finetuning_method(memory_gb, time_budget, quality_need):
    if memory_gb < 16:
        return "LoRA + 4-bit quantization"
    elif time_budget == "fast" and quality_need == "good":
        return "LoRA + Unsloth"
    elif quality_need == "highest":
        return "Full fine-tuning"
    else:
        return "LoRA + 8-bit"
```

### **Training Configuration Decision Tree**

**🎯 Your Priority?**
- **💾 Memory Efficiency**
  - Use 4-bit quantization
  - LoRA rank: 8-16
  - Batch size: 1
  - Gradient accumulation: 4-8
- **⚡ Training Speed**
  - Use Unsloth optimizations
  - Mixed precision (fp16/bf16)
  - Higher batch size
  - Gradient checkpointing: False
- **🎯 Model Quality**
  - Higher LoRA rank: 32-64
  - More training epochs: 3-5
  - Lower learning rate: 1e-4
  - Larger dataset
- **💰 Cost Efficiency**
  - Use Google Colab
  - Phi-3 Mini model
  - Short training runs
  - 4-bit quantization

## 📊 Dataset Size Decision Tree

**📚 How much data do you have?**

**< 100 examples**
- ⚠️ Too small for good results
- 💡 Use few-shot prompting instead
- 🔄 Or augment with synthetic data

**100 - 1,000 examples**
- ✅ Perfect for LoRA fine-tuning
- 📝 Focus on high-quality curation
- ⏱️ Training time: 10-30 minutes

**1,000 - 10,000 examples**
- ✅ Excellent for most use cases
- 🎯 Can use higher LoRA ranks
- ⏱️ Training time: 30 minutes - 2 hours

**10,000+ examples**
- ✅ Great for specialized domains
- 🔄 Consider full fine-tuning
- 📊 Split into train/validation sets
- ⏱️ Training time: 2+ hours

## 🎨 Use Case Specific Recommendations

### **🎓 Educational Applications**

**Math Tutoring**
- Model: Qwen2.5-7B (math-specialized)
- Method: LoRA + mathematical datasets
- Training: 1,000 problem-solution pairs
- Post-processing: DPO for explanation quality

**Language Learning**
- Model: Qwen2.5-7B (multilingual)
- Method: LoRA + conversation datasets
- Training: Native speaker dialogues
- Evaluation: Fluency + grammar checks

**Code Teaching**
- Model: Phi-3 Mini (code-optimized)
- Method: LoRA + coding instruction datasets
- Training: Code explanation pairs
- Testing: Code generation accuracy

### **💼 Business Applications**

**Customer Service**
- Model: Llama-3-8B + DPO training
- Dataset: Historical support tickets
- Training: Helpful vs unhelpful responses
- Deployment: API with safety filters

**Content Generation**
- Model: GPT-OSS-20B (creativity)
- Method: LoRA + brand voice data
- Training: Company writing samples
- Quality: Human review process

**Data Analysis**
- Model: Qwen2.5-7B (structured data)
- Method: LoRA + analysis examples
- Training: Question-insight pairs
- Output: JSON formatted results

### **🔬 Research Applications**

**Vision-Language Research**
- Model: Qwen2-VL-7B
- Method: LoRA + multimodal datasets
- Training: Image-text pairs
- Evaluation: Multimodal benchmarks

**Scientific Literature**
- Model: GPT-OSS-20B (reasoning)
- Method: Full fine-tuning or high-rank LoRA
- Training: Domain-specific papers
- Output: Research insights

## 🔄 Iterative Improvement Decision Tree

**🎯 After Initial Training**

**Model performance is...**
- **😞 Poor (< 60% accuracy)**
  - 🔍 Check data quality
  - 📊 Increase dataset size
  - ⚙️ Adjust learning rate
  - 🔄 Try different model
- **😐 Okay (60-80% accuracy)**
  - 📈 Increase LoRA rank
  - 🎯 Add more training epochs
  - 🔧 Try DPO training
  - 📚 Improve dataset quality
- **😊 Good (80%+ accuracy)**
  - 🚀 Ready for deployment!
  - 📊 Set up evaluation pipeline
  - 🔄 Consider model distillation
  - 📈 Monitor production performance

## 🛠️ Quick Decision Flowcharts

### **⚡ 30-Second Model Choice**

**I need a model for...**
- 📱 Mobile deployment → Phi-3 Mini
- 💬 Chatbot → Llama-3-8B + DPO
- 🌍 Multiple languages → Qwen2.5-7B
- 🧮 Math problems → Qwen2.5-7B
- 💻 Code tasks → Phi-3 Mini
- 🖼️ Vision + text → Qwen2-VL-7B
- 🔬 Research → GPT-OSS-20B

### **⚡ 30-Second Technique Choice**

**I want...**
- 💾 Lowest memory → LoRA + 4-bit
- ⚡ Fastest training → LoRA + Unsloth
- 🎯 Best quality → Full fine-tuning
- 💰 Cheapest → LoRA + Colab T4
- 🤖 Better responses → DPO training
- 🖼️ Vision capabilities → Vision model + LoRA

## 📋 Decision Checklist

Before starting fine-tuning, check:

- [ ] **Hardware Requirements**: Can my GPU handle the model?
- [ ] **Dataset Quality**: Do I have clean, relevant data?  
- [ ] **Time Budget**: How long can I train?
- [ ] **Quality Expectations**: What accuracy do I need?
- [ ] **Deployment Target**: Where will this model run?
- [ ] **Budget Constraints**: What can I afford to spend?

## 🎯 Common Decision Scenarios

### **Scenario 1: "I'm a student with limited budget"**
✅ **Solution:**
- Platform: Google Colab (free)
- Model: Phi-3 Mini
- Method: LoRA + 4-bit quantization
- Dataset: Small, curated dataset
- Time: Quick experiments

### **Scenario 2: "I need production-ready chatbot"** 
✅ **Solution:**
- Platform: Cloud GPU (Runpod/Lambda)
- Model: Llama-3-8B-Instruct
- Method: SFT + DPO training
- Dataset: Conversational data + preferences
- Evaluation: Human feedback loop

### **Scenario 3: "I want to experiment with techniques"**
✅ **Solution:**
- Platform: Mixed (Colab + cloud)
- Models: Multiple small models
- Method: Rapid LoRA experiments
- Dataset: Standardized benchmarks
- Focus: Technique comparison

### **Scenario 4: "I need vision-language capabilities"**
✅ **Solution:**
- Platform: Google Colab T4 or better
- Model: Qwen2-VL-7B
- Method: LoRA + 4-bit quantization
- Dataset: Image-text pairs
- Use case: OCR, VQA, image understanding

## 📊 Decision Matrix

| Priority | Memory | Speed | Quality | Cost | Recommended Setup |
|----------|--------|--------|---------|------|-------------------|
| **Learning** | Low | Medium | Medium | Low | Phi-3 + LoRA + Colab |
| **Research** | High | Low | High | High | GPT-OSS-20B + Full FT |
| **Production** | Medium | High | High | Medium | Llama-3-8B + DPO |
| **Experimentation** | Low | High | Medium | Low | Multiple small models |
| **Vision Tasks** | Medium | Medium | High | Medium | Qwen2-VL + LoRA |

---

**Still unsure?** → [Ask in Discussions](https://github.com/Abeshith/FineTuning_LanguageModels/discussions) or check [Troubleshooting Guide](./troubleshooting.md)
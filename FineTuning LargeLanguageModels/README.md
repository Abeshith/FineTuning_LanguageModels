# FineTuning LargeLanguageModels

This folder contains specialized notebooks for fine-tuning specific large language models using advanced frameworks and optimization techniques.

## Files in this Folder

### `Gemma_UnSloth_Finetuning.ipynb`

**Purpose**: Fine-tuning Google's Gemma model using the UnSloth framework for maximum efficiency

**Key Features**:
- 2x faster training with UnSloth optimization
- Efficient memory usage with 4-bit quantization
- LoRA (Low-Rank Adaptation) for parameter-efficient training
- Custom instruction dataset integration
- Chat template formatting for conversational AI

**Model**: Google Gemma (lightweight language model)
**Dataset**: Custom instruction dataset for conversational patterns
**Performance**: Significant speed and memory improvements over standard training

---

### `Infernecing_LLM.ipynb`

**Purpose**: Comprehensive inference pipeline specifically designed for large language models

**Key Features**:
- Advanced text generation techniques
- Performance optimization for inference
- Quality assessment frameworks
- Multiple sampling strategies comparison
- Parameter tuning for optimal outputs

**Techniques**: Temperature control, top-k/top-p sampling, repetition penalty optimization
**Output**: High-quality text generation with performance benchmarks

---

### `Instruction_FineTuning_UnSloth.ipynb`

**Purpose**: Training models to follow specific instructions using UnSloth framework

**Key Features**:
- Instruction-response pair training
- Command following optimization
- Conversational pattern learning
- Memory-efficient training with LoRA
- Specialized instruction dataset handling

**Model**: LLaMA 3.2 3B Instruct
**Focus**: Creating models that excel at following complex instructions and commands

---

### `LLM_FineTuning_CustomData.ipynb`

**Purpose**: Complete workflow for fine-tuning large language models on custom datasets

**Key Features**:
- Custom data preprocessing pipelines
- Domain-specific dataset integration
- Training configuration optimization
- Model evaluation on custom metrics
- Data formatting and tokenization strategies

**Use Cases**: Domain adaptation for medical, legal, technical, or specialized fields
**Output**: Models adapted to specific domains or use cases

---

### `LLM_Finetuning_Transformers.ipynb`

**Purpose**: Fine-tuning using the standard Hugging Face Transformers library

**Key Features**:
- Mistral architecture implementation
- 4-bit quantization with BitsAndBytes
- LoRA adaptation techniques
- Memory-efficient training strategies
- Comprehensive progress tracking

**Model**: Mistral-7B with quantization optimizations
**Libraries**: Transformers, BitsAndBytes, PEFT
**Focus**: Standard transformers workflow with efficiency optimizations

---

### `Mistral_UnSloth_FineTuning.ipynb`

**Purpose**: Fine-tuning Mistral-7B model using UnSloth for sentiment analysis

**Key Features**:
- IMDB dataset for sentiment classification
- Text classification training pipeline
- UnSloth 2x speed optimization
- Sentiment analysis capabilities
- Efficient tokenization and training

**Dataset**: IMDB movie reviews for sentiment analysis
**Application**: Text classification and sentiment analysis tasks
**Performance**: Improved training speed with maintained accuracy

---

### `Unsloth_LLama_3_2_3B_FineTuning.ipynb`

**Purpose**: Fine-tuning LLaMA 3.2 3B model using UnSloth with chat optimization

**Key Features**:
- ShareGPT format data handling
- Chat template standardization
- Conversational AI optimization
- Gradient checkpointing for memory efficiency
- LoRA configuration for efficient training

**Model**: LLaMA 3.2 3B (Meta's efficient language model)
**Dataset**: Conversational data in ShareGPT format
**Output**: Optimized conversational AI with improved dialogue capabilities

## Getting Started

1. Choose the notebook that matches your specific use case
2. Install required dependencies (see main repository README)
3. Follow the step-by-step instructions in each notebook
4. Customize parameters based on your dataset and requirements

## Performance Benefits

- **UnSloth Framework**: Up to 2x faster training
- **LoRA Techniques**: 90% reduction in trainable parameters
- **4-bit Quantization**: 75% memory usage reduction
- **Gradient Checkpointing**: Additional memory savings during training

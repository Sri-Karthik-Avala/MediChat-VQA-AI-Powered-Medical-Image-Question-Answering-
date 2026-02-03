# MediChat-VQA: AI-Powered Medical Image Question Answering

A comprehensive Visual Question Answering (VQA) system for medical imaging that fine-tunes state-of-the-art Vision-Language Models (VLMs) including **Llama 3.2 Vision**, **Qwen2-VL**, and **LLaMA-3** for answering diagnostic questions about medical images.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Models](#models)
- [Datasets](#datasets)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Future Scope](#future-scope)
- [Citation](#citation)
- [License](#license)

## Overview

Medical Visual Question Answering (Med-VQA) is a challenging task that requires understanding both visual medical imaging data and natural language questions. This project implements efficient fine-tuning of large vision-language models using **LoRA (Low-Rank Adaptation)** and **4-bit quantization** via **Unsloth** to make training feasible on consumer-grade GPUs.

### The Challenge

Healthcare professionals spend significant time analyzing medical images and formulating diagnoses. Automated systems that can answer questions about medical images have the potential to:
- Assist radiologists in preliminary screening
- Provide educational tools for medical students
- Enable telemedicine applications in resource-limited settings
- Reduce diagnostic turnaround time

## Key Features

- **Multi-Model Support**: Fine-tunes Llama 3.2-11B Vision, Qwen2-VL-2B/7B, and LLaMA-3-8B
- **Efficient Training**: Uses 4-bit quantization and LoRA for memory-efficient fine-tuning
- **Multiple Datasets**: Supports VQA-RAD and MEDPIX-ShortQA medical imaging datasets
- **Comprehensive Evaluation**: Includes BERTScore, BLEU, and accuracy metrics
- **GPU Optimized**: Runs on Tesla T4 (16GB) to A100 (80GB) GPUs
- **HuggingFace Integration**: Models are uploaded to HuggingFace Hub for easy deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MEDICAL IMAGE INPUT                              │
│                    (X-Ray, CT, MRI, Ultrasound)                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Vision Encoder         │
                    │  (ViT / Qwen2-VL Vision)  │
                    └─────────────┬─────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                         QUESTION INPUT                                   │
│              "Is there evidence of pneumonia?"                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Text Tokenizer         │
                    │    + Embedding Layer      │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Llama 3.2-11B  │   │   Qwen2-VL-7B   │   │   Qwen2-VL-2B   │
│     Vision      │   │    Instruct     │   │    Instruct     │
│   (4-bit LoRA)  │   │   (4-bit LoRA)  │   │   (4-bit LoRA)  │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Answer Generation  │
                    │  "Yes, bilateral    │
                    │   infiltrates seen" │
                    └─────────────────────┘
```

## Models

### 1. Llama 3.2-11B Vision (Medical VQA Fine-tuned)
- **Base Model**: `unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit`
- **Parameters**: 11 billion (4-bit quantized)
- **LoRA Rank**: 16
- **Trainable Parameters**: 67.17M
- **Fine-tuned Model**: [`titanhacker/Llama-3.2-11B-Vision-Med-Vqa-Finetuned`](https://huggingface.co/titanhacker/Llama-3.2-11B-Vision-Med-Vqa-Finetuned)

### 2. Qwen2-VL-7B Instruct
- **Base Model**: `unsloth/Qwen2-VL-7B-Instruct-bnb-4bit`
- **Parameters**: 7 billion (4-bit quantized)
- **LoRA Rank**: 16
- **Trainable Parameters**: 50.86M
- **GPU**: NVIDIA A100 80GB PCIe

### 3. Qwen2-VL-2B Instruct
- **Base Model**: `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`
- **Parameters**: 2 billion (4-bit quantized)
- **LoRA Rank**: 16
- **GPU**: Tesla T4 (16GB)

### 4. LLaMA-3-8B (Text Classification)
- **Base Model**: `unsloth/llama-3-8b-bnb-4bit`
- **Task**: Binary text classification with trimmed classification head
- **Trainable Parameters**: 41.95M
- **Custom Features**: Only Yes/No tokens in LM head, last-token loss

## Datasets

### VQA-RAD
- **Source**: [`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- **Domain**: Radiology images (CT, MRI, X-Ray)
- **Train Split**: 1,793 samples
- **Test Split**: 451 samples
- **Question Types**:
  - Closed-ended (Yes/No): ~50%
  - Open-ended (descriptive): ~50%

### MEDPIX-ShortQA
- **Source**: [`adishourya/MEDPIX-ShortQA`](https://huggingface.co/datasets/adishourya/MEDPIX-ShortQA)
- **Domain**: Multi-modality medical images
- **Train Split**: 11,574 samples
- **Test Split**: 980 samples
- **Validation Split**: Available
- **Question Types**: Diagnosis, anatomy, modality identification

## Results

### Llama 3.2-11B Vision on VQA-RAD

| Metric | Before Fine-tuning | After Fine-tuning |
|--------|-------------------|-------------------|
| **BERT Precision** | - | 0.9533 |
| **BERT Recall** | - | 0.9465 |
| **BERT F1 Score** | - | **0.9495** |
| **Overall Accuracy** | - | 42.35% |
| **Closed-ended Accuracy** | - | 59.36% |
| **Open-ended Accuracy** | - | 21.00% |
| **Average BLEU Score** | - | 0.4670 |

### Qwen2-VL-2B on VQA-RAD

| Metric | After Fine-tuning |
|--------|-------------------|
| **BERT Precision** | 0.8738 |
| **BERT Recall** | 0.8724 |
| **BERT F1 Score** | **0.8727** |

### Qwen2-VL-7B on MEDPIX-ShortQA

| Metric | After Fine-tuning |
|--------|-------------------|
| **BERT Precision** | 0.8738 |
| **BERT Recall** | 0.8724 |
| **BERT F1 Score** | **0.8727** |
| **Training Time** | 135.55 minutes |

### LLaMA-3-8B Text Classification

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **96.8%** |
| **Training Time** | 1.83 minutes |
| **Peak Memory** | 5.67 GB |

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ or CUDA 12.1
- GPU with 16GB+ VRAM (Tesla T4 minimum, A100 recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Sri-Karthik-Avala/MediChat-VQA-AI-Powered-Medical-Image-Question-Answering-.git
cd MediChat-VQA-AI-Powered-Medical-Image-Question-Answering-
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install unsloth
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers accelerate --upgrade
pip install datasets trl bert_score scikit-learn nltk wandb
```

4. **Login to HuggingFace (for model upload)**
```bash
huggingface-cli login
```

## Usage

### Fine-tuning Llama 3.2 Vision on VQA-RAD

```python
from unsloth import FastVisionModel
from datasets import load_dataset

# Load model with 4-bit quantization
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
    use_gradient_checkpointing="unsloth",
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
)

# Load dataset
train_dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
```

### Inference with Fine-tuned Model

```python
from unsloth import FastVisionModel
from PIL import Image

# Load fine-tuned model
model, tokenizer = FastVisionModel.from_pretrained(
    "titanhacker/Llama-3.2-11B-Vision-Med-Vqa-Finetuned"
)

# Prepare inference
FastVisionModel.for_inference(model)

# Load and process image
image = Image.open("chest_xray.jpg")
question = "Is there evidence of pneumonia in this chest X-ray?"

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image, input_text, return_tensors="pt").to("cuda")

# Generate answer
output = model.generate(**inputs, max_new_tokens=128, temperature=0.5)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

## Project Structure

```
MediChat-VQA/
├── llama3-2-11b-medvqa (1).ipynb      # Llama 3.2 Vision fine-tuning on VQA-RAD
├── qwen2-5-medvqa (1).ipynb           # Qwen2-VL-2B fine-tuning on VQA-RAD
├── qwen2-vl-fine-tuning-unsloth.ipynb # Qwen2-VL-7B fine-tuning on MEDPIX
├── unsloth_classification (1).ipynb   # LLaMA-3 text classification
└── README.md                          # This file
```

### Notebook Descriptions

| Notebook | Model | Dataset | GPU | Purpose |
|----------|-------|---------|-----|---------|
| `llama3-2-11b-medvqa` | Llama 3.2-11B Vision | VQA-RAD | Tesla T4 | Medical VQA |
| `qwen2-5-medvqa` | Qwen2-VL-2B | VQA-RAD | Tesla T4 | Medical VQA |
| `qwen2-vl-fine-tuning` | Qwen2-VL-7B | MEDPIX-ShortQA | A100 80GB | Medical VQA |
| `unsloth_classification` | LLaMA-3-8B | Finance Sentiment | RTX 3090 | Text Classification |

## Technical Details

### Training Configuration

| Parameter | Llama 3.2 Vision | Qwen2-VL-7B | Qwen2-VL-2B |
|-----------|------------------|-------------|-------------|
| Batch Size | 2 | 4 | 2 |
| Gradient Accumulation | 4 | 8 | 4 |
| Learning Rate | 2e-4 | 2e-4 | 2e-4 |
| Warmup Steps | 5 | 5 | 5 |
| Epochs | 1 | 2 | 2 |
| Optimizer | AdamW 8-bit | AdamW 8-bit | AdamW 8-bit |
| LR Scheduler | Linear | Linear | Linear |
| Max Sequence Length | 2048 | 2048 | 2048 |

### Memory Usage

| Model | GPU | Peak Memory | Training Memory |
|-------|-----|-------------|-----------------|
| Llama 3.2-11B | Tesla T4 | 14.57 GB | 1.47 GB |
| Qwen2-VL-7B | A100 80GB | 36.11 GB | 29.96 GB |
| Qwen2-VL-2B | Tesla T4 | ~12 GB | ~4 GB |

### Evaluation Metrics

- **BERTScore**: Semantic similarity between predictions and ground truth
- **BLEU Score**: N-gram overlap measurement
- **Accuracy**:
  - Overall accuracy
  - Closed-ended (Yes/No) accuracy
  - Open-ended accuracy

## Technologies Used

### Deep Learning Frameworks
- **PyTorch**: Core deep learning framework
- **Transformers**: HuggingFace model library
- **Unsloth**: 2x faster fine-tuning with 4-bit quantization
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA)
- **TRL**: Transformer Reinforcement Learning for SFT

### Vision-Language Models
- **Llama 3.2 Vision**: Meta's multimodal LLM
- **Qwen2-VL**: Alibaba's vision-language model
- **DINOv2**: Self-supervised vision transformer

### Utilities
- **Weights & Biases**: Experiment tracking
- **HuggingFace Hub**: Model hosting and sharing
- **xFormers**: Memory-efficient attention

## Future Scope

1. **Multi-Modal Ensemble**: Combine multiple VLMs for improved accuracy
2. **Domain-Specific Pre-training**: Pre-train on large medical image corpora
3. **Clinical Integration**: Deploy in PACS/RIS systems
4. **Multilingual Support**: Extend to non-English medical literature
5. **Explainability**: Add attention visualization and saliency maps
6. **Mobile Deployment**: Optimize for edge devices using quantization
7. **Active Learning**: Implement feedback loops with radiologists

## Citation

If you use this work in your research, please cite:

```bibtex
@software{avala2024medichatvqa,
  title={MediChat-VQA: AI-Powered Medical Image Question Answering},
  author={Avala, Sri Karthik},
  year={2024},
  url={https://github.com/Sri-Karthik-Avala/MediChat-VQA-AI-Powered-Medical-Image-Question-Answering-}
}
```

## Acknowledgments

- **Unsloth AI** - For efficient fine-tuning framework
- **Meta AI** - For Llama 3.2 Vision model
- **Alibaba Cloud** - For Qwen2-VL models
- **HuggingFace** - For datasets and model hosting
- **VQA-RAD Dataset** - Lau et al.
- **MEDPIX-ShortQA Dataset** - For medical image QA data

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Avala Sri Karthik
**Institution**: Vellore Institute of Technology, Chennai
**Year**: 2024

**Fine-tuned Model**: [titanhacker/Llama-3.2-11B-Vision-Med-Vqa-Finetuned](https://huggingface.co/titanhacker/Llama-3.2-11B-Vision-Med-Vqa-Finetuned)

For questions or collaborations, feel free to reach out!

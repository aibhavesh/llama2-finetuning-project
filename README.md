
# 🦙 Fine-Tuning Llama-2-7B with QLoRA

Fine-tune Meta's **Llama-2-7B-Chat** model on a custom instruction dataset using **QLoRA** (Quantized Low-Rank Adaptation) — enabling efficient training on consumer-grade hardware with minimal memory overhead.

---

## 📖 Overview

This project demonstrates how to fine-tune [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) on the [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) instruction-following dataset. It uses:

- **4-bit quantization** via [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to reduce GPU memory usage
- **LoRA** (Low-Rank Adaptation) via [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
- **SFTTrainer** from [TRL](https://github.com/huggingface/trl) for supervised fine-tuning
- **Hugging Face Transformers** & **Accelerate** for model loading and distributed training

The resulting fine-tuned model is saved as `Llama-2-7b-chat-guanaco-finetune`.

---

## 🏗️ Project Structure

```
llama2-finetuning-project/
├── train.py           # Main fine-tuning script
├── requirements.txt   # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: ≥ 16 GB VRAM; 4-bit quantization allows running on ~10 GB)
- [pip](https://pip.pypa.io/)

### 1. Clone the repository

```bash
git clone https://github.com/aibhavesh/llama2-finetuning-project.git
cd llama2-finetuning-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Key dependencies</summary>

| Package | Version | Purpose |
|---|---|---|
| `transformers` | ≥ 4.31.0 | Model loading & training utilities |
| `datasets` | ≥ 2.14.0 | Dataset loading from Hugging Face Hub |
| `accelerate` | ≥ 0.21.0 | Distributed & mixed-precision training |
| `peft` | ≥ 0.4.0 | LoRA adapter layers |
| `trl` | ≥ 0.4.7 | SFTTrainer for supervised fine-tuning |
| `bitsandbytes` | ≥ 0.41.0 | 4-bit / 8-bit quantization |
| `torch` | ≥ 2.0.0 | Deep learning backend |
| `gradio` | latest | Interactive demo UI |

</details>

### 3. Hugging Face authentication

Llama-2 is a gated model. Accept the license on Hugging Face and log in:

```bash
huggingface-cli login
```

---

## ▶️ How to Run

Start fine-tuning with:

```bash
python train.py
```

The script will:
1. Load the base `Llama-2-7b-chat-hf` model in 4-bit precision
2. Apply LoRA adapters via PEFT
3. Train on the `guanaco-llama2-1k` dataset using `SFTTrainer`
4. Save the fine-tuned model as `Llama-2-7b-chat-guanaco-finetune`

---

## 🔧 Configuration

Key parameters you can adjust in `train.py`:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `NousResearch/Llama-2-7b-chat-hf` | Base model to fine-tune |
| `dataset_name` | `mlabonne/guanaco-llama2-1k` | Training dataset |
| `new_model_name` | `Llama-2-7b-chat-guanaco-finetune` | Output model name |

LoRA & quantization settings can be tuned via `LoraConfig` and `BitsAndBytesConfig` in the script.

---

## 💡 How QLoRA Works

```
Base Model (frozen, 4-bit quantized)
        │
        ▼
  LoRA Adapters (trainable, low-rank matrices)
        │
        ▼
  Fine-tuned Model (adapters merged at inference)
```

QLoRA freezes the original model weights in 4-bit precision and only trains a small set of low-rank adapter parameters (~1–2% of total parameters), drastically reducing memory and compute requirements while preserving model quality.

---

## 📦 Model & Dataset

- **Base model:** [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
- **Dataset:** [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) — 1,000 high-quality instruction-following samples formatted for Llama-2 Chat

---

## 📄 License

This project is released under the [MIT License](LICENSE). Note that the Llama-2 model weights are subject to Meta's [Llama 2 Community License](https://ai.meta.com/llama/license/).

---

## 🙏 Acknowledgements

- [Meta AI](https://ai.meta.com/) for the Llama-2 model
- [Hugging Face](https://huggingface.co/) for the Transformers, PEFT, TRL, and Datasets libraries
- [Tim Dettmers](https://github.com/TimDettmers) for bitsandbytes quantization
- [mlabonne](https://huggingface.co/mlabonne) for the guanaco-llama2 dataset

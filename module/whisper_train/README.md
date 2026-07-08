# Installation

``` bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning transformers evaluate soundfile openai-whisper ctranslate2 tqdm numpy
pip install pytorch-lightning transformers evaluate soundfile openai-whisper ctranslate2 tqdm numpy torchaudio
```

---

# Configuration

Important training parameters are defined in `config.py`.

| Parameter                     | Default        | Description                                                     |
| ----------------------------- | -------------- | --------------------------------------------------------------- |
| `BATCH_SIZE`                  | `8`            | Number of samples processed per optimization step               |
| `gradient_accumulation_steps` | `1`            | Number of steps to accumulate gradients before updating weights |
| `learning_rate`               | `5e-4`         | Initial learning rate for AdamW                                 |
| `P_CLEAN`                     | `0.35`         | Probability of keeping audio clean                              |
| `SNR_DB_RANGE`                | `(15.0, 30.0)` | Noise injection SNR range                                       |
| `AUDIO_MAX_LENGTH`            | `480000`       | Maximum audio length (30 seconds at 16 kHz)                     |

---
# Preparation

``` bash
mkdir assets
```
```text
prepare dataset including wav file and prompt as labeling audio
```

---
# Training

```text
Just run the train.py file
```
---

# Whisper Fine-Tuning Pipeline

A production-ready pipeline for fine-tuning OpenAI Whisper on custom Automatic Speech Recognition (ASR) datasets using **PyTorch Lightning**. The project includes:

* Automated dataset indexing and filtering
* Speaker-aware train/validation splitting
* On-the-fly noise augmentation
* Mixed precision training
* Checkpoint management and experiment tracking
* Automatic conversion to **Hugging Face** and **CTranslate2 (faster-whisper)** formats

The pipeline is designed for efficient training on consumer GPUs while providing deployment-ready models optimized for real-time inference.

---

# System Architecture

The workflow consists of two major stages:

1. **Model Training**
2. **Deployment Optimization**

```text
Raw Audio + Transcripts
        │
        ▼
Data Indexing & Filtering
        │
        ▼
Train / Validation Split
(Speaker-based or Random)
        │
        ▼
On-the-Fly Augmentation
(Noise Mixing + Feature Extraction)
        │
        ▼
PyTorch Lightning Trainer
(Whisper Decoder Fine-Tuning)
        │
        ▼
Lightning Checkpoint (.ckpt)
        │
        ▼
Post-Training Conversion
        │
        ├── OpenAI Whisper (.pt)
        ├── Hugging Face Format
        └── CTranslate2 (INT8 / FP16)
```

---

# Training Pipeline

## 1. Dataset Indexing & Filtering

The pipeline automatically scans audio directories, pairs audio files with their corresponding transcripts, and filters out samples that exceed the maximum supported duration.

Benefits:

* Prevents GPU Out-of-Memory (OOM) issues
* Maintains consistent training batches
* Improves overall training stability

---

## 2. Train / Validation Split

The system prioritizes a **speaker-based split** whenever speaker metadata is available.

### Speaker-Based Split

* Speakers appearing in the validation set are excluded from training.
* Provides a more realistic evaluation of model generalization.

### Automatic Fallback

If the dataset contains only a single speaker, the pipeline automatically switches to a randomized train/validation split.

---

## 3. On-the-Fly Noise Augmentation

Noise is injected during batch loading rather than creating augmented files on disk.

Default behavior:

* 65% probability: Add environmental noise
* 35% probability: Keep audio clean

Noise is mixed using a random Signal-to-Noise Ratio (SNR) selected from a configurable range.

Advantages:

* Reduced storage requirements
* Infinite augmentation combinations
* Better robustness to real-world environments

---

## 4. Whisper Fine-Tuning

Training follows a decoder-focused strategy:

* Whisper Encoder: Frozen
* Whisper Decoder: Trainable

The encoder extracts acoustic representations while the decoder learns:

* Domain-specific vocabulary
* Pronunciation patterns
* Language-specific grammar
* Custom transcription conventions

Training objective:

```text
Cross-Entropy Loss
```

between predicted tokens and ground-truth transcripts.

---

## 5. Model Conversion Pipeline

After training completes, the best checkpoint is automatically converted through multiple formats:

```text
Lightning Checkpoint (.ckpt)
          │
          ▼
OpenAI Whisper (.pt)
          │
          ▼
Hugging Face Model
          │
          ▼
CTranslate2 Model
```

This conversion process:

* Removes Lightning-specific metadata
* Maps weights to Hugging Face naming conventions
* Generates optimized inference artifacts

The final CTranslate2 model can be directly used with:

```python
from faster_whisper import WhisperModel
```

for low-latency production inference.


## Recommended Settings for Large Datasets

For datasets around **20 GB** (~50,000 clips / 60+ hours):

### GPU Has Sufficient VRAM

```python
BATCH_SIZE = 16
```

or

```python
BATCH_SIZE = 32
```

---

### Limited GPU Memory

Keep a smaller batch size and use gradient accumulation:

```python
BATCH_SIZE = 8
gradient_accumulation_steps = 4
```

This simulates an effective batch size of:

```text
8 × 4 = 32
```

without increasing memory usage.

---

### Heavy Noise Environments

Reduce the minimum SNR:

```python
SNR_DB_RANGE = (10.0, 30.0)
```

to expose the model to more aggressive background noise.

---

# Encoder Freezing Strategy

By default, the encoder is frozen:

```python
for p in self.model.encoder.parameters():
    p.requires_grad = False
```

Only the decoder is updated during training.

---

## Why Freeze the Encoder?

OpenAI Whisper was pretrained on approximately **680,000 hours** of multilingual audio.

The encoder already serves as a highly effective acoustic feature extractor.

Freezing it provides several advantages:

### Lower Memory Usage

Training memory consumption can be reduced by more than 50%.

### Faster Training

Training speed is typically:

```text
2x – 3x faster
```

compared to full-model fine-tuning.

### Better Generalization

Freezing helps preserve Whisper's broad acoustic knowledge and reduces the risk of:

* Overfitting
* Catastrophic forgetting
* Loss of speaker diversity

---

# When Should You Unfreeze the Encoder?

For most datasets, encoder fine-tuning is unnecessary.

Consider unfreezing only when dealing with:

* Extreme radio communication noise
* Underwater audio
* Specialized medical microphones
* Child speech datasets
* Highly unusual accents
* Acoustic domains not represented in Whisper pretraining

---

# Recommended Two-Stage Fine-Tuning Strategy

Instead of training the full model from the beginning, use a staged approach.

## Stage 1: Decoder Adaptation

Epochs:

```text
1 – 7
```

Configuration:

```python
learning_rate = 5e-4
```

Encoder:

```text
Frozen
```

Goal:

* Adapt vocabulary
* Learn transcription style
* Stabilize decoder predictions

---

## Stage 2: Acoustic Adaptation

Epochs:

```text
8 – 10
```

Configuration:

```python
learning_rate = 1e-5
```

Encoder:

```text
Unfrozen
```

Goal:

* Refine acoustic representations
* Adapt to domain-specific audio characteristics
* Preserve pretrained knowledge while improving specialization

This staged strategy generally provides the best balance between adaptation performance and model stability.

---

# Output Artifacts

After training, the pipeline generates:

```text
checkpoints/
├── *.ckpt

model_finetuned.pt

huggingface_model/
├── config.json
├── generation_config.json
├── model.safetensors
├── preprocessor_config.json
├── tokenizer_config.json
└── tokenizer.json

models/ctranslate2_model
├── model.bin
├── config.json
└── vocabulary.json

logs/ -> not important
```

The generated CTranslate2 model is ready for deployment using:

* faster-whisper
* CPU inference
* GPU inference
* INT8 quantization
* FP16 acceleration

---

# Summary

This project provides a complete Whisper fine-tuning workflow optimized for production environments:

* Speaker-aware dataset splitting
* Dynamic noise augmentation
* Memory-efficient decoder fine-tuning
* Automated checkpoint conversion
* Hugging Face export
* CTranslate2 optimization for fast inference

For most custom ASR datasets, keeping the encoder frozen and training only the decoder offers the best trade-off between accuracy, training speed, and hardware efficiency.

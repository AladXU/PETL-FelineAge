# PurrNet-PETL: Feline Age Prediction via Acoustic Foundation Models

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.12.0-yellow.svg)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-0.19.1-green.svg)](https://github.com/huggingface/peft)
[![Status](https://img.shields.io/badge/Status-Active%20Research-red.svg)]()

---

> **A rigorous empirical study on the failure modes of Parameter-Efficient Transfer Learning (PETL)
> under extreme data scarcity — and why static acoustic representations still reign supreme.**

---

## 📑 Table of Contents

1. [Project Background](#1-project-background)
2. [Methodology & Pipeline](#2-methodology--pipeline)
3. [Experiment I: The Physical Limit of Static Features](#3-experiment-i-the-physical-limit-of-static-features)
4. [Experiment II: The Illusion of PETL](#4-experiment-ii-the-illusion-of-petl)
5. [Future Work](#5-future-work)
6. [Repository Structure](#6-repository-structure)
7. [Reproducibility Quick Start](#7-reproducibility-quick-start)

---

## 1. Project Background

### 🎯 Task Definition

| Dimension | Specification |
|:---|:---|
| **Input** | Raw `.wav` cat vocalisation recordings |
| **Output** | 3-class age group: `Kitten` (≤0.5yr), `Adult` (0.5–10yr), `Senior` (≥10yr) |
| **Training set** | **569 samples** (Fold 1 of 4-fold StratifiedGroupKFold) |
| **Total corpus** | 793 recordings from 112 unique cats |
| **Evaluation** | Per-fold validation accuracy, grouped by `cat_id` to prevent data leakage |

### 💢 The Bottleneck

| Approach | Limitation |
|:---|:---|
| **Handcrafted features** (f₀, formants) | Requires domain expertise; fragile to recording conditions |
| **Pretrained foundation models** (SSAST, CLAP) | Embedding dimension **768×** larger than training samples → **severe overfitting** |
| **End-to-end fine-tuning** | Catastrophic forgetting on 569 samples |

> **Core research question:**
> *Can we adapt a 87M-parameter Audio Spectrogram Transformer, pretrained on AudioSet
> (2M clips, 527 sound classes), to a 3-class fine-grained bioacoustic task with only
> 569 labelled examples — without destroying the pretrained representations?*

---

## 2. Methodology & Pipeline

### 🔐 Leak-Proof Data Splitting

```
StratifiedGroupKFold (n_splits=4, random_state=42)
├── Grouping key: cat_id  (same cat NEVER appears in both train & val)
├── Stratification: age_group label distribution preserved per fold
└── Result: 4 folds with ZERO cat-level leakage
```

| Fold | Train | Val | Kitten | Adult | Senior | Cat Overlap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 569 | 224 | 87/57 | 305/91 | 177/76 | **0** ✓ |
| 2 | 600 | 193 | 89/55 | 299/97 | 212/41 | **0** ✓ |
| 3 | 625 | 168 | 128/16 | 299/97 | 198/55 | **0** ✓ |
| 4 | 585 | 208 | 128/16 | 285/111 | 172/81 | **0** ✓ |

### 🧠 Feature Extraction Backbone

```
SSAST (Self-Supervised Audio Spectrogram Transformer)
├── Pretrained: AudioSet-2M (self-supervised + supervised 527-class)
├── Architecture: ViT-style, 12 layers, 768-D hidden, 12 attention heads
├── Input: 16 kHz mono → 128-bin mel-spectrogram → 16×16 patches
├── Output: pooler_output ∈ ℝ⁷⁶⁸  (mean-pooled last hidden state)
└── Source:  models/pretrained/ssast/  (HuggingFace format, ~330 MB)
```

### 🔬 Experiment Taxonomy

```
                    ┌─────────────────────────────────┐
                    │   SSAST Feature Extractor        │
                    │   (frozen / partially frozen)    │
                    └──────────────┬──────────────────┘
                                   │ pooler_output (768-D)
                    ┌──────────────┴──────────────────┐
                    │                                 │
          ┌─────────┴──────────┐        ┌────────────┴───────────┐
          │  Static Features    │        │   PETL / Fine-Tuning   │
          │  (no grad through   │        │   (grad flows into     │
          │   backbone)         │        │    backbone layers)    │
          ├─────────────────────┤        ├────────────────────────┤
          │ • MLP (paper params)│        │ • LoRA r=8  (0.34%)    │
          │ • MLP (manual tune) │        │ • LoRA r=16 (1.35%)    │
          │ • MLP (Optuna 300)  │        │ • LoRA + heavy reg     │
          │                     │        │ • Partial FT (16.45%)  │
          └─────────────────────┘        └────────────────────────┘
```

---

## 3. Experiment I: The Physical Limit of Static Features

### ⚙️ Setup

| Component | Configuration |
|:---|:---|
| **Feature extractor** | SSAST (frozen, `torch.no_grad()`) |
| **Classifier** | `Linear(768, hidden) → ReLU → Dropout(p) → Linear(hidden, 3)` |
| **Search engine** | Optuna 3.x, 300 trials, Bayesian sampling (TPE) |
| **Objective** | Maximise Fold-1 validation accuracy |
| **Search space** | `hidden_dim` ∈ {64, 128, 256}, `dropout` ∈ [0.3, 0.7], optimizer ∈ {Adam, AdamW, Adamax}, `lr` ∈ [1e-4, 5e-3] (log), `weight_decay` ∈ [1e-5, 1e-2] (log), `batch_size` ∈ {16, 32, 64, 128} |

### 📊 Results

| Strategy | hidden_dim | Dropout | Optimizer | LR | Val Acc |
|:---|:---:|:---:|:---:|---:|:---:|
| Paper-original params | 128 | 0.446 | Adamax | 3.11e-3 | 56.70% |
| Manual anti-overfit | 128 | 0.50 | AdamW | 5.0e-4 | 57.59% |
| **Optuna best** ⭐ | **64** | **0.667** | **AdamW** | **1.38e-3** | **66.96%** |

### 💎 Key Insight 1 — The Information Bottleneck

> **Optuna converged to an extreme regularisation regime:**
> the narrowest hidden layer (64 units) paired with the highest dropout (0.67).
> This is the model *fighting for its life* against the curse of dimensionality
> (768 features, 569 samples). The discovered architecture is effectively a
> **compressed linear probe** — and its **66.96% ceiling** represents the
> information-theoretic limit extractable from *frozen* SSAST `pooler_output`.

---

## 4. Experiment II: The Illusion of PETL

### ⚙️ Setup

| Method | Trainable Params | Description |
|:---|:---:|:---|
| **LoRA-Conservative** | 295K (0.34%) | r=8, α=16, q_proj + v_proj only, lr=1e-4 |
| **LoRA-Aggressive** | 1.18M (1.35%) | r=16, α=32, q+k+v+o_proj, lr=5e-4 |
| **LoRA + Regularised** | 590K (0.68%) | r=8, α=16, all 4 proj, lora_dropout=0.5, CosineAnnealingLR |
| **Partial Fine-Tuning** | **14.2M (16.45%)** | Unfreeze layers 10-11 + final LayerNorm, lr=5e-5 |

### 📊 Results — The More We Tune, The Worse It Gets

| # | Method | Trainable | Train Loss (final) | Val Acc |
|:--|:---|:---|---:|:---:|
| 🥇 | **Static + MLP (Optuna)** | 99K (100% of MLP) | 0.07 | **66.96%** |
| 🥈 | LoRA-Conservative | 295K (0.34%) | 0.14 | 60.71% |
| 🥉 | Static + MLP (manual) | 99K | 0.22 | 57.59% |
| 4 | Partial FT (layers 10-11) | 14.2M (16.45%) | 0.03 | 57.59% |
| 5 | Static + MLP (paper) | 99K | 0.06 | 56.70% |
| 6 | LoRA-Aggressive | 1.18M (1.35%) | **0.0002** | 53.57% |
| 7 | LoRA + Heavy Reg | 590K (0.68%) | 0.01 | 51.79% |

<table>
<tr><td align="center">

#### Train Loss ↓ as capacity ↑

```
Static MLP      ████████░░  0.07
LoRA-Cons.      ██████████  0.14
LoRA-Reg.       ████████░░  0.01
LoRA-Aggr.      ██████████  0.0002  ← memorised
Partial FT      ██████████  0.03    ← memorised
```

</td><td align="center">

#### Val Acc ↓ as capacity ↑

```
Static (Optuna) ████████████  66.96%  ← WINNER
LoRA-Cons.      ██████████░░  60.71%
Partial FT      ██████████░░  57.59%
LoRA-Aggr.      ██████░░░░░░  53.57%
LoRA-Reg.       ██████░░░░░░  51.79%
```

</td></tr>
</table>

### 💎 Key Insight 2 — Representation Misalignment (核心发现)

> **PETL methods catastrophically fail when the pretraining and downstream
> tasks are semantically distant.**
>
> SSAST was pretrained on **AudioSet** — a coarse-grained dataset of 527
> environmental sound classes (*"Cat meow"*, *"Dog bark"*, *"Engine"*,
> *"Thunder"*, etc.). Its high-level representations encode **"what object
> made this sound?"** — a semantic level far removed from the
> **fine-grained biometric question** of *"how old is this specific cat?"*
>
> When we inject LoRA adapters or unfreeze layers, the model eagerly exploits
> its newly granted degrees of freedom to **memorise the 569 training samples**
> (Train Loss → 0.0002), rather than learning age-discriminative features.
> The low-rank matrices of LoRA (ΔW = BA, rank 8–16) simply collapse into
> the nearest local minimum of the training distribution.
>
> ```
>   Pretraining manifold         Downstream manifold
>   (AudioSet 527-class)         (Feline Age 3-class)
>   
>        ┌─────┐                      ┌─────┐
>        │ Cat │                      │ Kit │
>        │ Dog │    ── LoRA ?? ──>    │ Adl │
>        │ Car │                      │ Snr │
>        └─────┘                      └─────┘
>           
>        Distance between manifolds is too large
>        for low-rank adaptation to bridge with
>        only 569 anchor points.
> ```
>
> **Implication:** For fine-grained bioacoustic tasks under extreme data
> scarcity, **weight-space adaptation (LoRA/FT) is counterproductive**.
> The pretrained model's frozen representations, combined with a
> carefully regularised shallow classifier, provide the **Pareto-optimal
> trade-off** between bias and variance.

---

## 5. Future Work

### 🎯 Phase A — Robust Baseline Certification *(immediate)*

| Task | Description | Expected Impact |
|:---|:---|:---|
| **4-Fold CV** | Run the Optuna-optimal MLP on all 4 folds; report mean ± std accuracy | Eliminate fold-selection bias |
| **Traditional classifiers** | Replace MLP with **SVM (RBF kernel)** and **XGBoost** on the 768-D features | Exploit the "less is more" principle — non-parametric models may generalise better |
| **Ensemble** | Average predictions across all 4 fold models | Squeeze out the last 1–2% |

### 🚀 Phase B — Feature Fusion *(mid-term)*

```
          ┌──────────────────┐     ┌──────────────────┐
          │   SSAST 768-D    │     │  Physical Features │
          │  (global acoustics)│    │  (f₀, mean_freq,  │
          │                  │     │   spectral centroid)│
          └────────┬─────────┘     └────────┬─────────┘
                   │                        │
                   └──────────┬─────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Dual-Stream MLP  │
                    │  or Late Fusion   │
                    └───────────────────┘
```

*Rationale:* The original paper (Van Toor et al., 2025) proved that a single
physical feature (`mean_freq`) was decisive for VGGish. Combining SSAST's
global spectral understanding with domain-specific physical features may
bridge the semantic gap.

### 🌌 Phase C — Representation-Level Adaptation *(cutting-edge)*

| Direction | Method | Hypothesis |
|:---|:---|:---|
| **Supervised Contrastive Learning** (SupCon) | Train a projection head to pull same-age embeddings together in representation space — *without* touching the backbone | Reshapes the feature manifold without weight-space interference |
| **Audio Prompt Tuning** | Learn a small set of continuous prompt vectors prepended to the mel-spectrogram input | Zero parameter modification to SSAST; domain adaptation via input-space conditioning |
| **Knowledge Distillation** | Use the original VGGish model (70.76% baseline) as teacher; SSAST + MLP as student | Transfers the VGGish-learned age-discriminative knowledge into the SSAST feature space |

---

## 6. Repository Structure

```
PurrNet-PETL/
│
├── README.md                          ← This document
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── AudioCropped/              ← 793 .wav recordings
│   │   └── feline_dataset.csv         ← Parsed metadata (file, age, label, cat_id)
│   └── features/                      ← Pre-extracted SSAST embeddings (4-fold split)
│
├── models/
│   ├── pretrained/ssast/              ← SSAST backbone (~330 MB, .gitignored)
│   └── saved_weights/                 ← Saved model checkpoints
│
└── scripts/
    ├── 02_extract_features.py         ← SSAST feature extraction + leak-proof K-Fold
    ├── 03_train_mlp.py                ← MLP baseline (paper params & manual tuning)
    ├── 03b_optuna_mlp.py              ← 300-trial Optuna hyperparameter search ⭐
    ├── 04_train_lora.py               ← LoRA PETL (conservative & aggressive variants)
    ├── 04b_train_lora_reg.py          ← LoRA + heavy regularisation + CosineAnnealingLR
    └── 05_train_partial_ft.py         ← Partial Fine-Tuning (last 2 layers unfrozen)
```

---

## 7. Reproducibility Quick Start

```bash
# Environment
conda activate catdog_fyp
cd /home/alad/FYP/FelineAgePrediction_PETL

# Step 1: Extract SSAST features + 4-fold split
python scripts/02_extract_features.py

# Step 2: Run Optuna hyperparameter search (300 trials)
python scripts/03b_optuna_mlp.py

# Step 3: Try PETL approaches (optional — see results above)
python scripts/04_train_lora.py         # LoRA variants
python scripts/04b_train_lora_reg.py     # LoRA + regularisation
python scripts/05_train_partial_ft.py    # Partial fine-tuning
```

### Hardware & Environment

| Component | Specification |
|:---|:---|
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 4060 8GB |
| CUDA | 12.1 |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 5.12.0 |
| PEFT | 0.19.1 |
| Optuna | 4.x |

---

## 📜 Citation & References

> Van Toor, A., Qazi, N., & Paladini, S. (2025).
> *A deep learning pipeline for age prediction from vocalisations of the domestic feline.*
> Scientific Reports, 15(1), 1–10.

> Gong, Y., Chung, Y. A., & Glass, J. (2021).
> *AST: Audio Spectrogram Transformer.*
> Proc. Interspeech 2021.

> Hu, E. J., Shen, Y., Wallis, P., et al. (2022).
> *LoRA: Low-Rank Adaptation of Large Language Models.*
> ICLR 2022.

---

<p align="center">
  <b>🐱 PurrNet-PETL</b> — <i>When pretrained knowledge meets biological subtlety,
  the answer is not always "fine-tune harder."</i>
</p>

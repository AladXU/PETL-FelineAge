# SSAST-LoRA Feline Age Prediction (WIP 🚧)

This repository contains the ongoing research and PyTorch implementation for feline age prediction using PEFT (LoRA) on Self-Supervised Audio Spectrogram Transformers (SSAST).

## 🚀 Current Status: Active Development
- [x] SSAST backbone integration with HuggingFace
- [x] LoRA injection pipeline implementation
- [x] Audio preprocessing and K-Fold cross-validation feature extraction (`02_extract_features.py`)
- [x] MLP Baseline Training (`03_train_mlp.py`)
- [x] Hyperparameter tuning via Optuna 300-trials (`03b_optuna_mlp.py`)
- [ ] Final evaluation and reporting

*Note: Large audio datasets and pre-trained weights are excluded from this repository. Complete evaluation results will be released upon project completion.*

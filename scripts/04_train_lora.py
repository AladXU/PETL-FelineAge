#!/usr/bin/env python3
"""
04_train_lora.py — LoRA Fine-Tuning of SSAST for Feline Age Prediction
======================================================================
PETL approach: freeze SSAST backbone, inject LoRA adapters into attention
layers, train a lightweight classification head on pooler_output.
"""

import os
import copy
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AUDIO_DIR = "./data/raw/AudioCropped"
CSV_PATH = "./data/raw/feline_dataset.csv"
MODEL_PATH = "./models/pretrained/ssast"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_LENGTH = 1024
BATCH_SIZE = 8
LR = 5e-4
EPOCHS = 30
PATIENCE = 8
NUM_CLASSES = 3
FOLD = 1

print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1.  Real-time data loading  (Fold 1, identical to previous splits)
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records from {CSV_PATH}")

sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
all_idx = np.arange(len(df))
splits = list(sgkf.split(all_idx, df["label"].values, df["cat_id"].values))
train_idx, val_idx = splits[FOLD - 1]  # Fold 1

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)
print(f"Fold {FOLD}:  train={len(df_train)}  val={len(df_val)}")


class FelineAudioDataset(Dataset):
    """Loads .wav on-the-fly, resamples → 16 kHz, extracts mel-spectrogram."""

    def __init__(self, dataframe, feature_extractor, audio_dir):
        self.df = dataframe
        self.feature_extractor = feature_extractor
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["file_name"])

        waveform, sr = torchaudio.load(audio_path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample → 16 kHz
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        # Feature-extractor expects 1-D numpy array
        raw_audio = waveform.squeeze().numpy()
        inputs = self.feature_extractor(
            raw_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        # input_values: (1, time, mel_bins) → squeeze batch dim
        return inputs.input_values.squeeze(0), row["label"]


print("Loading feature extractor …")
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_PATH)

train_dataset = FelineAudioDataset(df_train, feature_extractor, AUDIO_DIR)
val_dataset   = FelineAudioDataset(df_val,   feature_extractor, AUDIO_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# Quick shape check
sample_x, sample_y = train_dataset[0]
print(f"Sample input shape: {sample_x.shape}  (expected: ({MAX_LENGTH}, 128))")

# ---------------------------------------------------------------------------
# 2.  Build LoRA PEFT model
# ---------------------------------------------------------------------------
print(f"\nLoading base ASTModel from {MODEL_PATH} …")
base_model = ASTModel.from_pretrained(MODEL_PATH)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
)

peft_model = get_peft_model(base_model, lora_config)
print("\n--- Trainable Parameters ---")
peft_model.print_trainable_parameters()


class LoRAASTClassifier(nn.Module):
    """PEFT-wrapped SSAST + classification head."""

    def __init__(self, peft_backbone, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = peft_backbone
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_values):
        outputs = self.backbone(input_values=input_values)
        pooled = outputs.pooler_output  # (B, 768)
        return self.classifier(pooled)


model = LoRAASTClassifier(peft_model).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------------------------------------------------------------------
# 3.  Training loop
# ---------------------------------------------------------------------------
best_val_loss = float("inf")
best_weights = None
best_epoch = 0
patience_counter = 0

print(f"\n{'Epoch':>6s}  {'Train Loss':>10s}  {'Val Loss':>10s}  {'Val Acc':>8s}")
print("-" * 44)

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d} [Train]", ncols=80, leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    train_loss /= len(train_loader.dataset)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch:2d} [Val]  ", ncols=80, leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = 100.0 * correct / len(val_loader.dataset)

    print(f"{epoch:6d}  {train_loss:10.4f}  {val_loss:10.4f}  {val_acc:7.2f}%")

    # ---- Early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
        break

# ---------------------------------------------------------------------------
# 4.  Final result
# ---------------------------------------------------------------------------
model.load_state_dict(best_weights)
model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
best_val_acc = 100.0 * correct / len(val_loader.dataset)

print("\n" + "=" * 44)
print(f"  🏆 BEST Validation Accuracy (LoRA):  {best_val_acc:.2f}%")
print(f"     Achieved at Epoch:                {best_epoch}")
print("=" * 44)

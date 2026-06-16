#!/usr/bin/env python3
"""
05_train_partial_ft.py — Partial Fine-Tuning of SSAST (Last 2 Layers)
======================================================================
Strategy: freeze all SSAST weights → unfreeze last 2 transformer layers
          + final LayerNorm → train lightweight classification head.
Hypothesis: AudioSet high-level semantics → feline age via mid-depth adaptation.
"""

import os
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
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
LR = 5e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 30
PATIENCE = 7
NUM_CLASSES = 3
FOLD = 1

print(f"Device: {DEVICE}")
print(f"LR={LR}, weight_decay={WEIGHT_DECAY}, epochs={EPOCHS}, patience={PATIENCE}")

# ---------------------------------------------------------------------------
# 1.  Real-time data loading  (Fold 1)
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records from {CSV_PATH}")

sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
all_idx = np.arange(len(df))
train_idx, val_idx = list(sgkf.split(all_idx, df["label"], df["cat_id"]))[FOLD - 1]

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
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        raw_audio = waveform.squeeze().numpy()
        inputs = self.feature_extractor(
            raw_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        return inputs.input_values.squeeze(0), row["label"]


print("Loading feature extractor …")
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_PATH)

train_dataset = FelineAudioDataset(df_train, feature_extractor, AUDIO_DIR)
val_dataset   = FelineAudioDataset(df_val,   feature_extractor, AUDIO_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# 2.  Build model: freeze all → unfreeze last 2 layers + layernorm
# ---------------------------------------------------------------------------
print(f"\nLoading base ASTModel from {MODEL_PATH} …")
model = ASTModel.from_pretrained(MODEL_PATH)

# Step 1: freeze everything
for param in model.parameters():
    param.requires_grad = False

# Step 2: unfreeze last 2 transformer layers (layers 10, 11) + final layernorm
NUM_LAYERS = len(model.layers)  # 12
UNFREEZE_START = NUM_LAYERS - 2  # layer 10

for layer in model.layers[UNFREEZE_START:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.layernorm.parameters():
    param.requires_grad = True

print(f"Total layers: {NUM_LAYERS}, unfrozen layers: [{UNFREEZE_START}..{NUM_LAYERS-1}] + layernorm")

# Step 3: classification head
classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(768, NUM_CLASSES),
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
classifier_params = sum(p.numel() for p in classifier.parameters())
print(f"\n--- Trainable Parameters ---")
print(f"  Backbone (partial):  {trainable:>10,}  ({100*trainable/total:.2f}%)")
print(f"  Classifier head:     {classifier_params:>10,}")
print(f"  Total trainable:     {trainable+classifier_params:>10,}  ({100*(trainable+classifier_params)/(total+classifier_params):.2f}%)")

model.to(DEVICE)
classifier.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(classifier.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# ---------------------------------------------------------------------------
# 3.  Training loop
# ---------------------------------------------------------------------------
best_val_loss = float("inf")
best_state = None
best_epoch = 0
patience_counter = 0

print(f"\n{'Epoch':>6s}  {'Train Loss':>10s}  {'Val Loss':>10s}  {'Val Acc':>8s}")
print("-" * 44)

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    classifier.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d} [Train]", ncols=80, leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_values=xb)
        logits = classifier(outputs.pooler_output)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    train_loss /= len(train_loader.dataset)

    # ---- Validate ----
    model.eval()
    classifier.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch:2d} [Val]  ", ncols=80, leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(input_values=xb)
            logits = classifier(outputs.pooler_output)
            val_loss += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = 100.0 * correct / len(val_loader.dataset)

    print(f"{epoch:6d}  {train_loss:10.4f}  {val_loss:10.4f}  {val_acc:7.2f}%")

    # ---- Early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            "model": copy.deepcopy(model.state_dict()),
            "classifier": copy.deepcopy(classifier.state_dict()),
        }
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
model.load_state_dict(best_state["model"])
classifier.load_state_dict(best_state["classifier"])
model.eval()
classifier.eval()
correct = 0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        outputs = model(input_values=xb)
        logits = classifier(outputs.pooler_output)
        correct += (logits.argmax(dim=1) == yb).sum().item()
best_val_acc = 100.0 * correct / len(val_loader.dataset)

print("\n" + "=" * 44)
print(f"  🏆 BEST Validation Accuracy (Partial FT):  {best_val_acc:.2f}%")
print(f"     Achieved at Epoch:                      {best_epoch}")
print("=" * 44)

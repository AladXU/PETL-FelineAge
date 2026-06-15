#!/usr/bin/env python3
"""
02_extract_features.py — Feline Age Prediction (PETL Pipeline)
=============================================================
Task A: Parse WAV filenames → dataset CSV with age_group labels
Task B: Load pretrained SSAST → extract 768-dim pooler_output embeddings
Task C: StratifiedGroupKFold (4-fold) leak-free split by cat_id
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIO_DIR = "./data/raw/AudioCropped"
CSV_PATH = "./data/raw/feline_dataset.csv"
FEATURES_DIR = "./data/features"
MODEL_PATH = "./models/pretrained/ssast"

SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Audio directory: {AUDIO_DIR}")
print(f"Model path: {MODEL_PATH}")

# ===========================================================================
# TASK A — Parse filenames and generate dataset CSV
# ===========================================================================
print("\n" + "=" * 60)
print("TASK A: Parsing WAV filenames → dataset CSV")
print("=" * 60)

records = []
for fname in sorted(os.listdir(AUDIO_DIR)):
    if not fname.endswith(".wav"):
        continue

    name = fname[:-4]  # strip ".wav"
    parts = name.split("-")

    # --- exact_age from leading token (e.g., "0.5Y" → 0.5) ---
    age_str = parts[0]
    exact_age = float(age_str.replace("Y", ""))

    # --- cat_id: first segment matching 3-4 digits + uppercase letter ---
    cat_id = None
    for part in parts[1:]:
        if re.match(r"^\d{3,4}[A-Z]", part):
            cat_id = part
            break
    if cat_id is None:
        raise ValueError(f"Cannot extract cat_id from: {fname}")

    # --- age group assignment ---
    if exact_age <= 0.5:
        age_group = "Kitten"
        label = 0
    elif exact_age < 10.0:
        age_group = "Adult"
        label = 1
    else:
        age_group = "Senior"
        label = 2

    records.append(
        {
            "file_name": fname,
            "exact_age": exact_age,
            "age_group": age_group,
            "label": label,
            "cat_id": cat_id,
        }
    )

df = pd.DataFrame(records)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
df.to_csv(CSV_PATH, index=False)
print(f"\n✓ Saved {len(df)} records → {CSV_PATH}")

# --- class distribution ---
print("\nClass distribution:")
for lbl, name in enumerate(["Kitten", "Adult", "Senior"]):
    cnt = (df["label"] == lbl).sum()
    print(f"  {name:>7s}  (label={lbl}):  {cnt:4d}  samples")
print(f"  {'TOTAL':>7s}           {len(df):4d}")

# ===========================================================================
# TASK B — Load SSAST and extract 768-dim pooler_output
# ===========================================================================
print("\n" + "=" * 60)
print("TASK B: Loading SSAST → extracting 768-dim features")
print("=" * 60)

print(f"\nLoading feature extractor from {MODEL_PATH} …")
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_PATH)

print(f"Loading ASTModel from {MODEL_PATH} …")
model = ASTModel.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print(f"  model_type  = {model.config.model_type}")
print(f"  hidden_size = {model.config.hidden_size}")

X_list, y_list, groups_list = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting", ncols=80):
    audio_path = os.path.join(AUDIO_DIR, row["file_name"])

    # 1. Load audio
    waveform, sr = torchaudio.load(audio_path)

    # 2. Convert to mono  (mean across channels)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3. Resample to 16 kHz  ← CRITICAL for SSAST
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # 4. Feature-extractor expects 1-D numpy array
    raw_audio = waveform.squeeze().numpy()

    inputs = feature_extractor(
        raw_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5. Forward pass  (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. pooler_output → 768-dim vector
    embedding = outputs.pooler_output.squeeze().cpu().numpy()

    X_list.append(embedding)
    y_list.append(row["label"])
    groups_list.append(row["cat_id"])

# Stack into arrays
X = np.stack(X_list, axis=0).astype(np.float32)
y = np.array(y_list, dtype=np.int64)
groups = np.array(groups_list, dtype=str)

print(f"\nFeature matrix  X:      {X.shape}   (n_samples, 768)")
print(f"Labels          y:      {y.shape}")
print(f"Cat-ID groups   groups: {groups.shape}  ({len(np.unique(groups))} unique cats)")

# Quick sanity checks
assert X.shape == (len(df), 768), f"Unexpected X shape: {X.shape}"
assert y.shape == (len(df),), f"Unexpected y shape: {y.shape}"
assert not np.isnan(X).any(), "NaN values found in features!"
print("✓ All sanity checks passed")

# ===========================================================================
# TASK C — StratifiedGroupKFold  (4-fold, leak-free)
# ===========================================================================
print("\n" + "=" * 60)
print("TASK C: StratifiedGroupKFold  (4-fold, grouped by cat_id)")
print("=" * 60)

os.makedirs(FEATURES_DIR, exist_ok=True)

sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_cats = set(groups[train_idx])
    val_cats = set(groups[val_idx])
    overlap = train_cats & val_cats

    # Save fold arrays
    np.save(os.path.join(FEATURES_DIR, f"fold{fold_idx}_X_train.npy"), X_train)
    np.save(os.path.join(FEATURES_DIR, f"fold{fold_idx}_y_train.npy"), y_train)
    np.save(os.path.join(FEATURES_DIR, f"fold{fold_idx}_X_val.npy"), X_val)
    np.save(os.path.join(FEATURES_DIR, f"fold{fold_idx}_y_val.npy"), y_val)

    # Per-class counts for train/val
    train_dist = ", ".join(
        [f"{name}={np.sum(y_train == lbl)}" for lbl, name in enumerate(["K", "A", "S"])]
    )
    val_dist = ", ".join(
        [f"{name}={np.sum(y_val == lbl)}" for lbl, name in enumerate(["K", "A", "S"])]
    )

    print(
        f"Fold {fold_idx}:  train={len(train_idx):4d} [{train_dist}]  "
        f"val={len(val_idx):4d} [{val_dist}]  "
        f"cat_overlap={len(overlap)} ✓"
    )

print(f"\n✓ All folds saved → {FEATURES_DIR}/")
print("\n" + "=" * 60)
print("DONE — Feature extraction and data split completed.")
print("=" * 60)

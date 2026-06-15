#!/usr/bin/env python3
"""
03_train_mlp.py — MLP Baseline on Fold-1 SSAST Features (768-D)
==============================================================
Architecture: Linear(768,128) → ReLU → Dropout(0.44571) → Linear(128,3)
Optimizer:    Adamax  lr=0.00311
Loss:         CrossEntropyLoss
Early Stop:   patience=15 on Val Loss
"""

import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_DIR = "./data/features"
FOLD = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 100
PATIENCE = 15
NUM_CLASSES = 3

print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
X_train = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_y_train.npy"))
X_val   = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_X_val.npy"))
y_val   = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_y_val.npy"))

print(f"Loaded Fold {FOLD}: X_train={X_train.shape}, y_train={y_train.shape}, "
      f"X_val={X_val.shape}, y_val={y_val.shape}")

# Standardise features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.LongTensor(y_val)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                          batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# 2. Model  (Van Toor architecture)
# ---------------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(768, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, NUM_CLASSES),
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
best_val_loss = float("inf")
best_weights  = None
best_epoch    = 0
patience_counter = 0

print(f"\n{'Epoch':>6s}  {'Train Loss':>10s}  {'Val Loss':>10s}  {'Val Acc':>8s}")
print("-" * 44)

for epoch in range(1, EPOCHS + 1):
    # ----- Train -----
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # ----- Validate -----
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = 100.0 * correct / len(val_loader.dataset)

    print(f"{epoch:6d}  {train_loss:10.4f}  {val_loss:10.4f}  {val_acc:7.2f}%")

    # ----- Early stopping -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch} "
              f"(no improvement for {PATIENCE} epochs).")
        break

# ---------------------------------------------------------------------------
# 4. Final result
# ---------------------------------------------------------------------------
model.load_state_dict(best_weights)

# Recompute best val acc
model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
best_val_acc = 100.0 * correct / len(val_loader.dataset)

print("\n" + "=" * 44)
print(f"  BEST Validation Accuracy: {best_val_acc:.2f}%")
print(f"  Achieved at Epoch:        {best_epoch}")
print("=" * 44)

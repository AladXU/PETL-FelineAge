#!/usr/bin/env python3
"""
03b_optuna_mlp.py — Optuna Hyperparameter Search for MLP on SSAST Features
==========================================================================
300-trial search over hidden_dim, dropout, optimizer, lr, weight_decay, batch_size.
Objective: maximise validation accuracy on Fold 1.
"""

import os
import copy

import numpy as np
import optuna
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
MAX_EPOCHS = 40
PATIENCE = 6
N_TRIALS = 300

print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load and preprocess data (once)
# ---------------------------------------------------------------------------
X_train_raw = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_X_train.npy"))
y_train_raw = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_y_train.npy"))
X_val_raw   = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_X_val.npy"))
y_val_raw   = np.load(os.path.join(FEATURES_DIR, f"fold{FOLD}_y_val.npy"))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled   = scaler.transform(X_val_raw)

# Convert to tensors once
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train_raw)
X_val_t   = torch.FloatTensor(X_val_scaled)
y_val_t   = torch.LongTensor(y_val_raw)

print(f"Data: X_train={X_train_t.shape}, y_train={y_train_t.shape}, "
      f"X_val={X_val_t.shape}, y_val={y_val_t.shape}")

# ---------------------------------------------------------------------------
# Objective function for Optuna
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    # --- Sample hyperparameters ---
    hidden_dim    = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout_rate  = trial.suggest_float("dropout_rate", 0.3, 0.7)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adamax", "AdamW", "Adam"])
    lr            = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # --- DataLoaders ---
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size, shuffle=False)

    # --- Build model ---
    model = nn.Sequential(
        nn.Linear(768, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, 3),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr,
                                       weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
    else:  # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_weights = None
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Report intermediate value for pruning feedback
    trial.report(best_val_acc, step=1)

    return best_val_acc


# ---------------------------------------------------------------------------
# Run Optuna Study
# ---------------------------------------------------------------------------
print(f"\nStarting Optuna study: {N_TRIALS} trials, direction=maximize\n")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ---------------------------------------------------------------------------
# Final results
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  OPTUNA SEARCH COMPLETE")
print("=" * 60)
print(f"  Best Value  (Val Accuracy):  {study.best_value:.4f}  ({study.best_value*100:.2f}%)")
print(f"  Best Params:")
for k, v in study.best_params.items():
    print(f"    {k}:  {v}")
print("=" * 60)

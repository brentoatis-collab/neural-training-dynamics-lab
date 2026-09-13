"""
dataset.py — Data loading, splitting, normalization, and label corruption
for the double-descent experiment.

Design constraints (pre-registered):
  - 80/20 stratified train/test split, split_seed=0 (fixed for entire experiment)
  - Normalization: z-score using training-set statistics only (not regularization;
    needed for numerical stability with lr=0.01 on raw pixel values 0–16)
  - Exactly floor(0.15 * n_train) = 215 training labels corrupted to a different class
  - Corruption indices drawn from corruption_seed=0 (fixed; same indices for all exp seeds)
  - Test labels are never modified
"""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_and_prepare(
    corruption_fraction: float,
    split_seed: int,
    corruption_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """
    Load sklearn digits, split, normalize, and corrupt training labels.

    Returns
    -------
    X_train : (n_train, 64) float32 tensor, z-score normalized
    X_test  : (n_test, 64)  float32 tensor, normalized with train statistics
    y_train : (n_train,) long tensor — corrupted labels
    y_test  : (n_test,)  long tensor — clean labels (never modified)
    corrupt_indices : 1-D array of training indices whose labels were changed
    metadata : dict with split/corruption counts for audit logging
    """
    digits = load_digits()
    X = digits.data.astype(np.float32)   # (1797, 64), values 0–16
    y = digits.target                     # (1797,), classes 0–9

    # --- Fixed stratified split ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=split_seed,
    )
    n_train = len(y_tr)
    n_test  = len(y_te)

    # --- Normalization (training statistics only) ---
    # Pixel intensities span 0–16; without normalization, gradient magnitudes
    # are ~16× larger than expected, destabilizing full-batch GD at lr=0.01.
    # This is input scaling, not weight regularization — it does not add a
    # penalty term and does not change which optima exist.
    mean = X_tr.mean(axis=0)             # (64,)
    std  = X_tr.std(axis=0)              # (64,)
    std[std == 0] = 1.0                  # 3 constant pixels in digits; avoid /0

    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std           # test set uses TRAIN mean/std

    # --- Label corruption (training set only) ---
    n_corrupt = int(np.floor(corruption_fraction * n_train))

    rng  = np.random.default_rng(corruption_seed)
    perm = rng.permutation(n_train)
    corrupt_indices = perm[:n_corrupt]   # fixed set; same for all experimental seeds

    y_tr_corrupted = y_tr.copy()
    for idx in corrupt_indices:
        true_label   = int(y_tr_corrupted[idx])
        other_classes = [c for c in range(10) if c != true_label]
        y_tr_corrupted[idx] = int(rng.choice(other_classes))

    # Sanity check: every corrupted index must carry a different label now.
    n_actually_changed = int((y_tr_corrupted != y_tr).sum())
    assert n_actually_changed == n_corrupt, (
        f"Expected {n_corrupt} label changes, got {n_actually_changed}. "
        "This is a corruption implementation bug."
    )

    metadata = {
        "n_train":           n_train,
        "n_test":            n_test,
        "n_corrupt":         n_corrupt,
        "corruption_fraction": corruption_fraction,
        "split_seed":        split_seed,
        "corruption_seed":   corruption_seed,
        "train_mean_norm":   float(mean.mean()),   # for audit; not used in training
        "train_std_norm":    float(std.mean()),
    }

    return (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_tr_corrupted, dtype=torch.long),
        torch.tensor(y_te, dtype=torch.long),
        corrupt_indices,
        metadata,
    )

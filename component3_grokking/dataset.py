"""
dataset.py — Modular addition dataset for Experiment 3 (Grokking).

Generates all p^2 ordered pairs (a, b) with labels (a + b) % p.
Random 50/50 split using np.random.default_rng(split_seed).permutation(p^2)
to match Power et al. (2022) convention (not stratified).

Returns:
  a_train, b_train, y_train  — training inputs and labels (torch.long)
  a_test,  b_test,  y_test   — test inputs and labels (torch.long)
  data_meta                  — dict with n_train, n_test, p, per_class counts
"""

import numpy as np
import torch


def make_dataset(p: int = 97, split_seed: int = 0, train_fraction: float = 0.5):
    """Build and split the modular addition dataset.

    All p^2 ordered pairs (a, b) are enumerated in row-major order
    (a=0,b=0), (a=0,b=1), ..., (a=p-1, b=p-1).  A random permutation
    determined solely by split_seed partitions them into train and test.

    Args:
        p:               Modulus. Labels are (a + b) % p.
        split_seed:      Seed for the split permutation.
        train_fraction:  Fraction of pairs assigned to training.

    Returns:
        a_train (LongTensor), b_train (LongTensor), y_train (LongTensor),
        a_test  (LongTensor), b_test  (LongTensor), y_test  (LongTensor),
        data_meta (dict)
    """
    n_total = p * p          # 9409 for p=97

    # Enumerate all ordered pairs in row-major order
    a_all = torch.arange(p).repeat_interleave(p)   # [0,0,...,0,1,1,...,p-1]
    b_all = torch.arange(p).repeat(p)              # [0,1,...,p-1, 0,1,...,p-1]
    y_all = (a_all + b_all) % p

    # Random split — matches Power et al. (2022), not stratified
    rng   = np.random.default_rng(split_seed)
    perm  = rng.permutation(n_total)
    n_train = int(n_total * train_fraction)
    train_idx = torch.from_numpy(perm[:n_train]).long()
    test_idx  = torch.from_numpy(perm[n_train:]).long()

    a_train = a_all[train_idx]
    b_train = b_all[train_idx]
    y_train = y_all[train_idx]

    a_test  = a_all[test_idx]
    b_test  = b_all[test_idx]
    y_test  = y_all[test_idx]

    # Per-class counts (train) — for pilot reporting
    train_label_counts = torch.bincount(y_train, minlength=p)
    per_class_min = int(train_label_counts.min().item())
    per_class_max = int(train_label_counts.max().item())

    data_meta = {
        "p":              p,
        "n_total":        n_total,
        "n_train":        len(a_train),
        "n_test":         len(a_test),
        "per_class_min":  per_class_min,
        "per_class_max":  per_class_max,
        "split_seed":     split_seed,
    }

    return a_train, b_train, y_train, a_test, b_test, y_test, data_meta

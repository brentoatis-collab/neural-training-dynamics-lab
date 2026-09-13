"""
sweep.py — Full 24-run double-descent sweep for Experiment 2.

Parameters (all from config.py, locked before sweep was authorized):
  Widths:     [4, 8, 16, 32, 64, 128, 256, 512]
  Seeds:      [42, 123, 777]
  Epochs:     10,000
  Optimizer:  full-batch SGD, lr=0.01, momentum=0, weight_decay=0
  Data:       n_train=1437, n_test=360, n_corrupt=215 (15%), split_seed=0,
              corruption_seed=0 (same corrupted indices for all seeds)

Outputs:
  component2_double_descent/results/double_descent_raw.csv
      One row per (seed, width) — 24 rows. Final-state values only.
  component2_double_descent/results/double_descent_curves.csv
      One row per (seed, width, epoch) — 24 × 20 = 480 rows.

H*_seed determination:
  For each seed, H*_seed = the smallest H where train_error == 0.0 (exactly zero
  misclassifications on the training set at the final epoch). Determined from
  measured training error, not from the analytic parameter-count crossover.
  If no width achieves this, H*_seed is None and is recorded as -1 in the CSV.

Run:
  python3 component2_double_descent/sweep.py
"""

import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXP2_CORRUPTION_FRACTION,
    EXP2_CORRUPTION_SEED,
    EXP2_D_IN,
    EXP2_D_OUT,
    EXP2_EPOCHS,
    EXP2_LOG_EVERY,
    EXP2_LR,
    EXP2_MOMENTUM,
    EXP2_SEEDS,
    EXP2_SPLIT_SEED,
    EXP2_WEIGHT_DECAY,
    EXP2_WIDTHS,
)
from component2_double_descent.dataset import load_and_prepare
from component2_double_descent.model import MLP


def classification_error(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds != labels).float().mean().item())


def n_misclassified(logits: torch.Tensor, labels: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    return int((preds != labels).sum().item())


def run_sweep() -> pd.DataFrame:
    # --- Load data once; identical for all 24 runs ---
    X_train, X_test, y_train, y_test, corrupt_indices, data_meta = load_and_prepare(
        corruption_fraction=EXP2_CORRUPTION_FRACTION,
        split_seed=EXP2_SPLIT_SEED,
        corruption_seed=EXP2_CORRUPTION_SEED,
    )
    n_train = data_meta["n_train"]

    print("=" * 66)
    print("EXPERIMENT 2 — FULL DOUBLE-DESCENT SWEEP")
    print("=" * 66)
    print(f"Widths: {EXP2_WIDTHS}")
    print(f"Seeds:  {EXP2_SEEDS}")
    print(f"Epochs: {EXP2_EPOCHS}  |  log every: {EXP2_LOG_EVERY}")
    print(f"lr: {EXP2_LR}  |  momentum: {EXP2_MOMENTUM}  |  weight_decay: {EXP2_WEIGHT_DECAY}")
    print(f"n_train: {n_train}  |  n_test: {data_meta['n_test']}  "
          f"|  n_corrupt: {data_meta['n_corrupt']}")
    print()

    criterion = nn.CrossEntropyLoss()

    raw_records   = []   # final state per (seed, width)
    curve_records = []   # per (seed, width, epoch)

    sweep_start = time.perf_counter()

    for seed in EXP2_SEEDS:
        print(f"=== Seed {seed} ===")
        seed_rows = []   # collect raw rows for this seed to compute H*_seed after

        for H in EXP2_WIDTHS:
            torch.manual_seed(seed)

            model     = MLP(EXP2_D_IN, H, EXP2_D_OUT)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=EXP2_LR,
                momentum=EXP2_MOMENTUM,
                weight_decay=EXP2_WEIGHT_DECAY,
                nesterov=False,
            )

            run_start = time.perf_counter()
            model.train()

            for epoch in range(1, EXP2_EPOCHS + 1):
                optimizer.zero_grad()
                logits = model(X_train)
                loss   = criterion(logits, y_train)
                loss.backward()
                optimizer.step()

                if epoch % EXP2_LOG_EVERY == 0:
                    with torch.no_grad():
                        tr_logits = model(X_train)
                        te_logits = model(X_test)
                        tr_loss   = criterion(tr_logits, y_train).item()
                        te_loss   = criterion(te_logits,  y_test).item()
                        tr_err    = classification_error(tr_logits, y_train)
                        te_err    = classification_error(te_logits,  y_test)

                    curve_records.append({
                        "seed":        seed,
                        "hidden_width": H,
                        "epoch":       epoch,
                        "train_loss":  tr_loss,
                        "train_error": tr_err,
                        "test_loss":   te_loss,
                        "test_error":  te_err,
                    })

            # Final-state evaluation
            run_secs = time.perf_counter() - run_start
            with torch.no_grad():
                tr_logits = model(X_train)
                te_logits = model(X_test)
                final_tr_loss  = criterion(tr_logits, y_train).item()
                final_te_loss  = criterion(te_logits,  y_test).item()
                final_tr_err   = classification_error(tr_logits, y_train)
                final_te_err   = classification_error(te_logits,  y_test)
                final_tr_wrong = n_misclassified(tr_logits, y_train)
                final_l2       = model.l2_norm()

            interpolating = (final_tr_wrong == 0)

            print(f"  H={H:3d}  n_params={model.n_params():5d}  "
                  f"train_err={final_tr_err:.4f}  test_err={final_te_err:.4f}  "
                  f"interpolating={interpolating}  ({run_secs:.1f}s)")

            row = {
                "seed":               seed,
                "hidden_width":       H,
                "n_params":           model.n_params(),
                "train_error":        final_tr_err,
                "test_error":         final_te_err,
                "train_loss":         final_tr_loss,
                "test_loss":          final_te_loss,
                "epochs_run":         EXP2_EPOCHS,
                "corruption_fraction": EXP2_CORRUPTION_FRACTION,
                "n_corrupted":        data_meta["n_corrupt"],
                "interpolating":      interpolating,
                "l2_weight_norm":     final_l2,
                # H_star_seed filled after all widths for this seed are done
            }
            seed_rows.append(row)

        # --- Determine H*_seed from training error (not parameter count) ---
        H_star = -1   # -1 = undefined (no width achieved zero training error)
        for row in seed_rows:
            if row["interpolating"]:
                H_star = row["hidden_width"]
                break   # smallest H with train_error == 0

        for row in seed_rows:
            row["H_star_seed"] = H_star

        raw_records.extend(seed_rows)
        print(f"  --> H*_seed(seed={seed}) = {H_star if H_star != -1 else 'undefined'}")
        print()

    total_secs = time.perf_counter() - sweep_start
    print(f"Sweep complete in {total_secs:.1f}s")

    # --- Save CSVs ---
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    raw_path    = os.path.join(results_dir, "double_descent_raw.csv")
    curves_path = os.path.join(results_dir, "double_descent_curves.csv")

    df_raw    = pd.DataFrame(raw_records)
    df_curves = pd.DataFrame(curve_records)

    df_raw.to_csv(raw_path,    index=False)
    df_curves.to_csv(curves_path, index=False)

    # --- Summary across seeds ---
    print()
    print("FINAL TRAINING ERROR BY SEED AND WIDTH:")
    pivot_train = df_raw.pivot(index="hidden_width", columns="seed", values="train_error")
    pivot_train.columns = [f"seed_{s}_train_err" for s in pivot_train.columns]
    print(pivot_train.to_string())

    print()
    print("FINAL TEST ERROR BY SEED AND WIDTH:")
    pivot_test = df_raw.pivot(index="hidden_width", columns="seed", values="test_error")
    pivot_test.columns = [f"seed_{s}_test_err" for s in pivot_test.columns]
    print(pivot_test.to_string())

    print()
    print("INTERPOLATION THRESHOLD (H*_seed) PER SEED:")
    for seed in EXP2_SEEDS:
        h_star = df_raw[df_raw["seed"] == seed]["H_star_seed"].iloc[0]
        label  = str(h_star) if h_star != -1 else "undefined"
        print(f"  seed={seed}: H*_seed = {label}")

    print()
    print(f"Raw results saved to:    {raw_path}")
    print(f"Curve results saved to:  {curves_path}")

    return df_raw


if __name__ == "__main__":
    run_sweep()

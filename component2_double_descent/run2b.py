"""
run2b.py — Experiment 2 Run 2B: mini-batch reproduction attempt.

Design (pre-registered in LEDGER.md Deviation Log):
  batch_size : 32
  epochs     : 10,000
  steps/epoch: ceil(1437/32) = 45
  total steps: ~450,000 per (seed, width) combination
  Purpose    : test whether gradient noise + adequate step budget achieves
               interpolation and produces a double-descent test-error curve.

Logging: every 500 epochs (20 checkpoints per run). Both epoch and global_step recorded.

Outputs:
  component2_double_descent/results/double_descent_run2b_raw.csv    (24 rows)
  component2_double_descent/results/double_descent_run2b_curves.csv (480 rows)

All parameters identical to Run 1 and Run 2A except step count:
  Widths [4,8,16,32,64,128,256,512], seeds [42,123,777], split_seed=0,
  corruption_seed=0, same 215 corrupted indices, lr=0.01, momentum=0,
  weight_decay=0, architecture unchanged, evidence criteria unchanged.

Run:
  python3 component2_double_descent/run2b.py
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
    EXP2_LR,
    EXP2_MOMENTUM,
    EXP2_RUN2_BATCH_SIZE,
    EXP2_RUN2B_EPOCHS,
    EXP2_RUN2B_LOG_EVERY,
    EXP2_SEEDS,
    EXP2_SPLIT_SEED,
    EXP2_WEIGHT_DECAY,
    EXP2_WIDTHS,
)
from component2_double_descent.dataset import load_and_prepare
from component2_double_descent.model import MLP


def classification_error(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) != labels).float().mean().item())


def n_misclassified(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) != labels).sum().item())


def run_run2b() -> pd.DataFrame:
    X_train, X_test, y_train, y_test, corrupt_indices, data_meta = load_and_prepare(
        corruption_fraction=EXP2_CORRUPTION_FRACTION,
        split_seed=EXP2_SPLIT_SEED,
        corruption_seed=EXP2_CORRUPTION_SEED,
    )
    n_train         = data_meta["n_train"]
    steps_per_epoch = -(-n_train // EXP2_RUN2_BATCH_SIZE)   # ceil(1437/32) = 45
    total_steps     = EXP2_RUN2B_EPOCHS * steps_per_epoch

    print("=" * 66)
    print("EXPERIMENT 2 — RUN 2B: MINI-BATCH REPRODUCTION ATTEMPT")
    print("=" * 66)
    print(f"batch_size={EXP2_RUN2_BATCH_SIZE}  steps_per_epoch={steps_per_epoch}  "
          f"epochs={EXP2_RUN2B_EPOCHS}  total_steps={total_steps:,}")
    print(f"lr={EXP2_LR}  momentum={EXP2_MOMENTUM}  weight_decay={EXP2_WEIGHT_DECAY}")
    print(f"n_train={n_train}  n_test={data_meta['n_test']}  "
          f"n_corrupt={data_meta['n_corrupt']}")
    print(f"Log every {EXP2_RUN2B_LOG_EVERY} epochs  |  Widths: {EXP2_WIDTHS}")
    print(f"Seeds: {EXP2_SEEDS}")
    print()

    criterion     = nn.CrossEntropyLoss()
    raw_records   = []
    curve_records = []
    sweep_start   = time.perf_counter()

    for seed in EXP2_SEEDS:
        print(f"=== Seed {seed} ===")
        seed_rows = []

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

            run_start   = time.perf_counter()
            global_step = 0
            model.train()

            for epoch in range(1, EXP2_RUN2B_EPOCHS + 1):
                perm = torch.randperm(n_train)

                for start in range(0, n_train, EXP2_RUN2_BATCH_SIZE):
                    idx    = perm[start : start + EXP2_RUN2_BATCH_SIZE]
                    xb, yb = X_train[idx], y_train[idx]
                    optimizer.zero_grad()
                    criterion(model(xb), yb).backward()
                    optimizer.step()
                    global_step += 1

                if epoch % EXP2_RUN2B_LOG_EVERY == 0:
                    with torch.no_grad():
                        tr_logits = model(X_train)
                        te_logits = model(X_test)
                        tr_loss   = criterion(tr_logits, y_train).item()
                        te_loss   = criterion(te_logits,  y_test).item()
                        tr_err    = classification_error(tr_logits, y_train)
                        te_err    = classification_error(te_logits,  y_test)

                    curve_records.append({
                        "seed":         seed,
                        "hidden_width": H,
                        "epoch":        epoch,
                        "global_step":  global_step,
                        "train_loss":   tr_loss,
                        "train_error":  tr_err,
                        "test_loss":    te_loss,
                        "test_error":   te_err,
                    })

            run_secs = time.perf_counter() - run_start

            with torch.no_grad():
                tr_logits      = model(X_train)
                te_logits      = model(X_test)
                final_tr_err   = classification_error(tr_logits, y_train)
                final_te_err   = classification_error(te_logits,  y_test)
                final_tr_loss  = criterion(tr_logits, y_train).item()
                final_te_loss  = criterion(te_logits,  y_test).item()
                final_tr_wrong = n_misclassified(tr_logits, y_train)
                final_l2       = model.l2_norm()

            interpolating = (final_tr_wrong == 0)

            print(f"  H={H:3d}  steps={global_step:,}  "
                  f"train_err={final_tr_err:.4f}  test_err={final_te_err:.4f}  "
                  f"interpolating={interpolating}  ({run_secs:.1f}s)")

            row = {
                "seed":                seed,
                "hidden_width":        H,
                "n_params":            model.n_params(),
                "total_steps":         global_step,
                "train_error":         final_tr_err,
                "test_error":          final_te_err,
                "train_loss":          final_tr_loss,
                "test_loss":           final_te_loss,
                "epochs_run":          EXP2_RUN2B_EPOCHS,
                "corruption_fraction": EXP2_CORRUPTION_FRACTION,
                "n_corrupted":         data_meta["n_corrupt"],
                "interpolating":       interpolating,
                "l2_weight_norm":      final_l2,
            }
            seed_rows.append(row)

        H_star = -1
        for row in seed_rows:
            if row["interpolating"]:
                H_star = row["hidden_width"]
                break

        for row in seed_rows:
            row["H_star_seed"] = H_star

        raw_records.extend(seed_rows)
        print(f"  --> H*_seed(seed={seed}) = {H_star if H_star != -1 else 'undefined'}")
        print()

    total_secs = time.perf_counter() - sweep_start
    print(f"Run 2B complete in {total_secs:.1f}s")

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    raw_path    = os.path.join(results_dir, "double_descent_run2b_raw.csv")
    curves_path = os.path.join(results_dir, "double_descent_run2b_curves.csv")

    df_raw    = pd.DataFrame(raw_records)
    df_curves = pd.DataFrame(curve_records)
    df_raw.to_csv(raw_path,    index=False)
    df_curves.to_csv(curves_path, index=False)

    print()
    print("FINAL TRAINING ERROR (Run 2B):")
    pivot_tr = df_raw.pivot(index="hidden_width", columns="seed", values="train_error")
    pivot_tr.columns = [f"seed_{s}_train_err" for s in pivot_tr.columns]
    print(pivot_tr.to_string())

    print()
    print("FINAL TEST ERROR (Run 2B):")
    pivot_te = df_raw.pivot(index="hidden_width", columns="seed", values="test_error")
    pivot_te.columns = [f"seed_{s}_test_err" for s in pivot_te.columns]
    print(pivot_te.to_string())

    print()
    print("H*_seed per seed:")
    for seed in EXP2_SEEDS:
        h = df_raw[df_raw["seed"] == seed]["H_star_seed"].iloc[0]
        print(f"  seed={seed}: {h if h != -1 else 'undefined'}")

    print()
    print(f"Raw:    {raw_path}")
    print(f"Curves: {curves_path}")
    return df_raw


if __name__ == "__main__":
    run_run2b()

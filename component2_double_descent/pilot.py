"""
pilot.py — Pre-registered convergence and runtime pilot for Experiment 2.

Specification (pre-registered in LEDGER.md):
  - Seed:    42
  - Widths:  [16, 64, 512]
  - Epochs:  2,000
  - Log:     every 100 epochs
  - Data:    same fixed split and 15% corruption as the full sweep

Trigger 1 (lr adjustment):
  Any 100-epoch block where train_loss increases by more than 5% relative to
  the previous logged value. If fired: lr must be replaced with 0.001 before
  the full sweep. Fires per-width; one firing in any width triggers the change.

Trigger 2 (epoch budget adjustment):
  H=512 final train_loss at epoch 2000 > 1.5. If fired: epoch budget is raised
  to 30,000 for the full sweep.

Permitted changes if a trigger fires:
  Trigger 1 only → lr: 0.01 → 0.001
  Trigger 2 only → epochs: 10,000 → 30,000
  Both           → both changes apply

Nothing else may be changed based on pilot results.

Outputs:
  component2_double_descent/results/pilot_curves.csv   (per-epoch log)
  component2_double_descent/results/pilot_summary.csv  (final state per width)

Run this file directly:
    python3 component2_double_descent/pilot.py
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
    EXP2_PILOT_EPOCHS,
    EXP2_PILOT_LOG_EVERY,
    EXP2_PILOT_SEED,
    EXP2_PILOT_TRIGGER1_THRESHOLD,
    EXP2_PILOT_TRIGGER2_THRESHOLD,
    EXP2_PILOT_WIDTHS,
    EXP2_SPLIT_SEED,
    EXP2_WEIGHT_DECAY,
)
from component2_double_descent.dataset import load_and_prepare
from component2_double_descent.model import MLP


def classification_error(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of misclassified samples."""
    preds = logits.argmax(dim=1)
    return float((preds != labels).float().mean().item())


def run_pilot() -> dict:
    """
    Execute the pre-registered pilot and return a summary dict with trigger status.
    """
    # --- Load data (once; same for all widths) ---
    X_train, X_test, y_train, y_test, corrupt_indices, data_meta = load_and_prepare(
        corruption_fraction=EXP2_CORRUPTION_FRACTION,
        split_seed=EXP2_SPLIT_SEED,
        corruption_seed=EXP2_CORRUPTION_SEED,
    )

    print("=" * 62)
    print("EXPERIMENT 2 — CONVERGENCE / RUNTIME PILOT")
    print("=" * 62)
    print(f"Seed: {EXP2_PILOT_SEED}  |  Widths: {EXP2_PILOT_WIDTHS}")
    print(f"Epochs: {EXP2_PILOT_EPOCHS}  |  Log every: {EXP2_PILOT_LOG_EVERY}")
    print(f"lr: {EXP2_LR}  |  momentum: {EXP2_MOMENTUM}  |  weight_decay: {EXP2_WEIGHT_DECAY}")
    print(f"n_train: {data_meta['n_train']}  |  n_test: {data_meta['n_test']}  "
          f"|  n_corrupt: {data_meta['n_corrupt']}")
    print()

    criterion = nn.CrossEntropyLoss()

    curve_records   = []   # one row per (width, logged epoch)
    summary_records = []   # one row per width

    # Global trigger state
    trigger1_fired   = False
    trigger1_details = []   # list of (width, epoch, prev_loss, curr_loss) if fired

    final_loss_512 = None  # populated when H=512 run completes

    for H in EXP2_PILOT_WIDTHS:
        print(f"--- H = {H:3d}  (n_params = {75*H + 10}) ---")

        # Fresh seed for weight initialization
        torch.manual_seed(EXP2_PILOT_SEED)

        model     = MLP(EXP2_D_IN, H, EXP2_D_OUT)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=EXP2_LR,
            momentum=EXP2_MOMENTUM,
            weight_decay=EXP2_WEIGHT_DECAY,
            nesterov=False,
        )

        width_start_time = time.perf_counter()
        prev_logged_loss = None
        trigger1_this_width = False
        block_start = width_start_time

        model.train()

        for epoch in range(1, EXP2_PILOT_EPOCHS + 1):
            # --- Full-batch forward + backward ---
            optimizer.zero_grad()
            logits = model(X_train)
            loss   = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            if epoch % EXP2_PILOT_LOG_EVERY == 0:
                block_end = time.perf_counter()

                # Evaluate (no grad needed)
                with torch.no_grad():
                    train_logits = model(X_train)
                    test_logits  = model(X_test)
                    train_loss   = criterion(train_logits, y_train).item()
                    test_loss    = criterion(test_logits,  y_test).item()
                    train_err    = classification_error(train_logits, y_train)
                    test_err     = classification_error(test_logits,  y_test)

                elapsed     = block_end - width_start_time
                block_secs  = block_end - block_start
                block_start = block_end

                # --- Trigger 1 check ---
                if prev_logged_loss is not None:
                    relative_increase = (train_loss - prev_logged_loss) / (prev_logged_loss + 1e-12)
                    if relative_increase > EXP2_PILOT_TRIGGER1_THRESHOLD:
                        trigger1_this_width = True
                        trigger1_fired      = True
                        trigger1_details.append({
                            "width": H, "epoch": epoch,
                            "prev_loss": prev_logged_loss,
                            "curr_loss": train_loss,
                            "relative_increase": relative_increase,
                        })

                prev_logged_loss = train_loss

                curve_records.append({
                    "width":           H,
                    "epoch":           epoch,
                    "train_loss":      train_loss,
                    "train_error":     train_err,
                    "test_loss":       test_loss,
                    "test_error":      test_err,
                    "elapsed_seconds": round(elapsed, 3),
                    "block_seconds":   round(block_secs, 3),
                })

                if epoch == EXP2_PILOT_EPOCHS:
                    # Print final-epoch row
                    print(f"  epoch {epoch:5d}  train_loss={train_loss:.4f}  "
                          f"train_err={train_err:.4f}  test_err={test_err:.4f}  "
                          f"elapsed={elapsed:.1f}s")

        # Record first and last logged values for summary
        width_curves = [r for r in curve_records if r["width"] == H]
        first_row    = width_curves[0]
        last_row     = width_curves[-1]
        total_secs   = last_row["elapsed_seconds"]

        if H == 512:
            final_loss_512 = last_row["train_loss"]

        summary_records.append({
            "width":              H,
            "n_params":           75 * H + 10,
            "start_train_loss":   first_row["train_loss"],
            "final_train_loss":   last_row["train_loss"],
            "final_train_error":  last_row["train_error"],
            "final_test_error":   last_row["test_error"],
            "total_seconds":      round(total_secs, 2),
            "trigger1_fired":     trigger1_this_width,
        })

    # --- Trigger 2 check ---
    trigger2_fired = (final_loss_512 is not None and
                      final_loss_512 > EXP2_PILOT_TRIGGER2_THRESHOLD)

    # --- Save CSVs ---
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    curves_path  = os.path.join(results_dir, "pilot_curves.csv")
    summary_path = os.path.join(results_dir, "pilot_summary.csv")

    pd.DataFrame(curve_records).to_csv(curves_path,  index=False)
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)

    # --- Print trigger summary ---
    print()
    print("=" * 62)
    print("PILOT TRIGGER REPORT")
    print("=" * 62)

    if trigger1_fired:
        print("TRIGGER 1 (lr adjustment): FIRED")
        for d in trigger1_details:
            print(f"  Width H={d['width']}, epoch {d['epoch']}: "
                  f"loss {d['prev_loss']:.4f} → {d['curr_loss']:.4f} "
                  f"(+{d['relative_increase']*100:.1f}%)")
        print("  Permitted action: replace lr=0.01 with lr=0.001 for the full sweep.")
    else:
        print("TRIGGER 1 (lr adjustment):    did not fire")
        print(f"  Training loss was monotonically non-increasing for all pilot widths.")

    print()

    if trigger2_fired:
        print("TRIGGER 2 (epoch budget):  FIRED")
        print(f"  H=512 final train_loss = {final_loss_512:.4f} > threshold {EXP2_PILOT_TRIGGER2_THRESHOLD}")
        print("  Permitted action: increase epoch budget from 10,000 to 30,000.")
    else:
        print("TRIGGER 2 (epoch budget):     did not fire")
        print(f"  H=512 final train_loss = {final_loss_512:.4f} <= threshold {EXP2_PILOT_TRIGGER2_THRESHOLD}")

    print()
    print("Summary table:")
    df_summary = pd.DataFrame(summary_records)
    print(df_summary.to_string(index=False))
    print()
    print(f"Curves saved to:  {curves_path}")
    print(f"Summary saved to: {summary_path}")

    return {
        "trigger1_fired":   trigger1_fired,
        "trigger1_details": trigger1_details,
        "trigger2_fired":   trigger2_fired,
        "final_loss_512":   final_loss_512,
        "summary":          summary_records,
    }


if __name__ == "__main__":
    run_pilot()

"""
pilot.py — Experiment 3 pilot: 500 full-batch AdamW steps, seed=42.

Purpose:
  1. Verify numerical stability (gradients finite and non-zero).
  2. Measure throughput (seconds/step) to project full-experiment runtime.
  3. Check stability trigger: if any 50-step block shows relative loss increase
     > EXP3_PILOT_STABILITY_THRESHOLD (5%), the trigger fires and lr must be
     reduced to 1e-4 before the full experiment.

Logging: every EXP3_PILOT_LOG_EVERY (10) steps.

Outputs:
  component3_grokking/results/grokking_pilot_curves.csv   (50 rows)
  component3_grokking/results/grokking_pilot_summary.csv  (1 row)

Run:
  python3 component3_grokking/pilot.py
"""

import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXP3_BETAS,
    EXP3_D_FF,
    EXP3_D_MODEL,
    EXP3_LR,
    EXP3_N_HEADS,
    EXP3_P,
    EXP3_PILOT_LOG_EVERY,
    EXP3_PILOT_SEED,
    EXP3_PILOT_STABILITY_THRESHOLD,
    EXP3_PILOT_STEPS,
    EXP3_SPLIT_SEED,
    EXP3_TRAIN_FRACTION,
    EXP3_WEIGHT_DECAY,
)
from component3_grokking.dataset import make_dataset
from component3_grokking.model import GrokTransformer


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def run_pilot():
    # ------------------------------------------------------------------ data
    a_train, b_train, y_train, a_test, b_test, y_test, data_meta = make_dataset(
        p=EXP3_P,
        split_seed=EXP3_SPLIT_SEED,
        train_fraction=EXP3_TRAIN_FRACTION,
    )

    # ------------------------------------------------------------------ model
    torch.manual_seed(EXP3_PILOT_SEED)
    model     = GrokTransformer(p=EXP3_P, d_model=EXP3_D_MODEL,
                                n_heads=EXP3_N_HEADS, d_ff=EXP3_D_FF)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXP3_LR,
        betas=EXP3_BETAS,
        weight_decay=EXP3_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    print("=" * 66)
    print("EXPERIMENT 3 — PILOT (500 steps, seed=42)")
    print("=" * 66)
    print(f"p={EXP3_P}  d_model={EXP3_D_MODEL}  n_heads={EXP3_N_HEADS}  d_ff={EXP3_D_FF}")
    print(f"lr={EXP3_LR}  betas={EXP3_BETAS}  weight_decay={EXP3_WEIGHT_DECAY}")
    print(f"n_params={model.n_params():,}")
    print(f"n_train={data_meta['n_train']}  n_test={data_meta['n_test']}")
    print(f"per_class count range (train): [{data_meta['per_class_min']}, {data_meta['per_class_max']}]")
    print(f"pilot_steps={EXP3_PILOT_STEPS}  log_every={EXP3_PILOT_LOG_EVERY}")
    print()

    # ------------------------------------------------------------------ training loop
    curve_records  = []
    gradients_ok   = True    # finite and non-zero throughout
    trigger_fired  = False
    prev_block_loss = None   # last logged loss at end of previous 50-step block

    step_times = []
    model.train()

    for step in range(1, EXP3_PILOT_STEPS + 1):
        t0 = time.perf_counter()

        optimizer.zero_grad()
        logits = model(a_train, b_train)
        loss   = criterion(logits, y_train)
        loss.backward()

        # Gradient check
        for p_tensor in model.parameters():
            if p_tensor.grad is not None:
                if not torch.isfinite(p_tensor.grad).all():
                    gradients_ok = False
                if p_tensor.grad.abs().max() == 0:
                    gradients_ok = False

        optimizer.step()
        step_times.append(time.perf_counter() - t0)

        if step % EXP3_PILOT_LOG_EVERY == 0:
            with torch.no_grad():
                tr_loss = criterion(model(a_train, b_train), y_train).item()
                te_loss = criterion(model(a_test,  b_test),  y_test ).item()
                tr_acc  = accuracy(model(a_train, b_train), y_train)
                te_acc  = accuracy(model(a_test,  b_test),  y_test )
                l2      = model.l2_norm()

            curve_records.append({
                "step":       step,
                "train_loss": tr_loss,
                "train_acc":  tr_acc,
                "test_loss":  te_loss,
                "test_acc":   te_acc,
                "l2_norm":    l2,
            })

            # Stability trigger check (per 50-step block)
            if step % 50 == 0:
                if prev_block_loss is not None and prev_block_loss > 0:
                    rel_increase = (tr_loss - prev_block_loss) / prev_block_loss
                    if rel_increase > EXP3_PILOT_STABILITY_THRESHOLD:
                        trigger_fired = True
                prev_block_loss = tr_loss

            print(f"  step={step:4d}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}  l2={l2:.3f}")

    # ------------------------------------------------------------------ final evaluation
    with torch.no_grad():
        tr_logits_f = model(a_train, b_train)
        te_logits_f = model(a_test,  b_test)
        final_tr_loss = criterion(tr_logits_f, y_train).item()
        final_te_loss = criterion(te_logits_f, y_test).item()
        final_tr_acc  = accuracy(tr_logits_f, y_train)
        final_te_acc  = accuracy(te_logits_f, y_test)
        final_l2      = model.l2_norm()

    # ------------------------------------------------------------------ timing
    secs_per_step     = sum(step_times) / len(step_times)
    projected_per_seed = secs_per_step * 50_000
    projected_3seeds   = projected_per_seed * 3

    # ------------------------------------------------------------------ starting values (step 10)
    start_record = curve_records[0] if curve_records else {}

    # ------------------------------------------------------------------ summary
    print()
    print("=" * 66)
    print("PILOT SUMMARY")
    print("=" * 66)
    print(f"n_params:             {model.n_params():,}")
    print(f"n_train:              {data_meta['n_train']}")
    print(f"n_test:               {data_meta['n_test']}")
    print(f"per_class range(train):[{data_meta['per_class_min']}, {data_meta['per_class_max']}]")
    print()
    print(f"Starting (step {EXP3_PILOT_LOG_EVERY}):")
    print(f"  train_loss={start_record.get('train_loss', 'N/A'):.4f}  "
          f"train_acc={start_record.get('train_acc', 'N/A'):.4f}")
    print(f"  test_loss={start_record.get('test_loss', 'N/A'):.4f}   "
          f"test_acc={start_record.get('test_acc', 'N/A'):.4f}")
    print(f"  l2_norm={start_record.get('l2_norm', 'N/A'):.4f}")
    print()
    print(f"Ending (step {EXP3_PILOT_STEPS}):")
    print(f"  train_loss={final_tr_loss:.4f}  train_acc={final_tr_acc:.4f}")
    print(f"  test_loss={final_te_loss:.4f}   test_acc={final_te_acc:.4f}")
    print(f"  l2_norm={final_l2:.4f}")
    print()
    print(f"Gradients finite and non-zero throughout: {gradients_ok}")
    print(f"Stability trigger fired:                  {trigger_fired}")
    print()
    print(f"Measured seconds/step:          {secs_per_step:.4f} s")
    print(f"Projected time / seed (50k):    {projected_per_seed:.1f} s  ({projected_per_seed/60:.1f} min)")
    print(f"Projected time / 3 seeds:       {projected_3seeds:.1f} s  ({projected_3seeds/60:.1f} min)")

    # ------------------------------------------------------------------ save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    curves_path  = os.path.join(results_dir, "grokking_pilot_curves.csv")
    summary_path = os.path.join(results_dir, "grokking_pilot_summary.csv")

    df_curves = pd.DataFrame(curve_records)
    df_curves.to_csv(curves_path, index=False)

    summary = {
        "n_params":           model.n_params(),
        "n_train":            data_meta["n_train"],
        "n_test":             data_meta["n_test"],
        "per_class_min":      data_meta["per_class_min"],
        "per_class_max":      data_meta["per_class_max"],
        "start_step":         EXP3_PILOT_LOG_EVERY,
        "start_train_loss":   start_record.get("train_loss"),
        "start_train_acc":    start_record.get("train_acc"),
        "start_test_loss":    start_record.get("test_loss"),
        "start_test_acc":     start_record.get("test_acc"),
        "start_l2_norm":      start_record.get("l2_norm"),
        "end_train_loss":     final_tr_loss,
        "end_train_acc":      final_tr_acc,
        "end_test_loss":      final_te_loss,
        "end_test_acc":       final_te_acc,
        "end_l2_norm":        final_l2,
        "gradients_ok":       gradients_ok,
        "trigger_fired":      trigger_fired,
        "secs_per_step":      secs_per_step,
        "projected_per_seed_secs": projected_per_seed,
        "projected_3seeds_secs":   projected_3seeds,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print()
    print(f"Curves:  {curves_path}")
    print(f"Summary: {summary_path}")

    return summary


if __name__ == "__main__":
    run_pilot()

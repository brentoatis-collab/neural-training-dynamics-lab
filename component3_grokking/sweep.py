"""
sweep.py — Experiment 3 full sweep: grokking on modular addition mod 97.

Parameters locked at commit 125314c. Three seeds: [42, 123, 777].
Each seed: 50,000 full-batch AdamW steps.

Pre-registered criteria (evaluated mechanically after all runs complete):
  T_mem: earliest checkpoint with train_acc >= 0.99 for 5 consecutive logged checkpoints
  T_grok: earliest checkpoint with test_acc >= 0.95 for 5 consecutive logged checkpoints
  Signal A = T_mem defined AND T_grok defined AND T_grok - T_mem >= 2000
  Signal B = >=10 consecutive checkpoints strictly between T_mem and T_grok with test_acc < 0.30
  behavioral_grokking = Signal A AND Signal B
  REPRODUCED = behavioral_grokking for >=2 seeds

Outputs:
  component3_grokking/results/grokking_curves_seed{seed}.csv  (500 rows)
  component3_grokking/results/grokking_curves_all.csv         (1500 rows)
  component3_grokking/results/grokking_summary.csv            (3 rows, per-seed verdicts)
  component3_grokking/results/grokking_aggregate.csv          (1 row, aggregate verdict)
  component3_grokking/results/grokking_curves.png             (3-panel figure)

Run:
  python3 component3_grokking/sweep.py
"""

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXP3_BETAS,
    EXP3_BUDGET,
    EXP3_D_FF,
    EXP3_D_MODEL,
    EXP3_GROK_GAP_MIN,
    EXP3_LOG_EVERY,
    EXP3_LR,
    EXP3_N_HEADS,
    EXP3_P,
    EXP3_SEEDS,
    EXP3_SIGNAL_B_CHECKPOINTS,
    EXP3_SIGNAL_B_LOW_ACC,
    EXP3_SPLIT_SEED,
    EXP3_STABILITY_CHECKPOINTS,
    EXP3_T_GROK_THRESHOLD,
    EXP3_T_MEM_THRESHOLD,
    EXP3_TRAIN_FRACTION,
    EXP3_WEIGHT_DECAY,
)
from component3_grokking.dataset import make_dataset
from component3_grokking.model import GrokTransformer


# ---------------------------------------------------------------------------
# Criterion helpers
# ---------------------------------------------------------------------------

def find_T(steps, values, threshold, n_consecutive):
    """
    Return the earliest step in `steps` where `values` >= threshold
    for n_consecutive consecutive entries starting at that index.
    Returns None if no such point exists.
    """
    steps  = list(steps)
    values = list(values)
    n = len(steps)
    for i in range(n - n_consecutive + 1):
        if all(values[i + j] >= threshold for j in range(n_consecutive)):
            return steps[i]
    return None


def find_signal_B(steps, test_accs, t_mem, t_grok,
                  low_acc_threshold, n_consecutive):
    """
    Check whether there are >= n_consecutive consecutive logged checkpoints
    strictly between t_mem and t_grok with test_acc < low_acc_threshold.
    Returns (True/False, count_of_max_consecutive_run).
    """
    if t_mem is None or t_grok is None:
        return False, 0

    # Checkpoints strictly between t_mem and t_grok
    between = [(s, a) for s, a in zip(steps, test_accs)
               if s > t_mem and s < t_grok]

    if len(between) < n_consecutive:
        return False, 0

    # Longest run of consecutive entries with test_acc < threshold
    max_run = 0
    current_run = 0
    for _, acc in between:
        if acc < low_acc_threshold:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return (max_run >= n_consecutive), max_run


def accuracy(logits, labels):
    return float((logits.argmax(dim=1) == labels).float().mean().item())


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    # --- data (same split for all seeds) ---
    a_train, b_train, y_train, a_test, b_test, y_test, data_meta = make_dataset(
        p=EXP3_P,
        split_seed=EXP3_SPLIT_SEED,
        train_fraction=EXP3_TRAIN_FRACTION,
    )
    criterion = nn.CrossEntropyLoss()

    print("=" * 70)
    print("EXPERIMENT 3 — FULL GROKKING SWEEP")
    print("=" * 70)
    print(f"p={EXP3_P}  d_model={EXP3_D_MODEL}  n_heads={EXP3_N_HEADS}  d_ff={EXP3_D_FF}")
    print(f"lr={EXP3_LR}  betas={EXP3_BETAS}  weight_decay={EXP3_WEIGHT_DECAY}")
    print(f"budget={EXP3_BUDGET:,}  log_every={EXP3_LOG_EVERY}  seeds={EXP3_SEEDS}")
    print(f"n_train={data_meta['n_train']}  n_test={data_meta['n_test']}")
    print(f"per_class range (train): [{data_meta['per_class_min']}, {data_meta['per_class_max']}]")
    print()

    all_curve_records = []
    summary_rows = []
    sweep_start = time.perf_counter()

    for seed in EXP3_SEEDS:
        print(f"{'=' * 70}")
        print(f"SEED {seed}")
        print(f"{'=' * 70}")

        torch.manual_seed(seed)
        model = GrokTransformer(
            p=EXP3_P, d_model=EXP3_D_MODEL,
            n_heads=EXP3_N_HEADS, d_ff=EXP3_D_FF,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=EXP3_LR,
            betas=EXP3_BETAS,
            weight_decay=EXP3_WEIGHT_DECAY,
        )

        curve_records = []
        seed_start    = time.perf_counter()
        model.train()

        for step in range(1, EXP3_BUDGET + 1):
            optimizer.zero_grad()
            logits = model(a_train, b_train)
            loss   = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            if step % EXP3_LOG_EVERY == 0:
                with torch.no_grad():
                    tr_logits = model(a_train, b_train)
                    te_logits = model(a_test,  b_test)
                    tr_loss   = criterion(tr_logits, y_train).item()
                    te_loss   = criterion(te_logits, y_test ).item()
                    tr_acc    = accuracy(tr_logits, y_train)
                    te_acc    = accuracy(te_logits, y_test )
                    l2        = model.l2_norm()

                elapsed = time.perf_counter() - seed_start
                rec = {
                    "seed":       seed,
                    "step":       step,
                    "train_loss": tr_loss,
                    "train_acc":  tr_acc,
                    "test_loss":  te_loss,
                    "test_acc":   te_acc,
                    "l2_norm":    l2,
                    "elapsed_s":  elapsed,
                }
                curve_records.append(rec)
                all_curve_records.append(rec)

                if step % 5000 == 0:
                    print(f"  step={step:6,}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                          f"te_loss={te_loss:.4f}  te_acc={te_acc:.4f}  l2={l2:.3f}  "
                          f"({elapsed:.0f}s)")

        seed_elapsed = time.perf_counter() - seed_start
        print(f"  Seed {seed} complete in {seed_elapsed:.1f}s")

        # --- Save per-seed curves ---
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        df_seed = pd.DataFrame(curve_records)
        df_seed.to_csv(
            os.path.join(results_dir, f"grokking_curves_seed{seed}.csv"),
            index=False,
        )

        # --- Apply pre-registered criteria mechanically ---
        steps     = df_seed["step"].tolist()
        tr_accs   = df_seed["train_acc"].tolist()
        te_accs   = df_seed["test_acc"].tolist()
        l2_vals   = df_seed["l2_norm"].tolist()

        T_mem  = find_T(steps, tr_accs, EXP3_T_MEM_THRESHOLD,  EXP3_STABILITY_CHECKPOINTS)
        T_grok = find_T(steps, te_accs, EXP3_T_GROK_THRESHOLD, EXP3_STABILITY_CHECKPOINTS)

        signal_A = (
            T_mem  is not None and
            T_grok is not None and
            (T_grok - T_mem) >= EXP3_GROK_GAP_MIN
        )

        signal_B_fires, signal_B_max_run = find_signal_B(
            steps, te_accs, T_mem, T_grok,
            EXP3_SIGNAL_B_LOW_ACC, EXP3_SIGNAL_B_CHECKPOINTS,
        )

        behavioral_grokking = signal_A and signal_B_fires

        # L2 at key points
        l2_at_init = l2_vals[0] if l2_vals else None
        l2_at_T_mem = None
        if T_mem is not None:
            idx = steps.index(T_mem)
            l2_at_T_mem = l2_vals[idx]
        l2_at_end = l2_vals[-1] if l2_vals else None

        final = curve_records[-1]

        print()
        print(f"  [MEASURED] T_mem  = {T_mem}")
        print(f"  [MEASURED] T_grok = {T_grok}")
        if T_mem is not None and T_grok is not None:
            print(f"  [MEASURED] T_grok - T_mem = {T_grok - T_mem} steps")
        print(f"  [OPDEF]    Signal A (delayed generalization): {signal_A}")
        print(f"  [OPDEF]    Signal B (plateau >=10 consec checkpoints <0.30): "
              f"{signal_B_fires}  (max_run={signal_B_max_run})")
        print(f"  [OPDEF]    behavioral_grokking: {behavioral_grokking}")
        print(f"  [MEASURED] Final train_loss={final['train_loss']:.4f}  "
              f"train_acc={final['train_acc']:.4f}")
        print(f"  [MEASURED] Final test_loss={final['test_loss']:.4f}  "
              f"test_acc={final['test_acc']:.4f}")
        print(f"  [MEASURED] L2_norm: init={l2_at_init:.3f}  "
              f"T_mem={l2_at_T_mem if l2_at_T_mem else 'N/A'}  "
              f"end={l2_at_end:.3f}")
        print()

        summary_rows.append({
            "seed":               seed,
            "T_mem":              T_mem,
            "T_grok":             T_grok,
            "T_grok_minus_T_mem": (T_grok - T_mem) if (T_mem and T_grok) else None,
            "signal_A":           signal_A,
            "signal_B":           signal_B_fires,
            "signal_B_max_run":   signal_B_max_run,
            "behavioral_grokking": behavioral_grokking,
            "final_train_loss":   final["train_loss"],
            "final_train_acc":    final["train_acc"],
            "final_test_loss":    final["test_loss"],
            "final_test_acc":     final["test_acc"],
            "l2_norm_init":       l2_at_init,
            "l2_norm_at_T_mem":   l2_at_T_mem,
            "l2_norm_end":        l2_at_end,
            "elapsed_s":          seed_elapsed,
        })

    total_elapsed = time.perf_counter() - sweep_start

    # --- Aggregate verdict ---
    n_grokked = sum(1 for r in summary_rows if r["behavioral_grokking"])
    n_T_mem_defined   = sum(1 for r in summary_rows if r["T_mem"] is not None)
    n_T_grok_undefined = sum(1 for r in summary_rows
                             if r["T_mem"] is not None and r["T_grok"] is None)

    if n_grokked >= 2:
        aggregate_verdict = "REPRODUCED"
    elif n_grokked == 1:
        aggregate_verdict = "PARTIAL"
    else:
        # Distinguish sub-cases
        if n_T_mem_defined >= 2 and n_T_grok_undefined >= 2:
            aggregate_verdict = "NOT REPRODUCED — MEMORIZATION ONLY"
        elif n_T_mem_defined < 2:
            aggregate_verdict = "NOT REPRODUCED — TRAINING FAILED"
        else:
            aggregate_verdict = "NOT REPRODUCED"

    aggregate = {
        "n_seeds":                  len(EXP3_SEEDS),
        "n_behavioral_grokking":    n_grokked,
        "n_T_mem_defined":          n_T_mem_defined,
        "verdict":                  aggregate_verdict,
        "total_elapsed_s":          total_elapsed,
    }

    # --- Print aggregate ---
    print("=" * 70)
    print("AGGREGATE VERDICT (pre-registered criteria applied mechanically)")
    print("=" * 70)
    for row in summary_rows:
        print(f"  seed={row['seed']:3d}  T_mem={row['T_mem']}  T_grok={row['T_grok']}  "
              f"behavioral_grokking={row['behavioral_grokking']}")
    print()
    print(f"  n_behavioral_grokking = {n_grokked} / {len(EXP3_SEEDS)}")
    print(f"  VERDICT: {aggregate_verdict}")
    print(f"  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print()

    # --- Save all ---
    df_all     = pd.DataFrame(all_curve_records)
    df_summary = pd.DataFrame(summary_rows)
    df_agg     = pd.DataFrame([aggregate])

    df_all.to_csv(    os.path.join(results_dir, "grokking_curves_all.csv"),   index=False)
    df_summary.to_csv(os.path.join(results_dir, "grokking_summary.csv"),      index=False)
    df_agg.to_csv(    os.path.join(results_dir, "grokking_aggregate.csv"),    index=False)

    print("Files saved:")
    for fn in ["grokking_curves_all.csv", "grokking_summary.csv", "grokking_aggregate.csv"]:
        print(f"  {os.path.join(results_dir, fn)}")

    # --- Plots ---
    _make_plots(df_all, summary_rows, results_dir)

    return df_summary, aggregate


def _make_plots(df_all, summary_rows, results_dir):
    seeds = [r["seed"] for r in summary_rows]
    n     = len(seeds)

    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"train_acc": "#1f77b4", "test_acc": "#d62728"}

    for ax, row in zip(axes, summary_rows):
        seed = row["seed"]
        df   = df_all[df_all["seed"] == seed].copy()

        ax.plot(df["step"], df["train_acc"], color=colors["train_acc"],
                lw=1.0, label="train acc", alpha=0.85)
        ax.plot(df["step"], df["test_acc"],  color=colors["test_acc"],
                lw=1.5, label="test acc")

        # Mark T_mem
        if row["T_mem"] is not None:
            ax.axvline(row["T_mem"], color=colors["train_acc"], ls="--",
                       lw=1.0, alpha=0.7, label=f"T_mem={row['T_mem']:,}")
        # Mark T_grok
        if row["T_grok"] is not None:
            ax.axvline(row["T_grok"], color=colors["test_acc"], ls="--",
                       lw=1.0, alpha=0.7, label=f"T_grok={row['T_grok']:,}")

        # Threshold lines
        ax.axhline(0.99, color=colors["train_acc"], ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0.95, color=colors["test_acc"],  ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0.30, color="gray", ls=":", lw=0.8, alpha=0.5)

        verdict = "GROKKED" if row["behavioral_grokking"] else "not grokked"
        ax.set_title(f"Seed {seed} — {verdict}", fontsize=11)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="center left")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    axes[-1].set_xlabel("Optimizer step")
    fig.suptitle("Experiment 3 — Grokking on Modular Addition (mod 97)\n"
                 "Behavioral criteria: T_mem, T_grok, Signal A+B",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    plot_path = os.path.join(results_dir, "grokking_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {plot_path}")
    print()


if __name__ == "__main__":
    run_sweep()

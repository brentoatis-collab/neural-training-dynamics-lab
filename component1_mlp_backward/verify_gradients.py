"""
verify_gradients.py — Independent gradient verification harness.

Compares the hand-written MLP backward pass against two independent sources:
  1. Central finite differences (numerical approximation)
  2. PyTorch autograd (symbolic differentiation)

Pre-registered criteria (from LEDGER.md):
  - dtype:      float64
  - step size:  h = 1e-5
  - PASS:       100% of elements with rel_error < 1e-5 (vs. both FD and autograd)
  - IDEAL:      >= 95% of elements with rel_error < 1e-7
  - batch_size: >= 4
  - inputs:     >= 2 distinct random (X, y) pairs

Results are saved to component1_mlp_backward/results/gradient_check.csv.

Run this file directly:
    python component1_mlp_backward/verify_gradients.py
"""

import os
import sys

import pandas as pd
import torch

# Allow running from the project root or from within component1_mlp_backward/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXP1_BATCH_SIZE,
    EXP1_D_IN,
    EXP1_D_OUT,
    EXP1_DTYPE,
    EXP1_FD_STEP,
    EXP1_H,
    EXP1_N_INPUTS,
    EXP1_SEED,
    EXP1_TOL_IDEAL,
    EXP1_TOL_PASS,
)
from component1_mlp_backward.mlp import TwoLayerMLP


# ---------------------------------------------------------------------------
# Helper: relative error
# ---------------------------------------------------------------------------

def relative_error(a: float, b: float, eps: float = 1e-8) -> float:
    """
    Relative error between two scalars, safe against near-zero values.

        rel_error = |a - b| / max(|a|, |b|, eps)

    eps prevents division by zero when both values are near zero.
    The eps = 1e-8 default is well below our noise floor (~1e-10 in float64)
    so it only activates when both gradients are genuinely near zero, where
    absolute rather than relative error is the appropriate measure.
    """
    return abs(a - b) / max(abs(a), abs(b), eps)


# ---------------------------------------------------------------------------
# Central finite differences
# ---------------------------------------------------------------------------

def central_fd_gradient(
    mlp: TwoLayerMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    param_name: str,
    flat_idx: int,
    h: float,
) -> float:
    """
    Numerical gradient for a single parameter element via central differences.

        g_fd = [L(theta_i + h) - L(theta_i - h)] / (2h)

    The parameter tensor is perturbed in-place and restored afterward.
    Each evaluation is an independent forward pass; the cached gradients
    from the manual backward pass are not used or affected.

    Args
    ----
    mlp        : TwoLayerMLP instance (weights must already be set)
    X, y       : the specific (input, target) pair to evaluate on
    param_name : one of "W1", "b1", "W2", "b2"
    flat_idx   : flat index into the parameter's elements
    h          : step size (1e-5 for float64, per pre-registered criteria)

    Returns
    -------
    float : central finite-difference gradient estimate
    """
    param = dict(mlp.parameters())[param_name]
    original_val = param.data.view(-1)[flat_idx].item()

    # Evaluate at theta_i + h
    param.data.view(-1)[flat_idx] = original_val + h
    loss_plus = mlp.forward(X, y).item()

    # Evaluate at theta_i - h
    param.data.view(-1)[flat_idx] = original_val - h
    loss_minus = mlp.forward(X, y).item()

    # Restore original value — the MLP's weights are unchanged after this call
    param.data.view(-1)[flat_idx] = original_val

    return (loss_plus - loss_minus) / (2.0 * h)


# ---------------------------------------------------------------------------
# PyTorch autograd gradients
# ---------------------------------------------------------------------------

def autograd_gradients(
    mlp: TwoLayerMLP,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute gradients using PyTorch autograd on an INDEPENDENT copy of the weights.

    Independence is critical: the autograd graph must not share tensors with
    the hand-written forward pass. We copy each weight via .detach().clone()
    and attach requires_grad=True to the copies only. The forward pass is
    re-implemented with the same mathematical operations so that the only
    difference is the gradient computation mechanism.

    Returns
    -------
    dict mapping param name → gradient tensor (detached, same shape as param)
    """
    # Independent copies with gradient tracking
    W1 = mlp.W1.detach().clone().requires_grad_(True)
    b1 = mlp.b1.detach().clone().requires_grad_(True)
    W2 = mlp.W2.detach().clone().requires_grad_(True)
    b2 = mlp.b2.detach().clone().requires_grad_(True)

    # Identical forward pass — same operations, same numerical values
    z1   = X @ W1 + b1
    a1   = torch.clamp(z1, min=0.0)
    z2   = a1 @ W2 + b2
    loss = ((z2 - y) ** 2).mean()

    # Let autograd compute all gradients in one backward pass
    loss.backward()

    return {
        "W1": W1.grad.detach(),
        "b1": b1.grad.detach(),
        "W2": W2.grad.detach(),
        "b2": b2.grad.detach(),
    }


# ---------------------------------------------------------------------------
# Main verification loop
# ---------------------------------------------------------------------------

def run_gradient_check() -> pd.DataFrame:
    """
    Run the full pre-registered gradient check and save results to CSV.

    For each of EXP1_N_INPUTS random (X, y) inputs:
      1. Run the hand-written forward + backward to get manual gradients.
      2. Run autograd on independent weight copies to get autograd gradients.
      3. Run central FD for every individual parameter element.
      4. Compute rel_error for each element vs. both FD and autograd.
      5. Record kink-proximity diagnostic (minimum absolute pre-activation).

    Saves: component1_mlp_backward/results/gradient_check.csv
    Prints: pass/fail summary with element-level breakdown on failure.

    Returns
    -------
    DataFrame of all recorded rows (one row per parameter element per input).
    """
    # Deterministic seed for reproducibility of inputs and weights
    torch.manual_seed(EXP1_SEED)

    mlp = TwoLayerMLP(
        D_in=EXP1_D_IN,
        H=EXP1_H,
        D_out=EXP1_D_OUT,
        dtype=EXP1_DTYPE,
        seed=EXP1_SEED,
    )

    print(f"MLP: D_in={EXP1_D_IN}, H={EXP1_H}, D_out={EXP1_D_OUT}")
    print(f"Total parameters: {mlp.total_params()}")
    print(f"dtype={EXP1_DTYPE} | h={EXP1_FD_STEP} | batch_size={EXP1_BATCH_SIZE} | n_inputs={EXP1_N_INPUTS}")
    print()

    records = []

    for input_idx in range(EXP1_N_INPUTS):
        # Each input uses a fresh random tensor so the two checks exercise
        # different activation patterns (and different ReLU mask patterns).
        X = torch.randn(EXP1_BATCH_SIZE, EXP1_D_IN,  dtype=EXP1_DTYPE)
        y = torch.randn(EXP1_BATCH_SIZE, EXP1_D_OUT, dtype=EXP1_DTYPE)

        # --- Step 1: manual forward + backward ---
        mlp.forward(X, y)
        manual_grads = mlp.backward()

        # Capture the minimum absolute pre-activation BEFORE FD perturbations
        # alter the cache. This is the kink-proximity diagnostic: elements with
        # very small |z1| values are near the ReLU boundary and may have
        # elevated FD error due to the mask changing under perturbation.
        min_abs_preact = mlp._cache["z1"].abs().min().item()

        # --- Step 2: autograd gradients (independent computation) ---
        ag_grads = autograd_gradients(mlp, X, y)

        # --- Step 3 & 4: per-element FD comparison ---
        for param_name, param_tensor in mlp.parameters():
            n_elements = param_tensor.numel()
            manual_flat  = manual_grads[param_name].view(-1)
            ag_flat      = ag_grads[param_name].view(-1)

            for i in range(n_elements):
                g_manual   = manual_flat[i].item()
                g_autograd = ag_flat[i].item()
                g_fd       = central_fd_gradient(mlp, X, y, param_name, i, EXP1_FD_STEP)

                rel_err_fd       = relative_error(g_manual, g_fd)
                rel_err_autograd = relative_error(g_manual, g_autograd)

                # An element passes the hard threshold only if BOTH comparisons pass.
                passed_hard  = (rel_err_fd < EXP1_TOL_PASS and
                                rel_err_autograd < EXP1_TOL_PASS)
                passed_ideal = (rel_err_fd < EXP1_TOL_IDEAL and
                                rel_err_autograd < EXP1_TOL_IDEAL)

                records.append({
                    "input_idx":          input_idx,
                    "param_name":         param_name,
                    "param_index":        i,
                    "grad_manual":        g_manual,
                    "grad_autograd":      g_autograd,
                    "grad_fd":            g_fd,
                    "rel_error_vs_fd":    rel_err_fd,
                    "rel_error_vs_autograd": rel_err_autograd,
                    "passed_1e-5":        passed_hard,
                    "passed_1e-7":        passed_ideal,
                    # Kink diagnostic: the minimum |z1| value for this input.
                    # If this is small and an element fails, check whether
                    # the failing element is in W1/b1 (which feed into z1).
                    "min_abs_preact":     min_abs_preact,
                })

    df = pd.DataFrame(records)

    # --- Save results ---
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "gradient_check.csv")
    df.to_csv(csv_path, index=False)

    # --- Print summary ---
    n_total      = len(df)
    n_pass_hard  = int(df["passed_1e-5"].sum())
    n_pass_ideal = int(df["passed_1e-7"].sum())
    n_fail       = n_total - n_pass_hard

    # Borderline: pass hard threshold but miss ideal (potential kink elements)
    borderline = df[df["passed_1e-5"] & ~df["passed_1e-7"]]

    print("=" * 60)
    print("GRADIENT CHECK RESULTS")
    print("=" * 60)
    print(f"Hard pass  (rel_error < 1e-5):  {n_pass_hard}/{n_total}")
    print(f"Ideal pass (rel_error < 1e-7):  {n_pass_ideal}/{n_total}")
    print()
    print(f"Max rel_error vs FD:            {df['rel_error_vs_fd'].max():.3e}")
    print(f"Max rel_error vs autograd:      {df['rel_error_vs_autograd'].max():.3e}")
    print(f"Median rel_error vs FD:         {df['rel_error_vs_fd'].median():.3e}")
    print(f"Median rel_error vs autograd:   {df['rel_error_vs_autograd'].median():.3e}")
    print()

    if n_fail == 0:
        print("OVERALL VERDICT: PASSED")
        print("  All elements satisfy rel_error < 1e-5 vs. both FD and autograd.")
    else:
        print(f"OVERALL VERDICT: FAILED  ({n_fail} element(s) above 1e-5 threshold)")
        failures = df[~df["passed_1e-5"]]
        print("\nFailing elements:")
        print(failures[[
            "input_idx", "param_name", "param_index",
            "rel_error_vs_fd", "rel_error_vs_autograd",
            "grad_manual", "grad_fd",
        ]].to_string(index=False))

    if len(borderline) > 0:
        print(f"\nBorderline elements (1e-7 <= rel_error < 1e-5): {len(borderline)}")
        print("  If min_abs_preact is near 0 for these rows, likely a ReLU kink.")
        print("  If they cluster by param_name, suspect a structural bug.")
        print(borderline[[
            "input_idx", "param_name", "param_index",
            "rel_error_vs_fd", "min_abs_preact",
        ]].to_string(index=False))

    print()
    print(f"Full results saved to: {csv_path}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_gradient_check()

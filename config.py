"""
config.py — Central configuration for all Gradient Ledger experiments.

All hyperparameters, seeds, and tolerance thresholds are defined here.
Nothing is hard-coded inside experiment files. Any change to an experiment's
parameters must be made here so that the full parameter set is visible in one
place and the ledger entry can reference specific line numbers.
"""

import torch

# ---------------------------------------------------------------------------
# Experiment 1: Hand-written MLP backward pass and gradient verification
# ---------------------------------------------------------------------------

# Network dimensions. Small enough that every weight can be examined individually.
EXP1_D_IN = 8       # input features
EXP1_H = 16         # hidden layer width
EXP1_D_OUT = 4      # output dimension

# Float64 is required for numerical gradient checking.
# See LEDGER.md §1 for the truncation-vs-round-off justification.
EXP1_DTYPE = torch.float64

# Finite-difference step size.
# At float64 (eps ~ 2.22e-16), h=1e-5 gives:
#   truncation error  ~ h^2        ~ 1e-10
#   round-off error   ~ eps / h    ~ 2.2e-11
# Both sources are ~1e-10, well below any tolerance we care about.
EXP1_FD_STEP = 1e-5

# Pass/fail thresholds (pre-registered — do not adjust after seeing results).
# Hard threshold: any element with rel_error >= 1e-5 is a FAILURE.
# Ideal target:   elements with rel_error >= 1e-7 are flagged for investigation.
EXP1_TOL_PASS  = 1e-5   # hard pass/fail boundary
EXP1_TOL_IDEAL = 1e-7   # ideal target; exceeding this triggers kink-proximity check

# Batch size must be >= 4 so that the gradient summation over the batch
# dimension is exercised (a batch of 1 cannot catch missing mean/sum operations).
EXP1_BATCH_SIZE = 4

# Number of distinct random (X, y) inputs to run the check on.
# Two inputs required by pre-registered criteria.
EXP1_N_INPUTS = 2

# Seed for weight initialization and input generation.
EXP1_SEED = 42

# ---------------------------------------------------------------------------
# Experiment 2: Model-wise double descent on sklearn digits
# ---------------------------------------------------------------------------

# --- Data ---
# 80/20 stratified split; seed is fixed and never varied across the sweep.
EXP2_SPLIT_SEED       = 0
EXP2_TEST_FRACTION    = 0.2

# Label corruption applied to training set only.
# Exactly floor(0.15 * n_train) = 215 labels are changed to a different class.
# Corruption seed is fixed; the same 215 indices are corrupted for all exp seeds.
EXP2_CORRUPTION_SEED      = 0
EXP2_CORRUPTION_FRACTION  = 0.15

# --- Architecture ---
# Single hidden layer: Linear(64,H) -> ReLU -> Linear(H,10)
# D_in and D_out are fixed by the dataset; H is swept.
EXP2_D_IN   = 64
EXP2_D_OUT  = 10

# --- Full sweep ---
EXP2_WIDTHS  = [4, 8, 16, 32, 64, 128, 256, 512]
EXP2_SEEDS   = [42, 123, 777]
EXP2_EPOCHS  = 10_000   # may be increased to 30_000 only if pilot trigger 2 fires
EXP2_LR      = 0.01     # full-batch SGD; may be replaced with 0.001 only if pilot trigger 1 fires
EXP2_LOG_EVERY = 500    # checkpoint cadence for the full sweep

# No momentum, no weight decay — regularization would suppress interpolation.
EXP2_MOMENTUM     = 0.0
EXP2_WEIGHT_DECAY = 0.0

# --- Pilot ---
# Pre-registered convergence/runtime check: one seed, three widths, 2000 epochs.
# Results are stored separately and never mixed with full-sweep results.
EXP2_PILOT_SEED      = 42
EXP2_PILOT_WIDTHS    = [16, 64, 512]
EXP2_PILOT_EPOCHS    = 2_000
EXP2_PILOT_LOG_EVERY = 100

# Trigger 1: loss is non-monotone — any 100-epoch block shows a relative
# increase > 5% over the previous logged loss value.
# If fired: lr is replaced with 0.001 before the full sweep.
EXP2_PILOT_TRIGGER1_THRESHOLD = 0.05   # fractional increase threshold

# Trigger 2: H=512 training loss at epoch 2000 still above this value,
# indicating the model has barely moved from random-init cross-entropy (~2.3).
# If fired: epoch budget is increased to 30,000 for the full sweep.
EXP2_PILOT_TRIGGER2_THRESHOLD = 1.5

# --- Run 2A: step-matched mini-batch control ---
# batch_size=32, epochs=223 gives ceil(1437/32)=45 steps/epoch × 223 epochs
# = 10,035 optimizer steps (≈ 10,000 steps in Run 1, within 0.35%).
# Purpose: isolate gradient-noise effect at the same step budget as Run 1.
EXP2_RUN2_BATCH_SIZE   = 32
EXP2_RUN2A_EPOCHS      = 223    # ceil(10_000 / 45) = 223
# Log every epoch for Run 2A (223 rows/run): full resolution over the short run.

# --- Run 2B: mini-batch reproduction attempt ---
# Same batch_size=32. 10,000 epochs × 45 steps/epoch = 450,000 optimizer steps.
# Purpose: test whether gradient noise + adequate step budget achieves interpolation.
EXP2_RUN2B_EPOCHS      = 10_000
EXP2_RUN2B_LOG_EVERY   = 500    # 20 rows per run (same epoch cadence as Run 1)

# ---------------------------------------------------------------------------
# Experiment 3 will be added here after Experiment 2 is authorized.
# ---------------------------------------------------------------------------

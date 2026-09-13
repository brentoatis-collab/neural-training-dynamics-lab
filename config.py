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
# Experiments 2 and 3 will be added here after Experiment 1 is authorized.
# ---------------------------------------------------------------------------

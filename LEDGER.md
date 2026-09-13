# The Gradient Ledger — Experiment Ledger

This file is the running record of what was attempted, what failed or behaved
unexpectedly, what was changed from the original design, and what the evidence
supports. All pre-registered criteria are reproduced here for reference.

---

## Pre-Registered Criteria (locked before any results were seen)

### Experiment 1 — Finite-Difference Gradient Check

**Numerical precision rationale**

For central finite differences, total error = truncation (O(h²)) + round-off (O(eps/h)).
The optimal h minimizes their sum: h_opt ~ eps^(1/3).

| dtype   | eps        | h_opt  | error floor at h_opt |
|---------|------------|--------|----------------------|
| float32 | 1.19e-7    | ~5e-3  | ~4.5e-5              |
| float64 | 2.22e-16   | ~6e-6  | ~1e-10               |

float32 at h=1e-4 produces round-off ~1.2e-3, making a tolerance of 1e-3 identical
to the noise floor — the check cannot distinguish a 0.1% implementation error from
numerical noise. float64 at h=1e-5 gives an error floor of ~1e-10, providing five
orders of magnitude of headroom above the 1e-5 hard threshold.

**Chosen parameters:** dtype=float64, h=1e-5, batch_size>=4, n_inputs>=2.

**PASS requires ALL of the following:**
1. 100% of parameter elements: rel_error < 1e-5 (vs. both FD and autograd).
2. At least 95% of elements: rel_error < 1e-7 (ideal target; reported separately).
3. Elements with 1e-7 ≤ rel_error < 1e-5 must be confirmed to cluster near
   ReLU kink points (near-zero pre-activation), not by parameter type or index.
4. batch_size >= 4.
5. At least 2 distinct random input tensors checked.

**FAIL if:**
- Any single element has rel_error >= 1e-5.
- Elevated-error elements are distributed by parameter type (e.g., all biases),
  which would indicate a structural bug, not numerical noise.

---

## Experiment Log

### 2026-09-13 — Experiment 1 implementation started and executed

**Status: PASSED — all pre-registered criteria satisfied.**

**Measured results (python3, float64, h=1e-5, batch_size=4, n_inputs=2):**

| Metric | Value |
|---|---|
| Hard pass (rel_error < 1e-5) | 424/424 (100%) |
| Ideal pass (rel_error < 1e-7) | 424/424 (100%) |
| Max rel_error vs FD | 8.715e-08 |
| Max rel_error vs autograd | 0.000e+00 (exact match) |
| Median rel_error vs FD | 8.606e-11 |
| Borderline elements (1e-7 ≤ err < 1e-5) | 0 |

**Interpretation:**
- The autograd comparison is exact (0.0 error) because both paths execute the
  same float64 operations in the same order with no diverging accumulation.
- The FD maximum of 8.7e-08 falls between the noise floor (~1e-10) and the ideal
  threshold (1e-7), consistent with expected central-difference error at h=1e-5.
- No elements required kink-proximity investigation; all cleared the 1e-7 target.
- Full per-element record: component1_mlp_backward/results/gradient_check.csv

---

## Deviation Log

*No deviations recorded. Any post-results changes to criteria must be entered here
with: (a) what changed, (b) why the original criterion was wrong in principle,
(c) what the result would have been under the original criterion.*

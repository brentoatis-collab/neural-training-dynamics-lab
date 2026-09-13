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

### 2026-09-13 — Experiment 2 convergence/runtime pilot executed

**Pilot specification:** seed=42, widths=[16, 64, 512], 2000 epochs, full-batch SGD,
lr=0.01, momentum=0, weight_decay=0, log every 100 epochs.
Data: n_train=1437, n_test=360, n_corrupt=215 (15.0%).

**Wall-clock runtime:**

| Width | n_params | Total seconds (2000 epochs) | Projected (10,000 epochs) |
|-------|----------|----------------------------|---------------------------|
| 16    | 1210     | 0.34 s                     | ~1.7 s                    |
| 64    | 4810     | 0.47 s                     | ~2.4 s                    |
| 512   | 38410    | 1.01 s                     | ~5.0 s                    |

Projected full-sweep runtime (24 runs, 10,000 epochs): ~2 minutes on CPU.

**Starting and ending train loss per width:**

| Width | Train loss epoch 100 | Train loss epoch 2000 | Train error epoch 2000 | Test error epoch 2000 |
|-------|----------------------|-----------------------|------------------------|-----------------------|
| 16    | 2.1875               | 0.8731                | 0.1865                 | 0.0722                |
| 64    | 2.0736               | 0.8347                | 0.1747                 | 0.0722                |
| 512   | 1.5002               | 0.7350                | 0.1614                 | 0.0444                |

All losses decreased monotonically across all 100-epoch blocks for all widths.

**Trigger 1 (lr adjustment — fires if any block shows >5% loss increase):** DID NOT FIRE.
No lr change is permitted or will be made.

**Trigger 2 (epoch budget — fires if H=512 final loss > 1.5):** DID NOT FIRE.
H=512 final train_loss = 0.7350 < threshold 1.5. No epoch budget change is permitted or will be made.

**Observation (no action permitted):** At epoch 2000, training error for all three pilot widths
(16–19%) is close to the corruption fraction (14.97%). This is consistent with the model having
learned clean training labels but not yet memorized the corrupted ones. Whether 10,000 epochs
achieves zero training error (interpolation) for the overparameterized widths is unresolved by
this pilot — neither trigger is designed to detect slow convergence short of the 1.5 loss threshold.
This observation is recorded here for transparency. Since no trigger fired, the experimental
parameters remain unchanged per pre-registration.

**Full pilot data:** component2_double_descent/results/pilot_curves.csv,
component2_double_descent/results/pilot_summary.csv

### 2026-09-13 — Experiment 2 full sweep executed

**Status: NOT REPRODUCED — H*_seed undefined for all three seeds.**

**Evidence-criteria classification (pre-registered decision tree):**
Step 1: k = 0 seeds achieved zero training error at any recorded width.
k ≤ 1 → NOT REPRODUCED. No further steps evaluated.

**Measured final training error and test error (epoch 10,000):**

| H | n_params | seed42 train | seed123 train | seed777 train | seed42 test | seed123 test | seed777 test |
|---|---|---|---|---|---|---|---|
| 4 | 310 | 0.2484 | 0.2491 | 0.2811 | 0.1694 | 0.1722 | 0.1972 |
| 8 | 610 | 0.1698 | 0.1698 | 0.1719 | 0.0722 | 0.0889 | 0.0806 |
| 16 | 1210 | 0.1566 | 0.1566 | 0.1503 | 0.0528 | 0.0667 | 0.0639 |
| 32 | 2410 | 0.1468 | 0.1447 | 0.1482 | 0.0639 | 0.0389 | 0.0528 |
| 64 | 4810 | 0.1427 | 0.1406 | 0.1385 | 0.0556 | 0.0556 | 0.0556 |
| 128 | 9610 | 0.1336 | 0.1350 | 0.1336 | 0.0528 | 0.0667 | 0.0500 |
| 256 | 19210 | 0.1260 | 0.1246 | 0.1225 | 0.0556 | 0.0556 | 0.0583 |
| 512 | 38410 | 0.1086 | 0.1086 | 0.1093 | 0.0556 | 0.0639 | 0.0583 |

**H*_seed for all seeds: undefined (no interpolation).**

**What the data shows:**
- Training error decreased monotonically with width for all seeds — classical
  bias-variance behavior, not double descent.
- Minimum training error reached was ~10.9% (H=512), which is below the
  corruption fraction (14.97%). The model partially memorized corrupted labels
  but did not drive all training errors to zero.
- Test error shows no systematic U-shape; the variations at mid-widths are
  small and inconsistent across seeds.
- Total sweep wall-clock time: 59.9 s on CPU.

**Diagnosis:**
Full-batch GD with lr=0.01 converges to a neighborhood of the "approximately
clean" solution (fitting clean labels while partially fitting noisy ones) but
does not drive training loss to zero within 10,000 epochs. Cross-entropy loss
goes to zero only as logits diverge to ±∞, which requires either more
optimization steps, a larger learning rate, or gradient noise (mini-batches)
to push the optimizer out of the early-stopping basin.

This failure was anticipated in the pilot observation recorded above.
Neither pilot trigger was designed to detect this convergence mode.

**Full data:** component2_double_descent/results/double_descent_raw.csv,
component2_double_descent/results/double_descent_curves.csv

---

## Deviation Log

### 2026-09-13 — Experiment 2 Run 2 (mini-batch SGD): pre-registration and design correction

**Date of entry:** 2026-09-13
**Scope:** Experiment 2 — Double Descent, Runs 2A and 2B
**Nature of change:** Optimizer batch size (full-batch → mini-batch, batch_size=32)
**Status at time of writing:** Run 1 complete (NOT REPRODUCED); Runs 2A and 2B not yet executed

---

**1. Original full-batch design and its measured outcome**

Run 1 used full-batch SGD (batch_size=1437), lr=0.01, momentum=0.0, weight_decay=0.0,
10,000 epochs = 10,000 optimizer steps. Architecture: Linear(64,H)→ReLU→Linear(H,10).
Data: 1437 training samples, 215 corrupted (14.97%), split_seed=0, corruption_seed=0.

Result: H*_seed undefined for all three seeds. Minimum training error: 10.9% at H=512.
Training error decreased monotonically with width. No test-error U-shape.
Classified NOT REPRODUCED (Step 1, k=0).

---

**2. Why H*_seed was undefined**

For H*_seed to be defined, the model must correctly classify all 1437 training samples
including the 215 with corrupted labels. Full-batch GD computes gradients by averaging
over all 1437 samples simultaneously. Near the approximately-clean solution, the gradient
component from the 215 corrupted samples is, by sample count, outnumbered approximately
5.7-to-1 by the gradient from the 1222 clean samples — this ratio is a hypothesis about
relative gradient influence based on sample proportions, not a directly measured
gradient-magnitude ratio. The net gradient may therefore favor stabilizing the clean
solution over pushing toward memorization of corrupted labels. In 10,000 full-batch steps
the optimizer moved from ~25% to ~10.9% training error but did not reach zero.

---

**3. Evidence supporting optimization dynamics as the failure mode (with scope of inference)**

Three observations are consistent with optimization dynamics as the failure mode:

- **Training error at H=512 (10.9%) is below the corruption fraction (14.97%).** This
  shows the model correctly classified at least some corrupted-label training samples,
  consistent with having some capacity to fit at least some corrupted label mappings.
  This does not establish that the architecture can simultaneously interpolate all 1,437
  training samples — that would require zero misclassifications, which was not observed.

- **Training error decreases monotonically with width.** All widths show continuous
  improvement as H increases, consistent with optimizer dynamics being the bottleneck
  rather than a hard capacity ceiling at any particular width.

- **Loss was still falling at epoch 2,000 (pilot).** H=512 loss: 1.50 → 0.74 over
  2,000 epochs. This trajectory is consistent with slow convergence rather than
  convergence to a local minimum, though we cannot rule out that the optimizer was
  approaching an asymptote short of zero loss.

---

**3b. The step-count confound and why Option C is required**

Switching from full-batch to mini-batch with the same epoch count (10,000) delivers 45×
more optimizer steps (450,000 vs 10,000). Two things would change simultaneously:
gradient noise (stochasticity) and step count. A reproduction under that design could
not isolate which factor was necessary.

To address this, a step-matched mini-batch control (Run 2A, ~10,000 steps at
batch_size=32) is added alongside the reproduction attempt (Run 2B, 10,000 epochs /
~450,000 steps). Run 2A isolates the stochasticity effect at the original step budget;
Run 2B tests whether noise plus adequate steps achieves interpolation.

---

**4. Why mini-batch SGD is scientifically targeted, not post-hoc curve tuning**

Run 1 produced a monotone test-error curve with no bump — there is no test-curve feature
to align to. The change addresses the specific documented failure mode: the approximately-
clean basin hypothesis above. Mini-batch noise varies the gradient composition per step,
preventing stable convergence to the clean-solution basin. This mechanism is described in
Belkin et al. (2019) and Nakkiran et al. (2019), where interpolation is achieved via
mini-batch optimization. No parameter that shapes the test-error curve (width sweep,
corruption, architecture, test set) is adjusted.

---

**5. Parameters that change**

| Parameter | Run 1 | Run 2A | Run 2B |
|---|---|---|---|
| batch_size | 1437 (full) | 32 | 32 |
| Epochs | 10,000 | 223 | 10,000 |
| Optimizer steps | 10,000 | ~10,035 | ~450,000 |

Steps per epoch = ceil(1437/32) = 45.
Run 2A epoch count = ceil(10,000/45) = 223 epochs = 10,035 steps (0.35% above target).

---

**6. Parameters that remain locked across all runs**

Widths [4,8,16,32,64,128,256,512]; seeds [42,123,777]; split_seed=0;
corruption_seed=0; corruption_fraction=0.15; same 215 corrupted training indices;
architecture Linear(64,H)→ReLU→Linear(H,10); lr=0.01; momentum=0.0; weight_decay=0.0;
test labels (never modified); evidence criteria (decision tree, unchanged).

Run 1 data files are permanent and unmodified.
Run 2A saves to: double_descent_run2a_raw.csv, double_descent_run2a_curves.csv
Run 2B saves to: double_descent_run2b_raw.csv, double_descent_run2b_curves.csv

---

**7. Pre-registered success/partial/failure criteria for Runs 2A and 2B**

The pre-registered decision tree is unchanged and applied independently to each run.

Run 2A is expected to classify as NOT REPRODUCED (k=0, matching Run 1). If Run 2A
unexpectedly produces interpolation, this finding must be reported before Run 2B
results are examined.

Run 2B is the reproduction attempt. If it classifies as NOT REPRODUCED after ~450,000
steps, the experiment is closed as unresolvable within this design.

No criteria will be adjusted after examining either run's results.

---

**Estimated runtimes (CPU):**
Run 2A: ~2 minutes (24 runs × 223 epochs)
Run 2B: ~6 minutes (24 runs × 10,000 epochs)
Combined: ~8 minutes

---

### 2026-09-13 — Experiment 2 Run 2A executed (step-matched mini-batch control)

**Status: NOT REPRODUCED — H*_seed undefined for all three seeds. Result matches expectation.**

**Configuration:** batch_size=32, steps_per_epoch=45, epochs=223, total_steps=10,035 per run.

**Measured final training error and test error (epoch 223 / step 10,035):**

| H | n_params | seed42 train | seed123 train | seed777 train | seed42 test | seed123 test | seed777 test |
|---|---|---|---|---|---|---|---|
| 4 | 310 | 0.2533 | 0.2484 | 0.2763 | 0.1667 | 0.1694 | 0.1944 |
| 8 | 610 | 0.1677 | 0.1698 | 0.1712 | 0.0722 | 0.0861 | 0.0778 |
| 16 | 1210 | 0.1566 | 0.1552 | 0.1496 | 0.0556 | 0.0694 | 0.0639 |
| 32 | 2410 | 0.1468 | 0.1447 | 0.1475 | 0.0611 | 0.0361 | 0.0500 |
| 64 | 4810 | 0.1420 | 0.1406 | 0.1378 | 0.0556 | 0.0556 | 0.0556 |
| 128 | 9610 | 0.1343 | 0.1350 | 0.1343 | 0.0528 | 0.0667 | 0.0500 |
| 256 | 19210 | 0.1253 | 0.1260 | 0.1218 | 0.0556 | 0.0556 | 0.0528 |
| 512 | 38410 | 0.1086 | 0.1079 | 0.1079 | 0.0556 | 0.0639 | 0.0556 |

**H*_seed for all seeds: undefined.**

**Comparison to Run 1 (full-batch, same ~10,000 steps):**
Training error values at H=512: Run 1 ≈ 10.86–10.93%; Run 2A ≈ 10.79–10.86%.
The values are essentially identical. Mini-batch gradient noise at the same step
budget produced no meaningful improvement in interpolation progress.

**Interpretation:** This confirms the pre-registered expectation. Gradient noise alone
(at the same optimizer-step count as Run 1) does not overcome the step-budget
limitation. The step count, not the noise, is the operative bottleneck. Run 2B
(450,000 steps) is needed to test whether noise plus adequate steps achieves
interpolation. Run 2B awaiting authorization.

**Wall-clock time:** 29.6s for all 24 runs.
**Full data:** component2_double_descent/results/double_descent_run2a_raw.csv,
component2_double_descent/results/double_descent_run2a_curves.csv

---

### 2026-09-13 — Experiment 2 Run 2B executed (mini-batch reproduction attempt)

**Status: NOT REPRODUCED under pre-registered decision tree (Step 2, p=0).**
**A non-monotone test-error curve with double-descent shape was observed,
but the peak occurred at H=16 (adjacent to H*_seed=32), not at H*_seed=32.**

**Configuration:** batch_size=32, epochs=10,000, steps_per_epoch=45, total_steps=450,000.

**Decision-tree application:**
- Step 1: k=3 (all seeds have H*_seed=32). Proceed to Step 2.
- Step 2: p = seeds where test_error(H=32) > test_error(H=16) AND test_error(H=32) > test_error(H=64).
  For all three seeds: test_error(H=32) < test_error(H=16) → H=32 is NOT a local maximum.
  p = 0 → NOT REPRODUCED. Decision tree terminates here.

**Measured final training error and test error (epoch 10,000):**

| H | n_params | s42 train | s123 train | s777 train | s42 test | s123 test | s777 test |
|---|---|---|---|---|---|---|---|
| 4 | 310 | 0.2018 | 0.2387 | 0.2213 | 0.1667 | 0.1639 | 0.2028 |
| 8 | 610 | 0.1503 | 0.1503 | 0.1503 | 0.1944 | 0.2028 | 0.1583 |
| 16 | 1210 | 0.0160 | 0.0042 | 0.0042 | 0.3028 | 0.3139 | 0.3222 |
| 32 | 2410 | 0.0000 | 0.0000 | 0.0000 | 0.2361 | 0.2389 | 0.2361 |
| 64 | 4810 | 0.0000 | 0.0000 | 0.0000 | 0.1528 | 0.2028 | 0.1861 |
| 128 | 9610 | 0.0000 | 0.0000 | 0.0000 | 0.1194 | 0.1444 | 0.1417 |
| 256 | 19210 | 0.0000 | 0.0000 | 0.0000 | 0.1417 | 0.1222 | 0.1444 |
| 512 | 38410 | 0.0000 | 0.0000 | 0.0000 | 0.1083 | 0.1083 | 0.1000 |

**H*_seed: 32 for all three seeds (fully consistent).**

**Test error at H=16 (below threshold), H=32 (threshold), H=64 (above threshold):**

| Seed | H=16 (below) | H=32 (H*_seed) | H=64 (above) |
|---|---|---|---|
| 42 | 0.3028 | 0.2361 | 0.1528 |
| 123 | 0.3139 | 0.2389 | 0.2028 |
| 777 | 0.3222 | 0.2361 | 0.1861 |

The test error is highest at H=16, lower at H=32, and lower still at H=64.
The peak is at H=16, not H=32. H=32 is a local minimum relative to H=16 but
a local maximum relative to H=64 only — it satisfies the right-neighbor condition
but not the left-neighbor condition of the peak criterion.

**Observed curve shape (test error, per seed — all three seeds agree):**
H=4: ~17-20%  → H=8: ~16-20%  → H=16: ~30-32% (peak)
→ H=32: ~24% → H=64: ~15-20% → H=128-512: decreasing to ~10%

A non-monotone double-descent-like shape is clearly present: test error rises sharply
in the near-interpolating regime (H=16, where train_err = 0.4-1.6%), peaks there,
then declines monotonically through the over-parameterized regime.

**Why the peak is at H=16, not H=32:**
H=16 is the near-interpolating regime: train_err is 0.4-1.6%, meaning the model
nearly memorizes all training labels (including corrupted ones) but not completely.
This partial memorization appears to produce worse generalization than either the
under-fitted models (H=4, H=8) or the cleanly-interpolating models (H=32+). This
is consistent with theory: the sharpest generalization degradation occurs in the
regime where the model is constrained to nearly (but not exactly) satisfy all noisy
training constraints, not at the exact interpolation threshold.

**Seed-to-seed consistency:** Very high. All three seeds show:
- H*_seed = 32 (unanimous)
- Peak test error at H=16 (unanimous, range 30-32%)
- Monotone decline from H=32 through H=512
- Near-identical test error at H=512 (10-11%)

**Descent in over-parameterized regime:** Present and consistent.
All seeds: test_error(H=512) ≈ 10-11% < test_error(H=32) ≈ 24%. Substantial.

**Wall-clock time:** 1193.2s (~19.9 min) for all 24 runs on CPU.

**Full data:** component2_double_descent/results/double_descent_run2b_raw.csv,
component2_double_descent/results/double_descent_run2b_curves.csv

---

## Experiment 3 — Grokking on Modular Arithmetic

### Pre-Registration (locked 2026-09-13, before any code executed)

---

#### 3.1 Scientific question

Can a one-layer transformer trained on modular addition (a + b ≡ c mod 97) exhibit
grokking: a phase transition from near-random test accuracy to high test accuracy
that occurs long after training accuracy has saturated? [HYP, based on LIT: Power et al. 2022]

---

#### 3.2 Dataset

- All p² = 97² = 9,409 ordered pairs (a, b) with a, b ∈ {0, …, 96}. [OPDEF]
- Labels: y = (a + b) mod 97 (one of p=97 classes). [OPDEF]
- Random 50/50 train/test split using np.random.default_rng(EXP3_SPLIT_SEED=0).permutation(9409).
  Split is not stratified — matches Power et al. (2022) convention. [LIT]
- n_train ≈ 4705, n_test ≈ 4704. [OPDEF; exact counts depend on floor(9409 × 0.5)]

**Note on commutative pairs:** Approximately 50% of test pairs have their commuted
partner (b, a) in the training set. This is accepted as the literature convention
for this benchmark. It means test performance reflects generalization to orderings
and compositions not in training, not to entirely unseen operand pairs. [LIT, OPDEF]

---

#### 3.3 Architecture

1-layer transformer:
- Shared token embedding: nn.Embedding(97, 128) [OPDEF]
- Positional embedding: nn.Embedding(2, 128) — position 0 for operand a, position 1 for operand b [OPDEF]
- One TransformerEncoderLayer: d_model=128, n_heads=4, d_ff=512, activation=gelu,
  pre-norm (norm_first=True), dropout=0.0 [OPDEF]
- Mean pooling over the 2-token sequence [OPDEF]
- Output linear head: nn.Linear(128, 97) [OPDEF]

Motivation for pre-norm and gelu: aligns with modern transformer conventions
and the original grokking paper's setup. [LIT]

---

#### 3.4 Optimizer

AdamW with lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0. [LIT: Power et al. 2022]
Full-batch gradient descent (batch_size = n_train ≈ 4705 each step). [OPDEF]
Weight decay = 1.0 is the primary mechanism proposed in Power et al. (2022) to
induce the grokking phase transition. [LIT]
No learning rate schedule. [OPDEF]

---

#### 3.5 Training budget and seeds

Budget: 50,000 full-batch optimizer steps per seed. [OPDEF]
Seeds: [42, 123, 777] for model initialization (torch.manual_seed). [OPDEF]
Split seed: 0 (fixed; same split for all seeds). [OPDEF]
Logging: every 100 steps (500 checkpoints per seed). [OPDEF]

---

#### 3.6 Pilot

**Pilot parameters (seed=42 only, 500 steps, log every 10 steps):**
Purpose: verify numerical stability and measure throughput.
Permits one conditional hyperparameter change:

**Stability Trigger:** If any 50-step block shows a relative train loss increase
> 5% over the previous block's final logged value, the trigger fires and lr is
reduced to 3e-4 for all three full-experiment seeds. [OPDEF]

No other changes are permitted regardless of pilot outcomes.
Pilot data is saved to component3_grokking/results/grokking_pilot_curves.csv.

---

#### 3.7 Pre-registered evidence criteria

The grokking verdict is determined by behavioral signals (A and B) alone.
Signal C (L2 norm) is recorded as mechanistic supporting evidence but does
NOT gate the verdict.

**Definitions:**

T_mem = the earliest logged checkpoint where train_acc ≥ 0.99 AND train_acc
remains ≥ 0.99 for at least 5 consecutive logged checkpoints. With the full
experiment logging every 100 steps, this is a 500-step stability window.
[OPDEF] If no such checkpoint exists, T_mem is undefined.

T_grok = the earliest logged checkpoint where test_acc ≥ 0.95 AND test_acc
remains ≥ 0.95 for at least 5 consecutive logged checkpoints (500-step window).
[OPDEF] If no such checkpoint exists, T_grok is undefined.

**Signal A — Delayed generalization:** [OPDEF]
Signal A fires if and only if:
  (i)  T_mem is defined, AND
  (ii) T_grok is defined, AND
  (iii) T_grok − T_mem ≥ 2,000 optimizer steps.
Condition (iii) enforces that generalization is genuinely delayed relative to
memorization, not simply simultaneous with it.

**Signal B — Memorization plateau before generalization:** [OPDEF]
Signal B fires if and only if there exist at least 10 consecutive logged
checkpoints, all lying strictly between T_mem and T_grok, at which test_acc < 0.30.
At EXP3_LOG_EVERY=100 steps, this corresponds to ≥ 1,000 optimizer steps with
near-chance test accuracy after training has saturated.

**behavioral_grokking = Signal A AND Signal B.** [OPDEF]

**Signal C — L2 norm compression (mechanistic, non-gating):** [OPDEF]
The model's L2 weight norm decreases by ≥ 20% from its maximum observed value
during training, measured at the end of the 50,000-step run.
Recorded as supporting evidence of weight decay driving generalization; does not
enter the formal verdict. [HYP based on LIT: Power et al. 2022, Nanda et al. 2023]

---

#### 3.8 Aggregate verdict (pre-registered, mutually exclusive)

behavioral_grokking is evaluated independently for each of the 3 seeds.

- k ≥ 2 seeds have behavioral_grokking = True  → **REPRODUCED**
- k = 1 seed  has behavioral_grokking = True   → **PARTIAL**
- k = 0 seeds have behavioral_grokking = True  → **NOT REPRODUCED**
  - If k ≥ 2 seeds have T_mem defined but T_grok undefined: NOT REPRODUCED — MEMORIZATION ONLY
  - If k ≥ 2 seeds have T_mem undefined: NOT REPRODUCED — TRAINING FAILED

No criteria will be adjusted after examining any results.

---

#### 3.9 What will be recorded

For each seed:
- Final train/test loss and accuracy (step 50,000)
- T_mem (or "undefined")
- T_grok (or "undefined")
- behavioral_grokking verdict (True/False) with Signal A and Signal B sub-verdicts
- L2 norm at step 0 (init), at T_mem (if defined), and at step 50,000
- Wall-clock runtime

All data saved to component3_grokking/results/.

---

#### 3.10 Implementation-review discrepancy (caught before full experiment)

**Date:** 2026-09-13
**Status at time of discovery:** Pilot (500 steps, seed=42) complete; full experiment NOT yet run.

The initial code implementation of the pre-registration contained three errors
relative to the approved pre-registration text:

| # | What was implemented | What the pre-registration requires | Correction |
|---|---|---|---|
| 1 | Signal A: "train_acc ≥ 99% for 5,000 optimizer steps" | T_mem/T_grok defined as 5 consecutive *logged checkpoints* (500-step window at log_every=100) | Fixed in LEDGER §3.7 and config.py |
| 2 | Signal B: "test_acc rises from < 60% to ≥ 90%" | ≥ 10 consecutive checkpoints with test_acc < 0.30 between T_mem and T_grok | Fixed in LEDGER §3.7 and config.py |
| 3 | Pilot LR contingency: reduced to 1e-4 if trigger fires | Approved value was 3e-4 | Fixed in LEDGER §3.6 and config.py |

**Effect on pilot data:** None. The pilot trigger did not fire, so the LR contingency
value is irrelevant to the pilot result. The pilot CSV files (grokking_pilot_curves.csv,
grokking_pilot_summary.csv) are preserved exactly as generated and are unaffected
by this correction.

**Effect on full experiment:** None yet — the full experiment has not been run.
The corrected criteria are now locked in config.py and LEDGER.md before any
full-experiment outcomes are examined.

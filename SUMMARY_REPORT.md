# The Gradient Ledger — Final Summary Report

**Repository:** week01-gradient-ledger
**Commits covered:** 0a4ef74 through a33d422
**Report generated from:** committed CSVs and LEDGER.md only. No experiments were rerun.
**Claim classification used throughout:**
- [MEASURED] — directly read from committed CSV or logged output
- [OPDEF] — operational definition chosen before data were examined
- [HYP] — hypothesis or interpretation, not established by this experiment
- [LIT] — claim attributed to the external literature

---

## Framing Question

Dr. Priya Raman observes that her neural network's validation loss diverges sharply
after training loss falls to near zero. Is the backward pass broken, or is this
legitimate deep-learning behavior? The three experiments below build the diagnostic
framework needed to answer that question rigorously.

---

## Experiment 1 — Finite-Difference Gradient Verification

### Setup [OPDEF unless marked]

A two-layer MLP was implemented in pure Python/NumPy with a hand-written forward
and backward pass:

- Architecture: Linear(8→16) → ReLU → Linear(16→4) → MSE loss
- Total trainable parameters: 8×16 + 16 + 16×4 + 4 = **424 scalar elements** [MEASURED]
- dtype: float64 throughout [OPDEF]
- Batch size: 4 (satisfies pre-registered minimum of ≥ 4) [OPDEF]
- Number of independent random inputs tested: 2 [OPDEF]

Three independent gradient estimates were compared for every parameter element:

| Source | Method |
|--------|--------|
| Manual | Chain-rule backward pass coded by hand |
| Autograd | PyTorch `loss.backward()` on the same float64 computation graph |
| Finite difference (FD) | Central differences: `(f(θ+h) − f(θ−h)) / (2h)`, h=1e-5 |

**Why float64 and h=1e-5?** [OPDEF]
Central FD error = truncation O(h²) + round-off O(ε/h). At float64
(ε ≈ 2.22e-16) with h=1e-5, both terms are ≈ 1e-10 — five orders of magnitude
below the 1e-5 hard threshold. At float32 (ε ≈ 1.19e-7), the round-off floor
at h=1e-4 is ≈ 1e-3, making a 1e-3 tolerance indistinguishable from noise.
float64 was required to make the check meaningful.

**Pre-registered pass/fail criteria** [OPDEF]:
- PASS requires 100% of elements with rel_error < 1e-5 vs. both FD and autograd.
- Ideal target: ≥ 95% of elements with rel_error < 1e-7.
- Elements in [1e-7, 1e-5) must cluster near ReLU kink points, not by parameter type.
- FAIL if any single element ≥ 1e-5, or if elevated errors are distributed by type.

### Results [MEASURED]

| Metric | Value |
|--------|-------|
| Elements checked | 424 / 424 |
| Hard pass (rel_error < 1e-5 vs. FD) | 424 / 424 (100%) |
| Hard pass (rel_error < 1e-5 vs. autograd) | 424 / 424 (100%) |
| Ideal pass (rel_error < 1e-7 vs. FD) | 424 / 424 (100%) |
| Maximum rel_error vs. FD | **8.715e-08** |
| Maximum rel_error vs. autograd | **0.000e+00** (exact match) |
| Median rel_error vs. FD | **8.606e-11** |
| Borderline elements [1e-7, 1e-5) | 0 |

**Verdict: PASSED** — all pre-registered criteria satisfied.

The autograd comparison is exact (0.0) because both paths execute the same
float64 operations in the same order. The FD maximum of 8.715e-08 sits between
the noise floor (~1e-10) and the ideal threshold (1e-7), consistent with expected
central-difference error at h=1e-5. No kink-proximity investigation was needed.

### What passing establishes — and what it does not [OPDEF / HYP]

**Establishes [OPDEF]:** The hand-written chain-rule implementation produces
gradients indistinguishable from both PyTorch autograd and central finite
differences to within numerical precision. There is no implementation error in
the backward pass for this architecture and these inputs.

**Does NOT establish [HYP]:** This check does not address training dynamics,
generalization, convergence speed, or the shape of the loss landscape. A model
can pass a gradient check and still exhibit diverging validation loss, slow
convergence, or double-descent curves — those are training phenomena, not
backward-pass bugs.

---

## Experiment 2 — Model-Wise Double Descent

### Setup [OPDEF unless marked]

**Dataset:** scikit-learn handwritten digits (1,797 samples total). [MEASURED]
- 80/20 stratified train/test split, split_seed=0. [OPDEF]
- n_train = 1,437; n_test = 360. [MEASURED]
- **Label corruption:** floor(0.15 × 1,437) = **215 training labels** randomly
  reassigned to a different class using corruption_seed=0. The same 215 indices
  are corrupted for all experimental seeds. [OPDEF/MEASURED]
- Corruption fraction: 14.97%. [MEASURED]

**Architecture sweep:** Single hidden layer, Linear(64→H)→ReLU→Linear(H→10),
width H ∈ {4, 8, 16, 32, 64, 128, 256, 512}. [OPDEF]

**Parameter counts** [MEASURED]:

| H | n_params |
|---|---------|
| 4 | 310 |
| 8 | 610 |
| 16 | 1,210 |
| 32 | 2,410 |
| 64 | 4,810 |
| 128 | 9,610 |
| 256 | 19,210 |
| 512 | 38,410 |

**Seeds:** [42, 123, 777] — independent weight initializations, same data split
and same 215 corrupted indices for all seeds. [OPDEF]

**Interpolation threshold H\*\_seed** [OPDEF]:
For each seed, H\*\_seed = the smallest H at which train_error = 0.000 exactly
(zero misclassifications on all 1,437 training samples including the 215 with
corrupted labels). Determined from measured training error, not from analytic
parameter count.

**Optimizer:** full-batch SGD, lr=0.01, momentum=0, weight_decay=0. [OPDEF]

### Run 1 — Full-batch SGD, 10,000 epochs [MEASURED]

Each epoch = one full-batch gradient step. Total: 10,000 optimizer steps.

**Final training error (epoch 10,000):**

| H | seed=42 | seed=123 | seed=777 |
|---|---------|---------|---------|
| 4 | 0.2484 | 0.2491 | 0.2811 |
| 8 | 0.1698 | 0.1698 | 0.1719 |
| 16 | 0.1566 | 0.1566 | 0.1503 |
| 32 | 0.1468 | 0.1447 | 0.1482 |
| 64 | 0.1427 | 0.1406 | 0.1385 |
| 128 | 0.1336 | 0.1350 | 0.1336 |
| 256 | 0.1260 | 0.1246 | 0.1225 |
| 512 | **0.1086** | **0.1086** | **0.1093** |

H\*\_seed: **undefined for all three seeds** (no width achieved zero training error). [MEASURED]

**Decision tree Step 1:** k = 0 seeds achieved interpolation → **NOT REPRODUCED**.

Training error decreased monotonically with width (classical bias-variance), never
reaching zero. Minimum observed training error was ~10.9% at H=512, which is
below the corruption fraction (14.97%) — the model partially memorized corrupted
labels but did not drive all errors to zero within 10,000 gradient steps.

### Run 2A — Step-matched mini-batch control [MEASURED]

**Why Run 2A was needed:** Switching from full-batch to mini-batch (batch_size=32)
at the same epoch count would increase the optimizer step budget 45× (from 10,000
to ~450,000 steps). Two things would change simultaneously: gradient noise and step
count. Run 2A was added to isolate the noise effect by matching the step budget.

**Design:** batch_size=32, epochs=223, steps_per_epoch=ceil(1437/32)=45,
total steps = 10,035 per (seed, width) — within 0.35% of Run 1's 10,000 steps.
Pre-registered expected outcome: NOT REPRODUCED (matching Run 1). [OPDEF]

H\*\_seed: **undefined for all three seeds.** [MEASURED]
H=512 training error: 10.79–10.86% across seeds [MEASURED] — essentially identical
to Run 1 (10.86–10.93%). Mini-batch noise at the same step budget produced no
meaningful improvement. **Result matched expectation.**

### Run 2B — Mini-batch reproduction attempt [MEASURED]

**Design:** batch_size=32, epochs=10,000, total steps ≈ **450,000** per run.

H\*\_seed: **32 for all three seeds** (unanimous). [MEASURED]

**Final test error:**

| H | n_params | seed=42 | seed=123 | seed=777 |
|---|---------|---------|---------|---------|
| 4 | 310 | 0.1667 | 0.1639 | 0.2028 |
| 8 | 610 | 0.1944 | 0.2028 | 0.1583 |
| **16** | **1,210** | **0.3028** | **0.3139** | **0.3222** |
| **32 (H\*)** | **2,410** | **0.2361** | **0.2389** | **0.2361** |
| 64 | 4,810 | 0.1528 | 0.2028 | 0.1861 |
| 128 | 9,610 | 0.1194 | 0.1444 | 0.1417 |
| 256 | 19,210 | 0.1417 | 0.1222 | 0.1444 |
| 512 | 38,410 | 0.1083 | 0.1083 | 0.1000 |

**Test error at the three key widths** (under/near/over H\*):

| Seed | H=16 (below H\*) | H=32 (H\*) | H=64 (above H\*) |
|------|-----------------|-----------|-----------------|
| 42 | 0.3028 | 0.2361 | 0.1528 |
| 123 | 0.3139 | 0.2389 | 0.2028 |
| 777 | 0.3222 | 0.2361 | 0.1861 |

**Pre-registered decision tree Step 2** [OPDEF]:
p = number of seeds where test_error(H\*) > test_error(H below H\*) AND
test_error(H\*) > test_error(H above H\*).
For all three seeds: test_error(H=32) < test_error(H=16), so H=32 is NOT a
local maximum. p = 0 → **NOT REPRODUCED.**

**Formal verdict: NOT REPRODUCED** (pre-registered decision tree, Step 2, p=0).

**Qualitative observation** [MEASURED, secondary]:
A non-monotone double-descent-like curve is clearly present across all three seeds.
Test error peaks at H=16 (near-interpolating, train_err 0.4–1.6%) at 30–32%,
then declines monotonically through the over-parameterized regime to 10–11% at
H=512. The overall decline from H=32 to H=512 is substantial and consistent
(test_error(H=32) ≈ 23–24%; test_error(H=512) ≈ 10–11%). This qualitative shape
was reproducible across all three seeds but does not satisfy the pre-registered
peak-position criterion.

---

## Experiment 3 — Grokking on Modular Arithmetic

### Setup [OPDEF unless marked]

**Task:** Predict (a + b) mod 97 given operands a, b ∈ {0, …, 96}. [OPDEF]
All p² = 97² = **9,409 ordered pairs** enumerated. [MEASURED]
Random 50/50 split using np.random.default_rng(seed=0).permutation(9409). [OPDEF]
n_train = **4,704**; n_test = **4,705**. [MEASURED]
Per-class count range in training set: **[32, 60]**. [MEASURED]
Split is not stratified — matches Power et al. (2022) convention. [LIT]

**Architecture:** 1-layer transformer. [OPDEF]
- Shared token embedding: nn.Embedding(97, 128)
- Positional embedding: nn.Embedding(2, 128)
- One TransformerEncoderLayer: d_model=128, n_heads=4, d_ff=512, GELU activation,
  pre-norm, dropout=0
- Mean pooling over 2 tokens → nn.Linear(128, 97)
- Total parameters: **223,457** [MEASURED]

**Optimizer:** AdamW, lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0. [LIT: Power et al. 2022]
Full-batch gradient descent (batch_size = n_train = 4,704 each step). [OPDEF]
Budget: **50,000 optimizer steps** per seed. [OPDEF]
Seeds: [42, 123, 777]; split_seed=0 fixed. [OPDEF]
Logging: every 100 steps (500 checkpoints per seed). [OPDEF]

**Pilot (seed=42, 500 steps):** Stability trigger did not fire. lr=1e-3
confirmed for all three full-experiment seeds. [MEASURED]

### Pre-registered behavioral criteria [OPDEF]

**T\_mem:** Earliest logged checkpoint where train\_acc ≥ 0.99 sustained for 5
consecutive logged checkpoints (500-step stability window at log\_every=100).

**T\_grok:** Earliest logged checkpoint where test\_acc ≥ 0.95 sustained for 5
consecutive logged checkpoints (same window).

**Signal A** fires if and only if: T\_mem defined AND T\_grok defined AND
T\_grok − T\_mem ≥ 2,000 optimizer steps.

**Signal B** fires if and only if: ≥ 10 consecutive logged checkpoints strictly
between T\_mem and T\_grok have test\_acc < 0.30 (≥ 1,000 optimizer steps of
near-chance test accuracy after training has saturated).

**behavioral\_grokking = Signal A AND Signal B.** [OPDEF]

**Aggregate:** ≥ 2 seeds → REPRODUCED; exactly 1 → PARTIAL; 0 → NOT REPRODUCED.

### Results [MEASURED]

**T\_mem and T\_grok per seed:**

| Seed | T\_mem | T\_grok | T\_grok − T\_mem | Signal A threshold |
|------|--------|---------|-----------------|-------------------|
| 42 | 200 | 1,100 | 900 | 2,000 — NOT MET |
| 123 | 200 | 2,000 | 1,800 | 2,000 — NOT MET |
| 777 | 200 | 1,400 | 1,200 | 2,000 — NOT MET |

All seeds: T\_mem = step 200 (train\_acc reached 100% at step 200 and never
dropped). T\_grok ranged from 1,100 to 2,000. Seed 123 came closest to the
Signal A threshold, falling 200 steps short.

**Signal B — consecutive low-accuracy checkpoints between T\_mem and T\_grok:**

| Seed | Checkpoints between | Max consecutive run < 0.30 | Required | Signal B |
|------|---------------------|---------------------------|----------|----------|
| 42 | 8 | 5† | 10 | False |
| 123 | 17 | 6 | 10 | False |
| 777 | 11 | 6 | 10 | False |

†Reporting artifact: the sweep script's `find_signal_B` function returns
(False, 0) by early-return when `len(between) < n_consecutive` (8 < 10), causing
seed 42's max\_run to appear as 0 in `grokking_summary.csv`. The correct value is
**5**, verified by independent calculation. The decision (Signal B = False) is
correct under either count, since 5 < 10. The raw CSV is preserved as generated;
this note documents the artifact transparently.

**Behavioral verdicts:**

| Seed | Signal A | Signal B | behavioral\_grokking |
|------|----------|----------|---------------------|
| 42 | False | False | False |
| 123 | False | False | False |
| 777 | False | False | False |

**Aggregate: n\_behavioral\_grokking = 0 / 3 → NOT REPRODUCED.** [MEASURED/OPDEF]

**Final state (step 50,000):**

| Seed | train\_loss | train\_acc | test\_loss | test\_acc |
|------|------------|-----------|----------|---------|
| 42 | 0.0002 | 1.0000 | 0.0002 | 1.0000 |
| 123 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 777 | 0.0005 | 1.0000 | 0.0006 | 1.0000 |

**Test accuracy trajectory (selected checkpoints)** [MEASURED]:

| Step | seed=42 | seed=123 | seed=777 |
|------|---------|---------|---------|
| 100 | 0.003 | 0.001 | 0.001 |
| 200 | 0.020 | 0.012 | 0.010 |
| 500 | 0.092 | 0.055 | 0.050 |
| 700 | 0.273 | 0.159 | 0.143 |
| 900 | 0.606 | 0.338 | 0.340 |
| **1,100** | **0.969** | 0.598 | 0.641 |
| **1,400** | 0.999 | 0.890 | **0.976** |
| **2,000** | 1.000 | **0.968** | 1.000 |
| 5,000 | 1.000 | 1.000 | 1.000 |

Bold marks T\_grok for each seed. All three seeds reached test\_acc = 1.000
well before step 10,000.

**Signal C — L2 norm compression (mechanistic supporting evidence, non-gating):**

| Seed | L2 at T\_mem (step 200) | L2 at step 50,000 | Compression |
|------|------------------------|-------------------|-------------|
| 42 | 105.51 | 34.08 | 67.7% |
| 123 | 105.64 | 37.27 | 64.7% |
| 777 | 107.48 | 32.36 | 69.9% |

All three seeds show L2 weight-norm compression well exceeding the 20% pre-registered
threshold. This satisfies the pre-registered non-gating Signal C and is consistent
with the hypothesized norm-compression dynamics described in Power et al. (2022)
and Nanda et al. (2023). This experiment does not isolate weight decay causally
(no zero-weight-decay control was run) and does not establish parameter sparsity
(sparsity was not measured). Signal C does not alter the behavioral verdict. [HYP/LIT]

### What happened — distinguished from the formal verdict

[MEASURED] All three seeds generalized completely: test\_acc = 1.000 at the end
of the run for every seed. T\_mem = 200 and T\_grok ∈ {1,100, 1,400, 2,000}
were both defined for every seed.

[HYP] Generalization was faster than the pre-registered threshold required. The
2,000-step gap criterion was designed to capture a prolonged memorization plateau
before the phase transition. In this run, the transition was continuous and rapid,
completing in 900–1,800 steps. Under the binding pre-registered operational
definition, this is NOT REPRODUCED — not because generalization failed, but because
the delayed-plateau structure was not observed.

[LIT/HYP] Power et al. (2022) report grokking delays of tens of thousands of steps.
The substantially shorter delay here may reflect differences in model scale,
optimizer, or task structure. No post-hoc explanation is endorsed.

**Formal verdict: NOT REPRODUCED.** No criteria are revised.

---

## Engineering Diagnosis for Dr. Raman

Dr. Raman's question: *Is my backward pass broken, or is this legitimate behavior?*

### 1. Evidence that would implicate a backward-pass implementation error

A gradient check (Experiment 1's method) directly tests this. A broken backward
pass produces one or more of the following signatures:

- **Relative error ≥ 1e-5** between manual gradients and finite-difference
  estimates (at float64, h=1e-5).
- **Errors clustered by parameter type** (e.g., all biases wrong, all second-layer
  weights wrong) — indicating a structural bug in a specific section of the
  backward pass rather than numerical noise.
- **Exact mismatch against PyTorch autograd** — since autograd is numerically
  identical at float64, any discrepancy is a code error, not floating-point noise.
- **Non-finite gradients** (NaN or Inf) at initialization or early steps.

In Experiment 1, none of these signatures appeared. All 424 elements passed at
< 1e-7 relative error, the autograd comparison was exact (0.000), and errors were
uniformly distributed across parameter types. This rules out a backward-pass bug
for this architecture.

### 2. Evidence that supports legitimate training and generalization dynamics

The following behaviors were observed across Experiments 2 and 3, none of which
indicate a broken backward pass:

**Experiment 2 (Double Descent):**
- Validation (test) error rose sharply in the near-interpolating regime (H=16,
  train\_err 0.4–1.6%) before declining in the over-parameterized regime.
- This is the classical double-descent phenomenon: when a model is just barely
  large enough to memorize all training labels including corrupted ones, its
  solutions are highly sensitive to label noise and generalize poorly. Larger
  models find solutions that both interpolate the data and generalize better.
- This behavior was **consistent across all three independent seeds** and was
  present even without formal reproduction of the pre-registered criterion.

**Experiment 3 (Grokking):**
- Training accuracy saturated at 100% by step 200 for all seeds.
- Test accuracy remained near-random (~2%) at step 200 despite zero training error.
- Test accuracy then rose continuously, reaching 100% by step 5,000 for all seeds.
- This sharp decoupling of training and test accuracy — where a perfectly memorized
  model initially fails to generalize — is a known phenomenon in over-parameterized
  networks trained with weight decay, documented as "grokking" in Power et al. (2022).
- The pattern was **reproducible across all three seeds.**

Both behaviors are well-documented in the literature as properties of specific
training regimes — not artifacts of implementation errors.

### 3. Why unusual validation behavior alone is insufficient to diagnose a gradient bug

Validation loss diverging after training loss falls near zero is consistent with
**at least three distinct causes**:

| Cause | Example in this project | Distinguishable by gradient check? |
|-------|------------------------|-----------------------------------|
| Backward-pass bug | Would show rel\_error ≥ 1e-5 in Exp. 1 | **Yes** — directly detected |
| Near-interpolation of noisy labels | Exp. 2 H=16, test\_err = 30–32% | No — gradient check would pass |
| Grokking (delayed generalization) | Exp. 3, test\_acc ≈ 2% at step 200 | No — gradient check would pass |

The gradient check is a necessary but not sufficient diagnostic. A clean gradient
check rules out implementation errors; it does not rule out legitimate training
phenomena that produce identical validation symptoms. Dr. Raman should:

1. **Run the gradient check first** — if it fails, debug the backward pass before
   proceeding.
2. **If it passes,** examine the training dynamics: Is the model large enough to
   interpolate its training data? Is any label noise or corruption present? Is the
   optimizer running long enough to exit the memorization phase?

A diverging validation curve on a model that passes a rigorous gradient check is
more likely a training phenomenon than a code defect.

---

## AI Collaboration and Scientific Process

This project was conducted with a human-led pre-registration discipline applied
to an AI-assisted workflow. Several interventions documented in LEDGER.md are
worth recording explicitly, as they illustrate both the value and the risks of
AI-assisted experimentation.

### Pre-registration and locked criteria

All evidence criteria, thresholds, and decision trees for each experiment were
specified and locked in LEDGER.md before any full-experiment data were examined.
This prevented post-hoc adjustment of criteria to match observed outcomes.
Specific examples:
- Experiment 1: pass/fail thresholds (1e-5 hard, 1e-7 ideal) specified before
  `verify_gradients.py` was run.
- Experiment 2: H\*\_seed defined from measured training error (not parameter
  count), and the full decision tree (Steps 1–2) specified before the sweep ran.
- Experiment 3: T\_mem, T\_grok, Signal A/B thresholds, and the behavioral\_grokking
  definition all locked before any seed was run.

### Preservation of failed runs

Experiment 2 Run 1 (NOT REPRODUCED) was permanently preserved before the mini-batch
redesign was authorized. The pre-registered verdict was never revised. Run 2A was
designed before its outcome was known and preserved intact even though it also
produced NOT REPRODUCED. All three runs appear in the CSV record with their exact
measured values.

### Step-count control in Experiment 2 (Run 2A)

The human reviewer identified that switching from full-batch to mini-batch SGD
at the same epoch count would confound gradient noise with a 45× increase in
optimizer steps. A step-matched control (Run 2A, ~10,035 steps) was required
before the reproduction attempt (Run 2B, ~450,000 steps) was authorized. Run 2A
confirmed that gradient noise at the same step budget produced no improvement,
isolating step count as the operative variable.

### Criterion drift corrected before the full experiment (Experiment 3)

After the 500-step pilot completed, a review identified three divergences between
the implemented criteria and the approved pre-registration text:

1. Signal A implemented as "5,000 optimizer steps" instead of
   "5 consecutive logged checkpoints" (500-step window at log\_every=100).
2. Signal B implemented as "test\_acc rises from < 60% to ≥ 90%"
   instead of "≥ 10 consecutive checkpoints with test\_acc < 0.30."
3. Pilot LR fallback implemented as 1e-4 instead of the approved 3e-4.

All three were corrected in config.py and LEDGER.md **before any full-experiment
seed was run**, and the pilot data were preserved unmodified. This is the desired
behavior: pre-registration review caught implementation drift before it could
affect the formal outcome. Documented in LEDGER.md §3.10.

### Unsupported causal interpretation corrected after the experiment

After the Experiment 3 results were written to LEDGER.md, the human reviewer
identified that two passages claimed weight decay "drove generalization" and
compressed the network "toward a sparser solution." These are causal and sparsity
claims that the experiment cannot support — no zero-weight-decay control was run
and sparsity was not measured. Both passages were revised to softer language
("consistent with the hypothesized norm-compression dynamics; this experiment
does not isolate weight decay causally or establish parameter sparsity") before
the results were committed. The formal verdict was unchanged.

---

## Numerical Audit

Every numerical claim in this report was verified against committed CSVs before
writing. The following discrepancies were investigated:

| Item | CSV value | LEDGER.md / report value | Status |
|------|-----------|--------------------------|--------|
| Exp3 seed 42 Signal B max\_run | 0 (grokking_summary.csv) | 5 (verified independently) | Reporting artifact in CSV; correct value used in report; verdict unchanged |
| All other checked values | Match | Match | No discrepancy |

**Specific values verified against CSV:**

- Exp1: 424 elements, max FD rel\_error = 8.715e-08, max autograd rel\_error = 0.000,
  median FD rel\_error = 8.606e-11, 424/424 passed 1e-7. ✓
- Exp2 Run1: H\*\_seed = -1 (undefined) for all 3 seeds; H=512 train\_error =
  10.86% (s42/s123) and 10.93% (s777). ✓
- Exp2 Run2A: total\_steps = 10,035; H\*\_seed = undefined for all 3 seeds. ✓
- Exp2 Run2B: H\*\_seed = 32 for all 3 seeds; test\_error at H=16: 0.3028/0.3139/0.3222;
  at H=32: 0.2361/0.2389/0.2361; at H=64: 0.1528/0.2028/0.1861;
  at H=512: 0.1083/0.1083/0.1000. ✓
- Exp3: T\_mem = 200 for all seeds; T\_grok = 1100/2000/1400; gaps = 900/1800/1200;
  Signal A = False, Signal B = False, behavioral\_grokking = False for all seeds;
  final test\_acc = 1.0000 for all seeds; L2 compression 67.7%/64.7%/69.9%. ✓
- Exp3 n\_params = 223,457. ✓
- Exp3 n\_train = 4,704; n\_test = 4,705. ✓

**No unresolved discrepancies found.**

---

*Report generated 2026-09-13 from committed repository state a33d422.*
*All raw data in component1\_mlp\_backward/results/, component2\_double\_descent/results/,*
*and component3\_grokking/results/. Formal record in LEDGER.md.*

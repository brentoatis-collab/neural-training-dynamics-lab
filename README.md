> **Applied AI portfolio project by Brent Oatis, MBA**
> Examining optimization behavior, convergence, and training stability for dependable model development.

# The Gradient Ledger

**Course:** CST 627 — Week 01
**Student:** boatismbair

---

## Scenario

Dr. Priya Raman is a lead ML engineer at Northwind Diagnostics. She has trained a
neural network whose training loss converges cleanly to near-zero, but whose
validation loss diverges sharply afterwards. She asks: **is this a bug in my
backward pass, or legitimate unusual generalization behavior?**

This repository builds the experimental framework needed to answer that question
rigorously — using pre-registered criteria, controlled reproductions, and a formal
ledger that preserves every run, including the negative results.

---

## Central question

> Can we distinguish a broken gradient implementation from legitimate phenomena
> like double descent and grokking by using pre-registered evidence criteria,
> step-matched controls, and multi-seed replication?

---

## Experiments at a glance

| # | Experiment | Formal verdict |
|---|-----------|---------------|
| 1 | Finite-difference gradient verification of a hand-written 2-layer MLP backward pass | **PASSED** |
| 2 | Model-wise double descent on corrupted digits (three runs) | **NOT REPRODUCED** under pre-registered criterion; qualitative DD-like curve present |
| 3 | Grokking on modular addition mod 97 (three seeds × 50,000 steps) | **NOT REPRODUCED** under pre-registered behavioral criterion; all three seeds reached 100% test accuracy |

Negative results are preserved exactly as measured. No criteria were adjusted
after seeing data.

---

## Experiment 1 — Gradient Verification

**Architecture:** 2-layer MLP, Linear(8→16)→ReLU→Linear(16→4)→MSE loss.
Hand-written forward and backward pass in pure Python/NumPy compared against:
- PyTorch autograd (float64)
- Central finite differences (h=1e-5, float64)

**Pre-registered criteria:** 100% of 424 parameter elements with relative error
< 1e-5; ≥ 95% with < 1e-7.

**Measured results:**
- 424 / 424 elements passed at < 1e-7 (100%)
- Maximum relative error vs. finite differences: **8.715e-08**
- Maximum relative error vs. autograd: **0.000** (exact match at float64)

**Verdict: PASSED.**

**What this establishes:** The backward pass is correct.
**What it does not establish:** Correctness of the backward pass says nothing
about training dynamics, generalization, or the shape of validation curves.
Those require separate experiments.

---

## Experiment 2 — Model-Wise Double Descent

**Dataset:** scikit-learn digits, 1,797 samples.
- 80/20 stratified split (split\_seed=0): n\_train=1,437, n\_test=360.
- 215 training labels (14.97%) randomly corrupted to a different class
  (corruption\_seed=0, same 215 indices for all seeds).

**Width sweep:** H ∈ {4, 8, 16, 32, 64, 128, 256, 512}, seeds=[42, 123, 777].
**Optimizer:** Full-batch SGD, lr=0.01, momentum=0, weight\_decay=0.

**Interpolation threshold H\*\_seed** [operational definition]: smallest H where
training error = 0.000 exactly (zero misclassifications on all 1,437 labels
including corrupted ones). Determined from measured training error, not
parameter count.

**Three runs — preserved in full:**

| Run | Design | Steps | H\*\_seed | Formal verdict |
|-----|--------|-------|----------|----------------|
| Run 1 | Full-batch SGD, 10,000 epochs | 10,000 | undefined (all seeds) | NOT REPRODUCED (Step 1, k=0) |
| Run 2A | Mini-batch bs=32, 223 epochs (step-matched control) | ~10,035 | undefined (all seeds) | NOT REPRODUCED — matched expectation |
| Run 2B | Mini-batch bs=32, 10,000 epochs | ~450,000 | **32** (all 3 seeds) | NOT REPRODUCED (Step 2, p=0) |

**Why Run 2A existed:** Switching from full-batch to mini-batch at the same epoch
count would have increased the step budget 45×, confounding gradient noise with
step count. Run 2A matched steps (~10,035) to isolate the noise effect. It
confirmed gradient noise alone, at the same budget, produced no improvement.

**Run 2B key results:**

| H | n\_params | test\_err s42 | test\_err s123 | test\_err s777 |
|---|----------|-------------|--------------|--------------|
| 16 (below H\*) | 1,210 | 0.3028 | 0.3139 | 0.3222 |
| **32 (H\*)** | 2,410 | 0.2361 | 0.2389 | 0.2361 |
| 64 (above H\*) | 4,810 | 0.1528 | 0.2028 | 0.1861 |
| 512 | 38,410 | 0.1083 | 0.1083 | 0.1000 |

**Formal verdict: NOT REPRODUCED.** The pre-registered criterion required test
error to peak at exactly H\*=32. The peak was observed at H=16 (the adjacent
width below H\*), where the model is near-interpolating (train\_err 0.4–1.6%).

**Qualitative secondary observation:** A non-monotone double-descent-like curve
is clearly and reproducibly present across all three seeds — test error rises
sharply at H=16, then declines monotonically from H=32 through H=512 (from ~24%
to ~10%). This qualitative shape does not satisfy the pre-registered peak-position
criterion and is reported as a secondary observation only.

---

## Experiment 3 — Grokking on Modular Arithmetic

**Task:** Predict (a + b) mod 97 for all 97² = 9,409 ordered pairs.
Random 50/50 split (split\_seed=0): n\_train=4,704, n\_test=4,705.

**Architecture:** 1-layer transformer (d\_model=128, n\_heads=4, d\_ff=512,
pre-norm, GELU). 223,457 parameters.

**Optimizer:** AdamW, lr=1e-3, betas=(0.9, 0.98), weight\_decay=1.0.
Full-batch, 50,000 steps. Seeds=[42, 123, 777].

**Pre-registered behavioral criteria:**
- **T\_mem:** earliest checkpoint with train\_acc ≥ 0.99 sustained for 5
  consecutive logged checkpoints (500-step stability window at log\_every=100).
- **T\_grok:** same for test\_acc ≥ 0.95.
- **Signal A:** T\_mem and T\_grok both defined AND T\_grok − T\_mem ≥ 2,000 steps.
- **Signal B:** ≥ 10 consecutive checkpoints strictly between T\_mem and T\_grok
  with test\_acc < 0.30 (≥ 1,000 steps of near-chance accuracy after memorization).
- **behavioral\_grokking = Signal A AND Signal B.**
- **REPRODUCED** if behavioral\_grokking = True for ≥ 2 seeds.

**Key measured results:**

| Seed | T\_mem | T\_grok | Gap | Signal A | Signal B | behavioral\_grokking |
|------|--------|---------|-----|----------|----------|---------------------|
| 42 | 200 | 1,100 | 900 | False | False | False |
| 123 | 200 | 2,000 | 1,800 | False | False | False |
| 777 | 200 | 1,400 | 1,200 | False | False | False |

All three seeds reached **test\_acc = 1.000** well before step 10,000.

**Formal verdict: NOT REPRODUCED.** Generalization occurred in every seed, but
the gap between memorization and generalization (900–1,800 steps) was below the
pre-registered 2,000-step threshold. The continuous, rapid generalization ramp
observed here does not satisfy the pre-registered delayed-plateau definition of
grokking.

**L2 norm:** The L2 weight norm declined from ~110 to ~32–37 over the full run
(65–70% compression per seed). This satisfies the pre-registered non-gating
Signal C and is consistent with hypothesized norm-compression dynamics. This
experiment does not isolate weight decay causally and does not establish parameter
sparsity.

---

## Repository structure

```
week01-gradient-ledger/
├── README.md                          ← this file
├── LEDGER.md                          ← full experiment log, pre-registrations,
│                                         deviation log, all verdicts
├── SUMMARY_REPORT.md                  ← concise cross-experiment report for Dr. Raman
├── config.py                          ← all hyperparameters (single source of truth)
├── requirements.txt
│
├── component1_mlp_backward/
│   ├── mlp.py                         ← hand-written 2-layer MLP with manual backward
│   ├── verify_gradients.py            ← gradient check runner
│   └── results/
│       └── gradient_check.csv         ← 424 rows, one per parameter element
│
├── component2_double_descent/
│   ├── dataset.py                     ← digits load, split, corruption
│   ├── model.py                       ← MLP(D_in, H, D_out)
│   ├── pilot.py                       ← convergence/runtime pilot
│   ├── sweep.py                       ← Run 1: full-batch sweep
│   ├── run2a.py                       ← Run 2A: step-matched mini-batch control
│   ├── run2b.py                       ← Run 2B: mini-batch reproduction attempt
│   └── results/
│       ├── pilot_curves.csv
│       ├── pilot_summary.csv
│       ├── double_descent_raw.csv          ← Run 1 final state (24 rows)
│       ├── double_descent_curves.csv       ← Run 1 curves (480 rows)
│       ├── double_descent_run2a_raw.csv    ← Run 2A final state
│       ├── double_descent_run2a_curves.csv ← Run 2A curves
│       ├── double_descent_run2b_raw.csv    ← Run 2B final state
│       └── double_descent_run2b_curves.csv ← Run 2B curves
│
└── component3_grokking/
    ├── dataset.py                     ← mod-97 pair generation, random 50/50 split
    ├── model.py                       ← GrokTransformer (1-layer)
    ├── pilot.py                       ← 500-step stability/throughput pilot
    ├── sweep.py                       ← 3-seed × 50,000-step full sweep
    └── results/
        ├── grokking_pilot_curves.csv
        ├── grokking_pilot_summary.csv
        ├── grokking_curves_seed42.csv      ← 500 checkpoints, seed 42
        ├── grokking_curves_seed123.csv     ← 500 checkpoints, seed 123
        ├── grokking_curves_seed777.csv     ← 500 checkpoints, seed 777
        ├── grokking_curves_all.csv         ← all 1,500 checkpoints combined
        ├── grokking_summary.csv            ← per-seed verdicts (3 rows)
        ├── grokking_aggregate.csv          ← aggregate verdict (1 row)
        ├── grokking_curves.png             ← 3-panel accuracy plot
        └── sweep_log.txt                   ← full console output
```

---

## Reproducibility

All experiments are fully reproducible from the committed code and `config.py`.
No data is downloaded at runtime — the digits dataset ships with scikit-learn.

**Fixed randomness:**
- Experiment 2: split\_seed=0, corruption\_seed=0. The same 215 corrupted indices
  are used across all sweeps and seeds.
- Experiment 3: split\_seed=0 (fixed for all three model seeds). Model seeds
  applied via `torch.manual_seed(seed)` before each run.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Experiment 1 — Gradient check

```bash
python3 component1_mlp_backward/verify_gradients.py
# Output: component1_mlp_backward/results/gradient_check.csv
```

### Experiment 2 — Double descent

```bash
# Convergence pilot (run before sweep)
python3 component2_double_descent/pilot.py

# Run 1: full-batch SGD, 10,000 epochs
python3 component2_double_descent/sweep.py

# Run 2A: step-matched mini-batch control (~10,035 steps)
python3 component2_double_descent/run2a.py

# Run 2B: mini-batch reproduction attempt (~450,000 steps, ~20 min on CPU)
python3 component2_double_descent/run2b.py
```

### Experiment 3 — Grokking

```bash
# Stability/throughput pilot (500 steps, seed=42)
python3 component3_grokking/pilot.py

# Full sweep (3 seeds × 50,000 steps, ~97 min on CPU)
python3 component3_grokking/sweep.py
# Output: component3_grokking/results/grokking_curves_seed{42,123,777}.csv
#         component3_grokking/results/grokking_summary.csv
#         component3_grokking/results/grokking_aggregate.csv
#         component3_grokking/results/grokking_curves.png
```

---

## Key documents

| File | Purpose |
|------|---------|
| [`LEDGER.md`](LEDGER.md) | Full pre-registrations, deviation log, all measured results, and formal verdicts — the authoritative scientific record |
| [`SUMMARY_REPORT.md`](SUMMARY_REPORT.md) | Concise cross-experiment synthesis, engineering diagnosis for Dr. Raman, and AI collaboration process notes |
| [`config.py`](config.py) | Single source of truth for all hyperparameters and evidence-criterion constants |

---

## AI collaboration statement

Claude Code (Anthropic) assisted with implementation throughout this project:
writing Python code, translating design decisions into working scripts, and
drafting LEDGER.md entries. However, the scientific decisions were made and
actively reviewed by the student at every stage:

- All experimental criteria, thresholds, and decision trees were specified,
  challenged, and locked by the student before any full-experiment code was run.
- Implementation drift was caught in a student-led review of the Experiment 3
  pilot: three divergences from the approved pre-registration were identified and
  corrected before the full experiment ran (documented in LEDGER.md §3.10).
- An unsupported causal claim in the Experiment 3 Signal C wording ("weight decay
  drove generalization / compressed the network toward a sparser solution") was
  identified by the student and corrected before the results were committed.
- The step-matched control design for Experiment 2 (Run 2A) was required by the
  student after identifying the 45× step-count confound in the original proposal.
- Negative results in all three experiments were preserved intact rather than
  adjusted to achieve better outcomes.

The AI served as an implementation and drafting assistant. Experimental design,
criterion specification, auditing, corrections, and acceptance decisions were the
student's work.

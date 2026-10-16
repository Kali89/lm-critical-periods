# Pre-registration: sensitive periods for input structure in small language models

*Format follows the OSF Preregistration template (Study Information, Design Plan, Sampling Plan, Variables,
Analysis Plan). Items marked `[FILL: ...]` are computed from the clean baselines by `bin/lmcp prereg` and
must be filled, committed, and the commit hash recorded on OSF before any corrupted arm is queued.*

## Study information

**Title.** Does corrupting input structure during an early window of pretraining leave a permanent,
ability-specific deficit in small language models?

**Background.** Achille, Rovere & Soatto (ICLR 2019) showed that a temporary input deficit (image blur)
applied during the early phase of training a deep network leaves a permanent accuracy deficit, tracked by the
rise and fall of the Fisher information trace; label permutation, which preserves input statistics, leaves
none. Constantinescu, Pimentel, Cotterell & Warstadt (TACL 2025) found no critical period for the *identity*
of the language in LM pretraining: a second language introduced late is learned as well as one introduced
early. The untested case between these results is the *structure* of the input in language models.

**Hypotheses.** All directional, all evaluated against a clean baseline with the same model, token budget,
learning-rate schedule and seed set.

- **H1 (sensitive period).** Shuffling word order within sentences during the window [0, w] of the token
  budget, followed by clean training, lowers the primary endpoint at the end of training relative to (a) the
  clean baseline and (b) the matched-clean-tokens control (a clean run of (1 − w) × budget tokens). The
  same-length window placed late ([1 − w − ε, 1 − ε]) produces a smaller end-of-training deficit than the early
  window after the same amount of clean recovery, i.e. the deficit depends on onset, not only on recency.
- **H2 (ability specificity).** The sentence-shuffle deficit is concentrated on syntactic endpoints (BLiMP
  agreement terms) and absent on in-context copying (induction accuracy). The short-context deficit shows the
  reverse pattern.
- **H3 (negative control, predicted null).** A consistent permutation of the vocabulary during the same
  window leaves no deficit at end of training (equivalence test, margin `[FILL: 0.5 × seed SD of primary]`).
- **H4 (dose response).** Within the early window, the deficit increases monotonically with corruption
  intensity ∈ {0.25, 0.5, 1.0}.
- **H5 (plasticity timing).** The sensitive window closes where the clean baseline's Fisher trace has fallen
  to 50% of its peak. Predicted closing point: `[FILL: frac from predict_window_from_plasticity]`.
- **H0 (instrument check, positive control).** The same window code applied to CIFAR-10 ResNet-18 blur
  reproduces Achille et al.: deficit for blur in [0, 0.5] of training that exceeds the seed SD, and no deficit
  for label permutation. If H0 fails, no language-model result from this harness is reported as evidence
  either way.

**Operational definition of "permanent".** A deficit is permanent if, in the extended-recovery arm (same
window in absolute tokens, 2 × the clean recovery tokens), the 95% CI of the deficit excludes zero AND the
slope of the deficit over the second half of recovery is not significantly negative (95% CI of the slope
includes zero or is positive). Otherwise it is reported as "slow recovery".

## Design plan

Between-run design; each run is one (condition, seed) cell trained from random initialisation. No blinding
(automated pipeline; evaluation code frozen and hashed before the grid runs). Randomisation: seed sets model
initialisation, document order (a seed-specific permutation of the corpus, saved per run), label-noise draws
and stream mixing. Learning rate is constant (warmup 2%) in all critical-period arms so that a deficit cannot
be attributed to late data arriving at a small learning rate.

Conditions (grid file `configs/grid/pilot.yaml`, commit `[FILL: hash]`): clean; sentence shuffle at
windows [0, 0.10], [0, 0.25], [0, 0.50], [0.50, 0.75]; sentence shuffle at intensity 0.25 and 0.5 in
[0, 0.50]; short context (32-token blocks) at [0, 0.25], [0, 0.50], [0.50, 0.75]; vocabulary permutation at
[0, 0.50]; matched-clean-tokens controls at 50% budget; extended-recovery arm for the primary window.

## Sampling plan

Simulated data; no existing data are reused. Sample size = seeds per cell, chosen from a simulation-based
power analysis on the clean baselines' seed-to-seed SD of the primary endpoint: `[FILL: power table]`.
Minimum detectable effect targeted: 1 seed SD at 80% power. Seeds per cell: `[FILL: n]`. Stopping rule: the
grid is run to completion; no cells are added or dropped after the first corrupted run starts. Additional
seeds may be added only by a dated amendment to this document before they are queued.

## Variables

**Manipulated.** Corruption kind, window (onset, offset as fractions of the token budget), intensity.
**Measured at every eval.** Validation loss by position bucket and long-range gain; BLiMP accuracy per
linguistic term with item-level bootstrap CIs; induction accuracy and gain; Fisher trace (sampled and
empirical), gradient and parameter norms, effective rank of residual stream and embeddings, dead-unit
fraction. For the vision control: test accuracy.
**Primary endpoint.** `[FILL: metric]`, designated from the clean baselines as the ability with the largest
(mean − chance) / seed SD among those the m-size model clearly acquires (`bin/lmcp analyze baseline`). The
designation is made before any corrupted arm runs.
**Secondary endpoints.** The remaining per-ability metrics, Holm-corrected within each hypothesis.

## Analysis plan

Mixed-effects model per endpoint: `metric ~ condition` with a random intercept for seed (document order is
seed-linked by design; if a document-order-only replication is added it enters as a variance component).
Contrasts: each arm vs clean; early vs late window; corrupted vs matched-clean-tokens. Dose response: OLS of
endpoint on intensity within the [0, 0.5] window. Negative control: two one-sided tests against the margin
above. Inference threshold α = 0.05 after Holm correction within hypothesis; effect sizes reported in seed-SD
units with 95% CIs. Exclusions: runs whose config records a dirty git tree, NaN loss, or throughput below
50% of the measured norm (indicates a contended GPU) are re-run, never silently dropped; all exclusions are
listed. Exploratory: per-head induction scores and local learning coefficient on exported checkpoints,
landmark alignment with Pythia-14M/70M, reported as exploratory.

## What would falsify the claim

No deficit beyond the matched-token control at any window (H1 false), or a deficit that closes under
extended recovery (slow recovery, not sensitive period), or an equal deficit for the late window (recency),
or a deficit for the vocabulary-permutation control (the harness measures disruption, not structure).

## Filled values

`[FILL: paste output of bin/lmcp prereg here, with the baseline run names and their git revision]`

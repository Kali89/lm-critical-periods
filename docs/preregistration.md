# Pre-registration: sensitive periods for input structure in small language models

*Format follows the OSF Preregistration template (Study Information, Design Plan, Sampling Plan, Variables,
Analysis Plan). Numbers under "Filled values" are computed from the clean baselines by `bin/lmcp prereg`. This
document is committed, and the commit hash recorded on OSF, before any corrupted arm is queued; afterwards it
changes only by dated amendment.*

*Filled 2026-10-17 from the clean baseline ladder (`configs/grid/baselines.yaml`, 9 runs, harness revision
`52d1abb`; readout in `docs/baseline-readout-2026-10-17.md`) and revised 2026-10-19 after two independent
design reviews (endpoint family, controls, permanence, power; see `docs/review-pack-2026-10-19.md` if kept).
The baseline checkpoints were re-scored with the revised evaluation by the code in the registration commit itself (each `rescore.jsonl` line records the revision and dirty flag at the time).
Operating point: size m (6 layers, d=384, ~14M parameters), 600M tokens (4.3 epochs of the 138M-token
BabyLM-100M stream), constant LR 3e-4 with a 12M-token warmup, batch 32k tokens, sequence length 512.
Window fractions below refer to this 600M budget. The registered grid is `configs/grid/pilot.yaml` in the
same commit as this document.*

## Study information

**Title.** Does temporary corruption of input structure during different stages of language-model
pretraining leave a persistent, structure-specific deficit after matched clean recovery?

**Background.** Achille, Rovere & Soatto (ICLR 2019) showed that a temporary input deficit (image blur)
during the early phase of training a deep network leaves a lasting accuracy deficit, while interventions that
preserve low-level input statistics (label permutation, vertical flip) leave none; Kleinman, Achille & Soatto
(ICLR 2024) show the same phenomenon in deep linear networks, where it follows from competition between
features during learning. Constantinescu, Pimentel, Cotterell & Warstadt (TACL 2025) found no critical period
for the *identity* of the language in LM pretraining: a second language introduced late is learned as well as
one introduced early. Related evidence that early optimisation history constrains later learning comes from
loss of plasticity in continual learning (Dohare et al., Nature 2024) and the primacy bias in deep RL
(Nikishin et al., ICML 2022). Closest in setting, Dini, Domenichelli, Brunato & Dell'Orletta (Knowledge-Based
Systems 342, 2026) vary only the *order* of pretraining data in encoder- and decoder-only models and find that
ordering acts as a selective inductive bias rather than a performance lever, that decoder models are the more
stable architecture, and that early probing differences reflect differences in exposure rather than faster
acquisition; the no-exposure reference below exists to separate those two readings. The untested case between
these results is the *structure* of the input in language models: word order within sentences, sentence order across a document, and the span of context the
model can attend to. Two measurement cautions shape the design: Olsson et al. (2022) define an in-context
learning score from the loss difference between early and late sequence positions and tie its emergence to
induction heads, so position-bucket losses are not a pure measure of long-range structure; and Schaeffer,
Miranda & Koyejo (NeurIPS 2023) show that apparent discontinuities in ability can be artefacts of the metric.

**Framing.** This study is pre-registered after clean-baseline pilot work: the clean runs that fixed the
architecture, token budget, learning rate, windows and endpoint precision were run before this document was
finalised (their provenance is under Filled values), and no corrupted arm has been run. The claim under test
is training-stage-dependent vulnerability to temporary structural corruption. "Sensitive period" is used in that operational sense; the biological analogy is motivation, not
a claim. Four claims are kept apart throughout: (i) the model has acquired an ability; (ii) the measured
benefit from that ability has stopped increasing; (iii) the model's capacity to learn that ability has
diminished; (iv) a temporary corruption leaves an enduring impairment. The clean baselines speak to (i) and
(ii) at the checkpoints measured. The experiment tests (iv), and infers (iii) only from the pattern across
onset, recovery and structure specificity together: a late corruption without deficit may show robustness
of established representations rather than reduced plasticity, and an early deficit may reflect path
dependence of optimisation rather than a closed window. The write-up uses that language.

**What each test licenses.** H1 establishes an early-corruption deficit at 600M tokens, nothing more. H1b
establishes onset-dependent vulnerability, subject to both of its recovery comparisons. H2 establishes
differential structural effects. The extended-recovery arm establishes persistence through 1.0B tokens for
word-order shuffle only. No single significant result is evidence for the sensitive-period reading; that
reading requires H1, H1b and H2 together.

**Hypotheses.** All evaluated against the clean reference with the same model, token budget, learning-rate
schedule, warmup in absolute tokens, and seed set. Seed fixes initialisation and document order, so every
contrast is paired by seed. Three endpoints are defined on the *same* target tokens (positions 128-511 of each
512-token validation sequence, every 8th position from 128, 48 targets per sequence): the full-context loss
L_full, the local loss L_32 obtained by re-running the 32 tokens preceding each target as a fresh sequence
(exactly 32 tokens of context, in a configuration the model trained in), and the context gain
G = L_32 − L_full (how much the model benefits from context beyond 32 tokens). A masked sliding window was
rejected during instrument checks because its effective context grows with depth and hiding position 0
removes the attention sink (`docs/context-eval-check-2026-10-19.md`). Each corruption is mapped to
its endpoint by mechanism, fixed here before the baselines were re-scored:

| corruption (window [0, 0.333]) | what it removes | primary endpoint | support |
|---|---|---|---|
| `shuffle_sentence` (words shuffled within sentences) | within-sentence order | L_32 (higher = worse) | BLiMP overall |
| `shuffle_doc_sentences` (sentence order shuffled, syntax kept) | cross-sentence structure | G (lower = worse) | L_full |
| `short_context` (32-token attention blocks while training) | use of distant context | G (lower = worse) | L_full |

- **H1 (stage-dependent persistent deficit).** For each of the three corruptions applied in the primary
  window [0, 0.333] (first 200M tokens) and followed by 400M clean tokens, the mapped endpoint at 600M is worse
  than the clean reference at 600M, and worse than the no-exposure reference (the clean reference's own
  evaluation at 400M tokens, which under a constant learning rate and fixed warmup is a clean run of that
  length: it answers whether 200M corrupted tokens are worse than never having seen them). Decision rule:
  the three confirmatory tests are the corrupted-vs-clean contrasts at 600M, one per corruption, each a
  one-sided Welch t-test of the 4 corrupted runs against the 6 clean runs, Holm-corrected as one family; H1
  is supported for a corruption only if its test rejects. Welch rather than a paired test is registered
  because the within-seed correlation between a seed's clean and corrupted outcomes is unknown and the
  simulated procedure (Sampling plan) shows the paired test, with 3 degrees of freedom, has lower power
  unless that correlation exceeds about 0.5; the paired analysis over seeds 0-3 is reported as a sensitivity
  analysis. The no-exposure comparison is a qualifier, not a test: for a
  supported corruption, the no-exposure deficit is formed in the harmful direction for its endpoint, same
  seed: L_32,corrupt(600M) − L_32,clean(400M) for the local loss, and G_clean(400M) − G_corrupt(600M) for the
  context gain, positive when the corruption is worse than never having seen those 200M tokens. It is
  reported with its 95% CI, and the deficit is labelled "beyond no exposure" only if that CI excludes zero
  on the positive side, otherwise "relative to clean, within the cost of the lost clean tokens". The 600M outcome is the primary finding.
  "Persistent after extended recovery" (defined below) can be claimed only for word-order shuffle, the sole
  corruption with an extended-recovery arm; for the other two it is not assessed in this grid.
- **H1b (onset, not recency).** For `shuffle_sentence`, the 50M window [0, 0.083] produces a larger
  end-of-training deficit on L_32 than the same-length late window [0.5, 0.583], and the 150M window
  [0.083, 0.333] a larger one than [0.5, 0.75]; for `short_context`, [0, 0.333] a larger deficit on G than
  [0.5, 0.75]. Two estimands are registered because the early and late arms differ in both onset and in clean
  tokens after the window: (i) the deficit at the common 600M endpoint, and (ii) the deficit at an equal
  number of clean tokens after the window closes (250M for the 50M pair, 150M for the 150M pair), read from
  the 25M-token evaluation trajectory. Under either estimand the deficit is the paired difference against
  the same seed's clean run at the same token count (the clean runs are evaluated every 25M tokens, so a
  matched-time clean reference exists at every point). Onset dependence is claimed only if the early arm's
  deficit is larger under both estimands. If the late arm is worse at 600M but recovers faster per clean
  token, the result is reported as a recovery-time effect, not an onset effect. Neither estimand alone
  isolates timing: the equal-recovery comparison equalises clean exposure but compares models of different
  training age. The onset claim rests on the combined pattern of onset, recovery and structure specificity.
- **H2 (structure specificity, double dissociation).** Deterioration is defined per seed s, in the harmful
  direction for each endpoint: D_L(s) = L_32,corrupt(s) − L_32,clean(s) and D_G(s) = G_clean(s) −
  G_corrupt(s), both positive when the corruption hurts. Each is scaled by a fixed constant, its registered
  smallest effect of interest (0.016 nats for L_32, 0.0175 nats for G), not by a seed SD, so the two
  endpoints are compared in units of "smallest effects of interest". Each corrupted run r gets a
  specificity score S(r) = D_L(r)/0.016 − D_G(r)/0.0175, where D is computed against the clean mean over the
  6 clean seeds (the clean mean cancels in the contrast below, so its uncertainty does not enter). The
  interaction contrast is I = mean S over the 4 `shuffle_sentence` runs − mean S over the 4
  `shuffle_doc_sentences` runs. Prediction: I > 0 (word-order shuffle deteriorates local prediction more
  than context use, relative to what sentence-order shuffle does). Test: one-sided Welch t-test of S between
  the two corruptions (4 vs 4), α = 0.05, a single test with no further multiplicity. Sensitivity: the same
  contrast formed within seed, I(s) = S_shuffle_sentence(s) − S_shuffle_doc_sentences(s), tested by a
  one-sided paired t over seeds 0-3. All S and D values are reported in original units with CIs, and the
  across-run correlation of D_L and D_G is reported. H2's power is not simulated because the interaction
  magnitude has no baseline anchor; its minimum detectable effect at 80% power with 4 vs 4 and the observed
  SD of S is reported with the result. The
  interaction test, not two separate significance tests, is the confirmatory test. Interpretation rule:
  I > 0 establishes *differential* effects of the two corruptions across the two endpoints, and that is what
  H2 claims. The phrase "double dissociation" is used only if, in addition, the individual estimates show
  the crossover pattern descriptively: D_L larger than D_G (in scaled units) for word-order shuffle and D_G
  larger than D_L for sentence-order shuffle, each with its CI reported. A significant I produced by
  word-order shuffle harming both endpoints while sentence-order shuffle harms neither is reported as
  differential sensitivity, not as a dissociation. `short_context` is reported in the same form against
  `shuffle_sentence` as a secondary row. These are predicted preferential
  effects: sentence-order
  shuffling also disturbs local prediction around sentence boundaries and word-order shuffling may also
  disturb cross-sentence representations, so a partial dissociation is reported as partial.
  Secondary, within BLiMP: the `shuffle_sentence` deficit concentrates on phenomena still rising between
  150M and 600M in the clean baselines (plateau, by the H5 definition, at 449-600M: NPI licensing,
  quantifiers, anaphor agreement, control/raising, ellipsis) rather than those saturated early (plateau at
  100-175M: determiner-noun agreement, binding, filler-gap dependency).
  Secondary, induction: the mid window [0.083, 0.333], which covers induction formation in the baselines,
  lowers induction gain (nats, first minus second presentation of a repeated random sequence) more than the
  early window [0, 0.083] does. Induction accuracy is reported, not tested (seed SD 0.055 on a mean of 0.24).
  The position-bucket measure long_range_gain (loss at positions 0-31 minus 128-511) is retained as a
  descriptive statistic only; it is Olsson et al.'s in-context learning score and is expected to co-vary with
  induction gain.
- **H3 (lexical-remapping control).** A fixed permutation of the token vocabulary during [0, 0.333]
  preserves every structural statistic of the input and scrambles token identity; under tied embeddings the
  model must re-learn the embedding table when the permutation is lifted. Both outcomes are registered: no
  deficit at 600M (two one-sided tests on L_32 and G against the margins under Sampling plan) means a 200M
  disruption of the lexical interface is repairable and the structural deficits in H1 are not disruption per
  se; a deficit means lexical remapping is itself a substantial intervention. Neither outcome falsifies H1.
  If the remapping deficit on L_32 is as large as the `shuffle_sentence` deficit, the H2 interaction becomes
  the decisive test of structure specificity. This control is not the analogue of Achille et al.'s label
  permutation and is not described as such.
- **H4 (dose response).** Within [0, 0.333], the L_32 deficit increases monotonically with the fraction of
  sentences shuffled, intensity ∈ {0 (clean), 0.5, 1.0}.
- **H5 (timing from ability landmarks), amended before registration.** The original plan predicted the window
  from the Fisher trace's rise and fall (Achille et al.). In the clean baselines the sampled Fisher trace is
  largest at initialisation, falls to ~12 by 25M tokens and is flat to 1.4B at a constant learning rate, so it
  predicts nothing. H5 is restated on ability landmarks measured relative to chance (BLiMP: 0.5) or to the
  first evaluation (losses, gains): the half point is where the clean mean has acquired half of its final
  above-baseline range; the plateau is the first evaluation after which the clean mean stays within 10% of
  that range of its final value. The sensitive window for an ability closes at its plateau. Landmarks from
  the baselines (two seeds, provisional, re-estimated from the six clean seeds of this grid):
  (M tokens, evals every 25M; the context endpoints L_32 and G have no baseline trajectory and are added from
  this grid's clean seeds by amendment):

  | ability | half point | plateau |
  |---|---|---|
  | BLiMP overall | 25 | 200 |
  | long_range_gain (descriptive) | 50 | 125 |
  | induction gain | 75 | 150 |
  | validation loss | 25 | 50 |

  From the re-scored final checkpoints (150M, 600M, 1.4B; `docs/context-eval-check-2026-10-19.md`): context
  gain G is 0.17-0.18 nats at all three points, i.e. the measured benefit is stable from 150M onward. This
  establishes stabilisation of the endpoint, not when G emerged and not that the capacity to learn it has
  closed; the clean runs of this grid (evaluated every 25M tokens) will supply G's emergence curve, and the
  landmark table will be completed by dated amendment from them. The local loss L_32 is still improving at
  1.4B. Predicted closing points (fraction of 600M), as falsifiable predictions rather than baseline facts:
  G by 0.25, BLiMP by 0.33, induction by 0.25, local structure not closed within the budget. Concretely:
  corruption confined to [0.5, 0.75] is predicted to leave no persistent deficit on G or BLiMP; on L_32 the
  late window is predicted to leave a smaller deficit than the early window (H1b), not none, because local
  prediction is still being learned there. The Fisher trace is still recorded
  and reported as exploratory.
- **H0 (instrument check, positive control), completed 2026-10-18.** The same window code applied to
  CIFAR-10 ResNet-18 blur reproduced Achille et al. with 3 seeds per arm (`docs/vision-control.md`): deficits
  of 1.4, 4.0 and 9.4 points for blur in the first 20, 40 and 80 epochs, −0.1 for label permutation, 2.4 for
  the same 40 epochs placed late under the paper's decaying schedule. This validates the window machinery and
  the ability of the harness to detect a sensitive period; it does not validate the language corruptions or
  the language evaluations, which the controls above address.

**Operational definition of "persistent".** Assessed for word-order shuffle only. A deficit is *persistent
after extended recovery* if, in the extended-recovery arm (first 200M tokens shuffled, then 800M clean, 2 × the standard 400M), the paired
difference against the clean 1.0B reference at the 1.0B endpoint has a 95% CI excluding zero. Recovery
dynamics are reported alongside: the deficit trajectory over the clean phase (every 50M tokens for the
extended arm, 25M for the reference) with a fitted exponential recovery and its half-life. "Recovered" is
claimed only by equivalence (TOST within the margin under Sampling plan), never by failure to reject zero.
"Permanent" is not used: irreversibility cannot be established from a finite run.

## Design plan

Between-run design; each run is one (condition, seed) cell trained from random initialisation. No blinding
(automated pipeline; evaluation code frozen and hashed before the grid runs). Randomisation: seed sets model
initialisation, document order (a seed-specific permutation of the corpus, saved per run, independent of the
token budget), and stream mixing. Learning rate is constant in all arms so that a deficit cannot be
attributed to late data arriving at a small learning rate. Warmup is held at 12M tokens in absolute terms
(0.02 × 600M and 0.012 × 1.0B both give 366 optimiser steps), so a 1.0B clean run is identical to a 600M
clean run through its first 600M tokens, step for step; its 600M evaluation is therefore the 600M clean
reference and its 1.0B evaluation the extended-recovery reference.

Conditions (grid file `configs/grid/pilot.yaml`, committed together with this document; 50 runs):
clean reference, 6 seeds, 1.0B tokens; `shuffle_sentence` at windows [0, 0.083], [0, 0.333],
[0.083, 0.333], [0.5, 0.75], [0.5, 0.583]; `shuffle_sentence` at intensity 0.5 in [0, 0.333];
`shuffle_doc_sentences` at [0, 0.333]; `short_context` (32-token blocks) at [0, 0.333] and [0.5, 0.75];
`vocab_permute` at [0, 0.333]; extended recovery: `shuffle_sentence` in the first 200M tokens of a 1.0B
budget, seeds 0-3, paired with clean seeds 0-3 at 1.0B. Four seeds per corrupted cell. The no-exposure
reference is read from the clean runs' evaluations at (1 − w) × 600M tokens, so BLiMP is evaluated on all
1000 pairs per paradigm at every evaluation.

Corruption streams are produced before tokenisation from the same document file, document for document, and
read in the same seed-specific order; the `shuffle_doc_sentences` stream keeps each line's sentence count so
newline tokens are preserved. Token counts: clean 138.23M, `shuffle_sentence` 138.68M (+0.3%),
`shuffle_doc_sentences` 138.06M (−0.1%). The reader is document-indexed: all streams share the document ids
and the switch from corrupted to clean happens at the same document position, so a corrupted arm reaches any
given document within ±0.3% of the token count at which its clean twin does; the token budget, not the
document count, is fixed. Every evaluation records the readers' document positions, so document exposure at
each window boundary is audited per arm and seed from the run logs and reported. The `short_context` corruption uses block-aligned 32-token
attention (tokens attend within their own block; RoPE positions are not reset), while L_32 is computed from
fresh 32-token sequences; the evaluation deliberately does not mirror the training-time corruption.

## Sampling plan

Simulated data; no existing data are reused. Sample size = seeds per cell: 4 for every corrupted cell, 6 for
the shared clean reference (all 6 enter the Welch confirmatory tests; seeds 0-3 are also paired with every
corrupted cell for the sensitivity analyses and the no-exposure and matched-time references). Power was
simulated for the registered confirmatory procedure itself: the three H1 tests (word-order shuffle on L_32,
sentence-order shuffle on G, short context on G) at their smallest effects of interest, one-sided Welch
6-vs-4, Holm across the three, at the clean seed SD measured from the re-scored baselines and at 2 × and 3 ×
that SD (the SD comes from two seeds), and at within-seed correlations of −0.5, 0 and +0.5 between a seed's
clean and corrupted outcomes; the paired-t procedure was simulated alongside and is the sensitivity analysis
because it wins only when that correlation exceeds about 0.5 (tables under Filled values). Summary, power
after Holm at ρ = 0, 1x / 2x / 3x SD: word-order shuffle on L_32 1.00 / 0.90 / 0.63, sentence-order shuffle on G 1.00 / 1.00 / 1.00, short context on G
1.00 / 1.00 / 1.00, all three jointly 1.00 / 0.90 / 0.63. The L_32 test is the limiting one; the paired-t
sensitivity figures are lower at ρ ≤ 0 (0.75 and 0.47 at 2x and 3x SD) and only overtake Welch at ρ = 0.5. Smallest effects of interest, which also serve as equivalence margins for H3 and for
"recovered", are fixed in the metric's own units from the clean baselines, not from seed SDs: L_32 0.016 nats
(10% of what 450M further clean tokens buy in local loss, 3.75 → 3.59 from 150M to 600M); G 0.0175 nats (10%
of the clean model's context gain, 0.175); BLiMP 0.010 (one point, 5% of the above-chance range). Power at
these effects, the unpaired Welch 6-vs-4 sensitivity figures at 1x/2x/3x the measured seed SD are:
L_32 (SD 0.0037) 1.00 / 0.79 / 0.46; G (SD 0.0012) 1.00 / 1.00 / 1.00; BLiMP (SD 0.0082) 0.34 / 0.12 / 0.07.
With four seeds and SDs estimated from two, the variance and correlation estimates are themselves uncertain,
so the registered power figures are indicative, not guarantees. The smallest effects of interest are chosen
thresholds anchored to clean learning (what further clean tokens buy; 10% of the clean context gain), not
estimates of what the corruptions will do; nothing yet shows the interventions produce effects of that size.
The write-up therefore separates power to detect the pre-specified effect, the effect actually observed, and
the precision (CI width) available for smaller effects, and never equates high power at the threshold with
high power for any genuine sensitive-period effect. BLiMP is not powered for a one-point effect at four seeds: its minimum detectable effect at 80% power is about
1.7 points at the baseline SD and 3.5 points at twice it, so BLiMP serves as support for the word-order
contrast and the phenomenon-level predictions in H2 are secondary. Effects smaller than the minimum detectable
effect at 80% power are reported as inconclusive, not as null. Stopping rule: the
grid is run to completion; no cells are added or dropped after the first corrupted run starts. Additional
seeds may be added only by a dated amendment to this document before they are queued.

## Variables

**Manipulated.** Corruption kind, window (onset, offset as fractions of the token budget), intensity.
**Measured at every eval.** On 48 fixed validation targets per sequence (positions 128, 136, ..., 504):
L_full, L_32 (fresh 32-token context), G, each with per-sequence bootstrap CIs; validation loss overall and by position bucket, long_range_gain (descriptive); BLiMP accuracy
overall and per linguistic term (the dataset's 12 official phenomena, each the unweighted mean of its
paradigms' accuracies, 1000 pairs per paradigm, full-sentence summed log-probability, prefixed with the EOS
token, lower-cased to match the corpus), with item-level bootstrap CIs for description only; induction gain
and accuracy; Fisher trace (sampled and empirical), gradient and parameter norms, effective rank of residual
stream and embeddings, dead-unit fraction; the readers' document positions (exposure audit). For the vision
control: test accuracy.
**Primary endpoints.** The three mapped endpoints in the table above, at the 600M evaluation. The mapping is
fixed by mechanism, before the baselines were re-scored; baseline numbers are used only for precision and
power.
**Secondary endpoints.** BLiMP overall; the five rising and three early-saturating phenomena named in H2;
induction gain; L_full. Holm-corrected within each hypothesis.

## Analysis plan

The experimental unit is the training run (one seed in one condition); item-level counts (BLiMP pairs,
validation sequences) describe measurement precision and never enter the inference as replicates.
Confirmatory tests are the Welch t-tests specified in H1 and H2, with the paired versions over seeds 0-3 as
sensitivity analyses. Estimates and CIs for every
contrast come from a mixed-effects model per endpoint, `metric ~ condition` with a random intercept for seed,
fitted to all runs including clean seeds 4-5. Contrasts: each arm vs clean at 600M (H1 family, Holm across
the three mapped endpoints); corrupted vs no-exposure reference (qualifier); early vs late at 600M and at
equal clean tokens after the window (H1b, both estimands, matched-time clean reference by seed). H2: the
paired interaction contrast I defined above, with `short_context` as a secondary row. H3: TOST on L_32 and G at the margins above. H4: OLS of L_32 on intensity within
[0, 0.333] (three levels including clean) with seed as a random intercept. Persistence: paired difference at
1.0B against the clean 1.0B reference, plus the fitted recovery trajectory. Correlated endpoints (G and
induction gain, L_32 and BLiMP) are reported with their across-seed correlation in the clean arm.
Inference threshold α = 0.05 after Holm correction within hypothesis; effect sizes reported in the metric's
own units and in clean seed-SD units with 95% CIs. Exclusions: runs whose config records a dirty git tree,
NaN loss, or throughput below 50% of the measured norm (a contended GPU) are re-run, never silently dropped;
all exclusions are listed. Exploratory: per-head induction scores and local learning coefficient on exported
checkpoints, landmark alignment with Pythia-14M/70M, Fisher trace, long_range_gain; the fraction of L_32
targets whose 32-token context lies within a single sentence (at 15-20 tokens per sentence the context beyond
32 tokens almost always crosses a sentence boundary, so G is by construction mostly cross-sentence; this is
reported, and G is described as the predictive benefit of a longer prefix, not as discourse competence).

## What would falsify the claim

No corruption's mapped endpoint shows a deficit beyond the no-exposure reference at 600M (H1 false); the
deficit is absent at 1.0B or recovered by equivalence (recovering, not persistent); the late window matches
the early window under both estimands (recency or recovery time, not onset); or the corruption × endpoint
interaction is null or reversed (a general disruption effect, not structure-specific). A lexical-remapping
deficit alone does not falsify H1 (see H3).

## Filled values

Output of `bin/lmcp prereg --grid baselines --write` on 2026-10-19, after `bin/lmcp rescore runs/baselines__clean__wnone__* --force`
under the revised evaluation (fresh-context L_32 at stride 8). Baseline runs (trained at harness revision `52d1abb`):

- `baselines__clean__wnone__m__const__s0__3742cb` (600M)
- `baselines__clean__wnone__m__const__s0__481d3b` (1.4B)
- `baselines__clean__wnone__m__const__s0__a14b09` (150M)
- `baselines__clean__wnone__m__const__s1__4fa788` (150M)
- `baselines__clean__wnone__m__const__s1__b08f0f` (600M)
- `baselines__clean__wnone__s__const__s0__154386` (600M)
- `baselines__clean__wnone__s__const__s0__633f90` (150M)
- `baselines__clean__wnone__s__const__s1__64e8fe` (150M)
- `baselines__clean__wnone__s__const__s1__670d90` (600M)

The sections under "Historical (superseded 2026-10-19)" at the end are the tool's earlier signal-to-noise
proposal, equal-n power table and Fisher-based window; they have no authority and are kept for the audit
trail. Landmarks under the registered definition are in `docs/landmarks-baselines-2026-10-19.md`.

## Baseline summary (clean runs; seed SD, snr = |mean - chance| / SD)

| model.size | train.schedule | train.tokens | metric | mean | sd | n | chance | snr |
|---|---|---|---|---|---|---|---|---|
| m | const | 150000000 | val_loss | 3.620 | 0.002 | 2 | 9.011 | 2411.132 |
| m | const | 150000000 | val_loss_ctx_32 | 3.754 | 0.003 | 2 | 9.011 | 2001.257 |
| m | const | 150000000 | val_loss_ctx_full | 3.582 | 0.002 | 2 | 9.011 | 3176.713 |
| m | const | 150000000 | blimp_acc | 0.684 | 0.006 | 2 | 0.500 | 30.565 |
| m | const | 150000000 | induction_acc | 0.222 | 0.052 | 2 | 0.000 | 4.235 |
| m | const | 150000000 | induction_gain | 6.570 | 0.479 | 2 | 0.000 | 13.722 |
| m | const | 150000000 | long_range_gain | 0.449 | 0.002 | 2 | 0.000 | 222.767 |
| m | const | 150000000 | context_gain | 0.171 | 0.001 | 2 | 0.000 | 186.689 |
| m | const | 600000000 | val_loss | 3.458 | 0.001 | 2 | 9.011 | 11103.734 |
| m | const | 600000000 | val_loss_ctx_32 | 3.594 | 0.004 | 2 | 9.011 | 1471.249 |
| m | const | 600000000 | val_loss_ctx_full | 3.422 | 0.002 | 2 | 9.011 | 2281.119 |
| m | const | 600000000 | blimp_acc | 0.707 | 0.008 | 2 | 0.500 | 25.086 |
| m | const | 600000000 | induction_acc | 0.238 | 0.055 | 2 | 0.000 | 4.368 |
| m | const | 600000000 | induction_gain | 6.589 | 0.267 | 2 | 0.000 | 24.652 |
| m | const | 600000000 | long_range_gain | 0.471 | 0.002 | 2 | 0.000 | 296.117 |
| m | const | 600000000 | context_gain | 0.172 | 0.001 | 2 | 0.000 | 139.767 |
| m | const | 1400000000 | val_loss | 3.394 | nan | 1 | 9.011 | nan |
| m | const | 1400000000 | val_loss_ctx_32 | 3.531 | nan | 1 | 9.011 | nan |
| m | const | 1400000000 | val_loss_ctx_full | 3.355 | nan | 1 | 9.011 | nan |
| m | const | 1400000000 | blimp_acc | 0.713 | nan | 1 | 0.500 | nan |
| m | const | 1400000000 | induction_acc | 0.177 | nan | 1 | 0.000 | nan |
| m | const | 1400000000 | induction_gain | 5.717 | nan | 1 | 0.000 | nan |
| m | const | 1400000000 | long_range_gain | 0.488 | nan | 1 | 0.000 | nan |
| m | const | 1400000000 | context_gain | 0.176 | nan | 1 | 0.000 | nan |
| s | const | 150000000 | val_loss | 3.762 | 0.007 | 2 | 9.011 | 790.781 |
| s | const | 150000000 | val_loss_ctx_32 | 3.894 | 0.004 | 2 | 9.011 | 1140.548 |
| s | const | 150000000 | val_loss_ctx_full | 3.726 | 0.011 | 2 | 9.011 | 494.833 |
| s | const | 150000000 | blimp_acc | 0.658 | 0.005 | 2 | 0.500 | 28.687 |
| s | const | 150000000 | induction_acc | 0.089 | 0.047 | 2 | 0.000 | 1.907 |
| s | const | 150000000 | induction_gain | 4.704 | 0.901 | 2 | 0.000 | 5.222 |
| s | const | 150000000 | long_range_gain | 0.431 | 0.007 | 2 | 0.000 | 59.232 |
| s | const | 150000000 | context_gain | 0.169 | 0.006 | 2 | 0.000 | 27.233 |
| s | const | 600000000 | val_loss | 3.644 | 0.000 | 2 | 9.011 | 54970.838 |
| s | const | 600000000 | val_loss_ctx_32 | 3.780 | 0.001 | 2 | 9.011 | 3664.618 |
| s | const | 600000000 | val_loss_ctx_full | 3.609 | 0.001 | 2 | 9.011 | 3973.483 |
| s | const | 600000000 | blimp_acc | 0.685 | 0.000 | 2 | 0.500 | 515.481 |
| s | const | 600000000 | induction_acc | 0.199 | 0.060 | 2 | 0.000 | 3.294 |
| s | const | 600000000 | induction_gain | 6.153 | 0.726 | 2 | 0.000 | 8.473 |
| s | const | 600000000 | long_range_gain | 0.448 | 0.003 | 2 | 0.000 | 134.617 |
| s | const | 600000000 | context_gain | 0.171 | 0.000 | 2 | 0.000 | 2514.920 |

## Power for the mechanism-mapped endpoints (Welch, n_clean=6 vs n_arm, alpha 0.05)

Seed SD from the clean baselines at the best-populated group; the 2x/3x rows are what the SD might be if the baseline SD is underestimated. Effect = smallest effect of interest, in the metric's own units.

### val_loss_ctx_32

seed SD 0.003682 (n=2 seeds); effect 0.016

| SD multiplier | n_arm=3 | n_arm=4 | n_arm=5 | n_arm=6 |
|---|---|---|---|---|
| 1x (SD 0.003682) | 0.97 | 1.00 | 1.00 | 1.00 |
| 2x (SD 0.007364) | 0.63 | 0.79 | 0.88 | 0.91 |
| 3x (SD 0.01105) | 0.35 | 0.46 | 0.57 | 0.58 |

### context_gain

seed SD 0.001232 (n=2 seeds); effect 0.0175

| SD multiplier | n_arm=3 | n_arm=4 | n_arm=5 | n_arm=6 |
|---|---|---|---|---|
| 1x (SD 0.001232) | 1.00 | 1.00 | 1.00 | 1.00 |
| 2x (SD 0.002464) | 1.00 | 1.00 | 1.00 | 1.00 |
| 3x (SD 0.003695) | 0.98 | 1.00 | 1.00 | 1.00 |

### blimp_acc

seed SD 0.008232 (n=2 seeds); effect 0.01

| SD multiplier | n_arm=3 | n_arm=4 | n_arm=5 | n_arm=6 |
|---|---|---|---|---|
| 1x (SD 0.008232) | 0.28 | 0.34 | 0.41 | 0.47 |
| 2x (SD 0.01646) | 0.11 | 0.12 | 0.14 | 0.14 |
| 3x (SD 0.0247) | 0.07 | 0.07 | 0.10 | 0.08 |

## Power of the registered confirmatory procedure (one-sided Welch t, 6 clean vs 4 per arm, Holm across the three H1 tests, alpha 0.05)

Pairing by seed is reported as a sensitivity analysis because the within-seed correlation is unknown; the paired test only beats Welch when that correlation exceeds about 0.5.

All three true effects (the registered smallest effects of interest) present at once; seed SD scaled by the multiplier; rho = correlation between a seed's clean and corrupted run. Columns are the three H1 tests (corruption vs clean, with its mapped endpoint: shuffle_sentence on val_loss_ctx_32, shuffle_doc_sentences and short_context on context_gain); entries are the probability each test is rejected after Holm; 'all three' = all rejected.

### SD multiplier 1x

| rho | shuffle_sentence | shuffle_doc_sentences | short_context | all three |
|---|---|---|---|---|
| -0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |

### SD multiplier 2x

| rho | shuffle_sentence | shuffle_doc_sentences | short_context | all three |
|---|---|---|---|---|
| -0.50 | 0.87 | 1.00 | 1.00 | 0.87 |
| 0.00 | 0.90 | 1.00 | 1.00 | 0.90 |
| 0.50 | 0.93 | 1.00 | 1.00 | 0.93 |

### SD multiplier 3x

| rho | shuffle_sentence | shuffle_doc_sentences | short_context | all three |
|---|---|---|---|---|
| -0.50 | 0.61 | 1.00 | 1.00 | 0.61 |
| 0.00 | 0.63 | 1.00 | 1.00 | 0.63 |
| 0.50 | 0.67 | 1.00 | 1.00 | 0.67 |

### Sensitivity: paired t over the 4 seed pairs (df = 3)

#### SD multiplier 1x

| rho | shuffle_sentence | shuffle_doc_sentences | short_context | all three |
|---|---|---|---|---|
| -0.50 | 0.98 | 1.00 | 1.00 | 0.98 |
| 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |

#### SD multiplier 2x

| rho | shuffle_sentence | shuffle_doc_sentences | short_context | all three |
|---|---|---|---|---|
| -0.50 | 0.61 | 1.00 | 1.00 | 0.61 |
| 0.00 | 0.75 | 1.00 | 1.00 | 0.75 |
| 0.50 | 0.94 | 1.00 | 1.00 | 0.94 |

#### SD multiplier 3x

| rho | shuffle_sentence | shuffle_doc_sentences | short_context | all three |
|---|---|---|---|---|
| -0.50 | 0.34 | 0.91 | 0.91 | 0.34 |
| 0.00 | 0.47 | 0.99 | 0.99 | 0.47 |
| 0.50 | 0.72 | 1.00 | 1.00 | 0.72 |

## Historical (superseded 2026-10-19; kept for the audit trail)

## Proposed primary metric (proposal only)

**long_range_gain**: long_range_gain: highest signal-to-noise (|mean-chance|/seed SD = 296.1; mean 0.471, SD 0.0016, chance 0) among blimp_acc, induction_acc, induction_gain, long_range_gain, context_gain in the largest baseline group m/const/600,000,000 tokens (n=2 seeds); WARNING fewer than 3 seeds, SD estimate is unreliable

## Power (simulated Welch t-test, equal n per arm, alpha 0.05)

Seed SD of primary metric: 0.00159

| effect (SD) | n=2 | n=3 | n=4 | n=5 | n=6 | n=8 |
|---|---|---|---|---|---|---|
| 0.50 | 0.03 | 0.06 | 0.07 | 0.10 | 0.10 | 0.14 |
| 1.00 | 0.04 | 0.11 | 0.20 | 0.25 | 0.31 | 0.45 |
| 2.00 | 0.11 | 0.36 | 0.60 | 0.75 | 0.86 | 0.96 |
| 3.00 | 0.18 | 0.68 | 0.91 | 0.98 | 0.99 | 1.00 |

## Plasticity-predicted window

Fisher trace uninformative (peak at init); ability landmarks (mean point reaching half of final value): blimp_acc: frac 0.096, tokens 2.775e+07; induction_acc: frac 0.423, tokens 1.471e+08; long_range_gain: frac 0.106, tokens 3.33e+07

| run | frac_peak | frac_half |
|---|---|---|
| baselines__clean__wnone__m__const__s0__3742cb | 0.000 | 0.042 |
| baselines__clean__wnone__m__const__s0__481d3b | 0.000 | 0.036 |
| baselines__clean__wnone__m__const__s0__a14b09 | 0.000 | 0.166 |
| baselines__clean__wnone__m__const__s1__4fa788 | 0.000 | 0.166 |
| baselines__clean__wnone__m__const__s1__b08f0f | 0.000 | 0.042 |
| baselines__clean__wnone__s__const__s0__154386 | 0.000 | 0.042 |
| baselines__clean__wnone__s__const__s0__633f90 | 0.000 | 0.166 |
| baselines__clean__wnone__s__const__s1__64e8fe | 0.000 | 0.166 |
| baselines__clean__wnone__s__const__s1__670d90 | 0.000 | 0.042 |

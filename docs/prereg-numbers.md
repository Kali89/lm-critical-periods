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

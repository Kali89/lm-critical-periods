# Baseline ladder readout, 2026-10-17

Nine clean runs (`configs/grid/baselines.yaml`), constant LR (m 3e-4, s 1e-3), BabyLM-100M lowercased,
8192 vocab, seq 512, batch 32k tokens. All done, no errors; wall time 22.9 h on one M3 Max.
Seed reproducibility: final val loss differs by <= 0.003 between seeds in every cell.

## Final metrics (mean over seeds; n=2 except 1.4B n=1)

| size | tokens | epochs | val loss | BLiMP | induction acc | induction gain (nats) | long-range gain | h/run |
|---|---|---|---|---|---|---|---|---|
| s | 150M | 1 | 3.762 | 0.658 | 0.09 | 4.7 | 0.431 | 0.4 |
| s | 600M | 4 | 3.644 | 0.685 | 0.20 | 6.2 | 0.448 | 1.6 |
| m | 150M | 1 | 3.620 | 0.684 | 0.22 | 6.6 | 0.449 | 1.0 |
| m | 600M | 4 | 3.458 | 0.707 | 0.24 | 6.6 | 0.471 | 3.9 |
| m | 1.4B | 10 | 3.394 | 0.713 | 0.18 | 5.7 | 0.488 | 9.0 |

Seed SD at m/600M (n=2, unreliable): val 0.0005, BLiMP 0.008, induction acc 0.055, induction gain 0.27,
long-range gain 0.0016.

## Emergence (m, 600M; tokens at which each ability reaches half its final value)

| ability | half-max | plateau | notes |
|---|---|---|---|
| long-range gain | 25M | ~100M | 0.23 at 25M, 0.42 at 100M, 0.47 at 600M |
| BLiMP overall | 25-50M | ~300M | 0.63 at 25M, 0.70 at 200M; 1.4B adds 0.006 |
| induction | ~125M | 300-400M | 0 at 25M, 0.11 at 100M, 0.24 at 200M; noisy afterwards (seed 1: 0.25 -> 0.20) |

BLiMP phenomena still rising between 150M and 600M (candidates for a mid-window, ability-specific endpoint):
NPI licensing (+0.07), quantifiers (+0.06), anaphor agreement (+0.06), control raising (+0.04), ellipsis (+0.04).
Saturated by 150M: determiner-noun agreement, binding, irregular forms. Island effects stay at chance.

## Plasticity

The sampled Fisher trace does not show Achille's rise-then-fall. It is largest at initialisation (39-44), falls
to ~12 by 25M tokens and stays flat to 1.4B (s: 11.6 -> ~4.2, flat). Effective rank of the residual stream rises
251 -> 326. The Fisher-based window predictor therefore returns "peak at 0" and is uninformative at constant LR;
windows must be placed from the ability-emergence curves above. This is itself a result to report.

## External anchor

Pythia-14M (Pile, 50k vocab) reaches induction only at 2-4B tokens and BLiMP 0.68 at 300B. Our 14M model on
BabyLM shows induction by 100-200M and BLiMP 0.70 by 200M: the child-directed register and 8k vocabulary make
these abilities cheap. Comparisons to Pythia are qualitative only.

## Power (Welch t, alpha 0.05, seed SD from m/600M)

| endpoint | deficit | power n=3 | n=4 | n=5 |
|---|---|---|---|---|
| long-range gain | 0.005 | 0.71 | 0.93 | 0.99 |
| long-range gain | 0.010 | 1.00 | 1.00 | 1.00 |
| BLiMP overall | 0.02 | 0.51 | 0.77 | 0.91 |
| BLiMP overall | 0.03 | 0.83 | 0.99 | 1.00 |
| induction gain | 0.5 nats | 0.34 | 0.55 | 0.72 |
| induction gain | 1.0 nats | 0.85 | 0.99 | 1.00 |
| induction acc | 0.10 | 0.33 | 0.53 | 0.69 |

## Proposed decisions (for the pre-registration; not yet adopted)

1. Operating point: size m, 600M tokens (3.9 h/run). 1.4B costs 2.3x for +0.006 BLiMP and a noisier induction
   score. Size s at 600M (1.6 h/run) is the cheap tier for the sliding-window grid.
2. Primary endpoint: long-range gain (highest SNR by the pre-declared rule; the corruption targets exactly the
   cross-sentence context it measures). Secondary: BLiMP overall and a pre-specified phenomenon set (Holm);
   induction gain in nats (not induction accuracy, which is too noisy).
3. Windows in fraction of 600M: early [0, 0.08] (0-50M: BLiMP and long-range forming, induction absent);
   mid [0.08, 0.33] (50-200M: induction forming); late-matched [0.5, 0.58] and [0.5, 0.75] as recency controls.
   Ability-specific prediction: early window -> long-range/BLiMP deficit; mid window -> induction deficit.
4. Seeds: 4 per corruption cell, 6 for the clean reference (shared across comparisons). Caveat: SD from n=2.
5. Analysis tooling: `bin/lmcp analyze` and `prereg` pool all grids (smoke, lr_sweep) and need a `--grid`
   filter; the Fisher window predictor should fall back to ability landmarks. Both are analysis-only changes
   under `lmcp/`, held until the rule on mid-grid edits is relaxed for analysis code or the vision grid ends.

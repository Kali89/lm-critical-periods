# Instrument check: how to measure "local" loss and context gain (2026-10-19)

Question: the design review asked for a same-target measure of context use (loss on the same targets with and
without access to context beyond 32 tokens). Three definitions were evaluated on the saved m-size clean baseline
checkpoints (eval only, same 1M validation tokens, targets at positions >= 128 of each 512-token sequence).

- **A, sliding mask.** Attention restricted to the 32 most recent tokens including self; position 0 hidden.
- **B, sliding mask + sink.** As A, but position 0 always visible.
- **C, fresh context.** The 32 tokens preceding each target re-run as a fresh sequence from position 0; the
  target's loss is the last-token loss of that 33-token snippet. Targets every 32nd (12/sequence), 8th
  (48/sequence) or 4th (96/sequence) position from 128; full-context loss taken at the same targets.

## Results (nats; two seeds at 150M and 600M, one at 1.4B)

| tokens | seed | full (all >=128) | A | B | C (stride 8) | full (stride-8 targets) | G = C − full |
|---|---|---|---|---|---|---|---|
| 150M | 0 | 3.580 | 3.781 | 3.773 | 3.756 | 3.584 | 0.172 |
| 150M | 1 | 3.576 | 3.775 | 3.766 | 3.752 | 3.581 | 0.171 |
| 600M | 0 | 3.414 | 3.655 | 3.612 | 3.591 | 3.420 | 0.171 |
| 600M | 1 | 3.415 | 3.666 | 3.612 | 3.596 | 3.423 | 0.173 |
| 1.4B | 0 | 3.350 | 3.719 | 3.614 | 3.531 | 3.355 | 0.176 |

Sanity check for C: at 600M the fresh-context loss (3.609 at stride 32) matches the model's own loss at
position 31 of the full sequences (3.607, 3.595), i.e. the model is evaluated in a configuration it trained in.

## Reading

1. **A measures sink reliance, not local prediction.** A − B is 0.009 at 150M, 0.04 at 600M and 0.105 at 1.4B:
   the better-trained the model, the more a window that hides position 0 hurts it. Seed-to-seed difference at
   600M is 0.0105 under A against 0.0005 under B. A gets *worse* from 600M to 1.4B while every other measure
   improves.
2. **Neither mask is a 32-token measure.** With a masked window the representation of each token already
   contains its own window, so after L layers a target sees up to L × 32 tokens; B is flat from 600M to 1.4B
   while C keeps improving, consistent with B leaking longer context. Only C has exactly 32 tokens of context.
3. **Under C, context gain is flat from 150M on:** 0.17-0.18 nats at 150M, 600M and 1.4B, in both seeds. The
   benefit of context beyond 32 tokens is fully formed by 150M tokens; all later improvement in the full loss
   is local. The local loss L_32 is still improving at 1.4B (3.75 → 3.59 → 3.53).
4. **Precision.** G is a within-target difference, so item noise largely cancels: seed difference 0.002 at
   stride 8. L_32 and the full loss on 48 targets/sequence differ between seeds by about 0.004-0.005, of which
   a good part is target sampling (the full loss on all 384 targets differs by 0.001). Stride 8 costs ~15 s per
   evaluation at size m (3 full-pass equivalents); stride 4 doubles that for no visible gain in precision.

## Decision

`val_loss_ctx_32`, `val_loss_ctx_full` and `context_gain` are defined by C at stride 8 (48 fixed targets per
sequence: positions 128, 136, ..., 504). The masked-window implementation was removed. Baseline checkpoints are
re-scored under the new definition before the pre-registration numbers are filled.

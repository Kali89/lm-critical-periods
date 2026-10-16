# lm-critical-periods

Do tiny language models have critical periods? This harness trains small GPTs from scratch on the BabyLM
corpus in MLX (Apple silicon), corrupts the training input during a chosen window of training, and tracks
several abilities separately as training continues on clean data. If a corruption applied early leaves a
deficit that the clean tail never repairs, while the same corruption applied late does not, that is a
sensitive period. The design follows Achille, Rovere & Soatto (ICLR 2019) for vision networks and the null
result of Constantinescu et al. (TACL 2025) for language identity; the gap being tested is *input structure*.

Everything here runs on one laptop. Operating rules, model tiers and hardware rules: `program.md`.

## What is in the box

| Piece | Where | Notes |
|---|---|---|
| Models | trained from scratch, `lmcp/model.py` | pre-norm GPT, RoPE, SwiGLU, tied embeddings. Sizes xs 1.5M, s 5M, m 14M, l 30M params |
| Data | `bin/lmcp prepare-all` | BabyLM-100M (CLIMB mirror, lowercased), ~138M tokens; dev split ~13M tokens; byte-level BPE vocab 8192 |
| Corruptions | `lmcp/corrupt.py`, `lmcp/train.py` | text-level: `shuffle_sentence`, `shuffle_doc_sentences`, `shuffle_window k`, `char_noise p`; training-time: `short_context block`, `label_noise p` |
| Window | `corruption.window: [a, b]` | fractions of the token budget during which the corruption is active |
| Evals | `lmcp/evals.py` | val loss by position (long-range gain), BLiMP by linguistic term (syntax), induction-head copying |
| Queue | `bin/lmcp grid`, `scripts/run-queue.sh` | pre-registered grids, one run at a time, caffeinate, AC-power and thermal checks, resumable |
| Results | `bin/lmcp collect`, `bin/lmcp report` | `results/results.jsonl` (one row per run), `results/curves.jsonl` (every eval), summary table vs clean |

No pretrained models are downloaded: every run starts from random weights, which is the point. Pythia or Pico
checkpoints can be added later as external comparators.

## Setup on a new machine

    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
    bin/lmcp prepare-all                               # ~3 min: download, normalise, tokenizer, clean streams
    bin/lmcp tok encode --kind shuffle_sentence        # one corrupted stream per text-level corruption (~1 min each)
    bin/lmcp doctor

## Running an experiment

    bin/lmcp grid configs/grid/smoke.yaml      # 2 tiny runs, a few minutes: proves the pipeline
    scripts/run-queue.sh
    bin/lmcp report

Then baselines, then `configs/grid/pilot.yaml`. Commit a grid file before running it; that commit is the
pre-registration. Add `train.seed` values rather than re-running the same seed.

## Design notes

- **Alignment of streams.** Documents are shuffled once (seed 1234) when the corpus is prepared; every
  corrupted stream is built from the same document order with a per-document deterministic RNG. During a
  window the trainer reads the corrupted stream at the same *fraction* of the corpus, so the model sees the
  same documents, corrupted, that it would otherwise have seen clean.
- **Batches** come from B readers spaced evenly through the corpus, each advancing sequentially. This
  decorrelates a batch without random access and keeps "position in training" meaningful.
- **Learning rate.** Default `wsd` for general use; use `const` for critical-period arms so a deficit cannot be
  explained by late data arriving at a small learning rate. Cosine is available but confounded.
- **Controls to always include.** Clean baseline with the same budget; matched-clean-tokens control (a clean
  run of only the clean tokens the corrupted arm saw); a late-onset window of the same length (recency
  control); 3+ seeds. An extended-recovery arm (2x tokens, clean tail) separates permanent from slow.
- **BLiMP is lowercased** at eval time because the CLIMB-cleaned BabyLM text is lowercased. Set
  `eval.blimp_lowercase: false` if you switch to `data.variant: original`.
- **Short context** is implemented as a block-diagonal attention mask over the normal 512-token sequence, so
  shapes never change and compiled kernels are reused.

## Reading a run

    runs/<name>/train_curve.jsonl    loss every 20 steps with phase (corrupt|clean), lr, tok/s
    runs/<name>/metrics.jsonl        one line per eval: val_loss, val_loss_pos_*, long_range_gain,
                                     induction_acc/gain, blimp_acc, blimp/<term>, blimp_paradigms
    runs/<name>/status.json          running | done | failed | interrupted, with timings

Resume is automatic: a failed or interrupted run restarts from its last checkpoint when the queue picks it up
again (`bin/lmcp queue requeue-failed`).

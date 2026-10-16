# lm-critical-periods

Do small language models have sensitive periods for the *structure* of their input? This harness trains small
GPTs from scratch on the BabyLM corpus in MLX (Apple silicon), corrupts the training input during a chosen
window of training, and tracks several abilities and the network's plasticity as training continues on clean
data. It is built so that a result would survive review by the authors of the prior work: positive and
negative controls run through the same code, document order is randomised per seed, plasticity is measured
rather than assumed, every metric carries an item-level confidence interval, the analysis is a mixed model
with a pre-registered primary endpoint, and checkpoints export to PyTorch for mechanistic follow-up.

Prior work this sits between: Achille, Rovere & Soatto (ICLR 2019) found critical periods for input blur in
vision networks and none for label permutation. Constantinescu, Pimentel, Cotterell & Warstadt (TACL 2025)
found no critical period for *which language* arrives when in LM pretraining. Both are reproduced here as
controls (`configs/grid/vision_control.yaml`, `configs/grid/constantinescu.yaml`).

Everything runs on one laptop. Operating rules, phases and gates, model tiers, hardware rules: `program.md`.
Pre-registration template: `docs/preregistration.md`.

## What is in the box

| Piece | Where | Notes |
|---|---|---|
| Models | `lmcp/model.py` | pre-norm GPT, RoPE, SwiGLU, tied embeddings (= Llama layout). xs 1.5M, s 5M, m 14M, l 30M params. Trained from scratch; no pretrained weights |
| Data | `bin/lmcp prepare-all`, `lmcp/data.py` | BabyLM-100M (CLIMB mirror, lowercased), 138M tokens; dev 13M; BLiMP 67 paradigms; official BabyLM Dutch (`babylm_nld`) as L2 (`docs/data-babylm-nl.md`); CIFAR-10 for the vision control |
| Streams | `lmcp/tok.py` | byte-level BPE (vocab 8192) uint16 streams with a document index; one stream per corruption and dose |
| Document order | `lmcp/train.py` `DocReader` | per-seed permutation (`data.doc_order_seed`), saved per run; readers walk shards of it; clean and corrupted streams switch on the same document |
| Corruptions | `lmcp/corrupt.py`, `lmcp/train.py` | text: `shuffle_sentence`, `shuffle_doc_sentences`, `shuffle_window k`, `char_noise p`; training-time: `short_context block`, `label_noise p`, `vocab_permute` (predicted null). `corruption.window` = fraction range of the token budget; `corruption.intensity` = dose |
| Stream phases | `train.phases` | mixture schedule over streams, e.g. English then Dutch, for the Constantinescu replication |
| Evals | `lmcp/evals.py`, `lmcp/stats.py` | val loss by position (long-range gain), BLiMP by linguistic term, induction copying; every metric with a 95% item-level bootstrap CI; several val streams |
| Plasticity | `lmcp/plasticity.py` | Fisher trace (sampled and empirical), gradient and parameter norms, effective rank of residual stream and embeddings, dead-unit fraction, at every eval on a fixed probe batch |
| Positive control | `lmcp/vision/`, `docs/vision-control.md` | Achille et al. CIFAR-10 ResNet-18 blur / label-permute / vflip through the same window and queue code |
| Queue | `bin/lmcp grid`, `scripts/run-queue.sh` | pre-registered grids, one run at a time, caffeinate, AC-power and thermal checks, resumable |
| Analysis | `lmcp/analysis.py`, `bin/lmcp analyze`, `bin/lmcp prereg` | landmarks, baseline SNR and primary-endpoint proposal, mixed-effects model (seed random effect), Holm, dose-response, simulated power, Fisher-predicted window |
| Mech-interp export | `tools/export_tl.py`, `induction_heads.py`, `llc.py`, `tools/README.md` | MLX checkpoint to HF-Llama / TransformerLens (logits match to 5e-6), per-head induction scores, local learning coefficient via devinterp (`.venv-torch`) |
| External anchors | `tools/pythia_probe.py`, `tools/export_nanogpt.py`, `tools/README-external.md` | same probes on Pythia-14M/70M checkpoints (`results/external/`); nanoGPT config export for a CUDA rerun |

## Setup on a new machine

    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
    bin/lmcp prepare-all                                   # BabyLM, BLiMP, tokenizer, clean + shuffle_sentence streams
    bin/lmcp tok encode --kind shuffle_sentence --intensity 0.5
    bin/lmcp tok encode --kind shuffle_sentence --intensity 0.25
    .venv/bin/python -m lmcp.vision.data                   # CIFAR-10 for the positive control
    hf auth login   # needs access to BabyLM-community/babylm-nld (gated)
    bin/lmcp data download --corpus babylm_nld && bin/lmcp data prepare --corpus babylm_nld   # official BabyLM Dutch L2
    bin/lmcp tok train --corpora babylm100m_clean,babylm_nld_clean --name en_nld_v8192
    bin/lmcp tok encode --name en_nld_v8192 --split both
    bin/lmcp tok encode --name en_nld_v8192 --corpus babylm_nld --split both --as babylm_nld
    uv venv --python 3.12 .venv-torch && uv pip install --python .venv-torch/bin/python torch transformers transformer_lens devinterp "zarr<3.2" safetensors datasets
    bin/lmcp doctor

## Protocol (see program.md for gates)

1. `configs/grid/smoke.yaml`, `vision_smoke.yaml`, `constantinescu_smoke.yaml` prove the pipeline (minutes).
2. `vision_control.yaml` must reproduce Achille et al. before any LM claim is made (about a day of laptop time).
3. `baselines.yaml` (3 clean seeds, size m) then `bin/lmcp analyze baseline|landmarks` and `bin/lmcp prereg --write`.
4. Fill and commit `docs/preregistration.md`; set the pilot seed count from the power table.
5. `pilot.yaml`, then `bin/lmcp analyze mixed` and `dose`. `sliding.yaml` only if the pilot deficit survives the controls.
6. `constantinescu.yaml` for the prior-null replication; `tools/` for mechanism and external anchors.

## Design notes

- **Stream alignment.** Documents are shuffled once at corpus preparation; every corrupted stream is built
  from the same document list with a per-document deterministic RNG and carries a document index. The reader
  permutes document order per seed and switches streams on the same document id, so a corrupted window shows
  the model the same documents it would otherwise have seen clean.
- **Learning rate.** Constant (2% warmup) in critical-period arms so a deficit cannot be explained by late data
  arriving at a small learning rate. `wsd` and `cosine` exist for other uses.
- **Controls, always.** Clean baseline; matched-clean-tokens control; late window of the same length;
  vocabulary permutation (predicted null); extended recovery (2x clean tail); 3+ seeds, count from `prereg`.
- **BLiMP is lowercased** at eval time because the CLIMB-cleaned BabyLM text is lowercased (`eval.blimp_lowercase`).
- **Short context** is a block-diagonal attention mask over the normal 512-token sequence, so shapes never change.
- **Vision control caveats.** Achille et al. trained 160 epochs after deficit removal; the control grid uses 60
  epochs with a rescaled exponential decay to fit laptop time (about 78 min per run). Weight decay for SGD comes
  from `vision.weight_decay` (5e-4), not the AdamW value in `train.weight_decay`.
- **Dutch L2 caveats.** The BabyLM Dutch set is gated on Hugging Face; the fallback is a lowercased Dutch
  Wikipedia subsample, which is a register contrast as well as a language contrast. Swap it in once access is granted.

## Reading a run

    runs/<name>/config.json          resolved config, git rev, doc-order seed and hash, stream names
    runs/<name>/doc_order.npy        the document permutation
    runs/<name>/train_curve.jsonl    loss every 20 steps with phase (corrupt|clean|mix), lr, tok/s, epochs
    runs/<name>/metrics.jsonl        per eval: val_loss[/stream] (+CI), val_loss_pos_*, long_range_gain, blimp_acc,
                                     blimp/<term> (+CI), induction_*, plast/* (Fisher trace, norms, ranks)
    runs/<name>/status.json          running | done | failed | interrupted
    results/results.jsonl            one row per run (bin/lmcp collect); results/curves.jsonl every eval
    results/external/                Pythia landmark probes

Resume is automatic: a failed or interrupted run restarts from its last checkpoint when the queue picks it up
again (`bin/lmcp queue requeue-failed`). Runs whose names start with `_` are scratch and are never collected.

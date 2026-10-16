# External anchors and portability

## Pythia landmark probe (`tools/pythia_probe.py`)

Runs the same probes as `lmcp/evals.py` on EleutherAI Pythia checkpoints (a natural training run, 2,097,152 tokens/step)
so our from-scratch baselines can be compared on the same axes.

    .venv-torch/bin/python tools/pythia_probe.py --model pythia-14m --steps 0,1000,143000
    .venv-torch/bin/python tools/pythia_probe.py --model pythia-70m --steps log       # 0,1,2,4..512,1000,2000..64000,143000
    flags: --blimp-per-paradigm N (default 200, 0 = all 1000)  --val-tokens N (default 500000)  --out PATH

Appends one JSON line per checkpoint to `results/external/<model>.jsonl` (rerunning appends duplicates; dedupe on `step`).
Keys: model, step, tokens, tokens_per_param, n_params, blimp_acc, blimp/<term>, induction_loss_first/second, induction_gain,
induction_acc, val_loss, val_loss_pos_0_32 / _32_128 / _128_512, long_range_gain, eval_seconds.
Runs on MPS if available, else CPU; float32. The first run of each revision includes the HF download in `eval_seconds`.

Mirrors lmcp: BLiMP = summed log p of sentence tokens with <|endoftext|> (id 0) prepended, right-padded and masked;
induction = random ids in [2, 50277), L=64 repeated twice, 8x32 sequences, seed 99; val loss by position over 513-token packed
sequences (EOS after each doc), first N tokens.

### Caveats

- **Absolute values are not comparable with our runs.** BabyLM-clean is lowercased and uses our 8192 BPE; Pythia is cased
  (BLiMP sentences are used uncased-as-is, not lowercased) with a 50k vocab, so val loss is per Pythia token on lowercased text
  (loss per token is larger with a bigger vocab and shifts with tokenisation). Induction chance-level losses also differ with vocab.
- The purpose is **landmark timing**: when the induction jump and BLiMP rise happen, in tokens and in tokens-per-parameter
  (`tokens / n_params`; n_params counts embeddings, 14.07M for pythia-14m).
- Pythia saw the Pile (not BabyLM); its budget (300B tokens) is ~3000x our runs, so only the early checkpoints
  (steps 1 to ~2000, up to ~4B tokens) overlap our token range, and tokens-per-parameter is the fairer axis.

## nanoGPT export (`tools/export_nanogpt.py`)

    python3 tools/export_nanogpt.py runs/<run_dir>            # or a config.yaml; --out exports/nanogpt/<name>

Writes `config.py`, `README.md` (setup + fidelity caveats) and `window_patch.md` (corruption-window patch sketch) under
`exports/nanogpt/<name>/`. Design-level replication only: nanoGPT uses learned positions + GELU, random offsets, no
corruption window. See the generated README and `window_patch.md`; modded-nanogpt is the faster GPU alternative.

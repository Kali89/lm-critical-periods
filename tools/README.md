# tools/ — PyTorch interpretability tooling for MLX checkpoints

All three tools run in `.venv-torch` (torch, transformers, transformer_lens 4.2, devinterp 2.0.1);
the only MLX code is the `--check` reference dump, which `export_tl.py` runs itself via `.venv/bin/python`.
Nothing here trains, and nothing writes outside `exports/`.

## 1. Export

    .venv-torch/bin/python tools/export_tl.py runs/<name>            # uses runs/<name>/final/model.safetensors
    .venv-torch/bin/python tools/export_tl.py runs/<name>/ckpt/<x>/model.safetensors --out exports/tl/<name>-step<x>
    ... --check                                                      # also verify against MLX logits

`<name>/config.json` is found next to (or up to two levels above) the safetensors file and supplies
`config.model` (n_layer, d_model, n_head) and `config.data.tokenizer_vocab`. Output, in `exports/tl/<name>/`:

| File | What |
|---|---|
| `hf/` | HuggingFace `LlamaForCausalLM` checkpoint (+ tokenizer.json if `data/tok/*v<vocab>/` exists). **This is what TransformerLens 4.x loads** (`TransformerBridge`). |
| `tl_config.json`, `state_dict.pt` | The same weights in legacy `HookedTransformer` layout/names (TL < 4), as specified in the task. |
| `mlx_logits.npy`, `check.json` | `--check` only: MLX reference logits and the comparison. |

Load for analysis: `export_tl.load_export(dir)` returns a `TransformerBridge` (raw weights, no LN folding or centering;
`run_with_cache` hooks such as `blocks.0.attn.hook_pattern`, `hook_z`, `hook_resid_post` all exist).

`--check` feeds 4 fixed random sequences of 32 tokens (numpy seed 0) through the MLX model and compares:
(a) the TransformerLens bridge on the HF export, (b) a pure-torch forward over the legacy-layout tensors in `state_dict.pt`
(this validates the head/transposition mapping of the legacy files independently, since TL 4 cannot load them).
It exits non-zero if either max abs diff >= 1e-3.

## 2. Induction heads

    .venv-torch/bin/python tools/induction_heads.py exports/tl/<name> [--len 64 --n 32 --seed 0]

Writes `<export>/induction_heads.json`. Per layer/head: `induction_score` (= `prefix_matching_score`: attention from t in
the second copy to t-L+1, t in [L, 2L-2]), `prev_token_score`, `copying_score`; plus maxima and the uniform-attention baseline.
Copying score uses the OV circuit from token embeddings (ln1-scaled, RMS-normalised), so it is exact only for layer 0.

## 3. Local learning coefficient

    .venv-torch/bin/python tools/llc.py exports/tl/<name> --data data/tok/babylm100m_clean_v8192/val.clean.bin \
        [--n-chains 2 --n-steps 200 --batch 8 --seq 128 --lr 1e-4 --gamma 100 --nbeta X --device cpu]

Writes `<export>/llc.json` (LLC mean/std over chains, init loss, first/last-10-step sampling loss, runtime).
LLC = nbeta * (mean SGLD loss - loss at the checkpoint), loss in nats/token. **Check `sampling_loss_last10` vs `init_loss`**:
if the chain is still climbing the estimate is not converged. LLC is very sensitive to `--gamma`/`--lr`; calibrate per model
size (and compare checkpoints only at identical settings).

## Architecture mapping (MLX `lmcp/model.py` -> exports)

| MLX param (shape) | HF Llama | legacy TL `HookedTransformer` |
|---|---|---|
| `wte.weight` (V,d) | `model.embed_tokens.weight`, tied `lm_head` | `embed.W_E` (V,d); `unembed.W_U` = wte^T (d,V); `b_U`=0 |
| `blocks.i.n1.weight` | `input_layernorm.weight` | `blocks.i.ln1.w` (RMS, eps 1e-5) |
| `blocks.i.attn.wqkv.weight` (3d,d) rows [0:d]/[d:2d]/[2d:3d] | `q_proj` / `k_proj` / `v_proj` (d,d) | `W_Q/W_K/W_V` (h,d,dh) = rows.T.reshape(d,h,dh).permute(1,0,2); `b_*`=0 |
| `blocks.i.attn.wo.weight` (d,d) | `o_proj` | `W_O` (h,dh,d) = wo.T.reshape(h,dh,d); `b_O`=0 |
| RoPE `nn.RoPE(dh, traditional=False)`, base 10000 | Llama RoPE (rotate-half), `rope_theta`=1e4 | `rotary`, rotary_dim=dh, `rotary_adjacent_pairs=False`, base 10000 |
| `blocks.i.n2.weight` | `post_attention_layernorm.weight` | `ln2.w` |
| `mlp.w1.weight` (hid,d) (silu branch) | `gate_proj` | `W_gate` = w1.T |
| `mlp.w3.weight` (hid,d) (linear branch) | `up_proj` | `W_in` = w3.T |
| `mlp.w2.weight` (d,hid) | `down_proj` | `W_out` = w2.T |
| `nf.weight` | `model.norm.weight` | `ln_final.w` |

hid = 64*ceil(int(8d/3)/64). No biases anywhere; attention scale 1/sqrt(dh); causal mask; n_ctx from `data.seq_len` (512).
TL settings: `normalization_type="RMS"`, `act_fn="silu"`, `gated_mlp=True`, `default_prepend_bos=False`, `tokenizer_name=None`.

## Known limitations

- **TransformerLens 4.x removed `HookedTransformer`/`HookedTransformerConfig`**; the installed 4.2.0 only offers
  `TransformerBridge` over HF models. Hence the HF Llama export (Llama is exactly this architecture). The legacy
  `tl_config.json`/`state_dict.pt` are verified only by the pure-torch reference, not by loading into a real HookedTransformer.
  Bridge numerics are raw-weight (no `fold_ln`/`center_*`); call `enable_compatibility_mode()` yourself if you want those.
- **devinterp 2.0.1 vs zarr 3.4.1**: `devinterp.slt.llc` fails to import (`RegularChunkGrid`). `llc.py` then falls back
  to devinterp's `SGMCMC.sgld` optimiser in a plain loop (same estimator; matched the library path to 3 d.p. on the smoke
  model, with `zarr==3.1.5` supplied through PYTHONPATH). To use the library path in the venv: `uv pip install --python .venv-torch/bin/python "zarr<3.2"`.
- LLC defaults (lr 1e-4, gamma 100, nbeta = batch/ln batch) are the requested sane defaults, not calibrated: on the 1.5M-param
  model gamma=100 lets the chain drift (loss 6.04 -> 7.4). Treat smoke numbers as a plumbing test.
- Export is fp32 and loads on CPU; `--check` tolerance is 1e-3, observed ~5e-6.

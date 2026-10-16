#!/usr/bin/env python
"""Export an MLX checkpoint (lmcp/model.py GPT) to a TransformerLens HookedTransformer.

Usage (run with .venv-torch/bin/python):
    tools/export_tl.py <run_dir | model.safetensors> [--out exports/tl/<name>] [--check]

Writes <out>/tl_config.json and <out>/state_dict.pt (TransformerLens parameter names, fp32).
--check additionally runs the MLX model (via .venv/bin/python, this same script with --mlx-dump)
on 4 fixed random sequences of 32 tokens (seed 0), saves logits to <out>/mlx_logits.npy and
compares them with the TransformerLens model. Reports max/mean abs diff.

The MLX side and the torch side live in one file so the mapping and its test cannot drift apart.
Imports of mlx / torch are lazy: each venv only has one of them.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_SEQ, SEQ_LEN, SEED = 4, 32, 0


def resolve(src: Path):
    """-> (model.safetensors path, config.json path, run name)."""
    src = src.resolve()
    if src.is_dir():
        st = src / "final" / "model.safetensors"
        if not st.exists():
            cands = sorted(src.glob("**/model.safetensors"))
            if not cands:
                sys.exit(f"no model.safetensors under {src}")
            st = cands[-1]
    else:
        st = src
    for d in [st.parent, st.parent.parent, st.parent.parent.parent]:
        if (d / "config.json").exists():
            return st, d / "config.json", d.name
    sys.exit(f"no config.json found above {st}")


def fixed_tokens(vocab):
    import numpy as np
    return np.random.default_rng(SEED).integers(0, vocab, size=(N_SEQ, SEQ_LEN)).astype("int64")


def load_cfg(config_path: Path):
    c = json.loads(config_path.read_text())["config"]
    m = c["model"]
    return dict(n_layer=m["n_layer"], d_model=m["d_model"], n_head=m["n_head"],
                vocab=c["data"]["tokenizer_vocab"], n_ctx=c["data"].get("seq_len", 512))


# ---------------------------------------------------------------- MLX side
def mlx_dump(st: Path, config_path: Path, out_npy: Path):
    import mlx.core as mx
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from lmcp.model import GPT, causal_mask
    c = load_cfg(config_path)
    model = GPT(c["vocab"], c["n_layer"], c["d_model"], c["n_head"])
    model.load_weights(str(st), strict=True)
    toks = fixed_tokens(c["vocab"])
    logits = model(mx.array(toks), causal_mask(SEQ_LEN))
    mx.eval(logits)
    np.save(out_npy, np.array(logits.astype(mx.float32)))
    print(f"wrote {out_npy} {tuple(logits.shape)}")


# --------------------------------------------------------------- torch side
def build_tl_config(c):
    d, h = c["d_model"], c["n_head"]
    return dict(
        n_layers=c["n_layer"], d_model=d, n_heads=h, d_head=d // h,
        d_mlp=64 * ((int(8 * d / 3) + 63) // 64),   # lmcp.model.MLP hidden size
        d_vocab=c["vocab"], n_ctx=c["n_ctx"], act_fn="silu", gated_mlp=True,
        normalization_type="RMS", eps=1e-5,
        positional_embedding_type="rotary", rotary_dim=d // h, rotary_base=10000,
        rotary_adjacent_pairs=False, attn_only=False, attention_dir="causal",
        original_architecture="lmcp-mlx-gpt", default_prepend_bos=False, tokenizer_name=None,
        dtype="float32",
    )


def convert_state(flat, c):
    """flat: {mlx param name: torch fp32 tensor} -> TransformerLens state dict."""
    import torch
    d, h = c["d_model"], c["n_head"]
    dh = d // h
    sd = {}
    wte = flat["wte.weight"]
    sd["embed.W_E"] = wte.clone()
    sd["unembed.W_U"] = wte.T.contiguous().clone()
    sd["unembed.b_U"] = torch.zeros(c["vocab"])
    sd["ln_final.w"] = flat["nf.weight"].clone()
    for i in range(c["n_layer"]):
        p, t = f"blocks.{i}.", f"blocks.{i}."
        sd[t + "ln1.w"] = flat[p + "n1.weight"].clone()
        sd[t + "ln2.w"] = flat[p + "n2.weight"].clone()
        wqkv = flat[p + "attn.wqkv.weight"]               # (3d, d) = (out, in)
        for name, rows in (("Q", wqkv[:d]), ("K", wqkv[d:2 * d]), ("V", wqkv[2 * d:])):
            # y = x @ rows.T ; head k owns output columns [k*dh:(k+1)*dh]
            sd[t + f"attn.W_{name}"] = rows.T.reshape(d, h, dh).permute(1, 0, 2).contiguous()
            sd[t + f"attn.b_{name}"] = torch.zeros(h, dh)
        sd[t + "attn.W_O"] = flat[p + "attn.wo.weight"].T.reshape(h, dh, d).contiguous()
        sd[t + "attn.b_O"] = torch.zeros(d)
        sd[t + "mlp.W_gate"] = flat[p + "mlp.w1.weight"].T.contiguous()   # silu branch
        sd[t + "mlp.W_in"] = flat[p + "mlp.w3.weight"].T.contiguous()     # linear branch
        sd[t + "mlp.W_out"] = flat[p + "mlp.w2.weight"].T.contiguous()
        hid = flat[p + "mlp.w1.weight"].shape[0]
        sd[t + "mlp.b_in"] = torch.zeros(hid)
        sd[t + "mlp.b_out"] = torch.zeros(d)
    return sd


def hf_llama_config(c):
    from transformers import LlamaConfig
    d, h = c["d_model"], c["n_head"]
    return LlamaConfig(
        vocab_size=c["vocab"], hidden_size=d, intermediate_size=64 * ((int(8 * d / 3) + 63) // 64),
        num_hidden_layers=c["n_layer"], num_attention_heads=h, num_key_value_heads=h, head_dim=d // h,
        hidden_act="silu", max_position_embeddings=c["n_ctx"], rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=True, attention_bias=False, mlp_bias=False, torch_dtype="float32",
        bos_token_id=None, eos_token_id=None, pad_token_id=None)


def convert_hf(flat, c):
    """MLX names -> HF Llama names. Layouts are identical (Linear weight (out,in), q/k/v rows stacked)."""
    d = c["d_model"]
    sd = {"model.embed_tokens.weight": flat["wte.weight"], "model.norm.weight": flat["nf.weight"]}
    for i in range(c["n_layer"]):
        p, t = f"blocks.{i}.", f"model.layers.{i}."
        w = flat[p + "attn.wqkv.weight"]
        sd[t + "self_attn.q_proj.weight"] = w[:d]
        sd[t + "self_attn.k_proj.weight"] = w[d:2 * d]
        sd[t + "self_attn.v_proj.weight"] = w[2 * d:]
        sd[t + "self_attn.o_proj.weight"] = flat[p + "attn.wo.weight"]
        sd[t + "mlp.gate_proj.weight"] = flat[p + "mlp.w1.weight"]
        sd[t + "mlp.up_proj.weight"] = flat[p + "mlp.w3.weight"]
        sd[t + "mlp.down_proj.weight"] = flat[p + "mlp.w2.weight"]
        sd[t + "input_layernorm.weight"] = flat[p + "n1.weight"]
        sd[t + "post_attention_layernorm.weight"] = flat[p + "n2.weight"]
    return {k: v.clone().contiguous() for k, v in sd.items()}


def find_tokenizer(c):
    for p in sorted((ROOT / "data" / "tok").glob(f"*v{c['vocab']}/tokenizer.json")):
        return p
    return None


def export(st: Path, config_path: Path, out: Path):
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    c = load_cfg(config_path)
    flat = load_file(str(st))
    expected = 2 + c["n_layer"] * 7   # wte, nf, 7 tensors per layer
    if len(flat) != expected:
        sys.exit(f"unexpected checkpoint tensors: got {len(flat)}, expected {expected}: {sorted(flat)}")
    out.mkdir(parents=True, exist_ok=True)

    # (1) HF Llama checkpoint: what TransformerLens 4.x (TransformerBridge) consumes.
    model = LlamaForCausalLM(hf_llama_config(c)).float()
    res = model.load_state_dict(convert_hf(flat, c), strict=False)
    # tied lm_head.weight is the only key allowed to be absent (it is wte)
    missing = [k for k in res.missing_keys if k != "lm_head.weight"]
    if missing or res.unexpected_keys:
        sys.exit(f"HF state_dict mismatch. missing={missing} unexpected={res.unexpected_keys}")
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "lm_head not tied"
    model.save_pretrained(out / "hf", safe_serialization=True)
    tok = find_tokenizer(c)
    if tok is not None:
        PreTrainedTokenizerFast(tokenizer_file=str(tok)).save_pretrained(out / "hf")
    print(f"loaded {len(flat)} tensors strictly into HF Llama; tied lm_head verified")

    # (2) legacy HookedTransformer-format files (TransformerLens < 4); see tools/README.md.
    cfg_dict = build_tl_config(c)
    sd = convert_state(flat, c)
    (out / "tl_config.json").write_text(json.dumps(
        {**cfg_dict, "source_checkpoint": str(st), "source_config": str(config_path)}, indent=2))
    torch.save(sd, out / "state_dict.pt")
    print(f"exported -> {out}")
    return sd


def load_export(export_dir: Path):
    """Reload an export as a TransformerLens 4.x TransformerBridge (raw weights, no folding)."""
    import torch
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    from transformer_lens import TransformerBridge
    hf = Path(export_dir) / "hf"
    model = LlamaForCausalLM.from_pretrained(hf, torch_dtype=torch.float32, attn_implementation="eager")
    tok = PreTrainedTokenizerFast.from_pretrained(hf) if (hf / "tokenizer.json").exists() else None
    bridge = TransformerBridge.boot_transformers(str(hf), hf_model=model, tokenizer=tok,
                                                 device="cpu", dtype=torch.float32)
    return bridge


def ref_forward(sd, cfg, toks):
    """Pure-torch forward over the *TransformerLens-format* tensors (independent of any library):
    verifies the head layout / transposes of the legacy HookedTransformer files."""
    import torch
    d, h, dh = cfg["d_model"], cfg["n_heads"], cfg["d_head"]
    T = toks.shape[1]
    def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg["eps"]) * w
    inv = 1.0 / (cfg["rotary_base"] ** (torch.arange(0, dh, 2).float() / dh))
    ang = torch.arange(T)[:, None] * inv[None]
    cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()
    def rope(x):  # x (B,T,H,dh), half-split
        x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
        return x * cos[None, :, None] + torch.cat([-x2, x1], -1) * sin[None, :, None]
    x = sd["embed.W_E"][toks]
    for i in range(cfg["n_layers"]):
        t = f"blocks.{i}."
        n = rms(x, sd[t + "ln1.w"])
        q, k, v = (torch.einsum("btd,hde->bthe", n, sd[t + f"attn.W_{m}"]) for m in "QKV")
        q, k = rope(q), rope(k)
        s = torch.einsum("bthe,bshe->bhts", q, k) / dh ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        z = torch.einsum("bhts,bshe->bthe", s.softmax(-1), v)
        x = x + torch.einsum("bthe,hed->btd", z, sd[t + "attn.W_O"])
        n = rms(x, sd[t + "ln2.w"])
        x = x + (torch.nn.functional.silu(n @ sd[t + "mlp.W_gate"]) * (n @ sd[t + "mlp.W_in"])) @ sd[t + "mlp.W_out"]
    return rms(x, sd["ln_final.w"]) @ sd["unembed.W_U"]


def _stats(a, ref):
    import numpy as np
    d = np.abs(a - ref)
    return dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                argmax_agree=float((a.argmax(-1) == ref.argmax(-1)).mean()))


def check(out: Path, c, mlx_npy: Path):
    """Compare MLX logits with (a) the TransformerLens 4.x bridge on the HF export,
    (b) a pure-torch forward over the legacy HookedTransformer-format tensors."""
    import numpy as np
    import torch
    toks = torch.tensor(fixed_tokens(c["vocab"]))
    ref = np.load(mlx_npy)
    res = dict(logit_abs_max=float(np.abs(ref).max()))
    bridge = load_export(out)
    with torch.no_grad():
        res["transformerlens_bridge"] = _stats(bridge(toks, return_type="logits").float().numpy(), ref)
        meta = json.loads((out / "tl_config.json").read_text())
        sd = torch.load(out / "state_dict.pt")
        res["legacy_hookedtransformer_tensors_pure_torch"] = _stats(ref_forward(sd, meta, toks).numpy(), ref)
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="run dir or model.safetensors")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="compare against MLX logits (spawns .venv/bin/python)")
    ap.add_argument("--mlx-dump", type=Path, default=None, help="(MLX venv) dump logits to this .npy and exit")
    a = ap.parse_args()
    st, cfgp, name = resolve(a.src)
    if a.mlx_dump:
        return mlx_dump(st, cfgp, a.mlx_dump)
    out = (a.out or ROOT / "exports" / "tl" / name).resolve()
    export(st, cfgp, out)
    if a.check:
        npy = out / "mlx_logits.npy"
        subprocess.run([str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), str(st),
                        "--mlx-dump", str(npy)], check=True)
        res = check(out, load_cfg(cfgp), npy)
        (out / "check.json").write_text(json.dumps(res, indent=2))
        if max(res["transformerlens_bridge"]["max_abs_diff"],
               res["legacy_hookedtransformer_tensors_pure_torch"]["max_abs_diff"]) >= 1e-3:
            sys.exit("CHECK FAILED: max abs diff >= 1e-3")


if __name__ == "__main__":
    main()

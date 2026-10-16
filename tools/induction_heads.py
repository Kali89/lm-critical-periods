#!/usr/bin/env python
"""Per-head induction analysis (Olsson et al. 2022) of an export made by tools/export_tl.py.

Usage (run with .venv-torch/bin/python):
    tools/induction_heads.py <export_dir> [--len 64 --n 32] [--seed 0] [--out file.json]

Input: n sequences, each a random token block of length L repeated twice (2L tokens, no BOS).
  induction_score   mean attention from position t in the second copy to position t-L+1, i.e. the token
                    that FOLLOWED the previous occurrence of the current token (t in [L, 2L-2]).
                    This is Olsson's "prefix-matching score"; both names are reported (same number).
  prev_token_score  mean attention from t to t-1, over the same positions (the other half of the circuit).
  copying_score     OV-circuit copying (Olsson): for a head, M = E_n W_V W_O W_U on a random subset of
                    tokens (E_n = RMS-normalised, ln1-scaled embeddings); score = sum(eig(M)) / sum(|eig(M)|),
                    in [-1, 1]. Computed from embeddings, so it is an approximation for layers > 0.
Output: JSON on stdout (and --out) with [layer][head] matrices and the max.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_tl  # noqa: E402


def copying_scores(sd, cfg, n_tok=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(cfg["d_vocab"], generator=g)[:n_tok]
    E = sd["embed.W_E"][idx]
    E = E * torch.rsqrt(E.pow(2).mean(-1, keepdim=True) + cfg["eps"])
    U = sd["unembed.W_U"][:, idx]
    out = []
    for i in range(cfg["n_layers"]):
        t = f"blocks.{i}."
        En = E * sd[t + "ln1.w"]
        row = []
        for h in range(cfg["n_heads"]):
            M = En @ sd[t + "attn.W_V"][h] @ sd[t + "attn.W_O"][h] @ U
            ev = torch.linalg.eigvals(M).real
            row.append(float(ev.sum() / ev.abs().sum().clamp_min(1e-12)))
        out.append(row)
    return out


@torch.no_grad()
def analyse(export_dir, L=64, n=32, seed=0):
    bridge = export_tl.load_export(export_dir)
    meta = json.loads((Path(export_dir) / "tl_config.json").read_text())
    sd = torch.load(Path(export_dir) / "state_dict.pt")
    nl, nh, V = meta["n_layers"], meta["n_heads"], meta["d_vocab"]
    if 2 * L > meta["n_ctx"]:
        sys.exit(f"2*len={2 * L} exceeds n_ctx={meta['n_ctx']}")
    g = torch.Generator().manual_seed(seed)
    half = torch.randint(0, V, (n, L), generator=g)
    toks = torch.cat([half, half], 1)
    _, cache = bridge.run_with_cache(toks, names_filter=lambda k: k.endswith("attn.hook_pattern"))
    t = torch.arange(L, 2 * L - 1)
    ind, prev = [], []
    for i in range(nl):
        pat = cache[f"blocks.{i}.attn.hook_pattern"]            # (n, head, q, k)
        ind.append(pat[:, :, t, t - L + 1].mean((0, 2)).tolist())
        prev.append(pat[:, :, t, t - 1].mean((0, 2)).tolist())
    cp = copying_scores(sd, meta, seed=seed)
    flat = [(ind[i][h], i, h) for i in range(nl) for h in range(nh)]
    best = max(flat)
    return dict(
        export=str(export_dir), repeat_len=L, n_sequences=n, seed=seed,
        induction_score=ind, prefix_matching_score=ind, prev_token_score=prev, copying_score=cp,
        max_induction_score=best[0], max_induction_head={"layer": best[1], "head": best[2]},
        max_prev_token_score=max(max(r) for r in prev), max_copying_score=max(max(r) for r in cp),
        uniform_attention_baseline=float((1.0 / (t + 1).float()).mean()),   # a head attending uniformly to its causal window
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--len", type=int, default=64, dest="L")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: <export_dir>/induction_heads.json")
    a = ap.parse_args()
    res = analyse(a.export_dir, a.L, a.n, a.seed)
    out = a.out or a.export_dir / "induction_heads.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

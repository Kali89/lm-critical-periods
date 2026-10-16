"""Direct plasticity measurements for critical-period experiments.

Fisher information trace over training (Achille, Rovere & Soatto, ICLR 2019) plus standard plasticity-loss
diagnostics: gradient / parameter norms, effective rank of representations and embeddings (Lyle et al. 2023),
and dead-unit fraction (Dohare et al. 2024).
"""
import math
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .model import causal_mask  # noqa: F401  (callers build the mask with this)

MAX_RANK_ROWS = 2048
DEAD_EPS = 1e-6
DEAD_TOKEN_FRAC = 0.99


def _sqnorm(tree):
    return sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in tree_flatten(tree))


def _eff_rank(m):
    """exp(entropy of normalised singular values) of a 2-D matrix; float32 SVD on the CPU stream."""
    s = mx.linalg.svd(m.astype(mx.float32), compute_uv=False, stream=mx.cpu)
    p = s / mx.maximum(mx.sum(s), 1e-30)
    ent = -mx.sum(mx.where(p > 0, p * mx.log(mx.maximum(p, 1e-30)), 0.0))
    return mx.exp(ent)


def metrics(model, x, mask, vocab, n_fisher_seqs=16, seed=0) -> dict:
    """Plasticity diagnostics for `model` on a batch.

    x     : int32 (B, T+1) token window, as produced by train.make_batch (inputs x[:, :-1], targets x[:, 1:]).
            If you pass (B, T) the last token is used only as a target, so you lose one position; nothing breaks.
    mask  : additive causal (T, T) mask from lmcp.model.causal_mask; sliced to the input length, so a mask
            built for T or T+1 both work. (Short-context masks are sliced, not rebuilt.)
    vocab : vocabulary size (kept for interface symmetry; sampling uses the logits' own width).

    Normalisation. Every loss here is the per-token mean NLL over one sequence (or the whole batch for
    grad_norm). Fisher traces are the squared L2 norm of its gradient, summed over all parameters, averaged
    over the first `n_fisher_seqs` sequences (a single-sequence gradient per sequence, so this is the
    trace of the per-sequence-mean-gradient outer product, not the per-token Fisher). `fisher_trace` samples
    y~ from the model's own softmax (true Fisher); `fisher_trace_empirical` uses the real next tokens.
    All sampling is keyed by `seed`. Returns python floats.
    """
    t0 = time.time()
    x = mx.array(x).astype(mx.int32)
    inp, tgt = x[:, :-1], x[:, 1:]
    L = inp.shape[1]
    mask = mask[:L, :L]
    B = inp.shape[0]
    n = min(n_fisher_seqs, B)
    key = mx.random.key(seed)

    def nll_true(model, xi, yi):
        return nn.losses.cross_entropy(model(xi, mask), yi, reduction="mean")

    def nll_sampled(model, xi, k):
        logits = model(xi, mask).astype(mx.float32)
        y = mx.stop_gradient(mx.random.categorical(mx.stop_gradient(logits), key=k))
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    g_true = nn.value_and_grad(model, nll_true)
    g_samp = nn.value_and_grad(model, nll_sampled)

    out = {}
    # full-batch gradient on true targets
    _, g = g_true(model, inp, tgt)
    out["plast/grad_norm"] = mx.sqrt(_sqnorm(g))
    del g

    keys = mx.random.split(key, n)
    fs, fe = [], []
    for i in range(n):
        xi, yi = inp[i:i + 1], tgt[i:i + 1]
        _, gs = g_samp(model, xi, keys[i])
        _, ge = g_true(model, xi, yi)
        fs.append(_sqnorm(gs))
        fe.append(_sqnorm(ge))
        mx.eval(fs[-1], fe[-1])
    out["plast/fisher_trace"] = mx.mean(mx.stack(fs))
    out["plast/fisher_trace_empirical"] = mx.mean(mx.stack(fe))

    params = tree_flatten(model.parameters())
    out["plast/param_norm"] = mx.sqrt(sum(mx.sum(p.astype(mx.float32) ** 2) for _, p in params))
    emb = model.wte.weight.astype(mx.float32)
    out["plast/param_norm_emb"] = mx.sqrt(mx.sum(emb ** 2))

    # hidden-state effective rank on <= MAX_RANK_ROWS deterministically strided rows
    h = model.hidden(inp, mask).astype(mx.float32).reshape(-1, model.wte.weight.shape[1])
    N = h.shape[0]
    if N > MAX_RANK_ROWS:
        idx = mx.array([int(i * N / MAX_RANK_ROWS) for i in range(MAX_RANK_ROWS)], dtype=mx.int32)
        h = h[idx]
    out["plast/eff_rank_resid"] = _eff_rank(h)
    out["plast/eff_rank_emb"] = _eff_rank(emb)

    # dead MLP units: |silu(w1 h) * w3 h| < eps on > 99% of tokens
    dead, total = mx.array(0.0), 0
    xx = model.wte(inp)
    for b in model.blocks:
        xx = xx + b.attn(b.n1(xx), mask)
        hh = b.n2(xx)
        act = (nn.silu(b.mlp.w1(hh)) * b.mlp.w3(hh)).astype(mx.float32)
        small = (mx.abs(act) < DEAD_EPS).reshape(-1, act.shape[-1])
        frac_small = mx.mean(small.astype(mx.float32), axis=0)
        dead = dead + mx.sum((frac_small > DEAD_TOKEN_FRAC).astype(mx.float32))
        total += act.shape[-1]
        xx = xx + b.mlp.w2(act.astype(xx.dtype))
    out["plast/dead_frac_mlp"] = dead / max(total, 1)

    keys_ = list(out)
    mx.eval([out[k] for k in keys_])
    res = {k: float(out[k].item()) for k in keys_}
    res["plast/seconds"] = time.time() - t0
    return res

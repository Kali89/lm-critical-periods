"""A small pre-norm GPT in MLX: RMSNorm, RoPE, SwiGLU, tied embeddings, no biases."""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.rope = nn.RoPE(self.hd, traditional=False)

    def __call__(self, x, mask):
        B, T, D = x.shape
        q, k, v = mx.split(self.wqkv(x), 3, axis=-1)
        q = self.rope(q.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3))
        k = self.rope(k.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3))
        v = v.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.hd), mask=mask)
        return self.wo(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = 64 * ((int(8 * d / 3) + 63) // 64)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.attn = Attention(d, h)
        self.n2 = nn.RMSNorm(d)
        self.mlp = MLP(d)

    def __call__(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, vocab, n_layer, d_model, n_head):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.blocks = [Block(d_model, n_head) for _ in range(n_layer)]
        self.nf = nn.RMSNorm(d_model)
        self.n_layer = n_layer

    def hidden(self, idx, mask):
        """Final normalised hidden states (B, T, d): __call__ without the unembedding."""
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x, mask)
        return self.nf(x)

    def __call__(self, idx, mask):
        return self.wte.as_linear(self.hidden(idx, mask))


def init_gpt2_style(model, std=0.02):
    """Normal(0, 0.02) for all matrices, residual projections scaled by 1/sqrt(2*n_layer)."""
    scale = 1.0 / math.sqrt(2 * model.n_layer)
    new = {}
    for name, p in tree_flatten(model.parameters()):
        if name.endswith(".weight") and p.ndim == 2:
            s = std * scale if (".wo." in name or ".w2." in name) else std
            new[name] = mx.random.normal(p.shape) * s
        elif name.endswith("weight") and p.ndim == 1:   # norms
            new[name] = mx.ones(p.shape)
        else:
            new[name] = p
    model.update(tree_unflatten(list(new.items())))


def build_model(cfg, vocab):
    m = cfg["model"]
    model = GPT(vocab, m["n_layer"], m["d_model"], m["n_head"])
    init_gpt2_style(model)
    mx.eval(model.parameters())
    return model


def n_params(model):
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def causal_mask(T, block=None):
    """Additive (T,T) mask. block=k restricts attention to k-token blocks: the 'short context' corruption."""
    i = mx.arange(T)[:, None]
    j = mx.arange(T)[None, :]
    allowed = j <= i
    if block:
        allowed = allowed & ((i // block) == (j // block))
    return mx.where(allowed, mx.array(0.0), mx.array(-1e9))

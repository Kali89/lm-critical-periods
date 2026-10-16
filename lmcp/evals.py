"""Per-ability probes, all cheap enough to run every few million tokens.

  val_loss (+ by position bucket)  overall fit; long_range_gain = loss[0:32) - loss[128:) shows how much the
                                   model benefits from long context
  blimp                            67 minimal-pair paradigms grouped into 12 linguistic terms (syntax)
  induction                        random token sequence repeated twice; loss/accuracy on the second copy
                                   measures in-context copying (induction heads)
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import DATA
from .model import causal_mask
from .tok import EOS_ID, PAD_ID

POS_BUCKETS = [(0, 32), (32, 128), (128, 10 ** 9)]


def _token_logprobs(model, x):
    """x: (B, T) int array. Returns (B, T-1) log p(x[t+1] | x[<=t])."""
    T = x.shape[1]
    logits = model(x[:, :-1], causal_mask(T - 1))
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(lp, x[:, 1:, None], axis=-1)[..., 0]


def val_loss(model, val_stream, seq_len, max_tokens, batch=32):
    n_seq = min(len(val_stream) // (seq_len + 1), max(1, max_tokens // seq_len))
    data = np.array(val_stream[: n_seq * (seq_len + 1)]).reshape(n_seq, seq_len + 1).astype(np.int32)
    pos_sum = np.zeros(seq_len)
    for i in range(0, n_seq, batch):
        lp = _token_logprobs(model, mx.array(data[i:i + batch]))
        mx.eval(lp)
        pos_sum += -np.array(lp).sum(axis=0)
    pos_mean = pos_sum / n_seq
    out = {"val_loss": float(pos_mean.mean())}
    for a, b in POS_BUCKETS:
        out[f"val_loss_pos_{a}_{min(b, seq_len)}"] = float(pos_mean[a:b].mean())
    out["long_range_gain"] = out["val_loss_pos_0_32"] - out[f"val_loss_pos_128_{seq_len}"]
    return out


_BLIMP_CACHE = {}


def load_blimp(per_paradigm=None):
    key = per_paradigm
    if key in _BLIMP_CACHE:
        return _BLIMP_CACHE[key]
    items = []
    for p in sorted((DATA / "blimp").glob("*.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if per_paradigm:
            rows = rows[:per_paradigm]
        for r in rows:
            items.append((r["UID"], r["linguistics_term"], r["sentence_good"], r["sentence_bad"]))
    _BLIMP_CACHE[key] = items
    return items


def _sentence_logprobs(model, tok, sents, batch=128):
    encs = [[EOS_ID] + e.ids for e in tok.encode_batch(sents)]
    order = np.argsort([len(e) for e in encs])
    scores = np.zeros(len(encs))
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        L = max(len(encs[j]) for j in idx)
        arr = np.full((len(idx), L), PAD_ID, dtype=np.int32)
        for r, j in enumerate(idx):
            arr[r, :len(encs[j])] = encs[j]
        lp = _token_logprobs(model, mx.array(arr))
        mx.eval(lp)
        lp = np.array(lp)
        for r, j in enumerate(idx):
            scores[j] = lp[r, :len(encs[j]) - 1].sum()
    return scores


def blimp(model, tok, per_paradigm=None, lowercase=True):
    items = load_blimp(per_paradigm)
    norm = (lambda s: s.lower()) if lowercase else (lambda s: s)
    good = _sentence_logprobs(model, tok, [norm(g) for _, _, g, _ in items])
    bad = _sentence_logprobs(model, tok, [norm(b) for _, _, _, b in items])
    correct = good > bad
    by_term, by_uid = {}, {}
    for (uid, term, _, _), c in zip(items, correct):
        by_term.setdefault(term, []).append(c)
        by_uid.setdefault(uid, []).append(c)
    out = {"blimp_acc": float(correct.mean())}
    out.update({f"blimp/{t}": float(np.mean(v)) for t, v in by_term.items()})
    out["blimp_paradigms"] = {u: float(np.mean(v)) for u, v in by_uid.items()}
    return out


def induction(model, vocab, length=64, n_batches=8, batch=32, seed=99):
    rng = np.random.default_rng(seed)
    first, second, acc = [], [], []
    for _ in range(n_batches):
        r = rng.integers(2, vocab, size=(batch, length))
        x = mx.array(np.concatenate([r, r], axis=1).astype(np.int32))
        T = x.shape[1]
        logits = model(x[:, :-1], causal_mask(T - 1))
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = x[:, 1:]
        nll = -mx.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0]
        pred = mx.argmax(logits, axis=-1)
        mx.eval(nll, pred)
        nll, pred, tgt = np.array(nll), np.array(pred), np.array(tgt)
        first.append(nll[:, : length - 1].mean())
        second.append(nll[:, length:].mean())          # predicting tokens of the second copy
        acc.append((pred[:, length:] == tgt[:, length:]).mean())
    f, s = float(np.mean(first)), float(np.mean(second))
    return {"induction_loss_first": f, "induction_loss_second": s, "induction_gain": f - s,
            "induction_acc": float(np.mean(acc))}


def run_all(model, tok, val_stream, cfg, final=False):
    ev = cfg["eval"]
    vocab = cfg["data"]["tokenizer_vocab"]
    out = {}
    out.update(val_loss(model, val_stream, cfg["data"]["seq_len"], ev["val_tokens"]))
    out.update(induction(model, vocab, ev["induction_len"], ev["induction_batches"]))
    per = None if (final and ev["blimp_final_full"]) else ev["blimp_per_paradigm"]
    out.update(blimp(model, tok, per, lowercase=ev.get("blimp_lowercase", True)))
    return out

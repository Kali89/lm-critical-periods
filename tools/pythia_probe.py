#!/usr/bin/env python
"""Evaluate EleutherAI Pythia checkpoints with the same probes as lmcp/evals.py.

Run with .venv-torch:
  .venv-torch/bin/python tools/pythia_probe.py --model pythia-14m --steps 0,1000,143000
  .venv-torch/bin/python tools/pythia_probe.py --model pythia-70m --steps log

Appends one JSON line per checkpoint to results/external/<model>.jsonl. See tools/README-external.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

ROOT = Path(__file__).resolve().parent.parent
TOKENS_PER_STEP = 2_097_152
EOS_ID = 0  # <|endoftext|>; Pythia uses it as both BOS and EOS
POS_BUCKETS = [(0, 32), (32, 128), (128, 512)]
LOG_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]


def parse_steps(s):
    if s == "log":
        return LOG_STEPS
    return [int(x) for x in s.split(",") if x.strip()]


def device_name():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def token_logprobs(model, x, mask=None):
    """x (B,T) -> (B,T-1) log p(x[t+1] | x[<=t])."""
    logits = model(input_ids=x[:, :-1], attention_mask=None if mask is None else mask[:, :-1]).logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp.gather(-1, x[:, 1:, None])[..., 0]


@torch.no_grad()
def sentence_logprobs(model, tok, sents, dev, batch=128):
    encs = [[EOS_ID] + e for e in tok(sents, add_special_tokens=False)["input_ids"]]
    order = np.argsort([len(e) for e in encs])
    scores = np.zeros(len(encs))
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        L = max(len(encs[j]) for j in idx)
        arr = np.zeros((len(idx), L), dtype=np.int64)
        m = np.zeros((len(idx), L), dtype=np.int64)
        for r, j in enumerate(idx):
            arr[r, :len(encs[j])] = encs[j]
            m[r, :len(encs[j])] = 1
        lp = token_logprobs(model, torch.from_numpy(arr).to(dev), torch.from_numpy(m).to(dev)).cpu().numpy()
        for r, j in enumerate(idx):
            scores[j] = lp[r, :len(encs[j]) - 1].sum()
    return scores


def load_blimp(per_paradigm):
    items = []
    for p in sorted((ROOT / "data" / "blimp").glob("*.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if per_paradigm:
            rows = rows[:per_paradigm]
        for r in rows:
            items.append((r["UID"], r["linguistics_term"], r["sentence_good"], r["sentence_bad"]))
    return items


def blimp(model, tok, items, dev):
    good = sentence_logprobs(model, tok, [g for _, _, g, _ in items], dev)  # cased, as-is
    bad = sentence_logprobs(model, tok, [b for _, _, _, b in items], dev)
    correct = good > bad
    by_term = {}
    for (_, term, _, _), c in zip(items, correct):
        by_term.setdefault(term, []).append(c)
    out = {"blimp_acc": float(correct.mean())}
    out.update({f"blimp/{t}": float(np.mean(v)) for t, v in by_term.items()})
    return out


@torch.no_grad()
def induction(model, vocab, dev, length=64, n_batches=8, batch=32, seed=99):
    rng = np.random.default_rng(seed)
    first, second, acc = [], [], []
    for _ in range(n_batches):
        r = rng.integers(2, vocab, size=(batch, length))
        x = torch.from_numpy(np.concatenate([r, r], axis=1).astype(np.int64)).to(dev)
        logits = model(input_ids=x[:, :-1]).logits.float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = x[:, 1:]
        nll = -lp.gather(-1, tgt[..., None])[..., 0]
        pred = logits.argmax(-1)
        nll, pred, tgt = nll.cpu().numpy(), pred.cpu().numpy(), tgt.cpu().numpy()
        first.append(nll[:, : length - 1].mean())
        second.append(nll[:, length:].mean())
        acc.append((pred[:, length:] == tgt[:, length:]).mean())
    f, s = float(np.mean(first)), float(np.mean(second))
    return {"induction_loss_first": f, "induction_loss_second": s, "induction_gain": f - s,
            "induction_acc": float(np.mean(acc))}


def build_val(tok, max_tokens, seq_len=512):
    """Tokenise val docs, EOS after each, pack to (n, seq_len+1); same chunking as lmcp val_loss."""
    ids, need = [], max_tokens + seq_len + 1
    total = 0
    with open(ROOT / "data" / "corpora" / "babylm100m_clean" / "val.docs.jsonl") as f:
        for line in f:
            e = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"] + [EOS_ID]
            ids.extend(e)
            total += len(e)
            if total >= need:
                break
    n_seq = min(len(ids) // (seq_len + 1), max(1, max_tokens // seq_len))
    return np.array(ids[: n_seq * (seq_len + 1)], dtype=np.int64).reshape(n_seq, seq_len + 1)


@torch.no_grad()
def val_loss(model, data, dev, batch=32):
    seq_len = data.shape[1] - 1
    pos_sum = np.zeros(seq_len)
    for i in range(0, len(data), batch):
        x = torch.from_numpy(data[i:i + batch]).to(dev)
        pos_sum += -token_logprobs(model, x).sum(0).cpu().numpy()
    pos_mean = pos_sum / len(data)
    out = {"val_loss": float(pos_mean.mean())}
    for a, b in POS_BUCKETS:
        out[f"val_loss_pos_{a}_{b}"] = float(pos_mean[a:b].mean())
    out["long_range_gain"] = out["val_loss_pos_0_32"] - out["val_loss_pos_128_512"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-14m")
    ap.add_argument("--steps", default="log")
    ap.add_argument("--blimp-per-paradigm", type=int, default=200, help="0 = all 1000")
    ap.add_argument("--val-tokens", type=int, default=500_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = device_name()
    repo = f"EleutherAI/{a.model}"
    out_p = Path(a.out) if a.out else ROOT / "results" / "external" / f"{a.model}.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(repo)
    items = load_blimp(a.blimp_per_paradigm)
    val = build_val(tok, a.val_tokens)
    vocab = len(tok)  # real tokenizer vocab (50277); the 50304 embedding rows beyond it are padding
    print(f"device={dev} blimp_pairs={len(items)} val_seqs={len(val)} ({val.shape[0] * 512} scored tokens)", flush=True)

    for step in parse_steps(a.steps):
        t0 = time.time()
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision=f"step{step}", torch_dtype=torch.float32).to(dev).eval()
        n_params = sum(p.numel() for p in model.parameters())
        tokens = step * TOKENS_PER_STEP
        row = {"model": a.model, "step": step, "tokens": tokens, "tokens_per_param": tokens / n_params,
               "n_params": n_params}
        row.update(blimp(model, tok, items, dev))
        row.update(induction(model, vocab, dev))
        row.update(val_loss(model, val, dev))
        row["eval_seconds"] = round(time.time() - t0, 1)
        with open(out_p, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {step:>6}  blimp {row['blimp_acc']:.3f}  ind_acc {row['induction_acc']:.3f}  "
              f"val {row['val_loss']:.3f}  lrg {row['long_range_gain']:.3f}  {row['eval_seconds']}s", flush=True)
        del model


if __name__ == "__main__":
    main()

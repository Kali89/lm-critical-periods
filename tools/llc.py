#!/usr/bin/env python
"""Local learning coefficient (LLC) of an exported model via devinterp's SGLD sampler.

Usage (run with .venv-torch/bin/python):
    tools/llc.py <export_dir> --data data/tok/babylm100m_clean_v8192/val.clean.bin \
        [--n-chains 2 --n-steps 200 --batch 8 --seq 128] [--lr 1e-4 --gamma 100 --nbeta X] [--device cpu]

Loss: next-token cross-entropy (nats/token) on `seq`-token windows cut from a uint16 token stream
(inputs x[:-1], targets x[1:]; devinterp shifts internally, so each window contributes seq-1 predictions).
Windows are non-overlapping, drawn at random (--seed) from the stream; the pool is --pool windows, cycled.
Model: the HF Llama export in <export_dir>/hf (identical function to the MLX model; see export_tl.py).
Estimator (devinterp 2.x):  LLC = nbeta * (mean SGLD-chain loss - loss at the checkpoint),
SGLD with localization gamma pulling toward the checkpoint; --n-steps draws per chain, no burn-in.
nbeta defaults to devinterp.utils.default_nbeta(batch) = batch / ln(batch).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")   # one-line output; unset to see chain progress bars

import numpy as np
import torch


def make_dataset(path, seq, pool, seed):
    from datasets import Dataset
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    n_win = len(tokens) // seq
    rng = np.random.default_rng(seed)
    pool = min(pool, n_win)
    starts = rng.choice(n_win, size=pool, replace=False) * seq
    ids = np.stack([np.asarray(tokens[s:s + seq], dtype=np.int64) for s in starts])
    ds = Dataset.from_dict({"input_ids": ids.tolist()})
    ds.set_format("torch")
    return ds, len(tokens)


def fallback_llc(model, ds, a, nbeta):
    """Same estimator without devinterp.slt.sample (needs zarr): devinterp's SGMCMC.sgld optimizer driven
    by a plain loop. Used only if `devinterp.slt.llc` cannot be imported (zarr>=3.2 breaks devinterp 2.0.1)."""
    from torch.utils.data import DataLoader
    from devinterp.optim import SGMCMC
    from devinterp.slt.lm_loss import compute_per_token_loss

    def loader(seed):
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=a.batch, shuffle=True, generator=g, drop_last=True)

    model.to(a.device).train()
    with torch.no_grad():
        it, init = iter(loader(a.seed)), []
        for _ in range(a.init_loss_batches):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader(a.seed)); b = next(it)
            init.append(compute_per_token_loss(model, b["input_ids"].to(a.device)).mean().item())
    init_loss = float(np.mean(init))
    traces = []
    for c in range(a.n_chains):
        m = copy.deepcopy(model)
        torch.manual_seed(a.seed + c)
        opt = SGMCMC.sgld(m.parameters(), lr=a.lr, nbeta=nbeta, localization=a.gamma)
        feed, tr = iter(loader(a.seed)), []   # same data order in every chain, as devinterp's default
        for _ in range(a.n_steps):
            try:
                b = next(feed)
            except StopIteration:
                feed = iter(loader(a.seed)); b = next(feed)
            loss = compute_per_token_loss(m, b["input_ids"].to(a.device)).mean()
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            tr.append(loss.item())
        traces.append(tr)
    trace = np.array(traces)
    per_chain = nbeta * (trace.mean(1) - init_loss)
    return per_chain.mean(), per_chain.std(), per_chain, init_loss, trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-chains", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=200, help="SGLD draws (= steps) per chain")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=100.0, help="localization strength")
    ap.add_argument("--nbeta", type=float, default=None, help="default: default_nbeta(batch)")
    ap.add_argument("--pool", type=int, default=2048, help="number of data windows to draw from the stream")
    ap.add_argument("--init-loss-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--out", type=Path, default=None, help="default: <export_dir>/llc.json")
    a = ap.parse_args()

    from transformers import LlamaForCausalLM
    from devinterp.utils import default_nbeta

    nbeta = a.nbeta if a.nbeta is not None else float(default_nbeta(a.batch))
    model = LlamaForCausalLM.from_pretrained(a.export_dir / "hf", torch_dtype=torch.float32,
                                             attn_implementation="eager")
    ds, n_tok = make_dataset(a.data, a.seq, a.pool, a.seed)
    if len(ds) < a.batch * 2:
        sys.exit("data pool smaller than 2 batches")
    t0 = time.time()
    try:
        from devinterp.slt.llc import llc
    except ImportError as e:
        print(f"[llc] devinterp.slt.llc unavailable ({e}); using in-script SGLD loop with devinterp.optim", file=sys.stderr)
        llc = None
    if llc is not None:
        res = llc(model, ds, observables={}, lr=a.lr, n_beta=nbeta, localization=a.gamma,
                  num_chains=a.n_chains, num_draws=a.n_steps, batch_size=a.batch,
                  num_init_loss_batches=a.init_loss_batches, init_seed=a.seed, device=a.device)
        mean, std = float(res.llc_mean), float(res.llc_std)
        per_chain, init_loss, trace = res.llc_per_chain.values, float(res.init_loss), res.loss_trace.values
        impl = "devinterp.slt.llc.llc"
    else:
        mean, std, per_chain, init_loss, trace = fallback_llc(model, ds, a, nbeta)
        impl = "fallback: devinterp.optim.SGMCMC.sgld + manual loop"
    dt = time.time() - t0
    out = dict(
        export=str(a.export_dir), data=str(a.data), stream_tokens=int(n_tok),
        implementation=impl, llc_mean=float(mean), llc_std=float(std),
        llc_per_chain=[float(x) for x in per_chain], init_loss=float(init_loss),
        sampling_loss_mean=float(trace.mean()), sampling_loss_first10=float(trace[:, :10].mean()),
        sampling_loss_last10=float(trace[:, -10:].mean()),
        nbeta=nbeta, lr=a.lr, gamma=a.gamma, n_chains=a.n_chains, n_steps=a.n_steps, batch=a.batch,
        seq=a.seq, pool=len(ds), device=a.device, runtime_s=dt,
        n_params=sum(p.numel() for p in model.parameters()),
    )
    (a.out or a.export_dir / "llc.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"LLC = {out['llc_mean']:.3f} +/- {out['llc_std']:.3f} (std over {a.n_chains} chains); runtime {dt:.1f}s")


if __name__ == "__main__":
    main()

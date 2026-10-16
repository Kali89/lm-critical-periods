"""One training run: per-seed document order, windowed corruption schedule, optional stream phases,
periodic per-ability evals with plasticity measurements, resumable checkpoints.

Run directory layout (runs/<name>/):
  config.json        fully resolved config + environment/provenance (git rev, doc-order seed and hash)
  doc_order.npy      the document permutation this run read
  status.json        running | done | failed | interrupted, with step/tokens/timestamps
  train_curve.jsonl  every log_every steps: loss, lr, phase, throughput, epochs (loss bumps = phase transitions)
  metrics.jsonl      one line per eval: val loss by position (per val stream), BLiMP by term with bootstrap CIs,
                     induction, plasticity (Fisher trace, norms, effective ranks)
  ckpt/              resumable checkpoint (model + optimizer + reader cursor)
  final/model.safetensors
"""
import hashlib
import json
import platform
import socket
import subprocess
import time
import traceback
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from . import ROOT, RUNS, __version__, evals
from .config import TEXT_KINDS, TRAIN_KINDS, stream_dir
from .corrupt import stream_name
from .model import build_model, causal_mask, n_params
from .schedule import in_window, lr_at
from .tok import load_tokenizer, open_indexed, open_stream

try:
    from . import plasticity
except ImportError:          # pragma: no cover
    plasticity = None

DOC_ORDER_SALT = 20251016


def git_info():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "lmcp", "configs"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return {"git_rev": rev or None, "git_dirty": dirty}
    except Exception:
        return {"git_rev": None, "git_dirty": None}


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str))


def append_jsonl(p, obj):
    with open(p, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


# ---------------------------------------------------------------- streams and schedule

def phase_at(phases, frac):
    """Return the {streams: {name: weight}} phase active at frac (phases sorted by `until`)."""
    for ph in phases:
        if frac < float(ph["until"]):
            return ph
    return phases[-1]


def plan_streams(cfg):
    """Which train/val streams this run needs. Returns (train_names, corrupt_name_or_None, val_names)."""
    co, tr, ev = cfg["corruption"], cfg["train"], cfg["eval"]
    kind, param, inten = co["kind"], co["param"], co.get("intensity", 1.0)
    if kind not in TEXT_KINDS and kind not in TRAIN_KINDS:
        raise SystemExit(f"unknown corruption kind {kind}")
    train_names = {"clean"}
    if tr.get("phases"):
        for ph in tr["phases"]:
            train_names |= set(ph["streams"])
    corrupt = stream_name(kind, param, inten) if (kind in TEXT_KINDS and kind != "clean") else None
    if corrupt:
        train_names.add(corrupt)
    return sorted(train_names), corrupt, list(ev.get("val_streams") or ["clean"])


def required_streams(cfg):
    sd = stream_dir(cfg)
    train_names, corrupt, val_names = plan_streams(cfg)
    need = {"tokenizer": sd / "tokenizer.json"}
    for n in train_names:
        need[f"train.{n}"] = sd / f"train.{n}.bin"
        need[f"train.{n}.idx"] = sd / f"train.{n}.idx.npy"
    for n in val_names:
        need[f"val.{n}"] = sd / f"val.{n}.bin"
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        co = cfg["corruption"]
        hint = f"bin/lmcp tok encode --kind {co['kind']}" + (f" --param {co['param']}" if co["param"] is not None else "") \
               + (f" --intensity {co['intensity']}" if float(co.get("intensity", 1.0)) != 1.0 else "")
        raise SystemExit("missing data files:\n  " + "\n  ".join(missing) +
                         f"\nbuild them with: bin/lmcp prepare-all  and/or  {hint}  (streams need a .idx.npy: re-encode old ones)")
    return sd, train_names, corrupt, val_names


class DocReader:
    """B readers walking a seed-specific document permutation. Reader b owns the b-th contiguous shard of the
    permutation and reads its documents in order, wrapping to its shard start (a new epoch for that reader).
    All streams share the permutation and the document ids, so switching a reader from the clean stream to a
    corrupted one mid-document continues on the same document (offset clamped to that stream's doc length).
    Each step consumes T+1 tokens per reader; the one-token overlap between consecutive windows is not kept,
    so one token per reader per step is never a prediction target (~0.2%)."""

    def __init__(self, streams, perm, B, T):
        self.streams, self.perm, self.B, self.T = streams, perm, B, T
        self.shard = len(perm) // B
        if self.shard < 1:
            raise SystemExit("more readers than documents")
        self.doc = np.array([b * self.shard for b in range(B)], dtype=np.int64)
        self.off = np.zeros(B, dtype=np.int64)
        self.epochs = np.zeros(B, dtype=np.int64)

    def state(self):
        return {"doc": self.doc.tolist(), "off": self.off.tolist(), "epochs": self.epochs.tolist()}

    def load_state(self, st):
        self.doc, self.off, self.epochs = (np.array(st[k], dtype=np.int64) for k in ("doc", "off", "epochs"))

    def batch(self, names):
        out = np.empty((self.B, self.T + 1), dtype=np.int32)
        for b in range(self.B):
            data, idx = self.streams[names[b]]
            need, pos = self.T + 1, 0
            while need > 0:
                d = self.perm[self.doc[b]]
                s, L = idx[d], idx[d + 1] - idx[d]
                off = min(self.off[b], L)
                take = min(need, L - off)
                if take > 0:
                    out[b, pos:pos + take] = data[s + off:s + off + take]
                    pos += take; need -= take; off += take
                if off >= L:
                    nd = self.doc[b] + 1
                    if nd >= (b + 1) * self.shard:
                        nd = b * self.shard
                        self.epochs[b] += 1
                    self.doc[b], self.off[b] = nd, 0
                else:
                    self.off[b] = off
        return out


# ---------------------------------------------------------------- the run

def train(cfg, run_dir, resume=True):
    if cfg.get("task") == "vision":                      # positive-control task shares queue/status/schedule code
        from .vision.train import train as vision_train
        return vision_train(cfg, run_dir, resume)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_p, curve_p, metrics_p = run_dir / "status.json", run_dir / "train_curve.jsonl", run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "ckpt"
    sd, train_names, corrupt_name, val_names = required_streams(cfg)

    d, tr, co, ev = cfg["data"], cfg["train"], cfg["corruption"], cfg["eval"]
    T, vocab = d["seq_len"], d["tokenizer_vocab"]
    B = tr["batch_tokens"] // T
    total_steps = tr["tokens"] // (B * T)
    kind, param, window = co["kind"], co["param"], co["window"]
    inten = float(co.get("intensity", 1.0))
    if window is not None and kind == "clean":
        raise SystemExit("corruption.window set but kind is clean")
    if kind != "clean" and window is None:
        window = [0.0, 1.0]
    phases = tr.get("phases")
    if phases:
        phases = sorted(phases, key=lambda p: float(p["until"]))
        if float(phases[-1]["until"]) < 1.0:
            raise SystemExit("train.phases must end with until: 1.0")

    # ---- skip if done ----
    if resume and status_p.exists() and json.loads(status_p.read_text()).get("status") == "done":
        print(f"{run_dir.name}: already done, skipping (use --no-resume to retrain)", flush=True)
        lines = metrics_p.read_text().splitlines() if metrics_p.exists() else []
        return json.loads(lines[-1]) if lines else None

    mx.random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    tok = load_tokenizer(sd)
    streams = {n: open_indexed(sd, "train", n) for n in train_names}
    vals = {n: open_stream(sd / f"val.{n}.bin") for n in val_names}
    n_docs = min(len(idx) - 1 for _, idx in streams.values())
    doc_order_seed = d.get("doc_order_seed")
    doc_order_seed = tr["seed"] if doc_order_seed is None else int(doc_order_seed)
    perm = np.random.default_rng([DOC_ORDER_SALT, doc_order_seed]).permutation(n_docs)
    reader = DocReader(streams, perm, B, T)
    step_rng = lambda s: np.random.default_rng([tr["seed"], s])

    model = build_model(cfg, vocab)
    optimizer = optim.AdamW(learning_rate=tr["lr"], weight_decay=tr["weight_decay"])
    mask_full = causal_mask(T)
    mask_short = causal_mask(T, int(param or 32)) if kind == "short_context" else mask_full
    p_label = float(param if param is not None else 0.3) * inten if kind == "label_noise" else 0.0
    vperm = None
    if kind == "vocab_permute":
        # consistent relabelling of a fraction `intensity` of the vocabulary (ids >= 2); specials stay fixed
        rng = np.random.default_rng([DOC_ORDER_SALT + 1, tr["seed"]])
        ids = np.arange(2, vocab)
        chosen = rng.choice(ids, size=max(2, int(round(inten * len(ids)))), replace=False)
        vperm = np.arange(vocab)
        vperm[chosen] = rng.permutation(chosen)

    def loss_fn(model, x, y, mask):
        return nn.losses.cross_entropy(model(x, mask), y, reduction="mean")

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(x, y, mask):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y, mask)
        grads, _ = optim.clip_grad_norm(grads, tr["grad_clip"])
        optimizer.update(model, grads)
        return loss

    # ---- resume ----
    step = 0
    if resume and (ckpt_dir / "state.json").exists():
        st = json.loads((ckpt_dir / "state.json").read_text())
        model.load_weights(str(ckpt_dir / "model.safetensors"))
        optimizer.state = tree_unflatten(list(mx.load(str(ckpt_dir / "optimizer.safetensors")).items()))
        step = st["step"]
        reader.load_state(st["reader"])
        mx.eval(model.parameters(), optimizer.state)
        print(f"resumed from step {step}", flush=True)
    else:
        for p in (curve_p, metrics_p):
            p.unlink(missing_ok=True)
        np.save(run_dir / "doc_order.npy", perm)

    info = {"n_params": n_params(model), "batch_size": B, "total_steps": total_steps, "n_docs": n_docs,
            "docs_per_reader": reader.shard, "doc_order_seed": doc_order_seed,
            "doc_order_sha1": hashlib.sha1(perm.tobytes()).hexdigest()[:12],
            "train_streams": train_names, "corrupt_stream": corrupt_name, "val_streams": val_names,
            "host": socket.gethostname(), "platform": platform.platform(), "mlx": mx.__version__,
            "lmcp": __version__, **git_info()}
    write_json(run_dir / "config.json", {"config": cfg, "info": info})
    started = time.time()

    # fixed probe batch for plasticity so the Fisher trace is comparable across evals
    primary_val = vals[val_names[0]]
    n_probe = min(int(ev.get("fisher_seqs", 16)), len(primary_val) // (T + 1))
    probe = mx.array(np.array(primary_val[: n_probe * (T + 1)]).reshape(n_probe, T + 1).astype(np.int32))

    def set_status(s, **kw):
        write_json(status_p, {"status": s, "step": step, "tokens": step * B * T, "total_steps": total_steps,
                              "started": started, "updated": time.time(), **kw})

    def save_ckpt():
        ckpt_dir.mkdir(exist_ok=True)
        mx.save_safetensors(str(ckpt_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
        mx.save_safetensors(str(ckpt_dir / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
        write_json(ckpt_dir / "state.json", {"step": step, "reader": reader.state()})

    def do_eval(final=False):
        t0 = time.time()
        m = evals.run_all(model, tok, vals, cfg, final=final)
        if plasticity is not None and ev.get("plasticity", True):
            m.update(plasticity.metrics(model, probe, mask_full, vocab, n_fisher_seqs=n_probe))
        m.update({"step": step, "tokens": step * B * T, "frac": step / total_steps, "final": final,
                  "epochs": float(reader.epochs.mean()), "eval_seconds": round(time.time() - t0, 1)})
        append_jsonl(metrics_p, m)
        extra = f"  fisher {m['plast/fisher_trace']:.3g}" if "plast/fisher_trace" in m else ""
        print(f"  eval step {step}: val {m['val_loss']:.3f}  blimp {m['blimp_acc']:.3f}  "
              f"induction_acc {m['induction_acc']:.3f}  long_range_gain {m['long_range_gain']:.3f}{extra}  ({m['eval_seconds']}s)", flush=True)
        return m

    set_status("running")
    print(f"{run_dir.name}\n  params {info['n_params']/1e6:.2f}M  steps {total_steps}  batch {B}x{T}  docs {n_docs} (order seed {doc_order_seed})\n"
          f"  kind {kind} param {param} intensity {inten} window {window}  phases {'yes' if phases else 'no'}  schedule {tr['schedule']}", flush=True)
    eval_every = max(1, ev["every_tokens"] // (B * T))
    ckpt_every = max(1, cfg["checkpoint"]["every_tokens"] // (B * T)) if cfg["checkpoint"]["every_tokens"] else None
    if step == 0:
        do_eval()
    losses, t_log, tok_log = [], time.time(), 0
    try:
        while step < total_steps:
            frac = step / total_steps
            rng = step_rng(step)
            active = kind != "clean" and in_window(frac, window)
            if phases:
                ph = phase_at(phases, frac)["streams"]
                names_, w = list(ph), np.array([float(v) for v in ph.values()])
                names = list(rng.choice(names_, size=B, p=w / w.sum()))
            else:
                names = ["clean"] * B
            if active and corrupt_name:
                names = [corrupt_name if n == "clean" else n for n in names]
            batch = reader.batch(names)
            if active and vperm is not None:
                batch = vperm[batch]
            x, y = batch[:, :-1], batch[:, 1:]
            if active and p_label > 0:
                noise = rng.random(y.shape) < p_label
                y = np.where(noise, rng.integers(2, vocab, size=y.shape), y)
            short = active and kind == "short_context" and (inten >= 1.0 or rng.random() < inten)
            mask = mask_short if short else mask_full
            phase = "corrupt" if active else ("mix" if phases else "clean")
            lr = lr_at(step, total_steps, tr)
            optimizer.learning_rate = lr
            loss = step_fn(mx.array(x), mx.array(y), mask)
            mx.eval(state, loss)
            losses.append(loss)
            step += 1
            tok_log += B * T
            if step % tr["log_every"] == 0 or step == total_steps:
                mx.eval(losses)
                dt = time.time() - t_log
                rec = {"step": step, "tokens": step * B * T, "frac": round(step / total_steps, 4), "phase": phase,
                       "loss": float(mx.mean(mx.stack(losses)).item()), "lr": lr, "tok_s": round(tok_log / dt),
                       "epochs": round(float(reader.epochs.mean()), 3)}
                append_jsonl(curve_p, rec)
                eta = (total_steps - step) * dt / max(1, len(losses)) / 60
                print(f"step {step}/{total_steps} {phase:7s} loss {rec['loss']:.3f} lr {lr:.2e} {rec['tok_s']} tok/s  eta {eta:.0f}m", flush=True)
                losses, t_log, tok_log = [], time.time(), 0
            if step % eval_every == 0 and step < total_steps:
                do_eval()
                set_status("running")
            if ckpt_every and step % ckpt_every == 0 and step < total_steps:
                save_ckpt()
        final = do_eval(final=True)
        if cfg["checkpoint"]["final"]:
            (run_dir / "final").mkdir(exist_ok=True)
            mx.save_safetensors(str(run_dir / "final" / "model.safetensors"), dict(tree_flatten(model.parameters())))
        set_status("done", finished=time.time(), elapsed_min=round((time.time() - started) / 60, 1))
        return final
    except BaseException as e:
        save_ckpt()
        set_status("interrupted" if isinstance(e, KeyboardInterrupt) else "failed", error=repr(e))
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise

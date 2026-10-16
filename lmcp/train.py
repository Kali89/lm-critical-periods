"""One training run: windowed corruption schedule, periodic per-ability evals, resumable checkpoints.

Run directory layout (runs/<name>/):
  config.json        fully resolved config + environment/provenance
  status.json        running | done | failed, with step/tokens/timestamps
  train_curve.jsonl  every log_every steps: loss, lr, phase, throughput (loss bumps = phase transitions)
  metrics.jsonl      one line per eval: val loss by position, BLiMP by term, induction
  ckpt/              resumable checkpoint (model + optimizer + cursor)
  final/model.safetensors
"""
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
from .tok import load_tokenizer, open_stream


def git_info():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "lmcp", "configs"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return {"git_rev": rev or None, "git_dirty": dirty}
    except Exception:
        return {"git_rev": None, "git_dirty": None}


def required_streams(cfg):
    kind, param = cfg["corruption"]["kind"], cfg["corruption"]["param"]
    sd = stream_dir(cfg)
    need = {"train_clean": sd / "train.clean.bin", "val": sd / "val.clean.bin", "tokenizer": sd / "tokenizer.json"}
    if kind in TEXT_KINDS and kind != "clean":
        need["train_corrupt"] = sd / f"train.{stream_name(kind, param)}.bin"
    elif kind not in TEXT_KINDS and kind not in TRAIN_KINDS:
        raise SystemExit(f"unknown corruption kind {kind}")
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        hint = f"bin/lmcp tok encode --kind {kind}" + (f" --param {param}" if param is not None else "")
        raise SystemExit("missing data files:\n  " + "\n  ".join(missing) + f"\nbuild them with: bin/lmcp prepare-all  and/or  {hint}")
    return need


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str))


def append_jsonl(p, obj):
    with open(p, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def make_batch(stream, N, B, T, pos):
    """B readers spaced evenly through the reference corpus (length N), each at offset `pos` within its shard.
    A stream of different length L is read at the same *fraction*, so corrupted and clean streams stay aligned
    on the same documents to within a document or so."""
    L = len(stream)
    stride = N // B
    starts = (((np.arange(B, dtype=np.int64) * stride + pos) % N) * (L / N)).astype(np.int64)
    idx = (starts[:, None] + np.arange(T + 1)[None, :]) % L
    return np.asarray(stream[idx], dtype=np.int32)


def train(cfg, run_dir, resume=True):
    if cfg.get("task") == "vision":                      # positive-control task shares queue/status/schedule code
        from .vision.train import train as vision_train
        return vision_train(cfg, run_dir, resume)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_p, curve_p, metrics_p = run_dir / "status.json", run_dir / "train_curve.jsonl", run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "ckpt"
    paths = required_streams(cfg)

    d, tr, co, ev = cfg["data"], cfg["train"], cfg["corruption"], cfg["eval"]
    T, vocab = d["seq_len"], d["tokenizer_vocab"]
    B = tr["batch_tokens"] // T
    total_steps = tr["tokens"] // (B * T)
    kind, param, window = co["kind"], co["param"], co["window"]
    if window is not None and kind == "clean":
        raise SystemExit("corruption.window set but kind is clean")
    if kind != "clean" and window is None:
        window = [0.0, 1.0]

    mx.random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    tok = load_tokenizer(stream_dir(cfg))
    clean = open_stream(paths["train_clean"])
    corrupt = open_stream(paths["train_corrupt"]) if "train_corrupt" in paths else None
    val = open_stream(paths["val"])
    N = len(clean)

    model = build_model(cfg, vocab)
    optimizer = optim.AdamW(learning_rate=tr["lr"], weight_decay=tr["weight_decay"])
    mask_full = causal_mask(T)
    mask_short = causal_mask(T, int(param or 32)) if kind == "short_context" else mask_full
    p_label = float(param if param is not None else 0.3) if kind == "label_noise" else 0.0

    def loss_fn(model, x, y, mask):
        return nn.losses.cross_entropy(model(x, mask), y, reduction="mean")

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(x, y, mask):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y, mask)
        grads, _ = optim.clip_grad_norm(grads, tr["grad_clip"])
        optimizer.update(model, grads)
        return loss

    # ---- resume / skip ----
    step, pos = 0, 0
    if resume and status_p.exists() and json.loads(status_p.read_text()).get("status") == "done":
        print(f"{run_dir.name}: already done, skipping (use --no-resume to retrain)", flush=True)
        lines = metrics_p.read_text().splitlines() if metrics_p.exists() else []
        return json.loads(lines[-1]) if lines else None
    if resume and (ckpt_dir / "state.json").exists():
        st = json.loads((ckpt_dir / "state.json").read_text())
        model.load_weights(str(ckpt_dir / "model.safetensors"))
        optimizer.state = tree_unflatten(list(mx.load(str(ckpt_dir / "optimizer.safetensors")).items()))
        step, pos = st["step"], st["pos"]
        mx.eval(model.parameters(), optimizer.state)
        print(f"resumed from step {step}", flush=True)
    else:
        for p in (curve_p, metrics_p):
            p.unlink(missing_ok=True)

    info = {"n_params": n_params(model), "batch_size": B, "total_steps": total_steps, "ref_tokens": N,
            "host": socket.gethostname(), "platform": platform.platform(), "mlx": mx.__version__,
            "lmcp": __version__, **git_info()}
    write_json(run_dir / "config.json", {"config": cfg, "info": info})
    started = time.time()

    def set_status(s, **kw):
        write_json(status_p, {"status": s, "step": step, "tokens": step * B * T, "total_steps": total_steps,
                              "started": started, "updated": time.time(), **kw})

    def save_ckpt():
        ckpt_dir.mkdir(exist_ok=True)
        mx.save_safetensors(str(ckpt_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
        mx.save_safetensors(str(ckpt_dir / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
        write_json(ckpt_dir / "state.json", {"step": step, "pos": pos})

    def do_eval(final=False):
        t0 = time.time()
        m = evals.run_all(model, tok, val, cfg, final=final)
        m.update({"step": step, "tokens": step * B * T, "frac": step / total_steps, "final": final,
                  "eval_seconds": round(time.time() - t0, 1)})
        append_jsonl(metrics_p, m)
        print(f"  eval step {step}: val {m['val_loss']:.3f}  blimp {m['blimp_acc']:.3f}  "
              f"induction_acc {m['induction_acc']:.3f}  long_range_gain {m['long_range_gain']:.3f}  ({m['eval_seconds']}s)", flush=True)
        return m

    set_status("running")
    print(f"{run_dir.name}\n  params {info['n_params']/1e6:.2f}M  steps {total_steps}  batch {B}x{T}  "
          f"kind {kind} param {param} window {window}  schedule {tr['schedule']}", flush=True)
    eval_every = max(1, ev["every_tokens"] // (B * T))
    ckpt_every = max(1, cfg["checkpoint"]["every_tokens"] // (B * T)) if cfg["checkpoint"]["every_tokens"] else None
    if step == 0:
        do_eval()
    losses, t_log, tok_log = [], time.time(), 0
    try:
        while step < total_steps:
            frac = step / total_steps
            active = kind != "clean" and in_window(frac, window)
            phase = "corrupt" if active else "clean"
            src = corrupt if (active and corrupt is not None) else clean
            batch = make_batch(src, N, B, T, pos)
            x, y = batch[:, :-1], batch[:, 1:]
            if active and p_label > 0:
                rng = np.random.default_rng([tr["seed"], step])
                noise = rng.random(y.shape) < p_label
                y = np.where(noise, rng.integers(2, vocab, size=y.shape), y)
            mask = mask_short if (active and kind == "short_context") else mask_full
            lr = lr_at(step, total_steps, tr)
            optimizer.learning_rate = lr
            loss = step_fn(mx.array(x), mx.array(y), mask)
            mx.eval(state, loss)
            losses.append(loss)
            step += 1
            pos += T
            tok_log += B * T
            if step % tr["log_every"] == 0 or step == total_steps:
                mx.eval(losses)
                dt = time.time() - t_log
                rec = {"step": step, "tokens": step * B * T, "frac": round(step / total_steps, 4), "phase": phase,
                       "loss": float(mx.mean(mx.stack(losses)).item()), "lr": lr, "tok_s": round(tok_log / dt)}
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
        set_status("failed" if not isinstance(e, KeyboardInterrupt) else "interrupted", error=repr(e))
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise

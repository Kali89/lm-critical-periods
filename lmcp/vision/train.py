"""Vision positive control trainer (CIFAR-10 / ResNet-18). Same run-dir artefacts as lmcp/train.py.

Config:  task: vision;  vision: {epochs, batch_size, subset, test_every_epochs, lr_decay, weight_decay}
         train.lr / train.schedule (const|cosine|wsd via lmcp.schedule, or "exp" = lr * vision.lr_decay**epoch)
         train.seed;  corruption: {kind: clean|blur|label_permute|vflip, window: [start_frac, end_frac]}
Note: weight decay comes from vision.weight_decay (default 5e-4, the paper's ResNet value), NOT train.weight_decay,
whose base.yaml default (0.1) is an AdamW decoupled value that would be destructive as SGD L2.
"""
import json
import platform
import socket
import time
import traceback
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .. import __version__
from ..schedule import in_window, lr_at
from ..train import append_jsonl, git_info, write_json
from . import data as D
from .resnet import ResNet18

DEFAULTS = {"epochs": 160, "batch_size": 128, "subset": None, "test_every_epochs": 1,
            "lr_decay": 0.97, "weight_decay": 5e-4}


def train(cfg, run_dir, resume=True):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_p, curve_p, metrics_p = run_dir / "status.json", run_dir / "train_curve.jsonl", run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "ckpt"
    vc = {**DEFAULTS, **(cfg.get("vision") or {})}
    tr, co = cfg["train"], cfg["corruption"]
    kind, window = co["kind"], co.get("window")
    if kind not in D.KINDS:
        raise SystemExit(f"vision task: unknown corruption kind {kind}; expected one of {sorted(D.KINDS)}")
    if window is not None and kind == "clean":
        raise SystemExit("corruption.window set but kind is clean")
    if kind != "clean" and window is None:
        window = [0.0, 1.0]
    seed, B = tr["seed"], vc["batch_size"]

    if resume and status_p.exists() and json.loads(status_p.read_text()).get("status") == "done":
        print(f"{run_dir.name}: already done, skipping", flush=True)
        lines = metrics_p.read_text().splitlines() if metrics_p.exists() else []
        return json.loads(lines[-1]) if lines else None

    # ---- data: clean and (if needed) corrupted copies of the training images, held as uint8 ----
    tx, ty, ex, ey = D.load_cifar()
    n_train = vc["subset"] or len(tx)
    sub = np.random.default_rng([seed, 7]).permutation(len(tx))[:n_train]
    tx, ty = tx[sub], ty[sub]
    ty_c, tx_c = ty, tx
    if kind == "blur":
        tx_c = D.blur(tx)
    elif kind == "vflip":
        tx_c = np.ascontiguousarray(D.vflip(tx))
    elif kind == "label_permute":
        ty_c = D.label_perm(seed)[ty]
    ex_n, ey_mx = mx.array(D.normalise(ex)), mx.array(ey)

    spe = n_train // B
    epochs = vc["epochs"]
    total_steps = epochs * spe

    mx.random.seed(seed)
    model = ResNet18()
    optimizer = optim.SGD(learning_rate=tr["lr"], momentum=0.9, weight_decay=vc["weight_decay"])

    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x), y, reduction="mean")

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        optimizer.update(model, grads)
        return loss

    def lr_for(step):
        if tr["schedule"] == "exp":
            return tr["lr"] * vc["lr_decay"] ** (step // spe)
        return lr_at(step, total_steps, tr)

    step = 0
    if resume and (ckpt_dir / "state.json").exists():
        st = json.loads((ckpt_dir / "state.json").read_text())
        model.load_weights(str(ckpt_dir / "model.safetensors"))
        optimizer.state = tree_unflatten(list(mx.load(str(ckpt_dir / "optimizer.safetensors")).items()))
        step = st["step"]
        mx.eval(model.parameters(), optimizer.state)
        print(f"resumed from step {step}", flush=True)
    else:
        for p in (curve_p, metrics_p):
            p.unlink(missing_ok=True)

    n_p = sum(v.size for _, v in tree_flatten(model.parameters()))
    info = {"n_params": n_p, "batch_size": B, "total_steps": total_steps, "steps_per_epoch": spe, "n_train": n_train,
            "host": socket.gethostname(), "platform": platform.platform(), "mlx": mx.__version__,
            "lmcp": __version__, **git_info()}
    write_json(run_dir / "config.json", {"config": cfg, "info": info})
    started = time.time()

    def set_status(s, **kw):
        write_json(status_p, {"status": s, "step": step, "total_steps": total_steps, "started": started,
                              "updated": time.time(), "elapsed_min": round((time.time() - started) / 60, 2), **kw})

    def save_ckpt():
        ckpt_dir.mkdir(exist_ok=True)
        mx.save_safetensors(str(ckpt_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
        mx.save_safetensors(str(ckpt_dir / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
        write_json(ckpt_dir / "state.json", {"step": step})

    def evaluate(final=False, train_loss=None):
        t0 = time.time()
        model.eval()
        correct, tot_loss = 0, 0.0
        for i in range(0, len(ex_n), 1000):
            lg = model(ex_n[i:i + 1000])
            yb = ey_mx[i:i + 1000]
            correct += mx.sum(mx.argmax(lg, axis=1) == yb).item()
            tot_loss += mx.sum(nn.losses.cross_entropy(lg, yb, reduction="none")).item()
        model.train()
        frac = step / total_steps
        m = {"step": step, "frac": round(frac, 4), "phase": "corrupt" if kind != "clean" and in_window(frac, window) else "clean",
             "final": final, "test_acc": correct / len(ex_n), "test_loss": tot_loss / len(ex_n),
             "train_loss": train_loss, "eval_seconds": round(time.time() - t0, 1)}
        append_jsonl(metrics_p, m)
        print(f"  eval step {step}: test_acc {m['test_acc']:.4f}  test_loss {m['test_loss']:.3f}  ({m['eval_seconds']}s)", flush=True)
        return m

    set_status("running")
    print(f"{run_dir.name}\n  params {n_p/1e6:.2f}M  steps {total_steps} ({epochs} ep x {spe})  batch {B}  "
          f"kind {kind} window {window}  schedule {tr['schedule']}  lr {tr['lr']}", flush=True)
    model.train()
    log_every = tr.get("log_every", 20)
    losses, t_log, img_log, ep_losses = [], time.time(), 0, []
    final = None
    try:
        for ep in range(step // spe, epochs):
            order = np.random.default_rng([seed, ep]).permutation(n_train)
            aug = np.random.default_rng([seed, ep, 1])
            for bi in range(spe):
                frac = step / total_steps
                active = kind != "clean" and in_window(frac, window)
                phase = "corrupt" if active else "clean"
                idx = np.sort(order[bi * B:(bi + 1) * B])
                xb = D.augment((tx_c if active else tx)[idx], aug)
                yb = (ty_c if active else ty)[idx]
                lr = lr_for(step)
                optimizer.learning_rate = lr
                loss = step_fn(mx.array(D.normalise(xb)), mx.array(yb))
                mx.eval(state, loss)
                losses.append(loss)
                step += 1
                img_log += B
                if step % log_every == 0 or step == total_steps:
                    mx.eval(losses)
                    dt = time.time() - t_log
                    rec = {"step": step, "images": step * B, "frac": round(step / total_steps, 4), "phase": phase,
                           "loss": float(mx.mean(mx.stack(losses)).item()), "lr": lr, "img_s": round(img_log / dt)}
                    append_jsonl(curve_p, rec)
                    ep_losses.append(rec["loss"])
                    eta = (total_steps - step) * dt / max(1, len(losses)) / 60
                    print(f"step {step}/{total_steps} ep {ep} {phase:7s} loss {rec['loss']:.3f} lr {lr:.2e} "
                          f"{rec['img_s']} img/s  eta {eta:.0f}m", flush=True)
                    losses, t_log, img_log = [], time.time(), 0
            tl = float(np.mean(ep_losses)) if ep_losses else None
            ep_losses = []
            last = ep == epochs - 1
            if last or (ep + 1) % vc["test_every_epochs"] == 0:
                final = evaluate(final=last, train_loss=tl)
            if not last:
                save_ckpt()
                set_status("running")
        if cfg["checkpoint"].get("final", True):
            (run_dir / "final").mkdir(exist_ok=True)
            mx.save_safetensors(str(run_dir / "final" / "model.safetensors"), dict(tree_flatten(model.parameters())))
        set_status("done", finished=time.time())
        return final
    except BaseException as e:
        save_ckpt()
        set_status("failed" if not isinstance(e, KeyboardInterrupt) else "interrupted", error=repr(e))
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise

"""Grids, the run queue, result collection and reporting."""
import itertools
import json
import os
import shutil
import time
from pathlib import Path

from . import QUEUE, RESULTS, RUNS
from .config import load_config, run_name, set_dotted, load_yaml

PENDING, RUNNING, DONE, FAILED = (QUEUE / s for s in ("pending", "running", "done", "failed"))


def expand_grid(grid_path):
    g = load_yaml(grid_path)
    name = g.get("name") or Path(grid_path).stem
    base_over = g.get("overrides", {})
    axes = g.get("axes", {})
    keys = list(axes)
    runs = []
    combos = itertools.product(*[axes[k] for k in keys]) if keys else ([()] if not g.get("extra_runs") else [])
    for combo in combos:
        over = dict(base_over)
        for k, v in zip(keys, combo):
            over[k] = v
        runs.append(over)
    for extra in g.get("extra_runs", []):
        runs.append({**base_over, **extra})
    out = []
    for over in runs:
        cfg = load_config(g.get("base"), over)
        cfg["run"]["grid"] = name
        out.append((run_name(cfg, name), cfg))
    return name, out


def enqueue(grid_path, dry=False):
    name, runs = expand_grid(grid_path)
    for d in (PENDING, RUNNING, DONE, FAILED):
        d.mkdir(parents=True, exist_ok=True)
    added, skipped = [], []
    for rn, cfg in runs:
        status = run_status(rn)
        if status == "done" or any((d / f"{rn}.json").exists() for d in (PENDING, RUNNING)):
            skipped.append((rn, status or "queued"))
            continue
        if not dry:
            (PENDING / f"{rn}.json").write_text(json.dumps({"name": rn, "grid": name, "config": cfg}, indent=2))
        added.append(rn)
    return name, added, skipped


def run_status(rn):
    p = RUNS / rn / "status.json"
    return json.loads(p.read_text())["status"] if p.exists() else None


def queue_status():
    out = {}
    for d in (PENDING, RUNNING, DONE, FAILED):
        d.mkdir(parents=True, exist_ok=True)
        out[d.name] = sorted(p.stem for p in d.glob("*.json"))
    return out


def next_run():
    """Claim the first pending job, train it, file it under done/ or failed/. Returns (name, ok) or None."""
    from .train import train
    pending = sorted(PENDING.glob("*.json"))
    if not pending:
        return None
    job = pending[0]
    claimed = RUNNING / job.name
    job.rename(claimed)
    spec = json.loads(claimed.read_text())
    rn = spec["name"]
    ok = False
    try:
        train(spec["config"], RUNS / rn, resume=True)
        ok = True
    except KeyboardInterrupt:
        claimed.rename(PENDING / claimed.name)   # interrupted, not failed: it resumes from its checkpoint next time
        raise
    finally:
        if claimed.exists():
            claimed.rename((DONE if ok else FAILED) / claimed.name)
    return rn, ok


def requeue_failed():
    n = 0
    for p in FAILED.glob("*.json"):
        p.rename(PENDING / p.name); n += 1
    return n


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, kk + "."))
        else:
            out[kk] = v
    return out


def collect():
    RESULTS.mkdir(exist_ok=True)
    rows, curves = [], []
    for rd in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if rd.name.startswith("_"):      # scratch runs (engineer checks, throughput tests) are not results
            continue
        sp, cp, mp = rd / "status.json", rd / "config.json", rd / "metrics.jsonl"
        if not (sp.exists() and cp.exists() and mp.exists()):
            continue
        status = json.loads(sp.read_text())
        conf = json.loads(cp.read_text())
        metrics = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
        if not metrics:
            continue
        for m in metrics:
            curves.append({"run": rd.name, **{k: v for k, v in m.items() if k != "blimp_paradigms"}})
        final = metrics[-1]
        row = {"run": rd.name, "status": status["status"], "elapsed_min": status.get("elapsed_min"),
               **flatten(conf["config"]), "n_params": conf["info"]["n_params"], "git_rev": conf["info"].get("git_rev"),
               **{k: v for k, v in final.items() if k != "blimp_paradigms"}}
        rows.append(row)
    with open(RESULTS / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    with open(RESULTS / "curves.jsonl", "w") as f:
        for r in curves:
            f.write(json.dumps(r, default=str) + "\n")
    return rows


def report(metrics=("val_loss", "blimp_acc", "induction_acc", "long_range_gain")):
    import pandas as pd
    rows = collect()
    if not rows:
        return "no finished runs"
    df = pd.DataFrame(rows)
    df = df[df.status == "done"]
    if df.empty:
        return "no finished runs"
    df["window"] = df["corruption.window"].apply(lambda w: f"{w[0]:.2f}-{w[1]:.2f}" if isinstance(w, list) else "-")
    df["kind"] = df["corruption.kind"] + df["corruption.param"].apply(lambda p: "" if p is None or p != p else f"-{p}")
    keys = ["run.grid", "model.size", "train.schedule", "train.tokens", "kind", "window"]
    agg = df.groupby(keys)[list(metrics)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    # delta vs the clean baseline sharing grid/size/schedule/tokens
    base = df[df.kind == "clean"].groupby(keys[:4])[list(metrics)].mean()
    for m in metrics:
        agg[f"{m}_vs_clean"] = [
            row[f"{m}_mean"] - base.loc[idx[:4], m] if idx[:4] in base.index else float("nan")
            for idx, row in agg.iterrows()
        ]
    pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
    return agg.reset_index().to_string(index=False)

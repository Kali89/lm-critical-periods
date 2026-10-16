"""CLI entry point. Use bin/lmcp <command> from the repo root."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import DATA, QUEUE, ROOT, RUNS
from .config import TEXT_KINDS, load_config, parse_value, run_name, stream_dir


def parse_sets(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        out[k] = parse_value(v)
    return out


def cmd_doctor(a):
    import mlx.core as mx
    ok = True

    def line(flag, msg):
        nonlocal ok
        ok &= flag
        print(("  ok   " if flag else "  FAIL ") + msg)

    print("environment")
    line(mx.metal.is_available(), f"mlx {mx.__version__}, metal available")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    line(free_gb > 20, f"{free_gb:.0f} GB free disk")
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    line("AC Power" in batt, "on AC power" if "AC Power" in batt else "on battery (run-queue.sh will refuse to start)")
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    lim = [l for l in therm.splitlines() if "CPU_Speed_Limit" in l]
    line(not lim or "100" in lim[0], ("thermal: " + lim[0].strip()) if lim else "thermal: no limit reported")
    print("data")
    line(len(list((DATA / "blimp").glob("*.jsonl"))) == 67 if (DATA / "blimp").exists() else False, "BLiMP: 67 paradigms")
    for c in sorted((DATA / "corpora").glob("*")) if (DATA / "corpora").exists() else []:
        m = json.loads((c / "meta.json").read_text()) if (c / "meta.json").exists() else {}
        line(bool(m), f"corpus {c.name}: {m.get('train', {}).get('docs')} train docs, {m.get('val', {}).get('docs')} val docs")
    for t in sorted((DATA / "tok").glob("*")) if (DATA / "tok").exists() else []:
        streams = sorted(p.name[:-4] for p in t.glob("*.bin"))
        line((t / "tokenizer.json").exists() and "train.clean" in streams and "val.clean" in streams,
             f"tokenizer {t.name}: streams {streams}")
    print("queue")
    from .runs import queue_status
    q = queue_status()
    print("       " + "  ".join(f"{k}={len(v)}" for k, v in q.items()))
    lock = QUEUE / ".lock"
    if lock.exists():
        print(f"       run-queue lock present: {lock.read_text().strip()}")
    rev = subprocess.run(["git", "status", "--porcelain", "lmcp", "configs"], cwd=ROOT, capture_output=True, text=True).stdout
    line(not rev.strip(), "harness code committed (runs record the git rev)" if not rev.strip() else "uncommitted changes in lmcp/ or configs/ (runs will be marked dirty)")
    return 0 if ok else 1


def cmd_data(a):
    from . import data
    if a.what == "download":
        print(data.download(a.corpus, a.variant))
        print(data.download_blimp())
    elif a.what == "prepare":
        print(data.prepare(a.corpus, a.variant, a.max_train_docs))


def cmd_tok(a):
    from . import tok
    cd = DATA / "corpora" / f"{a.corpus}_{a.variant}"
    out = DATA / "tok" / f"{a.corpus}_{a.variant}_v{a.vocab}"
    if a.what == "train":
        tk = tok.train_tokenizer(cd / "train.docs.jsonl", out, a.vocab)
        print(f"trained vocab {tk.get_vocab_size()} -> {out}")
    elif a.what == "encode":
        tk = tok.load_tokenizer(out)
        print(tok.encode_stream(tk, cd / f"{a.split}.docs.jsonl", out, split=a.split, kind=a.kind, param=a.param))


def cmd_prepare_all(a):
    from . import data, tok
    data.download(a.corpus, a.variant); data.download_blimp()
    data.prepare(a.corpus, a.variant)
    cd = DATA / "corpora" / f"{a.corpus}_{a.variant}"
    out = DATA / "tok" / f"{a.corpus}_{a.variant}_v{a.vocab}"
    tk = tok.train_tokenizer(cd / "train.docs.jsonl", out, a.vocab)
    print(tok.encode_stream(tk, cd / "val.docs.jsonl", out, split="val"))
    print(tok.encode_stream(tk, cd / "train.docs.jsonl", out, split="train"))


def cmd_streams(a):
    for t in sorted((DATA / "tok").glob("*")):
        print(t.name)
        for m in sorted(t.glob("*.meta.json")):
            j = json.loads(m.read_text())
            print(f"  {j['split']}.{j['stream']:28s} {j['n_tokens']/1e6:8.1f}M tokens  {j['n_docs']} docs")


def cmd_run(a):
    from .train import train
    cfg = load_config(a.config, parse_sets(a.set))
    name = a.name or run_name(cfg)
    print(json.dumps(cfg, indent=1) if a.show else "")
    if a.dry:
        print(name); return
    train(cfg, RUNS / name, resume=not a.no_resume)


def cmd_grid(a):
    from .runs import enqueue
    name, added, skipped = enqueue(a.grid, dry=a.dry)
    print(f"grid {name}: {'would add' if a.dry else 'added'} {len(added)}, skipped {len(skipped)}")
    for r in added:
        print("  +", r)
    for r, s in skipped:
        print(f"  = {r}  ({s})")


def cmd_queue(a):
    from .runs import queue_status, next_run, requeue_failed, PENDING
    if a.what == "status":
        for k, v in queue_status().items():
            print(f"{k} ({len(v)})")
            for r in v:
                print("   ", r)
    elif a.what == "next":
        r = next_run()
        if r is None:
            print("queue empty"); sys.exit(3)
        print(("done " if r[1] else "FAILED ") + r[0])
    elif a.what == "requeue-failed":
        print(f"requeued {requeue_failed()}")
    elif a.what == "clear":
        n = 0
        for p in PENDING.glob("*.json"):
            p.unlink(); n += 1
        print(f"removed {n} pending jobs")


def cmd_collect(a):
    from .runs import collect
    rows = collect()
    print(f"collected {len(rows)} runs -> results/results.jsonl, results/curves.jsonl")


def cmd_report(a):
    from .runs import report
    print(report(tuple(a.metric) if a.metric else ("val_loss", "blimp_acc", "induction_acc", "long_range_gain")))


def main():
    p = argparse.ArgumentParser(prog="lmcp")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor").set_defaults(f=cmd_doctor)
    d = sp.add_parser("data"); d.add_argument("what", choices=["download", "prepare"]); d.add_argument("--corpus", default="babylm100m"); d.add_argument("--variant", default="clean"); d.add_argument("--max-train-docs", type=int); d.set_defaults(f=cmd_data)
    t = sp.add_parser("tok"); t.add_argument("what", choices=["train", "encode"]); t.add_argument("--corpus", default="babylm100m"); t.add_argument("--variant", default="clean"); t.add_argument("--vocab", type=int, default=8192); t.add_argument("--split", default="train"); t.add_argument("--kind", default="clean"); t.add_argument("--param", type=parse_value); t.set_defaults(f=cmd_tok)
    pa = sp.add_parser("prepare-all"); pa.add_argument("--corpus", default="babylm100m"); pa.add_argument("--variant", default="clean"); pa.add_argument("--vocab", type=int, default=8192); pa.set_defaults(f=cmd_prepare_all)
    sp.add_parser("streams").set_defaults(f=cmd_streams)
    r = sp.add_parser("run"); r.add_argument("config", nargs="?"); r.add_argument("--set", action="append", metavar="key=value"); r.add_argument("--name"); r.add_argument("--dry", action="store_true"); r.add_argument("--show", action="store_true"); r.add_argument("--no-resume", action="store_true"); r.set_defaults(f=cmd_run)
    g = sp.add_parser("grid"); g.add_argument("grid"); g.add_argument("--dry", action="store_true"); g.set_defaults(f=cmd_grid)
    q = sp.add_parser("queue"); q.add_argument("what", choices=["status", "next", "requeue-failed", "clear"]); q.set_defaults(f=cmd_queue)
    sp.add_parser("collect").set_defaults(f=cmd_collect)
    rp = sp.add_parser("report"); rp.add_argument("--metric", action="append"); rp.set_defaults(f=cmd_report)
    a = p.parse_args()
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()

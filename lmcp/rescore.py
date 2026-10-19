"""Re-evaluate saved checkpoints with the current evals (e.g. to add metrics introduced after a run finished).

Results go to RUN_DIR/rescore.jsonl (one JSON line per rescoring); metrics.jsonl is never touched. `collect()`
merges the last `which == "final"` line into results.jsonl, adding only keys the training-time final eval lacks.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from . import evals
from .model import build_model
from .tok import load_tokenizer, open_stream
from .train import append_jsonl, git_info, required_streams


def _has_line(path, which):
    if not path.exists():
        return False
    return any(json.loads(l).get("which") == which for l in path.read_text().splitlines() if l.strip())


def rescore_run(run_dir, which="final", force=False):
    """Returns the appended record, or a string explaining why the run dir was skipped."""
    import mlx.core as mx
    run_dir = Path(run_dir)
    sp, cp = run_dir / "status.json", run_dir / "config.json"
    if not (sp.exists() and cp.exists()):
        return "no status.json/config.json"
    status = json.loads(sp.read_text()).get("status")
    if status != "done":
        return f"status is {status!r}, not 'done'"
    cfg = json.loads(cp.read_text())["config"]
    if cfg.get("task") == "vision":
        return "vision task"
    out_p = run_dir / "rescore.jsonl"
    if not force and _has_line(out_p, which):
        return f"already rescored ({which}); use --force"
    if which == "final":
        weights, tokens = run_dir / "final" / "model.safetensors", None
    else:
        weights = run_dir / "ckpt" / "model.safetensors"
        st = json.loads((run_dir / "ckpt" / "state.json").read_text()) if (run_dir / "ckpt" / "state.json").exists() else None
        if st is None:
            return "no ckpt/state.json"
        tokens = int(st["step"]) * (cfg["train"]["batch_tokens"] // cfg["data"]["seq_len"]) * cfg["data"]["seq_len"]
    if not weights.exists():
        return f"missing {weights.relative_to(run_dir)}"
    sd, _, _, val_names = required_streams(cfg)
    tok = load_tokenizer(sd)
    vals = {n: open_stream(sd / f"val.{n}.bin") for n in val_names}
    model = build_model(cfg, cfg["data"]["tokenizer_vocab"])
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    m = evals.run_all(model, tok, vals, cfg, final=True)
    if tokens is None:
        tokens = int(json.loads(sp.read_text()).get("tokens", 0))
    rec = {**m, "which": which, "tokens": tokens, "rescored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rev": git_info()}
    append_jsonl(out_p, rec)
    return rec


def summary_line(name, rec):
    f = lambda k: f"{rec[k]:.4f}" if k in rec else "n/a"
    return (f"{name}: val_loss {f('val_loss')}  val_loss_ctx_32 {f('val_loss_ctx_32')}  "
            f"context_gain {f('context_gain')}  blimp_acc {f('blimp_acc')}")

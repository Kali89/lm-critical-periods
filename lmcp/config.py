"""Config loading, dotted overrides, run naming."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

from . import ROOT

SIZES = {
    "xs": dict(n_layer=2, d_model=128, n_head=4),   # ~1.4M params, smoke tests
    "s":  dict(n_layer=4, d_model=256, n_head=4),   # ~5M
    "m":  dict(n_layer=6, d_model=384, n_head=6),   # ~14M, pilot workhorse
    "l":  dict(n_layer=8, d_model=512, n_head=8),   # ~29M
}

TEXT_KINDS = {"clean", "shuffle_sentence", "shuffle_doc_sentences", "shuffle_window", "char_noise"}
TRAIN_KINDS = {"short_context", "label_noise", "vocab_permute"}


def load_yaml(p):
    return yaml.safe_load(Path(p).read_text()) or {}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def parse_value(s):
    """Parse a CLI override value: yaml-ish scalars and lists."""
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def set_dotted(d, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    if isinstance(value, dict) and isinstance(d.get(parts[-1]), dict):
        deep_update(d[parts[-1]], value)
    else:
        d[parts[-1]] = value


def load_config(path=None, overrides=None):
    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    if path:
        deep_update(cfg, load_yaml(path))
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    resolve_model(cfg)
    return cfg


def resolve_model(cfg):
    m = cfg["model"]
    base = SIZES[m.get("size", "s")]
    for k, v in base.items():
        if m.get(k) is None:
            m[k] = v
    return cfg


def config_hash(cfg):
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:6]


def run_name(cfg, grid=None):
    c = cfg["corruption"]
    kind = c["kind"]
    if c.get("param") is not None:
        kind += f"-{c['param']}"
    if c.get("intensity") is not None and float(c["intensity"]) != 1.0 and kind != "clean":
        kind += f"@{float(c['intensity']):g}"
    if cfg.get("train", {}).get("phases"):
        kind += "+phases"
    w = c.get("window")
    win = f"w{w[0]:.2f}-{w[1]:.2f}" if w else "wnone"
    parts = [grid] if grid else []
    if cfg.get("run", {}).get("tag"):
        parts.append(str(cfg["run"]["tag"]))
    parts += [kind, win, cfg["model"]["size"], cfg["train"]["schedule"], f"s{cfg['train']['seed']}", config_hash(cfg)]
    return "__".join(parts)


def tokenizer_name(cfg):
    d = cfg["data"]
    return d.get("tokenizer") or f"{d['corpus']}_{d['variant']}_v{d['tokenizer_vocab']}"


def stream_dir(cfg):
    return ROOT / "data" / "tok" / tokenizer_name(cfg)


def corpus_dir(cfg_or_corpus, variant=None):
    if isinstance(cfg_or_corpus, dict):
        d = cfg_or_corpus["data"]
        return ROOT / "data" / "corpora" / f"{d['corpus']}_{d['variant']}"
    return ROOT / "data" / "corpora" / f"{cfg_or_corpus}_{variant}"

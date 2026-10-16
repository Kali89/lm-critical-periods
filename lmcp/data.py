"""Download corpora + BLiMP, and normalise corpora into shuffled document files.

Output layout:
  data/raw/<corpus>/...                      downloaded files
  data/blimp/<paradigm>.jsonl                67 BLiMP paradigms
  data/corpora/<corpus>_<variant>/train.docs.jsonl   one JSON {"i","src","text"} per document, globally shuffled (seed 1234)
  data/corpora/<corpus>_<variant>/val.docs.jsonl
"""
import io
import json
import random
import re
import subprocess
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import DATA

BABYLM_REPO = "cambridge-climb/BabyLM"
BABYLM_SOURCES = ["aochildes", "bnc_spoken", "cbt", "children_stories", "gutenberg",
                  "open_subtitles", "qed", "simple_wikipedia", "switchboard", "wikipedia"]
TINY_REPO = "roneneldan/TinyStories"
BLIMP_ZIP = "https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip"
DOC_SHUFFLE_SEED = 1234
DOC_TARGET_CHARS = 2000   # documents are normalised to roughly this size (merge tiny lines, split huge blocks)


def download(corpus, variant="clean"):
    raw = DATA / "raw" / corpus
    raw.mkdir(parents=True, exist_ok=True)
    if corpus.startswith("babylm"):
        size = "100M" if corpus == "babylm100m" else "10M"
        for split in (size, "dev"):
            for src in BABYLM_SOURCES:
                hf_hub_download(BABYLM_REPO, f"{variant}/{split}/{src}.txt", repo_type="dataset", local_dir=raw)
    elif corpus == "tinystories":
        for f in ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"):
            hf_hub_download(TINY_REPO, f, repo_type="dataset", local_dir=raw)
    else:
        raise SystemExit(f"unknown corpus {corpus}")
    return raw


def download_blimp():
    out = DATA / "blimp"
    if out.exists() and len(list(out.glob("*.jsonl"))) == 67:
        return out
    out.mkdir(parents=True, exist_ok=True)
    buf = subprocess.run(["curl", "-sSL", "--max-time", "120", BLIMP_ZIP], check=True, capture_output=True).stdout
    with zipfile.ZipFile(io.BytesIO(buf)) as z:
        for name in z.namelist():
            if re.search(r"/data/[^/]+\.jsonl$", name):
                (out / Path(name).name).write_bytes(z.read(name))
    n = len(list(out.glob("*.jsonl")))
    assert n == 67, f"expected 67 BLiMP paradigms, got {n}"
    return out


def chunk_lines(lines, target=DOC_TARGET_CHARS):
    """Greedily merge consecutive non-blank lines into documents of ~target chars.
    Blank lines are soft boundaries: a block is flushed when it has reached target/2 chars."""
    docs, buf, n = [], [], 0
    for line in lines:
        s = line.rstrip("\n")
        if not s.strip():
            if n >= target // 2:
                docs.append("\n".join(buf)); buf, n = [], 0
            continue
        buf.append(s); n += len(s) + 1
        if n >= target:
            docs.append("\n".join(buf)); buf, n = [], 0
    if buf:
        docs.append("\n".join(buf))
    return docs


def iter_raw_docs(corpus, variant, split):
    raw = DATA / "raw" / corpus
    if corpus.startswith("babylm"):
        size = "100M" if corpus == "babylm100m" else "10M"
        sub = size if split == "train" else "dev"
        for src in BABYLM_SOURCES:
            with open(raw / variant / sub / f"{src}.txt", encoding="utf-8", errors="replace") as f:
                for d in chunk_lines(f):
                    yield src, d
    elif corpus == "tinystories":
        fn = "TinyStoriesV2-GPT4-train.txt" if split == "train" else "TinyStoriesV2-GPT4-valid.txt"
        buf = []
        with open(raw / fn, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("<|endoftext|>"):
                    if buf:
                        yield "tinystories", "".join(buf).strip()
                    buf = []
                else:
                    buf.append(line)
        if buf:
            yield "tinystories", "".join(buf).strip()


def prepare(corpus, variant="clean", max_train_docs=None):
    """Write shuffled train/val document files. Shuffling once here fixes the document order
    for every corrupted stream, which is what keeps the streams aligned during training."""
    out = DATA / "corpora" / f"{corpus}_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        docs = [(src, d) for src, d in iter_raw_docs(corpus, variant, split) if d.strip()]
        random.Random(DOC_SHUFFLE_SEED).shuffle(docs)
        if split == "train" and max_train_docs:
            docs = docs[:max_train_docs]
        chars = 0
        with open(out / f"{split}.docs.jsonl", "w", encoding="utf-8") as f:
            for i, (src, d) in enumerate(docs):
                chars += len(d)
                f.write(json.dumps({"i": i, "src": src, "text": d}, ensure_ascii=False) + "\n")
        stats[split] = {"docs": len(docs), "chars": chars}
    (out / "meta.json").write_text(json.dumps({"corpus": corpus, "variant": variant, **stats}, indent=2))
    return out, stats


def iter_docs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

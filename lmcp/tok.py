"""Train a byte-level BPE tokenizer and encode document files into flat uint16 token streams with a document index.

Layout: data/tok/<name>/tokenizer.json
                       /train.<stream>.bin        uint16 tokens, eos after every document
                       /train.<stream>.idx.npy    int64 document start offsets (n_docs + 1 entries)
                       /val.<stream>.bin, .idx.npy
                       /<split>.<stream>.meta.json
The document index is what lets the trainer read documents in a per-seed order and switch between a clean
and a corrupted stream on the same document.
"""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .corrupt import corrupt_doc, stream_name
from .data import iter_docs

EOS, PAD = "<|eos|>", "<|pad|>"
EOS_ID, PAD_ID = 0, 1


def train_tokenizer(docs_paths, out_dir, vocab, max_chars=300_000_000):
    """docs_paths: one path or a list (multilingual tokenizers are trained on several corpora, chars split evenly)."""
    paths = [Path(p) for p in (docs_paths if isinstance(docs_paths, (list, tuple)) else [docs_paths])]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOS, PAD],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    budget = max_chars // len(paths)

    def it():
        for p in paths:
            n = 0
            for d in iter_docs(p):
                n += len(d["text"])
                if n > budget:
                    break
                yield d["text"]

    tok.train_from_iterator(it(), trainer=trainer)
    tok.save(str(out_dir / "tokenizer.json"))
    (out_dir / "tokenizer.meta.json").write_text(json.dumps({"vocab": vocab, "trained_on": [str(p) for p in paths]}, indent=2))
    return tok


def load_tokenizer(tok_dir):
    return Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))


def encode_stream(tok, docs_path, out_dir, split="train", kind="clean", param=None, intensity=1.0, name=None, chunk=2000):
    name = name or stream_name(kind, param, intensity)
    out_dir = Path(out_dir)
    out = out_dir / f"{split}.{name}.bin"
    offsets = [0]
    n_tok = n_doc = 0
    with open(out, "wb") as f:
        batch = []

        def flush():
            nonlocal n_tok
            encs = tok.encode_batch(batch)
            parts = []
            for e in encs:
                ids = np.array(e.ids + [EOS_ID], dtype=np.uint16)
                parts.append(ids)
                n_tok += len(ids)
                offsets.append(n_tok)
            f.write(np.concatenate(parts).tobytes())
            batch.clear()

        for d in iter_docs(docs_path):
            batch.append(corrupt_doc(kind, param, d["i"], d["text"], intensity))
            n_doc += 1
            if len(batch) >= chunk:
                flush()
        if batch:
            flush()
    np.save(out_dir / f"{split}.{name}.idx.npy", np.array(offsets, dtype=np.int64))
    meta = {"split": split, "stream": name, "kind": kind, "param": param, "intensity": intensity,
            "source": str(docs_path), "n_tokens": n_tok, "n_docs": n_doc}
    (out_dir / f"{split}.{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return out, meta


def open_stream(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def open_indexed(tok_dir, split, name):
    tok_dir = Path(tok_dir)
    data = open_stream(tok_dir / f"{split}.{name}.bin")
    idx_p = tok_dir / f"{split}.{name}.idx.npy"
    if not idx_p.exists():
        raise SystemExit(f"{idx_p} missing: re-encode this stream (bin/lmcp tok encode ...) to get a document index")
    return data, np.load(idx_p)

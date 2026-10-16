"""Train a byte-level BPE tokenizer and encode document files into flat uint16 token streams.

Layout: data/tok/<corpus>_<variant>_v<vocab>/tokenizer.json
                                           /train.<stream>.bin  (uint16, eos after every document)
                                           /val.clean.bin
                                           /<stream>.meta.json
"""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .corrupt import corrupt_doc, stream_name
from .data import iter_docs

EOS, PAD = "<|eos|>", "<|pad|>"
EOS_ID, PAD_ID = 0, 1


def train_tokenizer(docs_path, out_dir, vocab, max_chars=300_000_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOS, PAD],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)

    def it():
        n = 0
        for d in iter_docs(docs_path):
            n += len(d["text"])
            if n > max_chars:
                return
            yield d["text"]

    tok.train_from_iterator(it(), trainer=trainer)
    tok.save(str(out_dir / "tokenizer.json"))
    return tok


def load_tokenizer(tok_dir):
    return Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))


def encode_stream(tok, docs_path, out_dir, split="train", kind="clean", param=None, chunk=2000):
    name = stream_name(kind, param)
    out = out_dir / f"{split}.{name}.bin"
    n_tok = n_doc = 0
    with open(out, "wb") as f:
        batch = []

        def flush():
            nonlocal n_tok
            encs = tok.encode_batch(batch)
            arr = np.concatenate([np.array(e.ids + [EOS_ID], dtype=np.uint16) for e in encs])
            f.write(arr.tobytes()); n_tok += len(arr); batch.clear()

        for d in iter_docs(docs_path):
            batch.append(corrupt_doc(kind, param, d["i"], d["text"]))
            n_doc += 1
            if len(batch) >= chunk:
                flush()
        if batch:
            flush()
    meta = {"split": split, "stream": name, "kind": kind, "param": param, "n_tokens": n_tok, "n_docs": n_doc}
    (out_dir / f"{split}.{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return out, meta


def open_stream(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

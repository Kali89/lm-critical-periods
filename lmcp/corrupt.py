"""Text-level corruptions, applied per document with a deterministic RNG keyed on the document index.

Each corruption is designed to remove one kind of structure while preserving others:
  shuffle_sentence       words shuffled within each sentence  -> kills syntax, keeps lexicon + sentence-level bag of words
  shuffle_doc_sentences  sentence order shuffled within a document -> kills discourse / long-range, keeps syntax
  shuffle_window(k)      words shuffled within windows of k -> keeps local co-occurrence (k small), kills sentences
  char_noise(p)          per-character delete/substitute/swap with prob p -> attacks the lexical/tokenization layer
Training-time corruptions (short_context, label_noise) live in train.py.
"""
import re

import numpy as np

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sentences(text):
    out = []
    for line in text.split("\n"):
        out.extend(s for s in SENT_SPLIT.split(line.strip()) if s)
    return out


def shuffle_sentence(text, rng, param=None):
    lines = []
    for line in text.split("\n"):
        sents = []
        for s in SENT_SPLIT.split(line.strip()):
            w = s.split()
            if len(w) > 1:
                rng.shuffle(w)
            sents.append(" ".join(w))
        lines.append(" ".join(s for s in sents if s))
    return "\n".join(lines)


def shuffle_doc_sentences(text, rng, param=None):
    s = sentences(text)
    rng.shuffle(s)
    return " ".join(s)


def shuffle_window(text, rng, param=5):
    k = int(param or 5)
    w = text.split()
    for i in range(0, len(w), k):
        seg = w[i:i + k]
        rng.shuffle(seg)
        w[i:i + k] = seg
    return " ".join(w)


def char_noise(text, rng, param=0.1):
    p = float(param if param is not None else 0.1)
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < p:
            op = rng.integers(3)
            if op == 0:          # delete
                pass
            elif op == 1:        # substitute
                out.append(LETTERS[rng.integers(26)] if c.islower() else LETTERS[rng.integers(26)].upper())
            else:                # swap with next
                if i + 1 < len(chars):
                    out.append(chars[i + 1]); out.append(c); i += 1
                else:
                    out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


CORRUPTIONS = {
    "shuffle_sentence": shuffle_sentence,
    "shuffle_doc_sentences": shuffle_doc_sentences,
    "shuffle_window": shuffle_window,
    "char_noise": char_noise,
}


def corrupt_doc(kind, param, idx, text):
    if kind == "clean":
        return text
    rng = np.random.default_rng([7, int(idx)])
    return CORRUPTIONS[kind](text, rng, param)


def stream_name(kind, param=None):
    return "clean" if kind == "clean" else (f"{kind}-{param}" if param is not None else kind)

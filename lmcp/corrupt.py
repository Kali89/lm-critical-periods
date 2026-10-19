"""Text-level corruptions, applied per document with a deterministic RNG keyed on the document index.

Each corruption removes one kind of structure while preserving others:
  shuffle_sentence       words shuffled within each sentence  -> kills syntax, keeps lexicon + sentence-level bag of words
  shuffle_doc_sentences  sentence order shuffled across a document (line structure kept) -> kills discourse / long-range, keeps syntax
  shuffle_window(k)      words shuffled within windows of k -> keeps local co-occurrence (k small), kills sentences
  char_noise(p)          per-character delete/substitute/swap with prob p -> attacks the lexical/tokenization layer
`intensity` in (0, 1] is the dose: the fraction of units (sentences, documents, windows, or the per-character
rate) that the corruption touches. intensity=1 is the full corruption.
Training-time corruptions (short_context, label_noise, vocab_permute) live in train.py.
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


def shuffle_sentence(text, rng, param=None, intensity=1.0):
    lines = []
    for line in text.split("\n"):
        sents = []
        for s in SENT_SPLIT.split(line.strip()):
            w = s.split()
            if len(w) > 1 and rng.random() < intensity:
                rng.shuffle(w)
            sents.append(" ".join(w))
        lines.append(" ".join(s for s in sents if s))
    return "\n".join(lines)


def shuffle_doc_sentences(text, rng, param=None, intensity=1.0):
    """Shuffle sentence order across the whole document, keeping the line structure: each line gets back the
    same number of sentences it had, so newline tokens and the within-line whitespace normalisation match
    shuffle_sentence (which rejoins words with single spaces). Token count then tracks the clean stream."""
    if rng.random() >= intensity:
        return text
    per_line = []
    for line in text.split("\n"):
        per_line.append([" ".join(s.split()) for s in SENT_SPLIT.split(line.strip()) if s.strip()])
    pool = [s for sents in per_line for s in sents]
    rng.shuffle(pool)
    out, k = [], 0
    for sents in per_line:
        out.append(" ".join(pool[k:k + len(sents)]))
        k += len(sents)
    return "\n".join(out)


def shuffle_window(text, rng, param=5, intensity=1.0):
    k = int(param or 5)
    w = text.split()
    for i in range(0, len(w), k):
        if rng.random() < intensity:
            seg = w[i:i + k]
            rng.shuffle(seg)
            w[i:i + k] = seg
    return " ".join(w)


def char_noise(text, rng, param=0.1, intensity=1.0):
    p = float(param if param is not None else 0.1) * intensity
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
                r = LETTERS[rng.integers(26)]
                out.append(r.upper() if c.isupper() else r)
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


def corrupt_doc(kind, param, idx, text, intensity=1.0):
    if kind == "clean":
        return text
    rng = np.random.default_rng([7, int(idx)])
    return CORRUPTIONS[kind](text, rng, param, float(intensity))


def stream_name(kind, param=None, intensity=1.0):
    if kind == "clean":
        return "clean"
    name = kind if param is None else f"{kind}-{param}"
    if intensity is not None and float(intensity) != 1.0:
        name += f"@{float(intensity):g}"
    return name

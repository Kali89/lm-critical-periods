# Dutch (L2) corpus: `babylm_nl`

Second-language corpus for the sequential L1-then-L2 design of Constantinescu, Pimentel, Cotterell & Warstadt
(TACL 2025). English side: `babylm100m` (CLIMB mirror of BabyLM-100M, lowercased).

## What it is and where it came from

**Dutch Wikipedia**, dump 20231101, HF dataset `wikimedia/wikipedia`, config `20231101.nl` (6 parquet files,
~1.4 GB, 2,135,977 articles, 367.7M whitespace words). Not gated.

Licence: Wikipedia text is CC BY-SA 4.0 / GFDL (from general knowledge; the dataset card's licence field was
not read in this task, so verify before redistributing derived text). Local research use is unaffected.

### Why not `BabyLM-community/babylm-nld`

Inspected via the HF API on 2026-10-16. It is the right kind of data (BabyLM multilingual collection, tier 100M,
304,611 documents, 109,885,564 whitespace tokens, 569 MB, 2 parquet files of 108 MB + 249 MB, one `train` split,
fields text/doc-id/category/data-source/age-estimate/license/misc, categories include child-books and
child-directed-speech) but:

- it is **gated (`auto`)**: README and parquet files return HTTP 401 without an authenticated, approved HF account,
  and no HF token exists on this machine;
- licence tag is `unknown` (per-document `license` field instead);
- it has **no dev split** (a val split would have to be held out).

If Matthew requests access (the dataset page, "Agree and access", then `hf auth login`), it is a better match for
a child-directed-input study than encyclopaedic text and would be a small change to `lmcp/data.py`. Not done.

## What `prepare` produces

Subsampling is deterministic (crc32 of the article id), no random state:

1. drop articles under 100 words (1.41M stub articles, bot-generated villages/species; 66% of articles, 15% of words);
2. keep 33% of the remaining articles (`NL_KEEP_PER_MILLE = 330`);
3. hold out 5% of the kept articles as `val` (article level, so no train/val leakage);
4. **lowercase all text** (see below);
5. stream the non-blank lines through the same `chunk_lines` as the English corpus, then shuffle documents with
   seed 1234. Blank lines are dropped first so documents come out ~2000 chars like English; a document can
   therefore span two neighbouring articles (as English documents span neighbouring lines of a source file).
   Every doc has `src = "wikipedia_nl"`.

Output: `data/corpora/babylm_nl_clean/{train,val}.docs.jsonl` and `meta.json`.

| | docs | chars | whitespace words | mean chars/doc |
|---|---|---|---|---|
| train | 291,370 | 647,881,095 | 98,858,844 | 2,223 |
| val | 15,124 | 33,608,789 | 5,132,950 | 2,222 |
| (English babylm100m_clean train, for reference) | 237,659 | 514,925,974 | 93,238,048 | 2,166 |

Dutch has more characters per word than English (about 6.6 vs 5.5), so token counts will differ from the
word match; the tokenizer has not been trained for Dutch here, so the BPE token count is unmeasured.

## Lowercasing

The English `clean` variant is fully lowercased, so Dutch is lowercased in `iter_raw_docs` (`text.lower()`),
so case is not a confound in the bilingual experiment. The `variant` argument is ignored for this corpus
(only the lowercased form exists; the CLI default `clean` is the one to use).

## Commands

    bin/lmcp data download --corpus babylm_nl     # ~1.4 GB into data/raw/babylm_nl/20231101.nl/ (also fetches BLiMP if missing)
    bin/lmcp data prepare  --corpus babylm_nl     # ~40 s, writes data/corpora/babylm_nl_clean/

## Caveats

- Encyclopaedic register, very different from BabyLM's child-directed and spoken mix; the L1/L2 contrast is
  therefore also a register contrast. The same applies to the English side's Wikipedia share (~10%).
- Wikipedia markup residue (section headings as short lines, lists, tables flattened) is left as is.
- Raw stub filtering threshold (100 words) and the 33% keep rate were chosen to hit ~100M words, not tuned otherwise.

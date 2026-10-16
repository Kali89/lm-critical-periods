"""Small statistics helpers: percentile bootstrap CIs (vectorised) used by evals and analysis."""
import numpy as np

_CHUNK_ELEMS = 4_000_000   # cap on elements materialised per bootstrap chunk


def _chunks(n_boot, n_items, k=1):
    step = max(1, min(n_boot, _CHUNK_ELEMS // max(1, n_items * k)))
    for s in range(0, n_boot, step):
        yield min(step, n_boot - s)


def _resampled_stat(values, idx, stat):
    """values: (n,) or (n, k); idx: (c, n) -> statistic per resample, shape (c,) or (c, k)."""
    samp = values[idx]                              # (c, n[, k])
    if stat is np.mean:
        return samp.mean(axis=1)
    try:
        return np.asarray(stat(samp, axis=1))
    except TypeError:
        return np.array([stat(s) for s in samp])


def bootstrap_dist(values, stat=np.mean, n=1000, seed=0):
    """Bootstrap distribution of `stat` over items (rows of `values`). Returns (n,) or (n, k)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    N = len(values)
    k = values.shape[1] if values.ndim == 2 else 1
    out = []
    for c in _chunks(n, N, k):
        out.append(_resampled_stat(values, rng.integers(0, N, size=(c, N)), stat))
    return np.concatenate(out, axis=0)


def _pct(dist, alpha):
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi


def bootstrap_ci(values, stat=np.mean, n=1000, alpha=0.05, seed=0):
    """95% (by default) percentile bootstrap CI of stat(values) resampling items. 2-D input = several
    metrics over the same items (shared resamples); returns arrays. Returns (nan, nan) if empty."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    lo, hi = _pct(bootstrap_dist(values, stat, n, seed), alpha)
    return (float(lo), float(hi)) if np.ndim(lo) == 0 else (lo, hi)


def bootstrap_diff_ci(a, b, stat=np.mean, n=1000, alpha=0.05, seed=0, paired=None):
    """CI for stat(a) - stat(b). paired (default: True iff equal lengths) resamples the same item indices
    for both; otherwise resamples a and b independently. Returns (estimate, lo, hi)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if paired is None:
        paired = len(a) == len(b)
    if paired and len(a) != len(b):
        raise ValueError("paired bootstrap needs equal lengths")
    est = float(stat(a) - stat(b))
    if len(a) == 0 or len(b) == 0:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = []
    for c in _chunks(n, max(len(a), len(b))):
        ia = rng.integers(0, len(a), size=(c, len(a)))
        ib = ia if paired else rng.integers(0, len(b), size=(c, len(b)))
        out.append(_resampled_stat(a, ia, stat) - _resampled_stat(b, ib, stat))
    lo, hi = _pct(np.concatenate(out), alpha)
    return est, float(lo), float(hi)

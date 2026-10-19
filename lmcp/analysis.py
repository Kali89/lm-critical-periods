"""Statistical analysis of results/results.jsonl (one row per run) and results/curves.jsonl (one row per eval).

Plain functions on pandas DataFrames; nothing here trains or touches runs/. Functions that fit models never
raise on small data: they return what can be computed and put a warning string in `.attrs["warning"]`
(DataFrames) or the "warning" key (dicts).
"""
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ABILITY_METRICS = ("blimp_acc", "induction_acc", "induction_gain", "long_range_gain", "context_gain")
# loss-like metrics: chance is the uniform-guess loss log(vocab), exactly as for val_loss. They are summarised but
# not proposed as the primary endpoint (their snr against uniform-guess loss is always huge, like val_loss's).
LOSS_METRICS = ("val_loss", "val_loss_ctx_32", "val_loss_ctx_full")
SUMMARY_METRICS = LOSS_METRICS + ABILITY_METRICS
CHANCE = {"blimp_acc": 0.5, "induction_acc": 0.0, "induction_gain": 0.0, "long_range_gain": 0.0, "context_gain": 0.0}
MECHANISM_ENDPOINTS = ("val_loss_ctx_32", "context_gain", "blimp_acc")
# registered smallest effects of interest, docs/preregistration.md 2026-10-19
MIN_EFFECTS = {"val_loss_ctx_32": 0.016, "context_gain": 0.0175, "blimp_acc": 0.010}
BASE_KEYS = ["model.size", "train.schedule", "train.tokens"]


# ----------------------------------------------------------------------------------------------- loading
def _fmt_window(w):
    if isinstance(w, (list, tuple, np.ndarray)) and len(w) == 2:
        return f"w{float(w[0]):.2f}-{float(w[1]):.2f}"
    return None


def _num(x, default):
    return default if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _condition(row):
    kind = row["corruption.kind"]
    if kind == "clean":
        return "clean"
    s = str(kind)
    p = row.get("corruption.param")
    if p is not None and not (isinstance(p, float) and math.isnan(p)):
        s += f"-{p:g}" if isinstance(p, (int, float)) else f"-{p}"
    inten = _num(row.get("corruption.intensity"), 1.0)
    if float(inten) != 1.0:
        s += f"@{float(inten):g}"
    return f"{s} {_fmt_window(row.get('corruption.window')) or 'all'}"


def _read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def add_derived(df):
    """Adds corruption.intensity (default 1.0), doc_order_seed (default train.seed) and condition."""
    df = df.copy()
    if df.empty:
        return df
    if "corruption.intensity" not in df:
        df["corruption.intensity"] = 1.0
    df["corruption.intensity"] = df["corruption.intensity"].apply(lambda v: float(_num(v, 1.0)))
    if "data.doc_order_seed" not in df:
        df["data.doc_order_seed"] = df["train.seed"]
    df["doc_order_seed"] = df["data.doc_order_seed"].where(df["data.doc_order_seed"].notna(), df["train.seed"])
    for c in ("corruption.param", "corruption.window"):
        if c not in df:
            df[c] = None
    df["condition"] = df.apply(_condition, axis=1)
    return df


def load_results(results_dir="results"):
    """-> (results_df, curves_df). results_df gains `condition` and `doc_order_seed`; curves_df gains
    `condition` too (joined by run) when the run is present in results."""
    d = Path(results_dir)
    res = add_derived(_read_jsonl(d / "results.jsonl"))
    cur = _read_jsonl(d / "curves.jsonl")
    if not cur.empty and not res.empty:
        cur = cur.merge(res[["run", "condition"]], on="run", how="left")
    return res, cur


def _grid_col(res):
    cols = [c for c in res.columns if c.endswith("grid")]
    return "run.grid" if "run.grid" in cols else (cols[0] if cols else None)


def filter_grids(res, cur, grids):
    """Restrict results to rows whose grid is in `grids` and curves to the runs that remain.
    grids falsy -> inputs returned unchanged."""
    if not grids:
        return res, cur
    col = _grid_col(res)
    if col is None:
        raise ValueError("results have no grid column; cannot apply --grid")
    res = res[res[col].isin(list(grids))].copy()
    if cur is not None and len(cur) and "run" in cur:
        cur = cur[cur["run"].isin(set(res["run"]))].copy()
    return res, cur


def pooled_grids(res):
    col = _grid_col(res) if res is not None and len(res) else None
    return sorted(res[col].dropna().astype(str).unique()) if col else []


def _done(df):
    return df[df["status"] == "done"] if "status" in df else df


# ------------------------------------------------------------------------------------------- landmarks
def landmarks(curves_df, metrics=("blimp_acc", "induction_acc", "long_range_gain", "val_loss_ctx_32", "context_gain"),
              min_range=0.02):
    """Acquisition point per run and metric: first eval after which the metric stays beyond the midpoint
    between its chance (blimp 0.5) or initial value and its final value. NaN frac/tokens if never acquired,
    or if the metric moved less than `min_range` from chance (not acquired at all; avoids calling noise a
    landmark)."""
    cols = ["run", "metric", "frac", "tokens", "initial", "final", "chance", "midpoint"]
    out = []
    if curves_df is None or len(curves_df) == 0:
        return pd.DataFrame(columns=cols)
    for run, g in curves_df.groupby("run"):
        g = g.sort_values("step")
        for m in metrics:
            if m not in g:
                continue
            s = g[["frac", "tokens", m]].dropna()
            if len(s) < 2:
                continue
            v = s[m].to_numpy(float)
            initial, final = v[0], v[-1]
            chance = CHANCE["blimp_acc"] if m == "blimp_acc" else initial
            mid = (chance + final) / 2
            sign = np.sign(final - chance)
            frac = tokens = np.nan
            if sign != 0 and abs(final - chance) >= min_range:
                ok = sign * (v - mid) > 0
                # first index from which every later eval is beyond the midpoint
                tail = np.logical_and.accumulate(ok[::-1])[::-1]
                if tail.any():
                    i = int(np.argmax(tail))
                    frac, tokens = float(s["frac"].iloc[i]), float(s["tokens"].iloc[i])
            out.append(dict(run=run, metric=m, frac=frac, tokens=tokens, initial=initial, final=final,
                            chance=chance, midpoint=mid))
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------------------------- baselines
def baseline_summary(results_df, metrics=SUMMARY_METRICS):
    """Clean runs grouped by (model.size, train.schedule, train.tokens); per metric mean, seed SD, n, chance,
    snr = |mean - chance| / SD. Long format: one row per group x metric."""
    cols = BASE_KEYS + ["metric", "mean", "sd", "n", "chance", "snr"]
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame(columns=cols)
    df = _done(results_df)
    df = df[df["corruption.kind"] == "clean"]
    rows = []
    for keys, g in df.groupby(BASE_KEYS):
        for m in metrics:
            if m not in g:
                continue
            v = g[m].dropna().to_numpy(float)
            if len(v) == 0:
                continue
            if m in LOSS_METRICS:
                vocab = g["data.tokenizer_vocab"].iloc[0] if "data.tokenizer_vocab" in g else np.nan
                chance = math.log(vocab) if vocab == vocab else np.nan     # uniform-guess loss
            else:
                chance = CHANCE.get(m, np.nan)
            sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            snr = abs(v.mean() - chance) / sd if sd and sd == sd and chance == chance else np.nan
            rows.append(dict(zip(BASE_KEYS, keys), metric=m, mean=float(v.mean()), sd=sd, n=len(v),
                             chance=chance, snr=snr))
    return pd.DataFrame(rows, columns=cols)


def propose_primary(summary, candidates=ABILITY_METRICS):
    """-> (metric or None, justification). Largest snr among ability metrics (val_loss excluded: it measures
    overall fit, not an ability, and its snr against uniform-guess loss is always huge) in the baseline group
    with the most seeds (ties: most tokens). Propose only; nothing is decided here."""
    if summary is None or len(summary) == 0:
        return None, "no clean baseline runs yet"
    s = summary
    gsize = s.groupby(BASE_KEYS)["n"].max().reset_index().sort_values(["n", "train.tokens"], ascending=False)
    top = gsize.iloc[0]
    g = s[(s["model.size"] == top["model.size"]) & (s["train.schedule"] == top["train.schedule"])
          & (s["train.tokens"] == top["train.tokens"]) & s["metric"].isin(candidates)]
    desc = f"{top['model.size']}/{top['train.schedule']}/{int(top['train.tokens']):,} tokens (n={int(top['n'])} seeds)"
    g = g.dropna(subset=["snr"])
    if g.empty:
        return None, (f"largest baseline group is {desc}; seed SD not estimable (need >=2 seeds), "
                      "so no primary metric can be proposed from data")
    best = g.sort_values("snr", ascending=False).iloc[0]
    msg = (f"{best['metric']}: highest signal-to-noise (|mean-chance|/seed SD = {best['snr']:.1f}; mean "
           f"{best['mean']:.3f}, SD {best['sd']:.4f}, chance {best['chance']:.3g}) among "
           f"{', '.join(candidates)} in the largest baseline group {desc}")
    if int(top["n"]) < 3:
        msg += "; WARNING fewer than 3 seeds, SD estimate is unreliable"
    return best["metric"], msg


# ---------------------------------------------------------------------------------------- inference
def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as input; NaNs pass through)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * pv[order]))
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def _coef_table(res, method, ref, extra=None):
    ci = res.conf_int()
    rows = []
    for name in res.params.index:
        mt = re.search(r"\[T\.(.*)\]$", name)
        if not mt:
            continue
        rows.append(dict(condition=mt.group(1), coef=res.params[name], se=res.bse[name],
                         ci_lo=ci.loc[name, 0], ci_hi=ci.loc[name, 1], p=res.pvalues[name]))
    t = pd.DataFrame(rows, columns=["condition", "coef", "se", "ci_lo", "ci_hi", "p"])
    t["p_holm"] = holm(t["p"].to_numpy())
    t["method"] = method
    t["reference"] = ref
    return t


def mixed_model(results_df, metric, reference="clean"):
    """metric ~ C(condition, Treatment(reference)) with random intercept for train.seed (+ variance component
    for doc_order_seed when it differs from train.seed). Falls back to OLS with HC3 standard errors if there
    are <3 seeds or the mixed fit fails. Result .attrs: method, warning. Never raises."""
    import statsmodels.formula.api as smf
    empty = pd.DataFrame(columns=["condition", "coef", "se", "ci_lo", "ci_hi", "p", "p_holm", "method", "reference"])
    warns = []

    def done(t, method):
        if len(t):
            method = t["method"].iloc[0]
        t.attrs.update(method=method, warning="; ".join(warns))
        return t

    try:
        if results_df is None or len(results_df) == 0 or metric not in results_df:
            warns.append("no data or metric missing")
            return done(empty, "none")
        df = _done(results_df)
        df = df.dropna(subset=[metric]).copy()
        if reference not in set(df["condition"]):
            warns.append(f"reference condition {reference!r} has no runs")
            return done(empty, "none")
        if df["condition"].nunique() < 2:
            warns.append("only one condition present")
            return done(empty, "none")
        df["y"] = df[metric].astype(float)
        df["seed"] = df["train.seed"].astype(str)
        df["doc"] = df["doc_order_seed"].astype(str)
        formula = f"y ~ C(condition, Treatment({reference!r}))"
        n_seeds = df["seed"].nunique()
        use_vc = (df["doc_order_seed"] != df["train.seed"]).any()
        if n_seeds >= 3:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    kw = dict(vc_formula={"doc_order": "0 + C(doc)"}, re_formula="1") if use_vc else {}
                    res = smf.mixedlm(formula, df, groups=df["seed"], **kw).fit(reml=True)
                    for w in caught:
                        warns.append(f"{w.category.__name__}: {str(w.message)[:120]}")
                if not np.isfinite(res.bse.to_numpy()).all():
                    raise RuntimeError("non-finite standard errors")
                if getattr(res, "converged", True) is False:
                    warns.append("mixed model did not converge")
                return done(_coef_table(res, "mixedlm" + ("+doc_order_vc" if use_vc else ""), reference),
                            "mixedlm")
            except Exception as e:                                          # fall through to OLS
                warns.append(f"MixedLM failed ({type(e).__name__}: {str(e)[:100]}); used OLS HC3")
        else:
            warns.append(f"only {n_seeds} seed(s) (<3); used OLS with HC3 errors, ignoring seed clustering")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.ols(formula, df).fit(cov_type="HC3")
        t = _coef_table(res, "ols_hc3", reference)
        if t[["se", "p"]].isna().any().any() or not np.isfinite(t["se"]).all():
            warns.append("HC3 standard errors not estimable (too few runs per condition)")
        return done(t, "ols_hc3")
    except Exception as e:
        warns.append(f"analysis failed: {type(e).__name__}: {str(e)[:150]}")
        return done(empty, "none")


def dose_response(results_df, metric, kind, window):
    """OLS of metric ~ intensity across runs of one corruption kind and window. Returns a dict: slope,
    ci_lo, ci_hi, p, r2, n, n_intensities, warning."""
    out = dict(metric=metric, kind=kind, window=window, slope=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
               r2=np.nan, n=0, n_intensities=0, warning="")
    try:
        import statsmodels.api as sm
        df = _done(results_df)
        w = tuple(window) if isinstance(window, (list, tuple)) else window
        sel = df["corruption.kind"] == kind
        sel &= df["corruption.window"].apply(lambda x: (tuple(x) if isinstance(x, (list, tuple)) else x) == w)
        d = df[sel].dropna(subset=[metric, "corruption.intensity"])
        out["n"], out["n_intensities"] = len(d), d["corruption.intensity"].nunique()
        if out["n_intensities"] < 2:
            out["warning"] = "need >=2 distinct intensities"
            return out
        X = sm.add_constant(d["corruption.intensity"].astype(float))
        res = sm.OLS(d[metric].astype(float), X).fit()
        ci = res.conf_int().loc["corruption.intensity"]
        out.update(slope=float(res.params["corruption.intensity"]), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                   p=float(res.pvalues["corruption.intensity"]), r2=float(res.rsquared))
        if len(d) <= 2:
            out["warning"] = "2 runs: CI not estimable"
    except Exception as e:
        out["warning"] = f"{type(e).__name__}: {str(e)[:150]}"
    return out


# ------------------------------------------------------------------------------------------- power
def power_simulation(seed_sd, effect_sizes, n_seeds=(2, 3, 4, 5, 6, 8), alpha=0.05, n_sims=2000, seed=0,
                     n_clean=None, n_arm=None, sd_mult=None):
    """Welch t-test power for a mean difference between a corrupted arm and the clean arm (Normal seed noise,
    SD `seed_sd` * sd_mult in both arms). Effects are in metric units; effect_sd = effect / (seed_sd * mult).

    Legacy call (n_clean and n_arm both None): equal n per arm taken from `n_seeds`, no SD scaling.
    n_clean: clean-arm size (int, default = the arm's n). n_arm: corrupted-arm size(s) (int or sequence,
    default `n_seeds`). sd_mult: SD multipliers; None -> (1.0, 2.0, 3.0) when n_clean/n_arm is given, else (1.0,).
    Returns columns effect, effect_sd, n (= corrupted-arm size), n_clean, n_arm, sd_mult, power."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    legacy = n_clean is None and n_arm is None
    arms = n_seeds if n_arm is None else n_arm
    arms = [int(arms)] if np.isscalar(arms) else [int(n) for n in arms]
    if sd_mult is None:
        sd_mult = (1.0,) if legacy else (1.0, 2.0, 3.0)
    mults = [float(sd_mult)] if np.isscalar(sd_mult) else [float(m) for m in sd_mult]
    rows = []
    for mult in mults:
        sd = seed_sd * mult
        for eff in effect_sizes:
            for n in arms:
                nc = int(n_clean) if n_clean is not None else n
                a = rng.normal(0.0, sd, size=(n_sims, nc))
                b = rng.normal(eff, sd, size=(n_sims, n))
                with np.errstate(all="ignore"):
                    p = stats.ttest_ind(b, a, axis=1, equal_var=False).pvalue
                rows.append(dict(effect=float(eff), effect_sd=float(eff / sd) if sd else np.nan, n=int(n),
                                 n_clean=nc, n_arm=int(n), sd_mult=mult, power=float(np.mean(p < alpha))))
    return pd.DataFrame(rows)


def _holm_rows(p):
    """Holm-adjusted p-values along the last axis of a 2-D array (vectorised over rows)."""
    m = p.shape[1]
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ps, axis=1))
    out = np.empty_like(adj)
    np.put_along_axis(out, order, adj, axis=1)
    return out


# The registered H1 confirmatory family (docs/preregistration.md, H1): one corrupted-vs-clean test per corruption at
# 600M, each mapped to one endpoint. Ordered (label, endpoint) pairs; Holm is taken across all members.
H1_FAMILY = [("shuffle_sentence", "val_loss_ctx_32"), ("shuffle_doc_sentences", "context_gain"),
             ("short_context", "context_gain")]


def power_confirmatory(seed_sd, effects, family=H1_FAMILY, rho=(-0.5, 0.0, 0.5), sd_mult=(1.0, 2.0, 3.0), n_pairs=4,
                       n_clean_extra=2, alpha=0.05, n_sims=4000, seed=0):
    """Simulated power of the registered H1 procedure: one corrupted-vs-clean test per `family` member, Holm across them.

    seed_sd, effects: dicts keyed by endpoint (every effect is a deficit in the harmful direction, so treated as
    positive). family: ordered list of (label, endpoint). Seeds 0..n_pairs-1 have a clean and a corrupted run; clean seeds
    n_pairs..n_pairs+n_clean_extra-1 are independent extra clean runs. Clean runs are simulated once per endpoint per
    simulation and shared by every family member on that endpoint (both are compared with the same clean reference);
    each member draws its own corrupted runs. Per endpoint, clean_s = u_s + e1_s and corrupt_s = sign(rho) * u_s + e2_s +
    effect with u_s ~ N(0, |rho| sd^2), e ~ N(0, (1-|rho|) sd^2), so each run has variance sd^2 and corr(clean_s,
    corrupt_s) = rho (rho < 0 gives the anti-correlated case). Extra clean runs are N(0, sd^2). Endpoints are simulated
    independently. Analyses, one-sided at alpha, Holm across the family: "paired" = paired t on the n_pairs pairs
    (df n_pairs-1); "welch" = Welch t, all n_pairs+n_clean_extra clean vs n_pairs corrupted, ignoring pairing.
    Returns columns label, endpoint, analysis, rho, sd_mult, power (label = endpoint = "all three" is the joint
    rejection probability)."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    eps = list(dict.fromkeys(ep for _, ep in family))
    rows = []
    for r in ([float(rho)] if np.isscalar(rho) else [float(x) for x in rho]):
        for mult in ([float(sd_mult)] if np.isscalar(sd_mult) else [float(x) for x in sd_mult]):
            pp = np.empty((n_sims, len(family)))
            pw = np.empty((n_sims, len(family)))
            ar = abs(r)
            for ep in eps:
                sd = seed_sd[ep] * mult
                u = rng.normal(0.0, sd * math.sqrt(ar), size=(n_sims, n_pairs))
                clean = u + rng.normal(0.0, sd * math.sqrt(1 - ar), size=(n_sims, n_pairs))
                extra = rng.normal(0.0, sd, size=(n_sims, n_clean_extra))
                all_clean = np.concatenate([clean, extra], axis=1)
                for k, (_, fep) in enumerate(family):
                    if fep != ep:
                        continue
                    corr = math.copysign(1.0, r) * u + rng.normal(0.0, sd * math.sqrt(1 - ar), size=(n_sims, n_pairs)) + effects[ep]
                    with np.errstate(all="ignore"):
                        pp[:, k] = stats.ttest_rel(corr, clean, axis=1, alternative="greater").pvalue
                        pw[:, k] = stats.ttest_ind(corr, all_clean, axis=1, equal_var=False, alternative="greater").pvalue
            for name, pv in (("paired", pp), ("welch", pw)):
                rej = _holm_rows(np.nan_to_num(pv, nan=1.0)) < alpha
                for k, (lab, ep) in enumerate(family):
                    rows.append(dict(label=lab, endpoint=ep, analysis=name, rho=r, sd_mult=mult, power=float(rej[:, k].mean())))
                rows.append(dict(label="all three", endpoint="all three", analysis=name, rho=r, sd_mult=mult,
                                 power=float(rej.all(axis=1).mean())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- plasticity
def predict_window_from_plasticity(curves_df, metric="plast/fisher_trace", results_df=None):
    """For clean runs: frac at the metric's peak and the first frac after the peak where it drops below 50% of
    the peak. Returns dict(per_run=DataFrame, mean_frac_peak, mean_frac_half) or None if unavailable."""
    if curves_df is None or len(curves_df) == 0 or metric not in curves_df:
        return None
    c = curves_df
    if results_df is not None and len(results_df):
        clean = set(results_df.loc[results_df["corruption.kind"] == "clean", "run"])
        c = c[c["run"].isin(clean)]
    else:
        c = c[c["run"].str.contains("__clean__")]
    rows = []
    for run, g in c.groupby("run"):
        g = g.sort_values("step").dropna(subset=[metric])
        if len(g) < 2:
            continue
        v, f = g[metric].to_numpy(float), g["frac"].to_numpy(float)
        i = int(np.argmax(v))
        after = np.where(v[i:] < 0.5 * v[i])[0]
        rows.append(dict(run=run, frac_peak=f[i], frac_half=f[i + after[0]] if len(after) else np.nan))
    if not rows:
        return None
    pr = pd.DataFrame(rows)
    mean_peak = float(pr["frac_peak"].mean())
    # monotone non-increasing after the first eval point -> the "peak" is just the init value
    mono = all(bool(np.all(np.diff(g.sort_values("step").dropna(subset=[metric])[metric].to_numpy(float)) <= 0))
               for _, g in c.groupby("run") if g[metric].notna().sum() >= 2)
    informative = not (mean_peak == 0 or mono)
    out = dict(per_run=pr, mean_frac_peak=mean_peak, mean_frac_half=float(pr["frac_half"].mean()),
               informative=informative)
    if not informative:
        out["fallback"] = _landmark_window(curves_df, set(pr["run"]))
    return out


def _landmark_window(curves_df, runs, metrics=("blimp_acc", "induction_acc", "long_range_gain")):
    """Ability-landmark window: mean frac/tokens at which each ability reaches half of its final value."""
    lm = landmarks(curves_df[curves_df["run"].isin(runs)], metrics=metrics)
    out = {}
    for m in metrics:
        g = lm[lm["metric"] == m] if len(lm) else lm
        out[m] = dict(frac_half_mean=float(g["frac"].mean()) if len(g) else float("nan"),
                      tokens_half_mean=float(g["tokens"].mean()) if len(g) else float("nan"))
    return out


def format_window(p):
    """One-paragraph text for a predict_window_from_plasticity result."""
    if p is None:
        return "no plasticity metric in curves yet"
    if p.get("informative", True):
        return (f"Mean frac of peak: {p['mean_frac_peak']:.3f}; mean frac where it first falls below 50% of peak: "
                f"{p['mean_frac_half']:.3f}")
    fb = "; ".join(f"{m}: frac {v['frac_half_mean']:.3f}, tokens {v['tokens_half_mean']:.4g}"
                   for m, v in p["fallback"].items())
    return f"Fisher trace uninformative (peak at init); ability landmarks (mean point reaching half of final value): {fb}"


# ----------------------------------------------------------------------------------------- prereg
def _endpoint_sd(summ, metric):
    """(seed SD, n seeds) of `metric` in its best-populated clean baseline group, or (nan, 0) if absent/not estimable."""
    if summ is None or len(summ) == 0 or metric not in set(summ["metric"]):
        return np.nan, 0
    top = summ[summ["metric"] == metric].sort_values(["n", "train.tokens"], ascending=False).iloc[0]
    return float(top["sd"]), int(top["n"])


def prereg_numbers(results_dir="results", effect_sds=(0.5, 1, 2, 3), n_sims=2000, grids=None, min_effects=None,
                   n_clean=6, n_arms=(3, 4, 5, 6), sd_mult=(1.0, 2.0, 3.0), endpoints=MECHANISM_ENDPOINTS):
    res, cur = filter_grids(*load_results(results_dir), grids)
    summ = baseline_summary(res)
    metric, why = propose_primary(summ)
    sd = np.nan
    if metric is not None:
        top = summ[summ["metric"] == metric].sort_values(["n", "train.tokens"], ascending=False).iloc[0]
        sd = float(top["sd"])
    unit = sd if sd == sd and sd > 0 else 1.0           # power depends only on effect/SD; fall back to SD units
    power = power_simulation(unit, [k * unit for k in effect_sds], n_sims=n_sims)
    me = {**MIN_EFFECTS, **(min_effects or {})}
    endpoint_power = {}
    for ep in endpoints:
        esd, en = _endpoint_sd(summ, ep)
        eff = me.get(ep, np.nan)
        tbl = None
        if esd == esd and esd > 0 and eff == eff:
            tbl = power_simulation(esd, [eff], n_sims=n_sims, n_clean=n_clean, n_arm=n_arms, sd_mult=sd_mult)
        endpoint_power[ep] = dict(sd=esd, n_seeds=en, min_effect=eff, table=tbl)
    fam_eps = list(dict.fromkeys(ep for _, ep in H1_FAMILY))
    sds = {ep: endpoint_power[ep]["sd"] for ep in fam_eps if ep in endpoint_power}
    if len(sds) == len(fam_eps) and all(v == v and v > 0 for v in sds.values()) and all(me.get(ep, np.nan) == me.get(ep, np.nan) for ep in fam_eps):
        confirmatory = power_confirmatory(sds, {ep: me[ep] for ep in fam_eps}, n_sims=max(n_sims, 4000))
    else:
        confirmatory = "not rescored"
    return dict(baseline_summary=summ, confirmatory_power=confirmatory, primary=metric, primary_justification=why, primary_seed_sd=sd,
                power=power, endpoint_power=endpoint_power, n_clean_arm=n_clean, plasticity=predict_window_from_plasticity(cur, results_df=res),
                n_runs=len(res), n_clean=int((res["corruption.kind"] == "clean").sum()) if len(res) else 0)


def _md(df, fmt="{:.3f}"):
    if df is None or len(df) == 0:
        return "_none_"
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def render_prereg_numbers(d):
    s = ["## Baseline summary (clean runs; seed SD, snr = |mean - chance| / SD)", "", _md(d["baseline_summary"]), ""]
    sd = d["primary_seed_sd"]
    pow_lines = ["## Power (simulated Welch t-test, equal n per arm, alpha 0.05)", "",
                 f"Seed SD of primary metric: {sd:.4g}" if sd == sd else
                 "Seed SD not yet estimable (need >=2 clean seeds); effects shown in SD units, which fully determines power.", ""]
    if len(d["power"]):
        pv = d["power"].pivot(index="effect_sd", columns="n", values="power").reset_index()
        pv.columns = ["effect (SD)"] + [f"n={c}" for c in pv.columns[1:]]
        pow_lines.append(_md(pv, "{:.2f}"))
    s += [f"## Power for the mechanism-mapped endpoints (Welch, n_clean={d.get('n_clean_arm', 6)} vs n_arm, alpha 0.05)", "",
          "Seed SD from the clean baselines at the best-populated group; the 2x/3x rows are what the SD might be "
          "if the baseline SD is underestimated. Effect = smallest effect of interest, in the metric's own units.", ""]
    for ep, e in (d.get("endpoint_power") or {}).items():
        s += [f"### {ep}", ""]
        if e["sd"] != e["sd"]:
            s += ["_not rescored_ (metric absent from clean baseline results, or fewer than 2 seeds: run "
                  "`bin/lmcp rescore` on the finished baselines, then re-run prereg)", ""]
            continue
        s += [f"seed SD {e['sd']:.4g} (n={e['n_seeds']} seeds); effect {e['min_effect']:g}", ""]
        pv = e["table"].pivot(index="sd_mult", columns="n_arm", values="power").reset_index()
        pv.columns = ["SD multiplier"] + [f"n_arm={c}" for c in pv.columns[1:]]
        pv["SD multiplier"] = pv["SD multiplier"].map(lambda m: f"{m:g}x (SD {e['sd'] * m:.4g})")
        s += [_md(pv, "{:.2f}"), ""]
    s += ["## Power of the registered confirmatory procedure (one-sided Welch t, 6 clean vs 4 per arm, Holm across the "
          "three H1 tests, alpha 0.05)", ""]
    cp = d.get("confirmatory_power")
    if cp is None or isinstance(cp, str):
        s += ["_not rescored_ (an endpoint SD is missing: run `bin/lmcp rescore` on the finished baselines, then re-run prereg)", ""]
    else:
        s += ["Pairing by seed is reported as a sensitivity analysis because the within-seed correlation is unknown; the "
              "paired test only beats Welch when that correlation exceeds about 0.5.", "",
              "All three true effects (the registered smallest effects of interest) present at once; seed SD scaled by the "
              "multiplier; rho = correlation between a seed's clean and corrupted run. Columns are the three H1 tests "
              "(corruption vs clean, with its mapped endpoint: shuffle_sentence on val_loss_ctx_32, shuffle_doc_sentences "
              "and short_context on context_gain); entries are the probability each test is rejected after Holm; "
              "'all three' = all rejected.", ""]
        for analysis, head, lvl in (("welch", None, "###"),
                                    ("paired", "### Sensitivity: paired t over the 4 seed pairs (df = 3)", "####")):
            if head:
                s += [head, ""]
            for mult in sorted(cp["sd_mult"].unique()):
                sub = cp[(cp["analysis"] == analysis) & (cp["sd_mult"] == mult)]
                if not len(sub):
                    continue
                pv = sub.pivot(index="rho", columns="label", values="power")
                cols = list(dict.fromkeys(lab for lab, _ in H1_FAMILY)) + ["all three"]
                pv = pv[cols].reset_index()
                s += [f"{lvl} SD multiplier {mult:g}x", "", _md(pv, "{:.2f}"), ""]
    s += ["## Historical (superseded 2026-10-19; kept for the audit trail)", "",
          "## Proposed primary metric (proposal only)", "", f"**{d['primary']}**: {d['primary_justification']}", ""]
    s += pow_lines
    s += ["", "## Plasticity-predicted window", ""]
    p = d["plasticity"]
    if p is None:
        s.append("_no plasticity metric in curves yet_")
    else:
        s += [format_window(p), "", _md(p["per_run"])]
    return "\n".join(s) + "\n"

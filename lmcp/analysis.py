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

ABILITY_METRICS = ("blimp_acc", "induction_acc", "induction_gain", "long_range_gain")
SUMMARY_METRICS = ("val_loss",) + ABILITY_METRICS
CHANCE = {"blimp_acc": 0.5, "induction_acc": 0.0, "induction_gain": 0.0, "long_range_gain": 0.0}
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


def _done(df):
    return df[df["status"] == "done"] if "status" in df else df


# ------------------------------------------------------------------------------------------- landmarks
def landmarks(curves_df, metrics=("blimp_acc", "induction_acc", "long_range_gain"), min_range=0.02):
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
            if m == "val_loss":
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
def power_simulation(seed_sd, effect_sizes, n_seeds=(2, 3, 4, 5, 6, 8), alpha=0.05, n_sims=2000, seed=0):
    """Welch t-test power for a mean difference between a corrupted arm and the clean arm (equal n per arm,
    Normal seed noise with SD `seed_sd` in both arms). Effects are in metric units; effect_sd = effect/sd."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    rows = []
    for eff in effect_sizes:
        for n in n_seeds:
            a = rng.normal(0.0, seed_sd, size=(n_sims, n))
            b = rng.normal(eff, seed_sd, size=(n_sims, n))
            with np.errstate(all="ignore"):
                p = stats.ttest_ind(b, a, axis=1, equal_var=False).pvalue
            rows.append(dict(effect=float(eff), effect_sd=float(eff / seed_sd) if seed_sd else np.nan, n=int(n),
                             power=float(np.mean(p < alpha))))
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
    return dict(per_run=pr, mean_frac_peak=float(pr["frac_peak"].mean()), mean_frac_half=float(pr["frac_half"].mean()))


# ----------------------------------------------------------------------------------------- prereg
def prereg_numbers(results_dir="results", effect_sds=(0.5, 1, 2, 3), n_sims=2000):
    res, cur = load_results(results_dir)
    summ = baseline_summary(res)
    metric, why = propose_primary(summ)
    sd = np.nan
    if metric is not None:
        top = summ[summ["metric"] == metric].sort_values(["n", "train.tokens"], ascending=False).iloc[0]
        sd = float(top["sd"])
    unit = sd if sd == sd and sd > 0 else 1.0           # power depends only on effect/SD; fall back to SD units
    power = power_simulation(unit, [k * unit for k in effect_sds], n_sims=n_sims)
    return dict(baseline_summary=summ, primary=metric, primary_justification=why, primary_seed_sd=sd,
                power=power, plasticity=predict_window_from_plasticity(cur, results_df=res),
                n_runs=len(res), n_clean=int((res["corruption.kind"] == "clean").sum()) if len(res) else 0)


def _md(df, fmt="{:.3f}"):
    if df is None or len(df) == 0:
        return "_none_"
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def render_prereg_numbers(d):
    s = ["## Baseline summary (clean runs; seed SD, snr = |mean - chance| / SD)", "", _md(d["baseline_summary"]), "",
         "## Proposed primary metric (proposal only)", "", f"**{d['primary']}**: {d['primary_justification']}", ""]
    sd = d["primary_seed_sd"]
    s += ["## Power (simulated Welch t-test, equal n per arm, alpha 0.05)", "",
          f"Seed SD of primary metric: {sd:.4g}" if sd == sd else
          "Seed SD not yet estimable (need >=2 clean seeds); effects shown in SD units, which fully determines power.", ""]
    if len(d["power"]):
        pv = d["power"].pivot(index="effect_sd", columns="n", values="power").reset_index()
        pv.columns = ["effect (SD)"] + [f"n={c}" for c in pv.columns[1:]]
        s.append(_md(pv, "{:.2f}"))
    s += ["", "## Plasticity-predicted window", ""]
    p = d["plasticity"]
    if p is None:
        s.append("_no plasticity metric in curves yet_")
    else:
        s += [f"Mean frac of peak: {p['mean_frac_peak']:.3f}; mean frac where it first falls below 50% of peak: "
              f"{p['mean_frac_half']:.3f}", "", _md(p["per_run"])]
    return "\n".join(s) + "\n"

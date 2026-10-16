"""Run: .venv/bin/python -m pytest tests/test_analysis.py   (or python tests/test_analysis.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmcp import analysis as A, stats  # noqa: E402

REAL = Path(__file__).resolve().parents[1] / "results"


def synth(effects=None, sd=0.01, n_seeds=4, seed=1, doc_seed_offset=False):
    effects = effects or {"clean": 0.0, "shuffle_sentence": -0.05, "char_noise": -0.02}
    rng = np.random.default_rng(seed)
    seed_eff = rng.normal(0, sd, n_seeds)
    rows = []
    for kind, eff in effects.items():
        for s in range(n_seeds):
            rows.append({"run": f"{kind}_{s}", "status": "done", "corruption.kind": kind,
                         "corruption.param": 0.1 if kind == "char_noise" else None,
                         "corruption.window": None if kind == "clean" else [0.0, 0.5],
                         "train.seed": s, "model.size": "xs", "train.schedule": "const", "train.tokens": 1000,
                         "data.tokenizer_vocab": 8192,
                         "blimp_acc": 0.7 + eff + seed_eff[s] + rng.normal(0, sd / 3),
                         "induction_acc": 0.5 + rng.normal(0, 0.02), "val_loss": 4 + rng.normal(0, 0.01),
                         "long_range_gain": 0.3 + rng.normal(0, 0.01), "induction_gain": 1 + rng.normal(0, 0.05)})
    df = pd.DataFrame(rows)
    if doc_seed_offset:
        df["data.doc_order_seed"] = df["train.seed"] + 100
    return A.add_derived(df)


def test_mixed_model_recovers_effect():
    t = A.mixed_model(synth(), "blimp_acc")
    assert t.attrs["method"].startswith("mixedlm"), t.attrs
    r = t[t.condition.str.startswith("shuffle_sentence")].iloc[0]
    assert r.ci_lo < -0.05 < r.ci_hi and r.p < 0.001
    assert len(t) == 2 and (t.p_holm >= t.p).all()


def test_mixed_model_with_doc_order_vc():
    t = A.mixed_model(synth(doc_seed_offset=True), "blimp_acc")
    r = t[t.condition.str.startswith("shuffle_sentence")].iloc[0]
    assert r.ci_lo < -0.05 < r.ci_hi, t.attrs


def test_mixed_model_fallback_few_seeds():
    t = A.mixed_model(synth(n_seeds=2), "blimp_acc")
    assert t.attrs["method"] == "ols_hc3" and "seed" in t.attrs["warning"]
    assert len(t) == 2


def test_mixed_model_never_raises():
    one = synth(n_seeds=1)
    t = A.mixed_model(one, "blimp_acc")
    assert t.attrs["method"] in ("ols_hc3", "none")
    assert A.mixed_model(pd.DataFrame(), "x").attrs["warning"]
    assert A.mixed_model(synth(), "nope").attrs["method"] == "none"


def test_holm():
    p = [0.01, 0.04, 0.03, np.nan]
    h = A.holm(p)
    assert np.allclose(h[:3], [0.03, 0.06, 0.06]) and np.isnan(h[3])


def test_power_monotone():
    pw = A.power_simulation(1.0, [0, 1, 2, 4], n_sims=1500)
    p = pw.pivot(index="effect", columns="n", values="power")
    assert (p.diff(axis=0).iloc[1:] > -0.03).all().all()
    assert (p.diff(axis=1).iloc[:, 1:].loc[1:] > -0.03).all().all()
    assert p.loc[4, 8] > 0.95 and abs(p.loc[0, 8] - 0.05) < 0.03


def test_landmarks_planted():
    fr = np.linspace(0, 1, 11)
    acc = np.where(fr < 0.45, 0.5, 0.9)           # crosses midpoint 0.7 at frac 0.5 (index 5)
    acc[2] = 0.95                                  # transient early spike must not count
    cur = pd.DataFrame({"run": "r", "step": range(11), "tokens": fr * 1000, "frac": fr,
                        "blimp_acc": acc, "induction_acc": np.where(fr < 0.7, 0.0, 1.0),
                        "long_range_gain": fr})
    lm = A.landmarks(cur).set_index("metric")
    assert abs(lm.loc["blimp_acc", "frac"] - 0.5) < 1e-9
    assert abs(lm.loc["induction_acc", "frac"] - 0.7) < 1e-9
    assert abs(lm.loc["long_range_gain", "frac"] - 0.6) < 1e-9     # mid 0.5 -> first strictly above is 0.6


def test_dose_response_and_baseline():
    df = synth()
    rows = []
    for i, k in enumerate([0.25, 0.5, 1.0]):
        for s in range(3):
            rows.append({**df.iloc[0].to_dict(), "run": f"d{i}{s}", "corruption.kind": "char_noise",
                         "corruption.window": [0.0, 0.5], "corruption.intensity": k, "train.seed": s,
                         "blimp_acc": 0.7 - 0.1 * k + 0.001 * s})
    d = A.dose_response(pd.DataFrame(rows), "blimp_acc", "char_noise", [0.0, 0.5])
    assert abs(d["slope"] + 0.1) < 0.01 and d["n"] == 9 and d["r2"] > 0.95
    s = A.baseline_summary(df)
    assert (s.n == 4).all() and set(s.metric) >= {"blimp_acc"}
    m, why = A.propose_primary(s)
    assert m in A.ABILITY_METRICS and why
    assert A.dose_response(df, "blimp_acc", "shuffle_sentence", [0, 0.5])["warning"]


def test_bootstrap():
    x = np.random.default_rng(0).normal(5, 1, 400)
    lo, hi = stats.bootstrap_ci(x)
    assert lo < 5 < hi and hi - lo < 0.4
    est, dl, dh = stats.bootstrap_diff_ci(x, x + 0.5)
    assert abs(est + 0.5) < 1e-9 and dl <= est <= dh
    assert stats.bootstrap_ci(x) == stats.bootstrap_ci(x)           # deterministic


def test_real_results_graceful():
    if not (REAL / "results.jsonl").exists():
        return
    res, cur = A.load_results(REAL)
    assert "condition" in res and "clean" in set(res.condition)
    A.landmarks(cur)
    s = A.baseline_summary(res)
    m, why = A.propose_primary(s)                     # 1 clean seed -> None, with explanation
    assert isinstance(why, str)
    t = A.mixed_model(res, "blimp_acc")
    assert t.attrs["method"] in ("ols_hc3", "none")
    assert A.predict_window_from_plasticity(cur) is None
    d = A.prereg_numbers(REAL, n_sims=200)
    assert "Power" in A.render_prereg_numbers(d)


def test_run_all_dict_streams():
    import json
    import pytest
    cfgp = Path(__file__).resolve().parents[1] / "runs" / "_engineer_check_stats" / "config.json"
    tokp = list((Path(__file__).resolve().parents[1] / "data").glob("**/tokenizer.json"))
    if not cfgp.exists() or not tokp:
        pytest.skip("needs the smoke run dir and a tokenizer")
    from tokenizers import Tokenizer
    from lmcp import evals
    from lmcp.model import build_model
    cfg = json.loads(cfgp.read_text())["config"]
    cfg["eval"].update(val_tokens=6000, blimp_per_paradigm=2)
    tok = Tokenizer.from_file(str(tokp[0]))
    model = build_model(cfg, cfg["data"]["tokenizer_vocab"])
    rng = np.random.default_rng(0)
    mk = lambda: rng.integers(2, 8000, size=40_000).astype(np.uint16)
    out = evals.run_all(model, tok, {"clean": mk(), "nl": mk()}, cfg)
    for k in ("val_loss", "val_loss_ci_lo", "val_loss/clean", "val_loss/nl", "val_loss/nl_ci_hi", "long_range_gain_ci_lo",
              "blimp_acc_ci_hi", "induction_acc_ci_lo"):
        assert k in out, k
    assert out["val_loss"] == out["val_loss/clean"]
    old = evals.run_all(model, tok, mk(), cfg)
    assert "val_loss/clean" in old and "val_loss/nl" not in old


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

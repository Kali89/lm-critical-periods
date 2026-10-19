"""Run: .venv/bin/python -m pytest -q tests"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmcp import analysis as A, evals  # noqa: E402
from lmcp.model import build_model, causal_mask  # noqa: E402


def _allowed(m):
    return {(i, j) for i in range(m.shape[0]) for j in range(m.shape[1]) if float(m[i, j]) == 0.0}


def test_causal_mask_window():
    T = 6
    got = _allowed(causal_mask(T, window=3))
    assert got == {(i, j) for i in range(T) for j in range(T) if j <= i and i - j < 3}
    # compose with block
    got = _allowed(causal_mask(T, block=4, window=2))
    assert got == {(i, j) for i in range(T) for j in range(T) if j <= i and i // 4 == j // 4 and i - j < 2}
    # window=None unchanged
    assert np.array_equal(np.array(causal_mask(T)), np.array(causal_mask(T, None, None)))
    assert _allowed(causal_mask(T)) == {(i, j) for i in range(T) for j in range(T) if j <= i}
    assert _allowed(causal_mask(T, window=T)) == _allowed(causal_mask(T))


def _tiny():
    mx.random.seed(0)
    return build_model({"model": {"n_layer": 1, "d_model": 32, "n_head": 2}}, 64)


def test_val_loss_ctx_fresh_context():
    model = _tiny()
    L = 160
    stream = np.random.default_rng(0).integers(2, 64, size=4 * (L + 1) + 7).astype(np.int32)
    out = evals.val_loss(model, stream, L, 4 * L)
    for k in ("val_loss_ctx_full", "val_loss_ctx_32", "context_gain"):
        assert k in out and np.isfinite(out[k])
        assert f"{k}_ci_lo" in out and f"{k}_ci_hi" in out
    data = stream[: 4 * (L + 1)].reshape(4, L + 1)
    targets = [128, 136, 144, 152]
    nll = -np.array(evals._token_logprobs(model, mx.array(data)))
    assert abs(out["val_loss_ctx_full"] - nll[:, targets].mean()) < 1e-6
    fresh = []
    for b in range(4):
        for t in targets:
            snip = mx.array(data[b:b + 1, t - 31:t + 2])              # 32 context tokens + target
            assert snip.shape == (1, 33)
            fresh.append(-float(np.array(evals._token_logprobs(model, snip))[0, -1]))
    assert abs(out["val_loss_ctx_32"] - np.mean(fresh)) < 1e-6
    assert abs(out["context_gain"] - (out["val_loss_ctx_32"] - out["val_loss_ctx_full"])) < 1e-9
    for k in ("val_loss", "val_loss_pos_0_32", "val_loss_pos_32_128", "long_range_gain"):
        assert k in out


def test_val_loss_ctx_nan_when_short():
    model = _tiny()
    stream = np.random.default_rng(0).integers(2, 64, size=4 * 65 + 3).astype(np.int32)
    out = evals.val_loss(model, stream, 64, 4 * 64)
    for k in ("val_loss_ctx_full", "val_loss_ctx_32", "context_gain"):
        assert np.isnan(out[k])
    for k in ("val_loss", "val_loss_pos_0_32", "val_loss_pos_32_64", "long_range_gain"):
        assert k in out


def test_power_unequal_arms():
    pw = A.power_simulation(1.0, [0.5], n_sims=200, n_clean=6, n_arm=(3, 4, 5, 6))
    assert len(pw) == 4 * 3 and set(pw.n_arm) == {3, 4, 5, 6} and set(pw.n_clean) == {6}
    assert set(pw.sd_mult) == {1.0, 2.0, 3.0}
    assert ((pw.power >= 0) & (pw.power <= 1)).all()
    legacy = A.power_simulation(1.0, [0, 1], n_sims=100)
    assert len(legacy) == 2 * 6 and (legacy.n_clean == legacy.n_arm).all()


def test_prereg_without_ctx_metrics(tmp_path):
    d = A.prereg_numbers(tmp_path, n_sims=50)
    assert "not rescored" in A.render_prereg_numbers(d)


def test_collect_merges_rescore_add_only(tmp_path, monkeypatch):
    import json
    from lmcp import runs
    rd = tmp_path / "runs" / "r1"
    rd.mkdir(parents=True)
    (rd / "status.json").write_text(json.dumps({"status": "done"}))
    (rd / "config.json").write_text(json.dumps({"config": {"a": 1}, "info": {"n_params": 5}}))
    (rd / "metrics.jsonl").write_text(json.dumps({"val_loss": 3.0, "final": True}) + "\n")
    (rd / "rescore.jsonl").write_text(
        json.dumps({"which": "final", "val_loss": 9.0, "context_gain": 0.1, "tokens": 1, "rev": {}}) + "\n")
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runs, "RESULTS", tmp_path / "results")
    (row,) = runs.collect()
    assert row["val_loss"] == 3.0 and row["context_gain"] == 0.1 and row["rescored"] is True
    assert "which" not in row and "rev" not in row


def test_fresh_context_equals_sequence_prefix():
    model = _tiny()
    x = np.random.default_rng(1).integers(2, 64, size=40).astype(np.int32)
    full = -np.array(evals._token_logprobs(model, mx.array(x[None, :])))[0]
    snip = -np.array(evals._token_logprobs(model, mx.array(x[None, :33])))[0]
    assert full.shape[0] == 39 and snip.shape[0] == 32
    assert abs(full[31] - snip[-1]) < 1e-6


def test_power_confirmatory_shape_and_monotone():
    sd = {"val_loss_ctx_32": 0.01, "context_gain": 0.01}
    eff = {"val_loss_ctx_32": 0.016, "context_gain": 0.0175}
    out = A.power_confirmatory(sd, eff, n_sims=400)
    assert list(out.columns) == ["label", "endpoint", "analysis", "rho", "sd_mult", "power"]
    labels = [lab for lab, _ in A.H1_FAMILY]
    assert set(out["label"]) == set(labels) | {"all three"} and set(out["analysis"]) == {"paired", "welch"}
    assert set(out["endpoint"]) == set(sd) | {"all three"}
    assert len(out) == 4 * 2 * 3 * 3

    def pw(lab, rho, mult, an="paired"):
        return float(out[(out.label == lab) & (out.rho == rho) & (out.sd_mult == mult) & (out.analysis == an)].power.iloc[0])

    for lab in labels + ["all three"]:
        assert pw(lab, 0.5, 1.0) >= pw(lab, 0.5, 3.0)
        assert pw(lab, 0.5, 1.0) >= pw(lab, -0.5, 1.0)
    for an in ("paired", "welch"):
        assert pw("all three", 0.0, 1.0, an) <= min(pw(lab, 0.0, 1.0, an) for lab in labels) + 1e-12

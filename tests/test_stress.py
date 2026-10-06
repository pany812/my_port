"""Stress: stationary bootstrap, path statistics, crisis windows, weights, spec, runner, report."""

import datetime as dt
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.fixtures.synthetic import small_market
from tests.test_runner import SMALL
from workbench.allocators._solve import Infeasible
from workbench.allocators.naive import funded_weights
from workbench.evaluation import stats
from workbench.evaluation.report import build_report
from workbench.evaluation.stress import (
    BOOTSTRAP_TEST,
    PRESET_WINDOWS,
    WINDOW_TEST,
    bootstrap_table,
    corridor_quantiles,
    path_stats,
    path_window_view,
    stationary_bootstrap,
    stress_evidence,
    stress_weights,
    window_table,
)
from workbench.grid.runner import run_experiment
from workbench.grid.spec import SpecError, parse_spec
from workbench.policy.saa import SAA
from workbench.registry.store import CellRecord, Registry

DATA = small_market()
R = DATA.returns
SAA_ = SAA.placeholder()
SAA_W = SAA_.weights[list(R.columns)]


# --- bootstrap --------------------------------------------------------------------------------


def _continuation_rate(idx: np.ndarray, n: int) -> float:
    return float((idx[:, 1:] == (idx[:, :-1] + 1) % n).mean())


@pytest.mark.parametrize("mean_block", [1, 3, 7, 12])
def test_stationary_bootstrap_block_lengths(mean_block):
    n = 1000
    idx = stationary_bootstrap(n, 2000, 120, mean_block, np.random.default_rng(0))
    assert idx.shape == (2000, 120) and idx.min() >= 0 and idx.max() < n
    # a new block starts with probability 1/L (and continues by chance with probability 1/n)
    expected = (1 - 1 / mean_block) + (1 / mean_block) / n
    assert _continuation_rate(idx, n) == pytest.approx(expected, abs=0.005)
    again = stationary_bootstrap(n, 2000, 120, mean_block, np.random.default_rng(0))
    assert (idx == again).all()
    with pytest.raises(ValueError, match="mean_block"):
        stationary_bootstrap(n, 1, 1, 0.5, np.random.default_rng(0))


def test_bootstrap_keeps_the_cross_section():
    idx = stationary_bootstrap(len(R), 2000, 120, 7, np.random.default_rng(1))
    sims = R.to_numpy()[idx].reshape(-1, R.shape[1])
    sim_corr = np.corrcoef(sims, rowvar=False)
    assert np.abs(sim_corr - R.corr().to_numpy()).max() < 0.03
    assert np.abs(sims.mean(axis=0) - R.mean().to_numpy()).max() < 5e-4


def test_path_stats_match_the_ex_post_definitions():
    r = np.random.default_rng(2).normal(0.004, 0.04, size=(5, 60))
    st = path_stats(r, "M")
    for i in range(5):
        s = pd.Series(r[i])
        assert st["max_dd"][i] == pytest.approx(stats.max_drawdown(s), abs=1e-14)
        assert st["cdar95"][i] == pytest.approx(stats.cdar(s), abs=1e-14)
        assert st["cvar95"][i] == pytest.approx(stats.cvar(s), abs=1e-14)
        assert st["ann_return"][i] == pytest.approx(stats.ann_return(s, "M"), abs=1e-12)
        assert st["ann_vol"][i] == pytest.approx(stats.ann_vol(s, "M"), abs=1e-12)


def test_iid_bootstrap_reproduces_the_true_drawdown_distribution():
    """Known answer: i.i.d. normal history, block 1 -> max-drawdown distribution of the truth."""
    rng = np.random.default_rng(3)
    mu, sigma, horizon = 0.005, 0.04, 120
    hist = rng.normal(mu, sigma, size=20_000)
    idx = stationary_bootstrap(len(hist), 4000, horizon, 1, rng)
    boot = path_stats(hist[idx], "M")["max_dd"]
    truth = path_stats(rng.normal(mu, sigma, size=(4000, horizon)), "M")["max_dd"]
    for q in (0.5, 0.9):
        assert np.quantile(boot, q) == pytest.approx(np.quantile(truth, q), rel=0.04)


# --- weights and policy portfolios ------------------------------------------------------------


def test_funded_weights():
    pro = funded_weights(SAA_W.copy(), "CAND", 0.05, "pro_rata", SAA_.asset_class)
    assert pro.sum() == pytest.approx(1) and pro["CAND"] == pytest.approx(0.05)
    assert pro["GL_EQ"] == pytest.approx(0.95 * SAA_W["GL_EQ"])
    cash = funded_weights(SAA_W.copy(), "CAND", 0.05, "asset:CASH", SAA_.asset_class)
    assert cash["CASH"] == pytest.approx(0.0) and cash["GL_EQ"] == SAA_W["GL_EQ"]
    fi = funded_weights(SAA_W.copy(), "CAND", 0.10, "class:fixed_income", SAA_.asset_class)
    assert fi[["SE_GOV", "GL_IG_H", "HY"]].sum() == pytest.approx(0.30)
    with pytest.raises(Infeasible, match="exhausted"):
        funded_weights(SAA_W.copy(), "CAND", 0.10, "asset:CASH", SAA_.asset_class)


def test_stress_weights_merge_roles_and_corridor_quantiles():
    ws = stress_weights({"corridor_p25": 0.0, "corridor_median": 0.05, "corridor_p75": 0.1},
                        (0.05, 0.2))  # fmt: skip
    assert [(w.x, w.roles) for w in ws] == [
        (0.0, ("saa", "corridor_p25")), (0.05, ("corridor_median", "spec")),
        (0.1, ("corridor_p75",)), (0.2, ("spec",))]  # fmt: skip
    assert ws[1].subject == "x=0.0500"

    def rec(i, variant, day, status, x):
        return CellRecord(f"c{i}", i, f"k{i}", "a", {}, None, "cs", variant, dt.date(2020, 1, day),
                          status, weights=None if x is None else {"CAND": x})  # fmt: skip

    records = [
        rec(0, "full", 1, "ok", 0.9),  # an earlier date: ignored
        *[rec(i, "full", 2, "ok", x) for i, x in enumerate([0.0, 0.02, 0.04, 0.06], 1)],
        rec(5, "full", 2, "infeasible", None),
        rec(6, "live_only", 3, "ok", 0.5),
        replace(rec(7, "full", 2, "ok", 0.9), cell_index=-1),  # the SAA reference row
    ]
    q = corridor_quantiles(records, "CAND")
    assert q == pytest.approx({"corridor_p25": 0.015, "corridor_median": 0.03,
                               "corridor_p75": 0.045})  # fmt: skip
    assert corridor_quantiles([], "CAND") == {}


# --- crisis windows ---------------------------------------------------------------------------


def _crashed():
    r = R.copy()
    crash = (r.index >= "2015-01-01") & (r.index <= "2015-06-30")
    r.loc[crash, "CAND"] = -0.10
    return r


def _evidence(returns, weights, windows, funding="pro_rata", **kw):
    rows = stress_evidence(returns, DATA.backfilled, SAA_W, "CAND", SAA_.asset_class, funding,
                           stress_weights({}, weights), windows, "M", kw.get("n_paths", 300),
                           kw.get("horizon_years", 5), None, 7, "full")  # fmt: skip
    return pd.DataFrame([{"data_variant": "full", **r} for r in rows])


def test_crisis_window_with_an_injected_crash():
    ev = _evidence(
        _crashed(), (0.05, 0.10), (("crash", "2015-01", "2015-06"), ("y1990", "1990-01", "1990-12"))
    )
    t = window_table(ev, ["crash", "y1990"]).set_index(["window", "x"])
    crash = t.loc["crash"]
    assert (crash.n_periods == 6).all()
    assert crash.candidate_return.iloc[0] == pytest.approx(0.9**6 - 1)
    assert crash.loc[0.0, ["d_return", "d_max_dd"]].abs().max() == 0  # the SAA vs itself
    assert 0 < crash.loc[0.05, "d_max_dd"] < crash.loc[0.10, "d_max_dd"]  # deeper with weight
    assert crash.loc[0.05, "d_return"] < 0
    assert (t.loc["y1990"].note == "window not covered by the data").all()


def test_window_return_is_the_compounded_policy_portfolio():
    ev = _evidence(R, (0.05,), (("w", "2017-03", "2017-08"),))
    row = window_table(ev).set_index("x").loc[0.05]
    w = funded_weights(SAA_W.copy(), "CAND", 0.05, "pro_rata", None)
    r = R.loc["2017-03-01":"2017-08-31"] @ w
    assert row["return"] == pytest.approx(float((1 + r).prod() - 1))
    assert row["max_dd"] == pytest.approx(stats.max_drawdown(r))


def test_bootstrap_rows_pair_with_the_saa():
    ev = _evidence(R, (0.05, 0.10), (("w", "2017-03", "2017-08"),), n_paths=500)
    bt = bootstrap_table(ev).set_index("x")
    assert bt.loc[0.0, ["d_max_dd_median", "d_max_dd_p95", "p_worse_max_dd"]].abs().max() == 0
    assert (bt.n_paths == 500).all() and (bt.horizon_years == 5).all()
    assert bt.loc[0.0, "mean_block"] == 5  # ceil(120^(1/3))
    again = bootstrap_table(_evidence(R, (0.05, 0.10), (("w", "2017-03", "2017-08"),),
                                      n_paths=500)).set_index("x")  # fmt: skip
    pd.testing.assert_frame_equal(bt, again)  # seeded


def test_unfundable_weight_is_noted_not_dropped():
    ev = _evidence(R, (0.10,), (("w", "2017-03", "2017-08"),), funding="asset:CASH")
    bt = bootstrap_table(ev)
    assert "exhausted" in bt.set_index("x").loc[0.10, "note"]
    assert set(ev.test) == {WINDOW_TEST + "w", BOOTSTRAP_TEST}


# --- spec -------------------------------------------------------------------------------------


def _raw(stress):
    raw = yaml.safe_load(SMALL)
    raw["stress"] = stress
    return raw


def test_spec_preset_hash_and_round_trip():
    base = parse_spec(SMALL)
    assert base.stress is None and "stress" not in base.canonical()
    s = parse_spec(_raw({"weights": [0.05]}))
    assert s.stress.windows == tuple((n, a, b) for n, (a, b) in PRESET_WINDOWS.items())
    assert s.stress.n_paths == 2000 and s.stress.block is None
    assert s.spec_hash != base.spec_hash
    assert parse_spec(s.yaml_text).spec_hash == s.spec_hash
    explicit = parse_spec(_raw({"windows": {n: list(v) for n, v in PRESET_WINDOWS.items()},
                                "weights": [0.05]}))  # fmt: skip
    assert explicit.spec_hash == s.spec_hash  # the preset hashes as its dates


@pytest.mark.parametrize(
    "stress,match",
    [
        ({"windows": {"bad name": ["2010-01", "2010-02"]}}, "letters, digits"),
        ({"windows": {"w": ["2010-05", "2010-02"]}}, "after end"),
        ({"windows": {"w": ["2010-05"]}}, r"\[start, end\]"),
        ({"windows": {}}, "non-empty mapping"),
        ({"weights": [0.0]}, r"in \(0, 1\)"),
        ({"bootstrap": {"n_paths": 50}}, ">= 100"),
        ({"bootstrap": {"block": 0}}, "block"),
        ({"bootstrap": {"seed": 1}}, "unknown keys"),
        ({"shocks": {}}, "unknown keys"),
    ],
)
def test_spec_errors(stress, match):
    with pytest.raises(SpecError, match=match):
        parse_spec(_raw(stress))


# --- runner and report ------------------------------------------------------------------------


def _run(tmp_path, mode="walk_forward", allocators=None):
    raw = _raw(
        {
            "windows": {
                "early": ["2012-01", "2012-12"],
                "mid": ["2017-01", "2017-06"],
                "late": ["2020-06", "2021-06"],
            },
            "weights": [0.05],
            "bootstrap": {"n_paths": 200, "horizon_years": 5},
        }
    )
    raw.update(backtest={"mode": mode}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"})  # fmt: skip
    if allocators is not None:
        raw["grid"]["allocators"] = allocators
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    return reg, run_experiment(parse_spec(raw), reg)


def test_walk_forward_paths_in_windows(tmp_path):
    reg, s = _run(tmp_path, allocators=[{"type": "static_saa"}])
    pv = path_window_view(reg, s.experiment_id, parse_spec(
        reg.experiment(s.experiment_id)["spec_yaml"]).stress.windows).set_index(
        ["data_variant", "window"])  # fmt: skip
    mid = pv.loc[("full", "mid")]
    assert mid.n_periods == 6 and mid.n_configs == 1
    # the static SAA configuration's path is the SAA reference path: every difference is 0
    assert mid[["d_return_median", "d_return_p10", "d_max_dd_p90"]].abs().max() == 0
    assert "outside the out-of-sample period" in pv.loc[("full", "early")].note
    assert pv.loc[("full", "late")].note.startswith("partial")


def test_runner_stores_stress_and_the_report_shows_it(tmp_path):
    reg, s = _run(tmp_path)
    ev = reg.evidence(s.experiment_id)
    st = ev[ev.test.str.startswith(("stress_window", "stress_bootstrap"))]
    assert set(st.data_variant) == {"full", "live_only"}
    roles = {r for x in st.extra_json.map(json.loads) for r in x.get("roles", [])}
    assert {"saa", "corridor_median", "spec"} <= roles
    md = build_report(reg, s.experiment_id).summary_md
    for heading in (
        "## Stress",
        "### Crisis windows: SAA plus the candidate",
        "### Crisis windows: walk-forward paths",
        "### Block-bootstrap paths",
    ):
        assert heading in md  # fmt: skip
    assert "Synthetic data: these dates carry no real crisis" in md
    assert "## Evidence net of search" in md  # stress rows do not disturb the evidence section


def test_in_sample_mode_has_windows_and_bootstrap_but_no_paths(tmp_path):
    reg, s = _run(tmp_path, mode="in_sample")
    md = build_report(reg, s.experiment_id).summary_md
    assert "### Block-bootstrap paths" in md
    assert "### Crisis windows: walk-forward paths" not in md

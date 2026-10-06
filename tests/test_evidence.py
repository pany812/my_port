"""Evidence stored by the runner and shown in the report."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_runner import SMALL
from workbench.evaluation.evidence import boot_seed, path_evidence, spanning_evidence
from workbench.evaluation.report import build_report
from workbench.grid.runner import run_experiment
from workbench.grid.spec import parse_spec
from workbench.registry.store import Registry


@pytest.fixture(scope="module")
def wf(tmp_path_factory):
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "Q"})  # fmt: skip
    reg = Registry(f"sqlite:///{tmp_path_factory.mktemp('ev') / 'r.db'}")
    return reg, run_experiment(parse_spec(raw), reg)


def test_rows_per_config_grid_and_candidate(wf):
    reg, s = wf
    ev = reg.evidence(s.experiment_id)
    full = ev[ev.data_variant == "full"]
    configs = set(reg.cells(s.experiment_id)["config_id"])
    for test in ("sharpe_diff", "dsr"):
        assert set(full.loc[full.test == test, "subject"]) == configs
    assert list(full.loc[full.subject == "grid", "test"]) == ["pbo"]
    assert set(full.loc[full.subject == "candidate", "test"]) == {
        "spanning_alpha", "spanning_alpha_robust"
    }  # CASH is the SAA's riskless asset  # fmt: skip
    assert set(ev.data_variant) == {"full", "live_only"}


def test_dsr_counts_every_configuration_and_bh_is_stored(wf):
    reg, s = wf
    ev = reg.evidence(s.experiment_id)
    full = ev[ev.data_variant == "full"]
    n_configs = reg.cells(s.experiment_id)["config_id"].nunique()
    dsr_extra = full[full.test == "dsr"]["extra_json"].map(json.loads)
    assert (dsr_extra.map(lambda e: e["n_trials"]) == n_configs).all()
    sd = full[full.test == "sharpe_diff"]
    extra = sd["extra_json"].map(json.loads)
    p = sd["p_value"].to_numpy(float)
    p_bh = extra.map(lambda e: e["p_bh"]).to_numpy(float)
    ok = np.isfinite(p)
    assert (p_bh[ok] >= p[ok] - 1e-12).all()  # adjustment never lowers a p-value
    assert (extra.map(lambda e: e["excess_over"]) == "CASH").all()


def test_static_saa_flagged_identical_not_zero(wf):
    reg, s = wf
    ev = reg.evidence(s.experiment_id)
    cells = reg.cells(s.experiment_id)
    static = cells.loc[cells.allocator == "static_saa", "config_id"].iloc[0]
    row = ev[(ev.subject == static) & (ev.test == "sharpe_diff") & (ev.data_variant == "full")]
    assert pd.isna(row["p_value"].iloc[0])
    assert json.loads(row["extra_json"].iloc[0])["note"] == "identical to SAA path"


def test_evidence_deterministic(wf, tmp_path):
    reg, s = wf
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "Q"})  # fmt: skip
    reg2 = Registry(f"sqlite:///{tmp_path / 'r2.db'}")
    s2 = run_experiment(parse_spec(raw), reg2)
    pd.testing.assert_frame_equal(reg.evidence(s.experiment_id), reg2.evidence(s2.experiment_id))
    assert boot_seed(42, "abc") == boot_seed(42, "abc") != boot_seed(43, "abc")


def test_report_section(wf):
    reg, s = wf
    md = build_report(reg, s.experiment_id).summary_md
    assert "## Evidence net of search" in md
    assert "Disclosures, not gates" in md
    assert "excess returns over the riskless asset CASH" in md
    assert "Sample too short" in md  # live-only OOS is short


def test_in_sample_mode_has_no_path_tests(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(SMALL), reg)  # SMALL is in-sample
    ev = reg.evidence(s.experiment_id)
    assert set(ev.subject) == {"candidate", "profile:candidate", "profile:saa"}
    assert not ev.test.isin(["sharpe_diff", "dsr", "pbo"]).any()


def test_spanning_without_riskless_uses_hk_and_kz():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0.005, 0.03, (120, 4)), columns=["A", "B", "C", "CAND"])
    tests = {row["test"] for row in spanning_evidence(r, "CAND", "M")}
    assert tests == {"spanning_hk", "spanning_kz_f1", "spanning_kz_f2", "spanning_hk_robust"}
    assert spanning_evidence(r.iloc[:5], "CAND", "M") == []  # too short to estimate


def test_sharpe_uses_excess_returns():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-31", periods=100, freq="ME")
    saa = pd.Series(rng.normal(0.004, 0.02, 100), index=idx)
    cashy = pd.Series(0.0017 + rng.normal(0, 0.001, 100), index=idx)  # cash-like path
    rf = pd.Series(0.0017, index=idx, name="CASH")
    raw = path_evidence({"c": cashy}, saa, "M", 1, riskless=0.0, n_boot=99)
    exc = path_evidence({"c": cashy}, saa, "M", 1, riskless=rf, n_boot=99)
    sr_raw = json.loads(raw[0]["extra_json"])["sr"]
    sr_exc = json.loads(exc[0]["extra_json"])["sr"]
    assert sr_raw > 1.0 and abs(sr_exc) < 0.3  # raw Sharpe of a cash-like path is inflated


def test_path_tests_not_computed_on_very_short_samples():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2026-01-31", periods=9, freq="ME")
    saa = pd.Series(rng.normal(0.004, 0.02, 9), index=idx)
    paths = {"a": saa + rng.normal(0, 0.01, 9), "b": saa + rng.normal(0.01, 0.01, 9)}
    rows = path_evidence(paths, saa, "M", 1, n_boot=99)
    assert {r["test"] for r in rows} == {"sharpe_diff", "dsr", "pbo"}
    assert all(r["statistic"] is None and r["p_value"] is None for r in rows)
    assert all("too few observations (9 < 24)" in r["extra_json"] for r in rows)

import json

import pandas as pd
import pytest
import yaml

from tests.fixtures.synthetic import small_market
from workbench.data.align import align_history, data_vintage
from workbench.grid.runner import run_experiment, select_window
from workbench.grid.spec import WindowSpec, parse_spec
from workbench.registry.store import Registry

SMALL = """
experiment: runner_test
seed: 7
data:
  source: synthetic
  frequency: M
  start: 2011-01
  end: 2020-12
  base_currency: SEK
  hedging: synthetic
  candidate: CAND
  synthetic: {live_start: 2016-01}
saa: {version: example}
window: {kind: rolling, periods: 60}
backtest: {mode: in_sample}
grid:
  estimators: [{method_mu: hist, method_cov: ledoit}]
  allocators:
    - {type: static_saa}
    - {type: saa_plus, x: [0.05, 0.15]}
    - {type: equal_weight}
    - {type: riskfolio_mean_risk, rm: [MV, CVaR], obj: [MinRisk]}
    - {type: riskfolio_hc, model: [HRP]}
  constraint_sets:
    - {name: cap10, candidate_cap: 0.10}
"""


@pytest.fixture()
def reg(tmp_path):
    return Registry(f"sqlite:///{tmp_path / 'reg.db'}")


def test_every_cell_recorded_with_status(reg):
    spec = parse_spec(SMALL)
    summary = run_experiment(spec, reg)
    cells = reg.cells(summary.experiment_id)
    # 7 configs x 2 variants (candidate is backfilled before 2016-01)
    assert summary.n_cells == len(cells) == 14
    assert set(cells["data_variant"]) == {"full", "live_only"}
    assert cells["status"].notna().all()
    by = cells.set_index(["data_variant", "allocator", "params_json"])["status"]
    # forced failures are recorded, not dropped
    assert by[("full", "saa_plus", '{"funding": "pro_rata", "x": 0.15}')] == "infeasible"
    assert by[("full", "equal_weight", "{}")] == "infeasible"  # 1/9 > 10% cap
    assert summary.status_counts == cells["status"].value_counts().to_dict()
    # weights only for ok cells, each summing to 1
    w = reg.weights_wide(summary.experiment_id)
    ok = cells.loc[cells["status"] == "ok", "cell_id"]
    assert set(w.index) == set(ok)
    assert (w.sum(axis=1) - 1).abs().max() < 1e-9


def test_experiment_row(reg):
    spec = parse_spec(SMALL)
    s = run_experiment(spec, reg)
    exp = reg.experiment(s.experiment_id)
    assert exp["spec_hash"] == spec.spec_hash
    assert exp["spec_yaml"] == SMALL
    assert exp["candidate_id"] == "CAND" and exp["seed"] == 7
    assert exp["data_vintage"] == s.data_vintage


def test_live_only_window_and_short_history(reg):
    raw = yaml.safe_load(SMALL)
    raw["window"]["periods"] = 72  # live history is 60 months
    s = run_experiment(parse_spec(raw), reg)
    cells = reg.cells(s.experiment_id)
    live = cells[cells["data_variant"] == "live_only"]
    assert (live["status"] == "exception").all()
    assert live["message"].str.contains("insufficient history: 60 < 72").all()
    assert (cells.loc[cells["data_variant"] == "full", "status"] != "exception").all()


def test_if_exists_modes(reg):
    spec = parse_spec(SMALL)
    first = run_experiment(spec, reg)
    assert run_experiment(spec, reg).skipped
    with pytest.raises(ValueError, match="already exists"):
        run_experiment(spec, reg, if_exists="error")
    replaced = run_experiment(spec, reg, if_exists="replace")
    assert not replaced.skipped and replaced.experiment_id == first.experiment_id
    assert len(reg.cells(first.experiment_id)) == 14


def test_ids_deterministic_and_seed_sensitive(tmp_path):
    a = run_experiment(parse_spec(SMALL), Registry(f"sqlite:///{tmp_path / 'a.db'}"))
    b = run_experiment(parse_spec(SMALL), Registry(f"sqlite:///{tmp_path / 'b.db'}"))
    assert a.experiment_id == b.experiment_id and a.data_vintage == b.data_vintage
    raw = yaml.safe_load(SMALL)
    raw["seed"] = 8
    c = run_experiment(parse_spec(raw), Registry(f"sqlite:///{tmp_path / 'c.db'}"))
    assert c.data_vintage != a.data_vintage and c.experiment_id != a.experiment_id


def test_construction_failure_is_a_cell(reg):
    raw = yaml.safe_load(SMALL)
    raw["grid"]["allocators"] = [{"type": "saa_plus", "x": [0.05, 1.5]}]
    s = run_experiment(parse_spec(raw), reg)
    cells = reg.cells(s.experiment_id)
    bad = cells[cells["params_json"].str.contains("1.5")]
    assert (bad["status"] == "exception").all()
    assert bad["message"].str.contains("x must be in").all()


def test_diagnostics_persisted(reg):
    s = run_experiment(parse_spec(SMALL), reg)
    cells = reg.cells(s.experiment_id)
    row = cells[(cells["allocator"] == "equal_weight") & (cells["data_variant"] == "full")].iloc[0]
    diag = json.loads(row["diagnostics_json"])
    assert any("CAND" in v for v in diag["violations"])
    assert set(diag["rejected_weights"]) >= {"CAND", "GL_EQ"}


def test_select_window():
    r = small_market().returns
    assert len(select_window(r, WindowSpec("rolling", periods=60))) == 60
    assert len(select_window(r, WindowSpec("expanding", min_periods=24))) == 120
    with pytest.raises(ValueError, match="insufficient"):
        select_window(r, WindowSpec("rolling", periods=121))


def test_align_history_and_vintage():
    d = small_market(live_start="2016-01", backfill=False)
    a = align_history(d)
    assert a.returns.index[0] == pd.Timestamp("2016-01-31") and not a.returns.isna().any().any()
    d2 = small_market()
    v = data_vintage(d2)
    r = d2.returns.copy()
    r.iloc[5, 3] += 1e-12
    assert data_vintage(type(d2)(r, d2.freq, d2.candidate, d2.asset_class, d2.backfilled)) != v
    gap = d2.returns.copy()
    gap.iloc[50, 0] = float("nan")
    with pytest.raises(ValueError, match="missing returns"):
        align_history(type(d2)(gap, d2.freq, d2.candidate, d2.asset_class, d2.backfilled))

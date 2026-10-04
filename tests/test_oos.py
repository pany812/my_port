"""Walk-forward runner integration and the out-of-sample table."""

import json

import pandas as pd
import pytest
import yaml

from tests.test_runner import SMALL
from workbench.evaluation.corridor import corridor
from workbench.evaluation.oos import oos_table
from workbench.evaluation.stats import ann_return
from workbench.grid.runner import run_experiment
from workbench.grid.spec import SpecError, parse_spec
from workbench.registry.store import REFERENCE_ALLOCATOR, Registry


def _wf_spec(**over):
    raw = yaml.safe_load(SMALL)
    raw["backtest"] = {"mode": "walk_forward"}
    raw["window"] = {"kind": "rolling", "periods": 36}
    raw["rebalance"] = {"kind": "calendar", "every": "Q"}
    raw.update(over)
    return parse_spec(raw)


@pytest.fixture(scope="module")
def wf(tmp_path_factory):
    reg = Registry(f"sqlite:///{tmp_path_factory.mktemp('wf') / 'r.db'}")
    return reg, run_experiment(_wf_spec(), reg)


def test_one_cell_per_config_and_rebalance(wf):
    reg, s = wf
    cells = reg.cells(s.experiment_id)
    full = cells[cells.data_variant == "full"]
    dates = sorted(full["window_end"].unique())
    # rolling 36 on 2011-01..2020-12 quarterly: 2013-12 .. 2020-09 = 28 dates; 7 configs
    assert len(dates) == 28 and len(full) == 28 * 7
    assert full.groupby("config_id").size().eq(28).all()
    live = cells[cells.data_variant == "live_only"]  # live from 2016-01: first fit 2018-12
    assert str(min(live["window_end"])) == "2018-12-31"
    assert s.n_cells == len(cells)


def test_paths_stored_for_every_config_and_saa(wf):
    reg, s = wf
    oos = reg.oos_returns(s.experiment_id, include_reference=True)
    full = oos[oos.data_variant == "full"]
    assert full["config_id"].nunique() == 7 + 1
    assert REFERENCE_ALLOCATOR in set(full["config_id"])
    assert REFERENCE_ALLOCATOR not in set(reg.oos_returns(s.experiment_id)["config_id"])
    per = full.groupby("config_id")["date"].agg(["min", "max", "size"])
    assert per["min"].nunique() == 1 and per["max"].nunique() == 1  # same span for all
    assert str(per["min"].iloc[0]) == "2014-01-31" and str(per["max"].iloc[0]) == "2020-12-31"


def test_oos_table(wf):
    reg, s = wf
    t = oos_table(reg, s.experiment_id)
    full = t[t.data_variant == "full"]
    assert full.iloc[0]["row"] == "SAA" and full.iloc[0]["te_vs_saa"] == pytest.approx(0, abs=1e-12)
    assert len(full) == 8
    cells = full[full.row == "cell"]
    assert (cells["n_rebalances"] == 28).all()
    # static SAA rebalanced quarterly is exactly the SAA reference path
    static = cells[cells.label == "static_saa"].iloc[0]
    assert static.te_vs_saa == pytest.approx(0, abs=1e-12)
    assert static.turnover_ann == pytest.approx(full.iloc[0].turnover_ann)  # same rebalancing
    # failures are counted: saa_plus 0.15 breaks the 10% cap at every rebalance
    plus15 = cells[cells.label == "saa_plus x=0.15"].iloc[0]
    assert plus15.n_failed_rebalances == 28 and plus15.te_vs_saa == pytest.approx(0, abs=1e-12)
    # ann_return recomputes from the stored path
    path = reg.oos_returns(s.experiment_id, include_reference=True)
    saa_path = path[(path.config_id == REFERENCE_ALLOCATOR) & (path.data_variant == "full")]
    assert full.iloc[0].ann_return == pytest.approx(
        ann_return(saa_path["portfolio_return"].reset_index(drop=True), "M")
    )


def test_corridor_per_rebalance_date(wf):
    reg, s = wf
    c = corridor(reg, s.experiment_id)
    cw = c[(c.measure == "capital_weight") & (c.data_variant == "full")]
    assert len(cw) == 28 and (cw["n_cells"] == 7).all()
    assert (cw["n_infeasible"] >= 2).all()  # saa_plus 0.15 and equal_weight every date


def test_reference_cells_per_rebalance(wf):
    reg, s = wf
    ref = reg.reference(s.experiment_id)
    assert len(ref[ref.data_variant == "full"]) == 28


def test_construction_failure_in_walk_forward_holds_saa(tmp_path):
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"})  # fmt: skip
    raw["grid"]["allocators"] = [{"type": "saa_plus", "x": [1.5]}]
    reg = Registry(f"sqlite:///{tmp_path / 'c.db'}")
    s = run_experiment(parse_spec(raw), reg)
    cells = reg.cells(s.experiment_id)
    assert (cells["status"] == "exception").all()
    assert cells["message"].str.contains("x must be in").all()
    t = oos_table(reg, s.experiment_id)
    row = t[(t.row == "cell") & (t.data_variant == "full")].iloc[0]
    assert row.te_vs_saa == pytest.approx(0, abs=1e-12)  # held SAA throughout


def test_too_short_for_any_rebalance(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 's.db'}")
    s = run_experiment(_wf_spec(window={"kind": "rolling", "periods": 119}), reg)
    cells = reg.cells(s.experiment_id)
    assert (cells["status"] == "exception").all()
    assert cells["message"].str.contains("no rebalance date").all()
    assert reg.oos_returns(s.experiment_id).empty


def test_backtest_start_and_spec_validation():
    spec = _wf_spec(backtest={"mode": "walk_forward", "start": "2019-01"})
    assert spec.backtest.start == "2019-01"
    with pytest.raises(SpecError, match="backtest.mode"):
        _wf_spec(backtest={"mode": "rolling"})
    with pytest.raises(SpecError, match="rebalance.every"):
        _wf_spec(rebalance={"every": "W"})


def test_metrics_and_diagnostics_per_rebalance(wf):
    reg, s = wf
    m = reg.metrics(s.experiment_id)
    cells = reg.cells(s.experiment_id)
    ok = cells[cells.status == "ok"]
    assert set(m["cell_id"]) == set(ok["cell_id"])
    bad = cells[cells.status == "infeasible"].iloc[0]
    assert json.loads(bad["diagnostics_json"])["violations"]
    assert isinstance(pd.Timestamp(bad["window_end"]), pd.Timestamp)

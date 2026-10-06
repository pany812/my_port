import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_runner import SMALL
from workbench.evaluation.corridor import corridor
from workbench.evaluation.expost import cell_label, expost_table
from workbench.grid.runner import run_experiment
from workbench.grid.spec import parse_spec
from workbench.registry.store import REFERENCE_ALLOCATOR, CellRecord, Registry

END = dt.date(2026, 9, 30)
SPEC_YAML = """
experiment: hand
seed: 1
data: {source: sql, frequency: M, start: 2006-01, end: 2026-09, base_currency: SEK,
       hedging: none, candidate: CAND}
saa: {version: example}
window: {kind: rolling, periods: 12}
grid:
  allocators: [{type: static_saa}]
  constraint_sets: [{name: cs}]
risk_lenses: [MV]
"""


def _cell(i, status, w_cand=None, share=None, allocator="riskfolio_mean_risk", rm="MV"):
    ok = status == "ok"
    return CellRecord(
        cell_id=f"c{i}",
        cell_index=i,
        config_id=f"{i:016d}",
        allocator=allocator,
        params={"rm": rm} if allocator == "riskfolio_mean_risk" else {},
        estimator={"method_mu": "hist", "method_cov": "ledoit"}
        if allocator == "riskfolio_mean_risk"
        else None,
        constraint_set="cs",
        data_variant="full",
        window_end=END,
        status=status,
        weights={"CAND": w_cand, "X": 1 - w_cand} if ok else None,
        metrics=[("candidate_risk_share", "MV", share)] if ok and share is not None else [],
    )


@pytest.fixture()
def hand_registry(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'h.db'}")
    cells = [
        _cell(0, "ok", 0.00, 0.000),
        _cell(1, "ok", 0.02, 0.010),
        _cell(2, "ok", 0.04, 0.020),
        _cell(3, "ok", 0.10, None),  # metric missing
        _cell(4, "ok", 0.06, 0.030, allocator="static_saa"),
        _cell(5, "infeasible"),
        _cell(6, "exception"),
        _cell(7, "solver_error", allocator="static_saa"),
        CellRecord(  # the SAA reference must never be counted
            cell_id="ref",
            cell_index=-1,
            config_id=REFERENCE_ALLOCATOR,
            allocator=REFERENCE_ALLOCATOR,
            params={},
            estimator=None,
            constraint_set="",
            data_variant="full",
            window_end=END,
            status="ok",
            weights={"CAND": 0.0, "X": 1.0},
            metrics=[("candidate_risk_share", "MV", 0.0)],
        ),  # fmt: skip
    ]
    reg.write_experiment(
        {
            "experiment_id": "e",
            "name": "hand",
            "spec_yaml": SPEC_YAML,
            "spec_hash": "h",
            "data_vintage": "v",
            "saa_version": "example",
            "candidate_id": "CAND",
            "riskfolio_version": "7.4.0",
            "seed": 1,
            "created_at": dt.datetime(2026, 10, 4, tzinfo=dt.UTC),
        },
        cells,
    )
    return reg


def test_corridor_counts_and_quantiles(hand_registry):
    c = corridor(hand_registry, "e").set_index("measure")
    cw = c.loc["capital_weight"]
    assert (cw.n_cells, cw.n_ok, cw.n_infeasible, cw.n_solver_error, cw.n_exception) == (
        8, 5, 1, 1, 1
    )  # fmt: skip
    vals = np.array([0.00, 0.02, 0.04, 0.10, 0.06])
    assert cw["median"] == pytest.approx(np.median(vals))
    assert cw.p10 == pytest.approx(np.quantile(vals, 0.10))
    assert cw.p90 == pytest.approx(np.quantile(vals, 0.90))
    assert cw["share_below_0.25pct"] == pytest.approx(1 / 5)
    assert cw.n_metric_missing == 0

    rs = c.loc["risk_share:MV"]
    assert rs.n_ok == 5 and rs.n_metric_missing == 1
    assert rs["median"] == pytest.approx(np.median([0.0, 0.01, 0.02, 0.03]))


def test_corridor_group_by_family(hand_registry):
    c = corridor(hand_registry, "e", by=["family"])
    cw = c[c.measure == "capital_weight"].set_index("family")
    assert cw.loc["mean_risk", "n_cells"] == 6 and cw.loc["mean_risk", "n_ok"] == 4
    assert cw.loc["naive", "n_cells"] == 2 and cw.loc["naive", "n_solver_error"] == 1
    assert cw["n_cells"].sum() == 8  # reference excluded from every group
    with pytest.raises(ValueError, match="unknown group keys"):
        corridor(hand_registry, "e", by=["colour"])


def test_reference_excluded_from_reads_but_available(hand_registry):
    assert REFERENCE_ALLOCATOR not in set(hand_registry.cells("e")["allocator"])
    assert "ref" not in set(hand_registry.weights("e")["cell_id"])
    assert "ref" not in set(hand_registry.metrics("e")["cell_id"])
    assert list(hand_registry.reference("e")["cell_id"]) == ["ref"]
    assert len(hand_registry.cells("e", include_reference=True)) == 9


def test_all_failed_group_gives_nan_stats(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'f.db'}")
    reg.write_experiment(
        {"experiment_id": "e", "name": "f", "spec_yaml": SPEC_YAML, "spec_hash": "h",
         "data_vintage": "v", "saa_version": "example", "candidate_id": "CAND",
         "riskfolio_version": "7.4.0", "seed": 1,
         "created_at": dt.datetime(2026, 10, 4, tzinfo=dt.UTC)},
        [_cell(0, "infeasible"), _cell(1, "exception")],
    )  # fmt: skip
    cw = corridor(reg, "e").set_index("measure").loc["capital_weight"]
    assert cw.n_cells == 2 and cw.n_ok == 0 and np.isnan(cw["median"])


def test_runner_writes_reference_and_metrics(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(SMALL), reg)
    ref = reg.reference(s.experiment_id)
    assert set(ref["data_variant"]) == {"full", "live_only"} and (ref["cell_index"] == -1).all()
    assert s.n_cells == 14  # reference rows are not grid cells
    m = reg.metrics(s.experiment_id)
    ok = reg.cells(s.experiment_id).query("status == 'ok'")["cell_id"]
    assert set(m["cell_id"]) == set(ok)  # every ok cell has metrics, failed cells none
    e = expost_table(reg, s.experiment_id)
    saa = e[e.row == "SAA"]
    assert len(saa) == 2 and (saa["te_vs_saa"].abs() < 1e-12).all()
    assert (e.groupby("data_variant").head(1)["row"] == "SAA").all()  # SAA row first


def test_live_only_identical_when_window_is_all_live(tmp_path):
    """Window (60) lies inside the live history (60 months from 2016-01): variants coincide."""
    reg = Registry(f"sqlite:///{tmp_path / 'l.db'}")
    s = run_experiment(parse_spec(SMALL), reg)
    c = corridor(reg, s.experiment_id).drop(columns="data_variant")
    full, live = c.iloc[: len(c) // 2], c.iloc[len(c) // 2 :]
    pd.testing.assert_frame_equal(full.reset_index(drop=True), live.reset_index(drop=True))
    raw = yaml.safe_load(SMALL)
    raw["window"] = {"kind": "expanding", "min_periods": 24}  # full now includes backfill
    s2 = run_experiment(parse_spec(raw), reg)
    c2 = corridor(reg, s2.experiment_id)
    m = c2[c2.measure == "risk_share:MV"].set_index("data_variant")["median"]
    assert m["full"] != m["live_only"]


def test_cell_label():
    assert cell_label("riskfolio_hc", '{"model": "HRP"}', float("nan")) == "riskfolio_hc model=HRP"
    assert cell_label("saa_plus", '{"funding": "pro_rata", "x": 0.05}', None) == "saa_plus x=0.05"
    assert cell_label("riskfolio_mean_risk", '{"rm": "MV"}',
                      '{"method_mu": "JS", "method_cov": "gerber1"}'
                      ) == "riskfolio_mean_risk rm=MV [JS/gerber1]"  # fmt: skip

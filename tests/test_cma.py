"""CMAs: spec section, point-in-time vectors, hashing, the ``cma`` mean method and the runner."""

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.fixtures.synthetic import fit_context, small_market
from tests.test_runner import SMALL
from workbench.allocators.riskfolio_hc import RiskfolioHC
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.allocators.skfolio_mr import SkfolioMeanRisk
from workbench.data.cma import CMA, CMAVector
from workbench.data.synthetic import PLACEHOLDER_SAA
from workbench.evaluation.report import build_report
from workbench.grid.runner import run_experiment
from workbench.grid.spec import SpecError, parse_spec
from workbench.registry.store import Registry

DATA = small_market()
R = DATA.returns
ASSETS = list(R.columns)


def _raw(**cma):
    raw = yaml.safe_load(SMALL)
    raw["grid"]["estimators"] = [{"method_mu": "hist", "method_cov": "ledoit"},
                                 {"method_mu": "cma", "method_cov": "ledoit"}]  # fmt: skip
    raw["cma"] = cma or {"version": "placeholder"}
    return raw


def _vectors(*pairs):
    """[(effective, {asset: r})] -> inline spec vectors over all assets (default 4%)."""
    out = []
    for eff, over in pairs:
        r = {a: 0.04 for a in ASSETS} | over
        out.append({"effective": eff, "returns_annual": r})
    return out


# --- spec ------------------------------------------------------------------------------------


def test_placeholder_is_the_synthetic_truth():
    spec = parse_spec(_raw())
    (v,) = spec.cma.vectors
    assert v.effective == spec.data.start == "2011-01"
    expected = PLACEHOLDER_SAA["mu_annual"].to_dict() | {"CAND": spec.data.synthetic.mu_annual}
    assert v.returns_annual == pytest.approx(expected)


def test_hash_covers_cma_values_and_round_trips():
    base = parse_spec(SMALL)
    assert "cma" not in base.canonical()  # existing specs keep their hash
    a = parse_spec(_raw(version="house", vectors=_vectors(("2011-01", {}))))
    b = parse_spec(_raw(version="house", vectors=_vectors(("2011-01", {"CAND": 0.05}))))
    assert a.spec_hash != b.spec_hash  # same version label, different values
    placeholder = parse_spec(_raw())
    assert parse_spec(placeholder.yaml_text).spec_hash == placeholder.spec_hash
    assert parse_spec(a.yaml_text).spec_hash == a.spec_hash


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r.pop("cma"), "no cma section"),
        (lambda r: r["grid"].update(estimators=[{"method_mu": "hist", "method_cov": "hist"}]),
         "no grid.estimators entry uses method_mu: cma"),
        (lambda r: r.update(cma={"version": "house"}), "give vectors inline"),
        (lambda r: r.update(cma={"version": "placeholder", "vectors": _vectors(("2011-01", {}))}),
         "placeholder vectors are generated"),
        (lambda r: r.update(cma={"version": "h", "vectors": _vectors(("2015-01", {}),
                                                                     ("2012-01", {}))}),
         "strictly increasing"),
        (lambda r: r.update(cma={"version": "h", "vectors": _vectors(("2011-01", {"HY": -1.0}))}),
         "> -1"),
        (lambda r: r.update(cma={"version": "h", "vectors": [{"effective": "2011-01"}]}),
         "missing keys"),
    ],
)  # fmt: skip
def test_spec_errors(mutate, match):
    raw = _raw()
    mutate(raw)
    with pytest.raises(SpecError, match=match):
        parse_spec(raw)


# --- point in time and units -----------------------------------------------------------------


def test_vector_at_is_point_in_time_and_converts_units():
    cma = CMA("h", (CMAVector("2015-01", {a: 0.04 for a in ASSETS}),
                    CMAVector("2018-07", {a: 0.10 for a in ASSETS})))  # fmt: skip
    assert cma.vector_at(pd.Timestamp("2018-06-30")).effective == "2015-01"
    assert cma.vector_at(pd.Timestamp("2018-07-31")).effective == "2018-07"
    with pytest.raises(ValueError, match="no vector in effect"):
        cma.vector_at(pd.Timestamp("2014-12-31"))
    mu = cma.mu_period(pd.Timestamp("2019-01-31"), "M", ASSETS)
    assert mu["CAND"] == pytest.approx(1.10 ** (1 / 12) - 1)
    with pytest.raises(ValueError, match="unknown"):
        CMA("h", (CMAVector("2015-01", {**dict.fromkeys(ASSETS, 0.04), "X": 0.1}),)).check_covers(
            ASSETS
        )


# --- allocators ------------------------------------------------------------------------------


def _cma_mu():
    mu = R.mean() * 0 + 0.004
    mu["CAND"] = 0.01
    return mu


@pytest.mark.parametrize("cls", [RiskfolioMeanRisk, SkfolioMeanRisk])
def test_cma_mean_method_needs_and_uses_the_override(cls):
    missing = cls("cma", "ledoit", "MV", "Sharpe").fit(R, fit_context(DATA))
    assert missing.status == "exception" and "needs ctx.mu_override" in missing.message
    ctx = fit_context(DATA, mu_override=_cma_mu())
    with_cma = cls("cma", "ledoit", "MV", "Sharpe").fit(R, ctx)
    same_mu = cls("hist", "ledoit", "MV", "Sharpe").fit(R, ctx)
    assert with_cma.status == "ok"
    assert (with_cma.weights - same_mu.weights).abs().max() < 1e-9


def test_cma_in_hierarchical_allocators():
    ctx = fit_context(DATA, mu_override=_cma_mu())
    assert RiskfolioHC(model="NCO", obj="Sharpe", method_mu="cma").fit(R, ctx).status == "ok"


# --- runner ----------------------------------------------------------------------------------


def _wf_raw(cma):
    raw = _raw(**cma)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"})  # fmt: skip
    raw["grid"]["allocators"] = [{"type": "riskfolio_mean_risk", "rm": ["MV"], "obj": ["Sharpe"]}]
    raw["grid"]["constraint_sets"] = [{"name": "cap", "candidate_cap": 0.3}]
    return raw


def test_runner_passes_the_cma_in_effect_at_each_date(tmp_path):
    # the candidate becomes attractive only from 2017-07: its weight jumps at that rebalance
    cma = {
        "version": "two_vintages",
        "vectors": _vectors(("2011-01", {"CAND": -0.05}), ("2017-07", {"CAND": 0.3})),
    }
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(_wf_raw(cma)), reg)
    cells = reg.cells(s.experiment_id)
    cells = cells[(cells.data_variant == "full") & cells.estimator_json.str.contains('"cma"')]
    w = reg.weights(s.experiment_id).query("asset_id == 'CAND'").set_index("cell_id")["weight"]
    by_date = cells.set_index(pd.to_datetime(cells.window_end))["cell_id"].map(w).sort_index()
    assert (by_date[:"2016-12-31"] < 0.01).all()
    assert (by_date["2017-12-31":] > 0.29).all()
    md = build_report(reg, s.experiment_id).summary_md
    assert "## Return assumptions (CMA)" in md and "| CMA | two_vintages" in md


def test_runner_refuses_fits_before_the_first_vector(tmp_path):
    cma = {"version": "late", "vectors": _vectors(("2016-01", {}))}
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    with pytest.raises(SpecError, match="set backtest.start on or after"):
        run_experiment(parse_spec(_wf_raw(cma)), reg)
    raw = _wf_raw(cma)
    raw["backtest"]["start"] = "2016-12"
    s = run_experiment(parse_spec(copy.deepcopy(raw)), reg)
    assert s.n_cells > 0 and np.isfinite(list(s.status_counts.values())).all()


def test_runner_checks_cma_coverage(tmp_path):
    vectors = _vectors(("2011-01", {}))
    del vectors[0]["returns_annual"]["HY"]
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    with pytest.raises(SpecError, match=r"missing \['HY'\]"):
        run_experiment(parse_spec(_wf_raw({"version": "h", "vectors": vectors})), reg)

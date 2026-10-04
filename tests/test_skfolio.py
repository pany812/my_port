"""skfolio backend: parity with Riskfolio-Lib, policy translation, failure classification."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
import yaml

from tests.fixtures.synthetic import fit_context, policy, small_market
from tests.test_runner import SMALL
from workbench.allocators.riskfolio_hc import RiskfolioHC
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.allocators.skfolio_hc import SkfolioHC
from workbench.allocators.skfolio_mr import SkfolioMeanRisk
from workbench.evaluation.agreement import agreement_summary, paired_cells
from workbench.evaluation.corridor import corridor
from workbench.evaluation.report import build_report
from workbench.grid.runner import run_experiment
from workbench.grid.spec import parse_spec
from workbench.policy.compiled import CompiledPolicy
from workbench.policy.skfolio import linear_infeasibility, mean_risk_kwargs
from workbench.registry.store import Registry

DATA = small_market()
R = DATA.returns


def _pair(rf_alloc, sk_alloc, pol=None):
    ctx = fit_context(DATA, pol)
    a, b = rf_alloc.fit(R, ctx), sk_alloc.fit(R, ctx)
    assert a.status == "ok", a.message
    assert b.status == "ok", b.message
    return a.weights, b.weights


# --- parity --------------------------------------------------------------------------------


@pytest.mark.parametrize("rm", ["CVaR", "CDaR"])
@pytest.mark.parametrize("obj", ["MinRisk", "Sharpe"])
@pytest.mark.parametrize("est", [("hist", "hist"), ("hist", "ledoit"), ("JS", "gerber1")])
def test_lp_risk_measures_match_riskfolio(rm, obj, est):
    mu, cov = est
    a, b = _pair(RiskfolioMeanRisk(mu, cov, rm, obj), SkfolioMeanRisk(mu, cov, rm, obj))
    assert (a - b).abs().max() < 1e-6


@pytest.mark.parametrize("obj", ["MinRisk", "Sharpe"])
def test_mv_matches_riskfolio_on_objective(obj):
    """MV weights sit in a flat region (solvers differ ~1e-4); compare the objective instead."""
    a, b = _pair(RiskfolioMeanRisk("hist", "hist", "MV", obj),
                 SkfolioMeanRisk("hist", "hist", "MV", obj))  # fmt: skip
    cov = R.cov().to_numpy()
    mu = R.mean().to_numpy()
    vol = lambda w: float(np.sqrt(w @ cov @ w))  # noqa: E731
    if obj == "MinRisk":
        # Cash-dominated min-variance is ill-conditioned: with default CLARABEL tolerances the
        # libraries' problem scalings leave a ~5e-5 relative gap in volatility.
        assert vol(b.to_numpy()) == pytest.approx(vol(a.to_numpy()), rel=1e-4)
    else:
        sharpe = lambda w: float(mu @ w) / vol(w)  # noqa: E731
        assert sharpe(b.to_numpy()) == pytest.approx(sharpe(a.to_numpy()), rel=1e-5)
    assert (a - b).abs().max() < 1e-3


def test_utility_maps_risk_aversion_one_to_one():
    a, b = _pair(RiskfolioMeanRisk("hist", "hist", "MV", "Utility", l=10),
                 SkfolioMeanRisk("hist", "hist", "MV", "Utility", l=10))  # fmt: skip
    assert (a - b).abs().max() < 1e-3


def test_hrp_identical_and_herc_identical_with_matching_clusters():
    a, b = _pair(RiskfolioHC("HRP", "pearson", "ward"), SkfolioHC("HRP", "pearson", "ward"))
    assert (a - b).abs().max() < 1e-12
    import riskfolio as rp  # riskfolio's chosen k, to force skfolio to the same

    hc = rp.HCPortfolio(returns=R)
    hc.optimization(model="HERC", codependence="pearson", rm="MV", linkage="ward")
    a, b = _pair(RiskfolioHC("HERC", "pearson", "ward"),
                 SkfolioHC("HERC", "pearson", "ward", max_clusters=hc.k))  # fmt: skip
    assert (a - b).abs().max() < 1e-12


# --- policy translation ----------------------------------------------------------------------


@pytest.mark.parametrize("rm", ["MV", "CVaR", "CDaR"])
def test_te_band_cap_and_class_limits_hold(rm):
    for pol in (
        policy(te_annual=0.02, candidate_cap=0.10),
        policy(band=0.05, candidate_cap=0.10, class_limits={"equity": [0.45, 0.55]}),
    ):
        res = SkfolioMeanRisk("hist", "ledoit", rm, "Sharpe").fit(R, fit_context(DATA, pol))
        assert res.status == "ok", res.message
        assert pol.violations(res.weights, R) == []


def test_mean_risk_kwargs_translation():
    pol = policy(
        band=0.05, te_annual=0.02, candidate_cap=0.10, class_limits={"equity": [0.45, 0.55]}
    )
    kw = mean_risk_kwargs(pol, list(R.columns))
    assert kw["max_weights"]["CAND"] == pytest.approx(0.10)
    assert kw["max_turnover"] == 0.05 and kw["previous_weights"]["GL_EQ"] == pytest.approx(0.30)
    assert kw["max_tracking_error"] == pytest.approx(0.02 / np.sqrt(12))
    assert kw["linear_constraints"] == ["equity >= 0.45", "equity <= 0.55"]
    assert kw["groups"]["SE_GOV"] == ["fixed_income"]


def test_hc_respects_cap_and_band():
    pol = policy(band=0.05, candidate_cap=0.10)
    for model in ("HRP", "HERC"):
        res = SkfolioHC(model).fit(R, fit_context(DATA, pol))
        assert res.status == "ok", res.message
        assert pol.violations(res.weights, R) == []


# --- failure classification ------------------------------------------------------------------


def test_linear_infeasibility_detection():
    assets = list(R.columns)
    assert linear_infeasibility(policy(band=0.05, te_annual=0.01), assets) is None  # SAA fits
    assert (
        linear_infeasibility(policy(band=0.05, class_limits={"equity": [0.6, 0.65]}), assets)
        is None
    )  # SAA breaks it, but the LP finds a point
    assert "infeasible" in linear_infeasibility(
        policy(band=0.05, class_limits={"equity": [0.7, 0.8]}), assets
    )


def test_infeasible_problem_recorded_as_infeasible(capsys):
    pol = policy(band=0.05, class_limits={"equity": [0.7, 0.8]})
    res = SkfolioMeanRisk("hist", "hist", "MV", "MinRisk").fit(R, fit_context(DATA, pol))
    assert res.status == "infeasible" and "linear constraints" in res.message
    assert res.diagnostics["solver_errors"]
    assert capsys.readouterr().out == ""


def test_solver_failure_on_feasible_problem_is_solver_error(monkeypatch):
    from skfolio.optimization import MeanRisk

    def boom(self, X, y=None, **kw):
        raise cp.error.SolverError("numerical trouble")

    monkeypatch.setattr(MeanRisk, "fit", boom)
    res = SkfolioMeanRisk().fit(R, fit_context(DATA, policy(candidate_cap=0.10)))
    assert res.status == "solver_error" and "numerical trouble" in res.message


def test_hc_infeasible_bounds_and_unsupported_names():
    assets = list(R.columns)
    p = CompiledPolicy(lower=pd.Series(0.0, index=assets), upper=pd.Series(0.1, index=assets))
    res = SkfolioHC("HRP").fit(R, fit_context(DATA, p))
    assert res.status == "infeasible" and "upper bounds sum" in res.message
    for alloc in (SkfolioMeanRisk(method_cov="oas"), SkfolioMeanRisk(rm="EVaR"),
                  SkfolioHC(codependence="gerber1")):  # fmt: skip
        res = alloc.fit(R, fit_context(DATA))
        assert res.status == "exception" and "unsupported in the skfolio backend" in res.message


def test_mu_override_honoured():
    mu = pd.Series(0.0, index=R.columns)
    mu["CAND"] = 0.02
    ctx = fit_context(DATA, policy(candidate_cap=0.10), mu_override=mu)
    res = SkfolioMeanRisk(rm="MV", obj="MaxRet").fit(R, ctx)
    assert res.status == "ok" and res.weights["CAND"] == pytest.approx(0.10, abs=1e-6)


# --- grid integration: library dimension and agreement --------------------------------------


@pytest.fixture(scope="module")
def two_lib(tmp_path_factory):
    raw = yaml.safe_load(SMALL)
    raw["grid"]["allocators"] = [
        {"type": "static_saa"},
        {"type": "riskfolio_mean_risk", "rm": ["CVaR", "MV"], "obj": ["MinRisk"]},
        {"type": "skfolio_mean_risk", "rm": ["CVaR", "MV"], "obj": ["MinRisk"]},
        {"type": "riskfolio_hc", "model": ["HRP", "HERC"]},
        {"type": "skfolio_hc", "model": ["HRP", "HERC"]},
    ]
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"})  # fmt: skip
    reg = Registry(f"sqlite:///{tmp_path_factory.mktemp('two') / 'r.db'}")
    return reg, run_experiment(parse_spec(raw), reg)


def test_paired_cells_and_summary(two_lib):
    reg, s = two_lib
    pairs = paired_cells(reg, s.experiment_id)
    assert set(pairs["family"]) == {"mean_risk", "hc"}
    summ = agreement_summary(pairs).set_index("label")
    cvar = summ[summ.index.str.contains("rm=CVaR")]
    assert (cvar["max_abs_diff"] < 1e-6).all()  # LP measures agree across libraries
    hrp = summ[summ.index.str.contains("model=HRP")]
    assert (hrp["max_abs_diff"] < 1e-9).all()
    assert (summ["n_status_mismatch"] == 0).all()


def test_library_group_and_report_section(two_lib):
    reg, s = two_lib
    c = corridor(reg, s.experiment_id, by=["library"])
    assert set(c["library"]) == {"own", "riskfolio", "skfolio"}
    md = build_report(reg, s.experiment_id).summary_md
    assert "## Library agreement (Riskfolio-Lib vs skfolio)" in md
    assert "### By library" in md


def test_single_library_report_has_no_agreement_section(tmp_path):
    raw = yaml.safe_load(SMALL)
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(raw), reg)
    assert paired_cells(reg, s.experiment_id).empty
    assert "Library agreement" not in build_report(reg, s.experiment_id).summary_md


def test_bounded_hrp_nan_defect_is_recorded_as_solver_error(monkeypatch):
    """skfolio 1.4.11 returns NaN weights when HRP lower bounds bind; record, never pass on."""
    from skfolio.optimization import HierarchicalRiskParity

    real_fit = HierarchicalRiskParity.fit

    def nan_fit(self, X, y=None, **kw):
        real_fit(self, X, y)
        self.weights_ = np.full_like(self.weights_, np.nan)
        return self

    monkeypatch.setattr(HierarchicalRiskParity, "fit", nan_fit)
    res = SkfolioHC("HRP").fit(R, fit_context(DATA, policy(band=0.05, candidate_cap=0.10)))
    assert res.status == "solver_error" and "non-finite weights" in res.message

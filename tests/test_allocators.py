import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import fit_context, policy, small_market
from workbench.allocators._solve import clean_weights, guarded_fit
from workbench.allocators.base import Allocator
from workbench.allocators.naive import EqualWeight, InverseVol, SAAPlus, StaticSAA
from workbench.allocators.riskfolio_hc import RiskfolioHC
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.allocators.skfolio_hc import SkfolioHC
from workbench.allocators.skfolio_mr import SkfolioMeanRisk
from workbench.policy.compiled import CompiledPolicy

ALLOCATORS = [
    StaticSAA(),
    SAAPlus(0.05),
    EqualWeight(),
    InverseVol(),
    RiskfolioMeanRisk(method_cov="ledoit", rm="MV", obj="Sharpe"),
    RiskfolioMeanRisk(method_cov="ledoit", rm="CVaR", obj="MinRisk"),
    RiskfolioMeanRisk(method_mu="JS", method_cov="gerber1", rm="CDaR", obj="Sharpe"),
    RiskfolioHC(model="HRP", codependence="pearson", linkage="ward"),
    RiskfolioHC(model="HERC", codependence="spearman", linkage="ward"),
    RiskfolioHC(model="NCO", codependence="pearson", linkage="ward", obj="MinRisk"),
    RiskfolioHC(model="NCO", obj="Sharpe", method_mu="JS", method_cov="gerber1"),
    SkfolioMeanRisk(method_cov="ledoit", rm="CVaR", obj="Sharpe"),
    SkfolioMeanRisk(method_mu="JS", method_cov="gerber1", rm="MV", obj="MinRisk"),
    SkfolioHC(model="HRP", codependence="pearson", linkage="ward"),
    SkfolioHC(model="HERC", codependence="spearman", linkage="ward"),
]
IDS = [f"{a.name}:{i}" for i, a in enumerate(ALLOCATORS)]


@pytest.fixture(scope="module")
def data():
    return small_market()


def _assert_ok(res, assets):
    assert res.status == "ok", res.message
    assert list(res.weights.index) == assets
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (res.weights >= 0).all()
    assert res.elapsed_s >= 0


# --- smoke: every allocator, unconstrained and under the example constraint sets ---------


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=IDS)
def test_smoke_unconstrained(alloc, data, capsys):
    assert isinstance(alloc, Allocator)
    res = alloc.fit(data.returns, fit_context(data))
    _assert_ok(res, data.assets)
    assert capsys.readouterr().out == ""  # nothing leaks to stdout


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=IDS)
def test_band_set_is_honoured_or_recorded(alloc, data):
    p = policy(band=0.05, candidate_cap=0.10)
    res = alloc.fit(data.returns, fit_context(data, p))
    if res.status == "ok":
        assert p.violations(res.weights, data.returns) == []
    else:  # heuristics that ignore the band are recorded, not dropped
        assert res.status == "infeasible" and "band" in res.message
        assert alloc.name in ("equal_weight", "inverse_vol")


@pytest.mark.parametrize("rm", ["MV", "CVaR", "CDaR"])
def test_mean_risk_respects_te_and_cap(rm, data):
    p = policy(te_annual=0.02, candidate_cap=0.10)
    res = RiskfolioMeanRisk(method_cov="ledoit", rm=rm, obj="Sharpe").fit(
        data.returns, fit_context(data, p)
    )
    _assert_ok(res, data.assets)
    assert p.tracking_error(res.weights, data.returns) <= p.te * (1 + 1e-4)
    assert res.weights["CAND"] <= 0.10 + 1e-6


def test_mean_risk_band_and_class_limits_hold(data):
    p = policy(band=0.05, candidate_cap=0.10, class_limits={"equity": [0.45, 0.55]})
    res = RiskfolioMeanRisk(method_cov="ledoit", rm="CVaR", obj="Sharpe").fit(
        data.returns, fit_context(data, p)
    )
    _assert_ok(res, data.assets)
    saa = fit_context(data).saa
    assert (res.weights - saa).abs().max() <= 0.05 + 1e-6
    eq = res.weights[["SE_EQ", "GL_EQ", "EM_EQ"]].sum()
    assert 0.45 - 1e-6 <= eq <= 0.55 + 1e-6


def test_hc_under_te_is_recorded_infeasible_when_breached(data):
    p = policy(te_annual=0.005, candidate_cap=0.10)
    res = RiskfolioHC(model="HRP").fit(data.returns, fit_context(data, p))
    assert res.status == "infeasible"
    assert "TE" in res.message and res.diagnostics["violations"]
    assert set(res.diagnostics["rejected_weights"]) == set(data.assets)


# --- forced-infeasible case for every allocator -----------------------------------------


@pytest.mark.parametrize(
    "alloc,constraint_set,expect",
    [
        (StaticSAA(), {"class_limits": {"equity": [0.6, 0.7]}}, "class equity"),
        (SAAPlus(0.15), {"candidate_cap": 0.10}, "CAND: weight 15.0000% above upper"),
        (EqualWeight(), {"candidate_cap": 0.10}, "CAND"),
        (InverseVol(), {"asset_ranges": True}, "CASH"),
        (
            RiskfolioMeanRisk(method_cov="ledoit", rm="MV", obj="MinRisk"),
            # band 5% on 3 equity blocks reaches at most 65% equity
            {"band": 0.05, "class_limits": {"equity": [0.7, 0.8]}},
            "doesn't have a solution",
        ),
        (
            RiskfolioHC(model="HRP"),
            {"candidate_cap": 0.10, "class_limits": {"equity": [0.6, 0.7]}},
            "class equity",
        ),
    ],
    ids=["static_saa", "saa_plus", "equal_weight", "inverse_vol", "mean_risk", "hc"],
)
def test_forced_infeasible_is_recorded(alloc, constraint_set, expect, data, capsys):
    res = alloc.fit(data.returns, fit_context(data, policy(**constraint_set)))
    assert res.status == "infeasible"
    assert res.weights is None
    assert expect in res.message
    assert capsys.readouterr().out == ""  # Riskfolio's print was captured into message


def test_hc_infeasible_bounds_recorded_before_solving(data):
    assets = data.assets
    p = CompiledPolicy(lower=pd.Series(0.0, index=assets), upper=pd.Series(0.1, index=assets))
    res = RiskfolioHC(model="HERC").fit(data.returns, fit_context(data, p))
    assert res.status == "infeasible" and "upper bounds sum to 0.9000" in res.message


# --- other status paths -----------------------------------------------------------------


def test_nan_returns_recorded_as_exception(data):
    r = data.returns.copy()
    r.iloc[0, 0] = np.nan
    res = InverseVol().fit(r, fit_context(data))
    assert res.status == "exception" and "NaN" in res.message


def test_look_ahead_rejected(data):
    ctx = fit_context(data)
    ctx = type(ctx)(as_of=data.returns.index[-2], saa=ctx.saa, candidate="CAND", policy=ctx.policy)
    res = StaticSAA().fit(data.returns, ctx)
    assert res.status == "exception" and "look-ahead" in res.message


def test_solver_error_mapped(data):
    def impl(r, c, d):
        raise cp.error.SolverError("boom")

    res = guarded_fit(impl, data.returns, fit_context(data))
    assert res.status == "solver_error" and "boom" in res.message


def test_printed_output_lands_in_message(data):
    def impl(r, c, d):
        print("solver chatter")  # noqa: T201 - simulating Riskfolio-Lib
        return None

    res = guarded_fit(impl, data.returns, fit_context(data))
    assert res.status == "infeasible" and "solver chatter" in res.message


def test_clean_weights_clips_noise_only():
    w, adj = clean_weights(pd.Series({"A": 0.6, "B": 0.4 + 1e-9, "C": -1e-9}), long_only=True)
    assert w["C"] == 0 and w.sum() == pytest.approx(1.0) and adj < 1e-8
    w, _ = clean_weights(pd.Series({"A": 1.1, "B": -0.1}), long_only=True)
    assert w["B"] == -0.1  # real negatives are left for the post-check


# --- behaviour -------------------------------------------------------------------------


def test_saa_plus_pro_rata(data):
    res = SAAPlus(0.05).fit(data.returns, fit_context(data))
    assert res.weights["CAND"] == pytest.approx(0.05)
    assert res.weights["GL_EQ"] == pytest.approx(0.30 * 0.95)
    with pytest.raises(ValueError):
        SAAPlus(0.05, funding="from_equity")


def test_inverse_vol_favours_low_vol(data):
    w = InverseVol().fit(data.returns, fit_context(data)).weights
    assert w.idxmax() == "CASH"
    assert w["SE_GOV"] > w["GL_EQ"]


def test_mu_override_drives_mean_risk(data):
    mu = pd.Series(0.0, index=data.assets)
    mu["CAND"] = 0.02  # per month
    p = policy(candidate_cap=0.10)
    res = RiskfolioMeanRisk(rm="MV", obj="MaxRet").fit(
        data.returns, fit_context(data, p, mu_override=mu)
    )
    _assert_ok(res, data.assets)
    assert res.weights["CAND"] == pytest.approx(0.10, abs=1e-6)


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=IDS)
def test_deterministic(alloc, data):
    ctx = fit_context(data, policy(band=0.10, candidate_cap=0.10))
    a, b = alloc.fit(data.returns, ctx), alloc.fit(data.returns, ctx)
    assert a.status == b.status
    if a.status == "ok":
        pd.testing.assert_series_equal(a.weights, b.weights)


def test_params_are_json_like():
    import json

    for alloc in ALLOCATORS:
        json.dumps({"name": alloc.name, **alloc.params()})


def test_js_mu_is_cast_to_real_and_recorded(data):
    res = RiskfolioMeanRisk(method_mu="JS", method_cov="ledoit").fit(
        data.returns, fit_context(data)
    )
    _assert_ok(res, data.assets)
    assert res.diagnostics["mu_cast_from_complex"] is True
    assert res.diagnostics["cov_cast_from_complex"] is False

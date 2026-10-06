"""Black–Litterman with the SAA as the prior: posterior, zero view, closed-form breakeven,
root-finding, both libraries, the runner and the report view."""

import math

import numpy as np
import pandas as pd
import pytest
import riskfolio as rp
import yaml
from skfolio.moments import EmpiricalCovariance, EquilibriumMu
from skfolio.prior import BlackLitterman, EmpiricalPrior

from tests.fixtures.synthetic import fit_context, policy, small_market
from tests.test_runner import SMALL
from workbench.allocators._bl import WEIGHT_TOL, BLParams, Prior, breakeven, posterior_excess
from workbench.allocators.riskfolio_bl import RiskfolioBlackLitterman
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.allocators.skfolio_bl import SkfolioBlackLitterman
from workbench.allocators.skfolio_mr import SkfolioMeanRisk
from workbench.data.synthetic import placeholder_saa
from workbench.evaluation.agreement import agreement_summary, paired_cells
from workbench.evaluation.report import build_report
from workbench.evaluation.views import breakeven_view, weight_by_view
from workbench.grid.runner import run_experiment
from workbench.grid.spec import SpecError, parse_spec
from workbench.policy.compiler import ConstraintSet, compile_policy
from workbench.policy.saa import SAA
from workbench.units import sharpe_annual_to_period

DATA = small_market()
R = DATA.returns
A = list(R.columns)
C = "CAND"
SAA_W = placeholder_saa()[A]
COV = R.cov()
SR = sharpe_annual_to_period(0.3, "M")
PRIOR = Prior.build(COV, SAA_W, C, SR)
LIBS = [RiskfolioBlackLitterman, SkfolioBlackLitterman]


def _annual_view(prior: Prior, premium_period: float) -> float:
    return 12 * (prior.pi_c + premium_period)


# --- posterior --------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("tau", [1 / len(R), 0.05, 1.0])
def test_posterior_matches_both_libraries_and_tau_cancels(confidence, tau):
    s = COV.to_numpy()
    q = PRIOR.pi_c + 0.02 / 12
    ours = PRIOR.pi.to_numpy() + PRIOR.premium(q, confidence) * PRIOR.beta.to_numpy()
    textbook = posterior_excess(PRIOR.pi.to_numpy(), s, A.index(C), q, confidence, tau)
    assert np.abs(ours - textbook).max() < 1e-15
    sk = BlackLitterman(
        views=[f"{C} = {q!r}"], view_confidences=[confidence], tau=tau,
        prior_estimator=EmpiricalPrior(mu_estimator=EquilibriumMu(
            risk_aversion=PRIOR.delta, weights=SAA_W.to_numpy(),
            covariance_estimator=EmpiricalCovariance())),
    ).fit(R)  # fmt: skip
    assert np.abs(sk.return_distribution_.mu - ours).max() < 1e-15
    if confidence == 0.5:  # Riskfolio's black_litterman: Omega = diag(P tau Sigma P'), tau = 1/T
        p = pd.DataFrame([[1.0 if a == C else 0.0 for a in A]], columns=A)
        mu_rf, _, _ = rp.ParamsEstimation.black_litterman(
            R, SAA_W.to_frame(), p, pd.DataFrame([[q]]), delta=PRIOR.delta, rf=0
        )
        assert np.abs(mu_rf.to_numpy().ravel() - ours).max() < 1e-15  # fmt: skip


def test_prior_scales_with_the_saa_sharpe():
    w = SAA_W.to_numpy()
    saa_excess = float(PRIOR.pi.to_numpy() @ w)
    assert saa_excess / PRIOR.sigma_saa == pytest.approx(SR)  # the SAA earns its assumed Sharpe
    assert PRIOR.delta > 0
    assert PRIOR.pi_c == pytest.approx(PRIOR.delta * float((COV.to_numpy() @ w)[A.index(C)]))


# --- zero view, closed form ------------------------------------------------------------------


@pytest.mark.parametrize("obj", ["Sharpe", "Utility"])
@pytest.mark.parametrize("cls", LIBS)
def test_view_at_equilibrium_returns_the_saa(cls, obj):
    view = _annual_view(PRIOR, 0.0)
    res = cls(view_annual=view, obj=obj).fit(R, fit_context(DATA))
    assert res.status == "ok", res.message
    assert res.weights[C] < 5e-4
    # skfolio's max-Sharpe is flat along the cash direction (CLAUDE.md): cash may drift ~0.2pp
    tol = 2.5e-3 if (cls is SkfolioBlackLitterman and obj == "Sharpe") else 5e-4
    assert (res.weights - SAA_W).abs().max() < tol


@pytest.mark.parametrize("x", [0.02, 0.05, 0.10])
def test_unconstrained_breakeven_is_closed_form_and_pro_rata(x):
    res = RiskfolioBlackLitterman(target_weight=x).fit(R, fit_context(DATA))
    d = res.diagnostics
    assert res.status == "ok" and abs(res.weights[C] - x) <= WEIGHT_TOL
    assert d["premium_annual"] == pytest.approx(12 * PRIOR.unconstrained_premium(x), rel=1e-9)
    assert d["n_solves"] == 2  # premium 0, then the closed form already hits the target
    pro_rata = SAA_W * (1 - x)
    pro_rata[C] = x
    assert (res.weights - pro_rata).abs().max() < 5e-4  # BL + Sharpe, no limits = saa_plus
    # required Sharpe = SR_SAA * (rho + (sigma_c / sigma_SAA) * x / (1 - x))
    sharpe = 0.3 * (PRIOR.rho + PRIOR.sigma_c / PRIOR.sigma_saa * x / (1 - x))
    assert d["candidate_sharpe_annual"] == pytest.approx(sharpe, rel=1e-9)
    assert d["unconstrained_sharpe_annual"] == pytest.approx(sharpe, rel=1e-9)


# --- constrained: monotone, root-found, unreachable -------------------------------------------


SETS = {
    "te1": policy(te_annual=0.01, candidate_cap=0.25),
    "ranges": policy(asset_ranges=True, candidate_cap=0.25),
    "band": policy(band=0.05, candidate_cap=0.10),
}


@pytest.mark.parametrize("name", list(SETS))
def test_weight_is_monotone_in_the_view(name):
    ctx = fit_context(DATA, SETS[name])
    ws = [RiskfolioBlackLitterman(view_annual=v).fit(R, ctx).weights[C]
          for v in np.linspace(-0.01, 0.03, 17)]  # fmt: skip
    assert (np.diff(ws) >= -1e-4).all()


@pytest.mark.parametrize("cls", LIBS)
def test_constrained_breakeven_hits_the_target(cls):
    res = cls(target_weight=0.10).fit(R, fit_context(DATA, SETS["te1"]))
    assert res.status == "ok", res.message
    assert abs(res.weights[C] - 0.10) <= WEIGHT_TOL
    d = res.diagnostics
    assert d["premium_annual"] > d["unconstrained_premium_annual"]  # the TE limit costs premium
    assert d["n_solves"] > 2


@pytest.mark.parametrize("cls", LIBS)
def test_unreachable_target_is_infeasible(cls):
    res = cls(target_weight=0.10).fit(R, fit_context(DATA, SETS["band"]))
    assert res.status == "infeasible"
    assert "unreachable" in res.message and "at most 5.00%" in res.message


def test_forced_infeasible_policy():
    pol = policy(class_limits={"equity": [0.9, 1.0]}, band=0.05)
    for cls in LIBS:
        for kw in ({"view_annual": 0.02}, {"target_weight": 0.05}):
            assert cls(**kw).fit(R, fit_context(DATA, pol)).status == "infeasible"


def test_root_finder_handles_targets_below_the_zero_premium_weight():
    """If the zero-premium solution already holds more than the target, search downwards."""

    def solve(a):  # a toy optimiser: weight rises linearly with the premium from 8%
        x = float(np.clip(0.08 + 50 * a, 0, 1))
        return pd.Series({C: x, "X": 1 - x})

    be = breakeven(solve, C, 0.05, first_guess=1e-4, max_premium=1.0)
    assert abs(be.weights[C] - 0.05) <= WEIGHT_TOL and be.premium < 0


def test_parameter_validation():
    for kw, match in [
        ({}, "exactly one"),
        ({"view_annual": 0.02, "target_weight": 0.05}, "exactly one"),
        ({"target_weight": 0.05, "confidence": 0.5}, "confidence applies to view_annual"),
        ({"target_weight": 1.0}, "target_weight"),
        ({"view_annual": 0.02, "confidence": 0.0}, "confidence"),
        ({"view_annual": 0.02, "obj": "MinRisk"}, "obj"),
        ({"view_annual": 0.02, "prior_sharpe": 0.0}, "prior_sharpe"),
    ]:
        for cls in LIBS:
            with pytest.raises(ValueError, match=match):
                cls(**kw)
    assert BLParams(0.02, 1.0, None, 0.3, "Sharpe").check() is None


@pytest.mark.parametrize("kw", [{"view_annual": 0.01, "confidence": 0.5},
                                {"target_weight": 0.05}])  # fmt: skip
def test_libraries_agree(kw):
    ctx = fit_context(DATA, SETS["ranges"])
    a = RiskfolioBlackLitterman(**kw).fit(R, ctx)
    b = SkfolioBlackLitterman(**kw).fit(R, ctx)
    assert a.status == b.status == "ok"
    assert abs(a.weights[C] - b.weights[C]) < 2 * WEIGHT_TOL
    assert (a.weights - b.weights).abs().max() < 1e-3  # flat max-Sharpe along cash in skfolio


# --- max-Sharpe undefined (surfaced by rf > 0 in the example) ---------------------------------


@pytest.mark.parametrize("cls", [RiskfolioMeanRisk, SkfolioMeanRisk])
def test_sharpe_without_positive_excess_return_is_infeasible_in_both_libraries(cls):
    pol = compile_policy(SAA.placeholder(), ConstraintSet.from_dict({"name": "rf"}), "M",
                         rf_annual=0.5)  # fmt: skip
    res = cls("hist", "ledoit", "MV", "Sharpe").fit(R, fit_context(DATA, pol))
    assert res.status == "infeasible"
    assert "max-Sharpe undefined" in res.message


# --- runner, views, report -------------------------------------------------------------------


@pytest.fixture(scope="module")
def bl_run(tmp_path_factory):
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"}, rf_annual=0.02)  # fmt: skip
    raw["grid"]["allocators"] = [
        {"type": "static_saa"},
        {"type": "riskfolio_bl", "target_weight": [0.05, 0.15]},
        {"type": "skfolio_bl", "target_weight": [0.05, 0.15]},
        {"type": "riskfolio_bl", "view_annual": [0.0, 0.02], "confidence": [0.5, 1.0]},
    ]
    raw["grid"]["constraint_sets"] = [{"name": "band", "band": 0.10, "candidate_cap": 0.10},
                                      {"name": "free", "candidate_cap": 0.5}]  # fmt: skip
    reg = Registry_(tmp_path_factory)
    return reg, run_experiment(parse_spec(raw), reg)


def Registry_(tmp_path_factory):  # noqa: N802 - local factory
    from workbench.registry.store import Registry

    return Registry(f"sqlite:///{tmp_path_factory.mktemp('bl') / 'r.db'}")


def test_breakeven_view_and_report(bl_run):
    reg, s = bl_run
    be = breakeven_view(reg, s.experiment_id)
    full = be[be.data_variant == "full"].set_index(["constraint_set", "target_weight", "library"])
    assert set(full.index.get_level_values("library")) == {"riskfolio", "skfolio"}
    capped = full.loc[("band", 0.15)]
    assert (capped.n_reached == 0).all() and (capped.n_unreachable == capped.n_cells).all()
    free = full.loc[("free", 0.05)]
    assert (free.n_reached == free.n_cells).all()
    # unconstrained: the realised breakeven equals the closed form at every date (skfolio's
    # max-Sharpe carries ~1e-4 weight noise, so its root-finder sometimes bisects past it)
    gap = (free.sharpe_median - free.sharpe_unconstrained_median).abs()
    assert gap["riskfolio"] < 1e-9 and gap["skfolio"] < 1e-3
    assert (full.loc[("free", 0.15)].excess_median > free.excess_median).all()
    views = weight_by_view(reg, s.experiment_id)
    v = views[(views.data_variant == "full") & (views.constraint_set == "free")]
    assert (
        v.sort_values("view_annual")
        .groupby("confidence")
        .weight_median.apply(lambda w: w.is_monotonic_increasing)
        .all()
    )
    md = build_report(reg, s.experiment_id).summary_md
    assert "## What would it have to earn? (Black–Litterman breakeven)" in md
    assert "### Weight at a stated view" in md and "prior SAA Sharpe 0.30" in md


def test_bl_pairs_across_libraries(bl_run):
    reg, s = bl_run
    summ = agreement_summary(paired_cells(reg, s.experiment_id))
    bl = summ[summ.family == "black_litterman"]
    assert len(bl) > 0 and (bl.n_status_mismatch == 0).all()
    assert (bl.median_abs_diff <= 2 * WEIGHT_TOL).all()
    assert (bl.max_abs_diff <= 5 * WEIGHT_TOL).all()  # a rare noisy jump (diagnostics "jump")


def test_bl_spec_validation():
    raw = yaml.safe_load(SMALL)
    raw["grid"]["allocators"] = [{"type": "riskfolio_bl", "target_weight": 0.05, "rm": "CVaR"}]
    with pytest.raises(SpecError, match="unknown parameters"):
        parse_spec(raw)
    raw["grid"]["allocators"] = [{"type": "riskfolio_bl", "target_weight": 0.05,
                                  "method_cov": ["hist", "ledoit"]}]  # fmt: skip
    assert parse_spec(raw).allocators[0].params["method_cov"] == ["hist", "ledoit"]
    assert math.isfinite(PRIOR.rho)

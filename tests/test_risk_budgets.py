"""Risk-budget allocators (both libraries), budgets, realised shares and the report view."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.fixtures.synthetic import fit_context, policy, small_market
from tests.test_runner import SMALL
from workbench.allocators._budget import BUDGET_FLOOR, covariance, risk_budget
from workbench.allocators.riskfolio_rb import RiskfolioRiskBudget
from workbench.allocators.skfolio_rb import SkfolioRiskBudget
from workbench.evaluation.agreement import agreement_summary, paired_cells
from workbench.evaluation.report import build_report
from workbench.evaluation.views import risk_budget_view
from workbench.grid.runner import run_experiment
from workbench.grid.spec import parse_spec
from workbench.policy.compiled import CompiledPolicy
from workbench.registry.store import Registry

DATA = small_market()
R = DATA.returns
CTX = fit_context(DATA)


def test_budget_vector():
    cov = covariance(R, "ledoit")
    for rest in ("saa", "equal"):
        b = risk_budget(R, CTX.saa, "CAND", 0.05, "MV", rest, cov, 0.0)
        assert b.sum() == pytest.approx(1.0) and b["CAND"] == pytest.approx(0.05)
        assert (b.drop("CAND") > 0).all()
    eq = risk_budget(R, CTX.saa, "CAND", 0.10, "MV", "equal", cov, 0.0)
    assert np.allclose(eq.drop("CAND"), 0.90 / 8)
    saa = risk_budget(R, CTX.saa, "CAND", 0.05, "CVaR", "saa", cov, 0.0)
    # diversifiers with negative SAA contributions are floored, not dropped
    assert saa.drop("CAND").min() >= BUDGET_FLOOR * 0.95 * (1 - 0.05) / 1.1
    with pytest.raises(ValueError, match="candidate_share"):
        risk_budget(R, CTX.saa, "CAND", 1.0, "MV", "saa", cov, 0.0)
    with pytest.raises(ValueError, match="rest"):
        RiskfolioRiskBudget(0.05, rest="parity")


@pytest.mark.parametrize("rm", ["MV", "MSV"])
@pytest.mark.parametrize("cls", [RiskfolioRiskBudget, SkfolioRiskBudget])
def test_smooth_budgets_are_exact(rm, cls):
    res = cls(0.05, rm, "saa", method_cov="ledoit").fit(R, CTX)
    assert res.status == "ok", res.message
    assert res.diagnostics["realised_share"] == pytest.approx(0.05, abs=1e-3)
    assert res.diagnostics["target_share"] == 0.05


@pytest.mark.parametrize("rm", ["MV", "MSV", "CVaR", "CDaR"])
@pytest.mark.parametrize("rest", ["saa", "equal"])
def test_libraries_agree(rm, rest):
    a = RiskfolioRiskBudget(0.05, rm, rest, method_cov="ledoit").fit(R, CTX)
    b = SkfolioRiskBudget(0.05, rm, rest, method_cov="ledoit").fit(R, CTX)
    assert a.status == b.status == "ok"
    assert (a.weights - b.weights).abs().max() < 1e-4


def test_cvar_budget_realised_share_differs_but_is_recorded():
    res = RiskfolioRiskBudget(0.05, "CVaR", "equal", method_cov="ledoit").fit(R, CTX)
    gap = abs(res.diagnostics["realised_share"] - 0.05)
    assert 1e-3 < gap < 0.03  # not exact on historical scenarios (documented), but close


def test_linear_constraints_can_stop_the_budget():
    res = RiskfolioRiskBudget(0.05, "MV").fit(R, fit_context(DATA, policy(candidate_cap=0.03)))
    assert res.status == "ok" and res.weights["CAND"] == pytest.approx(0.03, abs=1e-6)
    assert res.diagnostics["realised_share"] < 0.05  # cap binds before the budget is met


def test_numerical_failure_is_solver_error_not_infeasible():
    """FLPM budget: CLARABEL fails on a feasible problem; SCS solves it (verified)."""
    clarabel = RiskfolioRiskBudget(0.05, "FLPM", "saa", method_cov="ledoit").fit(R, CTX)
    assert clarabel.status == "solver_error" and "feasible problem" in clarabel.message
    scs_ctx = fit_context(DATA, CompiledPolicy(solvers=("CLARABEL", "SCS")))
    assert (
        RiskfolioRiskBudget(0.05, "FLPM", "saa", method_cov="ledoit").fit(R, scs_ctx).status == "ok"
    )


def test_conflicting_policy_is_infeasible():
    pol = policy(class_limits={"equity": [0.9, 1.0]}, band=0.05)
    for cls in (RiskfolioRiskBudget, SkfolioRiskBudget):
        assert cls(0.05, "MV").fit(R, fit_context(DATA, pol)).status == "infeasible"


@pytest.fixture(scope="module")
def rb_run(tmp_path_factory):
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"}, solvers=["CLARABEL", "SCS"])  # fmt: skip
    raw["grid"]["allocators"] = [
        {"type": "static_saa"},
        {"type": "riskfolio_risk_budget", "candidate_share": [0.02, 0.05], "rm": ["MV", "CVaR"]},
        {"type": "skfolio_risk_budget", "candidate_share": [0.02, 0.05], "rm": ["MV", "CVaR"]},
    ]
    raw["grid"]["constraint_sets"] = [{"name": "free"}]
    reg = Registry(f"sqlite:///{tmp_path_factory.mktemp('rb') / 'r.db'}")
    return reg, run_experiment(parse_spec(raw), reg)


def test_risk_budget_view_and_report(rb_run):
    reg, s = rb_run
    v = risk_budget_view(reg, s.experiment_id)
    full = v[v.data_variant == "full"]
    assert set(full.library) == {"riskfolio", "skfolio"}
    assert set(full.target_share) == {0.02, 0.05}
    mv = full[(full.rm == "MV") & (full.target_share == 0.05)]
    assert (mv.realised_share_median - 0.05).abs().max() < 1e-3
    # a larger risk share needs more capital: guaranteed for smooth MV, not for CVaR on a
    # 36-month window (about 2 tail scenarios: different targets can give the same weight)
    for lib, g in full[full.rm == "MV"].groupby("library"):
        assert g.sort_values("target_share").weight_median.is_monotonic_increasing, lib
    md = build_report(reg, s.experiment_id).summary_md
    assert "## How much of our risk should it carry? (risk budgets)" in md


def test_risk_budgets_pair_across_libraries(rb_run):
    reg, s = rb_run
    pairs = paired_cells(reg, s.experiment_id)
    summ = agreement_summary(pairs)
    rb = summ[summ.family == "risk_budget"]
    mv = rb.label.str.contains("rm=MV")
    assert mv.any() and (rb[mv].max_abs_diff < 1e-4).all()
    assert (rb.median_abs_diff < 1e-4).all()  # the libraries agree almost everywhere
    # CVaR on a 36-month window (about 2 tail scenarios) is degenerate: rare dates differ by
    # up to ~0.2 percentage points between equally valid solutions
    assert (rb[~mv].max_abs_diff < 5e-3).all()
    assert (rb.n_status_mismatch == 0).all()
    diag = reg.cells(s.experiment_id).query("allocator == 'skfolio_risk_budget'")
    assert diag.diagnostics_json.map(lambda d: "realised_share" in json.loads(d)).all()
    assert isinstance(pd.Timestamp(diag.window_end.iloc[0]), pd.Timestamp)

"""P2-M2 frictions: transaction costs, named funding, candidate liquidity, threshold trading."""

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.fixtures.synthetic import fit_context, policy, small_market
from tests.test_runner import SMALL
from workbench.allocators.base import AllocationResult
from workbench.allocators.naive import SAAPlus
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.backtest.schedule import period_ends, rebalance_dates
from workbench.backtest.walkforward import WalkForwardEngine
from workbench.data.synthetic import placeholder_saa
from workbench.evaluation.oos import oos_table
from workbench.evaluation.report import build_report
from workbench.grid.runner import run_experiment
from workbench.grid.spec import LiquiditySpec, SpecError, WindowSpec, parse_spec
from workbench.registry.store import Registry

DATA = small_market()
R = DATA.returns
SAA_W = placeholder_saa()[R.columns]
ROLL = WindowSpec("rolling", periods=36)
COSTS = pd.Series(0.001, index=R.columns)  # 10 bps one-way
MONTHLY = rebalance_dates(R.index, ROLL, "M", "M")
QUARTERLY = rebalance_dates(R.index, ROLL, "Q", "M")


def _const(w):
    return lambda window, as_of: AllocationResult(w.reindex(window.columns), "ok")


def _cand_target(fn):
    """Allocator that holds SAA but sets the candidate to fn(as_of), funded pro rata."""

    def fit(window, as_of):
        x = fn(as_of)
        w = SAA_W * (1 - x)
        w["CAND"] = x
        return AllocationResult(w.reindex(window.columns), "ok")

    return fit


# --- costs ---------------------------------------------------------------------------------


def test_costs_by_hand_and_net_returns():
    eng = WalkForwardEngine(ROLL, costs=COSTS)
    path = eng.run(R, QUARTERLY, _const(SAA_W), SAA_W)
    assert path.cost.iloc[0] == 0.0  # initial entry not charged
    k = 3  # first period after the second (quarterly) rebalance
    expected = 0.001 * 2 * path.turnover.iloc[k]  # sum |dw| = 2 x one-way turnover
    assert path.cost.iloc[k] == pytest.approx(expected, abs=1e-15)
    pd.testing.assert_series_equal(path.returns_net, (path.returns - path.cost).rename(
        "portfolio_return_net"))  # fmt: skip
    assert (path.cost[path.turnover == 0] == 0).all()


def test_zero_costs_identical_to_no_costs():
    a = WalkForwardEngine(ROLL).run(R, QUARTERLY, _const(SAA_W), SAA_W)
    b = WalkForwardEngine(ROLL, costs=COSTS * 0).run(R, QUARTERLY, _const(SAA_W), SAA_W)
    pd.testing.assert_series_equal(a.returns, b.returns)
    assert (b.returns_net.to_numpy() == a.returns.to_numpy()).all()


def test_costs_missing_asset_rejected():
    with pytest.raises(ValueError, match="costs missing"):
        WalkForwardEngine(ROLL, costs=COSTS.drop("CAND")).run(R, QUARTERLY, _const(SAA_W), SAA_W)


# --- liquidity -----------------------------------------------------------------------------


def _cand_path(liq, fn, schedule=MONTHLY):
    eng = WalkForwardEngine(ROLL, liquidity=liq, candidate="CAND")
    path = eng.run(R, schedule, _cand_target(fn), SAA_W)
    executed = pd.Series({f.as_of: f.executed["CAND"] for f in path.fits})
    return path, executed


def test_candidate_trades_only_on_dealing_dates():
    quarter_ends = period_ends(R.index, "Q")
    # wants 10% in odd months, 0% in even months
    path, ex = _cand_path(LiquiditySpec("Q"), lambda t: 0.10 if t.month % 2 else 0.0)
    held_prev = None
    for f, (t, w) in zip(path.fits, ex.items(), strict=True):
        if held_prev is not None and t not in quarter_ends:
            assert f.liquidity_adjusted or abs(w - f.result.weights["CAND"]) < 1e-12
            # on non-dealing dates the candidate only drifts: executed == drifted held weight
        held_prev = w
    on_dealing = [t for t in ex.index[1:] if t in quarter_ends]
    assert all(abs(ex[t] - (0.10 if t.month % 2 else 0.0)) < 1e-12 for t in on_dealing)
    assert path.liquidity_adjusted.any()
    # other assets keep the target's proportions after rescaling
    f = next(f for f in path.fits if f.liquidity_adjusted)
    others = f.executed.drop("CAND")
    tgt = f.result.weights.drop("CAND")
    assert np.allclose(others / others.sum(), tgt / tgt.sum())
    assert f.executed.sum() == pytest.approx(1.0)


def test_notice_delays_redemption_by_dealing_dates():
    first = MONTHLY[0]
    exit_at = MONTHLY[12]  # a later quarter-end decision to exit

    def want(t):
        return 0.10 if t < exit_at else 0.0

    _, ex0 = _cand_path(LiquiditySpec("Q", notice_periods=0), want)
    _, ex1 = _cand_path(LiquiditySpec("Q", notice_periods=1), want)
    q = sorted(d for d in period_ends(R.index, "Q") if d >= exit_at and d in ex1.index)
    assert ex0[q[0]] == pytest.approx(0.0)  # immediate without notice
    assert ex1[q[0]] > 0.05  # still invested at the decision date
    assert ex1[q[1]] == pytest.approx(0.0)  # executed one dealing date later
    assert first in ex0.index


def test_gate_caps_redemption_per_dealing_date():
    exit_at = MONTHLY[12]
    _, ex = _cand_path(LiquiditySpec("Q", gate=0.25), lambda t: 0.10 if t < exit_at else 0.0,
                       schedule=QUARTERLY)  # fmt: skip
    q = [d for d in ex.index if d >= exit_at]
    # each dealing date redeems exactly 25% of the position held then (the gate binds)
    assert ex[q[0]] == pytest.approx(0.075, rel=0.01)  # 25% of the (slightly drifted) 10%
    assert ex[q[0]] > ex[q[1]] > ex[q[2]] > ex[q[3]] > 0
    assert ex[q[3]] < 0.10 * 0.75**4 * 1.1


def test_subscription_is_immediate_on_dealing_date():
    enter_at = QUARTERLY[6]
    _, ex = _cand_path(LiquiditySpec("Q", notice_periods=2, gate=0.1),
                       lambda t: 0.08 if t >= enter_at else 0.0, schedule=QUARTERLY)  # fmt: skip
    assert ex[enter_at] == pytest.approx(0.08)  # notice and gate apply to redemptions only
    assert (ex[ex.index < enter_at] == 0).all()


def test_adjusted_target_policy_breach_is_reported():
    # enters 10% each December (annual dealing), wants out the rest of the year but is frozen
    pol = policy(band=0.05, candidate_cap=0.10)
    eng = WalkForwardEngine(ROLL, liquidity=LiquiditySpec("A"), candidate="CAND",
                            check=pol.violations)  # fmt: skip
    path = eng.run(R, MONTHLY, _cand_target(lambda t: 0.10 if t.month == 12 else 0.0), SAA_W)
    breaches = [f for f in path.fits if f.adjustment_violations]
    assert breaches and all(f.liquidity_adjusted for f in breaches)
    assert any("CAND" in v and "band" in v for f in breaches for v in f.adjustment_violations)


# --- threshold -------------------------------------------------------------------------------


def test_threshold_trades_only_beyond_band():
    eng = WalkForwardEngine(ROLL, costs=COSTS, threshold=0.02)
    path = eng.run(R, MONTHLY, _const(SAA_W), SAA_W)
    calendar = WalkForwardEngine(ROLL, costs=COSTS).run(R, MONTHLY, _const(SAA_W), SAA_W)
    traded = [f for f in path.fits[1:] if f.traded]
    held = [f for f in path.fits[1:] if not f.traded]
    assert traded and held
    assert path.turnover.sum() < calendar.turnover.sum()
    assert path.cost.sum() < calendar.cost.sum()
    for f in held:  # no-trade dates execute nothing
        assert f.executed is not None


# --- funding -------------------------------------------------------------------------------


def test_named_funding_sources():
    ctx = fit_context(DATA, policy(candidate_cap=0.2))
    w = SAAPlus(0.10, funding="class:fixed_income").fit(R, ctx).weights
    assert w["CAND"] == pytest.approx(0.10) and w["GL_EQ"] == pytest.approx(0.30)
    assert w[["SE_GOV", "GL_IG_H", "HY"]].sum() == pytest.approx(0.40 - 0.10)
    w = SAAPlus(0.10, funding="asset:SE_GOV").fit(R, ctx).weights
    assert w["SE_GOV"] == pytest.approx(0.10)
    res = SAAPlus(0.10, funding="asset:HY").fit(R, ctx)
    assert res.status == "infeasible" and "funding source exhausted" in res.message
    with pytest.raises(ValueError, match="funding must be"):
        SAAPlus(0.1, funding="from_equity")
    res = SAAPlus(0.1, funding="class:crypto").fit(R, ctx)
    assert res.status == "exception" and "no building blocks" in res.message


# --- look-ahead with frictions ---------------------------------------------------------------


def test_no_look_ahead_with_frictions():
    alloc = RiskfolioMeanRisk(method_cov="ledoit", rm="CVaR", obj="MinRisk")
    base = fit_context(DATA)

    def fit(window, as_of):
        ctx = type(base)(as_of=as_of, saa=base.saa, candidate="CAND", policy=base.policy)
        return alloc.fit(window, ctx)

    cut = QUARTERLY[len(QUARTERLY) // 2]
    poisoned = R.copy()
    poisoned.loc[poisoned.index > cut] *= -2.0
    eng = WalkForwardEngine(ROLL, costs=COSTS, liquidity=LiquiditySpec("Q", 1, 0.5),
                            candidate="CAND", threshold=0.01)  # fmt: skip
    a, b = eng.run(R, QUARTERLY, fit, SAA_W), eng.run(poisoned, QUARTERLY, fit, SAA_W)
    for fa, fb in zip(a.fits, b.fits, strict=True):
        if fa.as_of <= cut:
            pd.testing.assert_series_equal(fa.executed, fb.executed)


# --- spec and runner integration ------------------------------------------------------------

FRICTION_SPEC = {
    "costs": {"default_bps": 10, "per_asset": {"EM_EQ": 25, "CAND": 0}},
    "liquidity": {"dealing": "Q", "notice_periods": 1, "gate": 0.5},
    "rebalance": {"kind": "threshold", "every": "M", "band": 0.01},
}


def _spec(**over):
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36})
    raw.update(over)
    return raw


def test_spec_parsing_and_hash_stability():
    plain = parse_spec(_spec())
    fr = parse_spec(_spec(**FRICTION_SPEC))
    assert fr.costs.bps("EM_EQ") == 25 and fr.costs.bps("GL_EQ") == 10
    assert fr.liquidity == LiquiditySpec("Q", 1, 0.5) and fr.rebalance.band == 0.01
    assert "costs" not in plain.canonical() and "band" not in plain.canonical()["rebalance"]
    assert fr.spec_hash != plain.spec_hash
    for bad, match in [
        ({"costs": {"default_bps": -1}}, "costs must be"),
        ({"liquidity": {"dealing": "W"}}, "liquidity.dealing"),
        ({"liquidity": {"gate": 1.5}}, "liquidity.gate"),
        ({"rebalance": {"kind": "threshold"}}, "rebalance.band"),
        ({"rebalance": {"band": 0.02}}, "only used with kind: threshold"),
        ({"costs": {"bps": 10}}, "unknown keys"),
    ]:
        with pytest.raises(SpecError, match=match):
            parse_spec(_spec(**bad))


@pytest.fixture(scope="module")
def friction_run(tmp_path_factory):
    raw = _spec(**FRICTION_SPEC)
    raw["grid"]["allocators"] = [
        {"type": "static_saa"},
        {"type": "saa_plus", "x": [0.05], "funding": ["pro_rata", "class:fixed_income"]},
        {"type": "riskfolio_mean_risk", "rm": ["CVaR"], "obj": ["MinRisk"]},
    ]
    reg = Registry(f"sqlite:///{tmp_path_factory.mktemp('fr') / 'r.db'}")
    return reg, run_experiment(parse_spec(raw), reg)


def test_runner_stores_net_paths_and_reports_frictions(friction_run):
    reg, s = friction_run
    oos = reg.oos_returns(s.experiment_id, include_reference=True)
    assert np.allclose(oos.portfolio_return_net, oos.portfolio_return - oos.cost)
    assert (oos.cost > 0).any()
    saa = oos[oos.config_id == "saa_reference"]
    assert (saa.cost > 0).any()  # the SAA pays for its own rebalancing
    t = oos_table(reg, s.experiment_id)
    cells = t[(t.row == "cell") & (t.data_variant == "full")]
    assert (cells.cost_drag_ann >= 0).all() and (cells.ann_return <= cells.ann_return_gross).all()
    assert (cells.n_liquidity_adjusted > 0).any()
    md = build_report(reg, s.experiment_id).summary_md
    assert "Frictions: costs 10 bps one-way (EM_EQ 25, CAND 0)" in md
    assert "candidate deals Q, notice 1 dealing date(s), gate 50%" in md
    assert "threshold rebalancing at 1.0% drift" in md


def test_funding_is_a_grid_dimension(friction_run):
    reg, s = friction_run
    plus = reg.cells(s.experiment_id).query("allocator == 'saa_plus'")
    assert set(plus.params_json.map(lambda p: yaml.safe_load(p)["funding"])) == {
        "pro_rata", "class:fixed_income"
    }  # fmt: skip


def test_unknown_cost_asset_is_a_spec_error(tmp_path):
    raw = copy.deepcopy(_spec(costs={"per_asset": {"BTC": 50}}))
    with pytest.raises(SpecError, match="unknown assets"):
        run_experiment(parse_spec(raw), Registry(f"sqlite:///{tmp_path / 'r.db'}"))

"""Walk-forward engine: timing, drift, failure policy and the no-look-ahead guarantee."""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import fit_context, policy, small_market
from workbench.allocators.base import AllocationResult
from workbench.allocators.riskfolio_hc import RiskfolioHC
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.backtest.schedule import rebalance_dates
from workbench.backtest.walkforward import WalkForwardEngine
from workbench.data.synthetic import placeholder_saa
from workbench.grid.spec import WindowSpec

DATA = small_market()
R = DATA.returns
SAA_W = placeholder_saa()[R.columns]
ROLL = WindowSpec("rolling", periods=36)


def _fit_with(allocator, pol=None):
    base = fit_context(DATA, pol)

    def fit(window, as_of):
        ctx = type(base)(as_of=as_of, saa=base.saa, candidate="CAND", policy=base.policy)
        return allocator.fit(window, ctx)

    return fit


def _const(w):
    return lambda window, as_of: AllocationResult(w.reindex(window.columns), "ok")


# --- no look-ahead ---------------------------------------------------------------------


def test_spy_never_sees_the_future():
    seen = []

    def spy(window, as_of):
        seen.append((as_of, window.index.max(), len(window)))
        return AllocationResult(SAA_W, "ok")

    sched = rebalance_dates(R.index, ROLL, "M", "M")
    WalkForwardEngine(ROLL).run(R, sched, spy, SAA_W)
    assert [s[0] for s in seen] == sched
    assert all(latest == as_of and n == 36 for as_of, latest, n in seen)


@pytest.mark.parametrize(
    "allocator",
    [RiskfolioMeanRisk(method_cov="ledoit", rm="CVaR", obj="MinRisk"), RiskfolioHC(model="HRP")],
    ids=["mean_risk", "hc"],
)
def test_future_perturbation_does_not_change_past_weights(allocator):
    """Truncation invariance: scrambling everything after T leaves every fit at t <= T unchanged."""
    sched = rebalance_dates(R.index, ROLL, "Q", "M")
    cut = sched[len(sched) // 2]
    poisoned = R.copy()
    future = poisoned.index > cut
    poisoned.loc[future] = -3.0 * poisoned.loc[future].to_numpy()[::-1] + 0.01
    eng = WalkForwardEngine(ROLL)
    a = eng.run(R, sched, _fit_with(allocator), SAA_W)
    b = eng.run(poisoned, sched, _fit_with(allocator), SAA_W)
    past_a = [f for f in a.fits if f.as_of <= cut]
    past_b = [f for f in b.fits if f.as_of <= cut]
    assert len(past_a) == len(past_b) > 2
    for fa, fb in zip(past_a, past_b, strict=True):
        pd.testing.assert_series_equal(fa.result.weights, fb.result.weights)
    # ... while later fits do react to the changed data
    later = [(fa, fb) for fa, fb in zip(a.fits, b.fits, strict=True) if fa.as_of > cut]
    assert any(not fa.result.weights.equals(fb.result.weights) for fa, fb in later)


def test_weights_fitted_at_t_earn_return_of_t_plus_1():
    sched = rebalance_dates(R.index, ROLL, "M", "M")
    path = WalkForwardEngine(ROLL).run(R, sched, _fit_with(RiskfolioHC()), SAA_W)
    for f in path.fits[:5]:
        nxt = R.index[R.index.get_loc(f.as_of) + 1]
        expected = float(f.result.weights[R.columns] @ R.loc[nxt])
        assert path.returns[nxt] == pytest.approx(expected, abs=1e-15)
    assert path.returns.index[0] == R.index[R.index.get_loc(sched[0]) + 1]


# --- drift and turnover ----------------------------------------------------------------


def test_drift_between_quarterly_rebalances_by_hand():
    sched = rebalance_dates(R.index, ROLL, "Q", "M")
    path = WalkForwardEngine(ROLL).run(R, sched, _const(SAA_W), SAA_W)
    t0 = sched[0]
    i = R.index.get_loc(t0)
    w = SAA_W.to_numpy().copy()
    for k in range(1, 4):  # three months until the next quarter-end rebalance
        r = R.iloc[i + k].to_numpy()
        port = w @ r
        assert path.returns.iloc[k - 1] == pytest.approx(port, abs=1e-15)
        w = w * (1 + r) / (1 + port)
    # at the next rebalance, turnover = half the L1 distance from drifted weights back to SAA
    expected_to = 0.5 * np.abs(SAA_W.to_numpy() - w).sum()
    assert path.turnover.iloc[3] == pytest.approx(expected_to, abs=1e-15)
    assert path.turnover.iloc[0] == 0.0  # initial entry not counted
    assert (path.turnover.iloc[1:3] == 0).all()  # no trading between rebalances


def test_monthly_constant_target_matches_static_portfolio():
    sched = rebalance_dates(R.index, ROLL, "M", "M")
    path = WalkForwardEngine(ROLL).run(R, sched, _const(SAA_W), SAA_W)
    expected = R.loc[path.returns.index] @ SAA_W
    pd.testing.assert_series_equal(path.returns, expected, check_names=False, atol=1e-15)


# --- failure policy (1a) ---------------------------------------------------------------


def test_never_fitting_strategy_is_exactly_the_saa_path():
    sched = rebalance_dates(R.index, ROLL, "Q", "M")
    fail = WalkForwardEngine(ROLL).run(
        R, sched, lambda w, t: AllocationResult(None, "infeasible", "x"), SAA_W
    )
    saa = WalkForwardEngine(ROLL).run(R, sched, _const(SAA_W), SAA_W)
    pd.testing.assert_series_equal(fail.returns, saa.returns)
    pd.testing.assert_series_equal(fail.turnover, saa.turnover)


def test_failed_fit_holds_drifted_weights_and_fallback_before_first_success():
    sched = rebalance_dates(R.index, ROLL, "Q", "M")
    target = pd.Series(1 / len(R.columns), index=R.columns)
    calls = {"n": 0}

    def flaky(window, as_of):
        calls["n"] += 1
        if calls["n"] in (1, 3):  # first and third fits fail
            return AllocationResult(None, "infeasible", "nope")
        return AllocationResult(target, "ok")

    path = WalkForwardEngine(ROLL).run(R, sched, flaky, SAA_W)
    assert path.n_failed == 2
    i0 = R.index.get_loc(sched[0])
    first_period = R.iloc[i0 + 1]
    assert path.returns.iloc[0] == pytest.approx(float(SAA_W @ first_period))  # fallback SAA
    # third fit failed: weights keep drifting, so no turnover at that rebalance
    j = path.returns.index.get_loc(R.index[R.index.get_loc(sched[2]) + 1])
    assert path.turnover.iloc[j] == 0.0


def test_engine_rejects_bad_schedules():
    eng = WalkForwardEngine(ROLL)
    with pytest.raises(ValueError, match="empty"):
        eng.run(R, [], _const(SAA_W), SAA_W)
    with pytest.raises(ValueError, match="no following period"):
        eng.run(R, [R.index[-1]], _const(SAA_W), SAA_W)
    with pytest.raises(ValueError, match="not in returns index"):
        eng.run(R, [pd.Timestamp("2016-01-15")], _const(SAA_W), SAA_W)


def test_policy_breach_at_one_date_is_held_not_dropped():
    sched = rebalance_dates(R.index, ROLL, "Q", "M")
    pol = policy(te_annual=0.001, candidate_cap=0.10)  # HRP cannot meet this
    path = WalkForwardEngine(ROLL).run(R, sched, _fit_with(RiskfolioHC(), pol), SAA_W)
    assert path.n_failed == len(sched)
    expected = R.loc[path.returns.index] @ SAA_W  # SAA held throughout (drifting quarterly)
    assert path.returns.iloc[0] == pytest.approx(expected.iloc[0])

"""Walk-forward engine: fit on the past at each rebalance date, hold with drift.

Timing: an allocator fitted at t sees only rows dated <= t (``window_at``) and its weights
earn the return of t+1 onwards. Between rebalances weights drift with asset returns.

Failure policy (Phase 1, option 1a): until the first successful fit, every scheduled date
rebalances to ``fallback`` (the SAA), so a strategy that never fits *is* the SAA path. After the
first success, a failed fit keeps the current drifted weights. Failed fits are still reported.

Frictions (P2-M2), all optional; without them the path is identical to Phase 1:

- costs: one-way cost per asset (decimal of traded weight), charged on sum c_i |target_i -
  held_i| at each trade and deducted from the next period's return (``returns_net``). The
  initial entry is not charged, like turnover.
- liquidity (the candidate only): it trades only on dealing dates; on other dates its weight
  stays at the drifted value and the other assets' targets are rescaled proportionally. A
  reduction decided on a dealing date executes ``notice_periods`` dealing dates later, and at
  most ``gate`` of the candidate position is redeemed per dealing date (the rest carries
  forward). Notice counts calendar dealing dates; executions happen at rebalance dates that
  are dealing dates. Redemption amounts are tracked in portfolio weight at request time.
- threshold: on a scheduled date the portfolio trades only if some asset's weight differs from
  the (liquidity-adjusted) target by more than ``threshold``; otherwise nothing executes.

The interface (``run(returns, schedule, fit_fn, fallback) -> PathResult``) stays narrow so an
external engine such as riskbench can stand in later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from workbench.allocators.base import AllocationResult
from workbench.backtest.schedule import period_ends, window_at
from workbench.grid.spec import LiquiditySpec, WindowSpec

FitFn = Callable[[pd.DataFrame, pd.Timestamp], AllocationResult]
CheckFn = Callable[[pd.Series, pd.DataFrame], list[str]]
WEIGHT_TOL = 1e-12


@dataclass(frozen=True)
class RebalanceFit:
    """One scheduled fit: the date, the window it saw, the allocator's result and what the
    engine executed (after liquidity and threshold rules)."""

    as_of: pd.Timestamp
    window: pd.DataFrame
    result: AllocationResult
    traded: bool = True
    liquidity_adjusted: bool = False
    adjustment_violations: tuple[str, ...] = ()
    executed: pd.Series | None = None  # weights held after the trade (rules applied)


@dataclass
class PathResult:
    """Out-of-sample path.

    returns:  gross portfolio simple return per period (decimal), dated by the period earned
              (first date is the period after the first rebalance).
    turnover: one-way turnover (decimal, 0.5 * sum |target - drifted|) of the trade that set
              each period's weights; 0 between trades and for the initial entry.
    cost:     transaction cost (decimal) charged to that period; 0 without costs.
    liquidity_adjusted: True for periods whose weights were changed by the liquidity rules.
    fits:     every scheduled fit, in date order.
    """

    returns: pd.Series
    turnover: pd.Series
    cost: pd.Series
    liquidity_adjusted: pd.Series
    fits: list[RebalanceFit] = field(default_factory=list)

    @property
    def returns_net(self) -> pd.Series:
        return (self.returns - self.cost).rename("portfolio_return_net")

    @property
    def n_failed(self) -> int:
        return sum(f.result.status != "ok" for f in self.fits)


@dataclass
class _Liquidity:
    spec: LiquiditySpec
    candidate: int
    dealing_index: dict  # dealing date -> its position in the dealing calendar
    pending: list = field(default_factory=list)  # [due_dealing_index, weight]


class WalkForwardEngine:
    def __init__(
        self,
        window: WindowSpec,
        costs: pd.Series | None = None,
        liquidity: LiquiditySpec | None = None,
        candidate: str | None = None,
        threshold: float | None = None,
        check: CheckFn | None = None,
    ) -> None:
        """costs: one-way cost per asset id (decimal of traded weight). check: policy check run
        on liquidity-adjusted targets (violations are reported, not enforced)."""
        if liquidity is not None and candidate is None:
            raise ValueError("liquidity needs the candidate id")
        self.window = window
        self.costs = costs
        self.liquidity = liquidity
        self.candidate = candidate
        self.threshold = threshold
        self.check = check

    def run(
        self,
        returns: pd.DataFrame,
        schedule: list[pd.Timestamp],
        fit_fn: FitFn,
        fallback: pd.Series,
    ) -> PathResult:
        """Walk forward over ``returns`` (simple returns per period, complete, sorted)."""
        if not schedule:
            raise ValueError("empty rebalance schedule")
        cols = list(returns.columns)
        r = returns.to_numpy(dtype=float)
        dates = returns.index
        pos = {d: i for i, d in enumerate(dates)}
        missing = [d for d in schedule if d not in pos]
        if missing:
            raise ValueError(f"schedule dates not in returns index: {missing[:3]}")
        if pos[schedule[-1]] >= len(dates) - 1:
            raise ValueError("last rebalance date has no following period")
        to_fit = set(schedule)
        start = pos[schedule[0]]
        cost_vec = None if self.costs is None else self.costs.reindex(cols).to_numpy(float)
        if cost_vec is not None and np.isnan(cost_vec).any():
            raise ValueError("costs missing for some assets")
        liq = None
        if self.liquidity is not None:
            dealing = sorted(period_ends(dates, self.liquidity.dealing))
            liq = _Liquidity(self.liquidity, cols.index(self.candidate),
                             {d: k for k, d in enumerate(dealing)})  # fmt: skip

        held: np.ndarray | None = None  # current (drifted) weights
        fitted_once = False
        fits: list[RebalanceFit] = []
        out_ret, out_to, out_cost, out_adj = [], [], [], []
        pending_turnover = pending_cost = 0.0
        pending_adj = False
        for i in range(start, len(dates) - 1):
            t = dates[i]
            if t in to_fit:
                window = window_at(returns, t, self.window)
                res = fit_fn(window, t)
                if res.status == "ok":
                    target = res.weights.reindex(cols).to_numpy(dtype=float)
                    fitted_once = True
                elif fitted_once:
                    target = held
                else:
                    target = fallback.reindex(cols).to_numpy(dtype=float)
                traded, adjusted, violations = True, False, ()
                if held is not None:
                    if liq is not None:
                        target, adjusted, commit = self._apply_liquidity(liq, t, target, held)
                    if self.threshold is not None and (
                        np.max(np.abs(target - held)) <= self.threshold + WEIGHT_TOL
                    ):
                        target, traded, adjusted = held, False, False
                    elif liq is not None:
                        commit()
                    if adjusted and self.check is not None:
                        violations = tuple(self.check(pd.Series(target, index=cols), window))
                fits.append(RebalanceFit(t, window, res, traded, adjusted, violations,
                                         pd.Series(target, index=cols)))  # fmt: skip
                if held is None:
                    pending_turnover = pending_cost = 0.0
                else:
                    trade = np.abs(target - held)
                    pending_turnover = 0.5 * float(trade.sum())
                    pending_cost = 0.0 if cost_vec is None else float(cost_vec @ trade)
                pending_adj = adjusted
                held = target
            nxt = r[i + 1]
            port = float(held @ nxt)
            out_ret.append(port)
            out_to.append(pending_turnover)
            out_cost.append(pending_cost)
            out_adj.append(pending_adj)
            pending_turnover = pending_cost = 0.0
            pending_adj = False
            held = held * (1.0 + nxt) / (1.0 + port)

        idx = dates[start + 1 :]
        return PathResult(
            returns=pd.Series(out_ret, index=idx, name="portfolio_return"),
            turnover=pd.Series(out_to, index=idx, name="turnover"),
            cost=pd.Series(out_cost, index=idx, name="cost"),
            liquidity_adjusted=pd.Series(out_adj, index=idx, name="liquidity_adjusted"),
            fits=fits,
        )

    @staticmethod
    def _apply_liquidity(liq: _Liquidity, t, target: np.ndarray, held: np.ndarray):
        """Candidate weight the liquidity terms allow at t; returns (target, adjusted, commit).

        ``commit`` applies the redemption-queue changes; it is only called if the trade executes.
        """
        c = liq.candidate
        held_c, want_c = float(held[c]), float(target[c])
        pending = [list(p) for p in liq.pending]
        if t in liq.dealing_index:
            k = liq.dealing_index[t]
            if want_c > held_c + WEIGHT_TOL:  # wants more than it holds: subscribe now
                pending = []
                new_c = want_c
            else:
                # queue exactly the gap between holding and target: trim if over-queued
                # (target raised, or drift shrank the position), top up if under-queued
                gap = held_c - want_c
                queued = sum(w for _, w in pending)
                if queued > gap + WEIGHT_TOL:
                    excess, trimmed = queued - gap, []
                    for d, w in reversed(pending):  # cancel the latest requests first
                        cut = min(w, excess)
                        excess -= cut
                        if w - cut > WEIGHT_TOL:
                            trimmed.append([d, w - cut])
                    pending = trimmed[::-1]
                elif gap - queued > WEIGHT_TOL:
                    pending.append([k + liq.spec.notice_periods, gap - queued])
                due = sum(w for d, w in pending if d <= k)
                cap = due if liq.spec.gate is None else min(due, liq.spec.gate * held_c)
                redeemed, remaining = cap, []
                for d, w in pending:
                    if d <= k:
                        take = min(w, redeemed)
                        redeemed -= take
                        if w - take > WEIGHT_TOL:
                            remaining.append([k + 1, w - take])  # carries to next dealing date
                    else:
                        remaining.append([d, w])
                pending = remaining
                new_c = held_c - cap
        else:
            new_c = held_c
        adjusted = abs(new_c - want_c) > WEIGHT_TOL
        if adjusted:
            others = np.arange(len(target)) != c
            rest = float(target[others].sum())
            out = target.copy()
            out[c] = new_c
            scale = (1.0 - new_c) / rest if rest > WEIGHT_TOL else 0.0
            out[others] = target[others] * scale
            if rest <= WEIGHT_TOL:  # target held only the candidate: keep the others as held
                out[others] = held[others] * (1.0 - new_c) / max(held[others].sum(), WEIGHT_TOL)
            target = out

        def commit():
            liq.pending = pending

        return target, adjusted, commit

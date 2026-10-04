"""Walk-forward engine: fit on the past at each rebalance date, hold with drift, no costs.

Timing: an allocator fitted at t sees only rows dated <= t (``window_at``) and its weights
earn the return of t+1 onwards. Between rebalances weights drift with asset returns.

Failure policy (Phase 1, option 1a): until the first successful fit, every scheduled date
rebalances to ``fallback`` (the SAA), so a strategy that never fits *is* the SAA path. After the
first success, a failed fit keeps the current drifted weights. Failed fits are still reported.

The interface (``run(returns, schedule, fit_fn, fallback) -> PathResult``) is deliberately
narrow so an external engine such as riskbench can stand in later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from workbench.allocators.base import AllocationResult
from workbench.backtest.schedule import window_at
from workbench.grid.spec import WindowSpec

FitFn = Callable[[pd.DataFrame, pd.Timestamp], AllocationResult]


@dataclass(frozen=True)
class RebalanceFit:
    """One scheduled fit: the date, the window it saw and the allocator's result."""

    as_of: pd.Timestamp
    window: pd.DataFrame
    result: AllocationResult


@dataclass
class PathResult:
    """Out-of-sample path.

    returns:  portfolio simple return per period (decimal), dated by the period earned
              (first date is the period after the first rebalance).
    turnover: one-way turnover (decimal, 0.5 * sum |target - drifted|) of the rebalance that
              set each period's weights; 0 between rebalances and for the initial entry.
    fits:     every scheduled fit, in date order.
    """

    returns: pd.Series
    turnover: pd.Series
    fits: list[RebalanceFit] = field(default_factory=list)

    @property
    def n_failed(self) -> int:
        return sum(f.result.status != "ok" for f in self.fits)


class WalkForwardEngine:
    def __init__(self, window: WindowSpec) -> None:
        self.window = window

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

        held: np.ndarray | None = None  # current (drifted) weights
        fitted_once = False
        fits: list[RebalanceFit] = []
        out_ret, out_to = [], []
        pending_turnover = 0.0
        for i in range(start, len(dates) - 1):
            t = dates[i]
            if t in to_fit:
                window = window_at(returns, t, self.window)
                res = fit_fn(window, t)
                fits.append(RebalanceFit(t, window, res))
                if res.status == "ok":
                    target = res.weights.reindex(cols).to_numpy(dtype=float)
                    fitted_once = True
                elif fitted_once:
                    target = held
                else:
                    target = fallback.reindex(cols).to_numpy(dtype=float)
                pending_turnover = 0.0 if held is None else 0.5 * float(np.abs(target - held).sum())
                held = target
            nxt = r[i + 1]
            port = float(held @ nxt)
            out_ret.append(port)
            out_to.append(pending_turnover)
            pending_turnover = 0.0
            held = held * (1.0 + nxt) / (1.0 + port)

        idx = dates[start + 1 :]
        return PathResult(
            returns=pd.Series(out_ret, index=idx, name="portfolio_return"),
            turnover=pd.Series(out_to, index=idx, name="turnover"),
            fits=fits,
        )

"""Estimation windows and calendar rebalance schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from workbench.units import periods_per_year

if TYPE_CHECKING:
    from workbench.grid.spec import WindowSpec

_PERIOD_ALIAS = {"M": "M", "Q": "Q", "A": "Y"}


def min_observations(window: WindowSpec) -> int:
    """Observations needed before the first fit."""
    return window.periods if window.kind == "rolling" else window.min_periods


def window_at(returns: pd.DataFrame, as_of: pd.Timestamp, window: WindowSpec) -> pd.DataFrame:
    """Estimation window ending at ``as_of`` (inclusive): only rows dated <= as_of.

    Raises ValueError if fewer than ``min_observations(window)`` rows are available.
    """
    past = returns.loc[returns.index <= as_of]
    need = min_observations(window)
    if len(past) < need:
        raise ValueError(f"insufficient history: {len(past)} < {need} periods")
    return past.iloc[-window.periods :] if window.kind == "rolling" else past


def period_ends(index: pd.DatetimeIndex, every: str) -> set[pd.Timestamp]:
    """Dates in ``index`` that are the last observation of their ``every`` period (M, Q, A)."""
    if every not in _PERIOD_ALIAS:
        raise ValueError(f"period must be one of {sorted(_PERIOD_ALIAS)}, got {every!r}")
    periods = index.to_period(_PERIOD_ALIAS[every])
    last_in_period = pd.Series(index, index=index).groupby(periods).transform("max")
    return set(last_in_period[last_in_period.index == last_in_period.to_numpy()].index)


def rebalance_dates(
    index: pd.DatetimeIndex,
    window: WindowSpec,
    every: str,
    data_freq: str,
    start: str | None = None,
) -> list[pd.Timestamp]:
    """Calendar rebalance dates within ``index``.

    A date qualifies if it is the last observation of its ``every`` period (M, Q or A), the
    window is full there, at least one later observation exists (weights apply from the next
    period) and it is on or after ``start``.
    """
    if every not in _PERIOD_ALIAS:
        raise ValueError(f"rebalance every must be one of {sorted(_PERIOD_ALIAS)}, got {every!r}")
    if periods_per_year(every) > periods_per_year(data_freq):
        raise ValueError(f"cannot rebalance every {every} on {data_freq} data")
    need = min_observations(window)
    if len(index) < need + 1:
        return []
    candidates = index[need - 1 : -1]
    ends = period_ends(index, every)
    dates = [d for d in candidates if d in ends]
    if start is not None:
        dates = [d for d in dates if d >= pd.Timestamp(start)]
    return dates

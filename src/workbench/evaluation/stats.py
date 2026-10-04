"""Ex-post statistics on a per-period simple-return series.

These are the CIO-facing definitions (compounded wealth, demeaned TE). They intentionally differ
from Riskfolio-Lib's lens definitions (uncompounded CDaR, RMS TE), which are used only for
additive risk shares and the policy post-check. Annualisation goes through ``workbench.units``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from workbench.units import periods_per_year, vol_period_to_annual

ALPHA = 0.05


def wealth(r: pd.Series) -> pd.Series:
    """Compounded wealth starting at 1 before the first period."""
    return (1.0 + r).cumprod()


def drawdowns(r: pd.Series) -> pd.Series:
    """Drawdown of compounded wealth per period (decimal, >= 0). Initial wealth 1 is a peak."""
    w = wealth(r).to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return pd.Series(1.0 - w / peak, index=r.index)


def ann_return(r: pd.Series, freq: str) -> float:
    """Geometric annualised return (decimal): (prod(1 + r))^(n / T) - 1."""
    total = float((1.0 + r).prod())
    return total ** (periods_per_year(freq) / len(r)) - 1.0


def ann_vol(r: pd.Series, freq: str) -> float:
    """Annualised volatility (decimal): std(ddof=1) * sqrt(n)."""
    return vol_period_to_annual(float(r.std(ddof=1)), freq)


def _tail_count(n: int, alpha: float) -> int:
    return max(1, math.ceil(alpha * n))


def cvar(r: pd.Series, alpha: float = ALPHA) -> float:
    """Historical CVaR per period (decimal loss, positive = loss): mean of the worst
    ceil(alpha * T) period returns, sign flipped. Not annualised."""
    worst = np.sort(r.to_numpy())[: _tail_count(len(r), alpha)]
    return float(-worst.mean())


def cdar(r: pd.Series, alpha: float = ALPHA) -> float:
    """Conditional drawdown at risk (decimal): mean of the worst ceil(alpha * T) compounded
    drawdowns."""
    dd = np.sort(drawdowns(r).to_numpy())[::-1][: _tail_count(len(r), alpha)]
    return float(dd.mean())


def max_drawdown(r: pd.Series) -> float:
    """Maximum drawdown of compounded wealth (decimal, >= 0)."""
    return float(drawdowns(r).max())


def tracking_error(r: pd.Series, bench: pd.Series, freq: str) -> float:
    """Annualised ex-post tracking error (decimal): std(ddof=1) of r - bench, times sqrt(n)."""
    return vol_period_to_annual(float((r - bench).std(ddof=1)), freq)


def summary(r: pd.Series, bench: pd.Series, freq: str) -> dict[str, float]:
    """The ex-post table's statistics for portfolio returns ``r`` vs benchmark ``bench``."""
    return {
        "ann_return": ann_return(r, freq),
        "ann_vol": ann_vol(r, freq),
        "cvar95": cvar(r),
        "cdar95": cdar(r),
        "max_dd": max_drawdown(r),
        "te_vs_saa": tracking_error(r, bench, freq),
    }

"""Annual <-> per-period conversions.

This is the ONLY module where annual figures are converted to the return frequency of an
experiment (or back). Riskfolio-Lib reads ``rf``, ``TE``, ``lowerret`` and CVaR-type limits in
the period of the returns, so every annual input passes through here before it reaches a solver.

Frequency codes: ``"D"`` (business days, 252/yr), ``"W"`` (52/yr), ``"M"`` (12/yr),
``"Q"`` (4/yr), ``"A"`` (1/yr).
"""

from __future__ import annotations

import math
from typing import Literal

PERIODS_PER_YEAR: dict[str, int] = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}

Compounding = Literal["geometric", "arithmetic"]


def periods_per_year(freq: str) -> int:
    """Number of return periods per year for frequency code ``freq``."""
    try:
        return PERIODS_PER_YEAR[freq]
    except KeyError:
        raise ValueError(
            f"unknown frequency {freq!r}; expected one of {sorted(PERIODS_PER_YEAR)}"
        ) from None


def return_annual_to_period(
    r_annual: float, freq: str, compounding: Compounding = "geometric"
) -> float:
    """Convert an annual return or rate (decimal, e.g. 0.03) to one period of ``freq``.

    ``geometric``: (1 + r)^(1/n) - 1, for compound rates such as ``rf`` or a CMA.
    ``arithmetic``: r / n, for arithmetic mean returns such as ``mu`` or ``lowerret``.
    Returns a decimal per-period return.
    """
    n = periods_per_year(freq)
    if compounding == "geometric":
        if r_annual <= -1.0:
            raise ValueError(f"annual return must be > -1, got {r_annual}")
        return (1.0 + r_annual) ** (1.0 / n) - 1.0
    if compounding == "arithmetic":
        return r_annual / n
    raise ValueError(f"unknown compounding {compounding!r}")


def return_period_to_annual(
    r_period: float, freq: str, compounding: Compounding = "geometric"
) -> float:
    """Inverse of :func:`return_annual_to_period`. Decimal per-period in, decimal annual out."""
    n = periods_per_year(freq)
    if compounding == "geometric":
        if r_period <= -1.0:
            raise ValueError(f"per-period return must be > -1, got {r_period}")
        return (1.0 + r_period) ** n - 1.0
    if compounding == "arithmetic":
        return r_period * n
    raise ValueError(f"unknown compounding {compounding!r}")


def vol_annual_to_period(sigma_annual: float, freq: str) -> float:
    """Annual volatility (decimal) to per-period volatility: sigma / sqrt(n)."""
    if sigma_annual < 0:
        raise ValueError(f"volatility must be >= 0, got {sigma_annual}")
    return sigma_annual / math.sqrt(periods_per_year(freq))


def vol_period_to_annual(sigma_period: float, freq: str) -> float:
    """Per-period volatility (decimal) to annual volatility: sigma * sqrt(n)."""
    if sigma_period < 0:
        raise ValueError(f"volatility must be >= 0, got {sigma_period}")
    return sigma_period * math.sqrt(periods_per_year(freq))


def te_annual_to_period(te_annual: float, freq: str) -> float:
    """Annual tracking-error limit (decimal) to the per-period ``TE`` that Riskfolio-Lib expects.

    Monthly: te_annual / sqrt(12).
    """
    return vol_annual_to_period(te_annual, freq)


def sharpe_period_to_annual(sr_period: float, freq: str) -> float:
    """Per-period Sharpe (or information) ratio to annual: SR * sqrt(n) (i.i.d. scaling)."""
    return sr_period * math.sqrt(periods_per_year(freq))


def sharpe_annual_to_period(sr_annual: float, freq: str) -> float:
    """Annual Sharpe ratio to per-period: SR / sqrt(n) (i.i.d. scaling)."""
    return sr_annual / math.sqrt(periods_per_year(freq))

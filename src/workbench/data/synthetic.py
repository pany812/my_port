"""Synthetic market: placeholder SAA building blocks plus a configurable candidate.

Building blocks are drawn jointly from a fixed correlation matrix (Gaussian, or multivariate
Student-t when ``tail_df`` is set). The candidate is built from the reference equity block's
standardised shock and an independent skew-normal shock, so its correlation to equities and its
skewness hit their targets exactly in population (skewness only for Gaussian blocks).

All annual inputs are converted to the return frequency in ``workbench.units``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from workbench.data.base import MarketData
from workbench.units import return_annual_to_period, vol_annual_to_period

log = logging.getLogger(__name__)

# Placeholder SAA (synthetic phase only). mu_annual: expected annual return; the per-period mean
# is (1 + mu)^(1/n) - 1, so realised CAGR is about mu - vol^2/2. vol_annual: annual volatility.
# weight/min/max: policy weight and range. Decimals throughout.
# fmt: off
PLACEHOLDER_SAA = pd.DataFrame(
    [
        # asset_id,   asset_class,    mu,    vol,   weight, min,  max
        ("SE_EQ",    "equity",       0.070, 0.180, 0.15, 0.10, 0.20),
        ("GL_EQ",    "equity",       0.070, 0.150, 0.30, 0.25, 0.35),
        ("EM_EQ",    "equity",       0.080, 0.200, 0.05, 0.00, 0.08),
        ("SE_GOV",   "fixed_income", 0.025, 0.040, 0.20, 0.15, 0.25),
        ("GL_IG_H",  "fixed_income", 0.030, 0.050, 0.15, 0.10, 0.20),
        ("HY",       "fixed_income", 0.050, 0.090, 0.05, 0.00, 0.08),
        ("REAL_EST", "real_assets",  0.060, 0.160, 0.05, 0.00, 0.08),
        ("CASH",     "cash",         0.020, 0.005, 0.05, 0.00, 0.10),
    ],
    columns=["asset_id", "asset_class", "mu_annual", "vol_annual", "weight", "min", "max"],
).set_index("asset_id")

PLACEHOLDER_CORR = pd.DataFrame(
    [
        [1.00, 0.80, 0.70, -0.20, 0.00, 0.55, 0.55, 0.00],
        [0.80, 1.00, 0.75, -0.25, 0.05, 0.60, 0.60, 0.00],
        [0.70, 0.75, 1.00, -0.15, 0.10, 0.60, 0.50, 0.00],
        [-0.20, -0.25, -0.15, 1.00, 0.70, 0.10, 0.00, 0.10],
        [0.00, 0.05, 0.10, 0.70, 1.00, 0.45, 0.15, 0.10],
        [0.55, 0.60, 0.60, 0.10, 0.45, 1.00, 0.50, 0.00],
        [0.55, 0.60, 0.50, 0.00, 0.15, 0.50, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.10, 0.10, 0.00, 0.00, 1.00],
    ],
    index=PLACEHOLDER_SAA.index,
    columns=PLACEHOLDER_SAA.index,
)

# fmt: on

_PANDAS_FREQ = {"D": "B", "W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}
_MAX_SKEWNORM_SKEW = 0.99  # skew-normal supports |skew| < ~0.9953


@dataclass(frozen=True)
class CandidateSpec:
    """Synthetic candidate strategy.

    asset_id:        column name.
    mu_annual:       expected annual return (decimal); per-period mean is (1 + mu)^(1/n) - 1.
    vol_annual:      annual volatility (decimal).
    corr_to_equity:  correlation with ``equity_ref`` (in [-1, 1]).
    skew:            skewness of per-period returns (Gaussian blocks: exact in population).
    live_start:      first live period (e.g. "2019-01"); earlier observations are flagged as
                     backfilled. None = fully live history.
    backfill:        True keeps pre-live observations (flagged); False sets them to NaN.
    equity_ref:      building block whose shock defines "equities".
    asset_class:     class label.
    """

    asset_id: str = "CAND"
    mu_annual: float = 0.05
    vol_annual: float = 0.10
    corr_to_equity: float = 0.0
    skew: float = 0.3
    live_start: str | None = None
    backfill: bool = True
    equity_ref: str = "GL_EQ"
    asset_class: str = "alternatives"


def placeholder_saa(candidate: str = "CAND") -> pd.Series:
    """Placeholder SAA weights (sum 1) with the candidate appended at 0."""
    w = PLACEHOLDER_SAA["weight"].astype(float).copy()
    w[candidate] = 0.0
    w.name = "saa_weight"
    return w


def generate(
    seed: int,
    start: str = "2006-01",
    end: str = "2026-09",
    freq: str = "M",
    candidate: CandidateSpec | None = None,
    tail_df: float | None = None,
) -> MarketData:
    """Draw a synthetic market of simple total returns per period of ``freq``.

    seed:     RNG seed; same arguments and seed => identical output.
    start/end: inclusive period range (anything ``pd.Timestamp`` parses; "YYYY-MM" for monthly).
    tail_df:  degrees of freedom (> 2) for multivariate-t building blocks, scaled to unit
              variance so volatilities are unchanged. None = Gaussian.
    """
    cand = candidate or CandidateSpec()
    _validate(cand, tail_df)

    index = _period_index(start, end, freq)
    n = len(index)
    if n < 2:
        raise ValueError(f"need at least 2 periods between {start} and {end}")
    rng = np.random.default_rng(seed)

    blocks = PLACEHOLDER_SAA.index
    chol = np.linalg.cholesky(PLACEHOLDER_CORR.loc[blocks, blocks].to_numpy())
    z = rng.standard_normal((n, len(blocks))) @ chol.T
    if tail_df is not None:
        scale = np.sqrt(rng.chisquare(tail_df, size=(n, 1)) / (tail_df - 2.0))
        z = z / scale

    e = z[:, blocks.get_loc(cand.equity_ref)]
    u = _standard_skewnormal(_idio_skew(cand), n, rng)
    rho = cand.corr_to_equity
    zc = rho * e + math.sqrt(1.0 - rho**2) * u

    mu = [return_annual_to_period(m, freq) for m in PLACEHOLDER_SAA["mu_annual"]]
    vol = [vol_annual_to_period(s, freq) for s in PLACEHOLDER_SAA["vol_annual"]]
    r_blocks = np.asarray(mu) + z * np.asarray(vol)
    r_cand = return_annual_to_period(cand.mu_annual, freq) + zc * vol_annual_to_period(
        cand.vol_annual, freq
    )

    returns = pd.DataFrame(r_blocks, index=index, columns=list(blocks))
    returns[cand.asset_id] = r_cand
    if (returns <= -1.0).any().any():
        raise ValueError("synthetic draw produced a return <= -100%; lower vol or change seed")

    backfilled = pd.Series(False, index=index, name=f"{cand.asset_id}_backfilled")
    if cand.live_start is not None:
        pre_live = index < pd.Timestamp(cand.live_start)
        if cand.backfill:
            backfilled[pre_live] = True
        else:
            returns.loc[pre_live, cand.asset_id] = np.nan

    asset_class = PLACEHOLDER_SAA["asset_class"].copy()
    asset_class[cand.asset_id] = cand.asset_class
    log.debug("synthetic market: %d periods x %d assets, seed=%d", n, returns.shape[1], seed)
    return MarketData(
        returns=returns,
        freq=freq,
        candidate=cand.asset_id,
        asset_class=asset_class,
        backfilled=backfilled,
    )


def _validate(cand: CandidateSpec, tail_df: float | None) -> None:
    if cand.asset_id in PLACEHOLDER_SAA.index:
        raise ValueError(f"candidate id {cand.asset_id!r} collides with a building block")
    if cand.equity_ref not in PLACEHOLDER_SAA.index:
        raise ValueError(f"equity_ref {cand.equity_ref!r} is not a building block")
    if not -1.0 <= cand.corr_to_equity <= 1.0:
        raise ValueError(f"corr_to_equity must be in [-1, 1], got {cand.corr_to_equity}")
    if cand.vol_annual <= 0:
        raise ValueError(f"vol_annual must be > 0, got {cand.vol_annual}")
    if tail_df is not None and tail_df <= 2:
        raise ValueError(f"tail_df must be > 2, got {tail_df}")
    _idio_skew(cand)


def _idio_skew(cand: CandidateSpec) -> float:
    """Skew the idiosyncratic shock needs so the candidate's total skew equals ``cand.skew``.

    skew(rho*e + sqrt(1-rho^2)*u) = (1-rho^2)^1.5 * skew(u) for Gaussian e independent of u.
    """
    if cand.skew == 0:
        return 0.0
    w = (1.0 - cand.corr_to_equity**2) ** 1.5
    needed = cand.skew / w if w > 0 else math.inf
    if abs(needed) > _MAX_SKEWNORM_SKEW:
        raise ValueError(
            f"skew {cand.skew} unattainable with corr_to_equity {cand.corr_to_equity}: "
            f"idiosyncratic skew would be {needed:.3f}, limit is +/-{_MAX_SKEWNORM_SKEW}"
        )
    return needed


def _skewnorm_skew(delta: float) -> float:
    b = delta * math.sqrt(2.0 / math.pi)
    return (4.0 - math.pi) / 2.0 * b**3 / (1.0 - b**2) ** 1.5


def _standard_skewnormal(skew: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws with mean 0, variance 1 and the given skewness (skew-normal family)."""
    if skew == 0:
        return rng.standard_normal(n)
    delta = _bisect(lambda d: _skewnorm_skew(d) - skew, -0.999999, 0.999999)
    # Azzalini representation: delta*|z0| + sqrt(1-delta^2)*z1 is skew-normal.
    z0 = np.abs(rng.standard_normal(n))
    z1 = rng.standard_normal(n)
    x = delta * z0 + math.sqrt(1.0 - delta**2) * z1
    mean = delta * math.sqrt(2.0 / math.pi)
    sd = math.sqrt(1.0 - 2.0 * delta**2 / math.pi)
    return (x - mean) / sd


def _bisect(f, lo: float, hi: float, tol: float = 1e-12) -> float:
    """Root of an increasing function ``f`` on [lo, hi]."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _period_index(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    try:
        pfreq = _PANDAS_FREQ[freq]
    except KeyError:
        raise ValueError(
            f"unknown frequency {freq!r}; expected one of {sorted(_PANDAS_FREQ)}"
        ) from None
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if freq in ("M", "Q", "A"):
        e = e + pd.offsets.MonthEnd(0)  # "2026-09" -> 2026-09-30
    return pd.date_range(s, e, freq=pfreq, name="date")

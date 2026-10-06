"""Risk budgets for the candidate and realised risk shares, shared by both libraries.

The budget gives the candidate ``candidate_share`` of portfolio risk under ``rm``; the rest goes
to the other assets either equally (``rest="equal"``) or in proportion to their risk
contributions inside the SAA (``rest="saa"``). Risk budgeting needs strictly positive budgets,
and SAA contributions can be negative (diversifiers such as government bonds), so each other
asset gets at least ``BUDGET_FLOOR`` of the remaining budget before renormalising.

Realised shares use Riskfolio-Lib's ``Risk_Contribution`` with the same covariance the budget
was built on. They match the target exactly for smooth measures (MV, MSV); for CVaR and CDaR,
which are piecewise linear on historical scenarios, risk contributions are not unique and the
realised share differs from the target even at the optimum (both libraries agree on the weights).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import riskfolio as rp

from workbench.allocators._estimates import covariance

__all__ = ["BUDGET_FLOOR", "RESTS", "contributions", "covariance", "realised_share",
           "risk_budget"]  # fmt: skip

BUDGET_FLOOR = 0.01  # minimum fraction of the non-candidate budget per asset
RESTS = ("saa", "equal")
ALPHA = 0.05


def contributions(
    w: pd.Series, returns: pd.DataFrame, cov: pd.DataFrame, rm: str, rf: float
) -> np.ndarray:
    """Euler risk contributions of ``w`` under ``rm`` (Riskfolio definitions)."""
    cols = list(returns.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = rp.Risk_Contribution(
            w.reindex(cols).to_frame("weights"), returns, cov=cov, rm=rm, rf=rf, alpha=ALPHA
        )
    return np.ravel(rc)


def risk_budget(
    returns: pd.DataFrame,
    saa: pd.Series,
    candidate: str,
    candidate_share: float,
    rm: str,
    rest: str,
    cov: pd.DataFrame,
    rf: float,
) -> pd.Series:
    """Budget per asset (sums to 1): the candidate's share, the rest by ``rest``."""
    if not 0 < candidate_share < 1:
        raise ValueError(f"candidate_share must be in (0, 1), got {candidate_share}")
    if rest not in RESTS:
        raise ValueError(f"rest must be one of {RESTS}, got {rest!r}")
    cols = list(returns.columns)
    others = [a for a in cols if a != candidate]
    if rest == "equal":
        base = pd.Series(1.0, index=others)
    else:
        rc = pd.Series(contributions(saa, returns, cov, rm, rf), index=cols)[others]
        base = rc.clip(lower=0.0)
        base = base / base.sum() if base.sum() > 0 else pd.Series(1.0, index=others)
    base = base / base.sum()
    base = np.maximum(base, BUDGET_FLOOR)
    base = base / base.sum() * (1.0 - candidate_share)
    b = pd.concat([base, pd.Series({candidate: candidate_share})]).reindex(cols)
    return b


def realised_share(
    w: pd.Series, returns: pd.DataFrame, cov: pd.DataFrame, rm: str, rf: float, candidate: str
) -> float:
    rc = contributions(w, returns, cov, rm, rf)
    total = rc.sum()
    return float(rc[list(returns.columns).index(candidate)] / total) if total else float("nan")

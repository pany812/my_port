"""Translate a :class:`CompiledPolicy` into skfolio 1.4.11 estimator arguments.

Mean-risk: per-asset bounds -> ``min_weights``/``max_weights``; class limits -> ``groups`` +
``linear_constraints``; band -> ``max_turnover`` with ``previous_weights`` = SAA (element-wise,
same semantics as Riskfolio's ``allowTO``); TE -> ``max_tracking_error`` with ``y = R @ SAA``
passed to ``fit`` (non-demeaned RMS / (T - 1), same as Riskfolio's TE). HC: bounds intersected
with the band box. Class limits and TE are not enforceable in HC; the post-check catches them.

skfolio raises the same ``SolverError`` for infeasible and numerically failed problems, so
:func:`linear_infeasibility` decides which one it was.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from workbench.policy.compiled import LINEAR_TOL, CompiledPolicy

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def linear_kwargs(policy: CompiledPolicy, assets: list[str]) -> dict:
    """Per-asset bounds and class limits for skfolio estimators (MeanRisk, RiskBudgeting)."""
    if not policy.long_only:
        raise ValueError("the skfolio backend supports long-only policies only")
    lo, hi = policy.bounds(assets)
    kw: dict = {
        "min_weights": {a: float(lo[a]) for a in assets},
        "max_weights": {a: float(hi[a]) for a in assets},
        "budget": 1.0,
        "risk_free_rate": policy.rf,
    }
    if policy.class_limits:
        classes = policy.asset_class.reindex(assets)
        bad = [c for c in policy.class_limits if not _IDENT.match(c)]
        if bad:
            raise ValueError(f"class names {bad} are not valid skfolio constraint identifiers")
        kw["groups"] = {a: [str(classes[a])] for a in assets}
        cons = []
        for cls, (clo, chi) in policy.class_limits.items():
            if clo > 0:
                cons.append(f"{cls} >= {clo!r}")
            if chi < 1:
                cons.append(f"{cls} <= {chi!r}")
        kw["linear_constraints"] = cons
    return kw


def mean_risk_kwargs(policy: CompiledPolicy, assets: list[str]) -> dict:
    """Constraint arguments for ``skfolio.optimization.MeanRisk`` (per-period units).

    Risk caps map to ``max_standard_deviation``, ``max_cvar``, ``max_cdar`` and ``min_return``
    (same definitions as Riskfolio, verified). skfolio has no risk-contribution constraint, so
    the candidate risk-share cap is enforced by the post-check only.
    """
    kw = linear_kwargs(policy, assets)
    if policy.band is not None:
        kw["max_turnover"] = policy.band
        kw["previous_weights"] = _bench(policy, assets)
    if policy.te is not None:
        kw["max_tracking_error"] = policy.te
    if policy.max_vol is not None:
        kw["max_standard_deviation"] = policy.max_vol
    if policy.max_cvar is not None:
        kw["max_cvar"] = policy.max_cvar
    if policy.max_cdar is not None:
        kw["max_cdar"] = policy.max_cdar
    if policy.min_return is not None:
        kw["min_return"] = policy.min_return
    return kw


def tracking_target(policy: CompiledPolicy, returns: pd.DataFrame) -> pd.Series | None:
    """``y`` for ``MeanRisk.fit``: SAA returns per period, or None when no TE limit applies."""
    if policy.te is None:
        return None
    w = policy.benchweights.reindex(returns.columns)
    return returns @ w


def hc_kwargs(policy: CompiledPolicy, assets: list[str]) -> dict:
    """``min_weights``/``max_weights`` for skfolio hierarchical estimators."""
    lo, hi = policy.box_bounds(assets)
    return {
        "min_weights": {a: float(lo[a]) for a in assets},
        "max_weights": {a: float(hi[a]) for a in assets},
    }


def linear_infeasibility(policy: CompiledPolicy, assets: list[str]) -> str | None:
    """Why the policy's linear constraints admit no fully-invested long-only portfolio, or None.

    The SAA has zero TE and zero band deviation, so if it satisfies the linear constraints the
    problem is feasible. Otherwise an LP over bounds, band box, class limits and budget decides.
    """
    bench = _bench_series(policy, assets)
    if bench is not None and not _linear_breaches(policy, assets, bench):
        return None
    lo, hi = policy.box_bounds(assets)
    a_ub, b_ub = [], []
    if policy.class_limits:
        classes = policy.asset_class.reindex(assets).to_numpy()
        for cls, (clo, chi) in policy.class_limits.items():
            row = (classes == cls).astype(float)
            a_ub += [row, -row]
            b_ub += [chi, -clo]
    res = linprog(
        c=np.zeros(len(assets)),
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.ones((1, len(assets))),
        b_eq=np.array([1.0]),
        bounds=list(zip(lo.to_numpy(), hi.to_numpy(), strict=True)),
        method="highs",
    )
    if res.status == 2:
        return "linear constraints (bounds, band, class limits, budget) are infeasible"
    return None


def _linear_breaches(policy: CompiledPolicy, assets: list[str], w: pd.Series) -> bool:
    lo, hi = policy.box_bounds(assets)
    if ((w < lo - LINEAR_TOL) | (w > hi + LINEAR_TOL)).any():
        return True
    if policy.class_limits:
        by = w.groupby(policy.asset_class.reindex(assets)).sum()
        for cls, (clo, chi) in policy.class_limits.items():
            cw = float(by.get(cls, 0.0))
            if cw < clo - LINEAR_TOL or cw > chi + LINEAR_TOL:
                return True
    return False


def _bench_series(policy: CompiledPolicy, assets: list[str]) -> pd.Series | None:
    return None if policy.benchweights is None else policy.benchweights.reindex(assets)


def _bench(policy: CompiledPolicy, assets: list[str]) -> dict:
    return {a: float(v) for a, v in _bench_series(policy, assets).items()}

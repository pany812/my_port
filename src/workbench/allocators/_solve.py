"""Shared fit wrapper: input checks, stdout capture, status mapping, cleaning and post-check.

Every allocator routes its ``fit`` through :func:`guarded_fit`, so all cells are treated alike:

- stdout (Riskfolio-Lib prints on infeasibility) is captured into ``message``;
- warnings are silenced (cvxpy deprecation noise);
- ``None`` from the solver -> ``infeasible``; ``cvxpy.SolverError`` -> ``solver_error``;
  any other exception -> ``exception``;
- weights are reindexed to the return columns, tiny negatives clipped and renormalised;
- the result is checked against the policy; any breach -> ``infeasible`` (policy option A).
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
import warnings
from collections.abc import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from workbench.allocators.base import AllocationResult, FitContext, weights_from_riskfolio

log = logging.getLogger(__name__)

CLIP_TOL = 1e-7  # negatives above -CLIP_TOL are solver noise in a long-only problem

RawWeights = pd.Series | pd.DataFrame | None
FitImpl = Callable[[pd.DataFrame, FitContext, dict], RawWeights]


class Infeasible(Exception):
    """Raised inside a fit implementation when the problem is infeasible before solving."""


def validate_inputs(returns: pd.DataFrame, ctx: FitContext) -> None:
    """No NaNs, assets match the SAA, and no observation dated after ``ctx.as_of``."""
    if returns.empty or len(returns) < 2:
        raise ValueError("need at least 2 return observations")
    if returns.isna().any().any():
        cols = list(returns.columns[returns.isna().any()])
        raise ValueError(f"returns contain NaN in {cols}; align history before fitting")
    if set(returns.columns) != set(ctx.saa.index):
        raise ValueError("returns columns must match the SAA assets")
    if returns.index.max() > ctx.as_of:
        raise ValueError(f"look-ahead: returns extend past as_of {ctx.as_of.date()}")


def clean_weights(w: pd.Series, long_only: bool) -> tuple[pd.Series, float]:
    """Clip solver-noise negatives and renormalise. Returns (weights, max absolute change)."""
    orig = w.astype(float)
    out = orig.copy()
    if long_only:
        out[(out < 0) & (out > -CLIP_TOL)] = 0.0
    total = out.sum()
    if total > 0:
        out = out / total
    return out, float((out - orig).abs().max())


def guarded_fit(impl: FitImpl, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
    """Run ``impl(returns, ctx, diagnostics)`` and map every outcome to an AllocationResult."""
    t0 = time.perf_counter()
    buf = io.StringIO()
    diag: dict = {}

    def done(weights, status, message="") -> AllocationResult:
        captured = buf.getvalue().strip()
        msg = "\n".join(m for m in (message, captured) if m)
        return AllocationResult(weights, status, msg, time.perf_counter() - t0, diag)

    try:
        validate_inputs(returns, ctx)
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = impl(returns, ctx, diag)
    except Infeasible as e:
        return done(None, "infeasible", str(e))
    except cp.error.SolverError as e:
        return done(None, "solver_error", f"SolverError: {e}")
    except Exception as e:  # recorded, never dropped
        log.debug("fit raised", exc_info=True)
        return done(None, "exception", f"{type(e).__name__}: {e}")

    if raw is None:
        return done(None, "infeasible", "solver returned no solution")
    try:
        w = weights_from_riskfolio(raw) if isinstance(raw, pd.DataFrame) else raw.astype(float)
        assets = list(returns.columns)
        if set(w.index) != set(assets):
            raise ValueError(f"weights index {list(w.index)} does not match assets {assets}")
        w = w.reindex(assets)
        if not np.isfinite(w.to_numpy()).all():
            raise ValueError("non-finite weights")
        w, adj = clean_weights(w, ctx.policy.long_only)
        w.name = "weight"
    except Exception as e:
        return done(None, "exception", f"{type(e).__name__}: {e}")
    diag["clean_adjustment"] = adj

    breaches = ctx.policy.violations(w, returns)
    if breaches:
        diag["violations"] = breaches
        diag["rejected_weights"] = {k: float(v) for k, v in w.items()}
        return done(None, "infeasible", "policy violations: " + "; ".join(breaches))
    return done(w, "ok")

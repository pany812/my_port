"""Guards on Riskfolio-Lib estimates before they reach cvxpy, and the CMA mean.

Riskfolio-Lib 7.4.0 ``method_mu="JS"`` returns a complex128 mu (eigenvalues from
``np.linalg.eig``) with zero imaginary part; cvxpy then rejects the problem. We cast to real
when the imaginary part is negligible and fail loudly otherwise.

``method_mu="cma"`` (P2-M4) means the expected returns come from the spec's capital market
assumptions, which the runner passes as ``ctx.mu_override``. Library estimate steps then run
with ``"hist"`` (the mean is replaced afterwards); without an override the fit fails loudly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import riskfolio as rp

from workbench.allocators.base import FitContext

IMAG_TOL = 1e-12
CMA_MU = "cma"


def as_real(x: pd.DataFrame | pd.Series, what: str) -> tuple[pd.DataFrame | pd.Series, bool]:
    """Return (real-valued copy, whether a cast happened). Raise if the imaginary part matters."""
    values = np.asarray(x)
    if not np.iscomplexobj(values):
        return x, False
    imag = float(np.abs(values.imag).max()) if values.size else 0.0
    if imag > IMAG_TOL:
        raise ValueError(f"{what} has non-negligible imaginary part ({imag:.3e})")
    return x.apply(np.real).astype(float), True


def expected_returns(returns: pd.DataFrame, method_mu: str) -> tuple[pd.Series, bool]:
    """Riskfolio-Lib mean vector per period (decimal) as a real pd.Series indexed by asset."""
    mu = rp.ParamsEstimation.mean_vector(returns, method=method_mu)
    mu, cast = as_real(mu, f"mu ({method_mu})")
    return pd.Series(np.asarray(mu, dtype=float).ravel(), index=returns.columns), cast


def covariance(returns: pd.DataFrame, method_cov: str) -> pd.DataFrame:
    """Riskfolio covariance estimate (per period), real-valued."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = rp.ParamsEstimation.covar_matrix(returns, method=method_cov)
    cov, _ = as_real(pd.DataFrame(cov, index=returns.columns, columns=returns.columns), "cov")
    return cov


def estimate_mu_method(method_mu: str, ctx: FitContext) -> str:
    """The ``method_mu`` to pass to a library's estimate step.

    ``"cma"`` -> ``"hist"`` (replaced by ``ctx.mu_override`` afterwards), and the override must
    be present. Any other name passes through unchanged.
    """
    if method_mu != CMA_MU:
        return method_mu
    if ctx.mu_override is None:
        raise ValueError("method_mu 'cma' needs ctx.mu_override (set by the runner from the "
                         "spec's cma section)")  # fmt: skip
    return "hist"

"""Guards on Riskfolio-Lib estimates before they reach cvxpy.

Riskfolio-Lib 7.4.0 ``method_mu="JS"`` returns a complex128 mu (eigenvalues from
``np.linalg.eig``) with zero imaginary part; cvxpy then rejects the problem. We cast to real
when the imaginary part is negligible and fail loudly otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp

IMAG_TOL = 1e-12


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

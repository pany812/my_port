"""Riskfolio-Lib mean-risk allocator on ``model="Classic"``."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import riskfolio as rp

from workbench.allocators._estimates import as_real
from workbench.allocators._solve import guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.riskfolio import apply_mean_risk


@dataclass(frozen=True)
class RiskfolioMeanRisk:
    """Mean-risk optimisation with Riskfolio-Lib.

    method_mu / method_cov: Riskfolio-Lib estimators for ``assets_stats``.
    rm:  risk measure code (e.g. "MV", "CVaR", "CDaR").
    obj: "MinRisk", "Utility", "Sharpe" or "MaxRet".
    l:   risk-aversion for obj="Utility" (dimensionless).
    rf comes from ``ctx.policy.rf`` (per period); ``ctx.mu_override`` replaces mu (per period).
    """

    method_mu: str = "hist"
    method_cov: str = "hist"
    rm: str = "MV"
    obj: str = "Sharpe"
    l: float = 2.0  # noqa: E741 - Riskfolio-Lib's name
    name = "riskfolio_mean_risk"

    def params(self) -> dict:
        return asdict(self)

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r: pd.DataFrame, c: FitContext, diag: dict):
            assets = list(r.columns)
            port = rp.Portfolio(returns=r)
            port.assets_stats(method_mu=self.method_mu, method_cov=self.method_cov)
            port.mu, diag["mu_cast_from_complex"] = as_real(port.mu, f"mu ({self.method_mu})")
            port.cov, diag["cov_cast_from_complex"] = as_real(port.cov, f"cov ({self.method_cov})")
            if c.mu_override is not None:
                port.mu = c.mu_override.to_frame().T[assets]
            apply_mean_risk(port, c.policy, assets)
            diag["solvers"] = list(c.policy.solvers)
            return port.optimization(
                model="Classic", rm=self.rm, obj=self.obj, rf=c.policy.rf, l=self.l, hist=True
            )

        return guarded_fit(impl, returns, ctx)

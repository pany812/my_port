"""Riskfolio-Lib mean-risk allocator on ``model="Classic"``."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import riskfolio as rp

from workbench.allocators._estimates import as_real, estimate_mu_method
from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.riskfolio import apply_mean_risk
from workbench.policy.skfolio import sharpe_undefined


@dataclass(frozen=True)
class RiskfolioMeanRisk:
    """Mean-risk optimisation with Riskfolio-Lib.

    method_mu / method_cov: Riskfolio-Lib estimators for ``assets_stats``; ``method_mu="cma"``
                            takes the mean from the spec's CMA (``ctx.mu_override``).
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
            mu_method = estimate_mu_method(self.method_mu, c)
            port.assets_stats(method_mu=mu_method, method_cov=self.method_cov)
            port.mu, diag["mu_cast_from_complex"] = as_real(port.mu, f"mu ({mu_method})")
            port.cov, diag["cov_cast_from_complex"] = as_real(port.cov, f"cov ({self.method_cov})")
            if c.mu_override is not None:
                port.mu = c.mu_override.to_frame().T[assets]
            apply_mean_risk(port, c.policy, assets, rm=self.rm)
            if c.policy.max_vol is not None and port.upperdev is None:
                diag["vol_cap"] = "post-check only (Riskfolio upperdev + arcinequality bug)"
            diag["solvers"] = list(c.policy.solvers)
            out = port.optimization(
                model="Classic", rm=self.rm, obj=self.obj, rf=c.policy.rf, l=self.l, hist=True
            )
            if out is None and self.obj == "Sharpe":
                mu = pd.Series(port.mu.to_numpy(dtype=float).ravel(), index=assets)
                reason = sharpe_undefined(c.policy, mu, r)
                if reason:
                    raise Infeasible(reason)
            return out

        return guarded_fit(impl, returns, ctx)

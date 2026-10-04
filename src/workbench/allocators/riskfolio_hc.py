"""Riskfolio-Lib hierarchical-clustering allocator (HRP, HERC, NCO)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import riskfolio as rp

from workbench.allocators._estimates import expected_returns
from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.riskfolio import hc_bounds, hc_bounds_problem

HC_MODELS = ("HRP", "HERC", "NCO")


@dataclass(frozen=True)
class RiskfolioHC:
    """Hierarchical-clustering portfolio with Riskfolio-Lib.

    model:        "HRP", "HERC" or "NCO".
    codependence: e.g. "pearson", "spearman", "gerber1".
    linkage:      e.g. "ward", "single".
    rm / obj:     risk measure; objective (used by NCO; "MinRisk" etc.).
    method_mu:    mean estimator; mu is computed here (real-valued, see ``_estimates``) and
                  passed as ``custom_mu``. ``ctx.mu_override`` (per period) takes precedence.
    method_cov:   covariance estimator passed to ``HCPortfolio.optimization``.
    Asset bounds and the band are enforced via ``w_max``/``w_min``; class limits and TE are
    not enforceable in HC and are caught by the policy post-check.
    """

    model: str = "HRP"
    codependence: str = "pearson"
    linkage: str = "ward"
    rm: str = "MV"
    obj: str = "MinRisk"
    method_mu: str = "hist"
    method_cov: str = "hist"
    l: float = 2.0  # noqa: E741 - Riskfolio-Lib's name
    name = "riskfolio_hc"

    def __post_init__(self) -> None:
        if self.model not in HC_MODELS:
            raise ValueError(f"model must be one of {HC_MODELS}, got {self.model!r}")

    def params(self) -> dict:
        return asdict(self)

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r: pd.DataFrame, c: FitContext, diag: dict):
            assets = list(r.columns)
            w_max, w_min = hc_bounds(c.policy, assets)
            problem = hc_bounds_problem(w_max, w_min)
            if problem:
                raise Infeasible(f"HC bounds infeasible: {problem}")
            hc = rp.HCPortfolio(returns=r, w_max=w_max, w_min=w_min, solvers=list(c.policy.solvers))
            diag["solvers"] = list(c.policy.solvers)
            if c.mu_override is not None:
                mu = c.mu_override.reindex(assets).astype(float)
            else:
                mu, diag["mu_cast_from_complex"] = expected_returns(r, self.method_mu)
            return hc.optimization(
                model=self.model,
                codependence=self.codependence,
                linkage=self.linkage,
                rm=self.rm,
                obj=self.obj,
                rf=c.policy.rf,
                l=self.l,
                method_mu="custom_mu",
                custom_mu=mu,
                method_cov=self.method_cov,
            )

        return guarded_fit(impl, returns, ctx)

"""skfolio hierarchical-clustering allocator: the second-backend twin of ``RiskfolioHC``."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from skfolio.optimization import (
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    MeanRisk,
    NestedClustersOptimization,
)

from workbench.allocators import _skfolio_map as sk
from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.riskfolio import hc_bounds_problem
from workbench.policy.skfolio import hc_kwargs

HC_MODELS = ("HRP", "HERC", "NCO")


@dataclass(frozen=True)
class SkfolioHC:
    """HRP / HERC / NCO with skfolio; same parameters as ``RiskfolioHC`` plus ``max_clusters``.

    max_clusters: None = skfolio's own cluster-count selection (differs from Riskfolio's
                  two-difference gap statistic, so HERC/NCO weights differ); an integer forces it.
    Asset bounds and the band are enforced for HRP/HERC via min/max weights. NCO takes no
    composite bounds here; class limits, TE and (for NCO) bounds are caught by the post-check.
    Non-finite weights (a skfolio 1.4.11 defect with binding lower bounds) -> ``solver_error``.
    """

    model: str = "HRP"
    codependence: str = "pearson"
    linkage: str = "ward"
    rm: str = "MV"
    obj: str = "MinRisk"
    method_mu: str = "hist"
    method_cov: str = "hist"
    l: float = 2.0  # noqa: E741 - Riskfolio-Lib's name
    max_clusters: int | None = None
    name = "skfolio_hc"

    def __post_init__(self) -> None:
        if self.model not in HC_MODELS:
            raise ValueError(f"model must be one of {HC_MODELS}, got {self.model!r}")

    def params(self) -> dict:
        return asdict(self)

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r: pd.DataFrame, c: FitContext, diag: dict):
            assets = list(r.columns)
            prior = sk.prior(self.method_mu, self.method_cov, c.mu_override, assets)
            clustering = sk.clustering(self.linkage, self.max_clusters)
            if self.model == "NCO":
                inner = MeanRisk(
                    risk_measure=sk.risk_measure(self.rm, self.obj),
                    objective_function=sk.objective(self.obj),
                    risk_aversion=self.l,
                    prior_estimator=prior,
                    risk_free_rate=c.policy.rf,
                    cvar_beta=sk.BETA,
                    cdar_beta=sk.BETA,
                    **sk.mar_kwargs(self.rm, c.policy.rf),
                )
                est = NestedClustersOptimization(
                    inner_estimator=inner,
                    outer_estimator=inner,
                    distance_estimator=sk.distance(self.codependence),
                    clustering_estimator=clustering,
                )
            else:
                bounds = hc_kwargs(c.policy, assets)
                problem = hc_bounds_problem(
                    pd.Series(bounds["max_weights"]), pd.Series(bounds["min_weights"])
                )
                if problem:
                    raise Infeasible(f"HC bounds infeasible: {problem}")
                cls = HierarchicalRiskParity if self.model == "HRP" else (
                    HierarchicalEqualRiskContribution
                )  # fmt: skip
                est = cls(
                    risk_measure=sk.risk_measure(self.rm),
                    prior_estimator=prior,
                    distance_estimator=sk.distance(self.codependence),
                    hierarchical_clustering_estimator=clustering,
                    **bounds,
                )
            est.fit(r)
            if not np.isfinite(est.weights_).all():
                # skfolio 1.4.11 bounded bisection divides by a cluster's first weight; once a
                # weight hits 0 with binding lower bounds this is 0/0 (see CLAUDE.md).
                raise cp.error.SolverError(
                    f"skfolio {self.model} returned non-finite weights (bounded bisection "
                    "divides by zero when lower bounds bind; skfolio 1.4.11)"
                )
            n_clusters = getattr(
                getattr(est, "hierarchical_clustering_estimator_", None), "n_clusters_", None
            )
            if n_clusters is not None:
                diag["n_clusters"] = int(n_clusters)
            return pd.Series(est.weights_, index=assets)

        return guarded_fit(impl, returns, ctx)

"""skfolio mean-risk allocator: the second-backend twin of ``RiskfolioMeanRisk``."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cvxpy as cp
import pandas as pd
from skfolio.optimization import MeanRisk

from workbench.allocators import _skfolio_map as sk
from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.skfolio import linear_infeasibility, mean_risk_kwargs, tracking_target


@dataclass(frozen=True)
class SkfolioMeanRisk:
    """Mean-risk optimisation with skfolio; same parameters as ``RiskfolioMeanRisk``.

    method_mu / method_cov: estimator names in Riskfolio vocabulary (see ``_skfolio_map``).
    rm:  "MV", "CVaR" or "CDaR" (95% confidence).
    obj: "MinRisk", "Utility", "Sharpe" or "MaxRet".
    l:   risk aversion for obj="Utility" (maps to skfolio ``risk_aversion``).
    rf comes from ``ctx.policy.rf`` (per period); ``ctx.mu_override`` replaces mu (per period).
    """

    method_mu: str = "hist"
    method_cov: str = "hist"
    rm: str = "MV"
    obj: str = "Sharpe"
    l: float = 2.0  # noqa: E741 - Riskfolio-Lib's name
    name = "skfolio_mean_risk"

    def params(self) -> dict:
        return asdict(self)

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r: pd.DataFrame, c: FitContext, diag: dict):
            assets = list(r.columns)
            base = dict(
                risk_measure=sk.risk_measure(self.rm, self.obj),
                objective_function=sk.objective(self.obj),
                risk_aversion=self.l,
                prior_estimator=sk.prior(self.method_mu, self.method_cov, c.mu_override, assets),
                cvar_beta=sk.BETA,
                cdar_beta=sk.BETA,
                raise_on_failure=True,
                **mean_risk_kwargs(c.policy, assets),
            )
            y = tracking_target(c.policy, r)
            errors = []
            for solver in c.policy.solvers:
                try:
                    est = MeanRisk(solver=solver, **base)
                    est.fit(r, y) if y is not None else est.fit(r)
                    diag["solver"] = solver
                    return pd.Series(est.weights_, index=assets)
                except cp.error.SolverError as e:
                    errors.append(f"{solver}: {e}")
            diag["solver_errors"] = errors
            reason = linear_infeasibility(c.policy, assets)
            if reason:
                raise Infeasible(reason)
            raise cp.error.SolverError("; ".join(errors))

        return guarded_fit(impl, returns, ctx)

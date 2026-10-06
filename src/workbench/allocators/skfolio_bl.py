"""skfolio Black–Litterman: the second-backend twin of ``RiskfolioBlackLitterman``.

The posterior is skfolio's own ``BlackLitterman`` prior (``EquilibriumMu`` on the SAA weights),
not our formula, so every paired cell checks the posterior independently. Its posterior
covariance is not used: like the Riskfolio twin, the optimiser keeps the estimator's covariance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cvxpy as cp
import pandas as pd
from skfolio.moments import EquilibriumMu
from skfolio.optimization import MeanRisk
from skfolio.prior import BlackLitterman, EmpiricalPrior

from workbench.allocators import _skfolio_map as sk
from workbench.allocators._bl import BLParams, Prior, fit_black_litterman
from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.skfolio import linear_infeasibility, mean_risk_kwargs, tracking_target


@dataclass(frozen=True)
class SkfolioBlackLitterman:
    """Same parameters and modes as ``RiskfolioBlackLitterman`` (see there)."""

    view_annual: float | None = None
    confidence: float = 1.0
    target_weight: float | None = None
    prior_sharpe: float = 0.3
    obj: str = "Sharpe"
    method_cov: str = "hist"
    name = "skfolio_bl"

    def __post_init__(self) -> None:
        self.bl_params().check()

    def bl_params(self) -> BLParams:
        return BLParams(self.view_annual, self.confidence, self.target_weight, self.prior_sharpe,
                        self.obj)  # fmt: skip

    def params(self) -> dict:
        return asdict(self)

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r: pd.DataFrame, c: FitContext, diag: dict):
            assets = list(r.columns)
            cov_estimator = sk.covariance_estimator(self.method_cov)
            cov = pd.DataFrame(cov_estimator().fit(r).covariance_, index=assets, columns=assets)
            saa = c.saa.reindex(assets).to_numpy(dtype=float)
            rf = c.policy.rf
            kw = mean_risk_kwargs(c.policy, assets)
            y = tracking_target(c.policy, r)

            def make_solve(prior: Prior):
                def solve(view: float, confidence: float) -> pd.Series:
                    bl = BlackLitterman(
                        views=[f"{c.candidate} = {float(view)!r}"],
                        view_confidences=[confidence],
                        prior_estimator=EmpiricalPrior(
                            mu_estimator=EquilibriumMu(risk_aversion=prior.delta, weights=saa,
                                                       covariance_estimator=cov_estimator()),
                            covariance_estimator=cov_estimator(),
                        ),
                        risk_free_rate=rf,
                    ).fit(r)  # fmt: skip
                    mu = pd.Series(bl.return_distribution_.mu, index=assets)
                    errors = []
                    for solver in c.policy.solvers:
                        try:
                            est = MeanRisk(
                                risk_measure=sk.risk_measure("MV", self.obj),
                                objective_function=sk.objective(self.obj),
                                risk_aversion=prior.delta / 2.0,
                                prior_estimator=sk.prior("hist", self.method_cov, mu, assets),
                                raise_on_failure=True,
                                solver=solver,
                                **kw,
                            )
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

                return solve

            return fit_black_litterman(self.bl_params(), r, c, cov, make_solve, diag)

        return guarded_fit(impl, returns, ctx)

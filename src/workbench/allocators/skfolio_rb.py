"""skfolio risk budgeting: the second-backend twin of ``RiskfolioRiskBudget``."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cvxpy as cp
import pandas as pd
from skfolio.optimization import RiskBudgeting

from workbench.allocators import _skfolio_map as sk
from workbench.allocators._budget import RESTS, covariance, realised_share, risk_budget
from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.skfolio import linear_infeasibility, linear_kwargs


@dataclass(frozen=True)
class SkfolioRiskBudget:
    """skfolio ``RiskBudgeting`` with the same parameters and budget as the Riskfolio twin.

    The budget vector is built by the shared ``_budget.risk_budget`` (Riskfolio covariance and
    contributions), so both libraries solve the identical problem; weights agree to ~1e-5.
    """

    candidate_share: float = 0.05
    rm: str = "MV"
    rest: str = "saa"
    method_mu: str = "hist"
    method_cov: str = "hist"
    name = "skfolio_risk_budget"

    def __post_init__(self) -> None:
        if not 0 < self.candidate_share < 1:
            raise ValueError(f"candidate_share must be in (0, 1), got {self.candidate_share}")
        if self.rest not in RESTS:
            raise ValueError(f"rest must be one of {RESTS}, got {self.rest!r}")

    def params(self) -> dict:
        return asdict(self)

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r: pd.DataFrame, c: FitContext, diag: dict):
            assets = list(r.columns)
            cov = covariance(r, self.method_cov)
            b = risk_budget(r, c.saa, c.candidate, self.candidate_share, self.rm, self.rest,
                            cov, c.policy.rf)  # fmt: skip
            kw = linear_kwargs(c.policy, assets)
            kw.pop("budget")  # RiskBudgeting is fully invested by construction
            errors = []
            for solver in c.policy.solvers:
                try:
                    est = RiskBudgeting(
                        risk_measure=sk.risk_measure(self.rm),
                        risk_budget=b.to_numpy(),
                        prior_estimator=sk.prior(self.method_mu, self.method_cov, None, assets),
                        cvar_beta=sk.BETA,
                        cdar_beta=sk.BETA,
                        solver=solver,
                        raise_on_failure=True,
                        **kw,
                        **sk.mar_kwargs(self.rm, c.policy.rf),
                    ).fit(r)
                    w = pd.Series(est.weights_, index=assets)
                    diag["solver"] = solver
                    diag["target_share"] = self.candidate_share
                    diag["realised_share"] = realised_share(w, r, cov, self.rm, c.policy.rf,
                                                            c.candidate)  # fmt: skip
                    return w
                except cp.error.SolverError as e:
                    errors.append(f"{solver}: {e}")
            diag["solver_errors"] = errors
            reason = linear_infeasibility(c.policy, assets)
            if reason:
                raise Infeasible(reason)
            raise cp.error.SolverError("; ".join(errors))

        return guarded_fit(impl, returns, ctx)

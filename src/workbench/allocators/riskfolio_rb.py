"""Riskfolio-Lib risk budgeting: "how much of our risk should the candidate carry?"."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cvxpy as cp
import pandas as pd
import riskfolio as rp

from workbench.allocators._budget import RESTS, covariance, realised_share, risk_budget
from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.riskfolio import linear_constraints
from workbench.policy.skfolio import linear_infeasibility


@dataclass(frozen=True)
class RiskfolioRiskBudget:
    """Risk budgeting (``rp_optimization``) with the candidate at ``candidate_share`` of risk.

    candidate_share: target share of portfolio risk under ``rm`` (decimal, 0 < share < 1).
    rm:   risk measure (e.g. "MV", "MSV", "CVaR", "CDaR", "FLPM").
    rest: "saa" (others keep their relative SAA risk shares, floored) or "equal".
    method_mu: unused by risk budgeting (kept so grid estimators apply uniformly).
    method_cov: covariance estimator (per period).
    Asset bounds and class limits are passed as linear constraints (they can stop the budget
    being met); band, TE and risk caps are left to the post-check. Diagnostics record the
    target and the realised candidate share (Riskfolio ``Risk_Contribution``).
    """

    candidate_share: float = 0.05
    rm: str = "MV"
    rest: str = "saa"
    method_mu: str = "hist"
    method_cov: str = "hist"
    name = "riskfolio_risk_budget"

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
            port = rp.Portfolio(returns=r)
            port.assets_stats(method_mu="hist", method_cov=self.method_cov)
            port.cov = cov
            port.sht = False
            port.solvers = list(c.policy.solvers)
            a, bb = linear_constraints(c.policy, assets)
            if a is not None:
                port.ainequality, port.binequality = a, bb
            out = port.rp_optimization(model="Classic", rm=self.rm, rf=c.policy.rf,
                                       b=b.to_numpy().reshape(-1, 1), hist=True)  # fmt: skip
            if out is None:
                # Riskfolio returns None for numerical failures too. A risk budget with positive
                # budgets is feasible unless the linear constraints are, so classify by them
                # (seen: FLPM budget fails in CLARABEL and ECOS, solves in SCS).
                reason = linear_infeasibility(c.policy, assets)
                if reason:
                    raise Infeasible(reason)
                raise cp.error.SolverError(
                    "rp_optimization returned no solution on a feasible problem "
                    f"(solvers {list(c.policy.solvers)}); try adding SCS to the spec's solvers"
                )
            ws = out["weights"].astype(float)
            diag["target_share"] = self.candidate_share
            diag["realised_share"] = realised_share(ws, r, cov, self.rm, c.policy.rf,
                                                    c.candidate)  # fmt: skip
            return out

        return guarded_fit(impl, returns, ctx)

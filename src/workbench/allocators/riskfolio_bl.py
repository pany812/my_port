"""Riskfolio-Lib Black–Litterman: "what weight does our view give?" and "what must it earn?"."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import riskfolio as rp

from workbench.allocators._bl import BLParams, Prior, fit_black_litterman
from workbench.allocators._estimates import covariance
from workbench.allocators._solve import guarded_fit
from workbench.allocators.base import AllocationResult, FitContext
from workbench.policy.riskfolio import apply_mean_risk


@dataclass(frozen=True)
class RiskfolioBlackLitterman:
    """Black–Litterman with the SAA as the prior, solved with Riskfolio-Lib (Classic, MV).

    view_annual:   view mode: the candidate's expected excess return over rf (annual,
                   arithmetic, decimal).
    confidence:    view mode: in (0, 1]; the candidate's posterior moves this fraction of the way
                   from equilibrium to the view (Idzorek; exact for a single view).
    target_weight: breakeven mode: find at each date the posterior premium over equilibrium that
                   gives the candidate this weight (decimal, 0 < x < 1). The cell holds that
                   portfolio; an unreachable target is ``infeasible``.
    prior_sharpe:  assumed annual Sharpe ratio of the SAA: risk aversion delta = SR / sigma_SAA.
    obj:           "Sharpe" or "Utility" (Riskfolio l = delta / 2, the only value for which no view
                   returns the SAA).
    method_cov:    covariance estimator for the prior and the optimiser (per period).

    Exactly one of ``view_annual`` and ``target_weight``. The posterior is our formula
    (``_bl``), equal to Riskfolio-Lib's ``black_litterman`` at confidence 0.5, the only
    confidence that function supports.
    """

    view_annual: float | None = None
    confidence: float = 1.0
    target_weight: float | None = None
    prior_sharpe: float = 0.3
    obj: str = "Sharpe"
    method_cov: str = "hist"
    name = "riskfolio_bl"

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
            cov = covariance(r, self.method_cov)
            port = rp.Portfolio(returns=r)
            port.assets_stats(method_mu="hist", method_cov=self.method_cov)
            port.cov = cov
            apply_mean_risk(port, c.policy, assets, rm="MV")
            if c.policy.max_vol is not None and port.upperdev is None:
                diag["vol_cap"] = "post-check only (Riskfolio upperdev + arcinequality bug)"
            diag["solvers"] = list(c.policy.solvers)
            rf = c.policy.rf

            def make_solve(prior: Prior):
                def solve(view: float, confidence: float):
                    port.mu = prior.mu(prior.premium(view, confidence), rf).to_frame().T[assets]
                    return port.optimization(model="Classic", rm="MV", obj=self.obj, rf=rf,
                                             l=prior.delta / 2.0, hist=True)  # fmt: skip

                return solve

            return fit_black_litterman(self.bl_params(), r, c, cov, make_solve, diag)

        return guarded_fit(impl, returns, ctx)

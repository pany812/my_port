"""Spec names (Riskfolio-Lib vocabulary) -> skfolio 1.4.11 objects.

Only names verified to match Riskfolio-Lib on synthetic data are mapped (October 2026):
mu ``hist``/``JS`` and cov ``hist``/``ledoit``/``gerber1``/``fixed`` (denoised) are numerically
identical; Sharpe, MinRisk and Utility (``risk_aversion = l``) agree to solver tolerance; HRP is
identical; HERC and NCO are identical once the cluster count matches (skfolio picks its own by
default). Downside measures (P2-M3): ``MSV`` -> semi-deviation, ``FLPM`` -> first lower partial
moment and ``SLPM`` -> semi-deviation, the last two with the policy rf as the minimum acceptable
return (MinRisk weights agree to <= 3e-7).
Anything else raises ``ValueError("... unsupported in the skfolio backend")``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skfolio import RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import KendallDistance, PearsonDistance, SpearmanDistance
from skfolio.moments import (
    BaseMu,
    DenoiseCovariance,
    EmpiricalCovariance,
    EmpiricalMu,
    GerberCovariance,
    LedoitWolf,
    ShrunkMu,
    ShrunkMuMethods,
)
from skfolio.optimization import ObjectiveFunction
from skfolio.prior import EmpiricalPrior

BETA = 0.95  # CVaR / CDaR confidence, Riskfolio alpha = 0.05

_MU = {
    "hist": lambda: EmpiricalMu(),
    "JS": lambda: ShrunkMu(method=ShrunkMuMethods.JAMES_STEIN),
}
_COV = {
    "hist": lambda: EmpiricalCovariance(),
    "ledoit": lambda: LedoitWolf(),
    "gerber1": lambda: GerberCovariance(),
    "fixed": lambda: DenoiseCovariance(),  # Riskfolio "fixed" = denoised sample covariance
}
_RISK = {
    "CVaR": RiskMeasure.CVAR,
    "CDaR": RiskMeasure.CDAR,
    "MSV": RiskMeasure.SEMI_DEVIATION,
    "FLPM": RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    "SLPM": RiskMeasure.SEMI_DEVIATION,
}
_MAR_IS_RF = {"FLPM", "SLPM"}  # Riskfolio uses rf as the target return for these
_OBJ = {
    "MinRisk": ObjectiveFunction.MINIMIZE_RISK,
    "Sharpe": ObjectiveFunction.MAXIMIZE_RATIO,
    "Utility": ObjectiveFunction.MAXIMIZE_UTILITY,
    "MaxRet": ObjectiveFunction.MAXIMIZE_RETURN,
}
_DIST = {"pearson": PearsonDistance, "spearman": SpearmanDistance, "kendall": KendallDistance}
_LINK = {
    "ward": LinkageMethod.WARD,
    "single": LinkageMethod.SINGLE,
    "complete": LinkageMethod.COMPLETE,
    "average": LinkageMethod.AVERAGE,
}


class FixedMu(BaseMu):
    """Expected returns given directly (per period, decimal), for ``ctx.mu_override``."""

    def __init__(self, mu: np.ndarray | None = None) -> None:
        self.mu = mu

    def fit(self, X, y=None) -> FixedMu:  # noqa: N803 - sklearn convention
        self.mu_ = np.asarray(self.mu, dtype=float)
        return self


def _get(table: dict, key: str, what: str):
    try:
        return table[key]
    except KeyError:
        raise ValueError(
            f"{what} {key!r} unsupported in the skfolio backend; supported: {sorted(table)}"
        ) from None


def prior(method_mu: str, method_cov: str, mu_override: pd.Series | None = None,
          assets: list[str] | None = None) -> EmpiricalPrior:  # fmt: skip
    """EmpiricalPrior with mapped estimators (or a fixed mu from ``mu_override``)."""
    cov = _get(_COV, method_cov, "method_cov")()
    if mu_override is not None:
        mu = FixedMu(mu_override.reindex(assets).to_numpy(dtype=float))
    else:
        mu = _get(_MU, method_mu, "method_mu")()
    return EmpiricalPrior(mu_estimator=mu, covariance_estimator=cov)


def risk_measure(rm: str, obj: str | None = None) -> RiskMeasure:
    """Riskfolio ``rm`` -> skfolio RiskMeasure. MV: standard deviation for Sharpe (Riskfolio
    maximises mean / std), variance otherwise (same argmin; Utility uses variance)."""
    if rm == "MV":
        return RiskMeasure.STANDARD_DEVIATION if obj == "Sharpe" else RiskMeasure.VARIANCE
    return _get(_RISK, rm, "rm")


def mar_kwargs(rm: str, rf: float) -> dict:
    """``min_acceptable_return`` for lower-partial-moment measures (rf per period), else {}."""
    return {"min_acceptable_return": float(rf)} if rm in _MAR_IS_RF else {}


def objective(obj: str) -> ObjectiveFunction:
    return _get(_OBJ, obj, "obj")


def distance(codependence: str):
    return _get(_DIST, codependence, "codependence")()


def clustering(linkage: str, max_clusters: int | None) -> HierarchicalClustering:
    return HierarchicalClustering(
        max_clusters=max_clusters, linkage_method=_get(_LINK, linkage, "linkage")
    )

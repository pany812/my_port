"""Compiled policy: the constraint set handed to allocators, already in per-period units.

Built by ``workbench.policy.compiler.compile_policy`` from an SAA and a spec's
``constraint_set``. Translation to Riskfolio-Lib tables lives in ``workbench.policy.riskfolio``.

Every allocator's output is post-checked with :meth:`CompiledPolicy.violations`; any breach
makes the cell ``infeasible`` (policy decision, Phase 1 M2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from workbench.units import periods_per_year, vol_period_to_annual

LINEAR_TOL = 1e-6  # absolute, in weight units
TE_RTOL = 1e-4  # relative tolerance on the TE limit
RISK_RTOL = 1e-4  # relative tolerance on risk caps
ALPHA = 0.05  # CVaR / CDaR tail (95% confidence), same as the risk lenses


@dataclass(frozen=True)
class CompiledPolicy:
    """Constraints and limits for one grid cell.

    name:         constraint-set name from the spec.
    freq:         return frequency code of the experiment (see ``workbench.units``).
    rf:           risk-free rate per period of ``freq`` (decimal).
    long_only:    weights must be >= 0.
    lower/upper:  per-asset weight bounds (decimal); None = [0, 1].
    asset_class:  class label per asset id; required when ``class_limits`` is set.
    class_limits: {class: (lo, hi)} bounds on summed class weight (decimal).
    band:         per-asset band |w_i - benchweights_i| <= band (decimal weight), or None.
    te:           tracking-error limit vs ``benchweights`` per period of ``freq`` (decimal),
                  Riskfolio definition: ||R (w - b)|| / sqrt(T - 1). None = no limit.
    benchweights: the SAA, indexed by asset id; required for ``band`` and ``te``.
    solvers:      cvxpy solvers tried in order.
    candidate:    candidate asset id (needed for ``max_candidate_risk_share``).
    max_vol:      volatility cap per period (decimal std). Post-check: sample covariance.
    max_cvar:     CVaR 95% cap, loss per period (decimal; Riskfolio ``CVaR_Hist``).
    max_cdar:     CDaR 95% cap (decimal; Riskfolio ``CDaR_Abs``, uncompounded drawdowns).
    min_return:   floor on the mean return per period (decimal). Post-check: sample mean.
    max_candidate_risk_share: cap on the candidate's Euler share of variance (decimal).
                  Post-check: sample covariance of the fitting window (P2-M3 decision 2a: one
                  definition for every allocator; an optimiser using a shrunk covariance
                  enforces it under its own covariance and may breach the sample version).
    """

    name: str = "unconstrained"
    freq: str = "M"
    rf: float = 0.0
    long_only: bool = True
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    asset_class: pd.Series | None = None
    class_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    band: float | None = None
    te: float | None = None
    benchweights: pd.Series | None = None
    solvers: tuple[str, ...] = ("CLARABEL",)
    candidate: str | None = None
    max_vol: float | None = None
    max_cvar: float | None = None
    max_cdar: float | None = None
    min_return: float | None = None
    max_candidate_risk_share: float | None = None

    def __post_init__(self) -> None:
        periods_per_year(self.freq)  # validates the code
        if self.class_limits and self.asset_class is None:
            raise ValueError("class_limits require asset_class")
        if (self.band is not None or self.te is not None) and self.benchweights is None:
            raise ValueError("band and te require benchweights")
        for cls, (lo, hi) in self.class_limits.items():
            if not 0 <= lo <= hi <= 1:
                raise ValueError(f"class limit for {cls!r} must satisfy 0 <= lo <= hi <= 1")
        if self.max_candidate_risk_share is not None and self.candidate is None:
            raise ValueError("max_candidate_risk_share requires the candidate id")

    @property
    def has_risk_limits(self) -> bool:
        limits = (self.max_vol, self.max_cvar, self.max_cdar, self.min_return,
                  self.max_candidate_risk_share)  # fmt: skip
        return any(v is not None for v in limits)

    def bounds(self, assets: list[str]) -> tuple[pd.Series, pd.Series]:
        """Per-asset (lower, upper) bounds for ``assets``, defaulting to [0, 1]."""
        lo = pd.Series(0.0, index=assets) if self.lower is None else self.lower.reindex(assets)
        hi = pd.Series(1.0, index=assets) if self.upper is None else self.upper.reindex(assets)
        if lo.isna().any() or hi.isna().any():
            missing = sorted(set(lo.index[lo.isna()]) | set(hi.index[hi.isna()]))
            raise ValueError(f"no bounds for assets {missing}")
        return lo.astype(float), hi.astype(float)

    def box_bounds(self, assets: list[str]) -> tuple[pd.Series, pd.Series]:
        """Per-asset (lower, upper) bounds intersected with the band box [b - band, b + band].

        The band is a per-asset box, so methods that only take weight bounds (HC) can enforce
        it exactly this way.
        """
        lo, hi = self.bounds(assets)
        if self.band is not None:
            bench = self.benchweights.reindex(assets)
            lo = np.maximum(lo, bench - self.band)
            hi = np.minimum(hi, bench + self.band)
        return lo.clip(lower=0.0), hi.clip(upper=1.0)

    def tracking_error(self, w: pd.Series, returns: pd.DataFrame) -> float:
        """Ex-ante TE per period of ``w`` vs ``benchweights`` on ``returns`` (Riskfolio def.)."""
        if self.benchweights is None:
            raise ValueError("tracking_error requires benchweights")
        cols = list(returns.columns)
        active = returns.to_numpy() @ (w[cols] - self.benchweights[cols]).to_numpy()
        return float(np.linalg.norm(active) / math.sqrt(len(returns) - 1))

    def violations(self, w: pd.Series, returns: pd.DataFrame) -> list[str]:
        """Human-readable list of every breach of this policy by ``w``; empty if compliant.

        w:       weights indexed by asset id (decimal).
        returns: the fitting window (simple returns per period), used for the TE check.
        """
        out: list[str] = []
        assets = list(returns.columns)
        if set(w.index) != set(assets):
            return [f"weights index does not match assets {assets}"]
        w = w[assets]
        total = float(w.sum())
        if abs(total - 1.0) > LINEAR_TOL:
            out.append(f"weights sum to {total:.6f}, not 1")
        if self.long_only:
            for a in w.index[w < -LINEAR_TOL]:
                out.append(f"{a}: weight {w[a]:.4%} < 0 (long-only)")
        lo, hi = self.bounds(assets)
        for a in assets:
            if w[a] < lo[a] - LINEAR_TOL:
                out.append(f"{a}: weight {w[a]:.4%} below lower bound {lo[a]:.4%}")
            if w[a] > hi[a] + LINEAR_TOL:
                out.append(f"{a}: weight {w[a]:.4%} above upper bound {hi[a]:.4%}")
        if self.class_limits:
            by_class = w.groupby(self.asset_class.reindex(assets)).sum()
            for cls, (clo, chi) in self.class_limits.items():
                cw = float(by_class.get(cls, 0.0))
                if cw < clo - LINEAR_TOL or cw > chi + LINEAR_TOL:
                    out.append(f"class {cls}: weight {cw:.4%} outside [{clo:.4%}, {chi:.4%}]")
        if self.band is not None:
            dev = (w - self.benchweights.reindex(assets)).abs()
            for a in dev.index[dev > self.band + LINEAR_TOL]:
                out.append(f"{a}: |w - SAA| {dev[a]:.4%} exceeds band {self.band:.4%}")
        if self.te is not None:
            te = self.tracking_error(w, returns)
            if te > self.te * (1 + TE_RTOL) + 1e-10:
                ann = vol_period_to_annual(te, self.freq)
                lim = vol_period_to_annual(self.te, self.freq)
                out.append(f"TE {ann:.4%} p.a. exceeds limit {lim:.4%} p.a.")
        if self.has_risk_limits:
            out += self._risk_limit_violations(w, returns)
        return out

    def _risk_limit_violations(self, w: pd.Series, returns: pd.DataFrame) -> list[str]:
        """Risk caps on the fitting window: sample moments, Riskfolio CVaR/CDaR definitions."""
        from riskfolio.src import RiskFunctions as RF  # reference definitions

        out = []
        x = returns.to_numpy() @ w.to_numpy()
        cov = returns.cov().to_numpy()
        var = float(w.to_numpy() @ cov @ w.to_numpy())

        def over(value: float, cap: float) -> bool:
            return value > cap * (1 + RISK_RTOL) + 1e-12

        if self.max_vol is not None:
            vol = math.sqrt(max(var, 0.0))
            if over(vol, self.max_vol):
                ann, lim = (vol_period_to_annual(v, self.freq) for v in (vol, self.max_vol))
                out.append(f"volatility {ann:.4%} p.a. exceeds cap {lim:.4%} p.a.")
        if self.max_cvar is not None:
            cvar = float(RF.CVaR_Hist(x, ALPHA))
            if over(cvar, self.max_cvar):
                out.append(f"CVaR95 {cvar:.4%} per period exceeds cap {self.max_cvar:.4%}")
        if self.max_cdar is not None:
            cdar = float(RF.CDaR_Abs(x, ALPHA))
            if over(cdar, self.max_cdar):
                out.append(f"CDaR95 {cdar:.4%} exceeds cap {self.max_cdar:.4%}")
        if self.min_return is not None:
            mean = float(x.mean())
            if mean < self.min_return - 1e-12 - abs(self.min_return) * RISK_RTOL:
                out.append(f"mean return {mean:.4%} per period below floor {self.min_return:.4%}")
        if self.max_candidate_risk_share is not None:
            share = candidate_variance_share(w, cov, list(returns.columns), self.candidate)
            if share > self.max_candidate_risk_share + LINEAR_TOL:
                out.append(f"{self.candidate}: variance share {share:.4%} exceeds cap "
                           f"{self.max_candidate_risk_share:.4%}")  # fmt: skip
        return out


def candidate_variance_share(w: pd.Series, cov: np.ndarray, assets: list[str], candidate: str):
    """Euler share of portfolio variance carried by ``candidate``: w_c (S w)_c / (w' S w)."""
    wv = w.reindex(assets).to_numpy(dtype=float)
    total = float(wv @ cov @ wv)
    if total <= 0:
        return 0.0
    c = assets.index(candidate)
    return float(wv[c] * (cov @ wv)[c] / total)

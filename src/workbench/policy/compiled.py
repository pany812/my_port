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

    def __post_init__(self) -> None:
        periods_per_year(self.freq)  # validates the code
        if self.class_limits and self.asset_class is None:
            raise ValueError("class_limits require asset_class")
        if (self.band is not None or self.te is not None) and self.benchweights is None:
            raise ValueError("band and te require benchweights")
        for cls, (lo, hi) in self.class_limits.items():
            if not 0 <= lo <= hi <= 1:
                raise ValueError(f"class limit for {cls!r} must satisfy 0 <= lo <= hi <= 1")

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
        return out

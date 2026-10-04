"""Heuristic allocators: hold the SAA, SAA plus a candidate slice, equal weight, inverse vol.

None of these optimise against the policy; their weights are post-checked like every other
allocator's and a breach records the cell as ``infeasible``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from workbench.allocators._solve import Infeasible, guarded_fit
from workbench.allocators.base import AllocationResult, FitContext


@dataclass(frozen=True)
class StaticSAA:
    """Hold the SAA weights (candidate at 0)."""

    name = "static_saa"

    def params(self) -> dict:
        return {}

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        return guarded_fit(lambda r, c, d: c.saa.reindex(r.columns), returns, ctx)


@dataclass(frozen=True)
class SAAPlus:
    """SAA plus a candidate weight ``x`` (decimal), funded from a named source.

    funding="pro_rata":     w = (1 - x) * SAA + x * e_candidate (all other assets pro rata).
    funding="asset:<id>":   x comes out of one asset.
    funding="class:<name>": x comes out of one asset class, pro rata within it.
    A source holding less than x makes the cell infeasible ("funding source exhausted").
    """

    x: float
    funding: str = "pro_rata"
    name = "saa_plus"

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0:
            raise ValueError(f"x must be in [0, 1], got {self.x}")
        kind, _, name = self.funding.partition(":")
        if not (self.funding == "pro_rata" or (kind in ("asset", "class") and name)):
            raise ValueError(
                f"funding must be 'pro_rata', 'asset:<id>' or 'class:<name>', got {self.funding!r}"
            )

    def params(self) -> dict:
        return asdict(self)

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r, c, d):
            if self.funding == "pro_rata":
                w = (1.0 - self.x) * c.saa.reindex(r.columns)
                w[c.candidate] += self.x
                return w
            w = c.saa.reindex(r.columns).astype(float).copy()
            kind, _, name = self.funding.partition(":")
            if kind == "asset":
                if name not in w.index or name == c.candidate:
                    raise ValueError(f"funding asset {name!r} is not a building block")
                source = [name]
            else:
                classes = c.policy.asset_class
                if classes is None:
                    raise ValueError("class funding needs asset classes in the policy")
                source = [a for a in w.index if a != c.candidate and classes.get(a) == name]
                if not source:
                    raise ValueError(f"funding class {name!r} has no building blocks")
            available = float(w[source].sum())
            if available < self.x - 1e-12:
                raise Infeasible(
                    f"funding source exhausted: {self.funding} holds {available:.4%} "
                    f"< x {self.x:.4%}"
                )
            w[source] -= self.x * w[source] / available
            w[c.candidate] += self.x
            return w

        return guarded_fit(impl, returns, ctx)


@dataclass(frozen=True)
class EqualWeight:
    """1/N across all assets, candidate included."""

    name = "equal_weight"

    def params(self) -> dict:
        return {}

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        n = returns.shape[1]
        return guarded_fit(lambda r, c, d: pd.Series(1.0 / n, index=r.columns), returns, ctx)


@dataclass(frozen=True)
class InverseVol:
    """Weights proportional to 1 / sample volatility (ddof=1) over the fitting window."""

    name = "inverse_vol"

    def params(self) -> dict:
        return {}

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        def impl(r, c, d):
            vol = r.std(ddof=1)
            if (vol <= 0).any():
                raise ValueError(f"zero volatility for {list(vol.index[vol <= 0])}")
            inv = 1.0 / vol
            return inv / inv.sum()

        return guarded_fit(impl, returns, ctx)

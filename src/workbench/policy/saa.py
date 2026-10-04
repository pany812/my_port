"""Strategic asset allocation (SAA): policy weights, asset classes and per-asset ranges."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from workbench.data.synthetic import PLACEHOLDER_SAA

_TOL = 1e-9


@dataclass(frozen=True)
class SAA:
    """A versioned strategic allocation including the candidate at weight 0.

    weights:     policy weights (decimal) indexed by asset id, summing to 1.
    asset_class: class label per asset id.
    lower/upper: per-asset policy range (decimal weights), lower <= weight <= upper.
    candidate:   asset id of the candidate (weight 0).
    """

    version: str
    weights: pd.Series
    asset_class: pd.Series
    lower: pd.Series
    upper: pd.Series
    candidate: str

    def __post_init__(self) -> None:
        idx = self.weights.index
        if idx.has_duplicates:
            raise ValueError("SAA has duplicate asset ids")
        for name in ("asset_class", "lower", "upper"):
            if set(getattr(self, name).index) != set(idx):
                raise ValueError(f"SAA {name} must cover exactly the weight index")
        if self.candidate not in idx:
            raise ValueError(f"candidate {self.candidate!r} not in SAA")
        if self.weights[self.candidate] != 0:
            raise ValueError("candidate must have SAA weight 0")
        if abs(float(self.weights.sum()) - 1.0) > 1e-6:
            raise ValueError(f"SAA weights must sum to 1, got {self.weights.sum():.8f}")
        lo, hi = self.lower[idx], self.upper[idx]
        if ((lo - self.weights) > _TOL).any() or ((self.weights - hi) > _TOL).any():
            raise ValueError("SAA weights must lie within [lower, upper]")

    @property
    def assets(self) -> list[str]:
        return list(self.weights.index)

    @classmethod
    def placeholder(cls, candidate: str = "CAND", candidate_class: str = "alternatives") -> SAA:
        """The synthetic-phase placeholder SAA; candidate range [0, 1] (capped by the policy)."""
        t = PLACEHOLDER_SAA
        weights = t["weight"].astype(float).copy()
        asset_class = t["asset_class"].copy()
        lower = t["min"].astype(float).copy()
        upper = t["max"].astype(float).copy()
        weights[candidate], asset_class[candidate] = 0.0, candidate_class
        lower[candidate], upper[candidate] = 0.0, 1.0
        return cls("placeholder", weights, asset_class, lower, upper, candidate)

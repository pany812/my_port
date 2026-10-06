"""Capital market assumptions (CMAs): versioned, dated expected-return vectors.

A CMA version is a list of vectors, each effective from a period (e.g. "2016-01"). Values are
expected annual total returns (decimal): the expectation of the annual simple return in the
experiment's base currency and hedging convention, for every asset including the candidate.
Per period they become (1 + r)^(1/n) - 1 (``units.return_annual_to_period``, geometric), exact
for an expected annual simple return under i.i.d. returns. Compound (geometric, CAGR-style) CMAs
would need a variance-drag adjustment first; none is applied here.

Point in time: a fit dated t uses the latest vector effective on or before t. A fit before the
first vector is an error, never a silent look-ahead.

Sources: ``placeholder`` (the synthetic generator's true means, an oracle for testing; vectors
are generated) or vectors written inline in the spec. PostgreSQL comes with real data (P2-M8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from workbench.data.synthetic import PLACEHOLDER_SAA
from workbench.units import return_annual_to_period

PLACEHOLDER = "placeholder"


@dataclass(frozen=True)
class CMAVector:
    """effective: first period the vector applies to ("YYYY-MM" or an ISO date).
    returns_annual: expected annual total return per asset id (decimal)."""

    effective: str
    returns_annual: dict[str, float]

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.effective)


@dataclass(frozen=True)
class CMA:
    """A named CMA version: dated vectors in increasing effective order."""

    version: str
    vectors: tuple[CMAVector, ...]

    def __post_init__(self) -> None:
        if not self.vectors:
            raise ValueError(f"CMA {self.version!r} has no vectors")
        starts = [v.start for v in self.vectors]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError(f"CMA {self.version!r}: effective dates must be strictly increasing")
        for v in self.vectors:
            for asset, r in v.returns_annual.items():
                if not (isinstance(r, int | float) and math.isfinite(r) and r > -1):
                    raise ValueError(f"CMA {self.version!r} {v.effective} {asset}: expected "
                                     f"return must be a finite number > -1, got {r!r}")  # fmt: skip

    @property
    def first_effective(self) -> pd.Timestamp:
        return self.vectors[0].start

    def vector_at(self, as_of: pd.Timestamp) -> CMAVector:
        """The latest vector effective on or before ``as_of``. Raises before the first."""
        live = [v for v in self.vectors if v.start <= as_of]
        if not live:
            raise ValueError(f"CMA {self.version!r} has no vector in effect at "
                             f"{as_of.date()} (first: {self.vectors[0].effective})")  # fmt: skip
        return live[-1]

    def mu_period(self, as_of: pd.Timestamp, freq: str, assets: list[str]) -> pd.Series:
        """Expected return per period of ``freq`` (decimal) for ``assets`` at ``as_of``."""
        v = self.vector_at(as_of)
        missing = [a for a in assets if a not in v.returns_annual]
        if missing:
            raise ValueError(f"CMA {self.version!r} ({v.effective}) misses assets {missing}")
        return pd.Series({a: return_annual_to_period(float(v.returns_annual[a]), freq)
                          for a in assets})  # fmt: skip

    def check_covers(self, assets: list[str]) -> None:
        """Every vector must cover exactly ``assets`` (candidate included)."""
        for v in self.vectors:
            missing = sorted(set(assets) - set(v.returns_annual))
            extra = sorted(set(v.returns_annual) - set(assets))
            if missing or extra:
                raise ValueError(f"CMA {self.version!r} ({v.effective}): missing {missing}, "
                                 f"unknown {extra}")  # fmt: skip

    def canonical(self) -> dict[str, Any]:
        """Resolved, JSON-serialisable form (enters the spec hash)."""
        return {
            "version": self.version,
            "vectors": [{"effective": v.effective, "returns_annual": dict(v.returns_annual)}
                        for v in self.vectors],
        }  # fmt: skip


def placeholder_cma(candidate: str, candidate_mu_annual: float, start: str) -> CMA:
    """The synthetic truth as a CMA: building-block means from the placeholder table and the
    candidate's generator mean, effective from the first data period. An oracle: testing only."""
    returns = {a: float(m) for a, m in PLACEHOLDER_SAA["mu_annual"].items()}
    returns[candidate] = float(candidate_mu_annual)
    return CMA(PLACEHOLDER, (CMAVector(start, returns),))

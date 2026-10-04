"""Market data container and loader protocol.

Synthetic and PostgreSQL loaders both return :class:`MarketData`, so everything downstream is
indifferent to the source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import pandas as pd

from workbench.units import periods_per_year


@dataclass(frozen=True)
class MarketData:
    """Aligned return history for the SAA building blocks plus the candidate.

    returns:     simple total returns per period of ``freq`` (decimal), DatetimeIndex of period
                 ends, one column per asset id including the candidate. The candidate may be NaN
                 before its first observation.
    freq:        frequency code (see ``workbench.units``).
    candidate:   asset id of the candidate column.
    asset_class: asset class label per asset id.
    backfilled:  bool per date, True where the candidate observation is backfilled or proxied
                 rather than live. Kept outside ``returns`` so it is never mistaken for an asset.
    """

    returns: pd.DataFrame
    freq: str
    candidate: str
    asset_class: pd.Series
    backfilled: pd.Series

    def __post_init__(self) -> None:
        periods_per_year(self.freq)
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ValueError("returns must have a DatetimeIndex")
        if not self.returns.index.is_monotonic_increasing or self.returns.index.has_duplicates:
            raise ValueError("returns index must be strictly increasing")
        if self.candidate not in self.returns.columns:
            raise ValueError(f"candidate {self.candidate!r} not in returns columns")
        if set(self.asset_class.index) != set(self.returns.columns):
            raise ValueError("asset_class must cover exactly the returns columns")
        if not self.backfilled.index.equals(self.returns.index):
            raise ValueError("backfilled must share the returns index")
        if self.backfilled.dtype != bool:
            raise ValueError("backfilled must be boolean")

    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)

    def live_only(self) -> MarketData:
        """Variant with backfilled/proxied candidate observations removed (rows dropped)."""
        keep = ~self.backfilled
        return replace(self, returns=self.returns.loc[keep], backfilled=self.backfilled.loc[keep])


class Loader(Protocol):
    """Source of :class:`MarketData`. No live Bloomberg calls: production reads PostgreSQL."""

    def load(self) -> MarketData: ...

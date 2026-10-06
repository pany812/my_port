"""Loaders: spec ``data`` section -> :class:`MarketData`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from workbench.data.base import MarketData
from workbench.data.synthetic import CandidateSpec, generate

if TYPE_CHECKING:
    from workbench.grid.spec import DataSpec
    from workbench.policy.saa import SAA


def load_market(spec: DataSpec, seed: int, saa: SAA | None = None) -> MarketData:
    """Load the market described by ``spec``. Synthetic draws use ``seed``; the SQL source reads
    the SAA's assets and the candidate through the data contract (``data.sql``)."""
    if spec.source == "synthetic":
        s = spec.synthetic
        cand = CandidateSpec(
            asset_id=spec.candidate,
            mu_annual=s.mu_annual,
            vol_annual=s.vol_annual,
            corr_to_equity=s.corr_to_equity,
            skew=s.skew,
            live_start=s.live_start,
            backfill=s.backfill,
        )
        return generate(
            seed=seed,
            start=spec.start,
            end=spec.end,
            freq=spec.frequency,
            candidate=cand,
            tail_df=s.tail_df,
        )
    if spec.source == "sql":
        from workbench.data.sql import load_sql

        if saa is None:
            raise ValueError("the SQL source needs the SAA (its assets and classes)")
        return load_sql(spec, saa.assets, saa.asset_class).market
    raise ValueError(f"unknown data source {spec.source!r}")

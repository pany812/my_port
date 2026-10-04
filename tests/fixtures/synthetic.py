"""Synthetic fixtures. The only data source tests may use."""

from __future__ import annotations

import pandas as pd

from workbench.allocators.base import FitContext
from workbench.data.base import MarketData
from workbench.data.synthetic import CandidateSpec, generate, placeholder_saa
from workbench.policy.compiled import CompiledPolicy

SEED = 42


def small_market(seed: int = SEED, **candidate_kwargs) -> MarketData:
    """Ten years of monthly returns, 8 building blocks + CAND."""
    return generate(
        seed=seed, start="2011-01", end="2020-12", candidate=CandidateSpec(**candidate_kwargs)
    )


def long_daily_market(seed: int = SEED, tail_df: float | None = None, **candidate_kwargs):
    """~5,200 business days: enough observations to test distributional targets."""
    return generate(
        seed=seed,
        start="2000-01-03",
        end="2019-12-31",
        freq="D",
        candidate=CandidateSpec(**candidate_kwargs),
        tail_df=tail_df,
    )


def fit_context(data: MarketData) -> FitContext:
    return FitContext(
        as_of=pd.Timestamp(data.returns.index[-1]),
        saa=placeholder_saa(data.candidate),
        candidate=data.candidate,
        policy=CompiledPolicy(freq=data.freq),
    )

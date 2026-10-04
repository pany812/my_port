"""History alignment and data fingerprinting."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from workbench.data.base import MarketData


def align_history(data: MarketData) -> MarketData:
    """Trim leading periods until every asset has a return (common history).

    Gaps after the common start are an error: they must be fixed upstream, never filled here.
    """
    complete = data.returns.notna().all(axis=1)
    if not complete.any():
        raise ValueError("no period where all assets have returns")
    first = complete.idxmax()
    r = data.returns.loc[first:]
    if r.isna().any().any():
        gaps = list(r.columns[r.isna().any()])
        raise ValueError(f"missing returns after common start {first.date()} in {gaps}")
    return replace(data, returns=r, backfilled=data.backfilled.loc[first:])


def data_vintage(data: MarketData) -> str:
    """sha256 over the exact returns, dates, asset ids and backfill flags.

    Any change to a single observation changes the vintage, so same spec + same vintage
    implies the allocators saw identical inputs.
    """
    h = hashlib.sha256()
    h.update(data.freq.encode())
    h.update("\x1f".join(map(str, data.returns.columns)).encode())
    h.update(data.returns.index.to_numpy(dtype="datetime64[ns]").astype(np.int64).tobytes())
    h.update(np.ascontiguousarray(data.returns.to_numpy(dtype=np.float64)).tobytes())
    h.update(data.backfilled.to_numpy(dtype=bool).tobytes())
    return h.hexdigest()

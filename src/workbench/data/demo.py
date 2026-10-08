"""A demo data-contract database built from the synthetic market (P2-M8).

``wb data demo`` writes ``wb_assets`` and ``wb_returns`` tables into a SQLite file so the SQL
source can be exercised end to end before real data exists. The candidate's rows start at its
live start; a proxy series (``CAND_PROXY``) carries the full history, so loading with
``live_start`` and ``candidate_proxy`` reproduces the synthetic market exactly (same data
vintage hash). Period ends are written as the last business day to exercise period alignment.
"""

from __future__ import annotations

import pandas as pd
import sqlalchemy as sa

from workbench.data.synthetic import PLACEHOLDER_SAA, CandidateSpec, generate
from workbench.db import engine

DEMO_VINTAGE = "2026-10-01T06:00"


def contract_tables(metadata: sa.MetaData, schema: str | None = None) -> tuple[sa.Table, sa.Table]:
    """The contract as tables (production uses views with the same columns)."""
    assets = sa.Table(
        "wb_assets", metadata,
        sa.Column("asset_id", sa.String(100), primary_key=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("currency", sa.String(3)), sa.Column("hedged", sa.Boolean),
        sa.Column("description", sa.String(200)), schema=schema,
    )  # fmt: skip
    returns = sa.Table(
        "wb_returns", metadata,
        sa.Column("asset_id", sa.String(100), primary_key=True),
        sa.Column("period_end", sa.Date, primary_key=True),
        sa.Column("frequency", sa.String(2), primary_key=True),
        sa.Column("vintage", sa.String(40), primary_key=True),
        sa.Column("simple_return", sa.Float, nullable=False), schema=schema,
    )  # fmt: skip
    return assets, returns


def write_demo_contract(
    url: str, seed: int = 42, start: str = "2001-01", end: str = "2026-09",
    live_start: str = "2016-01", candidate: str = "CAND", proxy: str = "CAND_PROXY",
    vintage: str = DEMO_VINTAGE, schema: str | None = None, benchmark: str = "SAA_BENCH",
) -> dict:  # fmt: skip
    """Create the contract tables in an empty SQLite database and fill them. Returns a summary."""
    if not url.startswith("sqlite:///"):
        raise ValueError("the demo database is written to SQLite only (sqlite:///path.db)")
    market = generate(
        seed=seed,
        start=start,
        end=end,
        freq="M",
        candidate=CandidateSpec(asset_id=candidate, live_start=live_start),
    )
    md = sa.MetaData()  # fmt: skip
    assets_t, returns_t = contract_tables(md, schema)
    eng = engine(url)
    with eng.begin() as c:
        have = set(sa.inspect(c).get_table_names(schema=schema))
        if have & {"wb_assets", "wb_returns"}:
            raise ValueError(f"{url} already has contract tables; use a new file")
        md.create_all(c)
        blocks = list(PLACEHOLDER_SAA.index)
        rows = [{"asset_id": a, "kind": "cash" if a == "CASH" else "block", "currency": "SEK",
                 "hedged": PLACEHOLDER_SAA.loc[a, "asset_class"] == "fixed_income",
                 "description": f"synthetic {PLACEHOLDER_SAA.loc[a, 'asset_class']}"}
                for a in blocks]  # fmt: skip
        rows += [{"asset_id": candidate, "kind": "candidate", "currency": "SEK", "hedged": True,
                  "description": "synthetic candidate (live from live_start)"},
                 {"asset_id": proxy, "kind": "proxy", "currency": "SEK", "hedged": True,
                  "description": "synthetic proxy for the candidate's backfill"},
                 {"asset_id": benchmark, "kind": "benchmark", "currency": "SEK", "hedged": False,
                  "description": "official SAA benchmark (synthetic: the blocks at SAA weights, "
                                 "rebalanced monthly)"}]  # fmt: skip
        c.execute(sa.insert(assets_t), rows)
        r = market.returns
        # the last business day of each month, as many databases store month ends
        bday = [d - pd.offsets.BDay(1) if d.weekday() >= 5 else d for d in r.index]
        out = []
        live = pd.Timestamp(live_start)
        for a in [*blocks, candidate]:
            for d, b, v in zip(r.index, bday, r[a], strict=True):
                if a == candidate and d < live:
                    continue
                out.append({"asset_id": a, "period_end": b.date(), "frequency": "M",
                            "vintage": vintage, "simple_return": float(v)})  # fmt: skip
        for b, v in zip(bday, r[candidate], strict=True):
            out.append({"asset_id": proxy, "period_end": b.date(), "frequency": "M",
                        "vintage": vintage, "simple_return": float(v)})  # fmt: skip
        bench = r[blocks] @ PLACEHOLDER_SAA["weight"][blocks]
        for b, v in zip(bday, bench, strict=True):
            out.append({"asset_id": benchmark, "period_end": b.date(), "frequency": "M",
                        "vintage": vintage, "simple_return": float(v)})  # fmt: skip
        c.execute(sa.insert(returns_t), out)
    eng.dispose()
    return {"url": url, "assets": len(rows), "rows": len(out), "vintage": vintage,
            "periods": len(r), "live_start": live_start}  # fmt: skip

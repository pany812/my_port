"""SQL data source (P2-M8): read returns through the data contract (docs/DATA_CONTRACT.md).

Two views (or tables), mapped by the database team onto the Bloomberg-fed tables:

- ``wb_assets(asset_id, kind, currency, hedged, description)``; kind in block / candidate /
  proxy / cash.
- ``wb_returns(asset_id, period_end, frequency, simple_return, vintage)``: simple total returns
  per period (decimal) in the base currency, following the experiment's hedging convention; the
  candidate's net of fees. ``frequency`` is a ``workbench.units`` code. ``vintage`` is a load
  tag that sorts in load order (e.g. an ISO timestamp).

Reading is read-only (``workbench.db.engine``). Per (asset, period, frequency) the row with the
greatest vintage <= the requested tag wins ("latest" = the greatest tag among the requested
assets). Finer data is compounded to the experiment's frequency; periods with fewer than half
the asset's median observations per period (partial first or last periods) are dropped. Dates are
aligned by period, so month-end business-day conventions do not matter.

No live Bloomberg calls, ever: this module only reads the database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sqlalchemy as sa

from workbench.data.base import MarketData
from workbench.db import engine, redact
from workbench.units import periods_per_year

ASSETS_VIEW = "wb_assets"
RETURNS_VIEW = "wb_returns"
KINDS = ("block", "candidate", "proxy", "cash", "benchmark")
PANDAS_PERIOD = {"D": "B", "W": "W-FRI", "M": "M", "Q": "Q", "A": "Y"}
MIN_PERIOD_COVERAGE = 0.5


class DataSourceError(ValueError):
    """The data source cannot deliver the requested data (message says what to fix)."""


@dataclass(frozen=True)
class SqlRead:
    """What was read: the market, the resolved vintage tag and the asset metadata."""

    market: MarketData
    vintage_tag: str
    assets: pd.DataFrame  # wb_assets rows for the requested ids
    source_frequency: dict[str, str]  # asset -> frequency used before compounding
    benchmark: pd.Series | None = None  # the SAA benchmark per period (data.sql.benchmark)


def _views(schema: str | None):
    assets = sa.table(ASSETS_VIEW, sa.column("asset_id"), sa.column("kind"),
                      sa.column("currency"), sa.column("hedged"), sa.column("description"),
                      schema=schema)  # fmt: skip
    returns = sa.table(RETURNS_VIEW, sa.column("asset_id"), sa.column("period_end"),
                       sa.column("frequency"), sa.column("simple_return"), sa.column("vintage"),
                       schema=schema)  # fmt: skip
    return assets, returns


def database_url(url_env: str) -> str:
    url = os.environ.get(url_env)
    if not url:
        raise DataSourceError(f"set ${url_env} to the SQLAlchemy URL of the data source")
    return url


def to_periods(s: pd.Series, freq: str) -> pd.Series:
    """Map dated returns to period-end timestamps of ``freq`` (one value per period required)."""
    periods = pd.DatetimeIndex(s.index).to_period(PANDAS_PERIOD[freq])
    if periods.has_duplicates:
        dup = periods[periods.duplicated()][0]
        raise DataSourceError(f"{s.name}: more than one {freq} observation in period {dup}")
    return pd.Series(s.to_numpy(), index=periods.to_timestamp(how="end").normalize(), name=s.name)


def compound(s: pd.Series, freq: str) -> pd.Series:
    """Compound finer simple returns to ``freq`` periods; drop partial periods (fewer than half
    the median number of observations)."""
    periods = pd.DatetimeIndex(s.index).to_period(PANDAS_PERIOD[freq])
    g = (1.0 + s).groupby(periods)
    out, n = g.prod() - 1.0, g.size()
    out = out[n >= MIN_PERIOD_COVERAGE * float(np.median(n))]
    out.index = out.index.to_timestamp(how="end").normalize()
    return out.rename(s.name)


def load_sql(
    data_spec, assets: list[str], asset_class: pd.Series, url: str | None = None
) -> SqlRead:
    """Read ``assets`` (SAA blocks and the candidate) per ``data_spec`` (a ``DataSpec`` with
    ``source: sql``). Returns per period of ``data_spec.frequency``, unaligned (the runner
    aligns); the candidate's pre-``live_start`` periods are flagged backfilled (from the proxy
    when one is named)."""
    src = data_spec.sql
    url = url or database_url(src.url_env)
    candidate, freq = data_spec.candidate, data_spec.frequency
    extra = [x for x in (src.candidate_proxy, src.benchmark) if x]
    ids = list(dict.fromkeys([*assets, *extra]))
    a_view, r_view = _views(src.schema)
    eng = engine(url, read_only=True)
    try:
        with eng.connect() as c:
            meta = pd.DataFrame(c.execute(sa.select(a_view).where(
                a_view.c.asset_id.in_(ids))).mappings().all())  # fmt: skip
            missing = sorted(set(ids) - set(meta.get("asset_id", [])))
            if missing:
                raise DataSourceError(f"assets not in {ASSETS_VIEW}: {missing}")
            bad = sorted(set(meta["kind"]) - set(KINDS))
            if bad:
                raise DataSourceError(f"{ASSETS_VIEW}.kind values {bad} not in {list(KINDS)}")
            tag = src.vintage
            if tag == "latest":
                tag = c.execute(sa.select(sa.func.max(r_view.c.vintage)).where(
                    r_view.c.asset_id.in_(ids))).scalar()  # fmt: skip
                if tag is None:
                    raise DataSourceError(f"no rows in {RETURNS_VIEW} for {ids}")
            rows = pd.DataFrame(
                c.execute(
                    sa.select(r_view).where(
                        r_view.c.asset_id.in_(ids), r_view.c.vintage <= str(tag)
                    )
                )
                .mappings()
                .all()
            )
    except sa.exc.SQLAlchemyError as e:  # fmt: skip
        raise DataSourceError(f"reading {redact(url)}: {type(e).__name__}: "
                         f"{str(e).splitlines()[0]}") from None  # fmt: skip
    finally:
        eng.dispose()
    if rows.empty:
        raise DataSourceError(f"no rows in {RETURNS_VIEW} at vintage <= {tag}")
    rows["period_end"] = pd.to_datetime(rows["period_end"])
    rows["simple_return"] = pd.to_numeric(rows["simple_return"])
    rows = (rows.sort_values("vintage")
            .drop_duplicates(["asset_id", "period_end", "frequency"], keep="last"))  # fmt: skip

    series, used = {}, {}
    target = periods_per_year(freq)
    for a in ids:
        r = rows[rows["asset_id"] == a]
        if r.empty:
            raise DataSourceError(f"{a}: no returns at vintage <= {tag}")
        freqs = set(r["frequency"])
        unknown = sorted(f for f in freqs if f not in PANDAS_PERIOD)
        if unknown:
            raise DataSourceError(f"{a}: unknown frequency codes {unknown}")
        if freq in freqs:
            f = freq
        else:
            finer = [f for f in freqs if periods_per_year(f) > target]
            if not finer:
                raise DataSourceError(f"{a}: only coarser data than {freq} ({sorted(freqs)})")
            f = max(finer, key=periods_per_year)  # the finest available
        s = r[r["frequency"] == f].set_index("period_end")["simple_return"].sort_index()
        s.name = a
        series[a] = to_periods(s, freq) if f == freq else compound(s, freq)
        used[a] = f

    wide = pd.DataFrame(series).sort_index()
    wide.index.name = "date"
    lo = pd.Period(data_spec.start, PANDAS_PERIOD[freq]).to_timestamp(how="start")
    hi = pd.Period(data_spec.end, PANDAS_PERIOD[freq]).to_timestamp(how="end")
    wide = wide[(wide.index >= lo) & (wide.index <= hi)]
    backfilled = pd.Series(False, index=wide.index, name=f"{candidate}_backfilled")
    if src.live_start is not None:
        live = pd.Period(src.live_start, PANDAS_PERIOD[freq]).to_timestamp(how="start")
        pre = wide.index < live
        if src.candidate_proxy is not None:
            wide.loc[pre, candidate] = wide.loc[pre, src.candidate_proxy]
        backfilled[pre] = wide.loc[pre, candidate].notna()
    bench = wide[src.benchmark].rename("benchmark") if src.benchmark else None
    wide = wide[assets]
    market = MarketData(returns=wide, freq=freq, candidate=candidate,
                        asset_class=asset_class.reindex(assets), backfilled=backfilled,
                        vintage_tag=str(tag))  # fmt: skip
    meta = meta.set_index("asset_id").loc[ids].reset_index()
    return SqlRead(market, str(tag), meta, used, bench)

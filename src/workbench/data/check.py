"""Data health check (P2-M8): ``wb data check specs/<name>.yaml`` before any real experiment.

Per asset: kind, source frequency, first and last observation, observations, gaps inside its own
history, annualised return and volatility, worst and best period, outliers (more than 5 standard
deviations from the mean) and, for the candidate, the backfilled share. Plus the common start the
runner would use and the data vintage hash, or the alignment error that would stop the run.

Reconciliation (P2-M8b, when ``data.sql.benchmark`` names the official SAA benchmark series): the
SAA built from the building blocks (fixed weights, rebalanced every period, the workbench's
convention) against the benchmark, per calendar year; years more than ``RECON_TOL_BP`` apart are
flagged. Advisory: a different rebalancing convention or fee treatment in the official series
shows up here; it does not stop a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from workbench.data.align import align_history, data_vintage
from workbench.data.base import MarketData
from workbench.evaluation.stats import ann_return, ann_vol
from workbench.grid.spec import ExperimentSpec
from workbench.policy.saa import SAA

OUTLIER_SD = 5.0
RECON_TOL_BP = 10.0  # per calendar year, basis points


@dataclass
class DataCheck:
    summary: pd.DataFrame  # field / value
    assets: pd.DataFrame  # one row per asset
    reconciliation: pd.DataFrame | None = None  # one row per calendar year

    @property
    def ok(self) -> bool:
        s = self.summary.set_index("field")["value"]
        return s.get("alignment") == "ok"


def asset_table(market: MarketData, kinds: dict[str, str] | None = None,
                source_frequency: dict[str, str] | None = None) -> pd.DataFrame:  # fmt: skip
    """Per-asset health on the unaligned returns (decimal; annual figures geometric / std)."""
    rows = []
    for a in market.returns.columns:
        s = market.returns[a]
        obs = s.dropna()
        first, last = (obs.index[0], obs.index[-1]) if len(obs) else (None, None)
        inside = s.loc[first:last] if first is not None else s.iloc[:0]
        sd = float(obs.std(ddof=1)) if len(obs) > 1 else np.nan
        z = (obs - obs.mean()).abs() / sd if sd and np.isfinite(sd) else obs * 0
        rows.append({
            "asset_id": a, "kind": (kinds or {}).get(a, ""),
            "source_frequency": (source_frequency or {}).get(a, market.freq),
            "first": None if first is None else first.date().isoformat(),
            "last": None if last is None else last.date().isoformat(),
            "n_obs": len(obs), "n_gaps": int(inside.isna().sum()),
            "ann_return": ann_return(obs, market.freq) if len(obs) else np.nan,
            "ann_vol": ann_vol(obs, market.freq) if len(obs) > 1 else np.nan,
            "worst": float(obs.min()) if len(obs) else np.nan,
            "best": float(obs.max()) if len(obs) else np.nan,
            "n_outliers": int((z > OUTLIER_SD).sum()),
            "backfilled_share": (float(market.backfilled[obs.index].mean())
                                 if a == market.candidate and len(obs) else np.nan),
        })  # fmt: skip
    return pd.DataFrame(rows)


def reconcile(
    returns: pd.DataFrame, saa: pd.Series, candidate: str, benchmark: pd.Series
) -> pd.DataFrame:
    """Calendar-year returns of the building-block SAA vs the benchmark (decimal; diff in bp).

    Columns: year, n_periods, saa_return, benchmark_return, diff_bp, flag ("⚠" beyond
    ``RECON_TOL_BP``). Uses the periods where every block and the benchmark have a return.
    """
    blocks = [a for a in saa.index if a != candidate]
    df = returns[blocks].join(benchmark.rename("benchmark"), how="inner").dropna()
    port = df[blocks] @ saa[blocks]
    yearly = pd.DataFrame({"saa": 1.0 + port, "bench": 1.0 + df["benchmark"]})
    g = yearly.groupby(df.index.year)
    out = pd.DataFrame({"n_periods": g.size(), "saa_return": g["saa"].prod() - 1.0,
                        "benchmark_return": g["bench"].prod() - 1.0})  # fmt: skip
    out["diff_bp"] = (out["saa_return"] - out["benchmark_return"]) * 1e4
    out["flag"] = np.where(out["diff_bp"].abs() > RECON_TOL_BP, "⚠", "")
    return out.rename_axis("year").reset_index()


def _recon_summary(t: pd.DataFrame) -> str:
    if t.empty:
        return "no overlapping periods with the benchmark"
    i = t["diff_bp"].abs().idxmax()
    worst = f"max |Δ| {abs(t.loc[i, 'diff_bp']):.1f} bp ({t.loc[i, 'year']})"
    n = int((t["flag"] != "").sum())
    if n == 0:
        return f"ok: {len(t)} years within {RECON_TOL_BP:g} bp, {worst}"
    return f"⚠ {n} of {len(t)} years beyond {RECON_TOL_BP:g} bp, {worst}"


def data_check(spec: ExperimentSpec) -> DataCheck:
    """Load the spec's data the way the runner would and report its health."""
    saa = SAA.from_version(spec.saa_version, spec.data.candidate)
    kinds, used, tag, bench = {}, {}, None, None
    if spec.data.source == "sql":
        from workbench.data.sql import load_sql

        read = load_sql(spec.data, saa.assets, saa.asset_class)
        market, tag, bench = read.market, read.vintage_tag, read.benchmark
        kinds = dict(zip(read.assets["asset_id"], read.assets["kind"], strict=True))
        used = read.source_frequency
    else:
        from workbench.data.loaders import load_market

        market = load_market(spec.data, spec.seed, saa)
    rows = [
        ("source", spec.data.source),
        ("frequency", market.freq),
        ("vintage tag", tag or "–"),
        ("SAA version", f"{saa.version} ({saa.source})"),
    ]
    missing = sorted(set(saa.assets) - set(market.assets))
    if missing:
        rows.append(("alignment", f"SAA assets missing from the data: {missing}"))
    else:
        try:
            aligned = align_history(market)
            rows += [("alignment", "ok"),
                     ("common start", aligned.returns.index[0].date().isoformat()),
                     ("periods after alignment", str(len(aligned.returns))),
                     ("data vintage (sha256)", data_vintage(aligned))]  # fmt: skip
        except ValueError as e:
            rows.append(("alignment", f"error: {e}"))
    recon = None
    if bench is not None and not missing:
        recon = reconcile(market.returns, saa.weights, saa.candidate, bench)
        rows.append(("reconciliation", _recon_summary(recon)))
    elif spec.data.source == "sql":
        rows.append(("reconciliation", "no benchmark (set data.sql.benchmark)"))
    return DataCheck(pd.DataFrame(rows, columns=["field", "value"]),
                     asset_table(market, kinds, used), recon)  # fmt: skip

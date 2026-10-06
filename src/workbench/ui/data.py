"""Data for the read-only UI (P2-M7): plain functions over the registry, no Streamlit.

Everything is computed by the same functions as ``summary.md`` and ``memo.md`` (corridor, OOS,
evidence, views, stress, agreement, memo), so the UI cannot disagree with them. A stored
experiment never changes, so results are cached per experiment id by the app.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from workbench.evaluation.corridor import cell_frame
from workbench.evaluation.expost import cell_label
from workbench.evaluation.stats import drawdowns
from workbench.grid.spec import parse_spec
from workbench.registry.models import Cell
from workbench.registry.store import REFERENCE_ALLOCATOR, Registry, loads_or_none

BANDS = ("p10", "p25", "median", "p75", "p90")


def experiments_table(registry: Registry) -> pd.DataFrame:
    """One row per experiment, newest first: name, experiment_id, created_at, mode, candidate,
    spec_hash (12 chars), n_cells and counts by status (grid cells, SAA reference excluded)."""
    exps = registry.experiments()
    if exps.empty:
        return pd.DataFrame()
    q = (select(Cell.experiment_id, Cell.status, func.count())
         .where(Cell.allocator != REFERENCE_ALLOCATOR)
         .group_by(Cell.experiment_id, Cell.status))  # fmt: skip
    with registry.engine.connect() as c:
        counts = pd.DataFrame(c.execute(q).all(), columns=["experiment_id", "status", "n"])
    wide = counts.pivot_table(index="experiment_id", columns="status", values="n",
                              aggfunc="sum", fill_value=0)  # fmt: skip
    out = exps[["name", "experiment_id", "created_at", "candidate_id", "spec_hash"]].copy()
    out["mode"] = exps["spec_yaml"].map(lambda y: parse_spec(y).backtest.mode)
    out["spec_hash"] = out["spec_hash"].str[:12]
    out = out.merge(wide, left_on="experiment_id", right_index=True, how="left")
    statuses = [s for s in ("ok", "infeasible", "solver_error", "exception") if s in out]
    out[statuses] = out[statuses].fillna(0).astype(int)
    out.insert(5, "n_cells", out[statuses].sum(axis=1))
    return out.sort_values("created_at", ascending=False).reset_index(drop=True)


def provenance(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """field / value rows describing how the experiment was produced."""
    exp = registry.experiment(experiment_id)
    spec = parse_spec(exp["spec_yaml"])
    d = spec.data
    rows = [
        ("experiment", exp["name"]), ("experiment_id", exp["experiment_id"]),
        ("spec_hash", exp["spec_hash"]), ("data_vintage", exp["data_vintage"]),
        ("riskfolio-lib", exp["riskfolio_version"]), ("seed", str(exp["seed"])),
        ("created", f"{exp['created_at']:%Y-%m-%d %H:%M}" if exp["created_at"] else ""),
        ("data", f"{d.source}, {d.frequency}, {d.start} .. {d.end}"),
        ("data vintage tag", exp.get("data_vintage_tag") or "–"),
        ("currency / hedging", f"{d.base_currency} / {d.hedging}"),
        ("candidate", exp["candidate_id"]), ("SAA version", exp["saa_version"]),
        ("mode", spec.backtest.mode),
        ("CMA", "" if spec.cma is None else spec.cma.version),
        ("stress", "no" if spec.stress is None else "yes"),
        ("decision", "no" if spec.decision is None else "yes"),
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=["field", "value"])


def corridor_band(corr: pd.DataFrame, measure: str, variant: str, group: str | None = None,
                  value: str | None = None) -> pd.DataFrame:  # fmt: skip
    """Corridor quantiles through time for one measure, variant and (optional) group value:
    window_end (datetime) plus p10, p25, median, p75, p90, n_ok, n_cells."""
    c = corr[(corr["measure"] == measure) & (corr["data_variant"] == variant)]
    if group is not None and value is not None:
        c = c[c[group].astype(str) == str(value)]
    out = c[["window_end", *BANDS, "n_ok", "n_cells"]].copy()
    out["window_end"] = pd.to_datetime(out["window_end"])
    return out.sort_values("window_end").reset_index(drop=True)


def cells_table(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """Grid cells with labels: data_variant, window_end, label, constraint_set, status,
    capital_weight, risk_share:<lens>, family, library, allocator, elapsed_s, message (first
    line), cell_id."""
    frame = cell_frame(registry, experiment_id)
    cells = registry.cells(experiment_id).set_index("cell_id")
    frame["label"] = [cell_label(a, p, e) for a, p, e in zip(
        cells.loc[frame["cell_id"], "allocator"], cells.loc[frame["cell_id"], "params_json"],
        cells.loc[frame["cell_id"], "estimator_json"], strict=True)]  # fmt: skip
    frame["elapsed_s"] = cells.loc[frame["cell_id"], "elapsed_s"].to_numpy()
    frame["message"] = (cells.loc[frame["cell_id"], "message"].fillna("")
                        .str.split("\n").str[0].to_numpy())  # fmt: skip
    keep = ["data_variant", "window_end", "label", "constraint_set", "status", "capital_weight",
            *[c for c in frame.columns if c.startswith("risk_share:")], "family", "library",
            "allocator", "elapsed_s", "message", "cell_id"]  # fmt: skip
    return frame[keep].sort_values(["data_variant", "window_end", "label"]).reset_index(drop=True)


def cell_detail(registry: Registry, experiment_id: str, cell_id: str) -> dict:
    """One cell: its row, params, estimator, diagnostics, message, weights vs the SAA reference
    at the same date (DataFrame asset, weight, saa, active) and metrics (metric, lens, value)."""
    cells = registry.cells(experiment_id)
    row = cells[cells["cell_id"] == cell_id]
    if row.empty:
        raise KeyError(f"no cell {cell_id!r}")
    r = row.iloc[0]
    w = registry.weights(experiment_id, include_reference=True)
    ref = registry.reference(experiment_id)
    ref_id = ref[(ref["data_variant"] == r["data_variant"])
                 & (ref["window_end"] == r["window_end"])]["cell_id"]  # fmt: skip
    saa = (w[w["cell_id"].isin(ref_id)].set_index("asset_id")["weight"] if len(ref_id)
           else pd.Series(dtype=float))  # fmt: skip
    mine = w[w["cell_id"] == cell_id].set_index("asset_id")["weight"]
    assets = list(saa.index) or list(mine.index)
    weights = pd.DataFrame({"asset": assets, "weight": mine.reindex(assets).to_numpy(),
                            "saa": saa.reindex(assets).to_numpy()})  # fmt: skip
    weights["active"] = weights["weight"] - weights["saa"]
    m = registry.metrics(experiment_id)
    return {
        "row": r, "params": json.loads(r["params_json"]),
        "estimator": loads_or_none(r["estimator_json"]),
        "diagnostics": json.loads(r["diagnostics_json"] or "{}"), "message": r["message"] or "",
        "weights": weights if len(mine) else weights.assign(weight=np.nan, active=np.nan),
        "metrics": m[m["cell_id"] == cell_id][["metric", "lens", "value"]].reset_index(drop=True),
    }  # fmt: skip


def path_frames(
    registry: Registry, experiment_id: str, variant: str, config_ids: list[str],
    labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """(wealth, drawdown) frames for the SAA and ``config_ids`` on net out-of-sample returns:
    index date, one column per path (labelled), wealth starting at 1, drawdowns >= 0."""
    p = registry.oos_returns(experiment_id, include_reference=True)
    if p.empty:
        return pd.DataFrame(), pd.DataFrame()
    p = p[(p["data_variant"] == variant)
          & p["config_id"].isin([REFERENCE_ALLOCATOR, *config_ids])]  # fmt: skip
    wide = p.assign(date=pd.to_datetime(p["date"])).pivot(
        index="date", columns="config_id", values="portfolio_return_net")  # fmt: skip
    names = {REFERENCE_ALLOCATOR: "SAA", **(labels or {})}
    wide = wide[[c for c in [REFERENCE_ALLOCATOR, *config_ids] if c in wide]].rename(
        columns=lambda c: names.get(c, c))  # fmt: skip
    return (1.0 + wide).cumprod(), wide.apply(drawdowns)


def config_labels(registry: Registry, experiment_id: str) -> dict[str, str]:
    """config_id -> "label | constraint set" for every grid configuration."""
    cells = registry.cells(experiment_id).groupby("config_id").first()
    return {cid: f"{cell_label(r.allocator, r.params_json, r.estimator_json)} | "
                 f"{r.constraint_set}" for cid, r in cells.iterrows()}  # fmt: skip


def sharpe_table(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """Sharpe-difference tests and deflated Sharpe ratios per configuration and variant, with
    labels: data_variant, label, sr_ann, sr_saa_ann, diff_ann, p, p_bh, dsr, note."""
    ev = registry.evidence(experiment_id)
    if ev.empty:
        return pd.DataFrame()
    labels = config_labels(registry, experiment_id)
    sd = ev[ev["test"] == "sharpe_diff"]
    dsr = ev[ev["test"] == "dsr"].set_index(["data_variant", "subject"])["statistic"]
    rows = []
    for r in sd.itertuples():
        e = json.loads(r.extra_json)
        rows.append(
            {
                "data_variant": r.data_variant,
                "label": labels.get(r.subject, r.subject),
                "sr_ann": e.get("sr_ann"),
                "sr_saa_ann": e.get("sr_saa_ann"),
                "diff_ann": e.get("diff_ann"),
                "p": r.p_value,
                "p_bh": e.get("p_bh"),
                "dsr": dsr.get((r.data_variant, r.subject)),
                "note": e.get("note") or "",
            }
        )
    t = pd.DataFrame(rows)  # fmt: skip
    for c in ("sr_ann", "sr_saa_ann", "diff_ann", "p", "p_bh", "dsr"):
        if c in t:
            t[c] = pd.to_numeric(t[c])
    return t.sort_values(["data_variant", "p_bh"]).reset_index(drop=True) if not t.empty else t


def evidence_rows(registry: Registry, experiment_id: str, tests: tuple[str, ...]) -> pd.DataFrame:
    """Evidence rows whose test is in ``tests``, with their extra fields as columns."""
    ev = registry.evidence(experiment_id)
    ev = ev[ev["test"].isin(tests)]
    if ev.empty:
        return pd.DataFrame()
    extra = pd.DataFrame([json.loads(x) for x in ev["extra_json"]], index=ev.index)
    return pd.concat([ev.drop(columns="extra_json"), extra], axis=1).reset_index(drop=True)

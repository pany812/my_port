"""Allocation corridor: distribution of the candidate's capital weight and risk share.

Statistics are over ``ok`` cells; failed cells are counted by status, never dropped silently.
The SAA reference row is excluded by the registry reads (see ``REFERENCE_ALLOCATOR``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from workbench.evaluation.metrics import RISK_SHARE
from workbench.grid.spec import parse_spec
from workbench.registry.store import Registry, loads_or_none

ZERO_THRESHOLD = 0.0025  # "at zero": below 0.25%
FAILED_STATUSES = ("infeasible", "solver_error", "exception")
FAMILIES = {
    "static_saa": "naive",
    "saa_plus": "naive",
    "equal_weight": "naive",
    "inverse_vol": "naive",
    "riskfolio_mean_risk": "mean_risk",
    "riskfolio_hc": "hc",
}
GROUP_KEYS = ("family", "allocator", "rm", "estimator", "constraint_set")
QUANTILES = {"p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90}


def cell_frame(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """One row per grid cell: ids, status, group keys and the measures (NaN if unavailable)."""
    exp = registry.experiment(experiment_id)
    cand = exp["candidate_id"]
    lenses = parse_spec(exp["spec_yaml"]).risk_lenses
    cells = registry.cells(experiment_id)
    if cells.empty:
        raise ValueError(f"experiment {experiment_id} has no cells")
    df = cells[["cell_id", "data_variant", "window_end", "status", "allocator", "constraint_set"]]
    df = df.copy()
    params = cells["params_json"].map(json.loads)
    est = cells["estimator_json"].map(loads_or_none)
    df["family"] = df["allocator"].map(FAMILIES).fillna("other")
    df["rm"] = params.map(lambda p: p.get("rm", ""))
    df["estimator"] = est.map(
        lambda e: "none" if e is None else f"{e['method_mu']}/{e['method_cov']}"
    )

    w = registry.weights(experiment_id)
    cw = w.loc[w["asset_id"] == cand].set_index("cell_id")["weight"]
    df["capital_weight"] = df["cell_id"].map(cw)

    m = registry.metrics(experiment_id)
    rs = m[m["metric"] == RISK_SHARE]
    for lens in lenses:
        s = rs[rs["lens"] == lens].set_index("cell_id")["value"]
        df[f"risk_share:{lens}"] = df["cell_id"].map(s)
    return df


def corridor(
    registry: Registry,
    experiment_id: str,
    by: list[str] | tuple[str, ...] = (),
    threshold: float = ZERO_THRESHOLD,
) -> pd.DataFrame:
    """Corridor rows per (data_variant, window_end, *by, measure).

    Columns: n_cells, n_ok, n_infeasible, n_solver_error, n_exception, n_metric_missing,
    median, p25, p75, p10, p90 (decimal weights or shares) and share_below_0.25pct
    (fraction of ok cells with a value below ``threshold``).
    """
    unknown = set(by) - set(GROUP_KEYS)
    if unknown:
        raise ValueError(f"unknown group keys {sorted(unknown)}; allowed: {list(GROUP_KEYS)}")
    df = cell_frame(registry, experiment_id)
    measures = ["capital_weight"] + [c for c in df.columns if c.startswith("risk_share:")]
    keys = ["data_variant", "window_end", *by]
    rows = []
    for group_vals, g in df.groupby(keys, sort=True):
        base = dict(zip(keys, group_vals, strict=True))
        ok = g[g["status"] == "ok"]
        counts = {"n_cells": len(g), "n_ok": len(ok)}
        counts |= {f"n_{s}": int((g["status"] == s).sum()) for s in FAILED_STATUSES}
        for measure in measures:
            vals = ok[measure].dropna().to_numpy(dtype=float)
            row = {**base, "measure": measure, **counts, "n_metric_missing": len(ok) - len(vals)}
            if len(vals):
                q = np.quantile(vals, list(QUANTILES.values()))
                row |= dict(zip(QUANTILES, q, strict=True))
                row["share_below_0.25pct"] = float((vals < threshold).mean())
            else:
                row |= dict.fromkeys(QUANTILES, np.nan) | {"share_below_0.25pct": np.nan}
            rows.append(row)
    cols = [*keys, "measure", "n_cells", "n_ok", *(f"n_{s}" for s in FAILED_STATUSES),
            "n_metric_missing", "median", "p25", "p75", "p10", "p90",
            "share_below_0.25pct"]  # fmt: skip
    return pd.DataFrame(rows)[cols]

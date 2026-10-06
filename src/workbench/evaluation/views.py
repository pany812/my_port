"""Experiment views for P2-M3 report sections (DataFrames; rendering lives in ``report``).

- :func:`risk_budget_view`: "how much of our risk should it carry?" Capital weight implied by
  each target risk share, per lens, with the realised share.
- :func:`sweep_view`: "what fits inside a TE budget?" (or any swept constraint key): the
  candidate's corridor and realised out-of-sample TE per swept value.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from workbench.allocators.factory import full_params
from workbench.evaluation.corridor import FAMILIES, corridor, library_of
from workbench.evaluation.oos import oos_table
from workbench.grid.spec import parse_spec, sweep_metadata
from workbench.registry.store import Registry


def _candidate_weights(registry: Registry, experiment_id: str, cells: pd.DataFrame) -> pd.Series:
    cand = registry.experiment(experiment_id)["candidate_id"]
    w = registry.weights(experiment_id)
    cw = w.loc[w["asset_id"] == cand].set_index("cell_id")["weight"]
    return cells["cell_id"].map(cw)


def risk_budget_view(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """One row per (variant, rm, rest, target share, library) over all rebalance dates.

    Columns: n_cells, n_ok, weight_median, weight_p25, weight_p75 (candidate capital weight,
    decimal, over ok cells and dates), weight_latest (median at the latest date) and
    realised_share_median (candidate risk share realised under ``rm``).
    """
    cells = registry.cells(experiment_id)
    rb = cells[cells["allocator"].map(FAMILIES) == "risk_budget"].copy()
    if rb.empty:
        return pd.DataFrame()
    params = [full_params(a, json.loads(p)) for a, p in zip(rb["allocator"], rb["params_json"],
                                                             strict=True)]  # fmt: skip
    params = pd.Series(params, index=rb.index)
    rb["rm"] = params.map(lambda p: p["rm"])
    rb["rest"] = params.map(lambda p: p["rest"])
    rb["target_share"] = params.map(lambda p: p["candidate_share"])
    rb["library"] = rb["allocator"].map(library_of)
    rb["weight"] = _candidate_weights(registry, experiment_id, rb)
    rb["realised"] = rb["diagnostics_json"].map(lambda d: json.loads(d).get("realised_share"))
    keys = ["data_variant", "rm", "rest", "target_share", "library"]
    rows = []
    for k, g in rb.groupby(keys, sort=True):
        ok = g[g["status"] == "ok"]
        last = ok[ok["window_end"] == ok["window_end"].max()] if not ok.empty else ok
        w = ok["weight"].dropna().to_numpy(float)
        rows.append({
            **dict(zip(keys, k, strict=True)),
            "n_cells": len(g), "n_ok": len(ok),
            "weight_median": np.median(w) if len(w) else np.nan,
            "weight_p25": np.quantile(w, 0.25) if len(w) else np.nan,
            "weight_p75": np.quantile(w, 0.75) if len(w) else np.nan,
            "weight_latest": last["weight"].median() if not last.empty else np.nan,
            "realised_share_median": pd.to_numeric(ok["realised"]).median(),
        })  # fmt: skip
    return pd.DataFrame(rows)


def sweep_view(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """One row per (variant, swept constraint set) for sets expanded from ``{sweep: [...]}``.

    Columns: base, swept (e.g. "te_annual"), value, constraint_set, n_cells, n_ok,
    median / p25 / p75 (candidate capital weight at the latest date), median_through_time,
    n_configs_oos (configurations with at least one successful rebalance), oos_te_median
    (their realised OOS TE vs SAA, annual) and oos_weight_median (their median candidate
    weight). Configurations that failed at every date hold the SAA (TE exactly 0) and would
    otherwise drag the median to zero, so they are excluded and counted separately.
    """
    spec = parse_spec(registry.experiment(experiment_id)["spec_yaml"])
    swept = {cs.name: sweep_metadata(cs) for cs in spec.constraint_sets}
    swept = {k: v for k, v in swept.items() if v}
    if not swept:
        return pd.DataFrame()
    corr = corridor(registry, experiment_id, by=["constraint_set"])
    corr = corr[(corr["measure"] == "capital_weight") & corr["constraint_set"].isin(swept)]
    last = corr.groupby("data_variant")["window_end"].transform("max")
    latest = corr[corr["window_end"] == last].set_index(["data_variant", "constraint_set"])
    through = corr.groupby(["data_variant", "constraint_set"])["median"].median()
    oos = oos_table(registry, experiment_id)
    traded = (oos[(oos["row"] == "cell") & (oos["n_failed_rebalances"] < oos["n_rebalances"])]
              if not oos.empty else oos)  # fmt: skip
    oos_by = (traded.groupby(["data_variant", "constraint_set"])
              .agg(n_configs_oos=("config_id", "size"),
                   oos_te_median=("te_vs_saa", "median"),
                   oos_weight_median=("candidate_weight_median", "median"))
              if not traded.empty else pd.DataFrame())  # fmt: skip
    rows = []
    for (variant, cs_name), r in latest.iterrows():
        meta = swept[cs_name]
        row = {
            "data_variant": variant, "base": meta["base"],
            "swept": ",".join(meta["keys"]),
            "value": ",".join(f"{v:g}" if isinstance(v, int | float) else str(v)
                              for v in meta["keys"].values()),
            "constraint_set": cs_name, "n_cells": r["n_cells"], "n_ok": r["n_ok"],
            "median": r["median"], "p25": r["p25"], "p75": r["p75"],
            "median_through_time": through.get((variant, cs_name), np.nan),
        }  # fmt: skip
        if not oos_by.empty and (variant, cs_name) in oos_by.index:
            row |= oos_by.loc[(variant, cs_name)].to_dict()
        rows.append(row)

    def order(r):
        return (r["data_variant"], r["base"], r["swept"],
                tuple(swept[r["constraint_set"]]["keys"].values()))  # fmt: skip

    return pd.DataFrame(sorted(rows, key=order)).reset_index(drop=True)

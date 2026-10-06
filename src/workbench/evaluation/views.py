"""Experiment views for P2-M3 report sections (DataFrames; rendering lives in ``report``).

- :func:`risk_budget_view`: "how much of our risk should it carry?" Capital weight implied by
  each target risk share, per lens, with the realised share.
- :func:`sweep_view`: "what fits inside a TE budget?" (or any swept constraint key): the
  candidate's corridor and realised out-of-sample TE per swept value.
- :func:`breakeven_view` / :func:`weight_by_view` (P2-M4): "what would it have to earn?"
  (Black–Litterman breakeven per target weight) and "what weight does a view give?".
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


BL_SETTINGS = ["obj", "prior_sharpe", "method_cov"]


def _bl_cells(registry: Registry, experiment_id: str) -> pd.DataFrame:
    cells = registry.cells(experiment_id)
    bl = cells[cells["allocator"].map(FAMILIES) == "black_litterman"].copy()
    if bl.empty:
        return bl
    params = pd.Series([full_params(a, json.loads(p)) for a, p in
                        zip(bl["allocator"], bl["params_json"], strict=True)],
                       index=bl.index)  # fmt: skip
    for k in ("view_annual", "confidence", "target_weight", *BL_SETTINGS):
        bl[k] = params.map(lambda p, k=k: p[k])
    bl["library"] = bl["allocator"].map(library_of)
    bl["weight"] = _candidate_weights(registry, experiment_id, bl)
    diag = bl["diagnostics_json"].map(json.loads)
    for k in (
        "equilibrium_excess_annual",
        "premium_annual",
        "posterior_excess_annual",
        "candidate_sharpe_annual",
        "unconstrained_sharpe_annual",
        "rho_saa",
    ):
        bl[k] = pd.to_numeric(diag.map(lambda d, k=k: d.get(k)), errors="coerce")
    return bl


def _q(x: pd.Series, q: float) -> float:
    x = x.dropna().to_numpy(float)
    return float(np.quantile(x, q)) if len(x) else np.nan


def breakeven_view(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """Black–Litterman breakeven per (variant, constraint set, target weight, library, settings).

    Over rebalance dates (annual, decimal): excess_median / _p25 / _p75 = expected excess return
    over rf the candidate needs (posterior, full confidence) for the optimiser to give it the
    target weight; equilibrium_median = what the SAA prior implies; premium_median = the
    difference; sharpe_median = required Sharpe ratio; sharpe_unconstrained_median = the
    closed form without policy limits; excess_latest = median at the latest date. n_reached =
    ok cells; n_unreachable = targets a policy limit stops (``infeasible``, counted).
    """
    bl = _bl_cells(registry, experiment_id)
    if bl.empty or bl["target_weight"].isna().all():
        return pd.DataFrame()
    bl = bl[bl["target_weight"].notna()]
    keys = ["data_variant", "constraint_set", "target_weight", "library", *BL_SETTINGS]
    rows = []
    for k, g in bl.groupby(keys, sort=True):
        ok = g[g["status"] == "ok"]
        last = ok[ok["window_end"] == ok["window_end"].max()] if not ok.empty else ok
        rows.append({
            **dict(zip(keys, k, strict=True)),
            "n_cells": len(g), "n_reached": len(ok),
            "n_unreachable": int(g["message"].fillna("").str.contains("unreachable").sum()),
            "excess_median": _q(ok["posterior_excess_annual"], 0.5),
            "excess_p25": _q(ok["posterior_excess_annual"], 0.25),
            "excess_p75": _q(ok["posterior_excess_annual"], 0.75),
            "equilibrium_median": _q(g["equilibrium_excess_annual"], 0.5),
            "premium_median": _q(ok["premium_annual"], 0.5),
            "sharpe_median": _q(ok["candidate_sharpe_annual"], 0.5),
            "sharpe_unconstrained_median": _q(g["unconstrained_sharpe_annual"], 0.5),
            "excess_latest": _q(last["posterior_excess_annual"], 0.5),
        })  # fmt: skip
    return pd.DataFrame(rows)


def weight_by_view(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """Candidate weight per stated view (variant, constraint set, view, confidence, library,
    settings): weight_median / _p25 / _p75 over dates, weight_latest, equilibrium_median and
    posterior_median (the candidate's posterior expected excess return; annual, decimal)."""
    bl = _bl_cells(registry, experiment_id)
    if bl.empty or bl["view_annual"].isna().all():
        return pd.DataFrame()
    bl = bl[bl["view_annual"].notna()]
    keys = ["data_variant", "constraint_set", "view_annual", "confidence", "library",
            *BL_SETTINGS]  # fmt: skip
    rows = []
    for k, g in bl.groupby(keys, sort=True):
        ok = g[g["status"] == "ok"]
        last = ok[ok["window_end"] == ok["window_end"].max()] if not ok.empty else ok
        rows.append({
            **dict(zip(keys, k, strict=True)),
            "n_cells": len(g), "n_ok": len(ok),
            "weight_median": _q(ok["weight"], 0.5), "weight_p25": _q(ok["weight"], 0.25),
            "weight_p75": _q(ok["weight"], 0.75), "weight_latest": _q(last["weight"], 0.5),
            "equilibrium_median": _q(g["equilibrium_excess_annual"], 0.5),
            "posterior_median": _q(ok["posterior_excess_annual"], 0.5),
        })  # fmt: skip
    return pd.DataFrame(rows)

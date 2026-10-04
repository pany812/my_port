"""Out-of-sample (walk-forward) table: stats of each configuration's path vs the SAA path.

Computed at report time from the stored ``oos_returns`` paths (no data reload). Statistics use
the CIO-facing definitions in ``workbench.evaluation.stats``; both paths cover the same dates.
"""

from __future__ import annotations

import pandas as pd

from workbench.evaluation.expost import cell_label
from workbench.evaluation.stats import ann_return, summary
from workbench.grid.spec import parse_spec
from workbench.registry.store import REFERENCE_ALLOCATOR, Registry
from workbench.units import periods_per_year

OOS = "oos"


def oos_table(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """One row per (data_variant, configuration) plus the SAA row per variant.

    Statistics are on **net** returns (gross minus transaction costs; equal without costs).
    Columns: data_variant, row ("SAA"/"cell"), label, constraint_set, config_id, start, end,
    n_periods, n_rebalances, n_failed_rebalances, ann_return, ann_vol, cvar95 (per period),
    cdar95, max_dd, te_vs_saa, turnover_ann (one-way, decimal per year), ann_return_gross,
    cost_drag_ann (decimal per year), n_trades, n_liquidity_adjusted, n_adjustment_breaches
    (liquidity-adjusted targets that broke the policy) and the candidate's median weight over
    successful rebalances.
    """
    exp = registry.experiment(experiment_id)
    freq = parse_spec(exp["spec_yaml"]).data.frequency
    paths = registry.oos_returns(experiment_id, include_reference=True)
    if paths.empty:
        return pd.DataFrame()
    cells = registry.cells(experiment_id)
    w = registry.weights(experiment_id)
    cand_w = w.loc[w["asset_id"] == exp["candidate_id"]].set_index("cell_id")["weight"]
    cells = cells.assign(
        candidate_weight=cells["cell_id"].map(cand_w),
        adjustment_breach=cells["diagnostics_json"].str.contains('"adjustment_violations"'),
    )
    meta = cells.groupby(["data_variant", "config_id"]).agg(
        allocator=("allocator", "first"),
        params_json=("params_json", "first"),
        estimator_json=("estimator_json", "first"),
        constraint_set=("constraint_set", "first"),
        cell_index=("cell_index", "first"),
        n_rebalances=("status", "size"),
        n_failed_rebalances=("status", lambda s: int((s != "ok").sum())),
        candidate_weight_median=("candidate_weight", "median"),
        n_adjustment_breaches=("adjustment_breach", "sum"),
    )

    rows = []
    for variant, vp in paths.groupby("data_variant", sort=True):
        cols = ["portfolio_return", "portfolio_return_net", "turnover", "cost",
                "liquidity_adjusted"]  # fmt: skip
        series = {cid: g.set_index("date")[cols] for cid, g in vp.groupby("config_id")}
        bench = series[REFERENCE_ALLOCATOR]["portfolio_return_net"]
        years = len(bench) / periods_per_year(freq)
        for cid, df in series.items():
            r = df["portfolio_return_net"]
            if not r.index.equals(bench.index):
                raise ValueError(f"path {cid} ({variant}) does not cover the SAA path's dates")
            is_ref = cid == REFERENCE_ALLOCATOR
            row = {
                "data_variant": variant,
                "row": "SAA" if is_ref else "cell",
                "config_id": cid,
                "start": r.index[0],
                "end": r.index[-1],
                "n_periods": len(r),
                **summary(r, bench, freq),
                "turnover_ann": float(df["turnover"].sum() / years),
                "ann_return_gross": ann_return(df["portfolio_return"], freq),
                "cost_drag_ann": float(df["cost"].sum() / years),
                "n_trades": int((df["turnover"] > 0).sum()),
                "n_liquidity_adjusted": int(df["liquidity_adjusted"].astype(bool).sum()),
            }
            if is_ref:
                row |= {"label": "SAA", "constraint_set": "", "order": -1}
            else:
                m = meta.loc[(variant, cid)]
                row |= {
                    "label": cell_label(m.allocator, m.params_json, m.estimator_json),
                    "constraint_set": m.constraint_set,
                    "order": int(m.cell_index),
                    "n_rebalances": int(m.n_rebalances),
                    "n_failed_rebalances": int(m.n_failed_rebalances),
                    "candidate_weight_median": m.candidate_weight_median,
                    "n_adjustment_breaches": int(m.n_adjustment_breaches),
                }
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["data_variant", "order"]).drop(columns="order")
    cols = ["data_variant", "row", "label", "constraint_set", "config_id", "start", "end",
            "n_periods", "n_rebalances", "n_failed_rebalances", "candidate_weight_median",
            "ann_return", "ann_vol", "cvar95", "cdar95", "max_dd", "te_vs_saa",
            "turnover_ann", "ann_return_gross", "cost_drag_ann", "n_trades",
            "n_liquidity_adjusted", "n_adjustment_breaches"]  # fmt: skip
    return out.reindex(columns=cols).reset_index(drop=True)

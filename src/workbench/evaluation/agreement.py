"""Library agreement: the same configuration solved by Riskfolio-Lib and by skfolio.

Cells are paired when everything but the library matches: family, parameters, estimator,
constraint set, data variant and window end. ``skfolio_hc.max_clusters`` takes part in the key
only when set (None means skfolio's native cluster selection). Agreement is evidence that the
corridor is not an artefact of one implementation; disagreement flags implementation risk.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from workbench.evaluation.corridor import FAMILIES, library_of
from workbench.registry.store import Registry, loads_or_none

LIBRARY_ONLY_PARAMS = {"max_clusters": None}  # dropped from the key when at this default


def _key(row) -> str:
    params = {k: v for k, v in json.loads(row.params_json).items()
              if not (k in LIBRARY_ONLY_PARAMS and v == LIBRARY_ONLY_PARAMS[k])}  # fmt: skip
    return json.dumps(
        {
            "family": FAMILIES.get(row.allocator, "other"),
            "params": params,
            "estimator": loads_or_none(row.estimator_json),
            "constraint_set": row.constraint_set,
        },
        sort_keys=True,
    )


def paired_cells(registry: Registry, experiment_id: str) -> pd.DataFrame:
    """One row per (variant, window end, matched configuration) with both libraries' results.

    Columns: data_variant, window_end, family, label, constraint_set, status_riskfolio,
    status_skfolio, w_riskfolio, w_skfolio (candidate weight, decimal), abs_diff.
    """
    exp = registry.experiment(experiment_id)
    cells = registry.cells(experiment_id)
    cells = cells[cells["allocator"].map(library_of).isin(["riskfolio", "skfolio"])].copy()
    if cells.empty:
        return pd.DataFrame()
    w = registry.weights(experiment_id)
    cand = w.loc[w["asset_id"] == exp["candidate_id"]].set_index("cell_id")["weight"]
    cells["w"] = cells["cell_id"].map(cand)
    cells["library"] = cells["allocator"].map(library_of)
    cells["key"] = [_key(r) for r in cells.itertuples()]
    idx = ["data_variant", "window_end", "key"]
    wide = cells.pivot_table(index=idx, columns="library", values=["status", "w"],
                             aggfunc="first", dropna=False)  # fmt: skip
    if ("status", "riskfolio") not in wide or ("status", "skfolio") not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[("status", "riskfolio"), ("status", "skfolio")])
    out = pd.DataFrame(
        {
            "status_riskfolio": wide[("status", "riskfolio")],
            "status_skfolio": wide[("status", "skfolio")],
            "w_riskfolio": pd.to_numeric(wide.get(("w", "riskfolio")), errors="coerce"),
            "w_skfolio": pd.to_numeric(wide.get(("w", "skfolio")), errors="coerce"),
        }
    ).reset_index()
    keys = out["key"].map(json.loads)
    out["family"] = keys.map(lambda k: k["family"])
    out["constraint_set"] = keys.map(lambda k: k["constraint_set"])
    out["label"] = keys.map(_label)
    out["abs_diff"] = (out["w_riskfolio"] - out["w_skfolio"]).abs()
    cols = [
        "data_variant", "window_end", "family", "label", "constraint_set",
        "status_riskfolio", "status_skfolio", "w_riskfolio", "w_skfolio", "abs_diff",
    ]  # fmt: skip
    return out[cols].sort_values(cols[:4]).reset_index(drop=True)


def agreement_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, family, label): pairs, status mismatches, |Δ candidate weight| stats."""
    if pairs.empty:
        return pd.DataFrame()
    g = pairs.groupby(["data_variant", "family", "label"], sort=True)
    out = g.apply(
        lambda d: pd.Series(
            {
                "n_pairs": len(d),
                "n_status_mismatch": int((d["status_riskfolio"] != d["status_skfolio"]).sum()),
                "median_abs_diff": d["abs_diff"].median(),
                "max_abs_diff": d["abs_diff"].max(),
            }
        ),
        include_groups=False,
    )
    return out.reset_index()


def _label(key: dict) -> str:
    p = " ".join(f"{k}={v}" for k, v in key["params"].items())
    e = key["estimator"]
    est = "" if e is None else f" [{e['method_mu']}/{e['method_cov']}]"
    return f"{key['family']} {p}{est}".strip()


def max_abs_diff(pairs: pd.DataFrame) -> float:
    return float(np.nanmax(pairs["abs_diff"])) if not pairs.empty else float("nan")

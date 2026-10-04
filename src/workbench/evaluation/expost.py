"""Ex-post risk table: the SAA reference row followed by every ok cell."""

from __future__ import annotations

import json

import pandas as pd

from workbench.evaluation.metrics import IN_SAMPLE
from workbench.registry.store import REFERENCE_ALLOCATOR, Registry, loads_or_none

STATS = ("ann_return", "ann_vol", "cvar95", "cdar95", "max_dd", "te_vs_saa")


def cell_label(allocator: str, params_json: str, estimator_json: str | None) -> str:
    """Compact human-readable cell label, e.g. ``riskfolio_mean_risk rm=CVaR obj=Sharpe
    [hist/ledoit]``."""
    params = json.loads(params_json)
    parts = [allocator] + [f"{k}={v}" for k, v in params.items() if k != "funding"]
    e = loads_or_none(estimator_json)
    if e is not None:
        parts.append(f"[{e['method_mu']}/{e['method_cov']}]")
    return " ".join(parts)


def expost_table(registry: Registry, experiment_id: str, basis: str = IN_SAMPLE) -> pd.DataFrame:
    """One row per (data_variant, window_end) SAA reference and per ok cell.

    Columns: data_variant, window_end, row ("SAA" or "cell"), label, constraint_set, cell_id,
    candidate_weight, and ``STATS`` from metrics ``<basis>.<stat>``. Return, vol, TE and drawdowns
    are annual/decimal; cvar95 is per period (decimal loss).
    """
    cand = registry.experiment(experiment_id)["candidate_id"]
    cells = registry.cells(experiment_id, include_reference=True)
    cells = cells[cells["status"] == "ok"].copy()
    m = registry.metrics(experiment_id, include_reference=True)
    m = m[m["metric"].str.startswith(f"{basis}.")].copy()
    m["stat"] = m["metric"].str.removeprefix(f"{basis}.")
    wide = m.pivot(index="cell_id", columns="stat", values="value").reindex(columns=list(STATS))
    w = registry.weights(experiment_id, include_reference=True)
    cw = w.loc[w["asset_id"] == cand].set_index("cell_id")["weight"]

    ref = cells["allocator"] == REFERENCE_ALLOCATOR
    out = pd.DataFrame(
        {
            "data_variant": cells["data_variant"],
            "window_end": cells["window_end"],
            "row": ref.map({True: "SAA", False: "cell"}),
            "label": [
                "SAA" if r else cell_label(a, p, e)
                for r, a, p, e in zip(
                    ref,
                    cells["allocator"],
                    cells["params_json"],
                    cells["estimator_json"],
                    strict=True,
                )
            ],  # fmt: skip
            "constraint_set": cells["constraint_set"],
            "cell_id": cells["cell_id"],
            "order": cells["cell_index"],
        }
    )
    out["candidate_weight"] = out["cell_id"].map(cw)
    out = out.join(wide, on="cell_id")
    out = out.sort_values(["data_variant", "window_end", "order"]).drop(columns="order")
    return out.reset_index(drop=True)

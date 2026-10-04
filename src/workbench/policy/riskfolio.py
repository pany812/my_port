"""Translate a :class:`CompiledPolicy` into Riskfolio-Lib 7.4.0 inputs.

Classic (mean-risk) portfolios get A/B linear constraints (asset bounds, class limits),
the per-asset band via ``allowTO`` and the TE limit via ``allowTE``. Hierarchical-clustering
portfolios only take per-asset bounds; the band is a per-asset box so it is folded into them.
Class limits and TE are not enforceable in HC and are caught by the post-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp

from workbench.policy.compiled import CompiledPolicy

_CLASS_SET = "Class"


def _asset_classes(policy: CompiledPolicy, assets: list[str]) -> pd.DataFrame:
    cls = policy.asset_class.reindex(assets) if policy.asset_class is not None else None
    labels = ["" for _ in assets] if cls is None else list(cls.fillna(""))
    return pd.DataFrame({"Assets": assets, _CLASS_SET: labels})


def linear_constraints(
    policy: CompiledPolicy, assets: list[str]
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """A, B of A w <= B for per-asset bounds and class limits, columns in ``assets`` order."""
    lo, hi = policy.bounds(assets)
    rows: list[dict] = []

    def row(type_, set_, pos, sign, weight):
        rows.append(
            {
                "Disabled": False,
                "Type": type_,
                "Set": set_,
                "Position": pos,
                "Sign": sign,
                "Weight": weight,
                "Type Relative": "",
                "Relative Set": "",
                "Relative": "",
                "Factor": "",
            }
        )

    for a in assets:
        if lo[a] > 0:
            row("Assets", "", a, ">=", float(lo[a]))
        if hi[a] < 1:
            row("Assets", "", a, "<=", float(hi[a]))
    for cls, (clo, chi) in policy.class_limits.items():
        if clo > 0:
            row("Classes", _CLASS_SET, cls, ">=", float(clo))
        if chi < 1:
            row("Classes", _CLASS_SET, cls, "<=", float(chi))
    if not rows:
        return None, None
    table = pd.DataFrame(rows).astype({"Disabled": object})
    A, B = rp.assets_constraints(table, _asset_classes(policy, assets))
    return np.asarray(A, dtype=float), np.asarray(B, dtype=float)


def apply_mean_risk(port: rp.Portfolio, policy: CompiledPolicy, assets: list[str]) -> None:
    """Set constraints, TE, band and solvers on a Classic ``rp.Portfolio`` in place."""
    port.sht = not policy.long_only
    port.upperlng = 1.0
    port.solvers = list(policy.solvers)
    A, B = linear_constraints(policy, assets)
    if A is not None:
        port.ainequality, port.binequality = A, B
    if policy.benchweights is not None:
        port.kindbench = True
        port.benchweights = policy.benchweights.reindex(assets).to_frame("weights")
    if policy.te is not None:
        port.allowTE = True
        port.TE = policy.te
    if policy.band is not None:
        port.allowTO = True
        port.turnover = policy.band


def hc_bounds(policy: CompiledPolicy, assets: list[str]) -> tuple[pd.Series, pd.Series]:
    """(w_max, w_min) for ``rp.HCPortfolio``: asset bounds intersected with the band box.

    Built through ``rp.hrp_constraints`` with an object-dtype ``Disabled`` column; a numpy-bool
    column makes Riskfolio-Lib 7.4.0 silently ignore every row.
    """
    lo, hi = policy.bounds(assets)
    if policy.band is not None:
        bench = policy.benchweights.reindex(assets)
        lo = np.maximum(lo, bench - policy.band)
        hi = np.minimum(hi, bench + policy.band)
    lo, hi = lo.clip(lower=0.0), hi.clip(upper=1.0)
    rows = []
    for a in assets:
        rows.append({"Disabled": False, "Type": "Assets", "Set": "", "Position": a,
                     "Sign": "<=", "Weight": float(hi[a])})  # fmt: skip
        rows.append({"Disabled": False, "Type": "Assets", "Set": "", "Position": a,
                     "Sign": ">=", "Weight": float(lo[a])})  # fmt: skip
    table = pd.DataFrame(rows).astype({"Disabled": object})
    w_max, w_min = rp.hrp_constraints(table, _asset_classes(policy, assets))
    return w_max.reindex(assets).astype(float), w_min.reindex(assets).astype(float)


def hc_bounds_problem(w_max: pd.Series, w_min: pd.Series) -> str | None:
    """Why HC bounds are infeasible (Riskfolio would raise NameError), or None if fine."""
    if (w_min > w_max + 1e-12).any():
        bad = list(w_min.index[w_min > w_max + 1e-12])
        return f"lower bound above upper bound for {bad}"
    if w_max.sum() < 1.0 - 1e-12:
        return f"upper bounds sum to {w_max.sum():.4f} < 1"
    if w_min.sum() > 1.0 + 1e-12:
        return f"lower bounds sum to {w_min.sum():.4f} > 1"
    return None

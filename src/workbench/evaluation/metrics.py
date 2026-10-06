"""Per-cell metrics computed by the runner and stored in the ``metrics`` table.

- ``candidate_risk_share`` per lens: Riskfolio-Lib Euler risk contributions on the fitting
  window (additive: shares over all assets sum to 1). Lens definitions are Riskfolio-Lib's.
- ``in_sample.<stat>``: ``workbench.evaluation.stats.summary`` of fixed weights rebalanced
  every period over the fitting window. In-sample: these flatter optimisers. M5 adds ``oos.*``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import riskfolio as rp

from workbench.evaluation.stats import ALPHA, summary

IN_SAMPLE = "in_sample"
RISK_SHARE = "candidate_risk_share"


def risk_shares(w: pd.Series, window: pd.DataFrame, rm: str, rf: float = 0.0) -> pd.Series:
    """Share of total risk per asset under lens ``rm`` (sums to 1). Riskfolio-Lib definitions."""
    cols = list(window.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = np.ravel(
            rp.Risk_Contribution(
                w[cols].to_frame("weights"), window, cov=window.cov(), rm=rm, rf=rf, alpha=ALPHA
            )
        )
    total = rc.sum()
    if not np.isfinite(total) or abs(total) < 1e-14:
        raise ValueError(f"total {rm} risk is {total}; shares undefined")
    return pd.Series(rc / total, index=cols)


def cell_metrics(
    w: pd.Series,
    window: pd.DataFrame,
    saa: pd.Series,
    candidate: str,
    lenses: tuple[str, ...],
    freq: str,
    rf: float = 0.0,
) -> tuple[list[tuple[str, str, float]], dict[str, str]]:
    """Metrics for one ok cell. Returns (metric triples, errors by metric name).

    w, saa:  weights indexed by asset id (decimal).
    window:  the fitting window (simple returns per period of ``freq``).
    rf:      per-period target return for lower-partial-moment lenses (FLPM, SLPM).
    """
    out: list[tuple[str, str, float]] = []
    errors: dict[str, str] = {}
    for lens in lenses:
        try:
            out.append((RISK_SHARE, lens, float(risk_shares(w, window, lens, rf)[candidate])))
        except Exception as e:
            errors[f"{RISK_SHARE}:{lens}"] = f"{type(e).__name__}: {e}"
    try:
        cols = list(window.columns)
        port = window @ w[cols]
        bench = window @ saa[cols]
        for name, value in summary(port, bench, freq).items():
            out.append((f"{IN_SAMPLE}.{name}", "", float(value)))
    except Exception as e:
        errors[IN_SAMPLE] = f"{type(e).__name__}: {e}"
    return out, errors

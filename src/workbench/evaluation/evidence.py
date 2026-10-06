"""Experiment-level evidence: apply ``inference`` to an experiment's paths and returns.

Disclosures, not gates (Phase 2 decision): nothing here changes a cell's status.

Rows (one per subject and test, per data variant):
- subject = config_id: ``sharpe_diff`` (Ledoit–Wolf vs the SAA path; extra holds the
  BH-adjusted p-value) and ``dsr`` (deflated Sharpe of the information ratio vs the SAA,
  deflated for every configuration in the variant).
- subject = "grid": ``pbo`` (CSCV over the configuration x period active-return matrix).
- subject = "candidate": with a cash asset in the SAA, ``spanning_alpha`` and
  ``spanning_alpha_robust`` (excess returns over cash); otherwise ``spanning_hk``,
  ``spanning_kz_f1``, ``spanning_kz_f2`` and ``spanning_hk_robust``.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from workbench.evaluation.inference import (
    alpha_tests,
    benjamini_hochberg,
    deflated_sharpe,
    pbo_cscv,
    sharpe_diff_test,
    spanning_tests,
)
from workbench.units import return_period_to_annual, sharpe_period_to_annual

N_BOOT = 4999
PBO_SPLITS = 10
MIN_OBS = 24  # below this the path tests are not computed (a DSR of 1.00 on 9 months misleads)
GRID = "grid"
CANDIDATE = "candidate"


def boot_seed(spec_seed: int, config_id: str) -> int:
    """Deterministic per-configuration bootstrap seed."""
    return int(hashlib.sha256(f"{spec_seed}:{config_id}".encode()).hexdigest()[:8], 16)


def evidence_row(subject: str, test: str, statistic, p_value, extra: dict) -> dict:
    def clean(v):
        return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v

    return {
        "subject": subject,
        "test": test,
        "statistic": clean(None if statistic is None else float(statistic)),
        "p_value": clean(None if p_value is None else float(p_value)),
        "extra_json": json.dumps({k: clean(v) for k, v in extra.items()}, sort_keys=True),
    }


def path_evidence(
    paths: dict[str, pd.Series], saa_path: pd.Series, freq: str, spec_seed: int,
    riskless: pd.Series | float = 0.0, n_boot: int = N_BOOT, pbo_splits: int = PBO_SPLITS,
) -> list[dict]:  # fmt: skip
    """Sharpe-difference tests, deflated Sharpe ratios and PBO for walk-forward paths.

    paths:    config_id -> OOS portfolio returns (per period), same dates as ``saa_path``.
    saa_path: the SAA reference path.
    riskless: per-period riskless return (the SAA's cash asset over the same dates, or the
              policy rf). Sharpe ratios use returns in excess of it; the information ratio
              (DSR, PBO) is unaffected because it cancels in the active return.
    """
    rows: list[dict] = []
    if not paths:
        return rows
    ids = list(paths)
    b = saa_path.to_numpy(dtype=float)
    if len(b) < MIN_OBS:
        note = f"too few observations ({len(b)} < {MIN_OBS})"
        for cid in ids:
            extra = {"n_obs": len(b), "note": note}
            rows.append(evidence_row(cid, "sharpe_diff", None, None, extra))
            rows.append(evidence_row(cid, "dsr", None, None, {**extra, "n_trials": len(ids)}))
        rows.append(evidence_row(GRID, "pbo", None, None, {"n_obs": len(b), "note": note}))
        return rows
    rf = (riskless.reindex(saa_path.index).to_numpy(dtype=float)
          if isinstance(riskless, pd.Series) else np.full(len(b), float(riskless)))  # fmt: skip
    tests = {}
    for cid in ids:
        a = paths[cid].reindex(saa_path.index).to_numpy(dtype=float)
        tests[cid] = sharpe_diff_test(a - rf, b - rf, n_boot=n_boot,
                                      seed=boot_seed(spec_seed, cid))  # fmt: skip
    adj = benjamini_hochberg(np.array([tests[c].p_value for c in ids]))
    for cid, p_bh in zip(ids, adj, strict=True):
        r = tests[cid]
        rows.append(evidence_row(cid, "sharpe_diff", r.diff, r.p_value, {
            "sr": r.sr_a, "sr_saa": r.sr_b,
            "sr_ann": sharpe_period_to_annual(r.sr_a, freq),
            "sr_saa_ann": sharpe_period_to_annual(r.sr_b, freq),
            "diff_ann": sharpe_period_to_annual(r.diff, freq),
            "se": r.se, "p_bh": float(p_bh), "block": r.block, "n_boot": r.n_boot,
            "n_obs": len(b), "note": r.note,
            "excess_over": riskless.name if isinstance(riskless, pd.Series) else "rf",
        }))  # fmt: skip

    active = {cid: paths[cid].reindex(saa_path.index).to_numpy(dtype=float) - b for cid in ids}
    for cid, d in deflated_sharpe(active).items():
        rows.append(evidence_row(cid, "dsr", d.dsr, None, {
            "ir": d.sr, "ir_ann": sharpe_period_to_annual(d.sr, freq), "psr0": d.psr0,
            "sr_star": d.sr_star, "sr_star_ann": sharpe_period_to_annual(d.sr_star, freq),
            "n_trials": d.n_trials, "skew": d.skew, "kurt": d.kurt, "note": d.note,
        }))  # fmt: skip

    matrix = np.column_stack([active[c] for c in ids])
    if len(ids) >= 2 and len(b) >= pbo_splits:
        p = pbo_cscv(matrix, n_splits=pbo_splits)
        rows.append(evidence_row(GRID, "pbo", p.pbo, None, {
            "n_splits": p.n_splits, "n_combinations": p.n_combinations, "n_trials": p.n_trials,
            "median_logit": float(np.median(p.logits)), "n_obs": len(b),
        }))  # fmt: skip
    return rows


def spanning_evidence(
    returns: pd.DataFrame, candidate: str, freq: str, riskless: str | None = None
) -> list[dict]:
    """Spanning tests of the candidate against the other columns (the SAA building blocks).

    riskless: the SAA's cash asset, if exactly one. Then the test runs in excess returns over
    it and reduces to H0: alpha = 0 (``spanning_alpha`` / ``spanning_alpha_robust``); the
    Huberman–Kandel and Kan–Zhou F2 legs are not applicable. Without one: HK, KZ F1/F2 and the
    HC3-robust HK.
    """
    blocks = [c for c in returns.columns if c != candidate]
    if len(returns) <= len(blocks) + 2:
        return []
    out = []
    if riskless is not None:
        rf = returns[riskless].to_numpy(float)
        risky = [c for c in blocks if c != riskless]
        a = alpha_tests(returns[candidate].to_numpy(float) - rf,
                        returns[risky].to_numpy(float) - rf[:, None])  # fmt: skip
        common = {"alpha": a.alpha,
                  "alpha_ann": return_period_to_annual(a.alpha, freq, "arithmetic"),
                  "riskless": riskless, "n_obs": a.n_obs, "n_assets": a.n_assets}  # fmt: skip
        for test, f in (("spanning_alpha", a.exact), ("spanning_alpha_robust", a.robust)):
            out.append(evidence_row(CANDIDATE, test, f.statistic, f.p_value,
                            {**common, "df1": f.df1, "df2": f.df2}))  # fmt: skip
        return out
    s = spanning_tests(returns[candidate].to_numpy(float), returns[blocks].to_numpy(float))
    common = {
        "alpha": s.alpha,
        "alpha_ann": return_period_to_annual(s.alpha, freq, "arithmetic"),
        "beta_sum": s.beta_sum,
        "n_obs": s.n_obs,
        "n_assets": s.n_assets,
    }
    for test, f in (("spanning_hk", s.hk), ("spanning_kz_f1", s.kz_f1),
                    ("spanning_kz_f2", s.kz_f2), ("spanning_hk_robust", s.hk_robust)):  # fmt: skip
        out.append(evidence_row(CANDIDATE, test, f.statistic, f.p_value,
                        {**common, "df1": f.df1, "df2": f.df2}))  # fmt: skip
    return out

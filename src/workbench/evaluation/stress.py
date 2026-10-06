"""Stress (P2-M5): named crisis windows and stationary block-bootstrap paths.

Disclosures, like the evidence section: nothing here changes a cell's status.

Policy portfolios: the SAA, and the SAA plus the candidate at weight x funded per the spec's
``funding`` (``allocators.naive.funded_weights``), fixed weights rebalanced every period, no
costs. The weights stressed are the corridor's P25 / median / P75 at the latest date of the
``full`` variant plus any listed in the spec.

- Crisis windows: named, inclusive date ranges. Policy portfolios are evaluated on the
  variant's history at run time (stored as ``evidence`` rows); the stored walk-forward paths are
  evaluated at report time (:func:`path_window_view`). Inside a window, wealth starts at 1 at the
  window start, so ``max_dd`` is the worst fall from any level reached inside the window,
  including the start.
- Bootstrap: stationary block bootstrap (Politis & Romano 1994) of whole return rows, so the
  cross-section (dependence between assets) is kept and blocks keep time structure such as
  drawdown persistence. Mean block length ceil(T^(1/3)) as in the Sharpe test, unless set. The
  same paths are used for every weight, so comparisons with the SAA are paired.

Statistics use the ex-post definitions (``evaluation.stats``): compounded wealth drawdowns,
CDaR 95% as the mean of the worst 5% of drawdowns, CVaR 95% per period.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from workbench.allocators._solve import Infeasible
from workbench.allocators.naive import funded_weights
from workbench.evaluation.evidence import boot_seed
from workbench.evaluation.inference import block_length
from workbench.evaluation.stats import ALPHA, max_drawdown
from workbench.units import periods_per_year, vol_period_to_annual

PRESET_WINDOWS: dict[str, tuple[str, str]] = {
    "gfc": ("2007-11", "2009-02"),
    "covid": ("2020-02", "2020-03"),
    "rates_2022": ("2022-01", "2022-09"),
}
WINDOW_TEST = "stress_window:"
BOOTSTRAP_TEST = "stress_bootstrap"
WEIGHT_DECIMALS = 4


@dataclass(frozen=True)
class StressWeight:
    """A candidate weight to stress and why: "saa" (0), "corridor_p25/median/p75", "spec"."""

    x: float
    roles: tuple[str, ...]

    @property
    def subject(self) -> str:
        return f"x={self.x:.{WEIGHT_DECIMALS}f}"


# --- weights --------------------------------------------------------------------------------


def stress_weights(
    corridor_weights: dict[str, float], spec_weights: tuple[float, ...]
) -> list[StressWeight]:
    """The SAA (x = 0), corridor quantiles and spec weights; equal values merged, sorted by x."""
    roles: dict[float, list[str]] = {0.0: ["saa"]}
    for role, x in corridor_weights.items():
        roles.setdefault(round(float(x), WEIGHT_DECIMALS), []).append(role)
    for x in spec_weights:
        roles.setdefault(round(float(x), WEIGHT_DECIMALS), []).append("spec")
    return [StressWeight(x, tuple(r)) for x, r in sorted(roles.items())]


def corridor_quantiles(records, candidate: str, variant: str = "full") -> dict[str, float]:
    """Corridor P25 / median / P75 of the candidate's weight over ok grid cells at the latest
    rebalance date of ``variant`` ({} when no cell is ok). ``records``: the runner's cells."""
    grid = [r for r in records if r.data_variant == variant and r.cell_index >= 0]
    if not grid:
        return {}
    last = max(r.window_end for r in grid)
    w = [r.weights[candidate] for r in grid
         if r.window_end == last and r.status == "ok" and r.weights]  # fmt: skip
    if not w:
        return {}
    q = np.quantile(np.asarray(w, dtype=float), [0.25, 0.5, 0.75])
    return {"corridor_p25": float(q[0]), "corridor_median": float(q[1]),
            "corridor_p75": float(q[2])}  # fmt: skip


def policy_portfolios(
    saa: pd.Series, candidate: str, weights: list[StressWeight], funding: str,
    asset_class: pd.Series | None,
) -> tuple[dict[str, pd.Series], dict[str, str]]:  # fmt: skip
    """subject -> weights, and subject -> reason for weights that cannot be funded."""
    out, failed = {}, {}
    for sw in weights:
        try:
            out[sw.subject] = funded_weights(saa.copy(), candidate, sw.x, funding, asset_class)
        except (Infeasible, ValueError) as e:
            failed[sw.subject] = str(e)
    return out, failed


# --- path statistics (vectorised over paths; rows = paths, columns = periods) ----------------


def path_stats(r: np.ndarray, freq: str, alpha: float = ALPHA) -> dict[str, np.ndarray]:
    """Per-path statistics of simple returns ``r`` (n_paths, T); ``evaluation.stats`` definitions.

    max_dd, cdar95 (compounded drawdowns), cvar95 (per-period loss), ann_return (geometric),
    ann_vol (std ddof=1 x sqrt(n)). All decimal, losses positive.
    """
    n, t = r.shape
    wealth = np.cumprod(1.0 + r, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n, 1)), wealth], axis=1), axis=1)[:, 1:]
    dd = 1.0 - wealth / peak
    k = max(1, math.ceil(alpha * t))
    return {
        "max_dd": dd.max(axis=1),
        "cdar95": -np.sort(-dd, axis=1)[:, :k].mean(axis=1),
        "cvar95": -np.sort(r, axis=1)[:, :k].mean(axis=1),
        "ann_return": wealth[:, -1] ** (periods_per_year(freq) / t) - 1.0,
        "ann_vol": np.array([vol_period_to_annual(float(s), freq)
                             for s in r.std(axis=1, ddof=1)]),
    }  # fmt: skip


def stationary_bootstrap(
    n_obs: int, n_paths: int, horizon: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Row indices (n_paths, horizon) of a stationary bootstrap (Politis & Romano 1994).

    Each step starts a new block at a uniform random row with probability 1 / mean_block,
    otherwise continues with the next row (circular). Block lengths are geometric with mean
    ``mean_block``; ``mean_block = 1`` is i.i.d. resampling of rows.
    """
    if mean_block < 1:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    p = 1.0 / mean_block
    starts = rng.integers(0, n_obs, size=(n_paths, horizon))
    new = rng.random((n_paths, horizon)) < p
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, horizon):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n_obs)
    return idx


# --- run-time evidence rows -----------------------------------------------------------------


def window_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive bounds: "YYYY-MM" ends at the end of that month; ISO dates are taken as is."""
    lo = pd.Timestamp(start)
    hi = pd.Period(end, freq="M").end_time if len(end) == 7 else pd.Timestamp(end)
    return lo, hi


def _row(subject: str, test: str, statistic, extra: dict) -> dict:
    def clean(v):
        if isinstance(v, float | np.floating):
            return None if not np.isfinite(v) else float(v)
        return v

    import json

    return {"subject": subject, "test": test, "statistic": clean(statistic), "p_value": None,
            "extra_json": json.dumps({k: clean(v) for k, v in extra.items()})}  # fmt: skip


def window_rows(
    returns: pd.DataFrame, backfilled: pd.Series, candidate: str,
    portfolios: dict[str, pd.Series], weights: list[StressWeight],
    windows: tuple[tuple[str, str, str], ...],
) -> list[dict]:  # fmt: skip
    """One row per (window, weight): window return and max drawdown of the policy portfolio,
    their changes vs the SAA, and the candidate's own return over the window."""
    rows = []
    roles = {sw.subject: list(sw.roles) for sw in weights}
    saa_subject = weights[0].subject
    for name, start, end in windows:
        lo, hi = window_bounds(start, end)
        r = returns.loc[(returns.index >= lo) & (returns.index <= hi)]
        base = {"window": name, "start": start, "end": end, "n_periods": len(r)}
        if r.empty:
            for subject in portfolios:
                rows.append(
                    _row(
                        subject,
                        WINDOW_TEST + name,
                        None,
                        {**base, "roles": roles[subject], "note": "window not covered by the data"},
                    )
                )
            continue  # fmt: skip
        first, last = r.index[0].date().isoformat(), r.index[-1].date().isoformat()
        bf = float(backfilled.reindex(r.index).fillna(False).mean())
        cand = float((1.0 + r[candidate]).prod() - 1.0)
        saa_r = r @ portfolios[saa_subject].reindex(r.columns)
        saa_ret, saa_dd = float((1.0 + saa_r).prod() - 1.0), max_drawdown(saa_r)
        for subject, w in portfolios.items():
            pr = r @ w.reindex(r.columns)
            ret, dd = float((1.0 + pr).prod() - 1.0), max_drawdown(pr)
            rows.append(_row(subject, WINDOW_TEST + name, ret, {
                **base, "first": first, "last": last, "roles": roles[subject],
                "x": float(w[candidate]), "return": ret, "max_dd": dd,
                "d_return": ret - saa_ret, "d_max_dd": dd - saa_dd,
                "candidate_return": cand, "backfilled_share": bf}))  # fmt: skip
    return rows


def bootstrap_rows(
    returns: pd.DataFrame, backfilled: pd.Series, candidate: str,
    portfolios: dict[str, pd.Series], weights: list[StressWeight], freq: str, n_paths: int,
    horizon_years: int, block: int | None, seed: int,
) -> list[dict]:  # fmt: skip
    """One row per weight: distribution over bootstrap paths, paired with the SAA."""
    horizon = horizon_years * periods_per_year(freq)
    n_obs = len(returns)
    mean_block = block or block_length(n_obs)
    idx = stationary_bootstrap(n_obs, n_paths, horizon, mean_block, np.random.default_rng(seed))
    sims = returns.to_numpy(dtype=float)[idx]  # (n_paths, horizon, n_assets)
    roles = {sw.subject: list(sw.roles) for sw in weights}
    stats = {s: path_stats(sims @ w.reindex(returns.columns).to_numpy(dtype=float), freq)
             for s, w in portfolios.items()}  # fmt: skip
    saa = stats[weights[0].subject]
    common = {"n_paths": n_paths, "horizon_years": horizon_years, "mean_block": mean_block,
              "n_obs": n_obs, "backfilled_share": float(backfilled.mean())}  # fmt: skip
    rows = []
    for subject, st in stats.items():
        d = st["max_dd"] - saa["max_dd"]
        q = np.quantile
        rows.append(_row(subject, BOOTSTRAP_TEST, float(np.median(st["max_dd"])), {
            **common, "roles": roles[subject], "x": float(portfolios[subject][candidate]),
            "max_dd_median": float(np.median(st["max_dd"])),
            "max_dd_p95": float(q(st["max_dd"], 0.95)),
            "cdar95_median": float(np.median(st["cdar95"])),
            "cvar95_median": float(np.median(st["cvar95"])),
            "ann_return_median": float(np.median(st["ann_return"])),
            "ann_return_p05": float(q(st["ann_return"], 0.05)),
            "ann_vol_median": float(np.median(st["ann_vol"])),
            "d_max_dd_median": float(np.median(d)), "d_max_dd_p95": float(q(d, 0.95)),
            "p_worse_max_dd": float((d > 1e-12).mean()),
        }))  # fmt: skip
    return rows


def stress_evidence(
    returns: pd.DataFrame, backfilled: pd.Series, saa: pd.Series, candidate: str,
    asset_class: pd.Series | None, funding: str, weights: list[StressWeight],
    windows: tuple[tuple[str, str, str], ...], freq: str, n_paths: int, horizon_years: int,
    block: int | None, spec_seed: int, variant: str,
) -> list[dict]:  # fmt: skip
    """All stress rows for one data variant (``evidence`` rows without ``data_variant``)."""
    portfolios, failed = policy_portfolios(saa, candidate, weights, funding, asset_class)
    rows = [_row(s, BOOTSTRAP_TEST, None, {"note": why}) for s, why in failed.items()]
    if weights[0].subject not in portfolios:  # the SAA itself always funds
        raise ValueError("the SAA policy portfolio could not be built")
    rows += window_rows(returns, backfilled, candidate, portfolios, weights, windows)
    rows += bootstrap_rows(
        returns,
        backfilled,
        candidate,
        portfolios,
        weights,
        freq,
        n_paths,
        horizon_years,
        block,
        boot_seed(spec_seed, f"stress:{variant}"),
    )
    return rows  # fmt: skip


# --- report-time tables ---------------------------------------------------------------------


def _stress_frame(ev: pd.DataFrame, test_prefix: str) -> pd.DataFrame:
    rows = ev[ev["test"].str.startswith(test_prefix)] if not ev.empty else ev
    if rows.empty:
        return pd.DataFrame()
    extra = pd.DataFrame([json.loads(x) for x in rows["extra_json"]], index=rows.index)
    out = pd.concat([rows[["data_variant", "subject"]], extra], axis=1)
    out["role"] = out["roles"].map(lambda r: ", ".join(r) if isinstance(r, list) else "")
    if "x" not in out:
        out["x"] = np.nan
    out["x"] = out["x"].fillna(out["subject"].str.removeprefix("x=").astype(float))
    return out


def window_table(ev: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Crisis windows on policy portfolios, one row per (variant, window, weight), windows in
    ``order`` (the spec's) when given.

    Columns: data_variant, window, first, last, n_periods, backfilled_share, x, role, return,
    d_return, max_dd, d_max_dd (changes vs the SAA), candidate_return, note. Decimal.
    """
    t = _stress_frame(ev, WINDOW_TEST)
    if t.empty:
        return t
    for c in ("first", "last", "note", "return", "d_return", "max_dd", "d_max_dd",
              "candidate_return", "backfilled_share"):  # fmt: skip
        if c not in t:
            t[c] = np.nan if c not in ("first", "last", "note") else ""
    t["note"] = t["note"].fillna("")
    cols = [
        "data_variant",
        "window",
        "first",
        "last",
        "n_periods",
        "backfilled_share",
        "x",
        "role",
        "return",
        "d_return",
        "max_dd",
        "d_max_dd",
        "candidate_return",
        "note",
    ]
    rank = {name: i for i, name in enumerate(dict.fromkeys([*(order or []), *t["window"]]))}
    t = t.assign(_w=t["window"].map(rank)).sort_values(["data_variant", "_w", "x"])
    return t[cols].reset_index(drop=True)  # fmt: skip


def bootstrap_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap distribution per (variant, weight). Columns: data_variant, x, role,
    max_dd_median, max_dd_p95, cdar95_median, cvar95_median, ann_return_median, ann_return_p05,
    ann_vol_median, d_max_dd_median, d_max_dd_p95, p_worse_max_dd, n_paths, horizon_years,
    mean_block, n_obs, backfilled_share, note."""
    t = _stress_frame(ev, BOOTSTRAP_TEST)
    if t.empty:
        return t
    cols = ["data_variant", "x", "role", "max_dd_median", "max_dd_p95", "cdar95_median",
            "cvar95_median", "ann_return_median", "ann_return_p05", "ann_vol_median",
            "d_max_dd_median", "d_max_dd_p95", "p_worse_max_dd", "n_paths", "horizon_years",
            "mean_block", "n_obs", "backfilled_share", "note"]  # fmt: skip
    for c in cols:
        if c not in t:
            t[c] = "" if c == "note" else np.nan
    t["note"] = t["note"].fillna("")
    return t.sort_values(["data_variant", "x"])[cols].reset_index(drop=True)


def _spread(name: str, x: pd.Series) -> dict[str, float]:
    return {f"{name}_median": float(x.median()), f"{name}_p10": float(x.quantile(0.1)),
            f"{name}_p90": float(x.quantile(0.9))}  # fmt: skip


def path_window_view(
    registry, experiment_id: str, windows: tuple[tuple[str, str, str], ...]
) -> pd.DataFrame:
    """Crisis windows on the stored walk-forward paths (net returns), per (variant, window).

    Columns: data_variant, window, first, last, n_periods, n_configs, saa_return, saa_max_dd,
    d_return_median / _p10 / _p90 and d_max_dd_median / _p10 / _p90 (configuration minus the SAA
    path over the same dates), note ("partial" when the window extends past the out-of-sample
    period; out-of-sample period named when the window misses it).
    """
    from workbench.registry.store import REFERENCE_ALLOCATOR

    paths = registry.oos_returns(experiment_id, include_reference=True)
    if paths.empty:
        return pd.DataFrame()
    paths = paths.assign(date=pd.to_datetime(paths["date"]))
    rows = []
    for variant, g in paths.groupby("data_variant", sort=True):
        wide = g.pivot(index="date", columns="config_id", values="portfolio_return_net")
        ref = wide.pop(REFERENCE_ALLOCATOR)
        first, last = wide.index.min(), wide.index.max()
        for name, start, end in windows:
            lo, hi = window_bounds(start, end)
            sel = (wide.index >= lo) & (wide.index <= hi)
            base = {"data_variant": variant, "window": name}
            if not sel.any():
                rows.append(
                    {
                        **base,
                        "n_periods": 0,
                        "n_configs": 0,
                        "note": f"outside the out-of-sample period ({first.date()}..{last.date()})",
                    }
                )
                continue  # fmt: skip
            r, b = wide.loc[sel], ref.loc[sel]
            saa_ret, saa_dd = float((1.0 + b).prod() - 1.0), max_drawdown(b)
            d_ret = (1.0 + r).prod() - 1.0 - saa_ret
            d_dd = r.apply(max_drawdown) - saa_dd
            partial = lo < first or hi > last
            rows.append({
                **base, "first": r.index[0].date().isoformat(),
                "last": r.index[-1].date().isoformat(), "n_periods": int(sel.sum()),
                "n_configs": r.shape[1], "saa_return": saa_ret, "saa_max_dd": saa_dd,
                **_spread("d_return", d_ret), **_spread("d_max_dd", d_dd),
                "note": "partial: the window extends past the out-of-sample period" if partial
                else "",
            })  # fmt: skip
    return pd.DataFrame(rows)

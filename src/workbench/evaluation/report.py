"""Experiment report: CSV tables plus a markdown summary, built from the registry alone."""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from workbench.evaluation.agreement import agreement_summary, paired_cells
from workbench.evaluation.corridor import FAILED_STATUSES, corridor
from workbench.evaluation.expost import cell_label, expost_table
from workbench.evaluation.markdown import md_table, num, pct
from workbench.evaluation.oos import oos_table
from workbench.evaluation.views import risk_budget_view, sweep_view
from workbench.grid.spec import ExperimentSpec, parse_spec
from workbench.registry.store import Registry

MIN_OOS_PERIODS = 36
STAT_COLS = ["median", "p25", "p75", "p10", "p90", "share_below_0.25pct"]
GROUP_VIEWS = {
    "allocator family": "family",
    "library": "library",
    "risk measure": "rm",
    "estimator": "estimator",
    "constraint set": "constraint_set",
}
MEASURE_LABELS = {"capital_weight": "capital weight"}


@dataclass
class Report:
    experiment_id: str
    name: str
    corridor: pd.DataFrame
    expost: pd.DataFrame
    oos: pd.DataFrame
    summary_md: str

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Write corridor.csv, expost.csv, oos.csv and summary.md into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "corridor": out / "corridor.csv",
            "expost": out / "expost.csv",
            "oos": out / "oos.csv",
            "summary": out / "summary.md",
        }
        self.corridor.to_csv(paths["corridor"], index=False)
        self.expost.to_csv(paths["expost"], index=False)
        self.oos.to_csv(paths["oos"], index=False)
        paths["summary"].write_text(self.summary_md)
        return paths


def latest(corr: pd.DataFrame) -> pd.DataFrame:
    """Corridor rows at the latest window end of each data variant."""
    last = corr.groupby("data_variant")["window_end"].transform("max")
    return corr[corr["window_end"] == last].reset_index(drop=True)


def through_time(corr: pd.DataFrame) -> pd.DataFrame:
    """Per variant and measure: how the per-date corridor moved across rebalance dates."""
    g = corr.dropna(subset=["median"]).groupby(["data_variant", "measure"], sort=True)
    out = g.agg(
        n_dates=("window_end", "nunique"),
        median_of_medians=("median", "median"),
        min_median=("median", "min"),
        max_median=("median", "max"),
        typical_p25=("p25", "median"),
        typical_p75=("p75", "median"),
        typical_share_below=("share_below_0.25pct", "median"),
    )
    return out.reset_index()


def build_report(
    registry: Registry, experiment_id: str, min_oos_periods: int = MIN_OOS_PERIODS
) -> Report:
    exp = registry.experiment(experiment_id)
    spec = parse_spec(exp["spec_yaml"])
    corr = corridor(registry, experiment_id)
    expost = expost_table(registry, experiment_id)
    oos = oos_table(registry, experiment_id)
    md = _summary(registry, exp, spec, corr, expost, oos, min_oos_periods)
    return Report(experiment_id, exp["name"], corr, expost, oos, md)


def headline_text(corr: pd.DataFrame) -> str:
    """Plain-text corridor headline for the terminal (latest date per variant)."""
    t = latest(corr)[["data_variant", "window_end", "measure", "n_cells", "n_ok",
                      *[f"n_{s}" for s in FAILED_STATUSES], *STAT_COLS]]  # fmt: skip
    return t.to_string(index=False, float_format=lambda v: f"{v:.4f}")


# --- markdown --------------------------------------------------------------------------

_P = pct(1)
_P2 = pct(2)
_CORR_FMT = {c: _P2 for c in ("median", "p25", "p75", "p10", "p90")} | {
    "share_below_0.25pct": pct(0)
}


def _summary(registry, exp, spec: ExperimentSpec, corr, expost, oos, min_oos) -> str:
    cells = registry.cells(exp["experiment_id"])
    variants = sorted(cells["data_variant"].unique())
    lines = [f"# {exp['name']}", ""]
    lines += _provenance(exp, spec)
    lines += ["## Cells", "", _status_table(cells), ""]
    lines += _corridor_section(corr, spec)
    lines += _group_views(registry, exp["experiment_id"])
    lines += _failures(cells)
    lines += _agreement_section(registry, exp["experiment_id"])
    lines += _risk_budget_section(registry, exp["experiment_id"])
    lines += _sweep_section(registry, exp["experiment_id"])
    lines += _oos_section(oos, spec, min_oos)
    lines += _evidence_section(registry, exp["experiment_id"], min_oos)
    lines += _expost_section(expost)
    lines += _live_only_note(variants, corr)
    lines += _definitions(spec)
    return "\n".join(lines).rstrip() + "\n"


def _provenance(exp, spec: ExperimentSpec) -> list[str]:
    d = spec.data
    window = (f"rolling {spec.window.periods}" if spec.window.kind == "rolling"
              else f"expanding (min {spec.window.min_periods})")  # fmt: skip
    rows = pd.DataFrame(
        [
            ("experiment_id", exp["experiment_id"]),
            ("spec_hash", exp["spec_hash"]),
            ("data_vintage", exp["data_vintage"]),
            ("riskfolio-lib", exp["riskfolio_version"]),
            ("seed", exp["seed"]),
            ("created", f"{exp['created_at']:%Y-%m-%d %H:%M}" if exp["created_at"] else ""),
            ("data", f"{d.source}, {d.frequency}, {d.start} .. {d.end}"),
            ("candidate", exp["candidate_id"]),
            ("currency / hedging", f"{d.base_currency} / {d.hedging}"),
            ("SAA version", exp["saa_version"]),
            ("mode", spec.backtest.mode),
            (
                "window / rebalance",
                f"{window} / every {spec.rebalance.every}"
                + (
                    f", threshold {spec.rebalance.band:.1%}"
                    if spec.rebalance.kind == "threshold"
                    else ""
                ),
            ),  # fmt: skip
            ("risk lenses", ", ".join(spec.risk_lenses)),
        ],
        columns=["field", "value"],
    )
    return ["## Provenance", "", md_table(rows), ""]


def _status_table(cells: pd.DataFrame) -> str:
    t = cells.groupby("data_variant")["status"].value_counts().unstack(fill_value=0)
    for s in ("ok", *FAILED_STATUSES):
        if s not in t:
            t[s] = 0
    t = t[["ok", *FAILED_STATUSES]]
    t.insert(0, "n_cells", t.sum(axis=1))
    t.insert(1, "n_rebalance_dates", cells.groupby("data_variant")["window_end"].nunique())
    return md_table(t.reset_index())


def _corridor_section(corr: pd.DataFrame, spec: ExperimentSpec) -> list[str]:
    lines = ["## Allocation corridor", ""]
    lines += [
        "Distribution over `ok` cells of the candidate's capital weight and risk share per lens. "
        "Failed cells are counted, never dropped. A single optimal weight is not the answer.",
        "",
        "### Latest rebalance date",
        "",
    ]
    t = latest(corr)
    t = t.assign(measure=t["measure"].map(lambda m: MEASURE_LABELS.get(m, m)))
    cols = ["data_variant", "window_end", "measure", "n_cells", "n_ok",
            *[f"n_{s}" for s in FAILED_STATUSES], *STAT_COLS]  # fmt: skip
    lines += [md_table(t[cols], _CORR_FMT), ""]
    if corr["window_end"].nunique() > 1:
        tt = through_time(corr)
        tt = tt.assign(measure=tt["measure"].map(lambda m: MEASURE_LABELS.get(m, m)))
        fmt = {c: _P2 for c in ("median_of_medians", "min_median", "max_median",
                                "typical_p25", "typical_p75")}  # fmt: skip
        fmt["typical_share_below"] = pct(0)
        lines += ["### Through time (walk-forward)", "",
                  "How each date's corridor moved across rebalance dates.", "",
                  md_table(tt, fmt), ""]  # fmt: skip
    return lines


def _group_views(registry: Registry, experiment_id: str) -> list[str]:
    lines = ["## Corridor by group (latest date, capital weight)", ""]
    for title, key in GROUP_VIEWS.items():
        c = latest(corridor(registry, experiment_id, by=[key]))
        c = c[c["measure"] == "capital_weight"]
        cols = ["data_variant", key, "n_cells", "n_ok", "median", "p25", "p75",
                "share_below_0.25pct"]  # fmt: skip
        lines += [f"### By {title}", "", md_table(c[cols], _CORR_FMT), ""]
    return lines


def _failures(cells: pd.DataFrame) -> list[str]:
    failed = cells[cells["status"] != "ok"]
    lines = ["## Failures", ""]
    if failed.empty:
        return lines + ["No failed cells.", ""]
    t = (failed.groupby(["data_variant", "allocator", "constraint_set", "status"])
         .size().rename("n").reset_index())  # fmt: skip
    kinds = failed["message"].fillna("").map(message_kind)
    top = kinds.value_counts().head(5).rename_axis("message (numbers masked)").rename("n")
    return lines + [md_table(t), "", "Most common failure kinds:", "",
                    md_table(top.reset_index()), ""]  # fmt: skip


def message_kind(message: str) -> str:
    """First line of a failure message with numbers masked, so similar failures group."""
    first = message.split("\n")[0][:110]
    return re.sub(r"\d+(\.\d+)?", "#", first)


def _agreement_section(registry: Registry, experiment_id: str) -> list[str]:
    pairs = paired_cells(registry, experiment_id)
    if pairs.empty:
        return []  # single-library experiment: nothing to compare
    lines = ["## Library agreement (Riskfolio-Lib vs skfolio)", "",
             "Identical configurations solved by both libraries, compared on the candidate's "
             "capital weight at every rebalance date. Disagreement flags implementation risk; "
             "HERC/NCO differ by design unless `max_clusters` is set (cluster-count selection).",
             ""]  # fmt: skip
    s = agreement_summary(pairs)
    fmt = {"median_abs_diff": pct(3), "max_abs_diff": pct(3), "n_pairs": num(0),
           "n_status_mismatch": num(0)}  # fmt: skip
    return lines + [md_table(s, fmt), ""]


def _risk_budget_section(registry: Registry, experiment_id: str) -> list[str]:
    v = risk_budget_view(registry, experiment_id)
    if v.empty:
        return []
    lines = ["## How much of our risk should it carry? (risk budgets)", "",
             "Capital weight implied by giving the candidate a target share of risk under each "
             "lens (all rebalance dates; `latest` = latest date). Realised shares are exact for "
             "smooth measures (MV, MSV); for CVaR/CDaR on historical scenarios risk "
             "contributions are not unique, so the realised share differs from the target even "
             "at the optimum, and on short windows (few tail scenarios) different targets can "
             "give nearly the same weight.", ""]  # fmt: skip
    fmt = {c: _P2 for c in ("weight_median", "weight_p25", "weight_p75", "weight_latest",
                            "realised_share_median", "target_share")}  # fmt: skip
    fmt |= {"n_cells": num(0), "n_ok": num(0)}
    return lines + [md_table(v, fmt), ""]


def _sweep_section(registry: Registry, experiment_id: str) -> list[str]:
    v = sweep_view(registry, experiment_id)
    if v.empty:
        return []
    lines = ["## Constraint sweeps (e.g. what fits inside a TE budget?)", "",
             "Candidate capital weight per swept constraint value: corridor at the latest date, "
             "its median through time, and the realised out-of-sample TE and candidate weight "
             "over the configurations that traded (at least one successful rebalance; "
             "`n_configs_oos`).", ""]  # fmt: skip
    cols = [c for c in ("data_variant", "base", "swept", "value", "n_cells", "n_ok", "median",
                        "p25", "p75", "median_through_time", "n_configs_oos", "oos_te_median",
                        "oos_weight_median") if c in v.columns]  # fmt: skip
    fmt = {c: _P2 for c in ("median", "p25", "p75", "median_through_time", "oos_weight_median",
                            "oos_te_median")}  # fmt: skip
    fmt |= {"n_cells": num(0), "n_ok": num(0), "n_configs_oos": num(0)}
    return lines + [md_table(v[cols], fmt), ""]


def _oos_section(oos: pd.DataFrame, spec: ExperimentSpec, min_oos: int) -> list[str]:
    lines = ["## Out-of-sample (walk-forward) vs SAA", ""]
    if oos.empty:
        return lines + [
            "> **No out-of-sample evidence**: this experiment ran in `in_sample` mode.",
            "",
        ]
    te_limits = {cs.name: cs.te_annual for cs in spec.constraint_sets}
    for variant, o in oos.groupby("data_variant", sort=True):
        n = int(o["n_periods"].iloc[0])
        lines += [f"### {variant}: {o['start'].iloc[0]} .. {o['end'].iloc[0]} ({n} periods)", ""]
        if n < min_oos:
            lines += [f"> ⚠ **Sample too short** ({n} < {min_oos} periods): "
                      "treat these statistics as noise, not evidence.", ""]  # fmt: skip
        limit = pd.to_numeric(o["constraint_set"].map(te_limits), errors="coerce")
        t = o.assign(te_limit=limit.astype(float))
        t["te_breach"] = (t["te_vs_saa"] > t["te_limit"]).map({True: "yes", False: ""})
        cols = ["label", "constraint_set", "n_failed_rebalances", "candidate_weight_median",
                "ann_return", "ann_vol", "max_dd", "te_vs_saa", "te_limit", "te_breach",
                "turnover_ann"]  # fmt: skip
        if spec.costs is not None:
            cols += ["ann_return_gross", "cost_drag_ann"]
        if spec.liquidity is not None:
            cols += ["n_liquidity_adjusted", "n_adjustment_breaches"]
        if spec.rebalance.kind == "threshold":
            cols += ["n_trades"]
        pct_cols = ("candidate_weight_median", "ann_return", "ann_vol", "max_dd", "te_vs_saa",
                    "te_limit", "turnover_ann", "ann_return_gross")  # fmt: skip
        fmt = {c: _P for c in pct_cols}
        fmt["cost_drag_ann"] = pct(2)
        for c in ("n_failed_rebalances", "n_liquidity_adjusted", "n_adjustment_breaches",
                  "n_trades"):  # fmt: skip
            fmt[c] = num(0)
        lines += [md_table(t[cols], fmt), ""]
    lines += ["`te_limit` is the ex-ante limit enforced on each fitting window; `te_vs_saa` is "
              "realised out of sample. Returns are net of transaction costs when the spec sets "
              "them; the SAA path pays costs for its own rebalancing.", ""]  # fmt: skip
    lines += _frictions_note(spec)
    return lines


def _evidence_section(registry: Registry, experiment_id: str, min_oos: int) -> list[str]:
    ev = registry.evidence(experiment_id)
    lines = ["## Evidence net of search", "",
             "Disclosures, not gates. Sharpe test: Ledoit–Wolf studentised block bootstrap vs the "
             "SAA path on returns in excess of the riskless asset (or the policy rf), with p raw "
             "and Benjamini–Hochberg across configurations. DSR: deflated Sharpe "
             "of the information ratio vs the SAA, deflated for every configuration tried. PBO: "
             "share of CSCV splits where the in-sample best configuration ranks below the "
             "out-of-sample median (≈0.5 for pure noise).", ""]  # fmt: skip
    if ev.empty:
        return lines + ["_No evidence stored for this experiment._", ""]
    extra = ev["extra_json"].map(json.loads)
    ev = ev.assign(extra=extra)
    cells = registry.cells(experiment_id)
    meta = cells.groupby("config_id").first()
    for variant, g in ev.groupby("data_variant", sort=True):
        lines += [f"### {variant}", ""]
        sd = g[g["test"] == "sharpe_diff"]
        if not sd.empty:
            n_obs = int(sd["extra"].iloc[0]["n_obs"])
            pbo = g[g["test"] == "pbo"]
            dsr = g[g["test"] == "dsr"].set_index("subject")
            n_trials = int(dsr["extra"].iloc[0]["n_trials"]) if not dsr.empty else len(sd)
            sr_star = dsr["extra"].iloc[0].get("sr_star_ann") if not dsr.empty else None
            pbo_val = None if pbo.empty else pbo["statistic"].iloc[0]
            pbo_txt = "–" if pbo_val is None or pd.isna(pbo_val) else f"{pbo_val:.2f}"
            star_txt = "–" if sr_star is None else f"{sr_star:.2f}"
            lines += [f"Trials N = {n_trials}, OOS periods T = {n_obs}, PBO = {pbo_txt}, "
                      f"expected max IR under the null = {star_txt} (annualised).", ""]  # fmt: skip
            if n_obs < min_oos:
                lines += [f"> ⚠ **Sample too short** ({n_obs} < {min_oos} periods): tests have "
                          "almost no power here.", ""]  # fmt: skip
            rows = []
            for r in sd.itertuples():
                m = meta.loc[r.subject] if r.subject in meta.index else None
                d = dsr.loc[r.subject] if r.subject in dsr.index else None
                rows.append({
                    "label": "?" if m is None else cell_label(m.allocator, m.params_json,
                                                              m.estimator_json),
                    "constraint_set": "" if m is None else m.constraint_set,
                    "sr_ann": r.extra.get("sr_ann"), "sr_saa_ann": r.extra.get("sr_saa_ann"),
                    "diff_ann": r.extra.get("diff_ann"), "p": r.p_value,
                    "p_bh": r.extra.get("p_bh"),
                    "ir_ann": None if d is None else d.extra.get("ir_ann"),
                    "dsr": None if d is None else d.statistic,
                    "note": r.extra.get("note") or ("" if d is None else d.extra.get("note")),
                    "_order": 0 if m is None else int(m.cell_index),
                })  # fmt: skip
            t = pd.DataFrame(rows).sort_values("_order").drop(columns="_order")
            t = t.apply(lambda c: pd.to_numeric(c) if c.name in NUMERIC_EVIDENCE else c)
            fmt = {c: num(2) for c in ("sr_ann", "sr_saa_ann", "diff_ann", "ir_ann")}
            fmt |= {"p": num(3), "p_bh": num(3), "dsr": num(2)}
            lines += [md_table(t, fmt), ""]
        sp = g[g["subject"] == "candidate"]
        if not sp.empty:
            x0 = sp["extra"].iloc[0]
            how = (f"excess returns over the riskless asset {x0['riskless']}" if x0.get("riskless")
                   else "Huberman–Kandel / Kan–Zhou on raw returns")  # fmt: skip
            st = pd.DataFrame({
                "test": sp["test"].to_numpy(), "statistic": sp["statistic"].to_numpy(),
                "p_value": sp["p_value"].to_numpy(),
                "alpha_ann": [x.get("alpha_ann") for x in sp["extra"]],
                "n_obs": [x.get("n_obs") for x in sp["extra"]],
            })  # fmt: skip
            st["alpha_ann"] = pd.to_numeric(st["alpha_ann"])
            lines += [f"Spanning (candidate vs the SAA building blocks, {how}, full history of "
                      "this variant; low power on short histories):", "",
                      md_table(st, {"statistic": num(3), "p_value": num(3),
                                    "alpha_ann": pct(2)}), ""]  # fmt: skip
    return lines


NUMERIC_EVIDENCE = {"sr_ann", "sr_saa_ann", "diff_ann", "p", "p_bh", "ir_ann", "dsr"}


def _frictions_note(spec: ExperimentSpec) -> list[str]:
    parts = []
    if spec.costs is not None:
        per = ", ".join(f"{k} {v:g}" for k, v in (spec.costs.per_asset or {}).items())
        parts.append(f"costs {spec.costs.default_bps:g} bps one-way" + (f" ({per})" if per else ""))
    if spec.liquidity is not None:
        lq = spec.liquidity
        parts.append(f"candidate deals {lq.dealing}, notice {lq.notice_periods} dealing date(s), "
                     f"gate {'none' if lq.gate is None else f'{lq.gate:.0%}'}")  # fmt: skip
    if spec.rebalance.kind == "threshold":
        parts.append(f"threshold rebalancing at {spec.rebalance.band:.1%} drift")
    if spec.funding != "pro_rata":
        parts.append(f"default funding {spec.funding}")
    return [f"Frictions: {'; '.join(parts)}.", ""] if parts else []


def _expost_section(expost: pd.DataFrame) -> list[str]:
    lines = ["## In-sample ex-post (latest date)", "",
             "> In-sample: fixed weights on the window they were fitted on. These flatter the "
             "optimisers; use the out-of-sample section for evidence.", ""]  # fmt: skip
    last = expost.groupby("data_variant")["window_end"].transform("max")
    t = expost[expost["window_end"] == last]
    cols = ["data_variant", "label", "constraint_set", "candidate_weight", "ann_return",
            "ann_vol", "cvar95", "cdar95", "max_dd", "te_vs_saa"]  # fmt: skip
    fmt = {c: _P for c in cols[3:]}
    return lines + [md_table(t[cols], fmt), ""]


def _live_only_note(variants: list[str], corr: pd.DataFrame) -> list[str]:
    lines = ["## Live-only variant", ""]
    if "live_only" not in variants:
        return lines + ["The candidate has no backfilled observations: the full history is "
                        "live, so the live-only variant equals the full one.", ""]  # fmt: skip
    t = latest(corr)
    a = t[t["data_variant"] == "full"].drop(columns=["data_variant", "window_end"])
    b = t[t["data_variant"] == "live_only"].drop(columns=["data_variant", "window_end"])
    same = len(a) == len(b) and a.reset_index(drop=True).equals(b.reset_index(drop=True))
    note = ("At the latest date the live-only corridor is identical to the full one: the "
            "estimation window lies entirely in live history." if same else
            "The live-only corridor differs from the full one: backfilled observations "
            "influence the result. Compare both sections above.")  # fmt: skip
    return lines + [note, ""]


def _definitions(spec: ExperimentSpec) -> list[str]:
    return [
        "## Definitions",
        "",
        "- Risk shares: Riskfolio-Lib Euler risk contributions on the fitting window "
        "(lens definitions: CDaR on uncompounded cumulative returns).",
        "- Ex-post statistics: compounded wealth drawdowns; CVaR 95% is per period "
        f"({spec.data.frequency}), not annualised; TE is the annualised std of active returns.",
        "- Policy TE limits are enforced with Riskfolio-Lib's definition (RMS of active returns, "
        "not demeaned) on each fitting window.",
        "- Failure policy: until the first successful fit the path holds the SAA; afterwards a "
        "failed rebalance keeps the drifted weights. No transaction costs.",
        f"- Generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC from the registry.",
        "",
    ]

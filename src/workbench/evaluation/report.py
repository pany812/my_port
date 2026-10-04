"""Experiment report: CSV tables plus a markdown summary, built from the registry alone."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from workbench.evaluation.corridor import FAILED_STATUSES, corridor
from workbench.evaluation.expost import expost_table
from workbench.evaluation.markdown import md_table, num, pct
from workbench.evaluation.oos import oos_table
from workbench.grid.spec import ExperimentSpec, parse_spec
from workbench.registry.store import Registry

MIN_OOS_PERIODS = 36
STAT_COLS = ["median", "p25", "p75", "p10", "p90", "share_below_0.25pct"]
GROUP_VIEWS = {
    "allocator family": "family",
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
    lines += _oos_section(oos, spec, min_oos)
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
            ("window / rebalance", f"{window} / every {spec.rebalance.every}"),
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
        fmt = {c: _P for c in ("candidate_weight_median", "ann_return", "ann_vol", "max_dd",
                               "te_vs_saa", "te_limit", "turnover_ann")}  # fmt: skip
        fmt["n_failed_rebalances"] = num(0)
        lines += [md_table(t[cols], fmt), ""]
    lines += ["`te_limit` is the ex-ante limit enforced on each fitting window; `te_vs_saa` is "
              "realised out of sample.", ""]  # fmt: skip
    return lines


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

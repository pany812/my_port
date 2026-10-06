"""IC memo (P2-M6): a short SCQA decision document built from the registry alone.

The workbench never recommends. The recommendation, proposal, target corridor, conditions and
kill criteria come from the spec's ``decision`` block (people). The memo sets the evidence next
to them and flags where they disagree (:func:`memo_checks`). Flags are information, not gates.

The decision block is not part of ``spec_hash``: it is read from the spec stored with the run, or
from a revised spec file whose hash (ignoring the decision) equals the experiment's. Without a
decision block the memo is a DRAFT with the corridor median as the reference weight.

Kill criteria with a metric are replayed on the stored walk-forward path of the ``saa_plus``
configuration at the proposed weight and funding: how often the criterion would have fired.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from workbench.evaluation.agreement import agreement_summary, paired_cells
from workbench.evaluation.corridor import corridor
from workbench.evaluation.expost import cell_label
from workbench.evaluation.markdown import md_table, num, pct
from workbench.evaluation.oos import oos_table
from workbench.evaluation.report import MIN_OOS_PERIODS, latest
from workbench.evaluation.stress import bootstrap_table, window_table
from workbench.evaluation.views import breakeven_view
from workbench.grid.spec import DecisionSpec, ExperimentSpec, KillCriterion, load_spec, parse_spec
from workbench.registry.store import REFERENCE_ALLOCATOR, Registry
from workbench.units import periods_per_year

SIGNIFICANCE = 0.05  # pending the house statistical conventions (PHASE2 open question 10)
DSR_MIN = 0.95
PBO_MAX = 0.5
P_WORSE_MAX = 0.5
FAILED_MAX = 0.5
BACKFILL_MAX = 0.5
AGREEMENT_MAX = 0.01  # candidate weight, decimal
FLAG = "⚠"
_P = pct(1)
_P2 = pct(2)


# --- decision ---------------------------------------------------------------------------------


def resolve_decision(
    registry: Registry, experiment_id: str, spec_path: str | Path | None = None
) -> tuple[ExperimentSpec, DecisionSpec | None, str]:
    """(stored spec, decision, where the decision came from).

    ``spec_path``: a revised spec whose ``spec_hash`` (which ignores the decision block) must
    equal the experiment's, so the evidence still describes it. Raises ValueError otherwise.
    """
    exp = registry.experiment(experiment_id)
    stored = parse_spec(exp["spec_yaml"])
    if spec_path is None:
        return stored, stored.decision, "the spec stored with the run"
    current = load_spec(spec_path)
    if current.spec_hash != exp["spec_hash"]:
        raise ValueError(f"{spec_path} does not describe experiment {experiment_id}: its spec_hash "
                         "differs (only the decision block may change after a run)")  # fmt: skip
    return stored, current.decision, str(spec_path)


# --- kill criteria ----------------------------------------------------------------------------


def kill_triggers(port: pd.Series, saa: pd.Series, k: KillCriterion, freq: str) -> dict:
    """Replay a structured kill criterion on a path vs the SAA path (same dates, per period).

    te_vs_saa: rolling std (ddof=1) of active returns x sqrt(n). active_return: rolling
    compounded return minus the SAA's. Returns n_windows, n_triggered, first (date of the first
    trigger, or None) and worst (the most extreme rolling value), or a note.
    """
    ppy = periods_per_year(freq)
    n = round(k.months * ppy / 12)
    if n < 2 or not math.isclose(n, k.months * ppy / 12):
        return {"note": f"{k.months} months is not a whole number (>= 2) of {freq} periods"}
    if len(port) < n:
        return {"note": f"the path has {len(port)} periods, fewer than {n}"}
    if k.metric == "te_vs_saa":
        val = (port - saa).rolling(n).std(ddof=1) * math.sqrt(ppy)
    else:
        val = (np.exp(np.log1p(port).rolling(n).sum())
               - np.exp(np.log1p(saa).rolling(n).sum()))  # fmt: skip
    val = val.dropna()
    trig = val > k.above if k.above is not None else val < k.below
    worst = val.max() if k.above is not None else val.min()
    first = val.index[trig.to_numpy()][0] if trig.any() else None
    return {"n_windows": len(val), "n_triggered": int(trig.sum()),
            "first": None if first is None else pd.Timestamp(first).date().isoformat(),
            "worst": float(worst)}  # fmt: skip


# --- facts and checks -------------------------------------------------------------------------


@dataclass
class Facts:
    """Every number the checks look at (None = not available in this experiment)."""

    x_ref: float
    x_source: str  # "proposal" | "corridor median"
    target_corridor: tuple[float, float] | None = None
    corridor_full: dict | None = None  # p10, p25, median, p75, p90 (latest date, capital weight)
    corridor_live: dict | None = None
    n_configs: int | None = None
    n_significant: int | None = None
    proposal_p_bh: float | None = None
    proposal_label: str | None = None
    dsr_best: float | None = None
    pbo: float | None = None
    spanning_p: float | None = None
    spanning_alpha_ann: float | None = None
    oos_periods: int | None = None
    stress_x: float | None = None
    p_worse: float | None = None
    bl_x: float | None = None
    required_excess: float | None = None
    cma_excess: float | None = None
    agreement_max_diff: float | None = None
    agreement_mismatches: int | None = None
    failed_share: float | None = None
    backfilled_share: float | None = None
    notes: dict = field(default_factory=dict)


def _check(rows, check, value, flagged, note=""):
    rows.append({"check": check, "value": value, "flag": FLAG if flagged else "", "note": note})


def _nearest_note(x_used: float | None, x_ref: float) -> str:
    if x_used is None or math.isclose(x_used, x_ref, abs_tol=5e-5):
        return ""
    return f"nearest computed weight {x_used:.2%} (proposal {x_ref:.2%})"


def memo_checks(f: Facts) -> pd.DataFrame:
    """One row per check: check, value, flag ("⚠" where the evidence disagrees with the
    proposal), note. Missing evidence is "not evaluated", never flagged."""
    rows: list[dict] = []
    na = "not evaluated"
    who = "Proposal" if f.x_source == "proposal" else "Reference weight (corridor median)"
    for label, c in (("", f.corridor_full), ("live-only ", f.corridor_live)):
        if c is None:
            if not label:
                _check(rows, f"{who} within the corridor P25–P75", na, False, "no ok cells")
            continue
        inside = c["p25"] - 1e-9 <= f.x_ref <= c["p75"] + 1e-9
        wide = c["p10"] - 1e-9 <= f.x_ref <= c["p90"] + 1e-9
        _check(
            rows,
            f"{who} within the {label}corridor P25–P75",
            f"{f.x_ref:.1%} vs {c['p25']:.1%}–{c['p75']:.1%}",
            not inside,
            ""
            if inside
            else (
                "inside P10–P90" if wide else f"outside P10–P90 too ({c['p10']:.1%}–{c['p90']:.1%})"
            ),
        )
    if f.target_corridor is not None and f.corridor_full is not None:
        lo, hi = f.target_corridor
        c = f.corridor_full
        within = lo >= c["p10"] - 1e-9 and hi <= c["p90"] + 1e-9
        _check(
            rows,
            "Target corridor within the evidence corridor P10–P90",
            f"{lo:.1%}–{hi:.1%} vs {c['p10']:.1%}–{c['p90']:.1%}",
            not within,
            "" if within else "the approved range reaches weights few methods support",
        )
    if f.n_configs:
        _check(
            rows,
            f"Configurations beating the SAA (Sharpe, BH-adjusted p < {SIGNIFICANCE:.0%})",
            f"{f.n_significant} of {f.n_configs}",
            f.n_significant == 0,
        )
    else:
        _check(
            rows,
            "Configurations beating the SAA (Sharpe, BH-adjusted)",
            na,
            False,
            f.notes.get("sharpe", "no walk-forward evidence"),
        )
    if f.proposal_p_bh is not None:
        _check(
            rows,
            "The proposal's own Sharpe difference vs the SAA (BH-adjusted p)",
            f"{f.proposal_p_bh:.3f}",
            f.proposal_p_bh >= SIGNIFICANCE,
            f.proposal_label or "",
        )
    else:
        _check(
            rows,
            "The proposal's own Sharpe difference vs the SAA",
            na,
            False,
            f.notes.get("proposal", "no saa_plus configuration at the proposed weight"),
        )
    _optional(
        rows,
        "Deflated Sharpe ratio, best configuration",
        f.dsr_best,
        lambda v: f"{v:.2f}",
        lambda v: v < DSR_MIN,
        f"below {DSR_MIN}",
    )
    _optional(
        rows,
        "Probability of backtest overfitting (PBO)",
        f.pbo,
        lambda v: f"{v:.2f}",
        lambda v: v > PBO_MAX,
        f"above {PBO_MAX}: the in-sample winner tends to lose",
    )
    if f.spanning_p is not None:
        alpha = "" if f.spanning_alpha_ann is None else f"alpha {f.spanning_alpha_ann:.2%} p.a., "
        _check(
            rows,
            "Spanning: candidate alpha vs the SAA building blocks",
            f"{alpha}p = {f.spanning_p:.3f}",
            f.spanning_p >= SIGNIFICANCE,
        )
    else:
        _check(rows, "Spanning: candidate alpha vs the SAA building blocks", na, False)
    _optional(
        rows,
        "Out-of-sample length (periods)",
        f.oos_periods,
        lambda v: f"{v}",
        lambda v: v < MIN_OOS_PERIODS,
        f"fewer than {MIN_OOS_PERIODS}: little power",
    )
    _optional(
        rows,
        "Stress: bootstrap paths with a deeper max drawdown than the SAA",
        f.p_worse,
        lambda v: f"{v:.0%}",
        lambda v: v > P_WORSE_MAX,
        _nearest_note(f.stress_x, f.x_ref),
        f.notes.get("stress", "no stress section in the spec"),
    )
    if f.required_excess is not None:
        cma = "" if f.cma_excess is None else f" vs CMA {f.cma_excess:.2%}"
        flagged = f.cma_excess is not None and f.required_excess > f.cma_excess
        note = _nearest_note(f.bl_x, f.x_ref) or ("" if f.cma_excess is not None else "no CMA")
        _check(
            rows,
            "Expected excess return needed (BL breakeven)",
            f"{f.required_excess:.2%} p.a.{cma}",
            flagged,
            note,
        )
    else:
        _check(
            rows,
            "Expected excess return needed (BL breakeven)",
            na,
            False,
            f.notes.get("bl", "no riskfolio_bl / skfolio_bl target_weight in the grid"),
        )
    if f.agreement_max_diff is not None:
        _check(
            rows,
            "Library agreement (max |Δ candidate weight|)",
            f"{f.agreement_max_diff:.2%}, {f.agreement_mismatches} status mismatches",
            f.agreement_max_diff > AGREEMENT_MAX,
        )
    _optional(
        rows,
        "Failed cells (full variant)",
        f.failed_share,
        lambda v: f"{v:.0%}",
        lambda v: v > FAILED_MAX,
        "see Failures in summary.md",
    )
    _optional(
        rows,
        "Backfilled share of the candidate's history",
        f.backfilled_share,
        lambda v: f"{v:.0%}",
        lambda v: v > BACKFILL_MAX,
        "results lean on backfilled or proxied data",
    )
    return pd.DataFrame(rows, columns=["check", "value", "flag", "note"])


def _optional(rows, check, value, fmt, bad, note_if_bad="", note_if_missing=""):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        _check(rows, check, "not evaluated", False, note_if_missing)
        return
    flagged = bool(bad(value))
    _check(rows, check, fmt(value), flagged, note_if_bad if flagged else "")


# --- gathering --------------------------------------------------------------------------------


def _extra(ev: pd.DataFrame) -> pd.Series:
    return ev["extra_json"].map(json.loads)


def _proposal_configs(cells: pd.DataFrame, x: float, funding: str) -> pd.DataFrame:
    """saa_plus configurations at weight ``x`` and ``funding`` (full variant), fewest failures
    first."""
    sp = cells[(cells["allocator"] == "saa_plus") & (cells["data_variant"] == "full")]
    if sp.empty:
        return sp
    params = sp["params_json"].map(json.loads)
    match = params.map(lambda p: math.isclose(float(p.get("x", -1)), x, abs_tol=1e-9)
                       and p.get("funding", "pro_rata") == funding)  # fmt: skip
    sp = sp[match]
    if sp.empty:
        return sp
    fails = sp.groupby("config_id")["status"].apply(lambda s: int((s != "ok").sum()))
    first = sp.groupby("config_id").first()
    return first.assign(n_failed=fails).sort_values(["n_failed", "cell_index"]).reset_index()


@dataclass
class _Context:
    registry: Registry
    exp: dict
    spec: ExperimentSpec
    decision: DecisionSpec | None
    decision_source: str
    cells: pd.DataFrame
    ev: pd.DataFrame
    corr: pd.DataFrame
    oos: pd.DataFrame
    funding: str


def _breakeven_rows(ctx: _Context, x: float) -> pd.DataFrame | None:
    """BL breakeven rows (full variant, Riskfolio when both libraries ran) at the target weight
    nearest to ``x``, one per constraint set and setting."""
    be = breakeven_view(ctx.registry, ctx.exp["experiment_id"])
    if be.empty:
        return None
    be = be[be["data_variant"] == "full"]
    if (be["library"] == "riskfolio").any():
        be = be[be["library"] == "riskfolio"]
    t = be["target_weight"].iloc[(be["target_weight"] - x).abs().argmin()]
    return be[be["target_weight"] == t].reset_index(drop=True)


def gather(ctx: _Context) -> Facts:
    """Collect the facts the checks and the memo body need."""
    d, spec = ctx.decision, ctx.spec
    last = latest(ctx.corr)
    cw = last[last["measure"] == "capital_weight"].set_index("data_variant")

    def quantiles(variant):
        if variant not in cw.index or not cw.loc[variant, "n_ok"]:
            return None
        return {k: float(cw.loc[variant, k]) for k in ("p10", "p25", "median", "p75", "p90")}

    full, live = quantiles("full"), quantiles("live_only")
    if d is not None and d.proposal_weight is not None:
        x_ref, source = d.proposal_weight, "proposal"
    else:
        x_ref, source = (full or {"median": 0.0})["median"], "corridor median"
    f = Facts(x_ref=x_ref, x_source=source, corridor_full=full, corridor_live=live,
              target_corridor=None if d is None else d.target_corridor)  # fmt: skip

    ev = ctx.ev[ctx.ev["data_variant"] == "full"]
    sd = ev[ev["test"] == "sharpe_diff"]
    if not sd.empty and sd["p_value"].notna().any():
        p_bh = _extra(sd).map(lambda e: e.get("p_bh"))
        f.n_configs = int(p_bh.notna().sum())
        f.n_significant = int((p_bh < SIGNIFICANCE).sum())
    elif not sd.empty:
        f.notes["sharpe"] = _extra(sd).iloc[0].get("note") or "not computed"
    dsr = ev[(ev["test"] == "dsr") & ev["statistic"].notna()]
    f.dsr_best = None if dsr.empty else float(dsr["statistic"].max())
    pbo = ev[(ev["test"] == "pbo") & ev["statistic"].notna()]
    f.pbo = None if pbo.empty else float(pbo["statistic"].iloc[0])
    span = ev[ev["test"].isin(["spanning_alpha", "spanning_hk"])]
    if not span.empty and pd.notna(span["p_value"].iloc[0]):
        f.spanning_p = float(span["p_value"].iloc[0])
        f.spanning_alpha_ann = _extra(span).iloc[0].get("alpha_ann")
    if not ctx.oos.empty:
        o = ctx.oos[(ctx.oos["data_variant"] == "full") & (ctx.oos["row"] == "SAA")]
        f.oos_periods = None if o.empty else int(o["n_periods"].iloc[0])

    configs = _proposal_configs(ctx.cells, x_ref, ctx.funding)
    if not configs.empty:
        cid = configs["config_id"].iloc[0]
        f.proposal_label = (f"{cell_label('saa_plus', configs['params_json'].iloc[0], None)} "
                            f"in {configs['constraint_set'].iloc[0]}")  # fmt: skip
        row = sd[sd["subject"] == cid]
        if not row.empty:
            f.proposal_p_bh = _extra(row).iloc[0].get("p_bh")
    elif source == "proposal":
        f.notes["proposal"] = (f"no saa_plus configuration at x={x_ref:g}, funding "
                               f"{ctx.funding}: add it to the grid")  # fmt: skip

    bt = bootstrap_table(ctx.ev)
    if not bt.empty:
        b = bt[(bt["data_variant"] == "full") & bt["max_dd_median"].notna()]
        if not b.empty:
            i = (b["x"] - x_ref).abs().idxmin()
            f.stress_x, f.p_worse = float(b.loc[i, "x"]), float(b.loc[i, "p_worse_max_dd"])

    be = _breakeven_rows(ctx, x_ref)
    if be is not None and not be.empty and be["excess_median"].notna().any():
        f.bl_x = float(be["target_weight"].iloc[0])
        f.required_excess = float(be["excess_median"].median())
    elif be is not None and not be.empty:
        f.notes["bl"] = "the target weight was unreachable under every constraint set"
    if spec.cma is not None:
        v = spec.cma.vectors[-1].returns_annual.get(spec.data.candidate)
        f.cma_excess = None if v is None else float(v) - spec.rf_annual

    pairs = paired_cells(ctx.registry, ctx.exp["experiment_id"])
    if not pairs.empty:
        summ = agreement_summary(pairs)
        f.agreement_max_diff = float(summ["max_abs_diff"].max())
        f.agreement_mismatches = int(summ["n_status_mismatch"].sum())
    grid = ctx.cells[ctx.cells["data_variant"] == "full"]
    f.failed_share = float((grid["status"] != "ok").mean()) if len(grid) else None
    prof = ev[ev["subject"] == "profile:candidate"]
    if not prof.empty:
        f.backfilled_share = _extra(prof).iloc[0].get("backfilled_share")
    return f


# --- the memo ---------------------------------------------------------------------------------


@dataclass
class Memo:
    experiment_id: str
    name: str
    status: str  # "DRAFT" (no decision block, or no proposal / recommendation) or "PROPOSED"
    markdown: str
    checks: pd.DataFrame
    facts: Facts

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "memo.md"
        path.write_text(self.markdown)
        return path


def build_memo(
    registry: Registry, experiment_id: str, spec_path: str | Path | None = None,
    today: dt.date | None = None,
) -> Memo:  # fmt: skip
    """Build the IC memo for ``experiment_id`` from the registry (and, optionally, a revised
    decision block in ``spec_path``)."""
    spec, decision, source = resolve_decision(registry, experiment_id, spec_path)
    exp = registry.experiment(experiment_id)
    funding = (decision.proposal_funding if decision is not None and decision.proposal_funding
               else spec.funding)  # fmt: skip
    ctx = _Context(registry, exp, spec, decision, source, registry.cells(experiment_id),
                   registry.evidence(experiment_id), corridor(registry, experiment_id),
                   oos_table(registry, experiment_id), funding)  # fmt: skip
    facts = gather(ctx)
    checks = memo_checks(facts)
    proposed = (decision is not None and decision.recommendation is not None
                and decision.proposal_weight is not None)  # fmt: skip
    status = "PROPOSED" if proposed else "DRAFT"
    md = _render(ctx, facts, checks, status, today or dt.date.today())
    return Memo(experiment_id, exp["name"], status, md, checks, facts)


# --- rendering --------------------------------------------------------------------------------


def _render(ctx: _Context, f: Facts, checks: pd.DataFrame, status: str, today: dt.date) -> str:
    d, exp = ctx.decision, ctx.exp
    name = (d.candidate_name if d is not None and d.candidate_name else None) or exp["candidate_id"]
    lines = [f"# IC memo: {name} in the SAA", ""]
    meta = [
        f"**Status: {status}**",
        f"generated {today.isoformat()}",
        f"experiment `{exp['name']}` (`{exp['experiment_id']}`)",
        f"spec_hash `{exp['spec_hash'][:12]}`",
        f"data vintage `{exp['data_vintage'][:12]}`",
    ]
    if d is not None and d.owner:
        meta.append(f"owner {d.owner}")
    if d is not None and d.review:
        meta.append(f"review {d.review}")
    lines += [" · ".join(meta), "", f"_Decision text from {ctx.decision_source}._", ""]
    lines += _recommendation(ctx, f, checks)
    lines += _situation(ctx, f)
    lines += ["## Complication", "",
              f"{name} is not in the SAA. Adding it changes risk and return in ways a single "
              "optimiser would overstate, so the evidence is a corridor across "
              f"{_n_configs(ctx)} configurations, tested out of sample and net of how much was "
              "searched.", "",
              "## Question", "",
              f"Should the SAA hold {name}, and if so in what corridor?", ""]  # fmt: skip
    lines += _answer(ctx, f)
    lines += _conditions_and_kills(ctx, f)
    lines += _robustness(ctx, f)
    lines += ["## Appendix", "",
              "- Corridor: distribution over successful cells of the candidate's capital weight "
              "and risk share at the latest rebalance date; failed cells are counted, not dropped.",
              "- Evidence statistics are disclosures, never gates. Sharpe tests are on returns in "
              "excess of cash, Benjamini–Hochberg adjusted across configurations; the deflated "
              "Sharpe ratio and PBO account for the number of configurations tried.",
              f"- Checks use a {SIGNIFICANCE:.0%} significance level, DSR {DSR_MIN}, PBO "
              f"{PBO_MAX} and library agreement {AGREEMENT_MAX:.0%} (house conventions pending).",
              f"- Full evidence: `wb report {exp['name']}` (summary.md).", ""]  # fmt: skip
    return "\n".join(lines).rstrip() + "\n"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def _n_configs(ctx: _Context) -> int:
    return int(ctx.cells.loc[ctx.cells["data_variant"] == "full", "config_id"].nunique())


def _recommendation(ctx: _Context, f: Facts, checks: pd.DataFrame) -> list[str]:
    d = ctx.decision
    text = (
        d.recommendation
        if d is not None and d.recommendation
        else ("_To be written by the proposer (add `decision.recommendation` to the spec)._")
    )
    lines = ["## Recommendation", "", f"> {text}", ""]
    if f.x_source == "proposal":
        tc = (
            ""
            if d.target_corridor is None
            else (f"; target corridor {d.target_corridor[0]:.1%}–{d.target_corridor[1]:.1%}")
        )
        lines += [f"Proposal: {f.x_ref:.1%} funded `{ctx.funding}`{tc}.", ""]
    else:
        lines += [
            f"No proposal in the decision block: the reference weight below is the "
            f"corridor median, {f.x_ref:.1%}.",
            "",
        ]
    n_flag = int((checks["flag"] == FLAG).sum())
    lines += [
        "### Checks: where the proposal and the evidence disagree",
        "",
        md_table(checks),
        "",
        f"{n_flag} of {len(checks)} checks flagged. Flags are information, not gates.",
        "",
    ]
    return lines  # fmt: skip


def _situation(ctx: _Context, f: Facts) -> list[str]:
    spec, exp = ctx.spec, ctx.exp
    data = spec.data
    lines = ["## Situation", "",
             f"- SAA version `{exp['saa_version']}`; candidate `{exp['candidate_id']}`; data "
             f"{data.source}, {data.frequency}, {data.start}..{data.end}, {data.base_currency}, "
             f"{data.hedging}.",
             f"- Searched: {_plural(_n_configs(ctx), 'configuration')} "
             f"({_plural(len(spec.allocators), 'allocator entry', 'allocator entries')}, "
             f"{_plural(len(spec.estimators), 'estimator')}, "
             f"{_plural(len(spec.constraint_sets), 'constraint set')}) over "
             f"{_plural(ctx.cells['window_end'].nunique(), 'rebalance date')}; "
             f"{_plural(len(ctx.cells), 'cell')}.", ""]  # fmt: skip
    prof = ctx.ev[ctx.ev["test"] == "profile"]
    if prof.empty:
        return lines + ["_Candidate and SAA profiles not stored for this experiment (it predates "
                        "P2-M6); re-run to add them._", ""]  # fmt: skip
    rows = []
    for r in prof.itertuples():
        e = json.loads(r.extra_json)
        rows.append({"data_variant": r.data_variant,
                     "portfolio": r.subject.removeprefix("profile:"),
                     "period": f"{e['first']}..{e['last']}", "ann_return": e["ann_return"],
                     "ann_vol": e["ann_vol"], "max_dd": e["max_dd"],
                     "sharpe": e.get("sharpe_ann"), "corr_saa": e.get("corr_saa"),
                     "backfilled": e.get("backfilled_share")})  # fmt: skip
    t = pd.DataFrame(rows)
    fmt = {c: _P for c in ("ann_return", "ann_vol", "max_dd", "backfilled")}
    fmt |= {"sharpe": num(2), "corr_saa": num(2)}
    return lines + ["Standalone profiles (fixed weights for the SAA; Sharpe over cash):", "",
                    md_table(t, fmt), ""]  # fmt: skip


def _answer(ctx: _Context, f: Facts) -> list[str]:
    lines = ["## Answer", "", "### 1. The corridor", ""]
    last = latest(ctx.corr)
    t = last[["data_variant", "measure", "n_cells", "n_ok", "median", "p25", "p75", "p10", "p90",
              "share_below_0.25pct"]]  # fmt: skip
    fmt = {c: _P for c in ("median", "p25", "p75", "p10", "p90")}
    fmt |= {"share_below_0.25pct": pct(0)}
    lines += [f"Latest rebalance date {last['window_end'].max()}; capital weight and risk share "
              "per lens.", "", md_table(t, fmt), ""]  # fmt: skip
    fam = latest(corridor(ctx.registry, ctx.exp["experiment_id"], by=["family"]))
    fam = fam[(fam["measure"] == "capital_weight") & (fam["data_variant"] == "full")]
    lines += [
        "By allocator family (full variant, capital weight):",
        "",
        md_table(fam[["family", "n_cells", "n_ok", "median", "p25", "p75"]], fmt),
        "",
    ]

    lines += ["### 2. Evidence net of search", ""]
    if f.n_configs:
        lines.append(
            f"- {f.n_significant} of {f.n_configs} configurations beat the SAA's Sharpe "
            f"ratio at {SIGNIFICANCE:.0%} after Benjamini–Hochberg adjustment."
        )
    for label, v, fm in (
        ("Best deflated Sharpe ratio", f.dsr_best, "{:.2f}"),
        ("PBO", f.pbo, "{:.2f}"),
        ("Out-of-sample periods", f.oos_periods, "{}"),
    ):
        if v is not None:
            lines.append(f"- {label}: {fm.format(v)}.")
    if f.spanning_p is not None:
        a = "" if f.spanning_alpha_ann is None else f"alpha {f.spanning_alpha_ann:.2%} p.a., "
        lines.append(f"- Spanning (candidate vs the building blocks): {a}p = {f.spanning_p:.3f}.")
    if len(lines) and lines[-1] == "":
        lines.append("_No walk-forward evidence in this experiment._")
    lines += [""]

    lines += ["### 3. Risk at the proposal", ""]
    lines += _proposal_risk(ctx, f)
    lines += ["### 4. Return assumptions", ""]
    be = _breakeven_rows(ctx, f.x_ref)
    if be is not None and not be.empty:
        note = _nearest_note(f.bl_x, f.x_ref)
        cols = [
            "constraint_set",
            "n_cells",
            "n_reached",
            "excess_median",
            "excess_p25",
            "excess_p75",
            "equilibrium_median",
            "sharpe_median",
        ]
        fmt = {c: _P2 for c in cols if c.startswith(("excess", "equilibrium"))}
        fmt |= {"sharpe_median": num(2), "n_cells": num(0), "n_reached": num(0)}
        lines += [f"Black–Litterman breakeven at {f.bl_x:.1%}"
                  f"{f' ({note})' if note else ''}: the expected excess return over rf the "
                  "candidate needs for the optimiser to hold it, with the SAA as the prior.", "",
                  md_table(be[cols], fmt), ""]  # fmt: skip
    else:
        lines += ["_No Black–Litterman breakeven in this experiment._", ""]
    if f.cma_excess is not None:
        lines += [
            f"CMA `{ctx.spec.cma.version}`: the candidate's expected excess return is "
            f"{f.cma_excess:.2%} p.a.",
            "",
        ]
    return lines


def _proposal_risk(ctx: _Context, f: Facts) -> list[str]:
    lines = []
    configs = _proposal_configs(ctx.cells, f.x_ref, ctx.funding)
    if not configs.empty and not ctx.oos.empty:
        cid = configs["config_id"].iloc[0]
        o = ctx.oos[ctx.oos["data_variant"] == "full"]
        rows = o[(o["row"] == "SAA") | (o["config_id"] == cid)]
        cols = ["label", "constraint_set", "ann_return", "ann_vol", "max_dd", "te_vs_saa"]
        fmt = {c: _P2 for c in cols[2:]}
        lines += ["Out of sample (walk-forward, net), the proposal vs the SAA:", "",
                  md_table(rows[cols], fmt), ""]  # fmt: skip
    else:
        lines += [f"_No saa_plus path at {f.x_ref:.1%} funded `{ctx.funding}`: add "
                  f"`saa_plus` x={f.x_ref:g} to the grid._", ""]  # fmt: skip
    wt = window_table(ctx.ev, None if ctx.spec.stress is None else
                      [n for n, _, _ in ctx.spec.stress.windows])  # fmt: skip
    if not wt.empty and f.stress_x is not None:
        w = wt[(wt["data_variant"] == "full") & np.isclose(wt["x"], f.stress_x)]
        note = _nearest_note(f.stress_x, f.x_ref)
        cols = [
            "window",
            "first",
            "last",
            "return",
            "d_return",
            "max_dd",
            "d_max_dd",
            "backfilled_share",
            "note",
        ]
        fmt = {c: _P2 for c in ("return", "d_return", "max_dd", "d_max_dd", "backfilled_share")}
        lines += [f"Crisis windows at {f.stress_x:.1%}{f' ({note})' if note else ''}, SAA plus "
                  "the candidate (changes vs the SAA; positive `d_max_dd` = deeper):", "",
                  md_table(w[cols], fmt), ""]  # fmt: skip
        bt = bootstrap_table(ctx.ev)
        b = bt[(bt["data_variant"] == "full") & np.isclose(bt["x"], f.stress_x)]
        if not b.empty:
            r = b.iloc[0]
            lines += [f"Bootstrap ({int(r['n_paths']):,} paths of {int(r['horizon_years'])} "
                      f"years): median max drawdown {r['max_dd_median']:.1%} (SAA change "
                      f"{r['d_max_dd_median']:+.1%}), P95 {r['max_dd_p95']:.1%}; deeper than "
                      f"the SAA on {r['p_worse_max_dd']:.0%} of paths.", ""]  # fmt: skip
    elif ctx.spec.stress is None:
        lines += ["_No stress section in the spec._", ""]
    return lines


def _conditions_and_kills(ctx: _Context, f: Facts) -> list[str]:
    d = ctx.decision
    lines = ["## Conditions", ""]
    conds = [] if d is None else list(d.conditions)
    lines += [f"- {c}" for c in conds] or ["_None stated._"]
    lines += ["", "## Kill criteria", ""]
    kills = [] if d is None else list(d.kill_criteria)
    if not kills:
        return lines + ["_None stated._", ""]
    configs = _proposal_configs(ctx.cells, f.x_ref, ctx.funding)
    paths = ctx.registry.oos_returns(ctx.exp["experiment_id"], include_reference=True)
    port = saa = None
    if not configs.empty and not paths.empty:
        full = paths[paths["data_variant"] == "full"].assign(date=lambda x: pd.to_datetime(x.date))
        wide = full.pivot(index="date", columns="config_id", values="portfolio_return_net")
        cid = configs["config_id"].iloc[0]
        if cid in wide and REFERENCE_ALLOCATOR in wide:
            port, saa = wide[cid], wide[REFERENCE_ALLOCATOR]
    rows = []
    for k in kills:
        if k.text is not None:
            rows.append({"criterion": k.text, "backtest": "qualitative", "note": ""})
            continue
        if port is None:
            rows.append(
                {
                    "criterion": k.describe(),
                    "backtest": "not evaluated",
                    "note": f"no saa_plus path at {f.x_ref:.1%}",
                }
            )
            continue
        r = kill_triggers(port, saa, k, ctx.spec.data.frequency)
        if "note" in r:
            rows.append({"criterion": k.describe(), "backtest": "not evaluated", "note": r["note"]})
            continue
        worst = f"{r['worst']:.2%}"
        rows.append(
            {
                "criterion": k.describe(),
                "backtest": f"fired in {r['n_triggered']} of {r['n_windows']} windows",
                "note": (f"first {r['first']}; " if r["first"] else "") + f"worst {worst}",
            }
        )
    return lines + ["Structured criteria replayed on the walk-forward path of the proposal "
                    "(rolling windows, net returns):", "",
                    md_table(pd.DataFrame(rows)), ""]  # fmt: skip


def _robustness(ctx: _Context, f: Facts) -> list[str]:
    lines = ["## Robustness", ""]
    if f.corridor_live is not None and f.corridor_full is not None:
        lines.append(
            f"- Live-only history: corridor median {f.corridor_live['median']:.1%} "
            f"(P25–P75 {f.corridor_live['p25']:.1%}–{f.corridor_live['p75']:.1%}) vs "
            f"full {f.corridor_full['median']:.1%}."
        )
    if f.agreement_max_diff is not None:
        lines.append(
            f"- Riskfolio-Lib vs skfolio: largest candidate-weight difference "
            f"{f.agreement_max_diff:.2%}, {f.agreement_mismatches} status mismatches."
        )
    if f.failed_share is not None:
        lines.append(
            f"- Failed cells (full variant): {f.failed_share:.0%}, counted in the "
            "corridor's n_cells."
        )
    return lines + ([""] if len(lines) > 2 else ["_Nothing further._", ""])

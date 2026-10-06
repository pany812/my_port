"""Read-only workbench UI (P2-M7): browse experiments in the registry.

Start with ``wb ui`` (binds to localhost, usage statistics off). The registry URL comes from
``$WB_REGISTRY`` and is opened read-only: the database refuses writes. Every number comes from the
same functions as ``summary.md`` and ``memo.md``; this module only lays them out.
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from workbench.evaluation.agreement import agreement_summary, paired_cells
from workbench.evaluation.corridor import GROUP_KEYS, corridor
from workbench.evaluation.memo import FLAG, build_memo
from workbench.evaluation.oos import oos_table
from workbench.evaluation.report import build_report, latest
from workbench.evaluation.stress import bootstrap_table, path_window_view, window_table
from workbench.evaluation.views import (
    breakeven_view,
    risk_budget_view,
    sweep_view,
    weight_by_view,
)
from workbench.grid.spec import parse_spec
from workbench.registry.store import Registry, RegistrySchemaError
from workbench.ui import data as D

DEFAULT_REGISTRY = "sqlite:///out/registry.db"
PAGES = ("Experiments", "Corridor", "Cells", "Paths", "Evidence", "Libraries", "Memo")
PCT = st.column_config.NumberColumn(format="percent")
BAND_COLOR = "#4c78a8"


# --- cached access (experiments are immutable once stored) -----------------------------------


@st.cache_resource(show_spinner=False)
def registry(url: str) -> Registry:
    return Registry(url, read_only=True)


@st.cache_data(show_spinner=False)
def experiments(url: str) -> pd.DataFrame:
    return D.experiments_table(registry(url))


@st.cache_data(show_spinner=False)
def corridor_of(url: str, eid: str, by: tuple[str, ...] = ()) -> pd.DataFrame:
    return corridor(registry(url), eid, by=list(by))


@st.cache_data(show_spinner=False)
def cells_of(url: str, eid: str) -> pd.DataFrame:
    return D.cells_table(registry(url), eid)


@st.cache_data(show_spinner=False)
def oos_of(url: str, eid: str) -> pd.DataFrame:
    return oos_table(registry(url), eid)


@st.cache_data(show_spinner=False)
def labels_of(url: str, eid: str) -> dict[str, str]:
    return D.config_labels(registry(url), eid)


@st.cache_data(show_spinner=False)
def spec_yaml(url: str, eid: str) -> str:
    return registry(url).experiment(eid)["spec_yaml"]


@st.cache_data(show_spinner=False)
def memo_of(url: str, eid: str) -> tuple[str, pd.DataFrame, str]:
    m = build_memo(registry(url), eid)
    return m.markdown, m.checks, m.status


@st.cache_data(show_spinner=False)
def summary_of(url: str, eid: str) -> str:
    return build_report(registry(url), eid).summary_md


def _pct(df: pd.DataFrame) -> dict:
    """column_config showing every decimal-fraction column of ``df`` as a percentage."""
    keys = (
        "median",
        "p10",
        "p25",
        "p75",
        "p90",
        "weight",
        "share",
        "return",
        "vol",
        "dd",
        "te",
        "diff",
        "excess",
        "premium",
        "equilibrium",
        "posterior",
        "turnover",
        "x",
        "saa",
        "active",
        "target",
        "realised",
        "oos_",
    )
    skip = ("n_", "sr_", "dsr", "sharpe", "p_value", "p_bh", "confidence", "prior_sharpe")
    return {c: PCT for c in df.columns
            if pd.api.types.is_float_dtype(df[c]) and any(k in c for k in keys)
            and not c.startswith(skip) and c not in ("p", "p_bh")}  # fmt: skip


def table(df: pd.DataFrame, **kw) -> None:
    if df is None or df.empty:
        st.caption("(none)")
        return
    st.dataframe(df, hide_index=True, width="stretch", column_config=_pct(df), **kw)


# --- pages -------------------------------------------------------------------------------------


def page_experiments(url: str, eid: str) -> None:
    st.header("Experiments")
    table(experiments(url))
    st.subheader("Provenance")
    table(D.provenance(registry(url), eid))
    with st.expander("Spec (YAML, as run)"):
        st.code(spec_yaml(url, eid), language="yaml")


def page_corridor(url: str, eid: str) -> None:
    st.header("Allocation corridor")
    st.caption("Distribution over successful cells of the candidate's capital weight and risk "
               "share; failed cells are counted, never dropped.")  # fmt: skip
    corr = corridor_of(url, eid)
    c1, c2, c3 = st.columns(3)
    variant = c1.selectbox("Data variant", sorted(corr["data_variant"].unique()), key="c_var")
    measure = c2.selectbox("Measure", list(dict.fromkeys(corr["measure"])), key="c_measure")
    group = c3.selectbox("Group by", ["(none)", *GROUP_KEYS], key="c_group")
    last = latest(corr)
    st.subheader("Latest rebalance date")
    table(last[(last["data_variant"] == variant) & (last["measure"] == measure)])
    if group == "(none)":
        band = D.corridor_band(corr, measure, variant)
        if len(band) > 1:
            st.subheader("Through time")
            st.altair_chart(_band_chart(band, measure), width="stretch")
            st.caption("Line: median. Dark band: P25–P75. Light band: P10–P90. Below P10 the "
                       "background shows.")  # fmt: skip
    else:
        g = corridor_of(url, eid, (group,))
        g = g[(g["data_variant"] == variant) & (g["measure"] == measure)]
        st.subheader(f"Latest date by {group}")
        table(latest(g).drop(columns=["data_variant", "measure"]))
        if g["window_end"].nunique() > 1:
            st.subheader(f"Median through time by {group}")
            lines = g.assign(window_end=pd.to_datetime(g["window_end"]))
            chart = alt.Chart(lines).mark_line().encode(
                x=alt.X("window_end:T", title="rebalance date"),
                y=alt.Y("median:Q", title=measure, axis=alt.Axis(format="%")),
                color=alt.Color(f"{group}:N"))  # fmt: skip
            st.altair_chart(chart, width="stretch")


def _band_chart(band: pd.DataFrame, measure: str) -> alt.LayerChart:
    base = alt.Chart(band).encode(x=alt.X("window_end:T", title="rebalance date"))
    y = alt.Axis(format="%")
    outer = base.mark_area(color=BAND_COLOR, opacity=0.2).encode(
        y=alt.Y("p10:Q", title=f"{measure} (median, P25–P75, P10–P90)", axis=y), y2="p90:Q"
    )
    inner = base.mark_area(color=BAND_COLOR, opacity=0.45).encode(
        y=alt.Y("p25:Q", axis=y), y2="p75:Q"
    )
    line = base.mark_line(color=BAND_COLOR, strokeWidth=2).encode(
        y=alt.Y("median:Q", axis=y), tooltip=["window_end:T", "median:Q", "n_ok:Q", "n_cells:Q"]
    )
    return outer + inner + line  # fmt: skip


def page_cells(url: str, eid: str) -> None:
    st.header("Cells")
    cells = cells_of(url, eid)
    c1, c2, c3, c4 = st.columns(4)
    variant = c1.selectbox("Data variant", sorted(cells["data_variant"].unique()), key="x_var")
    dates = sorted(cells["window_end"].unique(), reverse=True)
    date = c2.selectbox("Rebalance date", dates, key="x_date")
    sets = c3.multiselect("Constraint sets", sorted(cells["constraint_set"].unique()), key="x_cs")
    status = c4.multiselect("Status", sorted(cells["status"].unique()), key="x_status")
    view = cells[(cells["data_variant"] == variant) & (cells["window_end"] == date)]
    if sets:
        view = view[view["constraint_set"].isin(sets)]
    if status:
        view = view[view["status"].isin(status)]
    table(view.drop(columns=["data_variant", "window_end"]))
    if view.empty:
        return
    options = list(view["cell_id"])
    names = dict(zip(view["cell_id"], view["label"] + " | " + view["constraint_set"] + " | "
                     + view["status"], strict=True))  # fmt: skip
    cell_id = st.selectbox("Cell", options, format_func=names.get, key="cell")
    d = D.cell_detail(registry(url), eid, cell_id)
    st.subheader(names[cell_id])
    left, right = st.columns([3, 2])
    w = d["weights"]
    if w["weight"].notna().any():
        long = w.melt(id_vars="asset", value_vars=["weight", "saa"], var_name="series")
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("asset:N", sort=list(w["asset"]), title=None),
            xOffset="series:N", y=alt.Y("value:Q", axis=alt.Axis(format="%"), title="weight"),
            color=alt.Color("series:N", legend=alt.Legend(title=None)),
            tooltip=["asset", "series", alt.Tooltip("value:Q", format=".2%")])  # fmt: skip
        left.altair_chart(chart, width="stretch")
        with left:
            table(w)
    else:
        left.warning(f"No weights: status {d['row']['status']}.")
    if d["message"]:
        right.code(d["message"], language=None)
    with right:
        st.markdown("**Parameters**")
        st.json({"allocator": d["row"]["allocator"], **d["params"], "estimator": d["estimator"]})
        st.markdown("**Metrics**")
        table(d["metrics"])
        with st.expander("Diagnostics"):
            st.json(d["diagnostics"])


def page_paths(url: str, eid: str) -> None:
    st.header("Out-of-sample paths")
    oos = oos_of(url, eid)
    if oos.empty:
        st.info("In-sample experiment: no walk-forward paths.")
        return
    labels = labels_of(url, eid)
    variant = st.selectbox("Data variant", sorted(oos["data_variant"].unique()), key="p_var")
    o = oos[oos["data_variant"] == variant]
    ids = [c for c in o["config_id"] if c in labels]
    chosen = st.multiselect("Configurations (vs the SAA)", ids, default=ids[:3],
                            format_func=labels.get, key="p_configs")  # fmt: skip
    wealth, dd = D.path_frames(registry(url), eid, variant, chosen, labels)
    c1, c2 = st.columns(2)
    c1.markdown("**Wealth (net)**")
    c1.line_chart(wealth, width="stretch")
    c2.markdown("**Drawdown**")
    c2.line_chart(-dd, width="stretch")
    st.subheader("Out-of-sample statistics vs the SAA (net)")
    table(o.drop(columns=["data_variant", "config_id"]))


def page_evidence(url: str, eid: str) -> None:
    st.header("Evidence")
    st.caption("Disclosures, not gates.")
    reg = registry(url)
    spec = parse_spec(spec_yaml(url, eid))
    tabs = st.tabs(["Net of search", "Candidate", "Risk budgets & sweeps", "Black–Litterman",
                    "Stress"])  # fmt: skip
    with tabs[0]:
        st.markdown("**Sharpe difference vs the SAA (BH-adjusted) and deflated Sharpe**")
        table(D.sharpe_table(reg, eid))
        st.markdown("**Probability of backtest overfitting**")
        table(D.evidence_rows(reg, eid, ("pbo",)))
    with tabs[1]:
        st.markdown("**Standalone profiles**")
        table(D.evidence_rows(reg, eid, ("profile",)))
        st.markdown("**Spanning**")
        table(D.evidence_rows(reg, eid, ("spanning_alpha", "spanning_alpha_robust", "spanning_hk",
                                         "spanning_kz_f1", "spanning_kz_f2",
                                         "spanning_hk_robust")))  # fmt: skip
    with tabs[2]:
        st.markdown("**Risk budgets**")
        table(risk_budget_view(reg, eid))
        st.markdown("**Constraint sweeps**")
        table(sweep_view(reg, eid))
    with tabs[3]:
        st.markdown("**Breakeven per target weight**")
        table(breakeven_view(reg, eid))
        st.markdown("**Weight at a stated view**")
        table(weight_by_view(reg, eid))
    with tabs[4]:
        if spec.stress is None:
            st.caption("No stress section in this experiment's spec.")
        else:
            ev = reg.evidence(eid)
            st.markdown("**Crisis windows: SAA plus the candidate**")
            table(window_table(ev, [n for n, _, _ in spec.stress.windows]))
            st.markdown("**Crisis windows: walk-forward paths**")
            table(path_window_view(reg, eid, spec.stress.windows))
            st.markdown("**Block-bootstrap paths**")
            table(bootstrap_table(ev))


def page_libraries(url: str, eid: str) -> None:
    st.header("Library agreement")
    pairs = paired_cells(registry(url), eid)
    if pairs.empty:
        st.info("Single-library experiment: nothing to compare.")
        return
    table(agreement_summary(pairs))
    both = pairs.dropna(subset=["w_riskfolio", "w_skfolio"])
    if not both.empty:
        pts = alt.Chart(both).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X("w_riskfolio:Q", axis=alt.Axis(format="%"), title="Riskfolio-Lib"),
            y=alt.Y("w_skfolio:Q", axis=alt.Axis(format="%"), title="skfolio"),
            color="family:N", tooltip=["label", "constraint_set", "window_end",
                                       alt.Tooltip("abs_diff:Q", format=".3%")])  # fmt: skip
        hi = float(max(both["w_riskfolio"].max(), both["w_skfolio"].max()))
        diag = alt.Chart(pd.DataFrame({"x": [0.0, hi], "y": [0.0, hi]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")  # fmt: skip
        st.markdown("**Candidate weight, paired cells**")
        st.altair_chart(diag + pts, width="stretch")


def page_memo(url: str, eid: str) -> None:
    md, checks, status = memo_of(url, eid)
    name = registry(url).experiment(eid)["name"]
    c1, c2, c3 = st.columns([2, 1, 1])
    n_flag = int((checks["flag"] == FLAG).sum())
    c1.metric("Memo status", status)
    c1.caption(f"{n_flag} of {len(checks)} checks flagged (information, not gates)")
    c2.download_button("memo.md", md, file_name=f"{name}_memo.md", key="dl_memo")
    c3.download_button("summary.md", summary_of(url, eid), file_name=f"{name}_summary.md",
                       key="dl_summary")  # fmt: skip
    st.markdown(md)


# --- main --------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Portfolio workbench", layout="wide")
    url = os.environ.get("WB_REGISTRY", DEFAULT_REGISTRY)
    st.sidebar.title("Portfolio workbench")
    st.sidebar.caption(f"Registry (read-only): `{url}`")
    try:
        registry(url)
    except (RegistrySchemaError, FileNotFoundError, ValueError) as e:
        st.error(f"Cannot open the registry: {e}")
        return
    exps = experiments(url)
    if exps.empty:
        st.info("No experiments yet: run `wb run specs/<name>.yaml`.")
        return
    names = {r.experiment_id: f"{r.name} · {r.experiment_id[:8]} · {r.created_at:%Y-%m-%d}"
             for r in exps.itertuples()}  # fmt: skip
    eid = st.sidebar.selectbox("Experiment", list(names), format_func=names.get, key="experiment")
    page = st.sidebar.radio("Page", PAGES, key="page")
    {"Experiments": page_experiments, "Corridor": page_corridor, "Cells": page_cells,
     "Paths": page_paths, "Evidence": page_evidence, "Libraries": page_libraries,
     "Memo": page_memo}[page](url, eid)  # fmt: skip


main()

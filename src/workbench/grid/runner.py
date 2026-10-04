"""Sequential grid runner with failure capture.

Every grid cell ends up in the registry with a status; nothing is dropped.

- ``backtest.mode: in_sample``: one fit per data variant at the end of the data.
- ``backtest.mode: walk_forward``: one fit per rebalance date (each a cell, ``window_end`` = the
  rebalance date) plus an out-of-sample path per configuration and for the SAA reference.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import riskfolio

from workbench.allocators.base import AllocationResult, FitContext
from workbench.allocators.factory import build
from workbench.backtest.schedule import rebalance_dates, window_at
from workbench.backtest.walkforward import PathResult, WalkForwardEngine
from workbench.data.align import align_history, data_vintage
from workbench.data.base import MarketData
from workbench.data.loaders import load_market
from workbench.evaluation.evidence import path_evidence, spanning_evidence
from workbench.evaluation.metrics import cell_metrics
from workbench.grid.expand import CellConfig, expand
from workbench.grid.spec import ExperimentSpec, WindowSpec
from workbench.policy.compiled import CompiledPolicy
from workbench.policy.compiler import compile_policy
from workbench.policy.saa import SAA
from workbench.registry.store import REFERENCE_ALLOCATOR, CellRecord, Registry

log = logging.getLogger(__name__)

IfExists = Literal["skip", "replace", "error"]


@dataclass(frozen=True)
class RunSummary:
    experiment_id: str
    spec_hash: str
    data_vintage: str
    n_cells: int
    status_counts: dict[str, int]
    skipped: bool = False


@dataclass
class _Context:
    """Per-experiment constants shared by every cell."""

    exp_id: str
    spec: ExperimentSpec
    saa: SAA
    policies: dict[str, CompiledPolicy]


def experiment_id_for(spec_hash: str, vintage: str, riskfolio_version: str) -> str:
    """Deterministic id: same spec + data vintage + Riskfolio-Lib version => same id."""
    key = f"{spec_hash}|{vintage}|{riskfolio_version}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def cell_id_for(experiment_id: str, config_id: str, variant: str, window_end: dt.date) -> str:
    return f"{experiment_id}:{config_id}:{variant}:{window_end.isoformat()}"


def select_window(returns: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    """Estimation window ending at the last observation. Raises if history is too short."""
    return window_at(returns, returns.index[-1], window)


def data_variants(data: MarketData) -> dict[str, MarketData]:
    """ "full" always; "live_only" too when the candidate has backfilled observations."""
    variants = {"full": data}
    if data.backfilled.any():
        variants["live_only"] = data.live_only()
    return variants


def run_experiment(
    spec: ExperimentSpec, registry: Registry, if_exists: IfExists = "skip"
) -> RunSummary:
    """Expand the grid, fit every cell and write the experiment to ``registry``."""
    data = align_history(load_market(spec.data, spec.seed))
    vintage = data_vintage(data)
    rf_version = riskfolio.__version__
    exp_id = experiment_id_for(spec.spec_hash, vintage, rf_version)

    if registry.has_experiment(exp_id):
        if if_exists == "skip":
            log.info("experiment %s already in registry; skipping", exp_id)
            counts = registry.cells(exp_id)["status"].value_counts().to_dict()
            return RunSummary(exp_id, spec.spec_hash, vintage, sum(counts.values()), counts, True)
        if if_exists == "error":
            raise ValueError(f"experiment {exp_id} already exists")
        registry.delete_experiment(exp_id)

    saa = SAA.from_version(spec.saa_version, spec.data.candidate)
    if set(saa.assets) != set(data.assets):
        raise ValueError(f"SAA assets {saa.assets} do not match data assets {data.assets}")
    policies = {
        cs.name: compile_policy(saa, cs, spec.data.frequency, spec.rf_annual, spec.solvers)
        for cs in spec.constraint_sets
    }
    ctx = _Context(exp_id, spec, saa, policies)
    configs = expand(spec)

    records: list[CellRecord] = []
    oos_rows: list[dict] = []
    evidence_rows: list[dict] = []
    t0 = time.perf_counter()
    freq = spec.data.frequency
    for variant, vdata in data_variants(data).items():
        returns = vdata.returns[saa.assets]
        if spec.backtest.mode == "in_sample":
            records += _in_sample_variant(ctx, configs, variant, returns)
            ev = []
        else:
            recs, rows, paths, saa_path = _walk_forward_variant(ctx, configs, variant, returns)
            records += recs
            oos_rows += rows
            rl = _riskless(saa)
            rf = returns[rl] if rl is not None else ctx.policies[spec.constraint_sets[0].name].rf
            ev = (path_evidence(paths, saa_path, freq, spec.seed, riskless=rf)
                  if saa_path is not None else [])  # fmt: skip
        ev += spanning_evidence(returns, saa.candidate, freq, _riskless(saa))
        evidence_rows += [{"data_variant": variant, **r} for r in ev]
        log.info("variant %s done (%.1fs)", variant, time.perf_counter() - t0)

    registry.write_experiment(
        {
            "experiment_id": exp_id,
            "name": spec.experiment,
            "spec_yaml": spec.yaml_text,
            "spec_hash": spec.spec_hash,
            "data_vintage": vintage,
            "saa_version": spec.saa_version,
            "candidate_id": spec.data.candidate,
            "riskfolio_version": rf_version,
            "seed": spec.seed,
            "created_at": dt.datetime.now(dt.UTC),
        },
        records,
        oos_rows,
        evidence_rows,
    )
    grid = [r for r in records if r.allocator != REFERENCE_ALLOCATOR]
    counts = pd.Series([r.status for r in grid]).value_counts().to_dict()
    log.info("experiment %s: %d cells in %.1fs %s", exp_id, len(grid),
             time.perf_counter() - t0, counts)  # fmt: skip
    return RunSummary(exp_id, spec.spec_hash, vintage, len(grid), counts)


# --- in-sample -------------------------------------------------------------------------


def _in_sample_variant(
    ctx: _Context, configs: list[CellConfig], variant: str, returns: pd.DataFrame
) -> list[CellRecord]:
    try:
        window = select_window(returns, ctx.spec.window)
    except ValueError as e:
        end = returns.index[-1]
        return [_failed_cell(ctx, cfg, variant, end, str(e)) for cfg in configs]
    as_of = window.index[-1]
    out = []
    for cfg in configs:
        try:
            allocator = build(cfg.allocator, cfg.params, cfg.estimator)
        except Exception as e:  # construction failures are cells too
            out.append(_failed_cell(ctx, cfg, variant, as_of, f"{type(e).__name__}: {e}"))
            continue
        res = allocator.fit(window, _fit_context(ctx, cfg, as_of))
        out.append(_cell_from_result(ctx, cfg, variant, as_of, window, res))
    out.append(_reference_cell(ctx, variant, window))
    return out


# --- walk-forward ----------------------------------------------------------------------


def _walk_forward_variant(
    ctx: _Context, configs: list[CellConfig], variant: str, returns: pd.DataFrame
) -> tuple[list[CellRecord], list[dict], dict[str, pd.Series], pd.Series | None]:
    """Cells, OOS path rows, OOS paths by config_id and the SAA path (None if no schedule)."""
    spec = ctx.spec
    schedule = rebalance_dates(returns.index, spec.window, spec.rebalance.every,
                               spec.data.frequency, spec.backtest.start)  # fmt: skip
    if not schedule:
        msg = (f"insufficient history: no rebalance date with a full window "
               f"({len(returns)} periods)")  # fmt: skip
        end = returns.index[-1]
        return [_failed_cell(ctx, cfg, variant, end, msg) for cfg in configs], [], {}, None

    engine = WalkForwardEngine(spec.window)
    records: list[CellRecord] = []
    rows: list[dict] = []
    paths: dict[str, pd.Series] = {}
    for cfg in configs:
        try:
            allocator = build(cfg.allocator, cfg.params, cfg.estimator)
            fit_fn = _fit_fn(ctx, cfg, allocator)
        except Exception as e:  # construction failure: every rebalance fails, SAA is held
            err = f"{type(e).__name__}: {e}"
            fit_fn = _failing_fit_fn(err)
        path = engine.run(returns, schedule, fit_fn, fallback=ctx.saa.weights)
        for f in path.fits:
            records.append(_cell_from_result(ctx, cfg, variant, f.as_of, f.window, f.result))
        rows += _path_rows(cfg.config_id, variant, path)
        paths[cfg.config_id] = path.returns

    ref_path = engine.run(returns, schedule, _saa_fit_fn(ctx), fallback=ctx.saa.weights)
    for f in ref_path.fits:
        records.append(_reference_cell(ctx, variant, f.window))
    rows += _path_rows(REFERENCE_ALLOCATOR, variant, ref_path)
    return records, rows, paths, ref_path.returns


def _fit_fn(ctx: _Context, cfg: CellConfig, allocator):
    def fit(window: pd.DataFrame, as_of: pd.Timestamp) -> AllocationResult:
        return allocator.fit(window, _fit_context(ctx, cfg, as_of))

    return fit


def _failing_fit_fn(message: str):
    def fit(window: pd.DataFrame, as_of: pd.Timestamp) -> AllocationResult:
        return AllocationResult(None, "exception", message)

    return fit


def _saa_fit_fn(ctx: _Context):
    def fit(window: pd.DataFrame, as_of: pd.Timestamp) -> AllocationResult:
        return AllocationResult(ctx.saa.weights.reindex(window.columns), "ok")

    return fit


def _path_rows(config_id: str, variant: str, path: PathResult) -> list[dict]:
    return [
        {"config_id": config_id, "data_variant": variant, "date": d.date(),
         "portfolio_return": float(r), "turnover": float(t)}
        for d, r, t in zip(path.returns.index, path.returns, path.turnover, strict=True)
    ]  # fmt: skip


# --- cells -----------------------------------------------------------------------------


def _fit_context(ctx: _Context, cfg: CellConfig, as_of: pd.Timestamp) -> FitContext:
    return FitContext(as_of=as_of, saa=ctx.saa.weights, candidate=ctx.saa.candidate,
                      policy=ctx.policies[cfg.constraint_set])  # fmt: skip


def _blank_cell(ctx: _Context, cfg: CellConfig, variant: str, as_of: pd.Timestamp) -> CellRecord:
    return CellRecord(
        cell_id=cell_id_for(ctx.exp_id, cfg.config_id, variant, as_of.date()),
        cell_index=cfg.index,
        config_id=cfg.config_id,
        allocator=cfg.allocator,
        params=cfg.params,
        estimator=cfg.estimator,
        constraint_set=cfg.constraint_set,
        data_variant=variant,
        window_end=as_of.date(),
        status="exception",
    )


def _failed_cell(
    ctx: _Context, cfg: CellConfig, variant: str, as_of: pd.Timestamp, message: str
) -> CellRecord:
    rec = _blank_cell(ctx, cfg, variant, as_of)
    rec.message = message
    return rec


def _cell_from_result(
    ctx: _Context,
    cfg: CellConfig,
    variant: str,
    as_of: pd.Timestamp,
    window: pd.DataFrame,
    res: AllocationResult,
) -> CellRecord:
    rec = _blank_cell(ctx, cfg, variant, as_of)
    rec.status, rec.message, rec.elapsed_s = res.status, res.message, res.elapsed_s
    rec.diagnostics = dict(res.diagnostics)
    if res.weights is not None:
        rec.weights = {k: float(v) for k, v in res.weights.items()}
        _add_metrics(rec, window, ctx.saa, ctx.spec)
    return rec


def _add_metrics(rec: CellRecord, window: pd.DataFrame, saa: SAA, spec: ExperimentSpec) -> None:
    """Attach evaluation metrics to an ok cell; failures go to diagnostics, status stays ok."""
    w = pd.Series(rec.weights)
    rec.metrics, errors = cell_metrics(
        w, window, saa.weights, saa.candidate, spec.risk_lenses, spec.data.frequency
    )
    if errors:
        rec.diagnostics = {**rec.diagnostics, "metric_errors": errors}


def _reference_cell(ctx: _Context, variant: str, window: pd.DataFrame) -> CellRecord:
    """The SAA benchmark row for one (variant, window end). Not a grid cell."""
    window_end = window.index[-1].date()
    rec = CellRecord(
        cell_id=cell_id_for(ctx.exp_id, REFERENCE_ALLOCATOR, variant, window_end),
        cell_index=-1,
        config_id=REFERENCE_ALLOCATOR,
        allocator=REFERENCE_ALLOCATOR,
        params={},
        estimator=None,
        constraint_set="",
        data_variant=variant,
        window_end=window_end,
        status="ok",
        weights={k: float(v) for k, v in ctx.saa.weights.items()},
    )
    _add_metrics(rec, window, ctx.saa, ctx.spec)
    return rec


def _riskless(saa: SAA) -> str | None:
    """The SAA's single cash-class asset (riskless proxy for spanning tests), else None."""
    cash = [a for a in saa.assets if saa.asset_class[a] == "cash" and a != saa.candidate]
    return cash[0] if len(cash) == 1 else None

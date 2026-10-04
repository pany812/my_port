"""Sequential grid runner with failure capture.

Every grid cell ends up in the registry with a status; nothing is dropped. M3 fits once per
data variant, at the end of the data, on the spec's window. M5 adds walk-forward window ends.
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

from workbench.allocators.base import FitContext
from workbench.allocators.factory import build
from workbench.data.align import align_history, data_vintage
from workbench.data.base import MarketData
from workbench.data.loaders import load_market
from workbench.evaluation.metrics import cell_metrics
from workbench.grid.expand import CellConfig, expand
from workbench.grid.spec import ExperimentSpec, WindowSpec
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


def experiment_id_for(spec_hash: str, vintage: str, riskfolio_version: str) -> str:
    """Deterministic id: same spec + data vintage + Riskfolio-Lib version => same id."""
    key = f"{spec_hash}|{vintage}|{riskfolio_version}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def cell_id_for(experiment_id: str, config_id: str, variant: str, window_end: dt.date) -> str:
    return f"{experiment_id}:{config_id}:{variant}:{window_end.isoformat()}"


def select_window(returns: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    """Estimation window ending at the last observation. Raises if history is too short."""
    if window.kind == "rolling":
        if len(returns) < window.periods:
            raise ValueError(f"insufficient history: {len(returns)} < {window.periods} periods")
        return returns.iloc[-window.periods :]
    if len(returns) < window.min_periods:
        raise ValueError(f"insufficient history: {len(returns)} < {window.min_periods} periods")
    return returns


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
    configs = expand(spec)

    records: list[CellRecord] = []
    t0 = time.perf_counter()
    for variant, vdata in data_variants(data).items():
        returns = vdata.returns[saa.assets]
        try:
            window, window_error = select_window(returns, spec.window), None
        except ValueError as e:
            window, window_error = None, str(e)
        window_end = (returns.index[-1] if window is None else window.index[-1]).date()
        for cfg in configs:
            rec = _run_cell(
                cfg, exp_id, variant, window, window_end, window_error,
                saa, policies[cfg.constraint_set],
            )  # fmt: skip
            if rec.status == "ok":
                _add_metrics(rec, window, saa, spec)
            records.append(rec)
        if window is not None:
            records.append(_reference_cell(exp_id, variant, window, saa, spec))
        log.info("variant %s: %d cells fitted", variant, len(configs))

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
    )
    grid = [r for r in records if r.allocator != REFERENCE_ALLOCATOR]
    counts = pd.Series([r.status for r in grid]).value_counts().to_dict()
    log.info("experiment %s: %d cells in %.1fs %s", exp_id, len(grid),
             time.perf_counter() - t0, counts)  # fmt: skip
    return RunSummary(exp_id, spec.spec_hash, vintage, len(grid), counts)


def _run_cell(
    cfg: CellConfig,
    exp_id: str,
    variant: str,
    window: pd.DataFrame | None,
    window_end: dt.date,
    window_error: str | None,
    saa: SAA,
    policy,
) -> CellRecord:
    rec = CellRecord(
        cell_id=cell_id_for(exp_id, cfg.config_id, variant, window_end),
        cell_index=cfg.index,
        config_id=cfg.config_id,
        allocator=cfg.allocator,
        params=cfg.params,
        estimator=cfg.estimator,
        constraint_set=cfg.constraint_set,
        data_variant=variant,
        window_end=window_end,
        status="exception",
    )
    if window_error is not None:
        rec.message = window_error
        return rec
    try:
        allocator = build(cfg.allocator, cfg.params, cfg.estimator)
        ctx = FitContext(as_of=window.index[-1], saa=saa.weights, candidate=saa.candidate,
                         policy=policy)  # fmt: skip
    except Exception as e:  # construction failures are cells too
        rec.message = f"{type(e).__name__}: {e}"
        return rec
    res = allocator.fit(window, ctx)
    rec.status, rec.message, rec.elapsed_s = res.status, res.message, res.elapsed_s
    rec.diagnostics = res.diagnostics
    rec.weights = None if res.weights is None else {k: float(v) for k, v in res.weights.items()}
    return rec


def _add_metrics(rec: CellRecord, window: pd.DataFrame, saa: SAA, spec: ExperimentSpec) -> None:
    """Attach evaluation metrics to an ok cell; failures go to diagnostics, status stays ok."""
    w = pd.Series(rec.weights)
    rec.metrics, errors = cell_metrics(
        w, window, saa.weights, saa.candidate, spec.risk_lenses, spec.data.frequency
    )
    if errors:
        rec.diagnostics = {**rec.diagnostics, "metric_errors": errors}


def _reference_cell(
    exp_id: str, variant: str, window: pd.DataFrame, saa: SAA, spec: ExperimentSpec
) -> CellRecord:
    """The SAA benchmark row for one (variant, window end). Not a grid cell."""
    window_end = window.index[-1].date()
    rec = CellRecord(
        cell_id=cell_id_for(exp_id, REFERENCE_ALLOCATOR, variant, window_end),
        cell_index=-1,
        config_id=REFERENCE_ALLOCATOR,
        allocator=REFERENCE_ALLOCATOR,
        params={},
        estimator=None,
        constraint_set="",
        data_variant=variant,
        window_end=window_end,
        status="ok",
        weights={k: float(v) for k, v in saa.weights.items()},
    )
    _add_metrics(rec, window, saa, spec)
    return rec

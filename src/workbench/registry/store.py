"""Registry access: write a whole experiment in one transaction, read back as DataFrames."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, delete, event, insert, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from workbench.registry.models import Base, Cell, Evidence, Experiment, Metric, OosReturn, Weight

# Pseudo-allocator for the SAA benchmark row written once per (variant, window end). It is a
# cells row so its weights and metrics can be stored, but it is NOT a grid cell: every read
# below excludes it unless ``include_reference=True``. Corridor counts depend on this.
REFERENCE_ALLOCATOR = "saa_reference"


@dataclass
class CellRecord:
    """Everything stored for one cell. ``weights`` is None unless status is "ok".

    metrics: (metric, lens, value) triples; lens is "" when not lens-specific.
    """

    cell_id: str
    cell_index: int
    config_id: str
    allocator: str
    params: dict
    estimator: dict | None
    constraint_set: str
    data_variant: str
    window_end: dt.date
    status: str
    message: str = ""
    elapsed_s: float = 0.0
    diagnostics: dict = field(default_factory=dict)
    weights: dict[str, float] | None = None
    metrics: list[tuple[str, str, float]] = field(default_factory=list)


class RegistrySchemaError(RuntimeError):
    """The registry database predates columns this version writes; run ``wb migrate``."""


# Values for columns added after a table first shipped, for rows written before them. Each is
# exact for older rows (e.g. experiments before P2-M2 had no costs, so net == gross).
BACKFILL: dict[tuple[str, str], str] = {
    ("oos_returns", "cost"): "0.0",
    ("oos_returns", "portfolio_return_net"): "portfolio_return",
    ("oos_returns", "liquidity_adjusted"): "0",
}


class Registry:
    """Experiment registry over any SQLAlchemy URL (``sqlite:///path.db``, ``postgresql://...``).

    Opening an existing database whose tables lack columns of the current models raises
    :class:`RegistrySchemaError` (pass ``check=False`` to open it for :meth:`migrate`).
    """

    def __init__(self, url: str, check: bool = True) -> None:
        self.url = url
        self.engine: Engine = create_engine(url)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _sqlite_foreign_keys)
        Base.metadata.create_all(self.engine)
        drift = self.schema_drift()
        if check and drift:
            missing = "; ".join(f"{t}: {', '.join(c)}" for t, c in drift.items())
            raise RegistrySchemaError(
                f"registry {url} predates this version (missing columns: {missing}). "
                f"Run: wb migrate --registry {url}"
            )

    def schema_drift(self) -> dict[str, list[str]]:
        """Columns defined in the models but missing from the database, per table."""
        insp = inspect(self.engine)
        drift = {}
        for table in Base.metadata.sorted_tables:
            have = {c["name"] for c in insp.get_columns(table.name)}
            missing = [c.name for c in table.columns if c.name not in have]
            if missing:
                drift[table.name] = missing
        return drift

    def migrate(self) -> list[str]:
        """Add missing columns (additive only) and backfill them. Returns the actions taken."""
        actions = []
        with self.engine.begin() as conn:
            for table_name, columns in self.schema_drift().items():
                table = Base.metadata.tables[table_name]
                for name in columns:
                    col = table.columns[name]
                    ddl_type = col.type.compile(dialect=self.engine.dialect)
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {name} {ddl_type}"))
                    fill = BACKFILL.get((table_name, name))
                    if fill is not None:
                        conn.execute(text(f"UPDATE {table_name} SET {name} = {fill}"))
                    actions.append(f"{table_name}.{name} added"
                                   + (f", backfilled = {fill}" if fill else ""))  # fmt: skip
        return actions

    # --- writes ----------------------------------------------------------------------

    def has_experiment(self, experiment_id: str) -> bool:
        with Session(self.engine) as s:
            return s.get(Experiment, experiment_id) is not None

    def delete_experiment(self, experiment_id: str) -> None:
        with Session(self.engine) as s, s.begin():
            cell_ids = select(Cell.cell_id).where(Cell.experiment_id == experiment_id)
            s.execute(delete(Weight).where(Weight.cell_id.in_(cell_ids)))
            s.execute(delete(Metric).where(Metric.cell_id.in_(cell_ids)))
            s.execute(delete(Cell).where(Cell.experiment_id == experiment_id))
            s.execute(delete(OosReturn).where(OosReturn.experiment_id == experiment_id))
            s.execute(delete(Evidence).where(Evidence.experiment_id == experiment_id))
            s.execute(delete(Experiment).where(Experiment.experiment_id == experiment_id))

    def write_experiment(
        self,
        experiment: dict,
        cells: Iterable[CellRecord],
        oos_rows: Iterable[dict] = (),
        evidence_rows: Iterable[dict] = (),
    ) -> None:
        """Insert the experiment, its cells, weights, metrics and OOS paths atomically.

        oos_rows: dicts with config_id, data_variant, date, portfolio_return, turnover.
        evidence_rows: dicts with data_variant, subject, test, statistic, p_value, extra_json.
        """
        cells = list(cells)
        cell_rows = [
            {
                "cell_id": c.cell_id,
                "experiment_id": experiment["experiment_id"],
                "cell_index": c.cell_index,
                "config_id": c.config_id,
                "allocator": c.allocator,
                "params_json": _dumps(c.params),
                "estimator_json": None if c.estimator is None else _dumps(c.estimator),
                "constraint_set": c.constraint_set,
                "data_variant": c.data_variant,
                "window_end": c.window_end,
                "status": c.status,
                "message": c.message,
                "elapsed_s": c.elapsed_s,
                "diagnostics_json": _dumps(c.diagnostics),
            }
            for c in cells
        ]
        weight_rows = [
            {"cell_id": c.cell_id, "asset_id": a, "weight": float(w)}
            for c in cells
            for a, w in (c.weights or {}).items()
        ]
        metric_rows = [
            {"cell_id": c.cell_id, "metric": m, "lens": lens, "value": float(v)}
            for c in cells
            for m, lens, v in c.metrics
        ]
        with Session(self.engine) as s, s.begin():
            s.execute(insert(Experiment), [experiment])
            if cell_rows:
                s.execute(insert(Cell), cell_rows)
            if weight_rows:
                s.execute(insert(Weight), weight_rows)
            if metric_rows:
                s.execute(insert(Metric), metric_rows)
            oos = [{"experiment_id": experiment["experiment_id"], **r} for r in oos_rows]
            if oos:
                s.execute(insert(OosReturn), oos)
            ev = [{"experiment_id": experiment["experiment_id"], **r} for r in evidence_rows]
            if ev:
                s.execute(insert(Evidence), ev)

    # --- reads -----------------------------------------------------------------------

    def experiments(self) -> pd.DataFrame:
        return self._frame(select(Experiment).order_by(Experiment.created_at))

    def experiment(self, experiment_id: str) -> dict:
        with Session(self.engine) as s:
            e = s.get(Experiment, experiment_id)
            if e is None:
                raise KeyError(f"no experiment {experiment_id!r}")
            return {c.name: getattr(e, c.name) for c in Experiment.__table__.columns}

    def resolve(self, name_or_id: str) -> str:
        """Experiment id for an id, or for the most recent experiment with that name."""
        with Session(self.engine) as s:
            if s.get(Experiment, name_or_id) is not None:
                return name_or_id
            q = (
                select(Experiment.experiment_id)
                .where(Experiment.name == name_or_id)
                .order_by(Experiment.created_at.desc())
                .limit(1)
            )
            found = s.execute(q).scalar_one_or_none()
        if found is None:
            raise KeyError(f"no experiment with id or name {name_or_id!r}")
        return found

    def cells(self, experiment_id: str, include_reference: bool = False) -> pd.DataFrame:
        """Grid cells of an experiment (the SAA reference row only if asked)."""
        q = select(Cell).where(Cell.experiment_id == experiment_id)
        q = _grid_only(q, include_reference)
        return self._frame(q.order_by(Cell.data_variant, Cell.window_end, Cell.cell_index))

    def reference(self, experiment_id: str) -> pd.DataFrame:
        """The SAA reference rows (one per data variant and window end)."""
        q = select(Cell).where(
            Cell.experiment_id == experiment_id, Cell.allocator == REFERENCE_ALLOCATOR
        )
        return self._frame(q.order_by(Cell.data_variant, Cell.window_end))

    def weights(self, experiment_id: str, include_reference: bool = False) -> pd.DataFrame:
        """Long format: cell_id, asset_id, weight."""
        q = (
            select(Weight)
            .join(Cell, Cell.cell_id == Weight.cell_id)
            .where(Cell.experiment_id == experiment_id)
        )
        q = _grid_only(q, include_reference)
        return self._frame(q.order_by(Weight.cell_id, Weight.asset_id))

    def weights_wide(self, experiment_id: str, include_reference: bool = False) -> pd.DataFrame:
        """One row per ok cell (index cell_id), one column per asset."""
        long = self.weights(experiment_id, include_reference)
        if long.empty:
            return pd.DataFrame()
        return long.pivot(index="cell_id", columns="asset_id", values="weight")

    def metrics(self, experiment_id: str, include_reference: bool = False) -> pd.DataFrame:
        """Long format: cell_id, metric, lens, value."""
        q = (
            select(Metric)
            .join(Cell, Cell.cell_id == Metric.cell_id)
            .where(Cell.experiment_id == experiment_id)
        )
        q = _grid_only(q, include_reference)
        return self._frame(q.order_by(Metric.cell_id, Metric.metric, Metric.lens))

    def oos_returns(self, experiment_id: str, include_reference: bool = False) -> pd.DataFrame:
        """Long format: config_id, data_variant, date, portfolio_return (gross), turnover, cost,
        portfolio_return_net, liquidity_adjusted."""
        q = select(OosReturn).where(OosReturn.experiment_id == experiment_id)
        if not include_reference:
            q = q.where(OosReturn.config_id != REFERENCE_ALLOCATOR)
        q = q.order_by(OosReturn.data_variant, OosReturn.config_id, OosReturn.date)
        return self._frame(q).drop(columns="experiment_id")

    def evidence(self, experiment_id: str) -> pd.DataFrame:
        """Long format: data_variant, subject, test, statistic, p_value, extra_json."""
        q = select(Evidence).where(Evidence.experiment_id == experiment_id)
        q = q.order_by(Evidence.data_variant, Evidence.subject, Evidence.test)
        return self._frame(q).drop(columns="experiment_id")

    def _frame(self, query) -> pd.DataFrame:
        with Session(self.engine) as s:
            rows = s.execute(query).scalars().all()
            if not rows:
                cols = [c.name for c in query.column_descriptions[0]["entity"].__table__.columns]
                return pd.DataFrame(columns=cols)
            table = type(rows[0]).__table__
            return pd.DataFrame([{c.name: getattr(r, c.name) for c in table.columns} for r in rows])


def loads_or_none(value) -> dict | None:
    """Parse a JSON column value; SQL NULL (None, or NaN after pandas 3 conversion) -> None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    return json.loads(value)


def _grid_only(query, include_reference: bool):
    return query if include_reference else query.where(Cell.allocator != REFERENCE_ALLOCATOR)


def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _sqlite_foreign_keys(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()

"""Registry access: write a whole experiment in one transaction, read back as DataFrames."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, delete, event, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from workbench.registry.models import Base, Cell, Experiment, Metric, Weight


@dataclass
class CellRecord:
    """Everything stored for one cell. ``weights`` is None unless status is "ok"."""

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


class Registry:
    """Experiment registry over any SQLAlchemy URL (``sqlite:///path.db``, ``postgresql://...``)."""

    def __init__(self, url: str) -> None:
        self.engine: Engine = create_engine(url)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _sqlite_foreign_keys)
        Base.metadata.create_all(self.engine)

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
            s.execute(delete(Experiment).where(Experiment.experiment_id == experiment_id))

    def write_experiment(self, experiment: dict, cells: Iterable[CellRecord]) -> None:
        """Insert the experiment row, its cells and their weights atomically (in FK order)."""
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
        with Session(self.engine) as s, s.begin():
            s.execute(insert(Experiment), [experiment])
            if cell_rows:
                s.execute(insert(Cell), cell_rows)
            if weight_rows:
                s.execute(insert(Weight), weight_rows)

    # --- reads -----------------------------------------------------------------------

    def experiments(self) -> pd.DataFrame:
        return self._frame(select(Experiment).order_by(Experiment.created_at))

    def experiment(self, experiment_id: str) -> dict:
        with Session(self.engine) as s:
            e = s.get(Experiment, experiment_id)
            if e is None:
                raise KeyError(f"no experiment {experiment_id!r}")
            return {c.name: getattr(e, c.name) for c in Experiment.__table__.columns}

    def cells(self, experiment_id: str) -> pd.DataFrame:
        q = select(Cell).where(Cell.experiment_id == experiment_id)
        return self._frame(q.order_by(Cell.data_variant, Cell.window_end, Cell.cell_index))

    def weights(self, experiment_id: str) -> pd.DataFrame:
        """Long format: cell_id, asset_id, weight."""
        q = (
            select(Weight)
            .join(Cell, Cell.cell_id == Weight.cell_id)
            .where(Cell.experiment_id == experiment_id)
            .order_by(Weight.cell_id, Weight.asset_id)
        )
        return self._frame(q)

    def weights_wide(self, experiment_id: str) -> pd.DataFrame:
        """One row per ok cell (index cell_id), one column per asset."""
        long = self.weights(experiment_id)
        if long.empty:
            return pd.DataFrame()
        return long.pivot(index="cell_id", columns="asset_id", values="weight")

    def _frame(self, query) -> pd.DataFrame:
        with Session(self.engine) as s:
            rows = s.execute(query).scalars().all()
            if not rows:
                cols = [c.name for c in query.column_descriptions[0]["entity"].__table__.columns]
                return pd.DataFrame(columns=cols)
            table = type(rows[0]).__table__
            return pd.DataFrame([{c.name: getattr(r, c.name) for c in table.columns} for r in rows])


def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _sqlite_foreign_keys(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()

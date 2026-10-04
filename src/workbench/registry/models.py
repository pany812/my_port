"""SQLAlchemy models. SQLite in tests, PostgreSQL in production.

Schema per docs/PHASE1.md plus four approved ``cells`` columns (``config_id``, ``cell_index``,
``data_variant``, ``diagnostics_json``) and the approved ``oos_returns`` table (M5).
Ask before changing it.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    spec_yaml: Mapped[str] = mapped_column(Text)
    spec_hash: Mapped[str] = mapped_column(String(64), index=True)
    data_vintage: Mapped[str] = mapped_column(String(64))
    saa_version: Mapped[str] = mapped_column(String(100))
    candidate_id: Mapped[str] = mapped_column(String(100))
    riskfolio_version: Mapped[str] = mapped_column(String(20))
    seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))


class Cell(Base):
    __tablename__ = "cells"

    cell_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.experiment_id", ondelete="CASCADE"), index=True
    )
    cell_index: Mapped[int] = mapped_column(Integer)
    config_id: Mapped[str] = mapped_column(String(16), index=True)
    allocator: Mapped[str] = mapped_column(String(100))
    params_json: Mapped[str] = mapped_column(Text)
    estimator_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    constraint_set: Mapped[str] = mapped_column(String(100))
    data_variant: Mapped[str] = mapped_column(String(20))
    window_end: Mapped[dt.date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), index=True)
    message: Mapped[str] = mapped_column(Text, default="")
    elapsed_s: Mapped[float] = mapped_column(Float, default=0.0)
    diagnostics_json: Mapped[str] = mapped_column(Text, default="{}")


class Weight(Base):
    __tablename__ = "weights"

    cell_id: Mapped[str] = mapped_column(
        ForeignKey("cells.cell_id", ondelete="CASCADE"), primary_key=True
    )
    asset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    weight: Mapped[float] = mapped_column(Float)


class Metric(Base):
    __tablename__ = "metrics"

    cell_id: Mapped[str] = mapped_column(
        ForeignKey("cells.cell_id", ondelete="CASCADE"), primary_key=True
    )
    metric: Mapped[str] = mapped_column(String(100), primary_key=True)
    lens: Mapped[str] = mapped_column(String(50), primary_key=True, default="")
    value: Mapped[float] = mapped_column(Float)


class OosReturn(Base):
    """Out-of-sample walk-forward path per configuration (and the SAA reference path)."""

    __tablename__ = "oos_returns"

    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.experiment_id", ondelete="CASCADE"), primary_key=True
    )
    config_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    data_variant: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    portfolio_return: Mapped[float] = mapped_column(Float)
    turnover: Mapped[float] = mapped_column(Float, default=0.0)

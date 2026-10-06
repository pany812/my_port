"""SQLAlchemy engines for the registry and the SQL data source.

Read-only mode is enforced by the database, not by convention: SQLite files open with
``mode=ro``; PostgreSQL sessions start with ``default_transaction_read_only=on`` (use a read-only
role as well).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def engine(url: str, read_only: bool = False) -> Engine:
    """An engine for ``url`` (any SQLAlchemy URL; read-only for SQLite files and PostgreSQL)."""
    if not read_only:
        return create_engine(url)
    if url.startswith("sqlite:///"):
        path = url.removeprefix("sqlite:///")
        if path in ("", ":memory:"):
            raise ValueError("an in-memory SQLite database cannot be opened read-only")
        if not Path(path).exists():
            raise FileNotFoundError(f"no database at {path}")
        return create_engine(f"sqlite:///file:{Path(path).resolve()}?mode=ro&uri=true")
    if url.startswith(("postgresql", "postgres")):
        return create_engine(url, connect_args={"options": "-c default_transaction_read_only=on"})
    raise ValueError(f"read-only mode is supported for SQLite and PostgreSQL, not {url}")


def redact(url: str) -> str:
    """``url`` with any password replaced by ``***`` (for messages and logs)."""
    from sqlalchemy.engine import make_url

    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return "<unparseable database URL>"

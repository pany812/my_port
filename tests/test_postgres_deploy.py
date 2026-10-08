"""The PostgreSQL deployment kit (sql/roles.sql, sql/contract_views.sql) on mock house tables.

Opt-in, like tests/test_postgres.py: WB_TEST_POSTGRES_URL must point to a throwaway ``wb_test*``
database as a superuser. The test creates schema ``house`` with the column names the view template
assumes, runs both SQL files unedited, then checks what each role can and cannot do.
"""

import os
import secrets
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
import yaml

from workbench.data.check import data_check
from workbench.data.demo import write_demo_contract
from workbench.data.sql import DataSourceError, load_sql
from workbench.grid.runner import run_experiment
from workbench.grid.spec import parse_spec
from workbench.policy.saa import SAA
from workbench.registry.store import Registry

URL = os.environ.get("WB_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="WB_TEST_POSTGRES_URL not set")
ROOT = Path(__file__).parents[1]
ROLES = ("wb_data_reader", "wb_writer", "wb_ui")
FIRST_LOAD = "2026-10-01 06:00:00+00"
REVISION = "2026-10-02 06:00:00+00"


def _role_url(role: str, password: str) -> str:
    u = sa.engine.make_url(URL)
    return u.set(username=role, password=password).render_as_string(hide_password=False)


def _run_sql_file(eng: sa.Engine, path: Path) -> None:
    raw = eng.raw_connection()
    try:
        raw.cursor().execute(path.read_text())  # several statements, no parameters
        raw.commit()
    finally:
        raw.close()


@pytest.fixture(scope="module")
def deployed():
    assert sa.engine.make_url(URL).database.startswith("wb_test"), "refusing a non-test database"
    admin = sa.create_engine(URL)
    with admin.begin() as c:
        c.execute(sa.text("DROP SCHEMA IF EXISTS house, workbench, workbench_data CASCADE"))
        for role in ROLES:
            if c.execute(sa.text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).first():
                c.execute(sa.text(f"DROP OWNED BY {role} CASCADE"))
                c.execute(sa.text(f"DROP ROLE {role}"))
        # mock house tables with the names contract_views.sql expects
        c.execute(sa.text("CREATE SCHEMA house"))
        c.execute(sa.text("CREATE TABLE house.series (series_id text PRIMARY KEY, currency text, "
                          "is_hedged boolean, name text)"))  # fmt: skip
        c.execute(
            sa.text(
                "CREATE TABLE house.monthly_returns (series_id text, period_end date, "
                "total_return_sek double precision, loaded_at timestamptz)"
            )
        )
    # the synthetic demo data, as if loaded into the house tables  # fmt: skip
    demo = sa.create_engine(f"sqlite:///{Path(tempfile.mkdtemp()) / 'demo.db'}")
    write_demo_contract(str(demo.url))
    assets = pd.read_sql("select * from wb_assets", demo)
    returns = pd.read_sql("select * from wb_returns", demo)
    with admin.begin() as c:
        c.execute(sa.text("INSERT INTO house.series VALUES (:s, :c, :h, :n)"),
                  [{"s": f"BBG_{r.asset_id}", "c": r.currency, "h": bool(r.hedged),
                    "n": r.description} for r in assets.itertuples()])  # fmt: skip
        c.execute(sa.text("INSERT INTO house.monthly_returns VALUES (:s, :d, :r, :t)"),
                  [{"s": f"BBG_{r.asset_id}", "d": r.period_end, "r": r.simple_return,
                    "t": FIRST_LOAD} for r in returns.itertuples()])  # fmt: skip
        c.execute(
            sa.text(
                "INSERT INTO house.monthly_returns VALUES ('BBG_GL_EQ', '2010-06-30', "
                f"0.0777, '{REVISION}')"
            )
        )  # a later revision of one month
    _run_sql_file(admin, ROOT / "sql" / "roles.sql")
    _run_sql_file(admin, ROOT / "sql" / "roles.sql")  # idempotent
    _run_sql_file(admin, ROOT / "sql" / "contract_views.sql")
    passwords = {r: secrets.token_hex(16) for r in ROLES}  # test-only credentials
    with admin.begin() as c:
        c.execute(sa.text("INSERT INTO workbench_data.asset_map VALUES (:a, :s, :k, NULL)"),
                  [{"a": r.asset_id, "s": f"BBG_{r.asset_id}", "k": r.kind}
                   for r in assets.itertuples()])  # fmt: skip
        for role, pw in passwords.items():
            c.execute(sa.text(f"ALTER ROLE {role} PASSWORD '{pw}'"))
    admin.dispose()
    return {r: _role_url(r, pw) for r, pw in passwords.items()}


def _sql_spec(**sql):
    raw = yaml.safe_load((ROOT / "specs" / "example_sql.yaml").read_text())
    raw["data"]["sql"] = {**raw["data"]["sql"], "schema": "workbench_data", **sql}
    return parse_spec(raw)


def test_data_reader_reads_the_contract_point_in_time(deployed):
    saa = SAA.from_version("synthetic_example")
    url = deployed["wb_data_reader"]
    latest = load_sql(_sql_spec().data, saa.assets, saa.asset_class, url=url)
    assert latest.vintage_tag == "2026-10-02T06:00:00"
    assert latest.market.returns.loc["2010-06-30", "GL_EQ"] == 0.0777
    first = load_sql(_sql_spec(vintage="2026-10-01T06:00:00").data, saa.assets,
                     saa.asset_class, url=url)  # fmt: skip
    assert first.market.returns.loc["2010-06-30", "GL_EQ"] != 0.0777
    assert first.market.backfilled.sum() == 15 * 12  # CAND_PROXY before 2016-01


def test_data_reader_sees_nothing_else(deployed):
    eng = sa.create_engine(deployed["wb_data_reader"])
    with eng.connect() as c:
        with pytest.raises(sa.exc.ProgrammingError, match="permission denied"):
            c.execute(sa.text("SELECT * FROM house.monthly_returns LIMIT 1"))
    with eng.connect() as c:
        with pytest.raises(sa.exc.DBAPIError, match="read-only|permission denied"):
            c.execute(sa.text("DELETE FROM workbench_data.asset_map"))
    eng.dispose()


def test_reconciliation_through_the_views(deployed, monkeypatch):
    monkeypatch.setenv("WB_DATA_URL", deployed["wb_data_reader"])
    raw = yaml.safe_load((ROOT / "specs" / "example_sql.yaml").read_text())
    raw["data"]["sql"].update(schema="workbench_data", vintage="2026-10-01T06:00:00")
    dc = data_check(parse_spec(raw))
    s = dc.summary.set_index("field")["value"]
    assert dc.ok and s["reconciliation"].startswith("ok: 26 years")


def test_writer_owns_the_registry_and_ui_only_reads(deployed):
    raw = yaml.safe_load((ROOT / "tests" / "golden" / "golden_spec.yaml").read_text())
    raw["backtest"]["mode"] = "in_sample"
    writer = Registry(deployed["wb_writer"])
    s = run_experiment(parse_spec(raw), writer)
    eng = sa.create_engine(URL)
    with eng.connect() as c:
        tables = set(
            c.execute(
                sa.text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'workbench'"
                )
            ).scalars()
        )
    eng.dispose()  # fmt: skip
    assert {"experiments", "cells", "weights", "evidence"} <= tables
    ui = Registry(deployed["wb_ui"])  # the role itself is read-only, even without the flag
    assert len(ui.cells(s.experiment_id)) > 0
    with pytest.raises(sa.exc.DBAPIError, match="read-only"):
        ui.delete_experiment(s.experiment_id)
    assert len(Registry(deployed["wb_ui"], read_only=True).cells(s.experiment_id)) > 0


def test_writer_cannot_read_house_tables(deployed):
    eng = sa.create_engine(deployed["wb_writer"])
    with eng.connect() as c, pytest.raises(sa.exc.ProgrammingError, match="permission denied"):
        c.execute(sa.text("SELECT * FROM house.series LIMIT 1"))
    eng.dispose()


def test_load_errors_name_the_role_not_the_password(deployed):
    saa = SAA.from_version("synthetic_example")
    bad = deployed["wb_data_reader"].replace("wb_test", "wb_test_missing")
    with pytest.raises(DataSourceError) as e:
        load_sql(_sql_spec().data, saa.assets, saa.asset_class, url=bad)
    pw = sa.engine.make_url(deployed["wb_data_reader"]).password
    assert pw not in str(e.value) and "***" in str(e.value)

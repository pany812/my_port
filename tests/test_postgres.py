"""PostgreSQL integration (opt-in): set WB_TEST_POSTGRES_URL to a throwaway database whose name
starts with ``wb_test`` (the suite drops and recreates the workbench tables there), e.g.

    docker run -d --rm --name wb-pg -e POSTGRES_PASSWORD=<test password> -e POSTGRES_DB=wb_test \\
        -p 55432:5432 postgres:17
    WB_TEST_POSTGRES_URL=postgresql+psycopg://postgres:<test password>@localhost:55432/wb_test \\
        uv run pytest tests/test_postgres.py
"""

import os

import pandas as pd
import pytest
import sqlalchemy as sa
import yaml

from tests.test_runner import SMALL
from workbench.data.demo import contract_tables, write_demo_contract
from workbench.data.sql import load_sql
from workbench.grid.runner import run_experiment
from workbench.grid.spec import parse_spec
from workbench.policy.saa import SAA
from workbench.registry.models import Base
from workbench.registry.store import Registry

URL = os.environ.get("WB_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="WB_TEST_POSTGRES_URL not set")
DATA_SCHEMA = "wb_data_test"


@pytest.fixture(scope="module")
def pg():
    assert sa.engine.make_url(URL).database.startswith("wb_test"), "refusing a non-test database"
    eng = sa.create_engine(URL)
    with eng.begin() as c:
        Base.metadata.drop_all(c)
        c.execute(sa.text(f"DROP SCHEMA IF EXISTS {DATA_SCHEMA} CASCADE"))
    eng.dispose()
    return URL


def _spec():
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"})  # fmt: skip
    return parse_spec(raw)


def _frames(reg: Registry, eid: str) -> dict:
    keys = ["cell_id"]
    return {
        "cells": reg.cells(eid).drop(columns=["elapsed_s"]).sort_values(keys)
        .reset_index(drop=True),
        "weights": reg.weights(eid).sort_values(["cell_id", "asset_id"]).reset_index(drop=True),
        "metrics": reg.metrics(eid).sort_values(["cell_id", "metric", "lens"])
        .reset_index(drop=True),
        "oos": reg.oos_returns(eid, include_reference=True)
        .sort_values(["config_id", "data_variant", "date"]).reset_index(drop=True),
        "evidence": reg.evidence(eid).sort_values(["data_variant", "subject", "test"])
        .reset_index(drop=True),
    }  # fmt: skip


@pytest.fixture(scope="module")
def runs(pg, tmp_path_factory):
    lite = Registry(f"sqlite:///{tmp_path_factory.mktemp('pg') / 'r.db'}")
    a = run_experiment(_spec(), lite)
    pgreg = Registry(pg)
    b = run_experiment(_spec(), pgreg)
    return lite, pgreg, a, b


def test_postgres_registry_matches_sqlite(runs):
    lite, pgreg, a, b = runs
    assert a.experiment_id == b.experiment_id
    fa, fb = _frames(lite, a.experiment_id), _frames(pgreg, b.experiment_id)
    for k in fa:
        x, y = fa[k], fb[k]
        for col in ("window_end", "date"):
            if col in x:
                x[col], y[col] = pd.to_datetime(x[col]), pd.to_datetime(y[col])
        pd.testing.assert_frame_equal(x, y, check_dtype=False, check_exact=True, obj=k)
    e = pgreg.experiment(b.experiment_id)
    assert e["data_source"] == "synthetic" and e["data_vintage_tag"] is None


def test_read_only_mode_on_postgres(runs, pg):
    _, _, _, b = runs
    ro = Registry(pg, read_only=True)
    assert len(ro.cells(b.experiment_id)) > 0
    with pytest.raises(sa.exc.DBAPIError, match="read-only"):
        ro.delete_experiment(b.experiment_id)


def test_migrate_on_postgres(runs, pg):
    _, _, _, b = runs
    eng = sa.create_engine(pg)
    with eng.begin() as c:
        c.execute(sa.text("ALTER TABLE experiments DROP COLUMN data_source"))
        c.execute(sa.text("ALTER TABLE oos_returns DROP COLUMN cost"))
    eng.dispose()
    actions = Registry(pg, check=False).migrate()
    assert len(actions) == 2
    reg = Registry(pg)
    assert reg.experiment(b.experiment_id)["data_source"] == "synthetic"
    assert (reg.oos_returns(b.experiment_id)["cost"] == 0.0).all()


def test_sql_source_on_postgres_with_a_schema(pg, tmp_path):
    lite = f"sqlite:///{tmp_path / 'demo.db'}"
    write_demo_contract(lite)
    src, dst = sa.create_engine(lite), sa.create_engine(pg)
    with dst.begin() as c:
        c.execute(sa.text(f"CREATE SCHEMA {DATA_SCHEMA}"))
        md = sa.MetaData()
        a_t, r_t = contract_tables(md, schema=DATA_SCHEMA)
        md.create_all(c)
        with src.connect() as s:
            c.execute(sa.insert(a_t), [dict(r) for r in s.execute(sa.text(
                "select * from wb_assets")).mappings()])  # fmt: skip
            c.execute(sa.insert(r_t), [dict(r) for r in s.execute(sa.text(
                "select * from wb_returns")).mappings()])  # fmt: skip
    raw = yaml.safe_load(open("specs/example_sql.yaml").read())
    saa = SAA.from_version("synthetic_example")
    from_lite = load_sql(parse_spec(raw).data, saa.assets, saa.asset_class, url=lite).market
    raw["data"]["sql"]["schema"] = DATA_SCHEMA
    from_pg = load_sql(parse_spec(raw).data, saa.assets, saa.asset_class, url=pg).market
    pd.testing.assert_frame_equal(from_lite.returns, from_pg.returns, check_exact=True)
    assert from_lite.vintage_tag == from_pg.vintage_tag


def test_ui_on_postgres(runs, pg, monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    import workbench.ui

    monkeypatch.setenv("WB_REGISTRY", pg)
    at = AppTest.from_file(str(Path(workbench.ui.__file__).parent / "app.py"),
                           default_timeout=120).run()  # fmt: skip
    for page in ("Experiments", "Corridor", "Cells", "Paths", "Evidence", "Memo"):
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception and not at.error, page

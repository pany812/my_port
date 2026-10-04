import datetime as dt

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from workbench.registry.models import Weight
from workbench.registry.store import CellRecord, Registry


def _experiment(eid="exp1"):
    return {
        "experiment_id": eid,
        "name": "t",
        "spec_yaml": "experiment: t\n",
        "spec_hash": "h" * 64,
        "data_vintage": "v" * 64,
        "saa_version": "example",
        "candidate_id": "CAND",
        "riskfolio_version": "7.4.0",
        "seed": 1,
        "created_at": dt.datetime(2026, 10, 4, tzinfo=dt.UTC),
    }


def _cell(cid, status="ok", weights=None, index=0):
    return CellRecord(
        cell_id=cid,
        cell_index=index,
        config_id="c" * 16,
        allocator="static_saa",
        params={},
        estimator=None,
        constraint_set="cs",
        data_variant="full",
        window_end=dt.date(2026, 9, 30),
        status=status,
        message="" if status == "ok" else "no solution",
        diagnostics={"violations": ["x"]} if status != "ok" else {},
        weights=weights,
    )


@pytest.fixture()
def reg(tmp_path):
    return Registry(f"sqlite:///{tmp_path / 'reg.db'}")


def test_round_trip(reg):
    reg.write_experiment(
        _experiment(),
        [_cell("a", weights={"X": 0.4, "Y": 0.6}), _cell("b", status="infeasible", index=1)],
    )
    assert reg.has_experiment("exp1")
    cells = reg.cells("exp1")
    assert list(cells["cell_id"]) == ["a", "b"]
    assert list(cells["status"]) == ["ok", "infeasible"]
    assert cells.loc[1, "diagnostics_json"] == '{"violations": ["x"]}'
    assert cells.loc[0, "estimator_json"] is None
    w = reg.weights_wide("exp1")
    assert w.loc["a", "Y"] == 0.6 and "b" not in w.index
    assert reg.experiment("exp1")["riskfolio_version"] == "7.4.0"


def test_write_is_atomic(reg):
    with pytest.raises(IntegrityError):
        reg.write_experiment(_experiment(), [_cell("a"), _cell("a")])  # duplicate PK
    assert not reg.has_experiment("exp1")
    assert reg.cells("exp1").empty


def test_delete_cascades_manually(reg):
    reg.write_experiment(_experiment(), [_cell("a", weights={"X": 1.0})])
    reg.delete_experiment("exp1")
    assert not reg.has_experiment("exp1")
    assert reg.weights("exp1").empty


def test_foreign_keys_enforced_on_sqlite(reg):
    with pytest.raises(IntegrityError), Session(reg.engine) as s, s.begin():
        s.execute(insert(Weight), [{"cell_id": "no-such-cell", "asset_id": "X", "weight": 1.0}])


def test_duplicate_experiment_rejected(reg):
    reg.write_experiment(_experiment("e2"), [])
    with pytest.raises(IntegrityError):
        reg.write_experiment(_experiment("e2"), [])


def test_empty_reads(reg):
    assert reg.experiments().empty
    assert list(reg.cells("nope").columns)[:2] == ["cell_id", "experiment_id"]
    with pytest.raises(KeyError):
        reg.experiment("nope")

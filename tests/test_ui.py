"""Read-only UI: registry read-only mode, the data layer, every page, `wb ui`."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
import yaml

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import workbench.ui  # noqa: E402
from tests.test_runner import SMALL  # noqa: E402
from workbench.cli import UI_FLAGS, main  # noqa: E402
from workbench.grid.runner import run_experiment  # noqa: E402
from workbench.grid.spec import parse_spec  # noqa: E402
from workbench.registry.store import Registry, RegistrySchemaError  # noqa: E402
from workbench.ui import data as D  # noqa: E402

APP = str(Path(workbench.ui.__file__).parent / "app.py")
PAGES = ("Experiments", "Corridor", "Cells", "Paths", "Evidence", "Libraries", "Memo")


def _spec():
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": "walk_forward"}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "A"}, solvers=["CLARABEL", "SCS"])  # fmt: skip
    raw["grid"]["allocators"] = [
        {"type": "static_saa"},
        {"type": "saa_plus", "x": [0.05]},
        {"type": "riskfolio_mean_risk", "rm": ["MV"], "obj": ["Sharpe"]},
        {"type": "skfolio_mean_risk", "rm": ["MV"], "obj": ["Sharpe"]},
        {"type": "riskfolio_bl", "target_weight": [0.05]},
        {"type": "riskfolio_risk_budget", "candidate_share": [0.05]},
    ]
    raw["grid"]["constraint_sets"] = [
        {"name": "te", "te_annual": {"sweep": [0.01, 0.02]}, "candidate_cap": 0.2}
    ]
    raw["stress"] = {"windows": {"mid": ["2017-01", "2017-06"]},
                     "bootstrap": {"n_paths": 200, "horizon_years": 5}}  # fmt: skip
    raw["decision"] = {
        "recommendation": "Allocate 5%.",
        "proposal": {"weight": 0.05},
        "kill_criteria": [{"metric": "te_vs_saa", "above": 0.01, "months": 12}],
    }
    return parse_spec(raw)  # fmt: skip


@pytest.fixture(scope="module")
def reg_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ui") / "r.db"
    s = run_experiment(_spec(), Registry(f"sqlite:///{path}"))
    return path, s.experiment_id


@pytest.fixture(scope="module")
def ro(reg_file):
    path, eid = reg_file
    return Registry(f"sqlite:///{path}", read_only=True), eid


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- read-only registry -----------------------------------------------------------------------


def test_read_only_registry_refuses_writes(ro, tmp_path):
    reg, eid = ro
    assert len(reg.cells(eid)) > 0
    with pytest.raises(sa.exc.OperationalError, match="readonly"):
        reg.delete_experiment(eid)
    with pytest.raises(FileNotFoundError):
        Registry(f"sqlite:///{tmp_path / 'missing.db'}", read_only=True)
    assert not (tmp_path / "missing.db").exists()  # never created
    empty = tmp_path / "empty.db"
    sa.create_engine(f"sqlite:///{empty}").connect().close()
    with pytest.raises(RegistrySchemaError, match="not a workbench registry"):
        Registry(f"sqlite:///{empty}", read_only=True)
    with pytest.raises(ValueError, match="in-memory"):
        Registry("sqlite:///:memory:", read_only=True)


# --- data layer -------------------------------------------------------------------------------


def test_experiments_and_provenance(ro):
    reg, eid = ro
    t = D.experiments_table(reg)
    assert list(t["experiment_id"]) == [eid] and t["mode"].iloc[0] == "walk_forward"
    statuses = [c for c in ("ok", "infeasible", "solver_error", "exception") if c in t]
    assert t["n_cells"].iloc[0] == t[statuses].sum(axis=1).iloc[0] == len(reg.cells(eid))
    p = D.provenance(reg, eid).set_index("field")["value"]
    assert p["experiment_id"] == eid and p["stress"] == "yes" and p["decision"] == "yes"


def test_corridor_band_and_cells(ro):
    from workbench.evaluation.corridor import corridor

    reg, eid = ro
    band = D.corridor_band(corridor(reg, eid), "capital_weight", "full")
    assert band["window_end"].is_monotonic_increasing and len(band) > 1
    q = band[list(D.BANDS)].dropna().to_numpy()
    assert (np.diff(q, axis=1) >= -1e-12).all()  # p10 <= p25 <= median <= p75 <= p90
    cells = D.cells_table(reg, eid)
    assert len(cells) == len(reg.cells(eid)) and "saa_reference" not in set(cells["allocator"])
    assert cells.columns[2] == "label" and cells.columns[-1] == "cell_id"


def test_cell_detail(ro):
    reg, eid = ro
    cells = D.cells_table(reg, eid)
    ok = cells[cells["status"] == "ok"].iloc[0]
    d = D.cell_detail(reg, eid, ok["cell_id"])
    w = d["weights"]
    assert w["weight"].sum() == pytest.approx(1) and w["saa"].sum() == pytest.approx(1)
    assert np.allclose(w["active"], w["weight"] - w["saa"])
    assert set(d["metrics"]["metric"]) >= {"candidate_risk_share"}
    failed = cells[cells["status"] != "ok"]
    if not failed.empty:
        assert D.cell_detail(reg, eid, failed["cell_id"].iloc[0])["weights"]["weight"].isna().all()
    with pytest.raises(KeyError):
        D.cell_detail(reg, eid, "nope")


def test_paths_and_sharpe_table(ro):
    reg, eid = ro
    labels = D.config_labels(reg, eid)
    static = [c for c, lab in labels.items() if lab.startswith("static_saa")]
    wealth, dd = D.path_frames(reg, eid, "full", static, labels)
    assert wealth.columns[0] == "SAA"
    # the static SAA configuration's path is the SAA reference path
    assert np.allclose(wealth["SAA"], wealth[labels[static[0]]])
    assert (dd >= 0).all().all() and (wealth > 0).all().all()
    t = D.sharpe_table(reg, eid)
    assert len(t) > 0 and t["p_bh"].notna().any()
    assert t["label"].str.contains(" | te").all()


# --- the app ----------------------------------------------------------------------------------


def _app(url: str, monkeypatch) -> AppTest:
    monkeypatch.setenv("WB_REGISTRY", url)
    return AppTest.from_file(APP, default_timeout=120)


def test_every_page_renders_and_the_registry_is_untouched(reg_file, monkeypatch):
    path, _ = reg_file
    before = _sha(path)
    at = _app(f"sqlite:///{path}", monkeypatch).run()
    assert not at.exception and not at.error
    for page in PAGES:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
    assert _sha(path) == before  # read-only: byte-identical after a session


def test_cells_drill_down_and_corridor_grouping(reg_file, ro, monkeypatch):
    path, _ = reg_file
    reg, eid = ro
    cells = D.cells_table(reg, eid)
    view = cells[(cells.data_variant == "full") & (cells.window_end == cells.window_end.max())]
    pick = view.iloc[1]  # the page lists the latest date's cells in this order
    at = _app(f"sqlite:///{path}", monkeypatch).run()
    at.sidebar.radio(key="page").set_value("Cells").run()
    at.selectbox(key="cell").set_value(pick["cell_id"]).run()
    assert not at.exception
    expected = f"{pick['label']} | {pick['constraint_set']} | {pick['status']}"
    assert any(h.value == expected for h in at.subheader)
    at.sidebar.radio(key="page").set_value("Corridor").run()
    at.selectbox(key="c_group").set_value("library").run()
    assert not at.exception and any("by library" in h.value for h in at.subheader)
    at.sidebar.radio(key="page").set_value("Memo").run()
    assert at.metric[0].value == "PROPOSED"
    assert {b.label for b in at.get("download_button")} == {"memo.md", "summary.md"}


def test_empty_and_missing_registries(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    Registry(url)  # tables, no experiments
    at = _app(url, monkeypatch).run()
    assert any("No experiments yet" in i.value for i in at.info)
    at = _app(f"sqlite:///{tmp_path / 'missing.db'}", monkeypatch).run()
    assert any("Cannot open the registry" in e.value for e in at.error)


# --- wb ui ------------------------------------------------------------------------------------


def test_wb_ui_starts_streamlit_with_safe_defaults(reg_file, monkeypatch, capsys):
    path, _ = reg_file
    seen = {}

    def fake_call(cmd, env):
        seen.update(cmd=cmd, env=env)
        return 0

    monkeypatch.setattr("workbench.cli.subprocess.call", fake_call)
    url = f"sqlite:///{path}"
    assert main(["ui", "--registry", url, "--port", "8600"]) == 0
    cmd = seen["cmd"]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("app.py")
    flags = dict(zip(cmd[5::2], cmd[6::2], strict=True))
    assert flags["--server.address"] == "localhost"
    assert flags["--browser.gatherUsageStats"] == "false"
    assert flags["--server.port"] == "8600" and set(UI_FLAGS[::2]) <= set(flags)
    assert seen["env"]["WB_REGISTRY"] == url
    assert "http://localhost:8600" in capsys.readouterr().out


def test_wb_ui_refuses_a_missing_registry(tmp_path, capsys):
    rc = main(["ui", "--registry", f"sqlite:///{tmp_path / 'none.db'}"])
    assert rc == 2 and "cannot open the registry read-only" in capsys.readouterr().err
    assert not (tmp_path / "none.db").exists()


def test_streamlit_config_is_safe():
    cfg = (Path(__file__).parents[1] / ".streamlit" / "config.toml").read_text()
    assert "gatherUsageStats = false" in cfg and 'address = "localhost"' in cfg

"""CLI (`wb run`, `wb report`) and the markdown report."""

import contextlib
import io

import pandas as pd
import pytest
import yaml

from tests.test_runner import SMALL
from workbench.cli import main
from workbench.evaluation.corridor import corridor
from workbench.evaluation.markdown import md_table, pct
from workbench.evaluation.report import build_report, message_kind
from workbench.registry.store import Registry

SECTIONS = [
    "## Provenance",
    "## Cells",
    "## Allocation corridor",
    "### Latest rebalance date",
    "## Corridor by group",
    "### By allocator family",
    "### By risk measure",
    "### By estimator",
    "### By constraint set",
    "## Failures",
    "## Out-of-sample (walk-forward) vs SAA",
    "## In-sample ex-post (latest date)",
    "## Live-only variant",
    "## Definitions",
]


def _write_spec(tmp_path, mode="walk_forward", **over):
    raw = yaml.safe_load(SMALL)
    raw.update(backtest={"mode": mode}, window={"kind": "rolling", "periods": 36},
               rebalance={"every": "Q"}, experiment="cli_test")  # fmt: skip
    raw.update(over)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cli")
    spec = _write_spec(tmp_path)
    reg = f"sqlite:///{tmp_path / 'reg.db'}"
    out = tmp_path / "out"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["run", str(spec), "--registry", reg, "--out", str(out)]) == 0
    return tmp_path, reg, out, buf.getvalue()


def test_run_writes_all_outputs_and_prints_headline(run_dir):
    _, reg, out, stdout = run_dir
    d = out / "cli_test"
    for name in ("corridor.csv", "expost.csv", "oos.csv", "summary.md"):
        assert (d / name).exists(), name
    assert "capital_weight" in stdout and "wrote" in stdout
    assert "experiment cli_test" in stdout


def test_corridor_csv_equals_corridor_function(run_dir):
    _, reg, out, _ = run_dir
    r = Registry(reg)
    exp_id = r.resolve("cli_test")
    expected = corridor(r, exp_id)
    got = pd.read_csv(out / "cli_test" / "corridor.csv")
    assert len(got) == len(expected)
    assert list(got.columns) == list(expected.columns)
    assert (got["n_cells"].to_numpy() == expected["n_cells"].to_numpy()).all()
    pd.testing.assert_series_equal(got["median"], expected["median"], check_names=False)


def test_summary_has_every_section_and_short_sample_banner(run_dir):
    _, _, out, _ = run_dir
    md = (out / "cli_test" / "summary.md").read_text()
    for s in SECTIONS:
        assert s in md, s
    # live-only OOS (from 2019-01) is 24 months < 36
    assert "Sample too short" in md
    assert "Most common failure kinds" in md
    assert "CAND: weight #% above upper bound #%" in md


def test_report_by_name_and_by_id(run_dir, capsys):
    tmp, reg, out, _ = run_dir
    exp_id = Registry(reg).resolve("cli_test")
    assert main(["report", "cli_test", "--registry", reg, "--out", str(tmp / "o2")]) == 0
    assert main(["report", exp_id, "--registry", reg, "--out", str(tmp / "o3")]) == 0
    a = (tmp / "o2" / "cli_test" / "corridor.csv").read_text()
    b = (out / "cli_test" / "corridor.csv").read_text()
    assert a == b
    assert "wrote" in capsys.readouterr().out


def test_rerun_skips_and_says_so(run_dir, capsys):
    tmp, reg, out, _ = run_dir
    assert main(["run", str(tmp / "spec.yaml"), "--registry", reg, "--out", str(out)]) == 0
    assert "already in registry" in capsys.readouterr().out


def test_unknown_experiment_and_bad_spec_exit_2(tmp_path, capsys):
    reg = f"sqlite:///{tmp_path / 'r.db'}"
    assert main(["report", "nope", "--registry", reg]) == 2
    assert "no experiment" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment: x\n")
    assert main(["run", str(bad), "--registry", reg]) == 2
    assert "missing keys" in capsys.readouterr().err


def test_registry_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WB_REGISTRY", f"sqlite:///{tmp_path / 'sub' / 'env.db'}")
    spec = _write_spec(tmp_path, mode="in_sample")
    assert main(["run", str(spec), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "sub" / "env.db").exists()


def test_in_sample_report_says_no_oos_evidence(tmp_path):
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    spec = _write_spec(tmp_path, mode="in_sample")
    assert main(["run", str(spec), "--registry", str(reg.engine.url),
                 "--out", str(tmp_path / "out")]) == 0  # fmt: skip
    md = build_report(reg, reg.resolve("cli_test")).summary_md
    assert "No out-of-sample evidence" in md
    assert "Through time" not in md  # single date


def test_md_table_and_message_kind():
    t = md_table(pd.DataFrame({"a": ["x|y", None], "w": [0.0512, float("nan")]}), {"w": pct(1)})
    assert t.splitlines()[1] == "| :--- | ---: |"
    assert "x\\|y" in t and "5.1%" in t and "| – | – |" in t
    assert md_table(pd.DataFrame()) == "_(none)_"
    assert message_kind("TE 7.69% p.a. exceeds limit 2.00%\nmore") == "TE #% p.a. exceeds limit #%"

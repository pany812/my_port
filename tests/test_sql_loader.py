"""SQL data source through the data contract, vintages, compounding, file SAA versions,
data check, and the registry's P2-M8 columns. SQLite only (PostgreSQL: tests/test_postgres.py)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
import yaml

from workbench.cli import main
from workbench.data.align import align_history, data_vintage
from workbench.data.check import data_check
from workbench.data.demo import DEMO_VINTAGE, contract_tables, write_demo_contract
from workbench.data.sql import DataSourceError, compound, load_sql
from workbench.data.synthetic import CandidateSpec, generate
from workbench.grid.runner import experiment_id_for, run_experiment
from workbench.grid.spec import parse_spec
from workbench.policy.saa import SAA, SAAError
from workbench.registry.store import Registry

SPEC = Path(__file__).parents[1] / "specs" / "example_sql.yaml"
SAA_FILE = SAA.from_version("synthetic_example")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    path = tmp_path_factory.mktemp("sql") / "demo.db"
    url = f"sqlite:///{path}"
    write_demo_contract(url)
    return url, path


def _spec(**sql):
    raw = yaml.safe_load(SPEC.read_text())
    raw["data"]["sql"] = {**raw["data"]["sql"], **sql}
    return parse_spec(raw)


def _load(url, **sql):
    return load_sql(_spec(**sql).data, SAA_FILE.assets, SAA_FILE.asset_class, url=url)


# --- the demo contract reproduces the synthetic market ---------------------------------------


def test_demo_round_trip_equals_the_synthetic_market(demo):
    url, path = demo
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    read = _load(url)
    g = generate(seed=42, start="2001-01", end="2026-09", freq="M",
                 candidate=CandidateSpec(live_start="2016-01"))  # fmt: skip
    m = read.market
    assert read.vintage_tag == DEMO_VINTAGE == m.vintage_tag
    pd.testing.assert_frame_equal(m.returns, g.returns[SAA_FILE.assets], check_freq=False)
    assert (m.backfilled.to_numpy() == g.backfilled.to_numpy()).all()
    assert data_vintage(align_history(m)) == data_vintage(align_history(g))
    assert set(read.assets["kind"]) == {"block", "cash", "candidate", "proxy"}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before  # read-only


def test_without_a_proxy_history_starts_at_live_start(demo):
    url, _ = demo
    m = _load(url, candidate_proxy=None).market
    assert not m.backfilled.any()
    aligned = align_history(m)
    assert aligned.returns.index[0] == pd.Timestamp("2016-01-31")


# --- a hand-built contract: vintages, compounding, errors -------------------------------------


def _contract(tmp_path, returns: list[dict], assets: dict[str, str] | None = None) -> str:
    url = f"sqlite:///{tmp_path / 'c.db'}"
    md = sa.MetaData()
    a_t, r_t = contract_tables(md)
    eng = sa.create_engine(url)
    with eng.begin() as c:
        md.create_all(c)
        assets = assets or {a: "block" for a in SAA_FILE.assets} | {"CAND": "candidate"}
        c.execute(sa.insert(a_t), [{"asset_id": a, "kind": k} for a, k in assets.items()])
        c.execute(sa.insert(r_t), returns)
    return url


def _monthly(vintage="v1", months=36, assets=None, value=0.01):
    idx = pd.date_range("2018-01-31", periods=months, freq="ME")
    return [
        {
            "asset_id": a,
            "period_end": d.date(),
            "frequency": "M",
            "vintage": vintage,
            "simple_return": value,
        }
        for a in (assets or SAA_FILE.assets)
        for d in idx
    ]


def _sql_spec(**kw):
    raw = yaml.safe_load(SPEC.read_text())
    raw["data"].update(start="2018-01", end="2020-12")
    raw["data"]["sql"] = {"vintage": "latest", **kw}
    return parse_spec(raw).data


def test_vintages_are_point_in_time(tmp_path):
    rows = _monthly("2026-01")
    rows.append(
        {
            "asset_id": "GL_EQ",
            "period_end": pd.Timestamp("2018-03-31").date(),
            "frequency": "M",
            "vintage": "2026-02",
            "simple_return": 0.05,
        }
    )  # revision
    url = _contract(tmp_path, rows)
    spec = _sql_spec()
    read = load_sql(spec, SAA_FILE.assets, SAA_FILE.asset_class, url=url)
    assert read.vintage_tag == "2026-02"
    assert read.market.returns.loc["2018-03-31", "GL_EQ"] == 0.05
    old = load_sql(_sql_spec(vintage="2026-01"), SAA_FILE.assets, SAA_FILE.asset_class, url=url)
    assert old.market.returns.loc["2018-03-31", "GL_EQ"] == 0.01
    with pytest.raises(DataSourceError, match="no rows"):
        load_sql(_sql_spec(vintage="2025-12"), SAA_FILE.assets, SAA_FILE.asset_class, url=url)


def test_daily_data_is_compounded_and_partial_periods_dropped(tmp_path):
    rows = [r for r in _monthly() if r["asset_id"] != "HY"]
    days = pd.bdate_range("2018-01-01", "2018-04-11")  # April is partial
    rng = np.random.default_rng(0)
    daily = rng.normal(0.0004, 0.006, len(days))
    rows += [
        {
            "asset_id": "HY",
            "period_end": d.date(),
            "frequency": "D",
            "vintage": "v1",
            "simple_return": float(v),
        }
        for d, v in zip(days, daily, strict=True)
    ]
    url = _contract(tmp_path, rows)
    read = load_sql(_sql_spec(), SAA_FILE.assets, SAA_FILE.asset_class, url=url)
    hy = read.market.returns["HY"].dropna()
    assert read.source_frequency["HY"] == "D" and read.source_frequency["SE_EQ"] == "M"
    jan = (1 + pd.Series(daily, index=days)["2018-01"]).prod() - 1
    assert hy.loc["2018-01-31"] == pytest.approx(jan, abs=1e-15)
    assert list(hy.index.strftime("%Y-%m")) == ["2018-01", "2018-02", "2018-03"]  # April dropped
    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-02-03",
        "2020-02-04",
        "2020-02-05",
        "2020-03-02",
    ]  # three, three and one observation(s)
    s = pd.Series([0.01] * len(dates), index=pd.to_datetime(dates), name="x")
    assert list(compound(s, "M").index.month) == [1, 2]  # March (1 < half of 3) is partial


def _only_quarterly_hy(rows, assets):
    hy = [r for r in rows if r["asset_id"] == "HY"]
    for r in hy:
        rows.remove(r)
    rows.extend([{**r, "frequency": "Q"} for r in hy])


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda rows, assets: assets.pop("HY"), r"assets not in wb_assets: \['HY'\]"),
        (lambda rows, assets: assets.update(HY="fund"), "kind values"),
        (_only_quarterly_hy, "only coarser data"),
        (lambda rows, assets: rows.append({**rows[0], "period_end":
                                           pd.Timestamp("2018-01-30").date()}),
         "more than one M observation"),
    ],
)  # fmt: skip
def test_contract_errors(tmp_path, mutate, match):
    rows = _monthly()
    assets = {a: "block" for a in SAA_FILE.assets} | {"CAND": "candidate"}
    mutate(rows, assets)
    url = _contract(tmp_path, rows, assets)
    with pytest.raises(DataSourceError, match=match):
        load_sql(_sql_spec(), SAA_FILE.assets, SAA_FILE.asset_class, url=url)


def test_url_from_the_environment_and_redacted_errors(monkeypatch):
    spec = _sql_spec()
    monkeypatch.delenv("WB_DATA_URL", raising=False)
    with pytest.raises(DataSourceError, match=r"set \$WB_DATA_URL"):
        load_sql(spec, SAA_FILE.assets, SAA_FILE.asset_class)
    with pytest.raises(DataSourceError) as e:
        load_sql(spec, SAA_FILE.assets, SAA_FILE.asset_class,
                 url="postgresql+psycopg://wb:s3cret@127.0.0.1:1/none")  # fmt: skip
    assert "s3cret" not in str(e.value) and "***" in str(e.value)


# --- runner, experiment ids, SAA files --------------------------------------------------------


def test_experiment_ids_unchanged_for_code_saa_and_cover_file_content():
    old = hashlib.sha256(b"h|v|7.4.0").hexdigest()[:16]
    assert experiment_id_for("h", "v", "7.4.0") == old
    assert experiment_id_for("h", "v", "7.4.0", "abc") != old


def test_run_with_sql_source_records_provenance(demo, monkeypatch, tmp_path):
    url, _ = demo
    monkeypatch.setenv("WB_DATA_URL", url)
    raw = yaml.safe_load(SPEC.read_text())
    raw.update(backtest={"mode": "in_sample"})
    reg = Registry(f"sqlite:///{tmp_path / 'r.db'}")
    s = run_experiment(parse_spec(raw), reg)
    e = reg.experiment(s.experiment_id)
    assert e["data_source"] == "sql" and e["data_vintage_tag"] == DEMO_VINTAGE
    # the same spec text with an edited SAA file is a different experiment
    saa_dir = tmp_path / "saa"
    saa_dir.mkdir()
    doc = yaml.safe_load((Path(__file__).parents[1] / "saa" / "synthetic_example.yaml").read_text())
    doc["assets"]["GL_EQ"]["weight"], doc["assets"]["SE_EQ"]["weight"] = 0.29, 0.16
    (saa_dir / "synthetic_example.yaml").write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("WB_SAA_DIR", str(saa_dir))
    s2 = run_experiment(parse_spec(raw), reg)
    assert s2.experiment_id != s.experiment_id and s2.spec_hash == s.spec_hash


def test_saa_files(tmp_path, monkeypatch):
    monkeypatch.setenv("WB_SAA_DIR", str(tmp_path))
    good = yaml.safe_load((Path(__file__).parents[1] / "saa" / "synthetic_example.yaml")
                          .read_text())  # fmt: skip

    def write(name, doc):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(doc))

    write("synthetic_example", good)
    assert SAA.from_version("synthetic_example").content_hash() == SAA_FILE.content_hash()
    for version, doc, match in [
        ("../etc", good, "use letters"),
        ("missing", None, "no file"),
        ("renamed", good, "must equal the file name"),
        ("cand", {**good, "version": "cand", "assets": {**good["assets"],
                                                        "CAND": good["assets"]["HY"]}},
         "must not be an SAA asset"),
        ("bad", {**good, "version": "bad", "assets": {"X": {"weight": 1.0}}}, "exactly class"),
    ]:  # fmt: skip
        if doc is not None:
            write(version if version != "../etc" else "etc", doc)
        with pytest.raises(SAAError, match=match):
            SAA.from_version(version)
    write(
        "off",
        {
            **good,
            "version": "off",
            "assets": {**good["assets"], "CASH": {**good["assets"]["CASH"], "weight": 0.2}},
        },
    )
    with pytest.raises(ValueError, match="sum to 1"):
        SAA.from_version("off")


# --- data check and CLI -----------------------------------------------------------------------


def test_data_check_reports_gaps_and_outliers(tmp_path, monkeypatch):
    rows = [
        r
        for r in _monthly()
        if not (r["asset_id"] == "HY" and r["period_end"] == pd.Timestamp("2019-06-30").date())
    ]
    for r in rows:
        if r["asset_id"] == "EM_EQ" and r["period_end"] == pd.Timestamp("2019-01-31").date():
            r["simple_return"] = 0.5
    em = [r for r in rows if r["asset_id"] == "EM_EQ"]
    for i, r in enumerate(em):  # some dispersion so the outlier stands out
        if r["simple_return"] != 0.5:
            r["simple_return"] = 0.01 + 0.001 * (i % 5)
    url = _contract(tmp_path, rows)
    monkeypatch.setenv("WB_DATA_URL", url)
    raw = yaml.safe_load(SPEC.read_text())
    raw["data"].update(start="2018-01", end="2020-12")
    raw["data"]["sql"] = {"vintage": "latest"}
    dc = data_check(parse_spec(raw))
    a = dc.assets.set_index("asset_id")
    assert a.loc["HY", "n_gaps"] == 1 and a.loc["EM_EQ", "n_outliers"] == 1
    assert (
        not dc.ok and "missing returns" in dc.summary.set_index("field").loc["alignment", "value"]
    )


def test_cli_data_demo_and_check(tmp_path, monkeypatch, capsys):
    url = f"sqlite:///{tmp_path / 'd.db'}"
    assert main(["data", "demo", "--url", url]) == 0
    assert main(["data", "demo", "--url", url]) == 2  # refuses to overwrite
    assert "already has contract tables" in capsys.readouterr().err
    assert main(["data", "demo", "--url", "postgresql+psycopg://x@localhost/db"]) == 2
    monkeypatch.setenv("WB_DATA_URL", url)
    assert main(["data", "check", str(SPEC), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "vintage tag" in out and "candidate" in out and "alignment" in out
    assert (tmp_path / "out" / "sql_demo_v1" / "data_check.csv").exists()

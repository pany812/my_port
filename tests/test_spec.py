import copy
from pathlib import Path

import pytest
import yaml

from workbench.grid.spec import SpecError, load_spec, parse_spec

EXAMPLE = Path(__file__).parents[1] / "specs" / "example_synthetic.yaml"


@pytest.fixture()
def raw():
    return yaml.safe_load(EXAMPLE.read_text())


def test_example_spec_parses():
    spec = load_spec(EXAMPLE)
    assert spec.experiment == "synthetic_trend_v1"
    assert spec.data.start == "2006-01" and spec.data.end == "2026-09"
    assert spec.data.synthetic.live_start == "2016-01"
    assert spec.data.hedging.startswith("synthetic placeholder")
    assert [cs.name for cs in spec.constraint_sets] == ["bands_5pct", "te_2pct"]
    saa_plus = spec.allocators[1]
    assert saa_plus.params == {"x": [0.02, 0.05, 0.10], "funding": ["pro_rata"]}
    assert spec.source_text == EXAMPLE.read_text()


def test_hash_ignores_name_and_formatting(raw):
    base = parse_spec(copy.deepcopy(raw))
    renamed = copy.deepcopy(raw)
    renamed["experiment"] = "something_else"
    assert parse_spec(renamed).spec_hash == base.spec_hash
    # scalar vs one-element list is the same grid
    scalar = copy.deepcopy(raw)
    scalar["grid"]["allocators"][3]["linkage"] = "ward"
    assert parse_spec(scalar).spec_hash == base.spec_hash
    # explicit defaults hash like implicit ones
    explicit = copy.deepcopy(raw)
    explicit["solvers"] = ["CLARABEL"]
    assert parse_spec(explicit).spec_hash == base.spec_hash


def test_hash_changes_with_content(raw):
    base = parse_spec(copy.deepcopy(raw)).spec_hash
    for path, value in [
        (("seed",), 43),
        (("data", "synthetic", "skew"), 0.2),
        (("window", "periods"), 60),
        (("rf_annual",), 0.02),
    ]:
        changed = copy.deepcopy(raw)
        node = changed
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        assert parse_spec(changed).spec_hash != base, path


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r.pop("seed"), "missing keys \\['seed'\\]"),
        (lambda r: r.update(sead=1), "unknown keys \\['sead'\\]"),
        (lambda r: r["data"].pop("hedging"), "data: missing keys \\['hedging'\\]"),
        (lambda r: r["data"].pop("base_currency"), "base_currency"),
        (lambda r: r["data"].update(frequency="monthly"), "data.frequency"),
        (lambda r: r["data"]["synthetic"].update(vol=0.1), "data.synthetic: unknown keys"),
        (lambda r: r["grid"]["allocators"].append({"type": "magic"}), "unknown allocator type"),
        (lambda r: r["grid"]["allocators"][2].update(rmm="MV"), "unknown parameters \\['rmm'\\]"),
        (lambda r: r["grid"]["allocators"][2].update(method_mu="hist"), "unknown parameters"),
        (lambda r: r["grid"]["constraint_sets"][0].update(te=0.02), "constraint_sets\\[0\\]"),
        (lambda r: r["grid"]["constraint_sets"].append({"name": "te_2pct"}), "duplicate names"),
        (lambda r: r["window"].update(kind="sliding"), "window.kind"),
        (lambda r: r["window"].pop("periods"), "window.periods"),
        (lambda r: r.update(seed="42"), "seed must be an integer"),
        (lambda r: r["grid"].update(allocators=[]), "non-empty list"),
    ],
)
def test_spec_errors(raw, mutate, match):
    mutate(raw)
    with pytest.raises(SpecError, match=match):
        parse_spec(raw)


def test_yaml_dates_normalised(raw):
    raw["data"]["start"] = yaml.safe_load("2006-01-31")  # a datetime.date
    assert parse_spec(raw).data.start == "2006-01-31"


def test_default_estimator_when_omitted(raw):
    raw["grid"].pop("estimators")
    assert parse_spec(raw).estimators == ({"method_mu": "hist", "method_cov": "hist"},)


def test_spec_built_from_dict_dumps_yaml(raw):
    spec = parse_spec(raw)
    assert spec.source_text is None
    assert parse_spec(spec.yaml_text).spec_hash == spec.spec_hash


def test_dump_round_trips_for_sql_and_expanding(raw):
    raw["data"]["source"] = "sql"
    raw["data"].pop("synthetic")
    raw["data"]["sql"] = {"live_start": "2016-01", "candidate_proxy": "CAND_PROXY"}
    raw["window"] = {"kind": "expanding", "min_periods": 36}
    spec = parse_spec(raw)
    assert spec.data.sql.url_env == "WB_DATA_URL" and spec.data.sql.vintage == "latest"
    assert parse_spec(spec.yaml_text).spec_hash == spec.spec_hash
    raw["data"]["source"] = "postgres"
    with pytest.raises(SpecError, match="data.source must be one of"):
        parse_spec(raw)


def test_readme_spec_example_parses():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    block = readme.split("## How to write a spec")[1].split("```yaml")[1].split("```")[0]
    spec = parse_spec(block)
    names = [cs.name for cs in spec.constraint_sets]
    assert len(names) == 7 and spec.data.synthetic.backfill is True  # te sweep -> 3 sets
    assert "te[te_annual=0.005]" in names and "caps" in names

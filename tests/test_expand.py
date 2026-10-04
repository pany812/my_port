import copy
from pathlib import Path

import pytest
import yaml

from workbench.grid.expand import expand
from workbench.grid.spec import parse_spec

EXAMPLE = Path(__file__).parents[1] / "specs" / "example_synthetic.yaml"


@pytest.fixture()
def raw():
    return yaml.safe_load(EXAMPLE.read_text())


def test_example_count_and_order(raw):
    cells = expand(parse_spec(raw))
    # per constraint set: static 1 + saa_plus 3 + mean-risk 3x2x2 est + HC 2x2x1x2 est = 24
    assert len(cells) == 48
    assert [c.index for c in cells] == list(range(48))
    assert [c.constraint_set for c in cells[:24]] == ["bands_5pct"] * 24
    first = cells[:6]
    assert [c.allocator for c in first] == ["static_saa"] + ["saa_plus"] * 3 + [
        "riskfolio_mean_risk"
    ] * 2
    assert [c.params.get("x") for c in first[1:4]] == [0.02, 0.05, 0.10]
    # parameter product in spec order (rm outer, obj inner), estimator innermost
    mr = [c for c in cells[:24] if c.allocator == "riskfolio_mean_risk"]
    assert [(c.params["rm"], c.params["obj"], c.estimator["method_mu"]) for c in mr[:4]] == [
        ("MV", "Sharpe", "hist"),
        ("MV", "Sharpe", "JS"),
        ("MV", "MinRisk", "hist"),
        ("MV", "MinRisk", "JS"),
    ]


def test_naive_allocators_not_repeated_per_estimator(raw):
    cells = expand(parse_spec(raw))
    static = [c for c in cells if c.allocator == "static_saa"]
    assert len(static) == 2 and all(c.estimator is None for c in static)


def test_config_ids_unique_and_stable_under_reordering(raw):
    base = {(c.allocator, str(c.params), str(c.estimator), c.constraint_set): c.config_id
            for c in expand(parse_spec(copy.deepcopy(raw)))}  # fmt: skip
    assert len(set(base.values())) == len(base)

    shuffled = copy.deepcopy(raw)
    shuffled["grid"]["allocators"].reverse()
    shuffled["grid"]["constraint_sets"].reverse()
    shuffled["grid"]["estimators"].reverse()
    shuffled["grid"]["allocators"][0]["model"] = ["HERC", "HRP"]  # HC entry is now first
    again = {(c.allocator, str(c.params), str(c.estimator), c.constraint_set): c.config_id
             for c in expand(parse_spec(shuffled))}  # fmt: skip
    assert again == base


def test_config_id_depends_on_constraint_content(raw):
    a = expand(parse_spec(copy.deepcopy(raw)))[0].config_id
    raw["grid"]["constraint_sets"][0]["band"] = 0.04
    assert expand(parse_spec(raw))[0].config_id != a


def test_duplicate_configurations_rejected(raw):
    raw["grid"]["allocators"].append({"type": "static_saa"})
    with pytest.raises(ValueError, match="duplicate"):
        expand(parse_spec(raw))

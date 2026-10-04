"""Golden determinism test: fixed spec + seed => identical spec_hash and weights.

Regenerate the golden file deliberately (and review the diff) with:
    WB_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from workbench.grid.runner import run_experiment
from workbench.grid.spec import load_spec
from workbench.registry.store import Registry

ROOT = Path(__file__).parents[1]
SPEC = ROOT / "specs" / "example_synthetic.yaml"
GOLDEN = ROOT / "tests" / "golden" / "example_synthetic_weights.csv"

# Changing the spec file or the canonical form changes this on purpose; update both together.
EXPECTED_SPEC_HASH = "d789b58883e7cd41b46a0948ffcd5a625f686955060a49c32c378835d9730e79"


def _weights(tmp_path, name):
    reg = Registry(f"sqlite:///{tmp_path / name}")
    s = run_experiment(load_spec(SPEC), reg)
    cells = reg.cells(s.experiment_id)[["cell_id", "config_id", "data_variant", "status"]]
    w = reg.weights(s.experiment_id).merge(cells, on="cell_id")
    w = w[["data_variant", "config_id", "asset_id", "weight"]]
    return s, cells, w.sort_values(["data_variant", "config_id", "asset_id"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("golden")
    return _weights(tmp, "a.db"), _weights(tmp, "b.db")


def test_spec_hash_pinned():
    assert load_spec(SPEC).spec_hash == EXPECTED_SPEC_HASH


def test_identical_weights_across_runs(runs):
    (sa, ca, wa), (sb, cb, wb) = runs
    assert sa.experiment_id == sb.experiment_id
    pd.testing.assert_frame_equal(ca, cb)
    pd.testing.assert_frame_equal(wa, wb, check_exact=True)


def test_weights_match_golden_file(runs):
    _, _, w = runs[0]
    if os.environ.get("WB_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        w.to_csv(GOLDEN, index=False, float_format="%.12g")
        pytest.skip("golden file regenerated")
    golden = pd.read_csv(GOLDEN)
    pd.testing.assert_frame_equal(
        w[["data_variant", "config_id", "asset_id"]],
        golden[["data_variant", "config_id", "asset_id"]],
    )
    assert (w["weight"] - golden["weight"]).abs().max() < 1e-8

"""Golden determinism test: fixed spec + seed => identical spec_hash, weights, paths, corridor.

Uses the small ``tests/golden/golden_spec.yaml`` (walk-forward, and the same spec in in-sample
mode) so ``pytest -q`` stays fast. Regenerate deliberately, then review the diff:
    WB_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
"""

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from workbench.evaluation.corridor import corridor
from workbench.grid.runner import run_experiment
from workbench.grid.spec import parse_spec
from workbench.registry.store import Registry

GOLDEN_DIR = Path(__file__).parent / "golden"
SPEC_TEXT = (GOLDEN_DIR / "golden_spec.yaml").read_text()

# Changing the spec file or the canonical form changes these on purpose; update them together.
EXPECTED_SPEC_HASH = {
    "walk_forward": "f1d84267d2e4416d78d12ccf7e98586596913aeab0c3b4daf889b9c867ba613e",
    "in_sample": "49c7d46f2941c9a47ceedb939842597c8226c3835e31b3e20549e6ba1c2b5919",
}
KEYS = ("status", "weights", "oos", "corridor")


def _spec(mode: str):
    if mode == "walk_forward":
        return parse_spec(SPEC_TEXT)
    raw = yaml.safe_load(SPEC_TEXT)
    raw["backtest"]["mode"] = mode
    return parse_spec(raw)


def _snapshot(tmp_path, mode: str, name: str) -> dict:
    reg = Registry(f"sqlite:///{tmp_path / name}")
    s = run_experiment(_spec(mode), reg)
    cells = reg.cells(s.experiment_id)
    keys = ["data_variant", "window_end", "config_id"]
    w = reg.weights(s.experiment_id).merge(cells[["cell_id", *keys]], on="cell_id")
    w = w[[*keys, "asset_id", "weight"]]
    return {
        "summary": s,
        "status": cells[[*keys, "status"]].sort_values(keys).reset_index(drop=True),
        "weights": w.sort_values([*keys, "asset_id"]).reset_index(drop=True),
        "oos": reg.oos_returns(s.experiment_id, include_reference=True).reset_index(drop=True),
        "corridor": corridor(reg, s.experiment_id),
    }


@pytest.fixture(scope="module", params=["walk_forward", "in_sample"])
def runs(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    mode = request.param
    return mode, _snapshot(tmp, mode, "a.db"), _snapshot(tmp, mode, "b.db")


def test_spec_hash_pinned(runs):
    mode, a, _ = runs
    assert a["summary"].spec_hash == EXPECTED_SPEC_HASH[mode]


def test_identical_across_runs(runs):
    _, a, b = runs
    assert a["summary"].experiment_id == b["summary"].experiment_id
    for key in KEYS:
        pd.testing.assert_frame_equal(a[key], b[key], check_exact=True)


@pytest.mark.parametrize("key", KEYS)
def test_matches_golden_files(runs, key):
    mode, a, _ = runs
    frame = a[key].copy()
    for col in ("window_end", "date"):
        if col in frame:
            frame[col] = frame[col].astype(str)
    path = GOLDEN_DIR / f"{mode}_{key}.csv"
    if os.environ.get("WB_UPDATE_GOLDEN") == "1":
        frame.to_csv(path, index=False, float_format="%.12g")
        pytest.skip(f"regenerated {path.name}")
    golden = pd.read_csv(path, dtype={"window_end": str, "date": str})
    assert list(golden.columns) == list(frame.columns)
    assert len(golden) == len(frame)
    num = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    other = [c for c in frame.columns if c not in num]
    pd.testing.assert_frame_equal(
        frame[other].astype(str).reset_index(drop=True),
        golden[other].astype(str).reset_index(drop=True),
    )
    diff = (frame[num].astype(float) - golden[num].astype(float)).abs().fillna(0.0)
    both_nan = frame[num].isna().to_numpy() == golden[num].isna().to_numpy()
    assert both_nan.all(), "NaN pattern differs from golden"
    assert (diff < 1e-8).all().all(), diff.max()

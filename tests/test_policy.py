import math

import numpy as np
import pandas as pd
import pytest
import riskfolio as rp

from tests.fixtures.synthetic import policy, small_market
from workbench.data.synthetic import placeholder_saa
from workbench.policy.compiled import CompiledPolicy
from workbench.policy.compiler import ConstraintSet, compile_policy
from workbench.policy.riskfolio import hc_bounds, hc_bounds_problem, linear_constraints
from workbench.policy.saa import SAA

ASSETS = SAA.placeholder().assets
EQUITY = ["SE_EQ", "GL_EQ", "EM_EQ"]


def test_placeholder_saa_is_valid():
    saa = SAA.placeholder()
    assert saa.weights["CAND"] == 0 and saa.upper["CAND"] == 1.0
    assert saa.asset_class["CAND"] == "alternatives"


def test_saa_rejects_weight_outside_range():
    saa = SAA.placeholder()
    lower = saa.lower.copy()
    lower["GL_EQ"] = 0.31
    with pytest.raises(ValueError, match="within"):
        SAA("bad", saa.weights, saa.asset_class, lower, saa.upper, "CAND")


def test_constraint_set_from_dict():
    cs = ConstraintSet.from_dict({"name": "x", "class_limits": {"equity": [0.4, 0.6]}})
    assert cs.class_limits == {"equity": (0.4, 0.6)}
    with pytest.raises(ValueError, match="unknown constraint_set keys"):
        ConstraintSet.from_dict({"name": "x", "te": 0.02})
    with pytest.raises(ValueError, match="needs a name"):
        ConstraintSet.from_dict({"band": 0.05})


def test_compile_converts_units():
    cs = ConstraintSet(name="te", te_annual=0.02, candidate_cap=0.10)
    p = compile_policy(SAA.placeholder(), cs, freq="M", rf_annual=0.03)
    assert p.te == pytest.approx(0.02 / math.sqrt(12))
    assert p.rf == pytest.approx(1.03 ** (1 / 12) - 1)
    lo, hi = p.bounds(ASSETS)
    assert hi["CAND"] == 0.10 and hi["GL_EQ"] == 1.0 and (lo == 0).all()
    assert np.allclose(p.benchweights, placeholder_saa()[ASSETS])


def test_compile_asset_ranges_with_cap():
    p = policy(asset_ranges=True, candidate_cap=0.05)
    lo, hi = p.bounds(ASSETS)
    assert lo["GL_EQ"] == 0.25 and hi["GL_EQ"] == 0.35 and hi["CAND"] == 0.05


def test_compile_rejects_bad_inputs():
    with pytest.raises(ValueError, match="unknown classes"):
        policy(class_limits={"crypto": [0, 0.1]})
    with pytest.raises(ValueError):
        policy(candidate_cap=1.5)
    with pytest.raises(ValueError):
        policy(te_annual=0.0)


def test_linear_constraints_match_hand_built():
    p = policy(candidate_cap=0.10, class_limits={"equity": [0.45, 0.55]})
    A, B = linear_constraints(p, ASSETS)
    e_cand = np.array([a == "CAND" for a in ASSETS], dtype=float)
    e_eq = np.array([a in EQUITY for a in ASSETS], dtype=float)
    expected_A = np.vstack([e_cand, -e_eq, e_eq])
    expected_B = np.array([0.10, -0.45, 0.55])
    assert A.shape == (3, len(ASSETS))
    assert np.allclose(A, expected_A) and np.allclose(B.ravel(), expected_B)


def test_linear_constraints_none_when_unconstrained():
    assert linear_constraints(CompiledPolicy(), ASSETS) == (None, None)


def test_hc_bounds_apply_cap_and_band():
    w_max, w_min = hc_bounds(policy(candidate_cap=0.10), ASSETS)
    assert w_max["CAND"] == pytest.approx(0.10) and w_max["GL_EQ"] == pytest.approx(1.0)
    w_max, w_min = hc_bounds(policy(band=0.05, candidate_cap=0.10), ASSETS)
    assert w_max["GL_EQ"] == pytest.approx(0.35) and w_min["GL_EQ"] == pytest.approx(0.25)
    assert w_max["CAND"] == pytest.approx(0.05) and w_min["CAND"] == 0.0
    assert hc_bounds_problem(w_max, w_min) is None


def test_riskfolio_hrp_constraints_bool_trap_still_present():
    """Documents the Riskfolio-Lib 7.4.0 trap our hc_bounds works around (see CLAUDE.md)."""
    classes = pd.DataFrame({"Assets": ASSETS, "Class": [""] * len(ASSETS)})
    table = pd.DataFrame(
        {"Disabled": [False], "Type": ["Assets"], "Set": [""], "Position": ["CAND"],
         "Sign": ["<="], "Weight": [0.03]}
    )  # fmt: skip
    w_max_bool, _ = rp.hrp_constraints(table, classes)
    w_max_obj, _ = rp.hrp_constraints(table.astype({"Disabled": object}), classes)
    assert w_max_bool["CAND"] == 1.0  # silently ignored
    assert w_max_obj["CAND"] == pytest.approx(0.03)


def test_hc_bounds_problem_detection():
    idx = ["A", "B"]
    assert "sum to 0.8" in hc_bounds_problem(pd.Series([0.4, 0.4], idx), pd.Series(0.0, idx))
    assert "above upper" in hc_bounds_problem(pd.Series([0.6, 0.6], idx), pd.Series([0.7, 0], idx))


def test_violations_detect_each_breach():
    r = small_market().returns
    saa = placeholder_saa()[r.columns]
    assert policy(band=0.05, te_annual=0.02, asset_ranges=True).violations(saa, r) == []

    w = saa * 0.85
    w["CAND"] = 0.15
    msgs = policy(candidate_cap=0.10, band=0.05, te_annual=0.001).violations(w, r)
    text = " ".join(msgs)
    assert "CAND: weight 15.0000% above upper bound 10.0000%" in text
    assert "exceeds band" in text and "TE" in text

    eq_heavy = saa.copy()
    eq_heavy["GL_EQ"] += 0.10
    eq_heavy["SE_GOV"] -= 0.10
    assert any("class equity" in m for m in policy(class_limits={"equity": [0.4, 0.55]})
               .violations(eq_heavy, r))  # fmt: skip

    neg = saa.copy()
    neg["CASH"], neg["GL_EQ"] = -0.01, 0.36
    assert any("long-only" in m for m in CompiledPolicy().violations(neg, r))
    assert any("sum to" in m for m in CompiledPolicy().violations(saa * 0.9, r))


def test_tracking_error_matches_riskfolio_definition():
    r = small_market().returns
    p = policy(te_annual=0.02)
    w = placeholder_saa()[r.columns] * 0.9
    w["CAND"] = 0.1
    active = r.to_numpy() @ (w - p.benchweights).to_numpy()
    assert p.tracking_error(w, r) == pytest.approx(np.sqrt((active**2).sum() / (len(r) - 1)))

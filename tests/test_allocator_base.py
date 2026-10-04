import pandas as pd
import pytest

from tests.fixtures.synthetic import fit_context, small_market
from workbench.allocators.base import (
    AllocationResult,
    Allocator,
    FitContext,
    check_weights,
    weights_from_riskfolio,
)
from workbench.data.synthetic import placeholder_saa
from workbench.policy.compiled import CompiledPolicy


def _ctx(saa, candidate="CAND"):
    return FitContext(
        as_of=pd.Timestamp("2020-12-31"), saa=saa, candidate=candidate, policy=CompiledPolicy()
    )


def test_fit_context_accepts_placeholder_saa():
    ctx = _ctx(placeholder_saa())
    assert ctx.saa["CAND"] == 0


def test_fit_context_rejects_bad_saa():
    saa = placeholder_saa()
    with pytest.raises(ValueError, match="not in SAA"):
        _ctx(saa, candidate="OTHER")
    nonzero = saa.copy()
    nonzero["CAND"], nonzero["CASH"] = 0.05, 0.0
    with pytest.raises(ValueError, match="must be 0"):
        _ctx(nonzero)
    with pytest.raises(ValueError, match="sum to 1"):
        _ctx(saa * 0.9)


def test_allocation_result_status_rules():
    w = pd.Series({"A": 0.4, "B": 0.6})
    assert AllocationResult(w, "ok").status == "ok"
    assert AllocationResult(None, "infeasible", "no solution").weights is None
    with pytest.raises(ValueError, match="requires weights"):
        AllocationResult(None, "ok")
    with pytest.raises(ValueError, match="must not carry"):
        AllocationResult(w, "infeasible")
    with pytest.raises(ValueError, match="status must be"):
        AllocationResult(None, "failed")
    with pytest.raises(ValueError, match="sum to 1"):
        AllocationResult(pd.Series({"A": 0.5, "B": 0.6}), "ok")


def test_check_weights_long_only():
    with pytest.raises(ValueError, match="negative"):
        check_weights(pd.Series({"A": 1.2, "B": -0.2}), long_only=True)
    check_weights(pd.Series({"A": 1.2, "B": -0.2}))  # allowed when not long-only


def test_weights_from_riskfolio():
    df = pd.DataFrame({"weights": [0.25, 0.75]}, index=["A", "B"])
    s = weights_from_riskfolio(df)
    assert isinstance(s, pd.Series) and list(s.index) == ["A", "B"]
    with pytest.raises(ValueError):
        weights_from_riskfolio(df.rename(columns={"weights": "w"}))


class _HoldSAA:
    """Minimal allocator used to check the protocol end to end."""

    name = "hold_saa"

    def params(self) -> dict:
        return {}

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        return AllocationResult(ctx.saa[returns.columns], "ok")


def test_protocol_conformance_on_synthetic_data():
    data = small_market()
    alloc = _HoldSAA()
    assert isinstance(alloc, Allocator)
    res = alloc.fit(data.returns, fit_context(data))
    assert res.status == "ok"
    assert res.weights["CAND"] == 0

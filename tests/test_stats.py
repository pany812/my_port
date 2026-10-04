import math

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import small_market
from workbench.data.synthetic import placeholder_saa
from workbench.evaluation import stats
from workbench.evaluation.metrics import cell_metrics, risk_shares

R = pd.Series([0.10, -0.20, 0.05, 0.10, -0.05, 0.02])


def test_wealth_and_drawdowns_by_hand():
    w = stats.wealth(R).to_numpy()
    assert w == pytest.approx([1.1, 0.88, 0.924, 1.0164, 0.96558, 0.9848916])
    dd = stats.drawdowns(R).to_numpy()
    # peak 1.1 until 1.0164 < 1.1 ... never regained
    assert dd == pytest.approx([0, 0.2, 0.16, 1 - 1.0164 / 1.1, 1 - 0.96558 / 1.1,
                                1 - 0.9848916 / 1.1])  # fmt: skip
    assert stats.max_drawdown(R) == pytest.approx(0.2)


def test_drawdown_counts_initial_wealth_as_peak():
    assert stats.drawdowns(pd.Series([-0.1, 0.05])).iloc[0] == pytest.approx(0.1)


def test_ann_return_and_vol():
    total = float(np.prod(1 + R))
    assert stats.ann_return(R, "M") == pytest.approx(total ** (12 / 6) - 1)
    assert stats.ann_vol(R, "M") == pytest.approx(R.std(ddof=1) * math.sqrt(12))


def test_cvar_and_cdar_tail_means():
    # T=6, alpha=5% -> ceil(0.3) = 1 observation in the tail
    assert stats.cvar(R) == pytest.approx(0.20)
    assert stats.cdar(R) == pytest.approx(0.20)
    r = pd.Series(np.linspace(-0.10, 0.09, 40))  # 40 obs -> 2 in the tail
    assert stats.cvar(r) == pytest.approx(-(r.iloc[0] + r.iloc[1]) / 2)


def test_tracking_error_is_demeaned():
    bench = R - 0.01  # constant active return -> zero TE
    assert stats.tracking_error(R, bench, "M") == pytest.approx(0.0, abs=1e-15)


def test_risk_shares_sum_to_one_and_zero_weight_is_zero():
    window = small_market().returns
    w = placeholder_saa()[window.columns]
    for rm in ("MV", "CVaR", "CDaR"):
        s = risk_shares(w, window, rm)
        assert s.sum() == pytest.approx(1.0)
        assert s["CAND"] == 0.0


def test_cell_metrics_contents_and_saa_reference_values():
    window = small_market().returns
    saa = placeholder_saa()[window.columns]
    metrics, errors = cell_metrics(saa, window, saa, "CAND", ("MV", "CVaR"), "M")
    assert errors == {}
    d = {(m, lens): v for m, lens, v in metrics}
    assert d[("candidate_risk_share", "MV")] == 0.0
    assert d[("in_sample.te_vs_saa", "")] == pytest.approx(0.0, abs=1e-15)
    assert {m for m, _ in d if m.startswith("in_sample.")} == {
        f"in_sample.{s}" for s in ("ann_return", "ann_vol", "cvar95", "cdar95", "max_dd",
                                   "te_vs_saa")
    }  # fmt: skip


def test_cell_metrics_reports_errors_without_raising():
    window = small_market().returns
    saa = placeholder_saa()[window.columns]
    metrics, errors = cell_metrics(saa, window, saa, "CAND", ("NOT_A_LENS",), "M")
    assert "candidate_risk_share:NOT_A_LENS" in errors
    assert any(m == "in_sample.ann_vol" for m, _, _ in metrics)

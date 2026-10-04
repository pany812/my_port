import math

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import long_daily_market, small_market
from workbench.data.synthetic import PLACEHOLDER_SAA, CandidateSpec, generate, placeholder_saa
from workbench.units import vol_annual_to_period


def _skew(x: pd.Series) -> float:
    x = x - x.mean()
    return float((x**3).mean() / (x**2).mean() ** 1.5)


def _excess_kurtosis(x: pd.Series) -> float:
    x = x - x.mean()
    return float((x**4).mean() / (x**2).mean() ** 2 - 3.0)


def test_shape_and_index():
    data = small_market()
    assert data.returns.shape == (120, 9)
    assert data.returns.index[0] == pd.Timestamp("2011-01-31")
    assert data.returns.index[-1] == pd.Timestamp("2020-12-31")
    assert data.returns.index.is_month_end.all()
    assert data.candidate == "CAND"
    assert not data.returns.isna().any().any()
    assert not data.backfilled.any()


def test_default_range_matches_example_spec():
    data = generate(seed=1)  # 2006-01 .. 2026-09 monthly
    assert len(data.returns) == 249
    assert data.returns.index[-1] == pd.Timestamp("2026-09-30")


def test_deterministic_given_seed():
    pd.testing.assert_frame_equal(small_market(seed=7).returns, small_market(seed=7).returns)
    assert not small_market(seed=7).returns.equals(small_market(seed=8).returns)


def test_placeholder_saa():
    saa = placeholder_saa()
    assert saa.sum() == pytest.approx(1.0)
    assert saa["CAND"] == 0
    assert (PLACEHOLDER_SAA["min"] <= PLACEHOLDER_SAA["weight"]).all()
    assert (PLACEHOLDER_SAA["weight"] <= PLACEHOLDER_SAA["max"]).all()


@pytest.mark.parametrize("rho,skew", [(0.0, 0.3), (0.5, 0.4), (-0.3, -0.5)])
def test_candidate_hits_vol_corr_skew_targets(rho, skew):
    data = long_daily_market(vol_annual=0.12, corr_to_equity=rho, skew=skew)
    c = data.returns["CAND"]
    assert c.std() == pytest.approx(vol_annual_to_period(0.12, "D"), rel=0.05)
    assert c.corr(data.returns["GL_EQ"]) == pytest.approx(rho, abs=0.05)
    assert _skew(c) == pytest.approx(skew, abs=0.12)


def test_building_block_vols_and_correlation():
    data = long_daily_market()
    r = data.returns
    for asset, row in PLACEHOLDER_SAA.iterrows():
        assert r[asset].std() == pytest.approx(vol_annual_to_period(row.vol_annual, "D"), rel=0.05)
    assert r["SE_EQ"].corr(r["GL_EQ"]) == pytest.approx(0.80, abs=0.04)
    assert r["SE_GOV"].corr(r["GL_EQ"]) == pytest.approx(-0.25, abs=0.04)


def test_fat_tails_option():
    gauss = long_daily_market(seed=3)
    fat = long_daily_market(seed=3, tail_df=5.0)
    assert _excess_kurtosis(fat.returns["GL_EQ"]) > _excess_kurtosis(gauss.returns["GL_EQ"]) + 1.0
    # volatility preserved by the unit-variance scaling
    assert fat.returns["GL_EQ"].std() == pytest.approx(vol_annual_to_period(0.15, "D"), rel=0.08)


def test_live_start_flags_backfill():
    data = small_market(live_start="2018-01")
    pre = data.returns.index < pd.Timestamp("2018-01")
    assert data.backfilled[pre].all() and not data.backfilled[~pre].any()
    assert data.backfilled.name == "CAND_backfilled"
    assert not data.returns["CAND"].isna().any()
    live = data.live_only()
    assert len(live.returns) == 36 and live.returns.index[0] == pd.Timestamp("2018-01-31")
    assert not live.backfilled.any()


def test_live_start_without_backfill_is_nan():
    data = small_market(live_start="2018-01", backfill=False)
    pre = data.returns.index < pd.Timestamp("2018-01")
    assert data.returns.loc[pre, "CAND"].isna().all()
    assert not data.returns.loc[~pre].isna().any().any()
    assert not data.backfilled.any()  # nothing proxied, just missing


def test_flag_never_appears_as_asset_column():
    data = small_market(live_start="2018-01")
    assert "CAND_backfilled" not in data.returns.columns
    assert set(data.asset_class.index) == set(data.returns.columns)
    assert data.asset_class["CAND"] == "alternatives"


def test_unattainable_skew_raises():
    with pytest.raises(ValueError, match="unattainable"):
        generate(seed=1, candidate=CandidateSpec(skew=0.9, corr_to_equity=0.6))


@pytest.mark.parametrize(
    "kwargs",
    [{"asset_id": "GL_EQ"}, {"equity_ref": "NOPE"}, {"corr_to_equity": 1.5}, {"vol_annual": 0.0}],
)
def test_invalid_candidate_specs(kwargs):
    with pytest.raises(ValueError):
        generate(seed=1, candidate=CandidateSpec(**kwargs))


def test_monthly_mean_matches_compound_target():
    # mean of simple monthly returns over a long draw ~ (1 + mu)^(1/12) - 1
    data = generate(seed=11, start="1700-01", end="2200-12", candidate=CandidateSpec(skew=0.0))
    target = 1.05 ** (1 / 12) - 1
    se = vol_annual_to_period(0.10, "M") / math.sqrt(len(data.returns))
    assert abs(data.returns["CAND"].mean() - target) < 4 * se
    assert np.isfinite(data.returns.to_numpy()).all()

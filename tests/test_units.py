import math

import pytest

from workbench import units


def test_periods_per_year():
    assert units.periods_per_year("M") == 12
    assert units.periods_per_year("D") == 252
    with pytest.raises(ValueError, match="unknown frequency"):
        units.periods_per_year("monthly")


def test_te_monthly_is_annual_over_sqrt12():
    assert units.te_annual_to_period(0.02, "M") == pytest.approx(0.02 / math.sqrt(12))


@pytest.mark.parametrize("freq", ["D", "W", "M", "Q", "A"])
@pytest.mark.parametrize("compounding", ["geometric", "arithmetic"])
def test_return_round_trip(freq, compounding):
    r = units.return_annual_to_period(0.05, freq, compounding)
    assert units.return_period_to_annual(r, freq, compounding) == pytest.approx(0.05)


def test_geometric_vs_arithmetic_monthly():
    assert units.return_annual_to_period(0.12, "M", "arithmetic") == pytest.approx(0.01)
    assert units.return_annual_to_period(0.12, "M") == pytest.approx(1.12 ** (1 / 12) - 1)


@pytest.mark.parametrize("freq", ["D", "W", "M", "Q", "A"])
def test_vol_round_trip(freq):
    s = units.vol_annual_to_period(0.15, freq)
    assert units.vol_period_to_annual(s, freq) == pytest.approx(0.15)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        units.vol_annual_to_period(-0.1, "M")
    with pytest.raises(ValueError):
        units.return_annual_to_period(-1.0, "M")
    with pytest.raises(ValueError):
        units.return_annual_to_period(0.05, "M", "continuous")

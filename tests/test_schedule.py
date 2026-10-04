import pandas as pd
import pytest

from tests.fixtures.synthetic import small_market
from workbench.backtest.schedule import rebalance_dates, window_at
from workbench.grid.spec import WindowSpec

IDX = small_market().returns.index  # 2011-01 .. 2020-12 month ends
ROLL60 = WindowSpec("rolling", periods=60)


def test_monthly_schedule_bounds():
    d = rebalance_dates(IDX, ROLL60, "M", "M")
    assert d[0] == pd.Timestamp("2015-12-31")  # 60th observation
    assert d[-1] == pd.Timestamp("2020-11-30")  # last obs has no following period
    assert len(d) == 60


def test_quarterly_and_annual():
    q = rebalance_dates(IDX, ROLL60, "Q", "M")
    assert all(x.month in (3, 6, 9, 12) for x in q)
    assert q[0] == pd.Timestamp("2015-12-31") and q[-1] == pd.Timestamp("2020-09-30")
    a = rebalance_dates(IDX, ROLL60, "A", "M")
    assert a == [pd.Timestamp(f"{y}-12-31") for y in range(2015, 2020)]


def test_partial_period_at_end_is_not_a_period_end():
    idx = IDX[IDX <= "2020-08-31"]  # data stops mid-quarter
    q = rebalance_dates(idx, ROLL60, "Q", "M")
    assert q[-1] == pd.Timestamp("2020-06-30")


def test_start_and_expanding():
    d = rebalance_dates(IDX, ROLL60, "M", "M", start="2018-01")
    assert d[0] == pd.Timestamp("2018-01-31")
    e = rebalance_dates(IDX, WindowSpec("expanding", min_periods=24), "M", "M")
    assert e[0] == pd.Timestamp("2012-12-31")


def test_insufficient_history_gives_empty_schedule():
    assert rebalance_dates(IDX[:60], ROLL60, "M", "M") == []


def test_invalid_frequency():
    with pytest.raises(ValueError, match="cannot rebalance"):
        rebalance_dates(IDX, ROLL60, "M", "Q")
    with pytest.raises(ValueError, match="must be one of"):
        rebalance_dates(IDX, ROLL60, "W", "M")


def test_window_at_never_includes_future():
    r = small_market().returns
    t = pd.Timestamp("2017-06-30")
    w = window_at(r, t, ROLL60)
    assert w.index[-1] == t and len(w) == 60 and (w.index <= t).all()
    assert len(window_at(r, t, WindowSpec("expanding", min_periods=24))) == 78
    with pytest.raises(ValueError, match="insufficient history"):
        window_at(r, pd.Timestamp("2014-01-31"), ROLL60)

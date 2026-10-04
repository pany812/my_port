"""Known-answer validation of the inference functions on synthetic data.

Monte Carlo tolerances are ~3 binomial standard errors; seeds are fixed so results are exact.
"""

import math

import numpy as np
import pytest

from workbench.evaluation.inference import (
    alpha_tests,
    benjamini_hochberg,
    block_length,
    deflated_sharpe,
    expected_max_sharpe,
    linear_f_test,
    pbo_cscv,
    probabilistic_sharpe,
    sharpe_diff_test,
    spanning_tests,
)

T = 129


def _pair(rng, sr_a, sr_b, rho=0.7, n=T):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return 0.03 * z[:, 0] + 0.03 * sr_a, 0.03 * z[:, 1] + 0.03 * sr_b


def _binom_ok(rate, p, n, k=3.0):
    return abs(rate - p) <= k * math.sqrt(p * (1 - p) / n)


# --- Sharpe difference -----------------------------------------------------------------------


def test_sharpe_test_size_and_power():
    rng = np.random.default_rng(1)
    size = np.mean([sharpe_diff_test(*_pair(rng, 0.15, 0.15), n_boot=499, seed=i).p_value < 0.05
                    for i in range(300)])  # fmt: skip
    assert size <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 300)  # not oversized
    assert size >= 0.01  # and not degenerate
    power = np.mean([sharpe_diff_test(*_pair(rng, 0.45, 0.0), n_boot=499, seed=i).p_value < 0.05
                     for i in range(100)])  # fmt: skip
    assert power > 0.9


def test_sharpe_test_degenerate_and_deterministic():
    rng = np.random.default_rng(2)
    a, b = _pair(rng, 0.2, 0.1)
    r1 = sharpe_diff_test(a, b, n_boot=199, seed=5)
    r2 = sharpe_diff_test(a, b, n_boot=199, seed=5)
    assert r1 == r2 and 0 < r1.p_value <= 1
    assert r1.diff == pytest.approx(a.mean() / a.std() - b.mean() / b.std())
    same = sharpe_diff_test(a, a.copy())
    assert np.isnan(same.p_value) and same.note == "identical to SAA path"
    assert block_length(129) == 6 and block_length(125) == 5


def test_sharpe_test_handles_paths_equal_on_most_dates():
    rng = np.random.default_rng(3)
    b = rng.normal(0.005, 0.03, T)
    a = b.copy()
    a[:3] += 0.01  # differs on three dates only: many resamples have no spread
    r = sharpe_diff_test(a, b, n_boot=499, seed=0)
    assert np.isfinite(r.p_value)


# --- deflated Sharpe ratio -------------------------------------------------------------------


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(7)
    n = 50
    maxs, stars = [], []
    for _ in range(300):
        r = rng.normal(0, 0.02, (n, T))
        sr = r.mean(1) / r.std(1, ddof=1)
        maxs.append(sr.max())
        stars.append(expected_max_sharpe(n, sr.var(ddof=1)))
    assert np.mean(maxs) == pytest.approx(np.mean(stars), rel=0.03)
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_dsr_false_positives_bounded_and_skill_detected():
    rng = np.random.default_rng(8)
    fp = []
    for _ in range(200):
        d = deflated_sharpe({str(j): rng.normal(0, 0.02, T) for j in range(50)})
        fp.append(max(d.values(), key=lambda r: r.sr).dsr > 0.95)
    assert np.mean(fp) <= 0.05  # conservative under the null
    hits = []
    for _ in range(100):
        trials = {str(j): rng.normal(0, 0.02, T) for j in range(9)}
        trials["skill"] = rng.normal(0.01, 0.02, T)  # SR 0.5 per month
        hits.append(deflated_sharpe(trials)["skill"].dsr > 0.95)
    assert np.mean(hits) > 0.8


def test_psr_formula_and_degenerate_trials():
    assert probabilistic_sharpe(0.0, 0.0, 100, 0.0, 3.0) == pytest.approx(0.5)
    # skewness and fat tails widen the SR standard error
    p_normal = probabilistic_sharpe(0.2, 0.0, 60, 0.0, 3.0)
    p_fat = probabilistic_sharpe(0.2, 0.0, 60, -1.0, 8.0)
    assert p_fat < p_normal
    d = deflated_sharpe({"flat": np.zeros(T), "x": np.random.default_rng(0).normal(0, 0.01, T)})
    assert np.isnan(d["flat"].dsr) and d["flat"].note == "no active risk"
    assert d["x"].n_trials == 2  # degenerate trials still count


# --- PBO -----------------------------------------------------------------------------------


def test_pbo_noise_near_half_and_real_signal_near_zero():
    rng = np.random.default_rng(9)
    noise = [pbo_cscv(rng.normal(0, 0.02, (T, 30))).pbo for _ in range(80)]
    assert np.mean(noise) == pytest.approx(0.5, abs=0.08)
    m = rng.normal(0, 0.02, (T, 30))
    m[:, 0] += 0.01
    assert pbo_cscv(m).pbo < 0.05
    r = pbo_cscv(m)
    assert r.n_combinations == math.comb(10, 5) and r.n_trials == 30


def test_pbo_input_validation():
    with pytest.raises(ValueError, match="at least 2"):
        pbo_cscv(np.zeros((50, 1)))
    with pytest.raises(ValueError, match="even"):
        pbo_cscv(np.zeros((50, 3)), n_splits=5)


# --- spanning ------------------------------------------------------------------------------


def _bench(rng, k=8):
    return rng.normal(0.005, 0.03, (T, k))


def test_hk_matches_restricted_regression():
    rng = np.random.default_rng(10)
    x = _bench(rng)
    y = x @ rng.dirichlet(np.ones(8)) + 0.001 + rng.normal(0, 0.01, T)
    s = spanning_tests(y, x)
    z = np.column_stack([np.ones(T), x])
    ssr_u = np.sum((y - z @ np.linalg.lstsq(z, y, rcond=None)[0]) ** 2)
    xr, yr = x[:, :-1] - x[:, [-1]], y - x[:, -1]  # alpha = 0, sum(beta) = 1
    ssr_r = np.sum((yr - xr @ np.linalg.lstsq(xr, yr, rcond=None)[0]) ** 2)
    f = ((ssr_r - ssr_u) / 2) / (ssr_u / (T - 8 - 1))
    assert s.hk.statistic == pytest.approx(f, rel=1e-10)
    assert (s.hk.df1, s.hk.df2) == (2, T - 9)


def test_spanning_size_and_power():
    rng = np.random.default_rng(11)
    x = _bench(rng)

    def cand(alpha, het=False):
        e = rng.normal(0, 1, T) * 0.01
        if het:
            e *= 0.3 + 3 * np.abs(x[:, 1]) / 0.03
        return x @ rng.dirichlet(np.ones(8)) + alpha + e

    n = 1000
    hk = np.mean([spanning_tests(cand(0.0), x).hk.p_value < 0.05 for _ in range(n)])
    f1 = np.mean([spanning_tests(cand(0.0), x).kz_f1.p_value < 0.05 for _ in range(n)])
    rob_het = np.mean([spanning_tests(cand(0.0, het=True), x).hk_robust.p_value < 0.05
                       for _ in range(n)])  # fmt: skip
    assert _binom_ok(hk, 0.05, n) and _binom_ok(f1, 0.05, n)
    assert rob_het <= 0.05 + 3 * math.sqrt(0.0475 / n)  # robust version holds size
    power = np.mean([spanning_tests(cand(0.004), x).hk.p_value < 0.05 for _ in range(200)])
    assert power > 0.9


def test_alpha_tests_with_riskless_asset():
    rng = np.random.default_rng(12)
    x = _bench(rng, 7)
    n = 1000
    size = np.mean([alpha_tests(x @ np.full(7, 0.1) + rng.normal(0, 0.01, T), x)
                    .exact.p_value < 0.05 for _ in range(n)])  # fmt: skip
    assert _binom_ok(size, 0.05, n)
    a = alpha_tests(x @ np.full(7, 0.1) + 0.006 + rng.normal(0, 0.01, T), x)
    assert a.exact.p_value < 0.01 and a.alpha == pytest.approx(0.006, abs=0.003)
    assert a.robust.df1 == 1 and a.robust.df2 == T - 8


def test_linear_f_test_single_restriction_equals_t_squared():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(T, 3))
    y = x @ [0.5, -0.2, 0.1] + rng.normal(size=T)
    z = np.column_stack([np.ones(T), x])
    theta, *_ = np.linalg.lstsq(z, y, rcond=None)
    resid = y - z @ theta
    s2 = resid @ resid / (T - 4)
    se0 = math.sqrt(s2 * np.linalg.inv(z.T @ z)[0, 0])
    f = linear_f_test(z, y, np.eye(1, 4), np.zeros(1))
    assert f.statistic == pytest.approx((theta[0] / se0) ** 2)


# --- multiple testing ----------------------------------------------------------------------


def test_benjamini_hochberg():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.2])
    adj = benjamini_hochberg(p)
    # step-up: sorted 0.01, 0.03, 0.04, 0.2 -> 0.04, 0.06, 0.0533, 0.2 -> running min from top
    assert np.allclose(adj[[0, 1, 2, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert np.isnan(adj[3])
    assert np.isnan(benjamini_hochberg(np.array([np.nan]))).all()

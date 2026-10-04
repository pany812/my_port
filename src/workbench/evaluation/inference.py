"""Statistical inference for out-of-sample evidence. Pure numpy/scipy, no I/O.

All inputs are per-period simple returns (decimal) on a common date index. Sharpe and
information ratios are per period unless a name says ``_ann``.

- :func:`sharpe_diff_test`: Ledoit & Wolf (2008), studentised circular-block bootstrap of
  H0: SR_a = SR_b.
- :func:`deflated_sharpe`: Bailey & López de Prado (2014) deflated Sharpe ratio.
- :func:`pbo_cscv`: probability of backtest overfitting by combinatorially symmetric
  cross-validation (Bailey, Borwein, López de Prado & Zhu 2017).
- :func:`spanning_tests`: Huberman–Kandel and Kan–Zhou step-down spanning tests (exact F
  under homoskedastic normal errors) plus an HC3-robust Wald version of the joint test, scaled
  to F(2, T-K-1). In our simulations (T=129, K=8) the HC3-F keeps ~4% size under both
  homoskedastic and heteroskedastic errors, where the exact F drifts to ~7% and a Newey–West
  chi2 Wald over-rejects at ~10%; it is the line to trust on real data.
- :func:`benjamini_hochberg`: FDR-adjusted p-values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329
DEGENERATE_TOL = 1e-12


def block_length(n: int) -> int:
    """Block-length rule for the bootstrap: ceil(T^(1/3)), at least 1."""
    return max(1, math.ceil(n ** (1.0 / 3.0)))


# --- Sharpe difference (Ledoit & Wolf 2008) --------------------------------------------


@dataclass(frozen=True)
class SharpeDiffResult:
    sr_a: float
    sr_b: float
    diff: float
    se: float
    p_value: float
    block: int
    n_boot: int
    note: str = ""


def _sr_diff_and_grad(m: np.ndarray):
    """m = (mu_a, mu_b, g_a, g_b) with g = E[r^2]. Returns (diff, gradient) along axis 0."""
    mu_a, mu_b, g_a, g_b = m
    va, vb = g_a - mu_a**2, g_b - mu_b**2
    diff = mu_a / np.sqrt(va) - mu_b / np.sqrt(vb)
    grad = np.stack([g_a / va**1.5, -g_b / vb**1.5, -mu_a / (2 * va**1.5), mu_b / (2 * vb**1.5)])
    return diff, grad


def _block_psi(y: np.ndarray, b: int) -> np.ndarray:
    """Block-based long-run covariance of y (..., n, 4), n = n_blocks * b: mean of zeta zeta'."""
    *lead, n, k = y.shape
    n_blocks = n // b
    yc = y - y.mean(axis=-2, keepdims=True)
    zeta = yc[..., : n_blocks * b, :].reshape(*lead, n_blocks, b, k).sum(axis=-2) / math.sqrt(b)
    return np.einsum("...ji,...jk->...ik", zeta, zeta) / n_blocks


def sharpe_diff_test(
    a: np.ndarray, b_: np.ndarray, n_boot: int = 4999, block: int | None = None, seed: int = 0
) -> SharpeDiffResult:
    """Two-sided test of equal Sharpe ratios for paired return series ``a`` and ``b_``.

    p-value = (1 + #{d* >= d}) / (1 + n_boot) with d = |diff| / se and d* the studentised
    bootstrap statistic centred at the sample difference (Ledoit & Wolf 2008, Remark 3.2).
    """
    a = np.asarray(a, dtype=float)
    b_ = np.asarray(b_, dtype=float)
    n = len(a)
    blk = block or block_length(n)
    if np.max(np.abs(a - b_)) < DEGENERATE_TOL:
        sr = float(a.mean() / a.std(ddof=0)) if a.std() > DEGENERATE_TOL else float("nan")
        return SharpeDiffResult(sr, sr, 0.0, 0.0, float("nan"), blk, 0, "identical to SAA path")
    if a.std() < DEGENERATE_TOL or b_.std() < DEGENERATE_TOL:
        return SharpeDiffResult(float("nan"), float("nan"), float("nan"), float("nan"),
                                float("nan"), blk, 0, "zero volatility")  # fmt: skip
    n_blocks = n // blk
    m_len = n_blocks * blk
    if n_blocks < 2:
        return SharpeDiffResult(float("nan"), float("nan"), float("nan"), float("nan"),
                                float("nan"), blk, 0, "too few observations")  # fmt: skip

    def moments(x, y):
        return np.stack([x.mean(-1), y.mean(-1), (x**2).mean(-1), (y**2).mean(-1)])

    def series(x, y):
        return np.stack([x, y, x**2, y**2], axis=-1)

    m0 = moments(a, b_)
    diff, grad = _sr_diff_and_grad(m0)
    psi = _block_psi(series(a, b_)[None, -m_len:], blk)[0]
    se = float(np.sqrt(grad @ psi @ grad / n))

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(blk)).reshape(n_boot, m_len) % n
    xa, xb = a[idx], b_[idx]
    mb = moments(xa, xb)
    diff_b, grad_b = _sr_diff_and_grad(mb)
    psi_b = _block_psi(series(xa, xb), blk)
    quad = np.einsum("ib,bij,jb->b", grad_b, psi_b, grad_b)
    se_b = np.sqrt(np.maximum(quad, 0.0) / m_len)  # rounding can make quad slightly < 0
    d0 = abs(diff) / se
    # A resample with no spread (paths equal on every drawn date) has se_b = 0: count it as
    # extreme, which can only raise the p-value (conservative).
    with np.errstate(divide="ignore", invalid="ignore"):
        d_star = np.where(se_b > DEGENERATE_TOL, np.abs(diff_b - diff) / se_b, np.inf)
    p = (1 + int(np.sum(d_star >= d0))) / (1 + n_boot)
    return SharpeDiffResult(float(m0[0] / math.sqrt(m0[2] - m0[0] ** 2)),
                            float(m0[1] / math.sqrt(m0[3] - m0[1] ** 2)),
                            float(diff), se, float(p), blk, n_boot)  # fmt: skip


# --- Deflated Sharpe ratio ---------------------------------------------------------------


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """SR* = sqrt(V) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))); 0 for N < 2."""
    if n_trials < 2 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe(sr: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """PSR: P(true SR > benchmark) given the sample SR of ``n`` observations (per period).

    kurt is the (non-excess) kurtosis.
    """
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0 or n < 2:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


@dataclass(frozen=True)
class DSRResult:
    sr: float
    dsr: float
    psr0: float
    sr_star: float
    n_trials: int
    skew: float
    kurt: float
    note: str = ""


def deflated_sharpe(returns_by_trial: dict[str, np.ndarray]) -> dict[str, DSRResult]:
    """Deflated Sharpe ratio for every trial, deflating for all trials in the dict.

    N counts every trial; the variance of Sharpe ratios uses the non-degenerate ones.
    Degenerate trials (no volatility) get NaN with a note.
    """
    n_trials = len(returns_by_trial)
    srs = {}
    for k, r in returns_by_trial.items():
        r = np.asarray(r, dtype=float)
        s = r.std(ddof=1)
        srs[k] = float(r.mean() / s) if s > DEGENERATE_TOL else float("nan")
    valid = np.array([v for v in srs.values() if np.isfinite(v)])
    var = float(valid.var(ddof=1)) if len(valid) > 1 else float("nan")
    sr_star = expected_max_sharpe(n_trials, var)
    out = {}
    for k, r in returns_by_trial.items():
        r = np.asarray(r, dtype=float)
        sr = srs[k]
        if not np.isfinite(sr):
            out[k] = DSRResult(sr, float("nan"), float("nan"), sr_star, n_trials,
                               float("nan"), float("nan"), "no active risk")  # fmt: skip
            continue
        sk = float(stats.skew(r, bias=False))
        ku = float(stats.kurtosis(r, fisher=False, bias=False))
        out[k] = DSRResult(sr, probabilistic_sharpe(sr, sr_star, len(r), sk, ku),
                           probabilistic_sharpe(sr, 0.0, len(r), sk, ku),
                           sr_star, n_trials, sk, ku)  # fmt: skip
    return out


# --- PBO by CSCV ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    n_combinations: int
    n_trials: int
    logits: np.ndarray = field(repr=False)


def _ratio(sums: np.ndarray, sq: np.ndarray, n: int) -> np.ndarray:
    mean = sums / n
    var = np.maximum(sq / n - mean**2, 0.0) * n / max(n - 1, 1)
    sd = np.sqrt(var)
    return np.where(sd > DEGENERATE_TOL, mean / np.where(sd > 0, sd, 1.0), 0.0)


def pbo_cscv(matrix: np.ndarray, n_splits: int = 10) -> PBOResult:
    """PBO for a (T x N) matrix of per-period returns of N trials.

    Rows are cut into ``n_splits`` contiguous blocks; for every half/half combination the trial
    with the best in-sample Sharpe is selected and its out-of-sample relative rank w recorded;
    PBO = share of combinations with logit(w) <= 0. Trials without volatility score 0.
    """
    m = np.asarray(matrix, dtype=float)
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least 2 trials")
    if n_splits % 2 or n_splits < 2 or t < n_splits:
        raise ValueError("n_splits must be even, >= 2 and <= number of periods")
    blocks = np.array_split(np.arange(t), n_splits)
    bsum = np.stack([m[b].sum(0) for b in blocks])
    bsq = np.stack([(m[b] ** 2).sum(0) for b in blocks])
    blen = np.array([len(b) for b in blocks])
    logits = []
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        mask = np.zeros(n_splits, bool)
        mask[list(is_blocks)] = True
        is_r = _ratio(bsum[mask].sum(0), bsq[mask].sum(0), int(blen[mask].sum()))
        oos_r = _ratio(bsum[~mask].sum(0), bsq[~mask].sum(0), int(blen[~mask].sum()))
        best = int(np.argmax(is_r))
        rank = stats.rankdata(oos_r)[best]  # 1..N, average ties
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
    logits_arr = np.array(logits)
    return PBOResult(float(np.mean(logits_arr <= 0)), n_splits, len(logits), n, logits_arr)


# --- Spanning tests ------------------------------------------------------------------------


@dataclass(frozen=True)
class FTest:
    statistic: float
    p_value: float
    df1: int
    df2: int


@dataclass(frozen=True)
class SpanningResult:
    alpha: float  # per period
    beta_sum: float
    hk: FTest  # H0: alpha = 0 and sum(beta) = 1
    kz_f1: FTest  # H0: alpha = 0 (tangency portfolio not improved)
    kz_f2: FTest  # H0: sum(beta) = 1 given alpha = 0 (GMV portfolio not improved)
    hk_robust: FTest  # HC3-robust Wald of the joint test, W / 2 ~ F(2, T - K - 1)
    n_obs: int
    n_assets: int


def _ols(z: np.ndarray, y: np.ndarray):
    zz_inv = np.linalg.inv(z.T @ z)
    theta = zz_inv @ z.T @ y
    resid = y - z @ theta
    return theta, resid, zz_inv


def linear_f_test(z: np.ndarray, y: np.ndarray, r: np.ndarray, q: np.ndarray) -> FTest:
    """Exact F test of R theta = q in y = Z theta + e (homoskedastic normal errors)."""
    theta, resid, zz_inv = _ols(z, y)
    t, p = z.shape
    s2 = resid @ resid / (t - p)
    d = r @ theta - q
    f = float(d @ np.linalg.inv(r @ zz_inv @ r.T) @ d / len(q) / s2)
    df1, df2 = len(q), t - p
    return FTest(f, float(stats.f.sf(f, df1, df2)), df1, df2)


def _hc3_cov(z: np.ndarray, resid: np.ndarray, zz_inv: np.ndarray) -> np.ndarray:
    """HC3 heteroskedasticity-robust covariance of OLS coefficients."""
    h = np.einsum("ij,jk,ik->i", z, zz_inv, z)
    u = resid / (1 - h)
    return zz_inv @ (z.T * u**2) @ z @ zz_inv


@dataclass(frozen=True)
class AlphaResult:
    alpha: float  # per period, excess return over the riskless asset
    exact: FTest  # H0: alpha = 0, homoskedastic normal errors
    robust: FTest  # HC3-robust Wald, F(1, T - K - 1)
    n_obs: int
    n_assets: int


def alpha_tests(y_excess: np.ndarray, x_excess: np.ndarray) -> AlphaResult:
    """Spanning with a riskless asset: H0 alpha = 0 in excess returns (tangency not improved).

    With a riskless asset among the benchmarks the GMV leg of the Huberman–Kandel test is
    degenerate (the near-constant asset absorbs any beta shortfall), so only alpha is tested.
    """
    y = np.asarray(y_excess, dtype=float)
    x = np.asarray(x_excess, dtype=float)
    t, k = x.shape
    z = np.column_stack([np.ones(t), x])
    theta, resid, zz_inv = _ols(z, y)
    r = np.zeros((1, k + 1))
    r[0, 0] = 1.0
    exact = linear_f_test(z, y, r, np.zeros(1))
    v = _hc3_cov(z, resid, zz_inv)
    f_rob = float(theta[0] ** 2 / v[0, 0])
    robust = FTest(f_rob, float(stats.f.sf(f_rob, 1, t - k - 1)), 1, t - k - 1)
    return AlphaResult(float(theta[0]), exact, robust, t, k)


def spanning_tests(y: np.ndarray, x: np.ndarray) -> SpanningResult:
    """Does test asset ``y`` (T) expand the frontier of benchmark assets ``x`` (T x K)?"""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    t, k = x.shape
    z = np.column_stack([np.ones(t), x])
    theta, resid, zz_inv = _ols(z, y)
    r_hk = np.zeros((2, k + 1))
    r_hk[0, 0] = 1.0
    r_hk[1, 1:] = 1.0
    q_hk = np.array([0.0, 1.0])
    hk = linear_f_test(z, y, r_hk, q_hk)
    f1 = linear_f_test(z, y, r_hk[:1], q_hk[:1])
    f2 = linear_f_test(x, y, np.ones((1, k)), np.array([1.0]))  # model without intercept
    v = _hc3_cov(z, resid, zz_inv)
    d = r_hk @ theta - q_hk
    f_rob = float(d @ np.linalg.inv(r_hk @ v @ r_hk.T) @ d) / 2
    robust = FTest(f_rob, float(stats.f.sf(f_rob, 2, t - k - 1)), 2, t - k - 1)
    return SpanningResult(float(theta[0]), float(theta[1:].sum()), hk, f1, f2, robust, t, k)


# --- multiple testing ----------------------------------------------------------------------


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (FDR); NaNs are ignored and stay NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out

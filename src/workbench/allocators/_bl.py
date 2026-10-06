"""Black–Litterman with the SAA as the prior, and the candidate's breakeven view.

Shared by ``riskfolio_bl`` and ``skfolio_bl``. All quantities per period of the returns unless
named ``*_annual``; excess returns are over the policy ``rf``.

Prior. Equilibrium excess returns implied by holding the SAA: ``pi = delta * Sigma @ w_SAA``, with
risk aversion ``delta = SR / sigma_SAA`` from an assumed SAA Sharpe ratio (frequency-free and
always positive; Riskfolio-Lib's default delta comes from the window's historical mean and is
negative in about a third of 36-month windows on synthetic data).

View. One absolute view on the candidate c, "expected excess return q", with confidence k in
(0, 1]. With the view uncertainty proportional to the prior's (Omega = tau (1/k - 1) P Sigma P',
He–Litterman / Idzorek), tau cancels and the posterior mean is

    mu_BL - rf = pi + k (q - pi_c) * Sigma[:, c] / Sigma[c, c]

The candidate moves the fraction k of the way from equilibrium to the view (the posterior
*premium* a = k (q - pi_c)); every other asset moves by its regression beta on the candidate.
Verified against Riskfolio-Lib ``black_litterman`` (k = 0.5) and skfolio ``BlackLitterman`` (any
tau and k). The covariance stays the estimator's (our ``hist=True`` convention): BL changes mu.

Breakeven. Without constraints, max-Sharpe on (mu_BL, Sigma) returns the SAA plus the candidate
funded pro rata, with candidate weight x = a / (delta sigma_c^2 + a). So the posterior premium for
weight x is a*(x) = delta sigma_c^2 x / (1 - x), and the candidate's required Sharpe ratio is
SR_SAA * (rho + (sigma_c / sigma_SAA) * x / (1 - x)), rho = corr(candidate, SAA). Under policy
constraints there is no closed form; :func:`breakeven` root-finds a on the optimiser itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from workbench.allocators._solve import Infeasible
from workbench.allocators.base import FitContext
from workbench.policy.skfolio import max_candidate_weight
from workbench.units import (
    return_annual_to_period,
    return_period_to_annual,
    sharpe_annual_to_period,
    sharpe_period_to_annual,
    vol_period_to_annual,
)

OBJECTIVES = ("Sharpe", "Utility")
WEIGHT_TOL = 1e-4  # breakeven: |candidate weight - target| <= WEIGHT_TOL (decimal weight)
MAX_SHARPE_ANNUAL = 3.0  # the root-finder gives up where the candidate would need a Sharpe > 3
MAX_ITER = 60
MIN_WIDTH = 1e-9  # per period: stop bisecting; solver noise (~1e-4 in weight) can jump the target

# solve(view excess per period, confidence) -> weights, or None when the optimiser finds none
Solve = Callable[[float, float], pd.Series | pd.DataFrame | None]
MakeSolve = Callable[["Prior"], Solve]


@dataclass(frozen=True)
class Prior:
    """Equilibrium moments on one fitting window (per period).

    delta:     risk aversion (dimensionless).
    pi:        equilibrium excess return per asset (Series).
    beta:      Sigma[:, c] / Sigma[c, c] per asset: the move per unit of candidate premium.
    sigma_c:   candidate volatility; sigma_saa: SAA volatility; rho: corr(candidate, SAA).
    """

    candidate: str
    delta: float
    pi: pd.Series
    beta: pd.Series
    sigma_c: float
    sigma_saa: float
    rho: float

    @classmethod
    def build(cls, cov: pd.DataFrame, saa: pd.Series, candidate: str, sharpe: float) -> Prior:
        """``cov`` per period; ``saa`` weights; ``sharpe`` the SAA's Sharpe ratio per period."""
        assets = list(cov.columns)
        s = cov.to_numpy(dtype=float)
        w = saa.reindex(assets).to_numpy(dtype=float)
        sigma_saa = math.sqrt(float(w @ s @ w))
        if not sigma_saa > 0:
            raise ValueError("SAA volatility is zero on this window; the prior is undefined")
        ic = assets.index(candidate)
        var_c = float(s[ic, ic])
        if not var_c > 0:
            raise ValueError("candidate variance is zero on this window")
        delta = sharpe / sigma_saa
        pi = pd.Series(delta * s @ w, index=assets)
        beta = pd.Series(s[:, ic] / var_c, index=assets)
        rho = float((s @ w)[ic] / (sigma_saa * math.sqrt(var_c)))
        return cls(candidate, delta, pi, beta, math.sqrt(var_c), sigma_saa, rho)

    @property
    def pi_c(self) -> float:
        return float(self.pi[self.candidate])

    def premium(self, view: float, confidence: float) -> float:
        """Posterior premium of the candidate over equilibrium for a stated view."""
        return confidence * (view - self.pi_c)

    def mu(self, premium: float, rf: float) -> pd.Series:
        """Posterior expected total returns per period: rf + pi + premium * beta."""
        return rf + self.pi + premium * self.beta

    def unconstrained_premium(self, x: float) -> float:
        """Closed-form posterior premium giving weight ``x`` with no constraints (max-Sharpe)."""
        return self.delta * self.sigma_c**2 * x / (1.0 - x)

    def candidate_sharpe(self, premium: float) -> float:
        """The candidate's posterior Sharpe ratio per period at ``premium``."""
        return (self.pi_c + premium) / self.sigma_c


def posterior_excess(
    pi: np.ndarray, cov: np.ndarray, ic: int, view: float, confidence: float, tau: float
) -> np.ndarray:
    """Textbook BL posterior mean (excess) for one absolute view on asset ``ic``.

    Reference implementation for tests: Omega = tau (1/k - 1) P Sigma P' (zero at k = 1).
    """
    p = np.zeros((1, len(pi)))
    p[0, ic] = 1.0
    v = tau * cov @ p.T
    omega = tau * (1.0 / confidence - 1.0) * (p @ cov @ p.T)
    return pi + (v @ np.linalg.solve(p @ v + omega, np.array([view]) - p @ pi)).ravel()


@dataclass(frozen=True)
class BLParams:
    """The parameters both BL allocators share (see ``RiskfolioBlackLitterman``)."""

    view_annual: float | None
    confidence: float
    target_weight: float | None
    prior_sharpe: float
    obj: str

    def check(self) -> None:
        if (self.view_annual is None) == (self.target_weight is None):
            raise ValueError("give exactly one of view_annual (view mode) and target_weight "
                             "(breakeven mode)")  # fmt: skip
        if not 0 < self.confidence <= 1:
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}")
        if self.target_weight is not None:
            if not 0 < self.target_weight < 1:
                raise ValueError(f"target_weight must be in (0, 1), got {self.target_weight}")
            if self.confidence != 1.0:
                raise ValueError("confidence applies to view_annual only; a breakeven is the "
                                 "posterior premium (at confidence k the stated view is "
                                 "equilibrium + premium / k)")  # fmt: skip
        if not self.prior_sharpe > 0:
            raise ValueError(f"prior_sharpe must be > 0, got {self.prior_sharpe}")
        if self.obj not in OBJECTIVES:
            raise ValueError(f"obj must be one of {OBJECTIVES}, got {self.obj!r}")


@dataclass
class Breakeven:
    premium: float  # posterior premium per period
    weights: pd.Series
    n_solves: int
    jump: bool  # the weight jumps over the target (solver noise or a kink): first one above it


def breakeven(
    solve_premium: Callable[[float], pd.Series],
    candidate: str,
    target: float,
    first_guess: float,
    max_premium: float,
    ceiling: float = 1.0,
    annual: Callable[[float], float] = lambda a: a,
    tol: float = WEIGHT_TOL,
) -> Breakeven:
    """Posterior premium at which the optimiser gives the candidate weight ``target``.

    ``solve_premium(a)`` returns the optimal weights at premium ``a`` (per period). The weight is
    non-decreasing in the premium for these objectives (checked in the tests), so: bracket from
    premium 0 (the SAA when it is feasible) and the unconstrained closed form ``first_guess``,
    doubling until the target is reached, then bisect to ``tol``.

    Raises ``Infeasible`` ("unreachable") when the target exceeds ``ceiling`` (the most the
    policy allows the candidate), when it is not reached below ``max_premium``, or when the
    optimiser fails at the extreme premiums of the bracket search. ``annual`` converts a
    per-period premium for messages.
    """
    if target > ceiling + tol:
        raise Infeasible(f"target weight {target:.2%} unreachable: the policy allows the "
                         f"candidate at most {ceiling:.2%}")  # fmt: skip
    n = 0

    def at(a: float) -> tuple[float, pd.Series]:
        nonlocal n
        n += 1
        w = solve_premium(a)
        return float(w[candidate]), w

    def found(a: float, w: pd.Series, jump: bool = False) -> Breakeven:
        return Breakeven(a, w, n, jump)

    def gave_up(x: float, a: float, why: str) -> Infeasible:
        return Infeasible(f"target weight {target:.2%} unreachable: the candidate's weight "
                          f"reaches {x:.2%} at a posterior premium of {annual(a):.2%} p.a. and "
                          f"{why}")  # fmt: skip

    lo, (x_lo, w_lo) = 0.0, at(0.0)
    if abs(x_lo - target) <= tol:
        return found(lo, w_lo)
    step = max(first_guess, 1e-9)
    if x_lo < target:
        hi = step
        while True:
            try:
                x_hi, w_hi = at(hi)
            except (Infeasible, cp.error.SolverError):
                raise gave_up(x_lo, lo, "the optimiser fails beyond it") from None
            if x_hi >= target - tol:
                break
            lo, x_lo, w_lo = hi, x_hi, w_hi
            hi *= 2.0
            if hi > max_premium:
                raise gave_up(x_lo, lo, "more would need a Sharpe ratio above "
                                        f"{MAX_SHARPE_ANNUAL:g}")  # fmt: skip
    else:  # the SAA is not the zero-premium solution here and already holds more than target
        hi, x_hi, w_hi = lo, x_lo, w_lo
        lo = -step
        x_lo, w_lo = at(lo)
        while x_lo > target + tol:
            if lo < -max_premium:
                raise Infeasible(f"target weight {target:.2%} unreachable: the candidate's "
                                 f"weight stays at {x_lo:.2%} or above")  # fmt: skip
            hi, x_hi, w_hi = lo, x_lo, w_lo
            lo *= 2.0
            x_lo, w_lo = at(lo)
    for _ in range(MAX_ITER):
        if abs(x_lo - target) <= tol:
            return found(lo, w_lo)
        if abs(x_hi - target) <= tol:
            return found(hi, w_hi)
        if hi - lo < MIN_WIDTH:
            return found(hi, w_hi, jump=True)
        mid = 0.5 * (lo + hi)
        x_mid, w_mid = at(mid)
        if x_mid < target:
            lo, x_lo, w_lo = mid, x_mid, w_mid
        else:
            hi, x_hi, w_hi = mid, x_mid, w_mid
    return found(hi, w_hi, jump=abs(x_hi - target) > tol)


def fit_black_litterman(
    p: BLParams,
    returns: pd.DataFrame,
    ctx: FitContext,
    cov: pd.DataFrame,
    make_solve: MakeSolve,
    diag: dict,
) -> pd.Series | pd.DataFrame | None:
    """Run one BL cell: view mode solves once; breakeven mode root-finds the premium.

    ``make_solve(prior)`` returns the library-specific ``solve(view, confidence)`` (posterior +
    optimiser). Diagnostics are annual (excess returns arithmetic x n, Sharpe x sqrt(n)).
    """
    freq = ctx.policy.freq
    rf = ctx.policy.rf
    prior = Prior.build(cov, ctx.saa, ctx.candidate, sharpe_annual_to_period(p.prior_sharpe, freq))
    solve = make_solve(prior)

    def annual(x: float) -> float:
        return return_period_to_annual(x, freq, "arithmetic")

    diag |= {
        "delta": prior.delta,
        "equilibrium_excess_annual": annual(prior.pi_c),
        "rho_saa": prior.rho,
        "candidate_vol_annual": vol_period_to_annual(prior.sigma_c, freq),
        "rf_annual": annual(rf),
    }  # fmt: skip
    if p.target_weight is None:
        view = return_annual_to_period(p.view_annual, freq, "arithmetic")
        a = prior.premium(view, p.confidence)
        diag |= {"mode": "view", "premium_annual": annual(a),
                 "posterior_excess_annual": annual(prior.pi_c + a),
                 "candidate_sharpe_annual": sharpe_period_to_annual(prior.candidate_sharpe(a),
                                                                    freq)}  # fmt: skip
        return solve(view, p.confidence)

    x = p.target_weight
    a0 = prior.unconstrained_premium(x)
    diag |= {"mode": "breakeven",
             "unconstrained_premium_annual": annual(a0),
             "unconstrained_sharpe_annual": sharpe_period_to_annual(prior.candidate_sharpe(a0),
                                                                    freq)}  # fmt: skip

    def solve_premium(a: float) -> pd.Series:
        w = solve(prior.pi_c + a, 1.0)
        if w is None:
            raise Infeasible(f"optimiser found no solution at a posterior premium of "
                             f"{annual(a):.2%} p.a.")  # fmt: skip
        return w["weights"].astype(float) if isinstance(w, pd.DataFrame) else w.astype(float)

    max_premium = sharpe_annual_to_period(MAX_SHARPE_ANNUAL, freq) * prior.sigma_c - prior.pi_c
    ceiling = max_candidate_weight(ctx.policy, returns, ctx.candidate)
    diag["max_candidate_weight"] = ceiling
    be = breakeven(solve_premium, ctx.candidate, x, a0, max_premium, ceiling, annual)
    diag |= {"premium_annual": annual(be.premium),
             "posterior_excess_annual": annual(prior.pi_c + be.premium),
             "candidate_sharpe_annual": sharpe_period_to_annual(
                 prior.candidate_sharpe(be.premium), freq),
             "achieved_weight": float(be.weights[ctx.candidate]),
             "n_solves": be.n_solves, "jump": be.jump}  # fmt: skip
    return be.weights

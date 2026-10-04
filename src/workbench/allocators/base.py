"""Allocator protocol and the objects that cross the allocator boundary.

Core signatures (ask before changing): ``Allocator.fit(returns, ctx) -> AllocationResult``,
``FitContext`` and ``AllocationResult``. ``fit`` is deterministic given its inputs and seed and
does no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from workbench.policy.compiled import CompiledPolicy

Status = Literal["ok", "infeasible", "solver_error", "exception"]
STATUSES: tuple[str, ...] = ("ok", "infeasible", "solver_error", "exception")

WEIGHT_SUM_TOL = 1e-6


@dataclass(frozen=True)
class FitContext:
    """Everything an allocator may know besides the return history.

    as_of:       last date of information; the allocator sees returns dated <= as_of and its
                 weights apply from the next period.
    saa:         strategic policy weights indexed by asset id, summing to 1, candidate at 0.
    candidate:   asset id of the candidate strategy; must be in ``saa.index``.
    policy:      compiled constraints, already in per-period units.
    mu_override: optional expected returns per period (decimal), indexed by asset id.
    """

    as_of: pd.Timestamp
    saa: pd.Series
    candidate: str
    policy: CompiledPolicy
    mu_override: pd.Series | None = None

    def __post_init__(self) -> None:
        if self.candidate not in self.saa.index:
            raise ValueError(f"candidate {self.candidate!r} not in SAA index")
        if self.saa[self.candidate] != 0:
            raise ValueError(f"SAA weight of candidate must be 0, got {self.saa[self.candidate]}")
        if abs(float(self.saa.sum()) - 1.0) > WEIGHT_SUM_TOL:
            raise ValueError(f"SAA weights must sum to 1, got {self.saa.sum():.8f}")
        if self.saa.index.has_duplicates:
            raise ValueError("SAA index has duplicate asset ids")


@dataclass
class AllocationResult:
    """Outcome of one ``fit``.

    weights:     pd.Series indexed by asset id summing to 1 when ``status == "ok"``, else None.
    status:      one of ``STATUSES``. Failures are recorded, never dropped.
    message:     solver output captured from stdout, or the exception text.
    elapsed_s:   wall-clock seconds spent in ``fit``.
    diagnostics: free-form, JSON-serialisable details (solver used, realised risk shares, ...).
    """

    weights: pd.Series | None
    status: Status
    message: str = ""
    elapsed_s: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.status == "ok":
            if self.weights is None:
                raise ValueError('status "ok" requires weights')
            check_weights(self.weights)
        elif self.weights is not None:
            raise ValueError(f'status "{self.status}" must not carry weights')


def check_weights(w: pd.Series, long_only: bool = False, tol: float = WEIGHT_SUM_TOL) -> None:
    """Raise ValueError unless ``w`` is a finite weight vector summing to 1 (within ``tol``)."""
    if not isinstance(w, pd.Series):
        raise ValueError(f"weights must be a pd.Series, got {type(w).__name__}")
    if w.index.has_duplicates:
        raise ValueError("weights index has duplicate asset ids")
    if not np.isfinite(w.to_numpy(dtype=float)).all():
        raise ValueError("weights contain non-finite values")
    if abs(float(w.sum()) - 1.0) > tol:
        raise ValueError(f"weights must sum to 1, got {w.sum():.8f}")
    if long_only and (w < -tol).any():
        raise ValueError("long-only weights contain negative values")


def weights_from_riskfolio(w: pd.DataFrame) -> pd.Series:
    """Convert Riskfolio-Lib output (one-column DataFrame named ``weights``) to a pd.Series."""
    if not isinstance(w, pd.DataFrame) or list(w.columns) != ["weights"]:
        raise ValueError("expected a one-column DataFrame named 'weights'")
    s = w["weights"].astype(float).copy()
    s.name = "weight"
    return s


@runtime_checkable
class Allocator(Protocol):
    """A portfolio construction rule. Implementations live in ``workbench.allocators``."""

    name: str

    def params(self) -> dict:
        """JSON-serialisable parameters that, with ``name``, fully identify the allocator."""
        ...

    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
        """Fit on ``returns`` (simple returns per period, dated <= ctx.as_of)."""
        ...

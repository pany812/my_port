"""Compiled policy: the constraint set handed to allocators, already in per-period units.

M1 holds only what every allocator needs. M2 adds the Riskfolio-Lib constraint tables
(A/B from ``rp.assets_constraints``), the TE limit vs the SAA, the per-asset band and HC bounds,
all produced by ``workbench.policy.compiler`` from a spec's ``constraint_set``.
"""

from __future__ import annotations

from dataclasses import dataclass

from workbench.units import periods_per_year


@dataclass(frozen=True)
class CompiledPolicy:
    """Constraints and limits for one grid cell.

    freq: return frequency code of the experiment (see ``workbench.units``).
    rf:   risk-free rate per period of ``freq`` (decimal). Convert from annual with
          ``units.return_annual_to_period`` before constructing.
    long_only: weights must be >= 0.
    """

    freq: str = "M"
    rf: float = 0.0
    long_only: bool = True

    def __post_init__(self) -> None:
        periods_per_year(self.freq)  # validates the code

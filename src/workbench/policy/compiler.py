"""Policy compiler: SAA + spec ``constraint_set`` -> :class:`CompiledPolicy` (per-period units)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import pandas as pd

from workbench.policy.compiled import CompiledPolicy
from workbench.policy.saa import SAA
from workbench.units import return_annual_to_period, te_annual_to_period


@dataclass(frozen=True)
class ConstraintSet:
    """A named constraint set as written in a spec (annual units).

    candidate_cap: maximum candidate weight (decimal), or None.
    asset_ranges:  apply the SAA's per-asset [lower, upper] ranges.
    class_limits:  {class: [lo, hi]} on summed class weight (decimal).
    band:          per-asset band |w_i - saa_i| <= band (decimal weight).
    te_annual:     annual tracking-error limit vs the SAA (decimal).
    """

    name: str
    candidate_cap: float | None = None
    asset_ranges: bool = False
    class_limits: dict[str, tuple[float, float]] | None = None
    band: float | None = None
    te_annual: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConstraintSet:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"unknown constraint_set keys {sorted(unknown)}; known: {sorted(known)}"
            )
        if "name" not in d:
            raise ValueError("constraint_set needs a name")
        kw = dict(d)
        if kw.get("class_limits"):
            kw["class_limits"] = {
                k: (float(v[0]), float(v[1])) for k, v in kw["class_limits"].items()
            }
        return cls(**kw)


def compile_policy(
    saa: SAA,
    cs: ConstraintSet,
    freq: str = "M",
    rf_annual: float = 0.0,
    solvers: tuple[str, ...] = ("CLARABEL",),
) -> CompiledPolicy:
    """Compile ``cs`` against ``saa`` for return frequency ``freq``.

    rf_annual: annual risk-free rate (decimal), converted geometrically to per period.
    TE is converted annual -> per period in ``workbench.units``.
    """
    assets = saa.assets
    lower = saa.lower[assets].astype(float) if cs.asset_ranges else None
    upper = saa.upper[assets].astype(float) if cs.asset_ranges else None
    if cs.candidate_cap is not None:
        if not 0 <= cs.candidate_cap <= 1:
            raise ValueError(f"candidate_cap must be in [0, 1], got {cs.candidate_cap}")
        upper = pd.Series(1.0, index=assets) if upper is None else upper.copy()
        upper[saa.candidate] = min(float(upper[saa.candidate]), cs.candidate_cap)
        if lower is None:
            lower = pd.Series(0.0, index=assets)
    if cs.band is not None and cs.band < 0:
        raise ValueError(f"band must be >= 0, got {cs.band}")
    if cs.te_annual is not None and cs.te_annual <= 0:
        raise ValueError(f"te_annual must be > 0, got {cs.te_annual}")
    if cs.class_limits:
        unknown = set(cs.class_limits) - set(saa.asset_class)
        if unknown:
            raise ValueError(f"class_limits for unknown classes {sorted(unknown)}")

    return CompiledPolicy(
        name=cs.name,
        freq=freq,
        rf=return_annual_to_period(rf_annual, freq),
        lower=lower,
        upper=upper,
        asset_class=saa.asset_class[assets].copy(),
        class_limits=dict(cs.class_limits or {}),
        band=cs.band,
        te=None if cs.te_annual is None else te_annual_to_period(cs.te_annual, freq),
        benchweights=saa.weights[assets].astype(float).copy(),
        solvers=tuple(solvers),
    )

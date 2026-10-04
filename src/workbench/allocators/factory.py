"""Spec ``type`` name -> allocator class, parameter validation and construction."""

from __future__ import annotations

from dataclasses import fields

from workbench.allocators.base import Allocator
from workbench.allocators.naive import EqualWeight, InverseVol, SAAPlus, StaticSAA
from workbench.allocators.riskfolio_hc import RiskfolioHC
from workbench.allocators.riskfolio_mr import RiskfolioMeanRisk
from workbench.allocators.skfolio_hc import SkfolioHC
from workbench.allocators.skfolio_mr import SkfolioMeanRisk

ALLOCATOR_TYPES: dict[str, type] = {
    "static_saa": StaticSAA,
    "saa_plus": SAAPlus,
    "equal_weight": EqualWeight,
    "inverse_vol": InverseVol,
    "riskfolio_mean_risk": RiskfolioMeanRisk,
    "riskfolio_hc": RiskfolioHC,
    "skfolio_mean_risk": SkfolioMeanRisk,
    "skfolio_hc": SkfolioHC,
}

ESTIMATOR_KEYS = ("method_mu", "method_cov")


def uses_estimator(type_: str) -> bool:
    """True if the allocator takes ``method_mu``/``method_cov`` from the spec's estimators."""
    names = {f.name for f in fields(_cls(type_))}
    return set(ESTIMATOR_KEYS) <= names


def allowed_params(type_: str) -> set[str]:
    """Parameters a spec may set for ``type_`` (estimator keys come from ``grid.estimators``)."""
    return {f.name for f in fields(_cls(type_))} - set(ESTIMATOR_KEYS)


def build(type_: str, params: dict, estimator: dict | None = None) -> Allocator:
    """Construct an allocator from a spec type, scalar params and an optional estimator."""
    kwargs = dict(params)
    if estimator is not None:
        if not uses_estimator(type_):
            raise ValueError(f"{type_} does not take an estimator")
        kwargs.update(estimator)
    return _cls(type_)(**kwargs)


def _cls(type_: str) -> type:
    try:
        return ALLOCATOR_TYPES[type_]
    except KeyError:
        raise ValueError(
            f"unknown allocator type {type_!r}; known: {sorted(ALLOCATOR_TYPES)}"
        ) from None

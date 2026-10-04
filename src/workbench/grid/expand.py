"""Deterministic grid expansion: spec -> ordered list of cell configurations.

Order: constraint set -> allocator entry (spec order) -> parameter product (keys and values in
spec order) -> estimator. Allocators without an estimator are not repeated per estimator.
``config_id`` hashes the configuration's content, so it is stable under spec reordering.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

from workbench.allocators.factory import uses_estimator
from workbench.grid.spec import ExperimentSpec, canonical_json, constraint_set_dict


@dataclass(frozen=True)
class CellConfig:
    """One point of the grid (before choosing a window end or data variant).

    index:          position in the deterministic expansion order (0-based).
    config_id:      content hash of (allocator, params, estimator, constraint set).
    allocator:      spec type name.
    params:         scalar parameters for the allocator.
    estimator:      {"method_mu", "method_cov"} or None for allocators that take none.
    constraint_set: constraint-set name.
    """

    index: int
    config_id: str
    allocator: str
    params: dict
    estimator: dict | None
    constraint_set: str


def expand(spec: ExperimentSpec) -> list[CellConfig]:
    cells: list[CellConfig] = []
    for cs in spec.constraint_sets:
        cs_content = constraint_set_dict(cs)
        for entry in spec.allocators:
            keys = list(entry.params)
            estimators = spec.estimators if uses_estimator(entry.type) else (None,)
            for values in itertools.product(*(entry.params[k] for k in keys)):
                params = dict(zip(keys, values, strict=True))
                for est in estimators:
                    est = None if est is None else dict(est)
                    cid = config_id(entry.type, params, est, cs_content)
                    cells.append(CellConfig(len(cells), cid, entry.type, params, est, cs.name))
    ids = [c.config_id for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("grid contains duplicate configurations")
    return cells


def config_id(allocator: str, params: dict, estimator: dict | None, cs_content: dict) -> str:
    payload = {"allocator": allocator, "params": params, "estimator": estimator,
               "constraint_set": cs_content}  # fmt: skip
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]

"""Strategic asset allocation (SAA): policy weights, asset classes and per-asset ranges.

Versions: ``example`` / ``placeholder`` are the synthetic placeholder (code). Any other version
is a reviewed YAML file ``<WB_SAA_DIR or ./saa>/<version>.yaml`` (P2-M8), so git history is the
governance trail. A file version's content hash enters the experiment id: editing the file
under the same version name gives a new experiment, never a silent re-use.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from workbench.data.synthetic import PLACEHOLDER_SAA

SAA_DIR_ENV = "WB_SAA_DIR"
CODE_VERSIONS = ("example", "placeholder")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class SAAError(ValueError):
    """An SAA version cannot be found or its file is invalid."""


_TOL = 1e-9


@dataclass(frozen=True)
class SAA:
    """A versioned strategic allocation including the candidate at weight 0.

    weights:     policy weights (decimal) indexed by asset id, summing to 1.
    asset_class: class label per asset id.
    lower/upper: per-asset policy range (decimal weights), lower <= weight <= upper.
    candidate:   asset id of the candidate (weight 0).
    """

    version: str
    weights: pd.Series
    asset_class: pd.Series
    lower: pd.Series
    upper: pd.Series
    candidate: str
    source: str = "code"  # "code" (placeholder) or "file" (P2-M8)

    def __post_init__(self) -> None:
        idx = self.weights.index
        if idx.has_duplicates:
            raise ValueError("SAA has duplicate asset ids")
        for name in ("asset_class", "lower", "upper"):
            if set(getattr(self, name).index) != set(idx):
                raise ValueError(f"SAA {name} must cover exactly the weight index")
        if self.candidate not in idx:
            raise ValueError(f"candidate {self.candidate!r} not in SAA")
        if self.weights[self.candidate] != 0:
            raise ValueError("candidate must have SAA weight 0")
        if abs(float(self.weights.sum()) - 1.0) > 1e-6:
            raise ValueError(f"SAA weights must sum to 1, got {self.weights.sum():.8f}")
        lo, hi = self.lower[idx], self.upper[idx]
        if ((lo - self.weights) > _TOL).any() or ((self.weights - hi) > _TOL).any():
            raise ValueError("SAA weights must lie within [lower, upper]")

    @property
    def assets(self) -> list[str]:
        return list(self.weights.index)

    def content_hash(self) -> str:
        """sha256 over the version's weights, classes and ranges (asset order irrelevant)."""
        doc = {a: [float(self.weights[a]), str(self.asset_class[a]), float(self.lower[a]),
                   float(self.upper[a])] for a in sorted(self.weights.index)}  # fmt: skip
        payload = json.dumps({"candidate": self.candidate, "assets": doc}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_version(cls, version: str, candidate: str = "CAND") -> SAA:
        """The placeholder for ``example`` / ``placeholder``, else ``saa/<version>.yaml``."""
        if version in CODE_VERSIONS:
            return cls.placeholder(candidate)
        if not _VERSION.match(version):
            raise SAAError(f"SAA version {version!r}: use letters, digits, '_', '.', '-'")
        path = Path(os.environ.get(SAA_DIR_ENV, "saa")) / f"{version}.yaml"
        if not path.exists():
            raise SAAError(f"unknown SAA version {version!r}: no file {path} (set "
                             f"${SAA_DIR_ENV} to the SAA directory)")  # fmt: skip
        return cls.from_file(path, candidate)

    @classmethod
    def from_file(cls, path: str | Path, candidate: str = "CAND") -> SAA:
        """Read an SAA file::

            version: house_2026
            description: "..."
            candidate_class: alternatives
            assets:
              SE_EQ: {class: equity, weight: 0.15, min: 0.10, max: 0.20}

        Weights and ranges are decimals; the candidate is added at weight 0, range [0, 1]."""
        path = Path(path)
        d = yaml.safe_load(path.read_text())
        if not isinstance(d, dict) or not isinstance(d.get("assets"), dict) or not d["assets"]:
            raise SAAError(f"{path}: expected a mapping with a non-empty 'assets' mapping")
        unknown = set(d) - {"version", "description", "candidate_class", "assets"}
        if unknown:
            raise SAAError(f"{path}: unknown keys {sorted(unknown)}")
        if str(d.get("version")) != path.stem:
            raise SAAError(f"{path}: version {d.get('version')!r} must equal the file name")
        if candidate in d["assets"]:
            raise SAAError(f"{path}: the candidate {candidate!r} must not be an SAA asset")
        rows = {}
        for a, v in d["assets"].items():
            if not isinstance(v, dict) or set(v) != {"class", "weight", "min", "max"}:
                raise SAAError(f"{path}: asset {a} needs exactly class, weight, min, max")
            rows[str(a)] = v
        weights = pd.Series({a: float(v["weight"]) for a, v in rows.items()})
        asset_class = pd.Series({a: str(v["class"]) for a, v in rows.items()})
        lower = pd.Series({a: float(v["min"]) for a, v in rows.items()})
        upper = pd.Series({a: float(v["max"]) for a, v in rows.items()})
        weights[candidate], asset_class[candidate] = 0.0, str(d.get("candidate_class", "candidate"))
        lower[candidate], upper[candidate] = 0.0, 1.0
        return cls(path.stem, weights, asset_class, lower, upper, candidate, source="file")

    @classmethod
    def placeholder(cls, candidate: str = "CAND", candidate_class: str = "alternatives") -> SAA:
        """The synthetic-phase placeholder SAA; candidate range [0, 1] (capped by the policy)."""
        t = PLACEHOLDER_SAA
        weights = t["weight"].astype(float).copy()
        asset_class = t["asset_class"].copy()
        lower = t["min"].astype(float).copy()
        upper = t["max"].astype(float).copy()
        weights[candidate], asset_class[candidate] = 0.0, candidate_class
        lower[candidate], upper[candidate] = 0.0, 1.0
        return cls("placeholder", weights, asset_class, lower, upper, candidate)

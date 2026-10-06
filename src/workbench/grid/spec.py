"""``ExperimentSpec``: strict YAML parsing, defaults, canonical form and ``spec_hash``.

Unknown keys are errors. Allocator types and parameter names are validated at load time.
``spec_hash`` is the sha256 of the canonical JSON of the normalised spec, excluding the
cosmetic ``experiment`` name.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from workbench.allocators._estimates import CMA_MU
from workbench.allocators.factory import allowed_params, uses_estimator
from workbench.data.cma import CMA, PLACEHOLDER, CMAVector, placeholder_cma
from workbench.policy.compiler import ConstraintSet
from workbench.units import periods_per_year


class SpecError(ValueError):
    """The spec is malformed. The message names the offending path."""


@dataclass(frozen=True)
class SyntheticParams:
    """Synthetic candidate and market parameters (annual units, decimals)."""

    mu_annual: float = 0.05
    vol_annual: float = 0.10
    corr_to_equity: float = 0.0
    skew: float = 0.3
    live_start: str | None = None
    backfill: bool = True
    tail_df: float | None = None


@dataclass(frozen=True)
class DataSpec:
    """Data source and conventions. ``base_currency`` and ``hedging`` are always explicit."""

    source: str
    frequency: str
    start: str
    end: str
    base_currency: str
    hedging: str
    candidate: str
    synthetic: SyntheticParams | None = None


@dataclass(frozen=True)
class WindowSpec:
    """Estimation window. rolling: last ``periods`` observations. expanding: all, at least
    ``min_periods``."""

    kind: str
    periods: int | None = None
    min_periods: int = 24


@dataclass(frozen=True)
class RebalanceSpec:
    """calendar: trade to the new target at every ``every`` date. threshold: fit at every
    ``every`` date but trade only when some asset drifts more than ``band`` (decimal weight)
    from the target."""

    kind: str = "calendar"
    every: str = "M"
    band: float | None = None


@dataclass(frozen=True)
class CostsSpec:
    """One-way transaction costs in basis points of traded weight, per building block."""

    default_bps: float = 0.0
    per_asset: dict[str, float] | None = None

    def bps(self, asset: str) -> float:
        return float((self.per_asset or {}).get(asset, self.default_bps))


@dataclass(frozen=True)
class LiquiditySpec:
    """Candidate dealing terms. dealing: M/Q/A calendar of dealing dates. notice_periods:
    dealing dates between a redemption decision and its execution. gate: maximum fraction of
    the candidate position redeemable per dealing date (None = no gate)."""

    dealing: str = "M"
    notice_periods: int = 0
    gate: float | None = None


@dataclass(frozen=True)
class BacktestSpec:
    """walk_forward: fit at every rebalance date, out-of-sample path (default).
    in_sample: a single fit at the end of the data, no path. ``start``: first rebalance date."""

    mode: str = "walk_forward"
    start: str | None = None


@dataclass(frozen=True)
class AllocatorEntry:
    """One allocator block; every param value is a list of alternatives (grid dimension)."""

    type: str
    params: dict[str, list]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    seed: int
    data: DataSpec
    saa_version: str
    funding: str
    window: WindowSpec
    rebalance: RebalanceSpec
    backtest: BacktestSpec
    estimators: tuple[dict[str, str], ...]
    allocators: tuple[AllocatorEntry, ...]
    constraint_sets: tuple[ConstraintSet, ...]
    risk_lenses: tuple[str, ...]
    costs: CostsSpec | None = None
    liquidity: LiquiditySpec | None = None
    cma: CMA | None = None
    rf_annual: float = 0.0
    solvers: tuple[str, ...] = ("CLARABEL",)
    source_text: str | None = None  # raw YAML as written; stored in the registry, not hashed

    def canonical(self) -> dict[str, Any]:
        """Normalised, JSON-serialisable form with every default filled in (no name, no text).

        Optional sections added after Phase 1 (costs, liquidity, rebalance.band, cma) enter only
        when set, so specs that do not use them keep their spec_hash. The CMA enters resolved
        (every vector's values), so changing a CMA changes the hash even under the same version.
        """
        rebalance = asdict(self.rebalance)
        if rebalance["band"] is None:
            del rebalance["band"]
        out = {
            "seed": self.seed,
            "data": asdict(self.data),
            "saa": {"version": self.saa_version},
            "funding": self.funding,
            "window": asdict(self.window),
            "rebalance": rebalance,
            "backtest": asdict(self.backtest),
            "grid": {
                "estimators": [dict(e) for e in self.estimators],
                "allocators": [{"type": a.type, **a.params} for a in self.allocators],
                "constraint_sets": [constraint_set_dict(cs) for cs in self.constraint_sets],
            },
            "risk_lenses": list(self.risk_lenses),
            "rf_annual": self.rf_annual,
            "solvers": list(self.solvers),
        }
        if self.costs is not None:
            out["costs"] = asdict(self.costs)
        if self.liquidity is not None:
            out["liquidity"] = asdict(self.liquidity)
        if self.cma is not None:
            out["cma"] = self.cma.canonical()
        return out

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical()).encode()).hexdigest()

    @property
    def yaml_text(self) -> str:
        """The YAML as written, or a dump of the canonical form for specs built in code."""
        if self.source_text is not None:
            return self.source_text
        doc = _drop_none({"experiment": self.experiment, **self.canonical()})
        return yaml.safe_dump(doc, sort_keys=False)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_spec(path: str | Path) -> ExperimentSpec:
    text = Path(path).read_text()
    return parse_spec(text)


def parse_spec(src: str | dict) -> ExperimentSpec:
    """Parse YAML text or an already-loaded mapping into a validated ``ExperimentSpec``."""
    text = src if isinstance(src, str) else None
    raw = yaml.safe_load(src) if isinstance(src, str) else src
    if not isinstance(raw, dict):
        raise SpecError("spec must be a mapping")
    top = _keys(
        raw,
        "",
        required={"experiment", "seed", "data", "saa", "window", "grid"},
        optional={
            "funding",
            "rebalance",
            "backtest",
            "risk_lenses",
            "rf_annual",
            "solvers",
            "costs",
            "liquidity",
            "cma",
        },
    )
    data = _data(top["data"])
    grid = _keys(
        top["grid"], "grid", required={"allocators", "constraint_sets"}, optional={"estimators"}
    )
    saa = _keys(top["saa"], "saa", required={"version"})
    funding = top.get("funding", "pro_rata")
    allocator_list = _list(grid["allocators"], "grid.allocators")
    allocators = tuple(
        _allocator(a, f"grid.allocators[{i}]", funding) for i, a in enumerate(allocator_list)
    )
    estimators = _estimators(grid.get("estimators"), allocators)
    cma = _cma(top["cma"], data) if "cma" in top else None
    uses_cma = any(e["method_mu"] == CMA_MU for e in estimators)
    if uses_cma and cma is None:
        raise SpecError("grid.estimators uses method_mu: cma but the spec has no cma section")
    if cma is not None and not uses_cma:
        raise SpecError("cma is set but no grid.estimators entry uses method_mu: cma")
    constraint_sets = _constraint_sets(grid["constraint_sets"])
    seed = top["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SpecError(f"seed must be an integer, got {seed!r}")
    return ExperimentSpec(
        experiment=str(top["experiment"]),
        seed=seed,
        data=data,
        saa_version=str(saa["version"]),
        funding=str(funding),
        window=_window(top["window"]),
        rebalance=_rebalance(top.get("rebalance", {}), data.frequency),
        backtest=_backtest(top.get("backtest", {})),
        costs=_costs(top["costs"]) if "costs" in top else None,
        liquidity=_liquidity(top["liquidity"], data.frequency) if "liquidity" in top else None,
        cma=cma,
        estimators=estimators,
        allocators=allocators,
        constraint_sets=constraint_sets,
        risk_lenses=tuple(str(x) for x in top.get("risk_lenses", ["MV"])),
        rf_annual=float(top.get("rf_annual", 0.0)),
        solvers=tuple(str(x) for x in top.get("solvers", ["CLARABEL"])),
        source_text=text,
    )


# --- section parsers -------------------------------------------------------------------


def _data(d: Any) -> DataSpec:
    d = _keys(
        d,
        "data",
        required={"source", "frequency", "start", "end", "base_currency", "hedging", "candidate"},
        optional={"synthetic"},
    )
    try:
        periods_per_year(d["frequency"])
    except ValueError as e:
        raise SpecError(f"data.frequency: {e}") from None
    synthetic = None
    if d["source"] == "synthetic":
        s = _keys(d.get("synthetic", {}), "data.synthetic",
                  optional=set(SyntheticParams.__dataclass_fields__))  # fmt: skip
        if s.get("live_start") is not None:
            s["live_start"] = _period_str(s["live_start"])
        synthetic = SyntheticParams(**s)
    elif "synthetic" in d:
        raise SpecError("data.synthetic is only allowed with source: synthetic")
    return DataSpec(
        source=str(d["source"]),
        frequency=str(d["frequency"]),
        start=_period_str(d["start"]),
        end=_period_str(d["end"]),
        base_currency=str(d["base_currency"]),
        hedging=str(d["hedging"]),
        candidate=str(d["candidate"]),
        synthetic=synthetic,
    )


def _window(d: Any) -> WindowSpec:
    d = _keys(d, "window", required={"kind"}, optional={"periods", "min_periods"})
    if d["kind"] == "rolling":
        if not isinstance(d.get("periods"), int) or d["periods"] < 2:
            raise SpecError("window.periods must be an integer >= 2 for a rolling window")
    elif d["kind"] == "expanding":
        if "periods" in d:
            raise SpecError("window.periods is not used by an expanding window; use min_periods")
    else:
        raise SpecError(f"window.kind must be 'rolling' or 'expanding', got {d['kind']!r}")
    return WindowSpec(**d)


def _rebalance(d: Any, data_freq: str) -> RebalanceSpec:
    d = _keys(d, "rebalance", optional={"kind", "every", "band"})
    spec = RebalanceSpec(**d)
    if spec.kind not in ("calendar", "threshold"):
        raise SpecError(f"rebalance.kind must be 'calendar' or 'threshold', got {spec.kind!r}")
    if spec.kind == "threshold" and not (isinstance(spec.band, int | float) and spec.band > 0):
        raise SpecError("rebalance.band must be a positive weight for kind: threshold")
    if spec.kind == "calendar" and spec.band is not None:
        raise SpecError("rebalance.band is only used with kind: threshold")
    if spec.every not in ("M", "Q", "A"):
        raise SpecError(f"rebalance.every must be M, Q or A, got {spec.every!r}")
    if periods_per_year(spec.every) > periods_per_year(data_freq):
        raise SpecError(f"rebalance.every {spec.every} is finer than data.frequency {data_freq}")
    return spec


def _costs(d: Any) -> CostsSpec:
    d = _keys(d, "costs", optional={"default_bps", "per_asset"})
    spec = CostsSpec(
        default_bps=float(d.get("default_bps", 0.0)),
        per_asset={str(k): float(v) for k, v in (d.get("per_asset") or {}).items()} or None,
    )
    if spec.default_bps < 0 or any(v < 0 for v in (spec.per_asset or {}).values()):
        raise SpecError("costs must be >= 0 bps")
    return spec


def _liquidity(d: Any, data_freq: str) -> LiquiditySpec:
    d = _keys(d, "liquidity", optional={"dealing", "notice_periods", "gate"})
    spec = LiquiditySpec(**d)
    if spec.dealing not in ("M", "Q", "A"):
        raise SpecError(f"liquidity.dealing must be M, Q or A, got {spec.dealing!r}")
    if periods_per_year(spec.dealing) > periods_per_year(data_freq):
        raise SpecError(f"liquidity.dealing {spec.dealing} is finer than data.frequency")
    if not isinstance(spec.notice_periods, int) or spec.notice_periods < 0:
        raise SpecError("liquidity.notice_periods must be an integer >= 0")
    if spec.gate is not None and not 0 < spec.gate <= 1:
        raise SpecError("liquidity.gate must be in (0, 1]")
    return spec


def _cma(d: Any, data: DataSpec) -> CMA:
    """``cma: {version: placeholder}`` (synthetic truth, generated) or
    ``cma: {version: <label>, vectors: [{effective: YYYY-MM, returns_annual: {asset: r}}]}``."""
    d = _keys(d, "cma", required={"version"}, optional={"vectors"})
    version = str(d["version"])
    if version == PLACEHOLDER:
        if data.source != "synthetic":
            raise SpecError("cma.version placeholder needs data.source: synthetic")
        cma = placeholder_cma(data.candidate, data.synthetic.mu_annual, data.start)
        if "vectors" in d and _cma_vectors(d["vectors"]) != cma.vectors:
            raise SpecError("cma: placeholder vectors are generated from the synthetic truth; "
                            "remove 'vectors' or use another version name")  # fmt: skip
        return cma
    if "vectors" not in d:
        raise SpecError(f"cma.version {version!r}: only 'placeholder' resolves by name until the "
                        "PostgreSQL source exists (P2-M8); give vectors inline")  # fmt: skip
    try:
        return CMA(version, _cma_vectors(d["vectors"]))
    except SpecError:
        raise
    except (ValueError, TypeError) as e:
        raise SpecError(f"cma: {e}") from None


def _cma_vectors(raw: Any) -> tuple[CMAVector, ...]:
    out = []
    for i, v in enumerate(_list(raw, "cma.vectors")):
        v = _keys(v, f"cma.vectors[{i}]", required={"effective", "returns_annual"})
        r = v["returns_annual"]
        if not isinstance(r, dict) or not r:
            raise SpecError(f"cma.vectors[{i}].returns_annual: expected a non-empty mapping")
        out.append(CMAVector(_period_str(v["effective"]), {str(k): x for k, x in r.items()}))
    return tuple(out)


def _backtest(d: Any) -> BacktestSpec:
    d = _keys(d, "backtest", optional={"mode", "start"})
    if d.get("mode", "walk_forward") not in ("walk_forward", "in_sample"):
        raise SpecError(f"backtest.mode must be walk_forward or in_sample, got {d['mode']!r}")
    if d.get("start") is not None:
        d["start"] = _period_str(d["start"])
    return BacktestSpec(**d)


def _allocator(d: Any, path: str, funding: str) -> AllocatorEntry:
    if not isinstance(d, dict) or "type" not in d:
        raise SpecError(f"{path}: needs a 'type'")
    type_ = d["type"]
    try:
        allowed = allowed_params(type_)
    except ValueError as e:
        raise SpecError(f"{path}: {e}") from None
    params = {k: v for k, v in d.items() if k != "type"}
    unknown = set(params) - allowed
    if unknown:
        raise SpecError(f"{path} ({type_}): unknown parameters {sorted(unknown)}; "
                        f"allowed: {sorted(allowed)}")  # fmt: skip
    if "funding" in allowed and "funding" not in params:
        params["funding"] = funding
    grid_params = {}
    for k, v in params.items():
        values = v if isinstance(v, list) else [v]
        if not values:
            raise SpecError(f"{path}.{k}: empty list")
        grid_params[k] = values
    return AllocatorEntry(type=type_, params=grid_params)


def _estimators(raw: Any, allocators: tuple[AllocatorEntry, ...]) -> tuple[dict[str, str], ...]:
    if raw is None:
        return ({"method_mu": "hist", "method_cov": "hist"},)
    out = []
    for i, e in enumerate(_list(raw, "grid.estimators")):
        e = _keys(e, f"grid.estimators[{i}]", required={"method_mu", "method_cov"})
        out.append({"method_mu": str(e["method_mu"]), "method_cov": str(e["method_cov"])})
    if not out and any(uses_estimator(a.type) for a in allocators):
        raise SpecError("grid.estimators is empty but an allocator needs one")
    return tuple(out)


def _constraint_sets(raw: Any) -> tuple[ConstraintSet, ...]:
    sets = []
    for i, d in enumerate(_list(raw, "grid.constraint_sets")):
        try:
            sets.extend(_expand_sweeps(d))
        except (ValueError, TypeError) as e:
            raise SpecError(f"grid.constraint_sets[{i}]: {e}") from None
    names = [cs.name for cs in sets]
    if len(set(names)) != len(names):
        raise SpecError(f"grid.constraint_sets: duplicate names {names}")
    return tuple(sets)


def _expand_sweeps(d: Any) -> list[ConstraintSet]:
    """``{key: {sweep: [v1, v2]}}`` -> one set per value (cartesian over swept keys, spec order),
    named ``base[key=v1,...]``. Equivalent to listing the expanded sets by hand."""
    if not isinstance(d, dict):
        raise ValueError("expected a mapping")
    swept = {k: v["sweep"] for k, v in d.items() if isinstance(v, dict) and set(v) == {"sweep"}}
    if not swept:
        return [ConstraintSet.from_dict(d)]
    for k, values in swept.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{k}.sweep must be a non-empty list")
    if "name" not in d:
        raise ValueError("constraint_set needs a name")
    out = []
    for combo in itertools.product(*swept.values()):
        chosen = dict(zip(swept, combo, strict=True))
        label = ",".join(f"{k}={v:g}" if isinstance(v, int | float) else f"{k}={v}"
                         for k, v in chosen.items())  # fmt: skip
        concrete = {**d, **chosen, "name": f"{d['name']}[{label}]"}
        cs = ConstraintSet.from_dict(concrete)
        out.append(replace(cs, sweep={"base": d["name"], "keys": chosen}))
    return out


_SWEEP_NAME = re.compile(r"^(?P<base>.+)\[(?P<pairs>[^\[\]]+)\]$")


def sweep_metadata(cs: ConstraintSet) -> dict | None:
    """The set's sweep ({"base", "keys"}): from the parser, or recovered from a name of the form
    ``base[key=value,...]`` whose values match the set's fields. An explicitly listed set named
    that way hashes identically to the swept one, so it is treated the same."""
    if cs.sweep:
        return cs.sweep
    m = _SWEEP_NAME.match(cs.name)
    if not m:
        return None
    keys = {}
    for pair in m["pairs"].split(","):
        key, _, value = pair.partition("=")
        if not hasattr(cs, key) or key in ("name", "sweep"):
            return None
        actual = getattr(cs, key)
        try:
            if not math.isclose(float(value), float(actual)):
                return None
        except (TypeError, ValueError):
            return None
        keys[key] = actual
    return {"base": m["base"], "keys": keys}


# --- helpers ---------------------------------------------------------------------------


def _keys(d: Any, path: str, required: set[str] = frozenset(), optional: set[str] = frozenset()):
    if not isinstance(d, dict):
        raise SpecError(f"{path or 'spec'}: expected a mapping, got {type(d).__name__}")
    missing = set(required) - set(d)
    if missing:
        raise SpecError(f"{path or 'spec'}: missing keys {sorted(missing)}")
    unknown = set(d) - set(required) - set(optional)
    if unknown:
        raise SpecError(f"{path or 'spec'}: unknown keys {sorted(unknown)}")
    return dict(d)


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


def _list(v: Any, path: str) -> list:
    if not isinstance(v, list) or not v:
        raise SpecError(f"{path}: expected a non-empty list")
    return v


def _period_str(v: Any) -> str:
    """YAML turns 2006-01-31 into a date but leaves 2006-01 a string; normalise both."""
    if isinstance(v, dt.date):
        return v.isoformat()
    return str(v)


# Constraint-set keys added after Phase 1 enter the canonical form (and config_id) only when
# set, so existing specs keep their spec_hash and config_ids. ``sweep`` is metadata: never hashed.
_LATER_CS_KEYS = ("max_vol_annual", "max_cvar_period", "max_cdar", "min_return_annual",
                  "candidate_max_risk_share")  # fmt: skip


def constraint_set_dict(cs: ConstraintSet) -> dict:
    d = asdict(cs)
    d.pop("sweep")
    for k in _LATER_CS_KEYS:
        if d[k] is None:
            del d[k]
    if d["class_limits"] is not None:
        d["class_limits"] = {k: list(v) for k, v in d["class_limits"].items()}
    return d

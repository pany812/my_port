# Phase 1: the core loop

## Goal

Run one candidate through a grid of allocators and constraint sets on the SAA universe,
in-sample and in a basic walk-forward, and produce an allocation corridor stored in a registry.
Build end to end on synthetic data first; real data plugs in behind the same loader interface.

## Definition of done

1. `wb run specs/example_synthetic.yaml` runs end to end on synthetic data, writes the registry,
   prints the corridor table and saves `out/<experiment>/corridor.csv`.
2. `pytest -q` is green, including allocator smoke tests, a forced-infeasible case that is
   recorded rather than dropped, the golden determinism test and the no-look-ahead test.
3. Every grid cell appears in the registry with a status; the corridor report shows n_ok and
   n_failed by status.
4. README sections: how to write a spec, how to add an allocator.

## Milestones (one at a time; stop and show results after each)

**M0 Orientation.** Read this file and CLAUDE.md, inspect riskbench and the mandate-adherence
analyzer, then ask the open questions below. Propose the package skeleton before writing code.

**M1 Foundations.** `pyproject.toml` (Python ≥ 3.10, riskfolio-lib==7.4.0 pinned, pytest, ruff),
`units.py`, the `Allocator` protocol, `FitContext`, `AllocationResult`, and a synthetic data
generator: SAA building blocks plus a candidate with configurable volatility, correlation to
equities, skew, and an optional short live history.

**M2 Allocators and policy.**
- `StaticSAA`, `SAAPlus(x, funding="pro_rata")`, `EqualWeight`, `InverseVol`.
- `RiskfolioMeanRisk(method_mu, method_cov, rm, obj, rf, l)` on `model="Classic"`.
- `RiskfolioHC(model in {HRP, HERC, NCO}, codependence, linkage, rm, obj)`.
- Policy application: class limits and candidate cap via `rp.assets_constraints`; TE vs SAA;
  per-asset band via `allowTO`; HC bounds via `rp.hrp_constraints`.
- Capture stdout around every solve; map `None` to `status="infeasible"`; wrap exceptions.

**M3 Grid and registry.** YAML spec → cartesian expansion in a deterministic order with stable
`cell_id`s; sequential runner (parallelism later); SQLAlchemy registry (SQLite in tests).

**M4 Evaluation.** Corridor statistics for the candidate's capital weight and risk share under
MV, CVaR and CDaR (`rp.Risk_Contribution`); ex-post table for the SAA versus each cell:
annualised return and volatility, CVaR 95%, CDaR 95%, max drawdown, TE vs SAA.

**M5 Basic walk-forward.** Rolling or expanding window, calendar rebalancing, drift between
rebalances, no costs, using the same `Allocator` objects. Reuse riskbench's engine if its
interface fits; otherwise wrap it. Propose the choice before building.

**M6 CLI and report.** `wb run`, `wb report <experiment>`; corridor CSV plus a short markdown
summary with group-by views (by allocator family, risk measure, estimator).

## Interfaces

```python
@dataclass(frozen=True)
class FitContext:
    as_of: pd.Timestamp
    saa: pd.Series                 # policy weights, candidate at 0
    candidate: str
    policy: CompiledPolicy         # constraint tables and limits, per-period units
    mu_override: pd.Series | None = None

@dataclass
class AllocationResult:
    weights: pd.Series | None
    status: Literal["ok", "infeasible", "solver_error", "exception"]
    message: str = ""
    elapsed_s: float = 0.0
    diagnostics: dict = field(default_factory=dict)

class Allocator(Protocol):
    name: str
    def params(self) -> dict: ...
    def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult: ...
```

Riskfolio-Lib wrapper pattern (verified on 7.4.0):

```python
def fit(self, returns, ctx):
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu=self.method_mu, method_cov=self.method_cov)
    if ctx.mu_override is not None:
        port.mu = ctx.mu_override.to_frame().T[returns.columns]
    apply_policy(port, ctx.policy)        # A/B, TE, band, solvers
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        w = port.optimization(model="Classic", rm=self.rm, obj=self.obj,
                              rf=ctx.policy.rf, l=self.l, hist=True)
    if w is None:
        return AllocationResult(None, "infeasible", buf.getvalue().strip())
    return AllocationResult(w["weights"], "ok")
```

## Example spec

```yaml
experiment: synthetic_trend_v1
seed: 42
data:
  source: synthetic            # later: postgres
  frequency: M
  start: 2006-01
  end: 2026-09
  base_currency: SEK
  candidate: CAND
saa:
  version: example
funding: pro_rata
window: {kind: rolling, periods: 120}
rebalance: {kind: calendar, every: M}
grid:
  estimators:
    - {method_mu: hist, method_cov: ledoit}
    - {method_mu: JS, method_cov: gerber1}
  allocators:
    - {type: static_saa}
    - {type: saa_plus, x: [0.02, 0.05, 0.10]}
    - {type: riskfolio_mean_risk, rm: [MV, CVaR, CDaR], obj: [Sharpe, MinRisk]}
    - {type: riskfolio_hc, model: [HRP, HERC], codependence: [pearson, spearman], linkage: [ward]}
  constraint_sets:
    - {name: bands_5pct, band: 0.05, candidate_cap: 0.10}
    - {name: te_2pct, te_annual: 0.02, candidate_cap: 0.10}
risk_lenses: [MV, CVaR, CDaR]
```

## Registry schema (initial)

```
experiments(experiment_id, name, spec_yaml, spec_hash, data_vintage, saa_version,
            candidate_id, riskfolio_version, seed, created_at)
cells(cell_id, experiment_id, cell_index, config_id, allocator, params_json, estimator_json,
      constraint_set, data_variant, window_end, status, message, elapsed_s, diagnostics_json)
weights(cell_id, asset_id, weight)
metrics(cell_id, metric, lens, value)
```

`experiment_id` = hash(spec_hash, data_vintage, riskfolio_version); `cell_id` =
`<experiment_id>:<config_id>:<data_variant>:<window_end>`. `data_vintage` is a sha256 of the
exact returns, dates and backfill flags. `spec_hash` excludes the cosmetic `experiment` name.

## Corridor definition

Over cells with `status="ok"` (per `window_end` in walk-forward mode): candidate capital weight
median, P25, P75, P10, P90 and share of cells below 0.25%; the same for candidate risk share
per lens. Always report n_cells, n_ok and n_failed by status.

## Out of scope for Phase 1

Transaction costs, liquidity terms, funding sources other than pro rata, Black–Litterman,
entropy pooling, factor models, worst-case optimization, spanning and Sharpe-difference tests,
deflated Sharpe and PBO, the Streamlit UI, and live Bloomberg calls.

## Open questions for Patrik (ask at M0)

1. Where does riskbench live, and what is its backtest entry point?
2. SAA: building blocks, weights, ranges, and which version to start from.
3. Return frequency, base currency and hedging convention.
4. The first real candidate to test.
5. Registry: a new PostgreSQL schema, or a schema in the existing analytics database?

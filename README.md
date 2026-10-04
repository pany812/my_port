# Portfolio construction workbench

Tests whether a candidate strategy improves the strategic portfolio (SAA), and at what size. It
runs the candidate through a grid of estimators, construction methods, risk measures and
constraint sets, and judges it on out-of-sample evidence. The output is an **allocation
corridor**: the distribution of the candidate's capital weight and risk share across all grid
cells, with failed cells counted, never dropped.

Riskfolio-Lib 7.4.0 does estimation and construction, with skfolio 1.4.11 as a second backend:
the same configuration can be solved by both and compared. The backtest, statistics, registry
and reporting are our own code. See `CLAUDE.md` for conventions and verified library behaviour and
`docs/PHASE1.md` for scope.

## Quickstart

```bash
uv sync                                   # Python 3.12 venv with pinned dependencies
uv run pytest -q                          # must pass before every commit
uv run wb run specs/example_synthetic.yaml -v
uv run wb report synthetic_trend_v1       # rebuild the report from the registry
```

`wb run` writes the registry (default `sqlite:///out/registry.db`; override with `--registry` or
`$WB_REGISTRY`, e.g. a PostgreSQL URL), prints the corridor headline and writes
`out/<experiment>/`:

| file | content |
|---|---|
| `corridor.csv` | corridor per data variant × rebalance date × measure (counts by status, P10–P90, share below 0.25%) |
| `oos.csv` | walk-forward out-of-sample statistics per configuration vs the SAA path |
| `expost.csv` | in-sample statistics per cell and rebalance date (diagnostic only) |
| `summary.md` | provenance, cell counts, corridor (latest and through time), group-by views, failures, OOS vs SAA, **evidence net of search** (Sharpe test vs SAA with BH-adjusted p, deflated Sharpe, PBO, spanning), live-only variant, definitions |

Re-running an identical spec on identical data is skipped (`--if-exists replace` to redo):
`experiment_id` is a hash of the spec, the data vintage and the Riskfolio-Lib version.

The example spec runs about 6,600 fits (129 monthly rebalances × 48 configurations, plus the
live-only variant) in roughly 3 minutes.

## How to write a spec

A spec is YAML. Unknown keys are errors, and every annual figure is converted to the return
frequency in `workbench.units`, never elsewhere.

```yaml
experiment: my_candidate_v1          # cosmetic name; not part of spec_hash
seed: 42                             # stored in the registry; drives synthetic data
data:
  source: synthetic                  # synthetic | postgres (loader pending)
  frequency: M                       # one return frequency per experiment
  start: 2006-01
  end: 2026-09
  base_currency: SEK                 # required: never assumed
  hedging: "FI SEK-hedged, equity unhedged"   # required: never assumed
  candidate: CAND
  synthetic:                         # only for source: synthetic (annual, decimals)
    mu_annual: 0.05
    vol_annual: 0.10
    corr_to_equity: -0.1
    skew: 0.3
    live_start: 2016-01              # earlier observations are flagged as backfilled
    backfill: true                   # false: NaN before live_start (history is trimmed)
    tail_df: 6                       # optional fat tails for the building blocks
saa: {version: example}
funding: pro_rata                    # how saa_plus funds the candidate (Phase 1: pro_rata only)
rf_annual: 0.0
window: {kind: rolling, periods: 120}          # or {kind: expanding, min_periods: 36}
rebalance: {kind: calendar, every: M}          # M | Q | A
backtest: {mode: walk_forward, start: null}    # walk_forward (default) | in_sample
grid:
  estimators:                        # applied to Riskfolio-Lib allocators only
    - {method_mu: hist, method_cov: ledoit}
    - {method_mu: JS, method_cov: gerber1}
  allocators:                        # a list value is a grid dimension
    - {type: static_saa}
    - {type: saa_plus, x: [0.02, 0.05, 0.10]}
    - {type: equal_weight}
    - {type: inverse_vol}
    - {type: riskfolio_mean_risk, rm: [MV, CVaR, CDaR], obj: [Sharpe, MinRisk]}
    - {type: riskfolio_hc, model: [HRP, HERC], codependence: [pearson, spearman], linkage: [ward]}
    - {type: skfolio_mean_risk, rm: [CVaR], obj: [MinRisk]}   # same parameters, second library
    - {type: skfolio_hc, model: [HRP], max_clusters: 3}      # max_clusters: skfolio only
  constraint_sets:
    - {name: bands_5pct, band: 0.05, candidate_cap: 0.10}
    - {name: te_2pct, te_annual: 0.02, candidate_cap: 0.10}
    - {name: ranges, asset_ranges: true, class_limits: {equity: [0.45, 0.55]}}
risk_lenses: [MV, CVaR, CDaR]        # lenses for the candidate's risk share
solvers: [CLARABEL]
```

**Constraint-set keys.** `candidate_cap` is the maximum candidate weight. `asset_ranges` applies
the SAA's per-asset ranges. `class_limits` bounds summed class weights. `band` is a per-asset band
`|w − SAA| ≤ band`. `te_annual` is the annual tracking-error limit vs the SAA.

Mean-risk allocators enforce all of these keys. HC enforces asset bounds and the band only.
**Every allocator's output is post-checked against the full constraint set**: a breach records
the cell as `infeasible` with the violations in `diagnostics_json`.

**Grid expansion order** is deterministic: constraint set → allocator entry → parameter product
(spec order) → estimator. Each configuration gets a content-hashed `config_id` that is stable
under reordering. In walk-forward mode every rebalance date is a cell.

**Data variants.** If the candidate has backfilled observations, every experiment also runs a
`live_only` variant. The report always shows it and flags out-of-sample samples shorter than 36
periods.

**Two libraries.** `skfolio_mean_risk` and `skfolio_hc` take the same parameters as their
`riskfolio_*` twins (estimator names in Riskfolio vocabulary; unmapped names are recorded as
`exception` cells). When both libraries are in a grid, the report adds a "By library" view and a
**library agreement** section: matched configurations compared on the candidate's weight at
every date. HERC/NCO differ by design unless `max_clusters` is set (cluster-count selection).
See `specs/example_two_libraries.yaml`.

**Allocator parameters** are the fields of the allocator dataclass (see
`src/workbench/allocators/`). `wb run` rejects unknown names before anything runs.

## How to add an allocator

1. **Implement the protocol** in `src/workbench/allocators/`. Use a frozen dataclass whose fields
   are its parameters, a `name` class attribute and `params()`. Route `fit` through
   `guarded_fit`, which handles input checks, stdout capture, status mapping, weight cleaning and
   the policy post-check:

   ```python
   from dataclasses import asdict, dataclass

   import pandas as pd

   from workbench.allocators._solve import Infeasible, guarded_fit
   from workbench.allocators.base import AllocationResult, FitContext


   @dataclass(frozen=True)
   class MinVariance:
       """Long-only minimum variance on the sample covariance (per-period returns)."""

       shrink: float = 0.0
       name = "min_variance"

       def params(self) -> dict:
           return asdict(self)

       def fit(self, returns: pd.DataFrame, ctx: FitContext) -> AllocationResult:
           def impl(r: pd.DataFrame, c: FitContext, diag: dict):
               # deterministic, no I/O; sees only rows dated <= c.as_of
               ...  # return a pd.Series of weights indexed by asset, a Riskfolio-Lib
               # one-column "weights" DataFrame, None (solver found nothing -> infeasible),
               # or raise Infeasible("why") when the problem is infeasible before solving
           return guarded_fit(impl, returns, ctx)
   ```

   `ctx.policy` (a `CompiledPolicy`, already in per-period units) holds the bounds, class
   limits, band, TE limit and SAA. Enforce what your method can. The post-check catches the rest.
2. **Register it** in `ALLOCATOR_TYPES` in `src/workbench/allocators/factory.py`. If it takes
   `method_mu` and `method_cov` fields, the spec's `grid.estimators` are applied to it
   automatically.
3. **Classify it** for the group-by views: add the type to `FAMILIES` in
   `src/workbench/evaluation/corridor.py`.
4. **Test it** in `tests/test_allocators.py`, using synthetic fixtures only. Add a smoke test,
   determinism, and a forced-infeasible case that must be recorded with `status="infeasible"`
   (`CLAUDE.md`). Add the instance to `ALLOCATORS` there and the parametrised smoke, band and
   determinism tests pick it up. If your method ignores the band, add it to the band test's
   allow-list.
5. If it wraps a library, verify the library's behaviour on synthetic data and record any traps
   in `CLAUDE.md`.

## Layout

```
src/workbench/
  units.py        the only annual <-> per-period conversions
  data/           MarketData, synthetic generator, loaders, alignment, data vintage
  policy/         SAA, constraint sets -> CompiledPolicy, Riskfolio-Lib and skfolio translation
  allocators/     protocol, guarded_fit, naive, Riskfolio-Lib and skfolio allocators, factory
  grid/           spec parsing, deterministic expansion, runner
  backtest/       rebalance schedule, walk-forward engine
  evaluation/     stats, risk shares, corridor, ex-post, out-of-sample, inference + evidence,
                  library agreement, report
  registry/       SQLAlchemy models and store
  cli.py          wb run / wb report
specs/            experiment specs
tests/            pytest on synthetic fixtures; tests/golden pins determinism
```

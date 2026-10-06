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
uv run wb memo synthetic_trend_v1         # IC memo (SCQA) from the registry
```

`wb run` writes the registry (default `sqlite:///out/registry.db`; override with `--registry` or
`$WB_REGISTRY`, e.g. a PostgreSQL URL), prints the corridor headline and writes
`out/<experiment>/`:

| file | content |
|---|---|
| `corridor.csv` | corridor per data variant × rebalance date × measure (counts by status, P10–P90, share below 0.25%) |
| `oos.csv` | walk-forward out-of-sample statistics per configuration vs the SAA path |
| `expost.csv` | in-sample statistics per cell and rebalance date (diagnostic only) |
| `summary.md` | provenance (and CMA), cell counts, corridor (latest and through time), group-by views, failures, library agreement, risk budgets, **Black–Litterman breakeven**, constraint sweeps, OOS vs SAA, **evidence net of search** (Sharpe test vs SAA with BH-adjusted p, deflated Sharpe, PBO, spanning), **stress** (crisis windows, bootstrap paths), live-only variant, definitions |

If a registry was created by an older version, `wb` refuses to write to it and asks for
`wb migrate`, which adds the new columns and backfills them exactly (older experiments had no
frictions, so net = gross).

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
funding: pro_rata                    # saa_plus default: pro_rata | asset:<id> | class:<name>
rf_annual: 0.0
cma: {version: placeholder}          # optional: expected returns for method_mu: cma (see below)
window: {kind: rolling, periods: 120}          # or {kind: expanding, min_periods: 36}
rebalance: {kind: calendar, every: M}          # M | Q | A; or {kind: threshold, every: M, band: 0.01}
costs: {default_bps: 5, per_asset: {EM_EQ: 20, CAND: 0}}  # optional: one-way bps of traded weight
liquidity: {dealing: Q, notice_periods: 1, gate: 0.25}   # optional: the candidate's dealing terms
backtest: {mode: walk_forward, start: null}    # walk_forward (default) | in_sample
grid:
  estimators:                        # applied to Riskfolio-Lib allocators only
    - {method_mu: hist, method_cov: ledoit}
    - {method_mu: JS, method_cov: gerber1}
    - {method_mu: cma, method_cov: ledoit}     # mean from the spec's cma, point in time
  allocators:                        # a list value is a grid dimension
    - {type: static_saa}
    - {type: saa_plus, x: [0.02, 0.05, 0.10]}
    - {type: equal_weight}
    - {type: inverse_vol}
    - {type: riskfolio_mean_risk, rm: [MV, CVaR, CDaR], obj: [Sharpe, MinRisk]}
    - {type: riskfolio_hc, model: [HRP, HERC], codependence: [pearson, spearman], linkage: [ward]}
    - {type: skfolio_mean_risk, rm: [CVaR], obj: [MinRisk]}   # same parameters, second library
    - {type: skfolio_hc, model: [HRP], max_clusters: 3}      # max_clusters: skfolio only
    - {type: riskfolio_bl, target_weight: [0.05, 0.10]}      # breakeven: what must it earn?
    - {type: riskfolio_bl, view_annual: [0.02], confidence: [0.5]}  # weight at a stated view
  constraint_sets:
    - {name: bands_5pct, band: 0.05, candidate_cap: 0.10}
    - {name: te_2pct, te_annual: 0.02, candidate_cap: 0.10}
    - {name: ranges, asset_ranges: true, class_limits: {equity: [0.45, 0.55]}}
    - {name: te, te_annual: {sweep: [0.005, 0.01, 0.02]}}      # one set per value
    - {name: caps, max_vol_annual: 0.09, max_cvar_period: 0.03, max_cdar: 0.25,
       min_return_annual: 0.02, candidate_max_risk_share: 0.10}
risk_lenses: [MV, CVaR, CDaR]        # lenses for the candidate's risk share
solvers: [CLARABEL]
stress:                              # optional: crisis windows and bootstrap paths (see below)
  windows: default                   # or {gfc: [2007-11, 2009-02], ...}
  weights: [0.05]
  bootstrap: {n_paths: 2000, horizon_years: 10, block: null}
decision:                            # optional, not hashed: the IC decision for `wb memo`
  candidate_name: "Trend programme X"
  recommendation: "Allocate 3% inside a 2-5% corridor, funded pro rata."
  proposal: {weight: 0.03, funding: pro_rata}
  target_corridor: [0.02, 0.05]
  conditions: ["Rebalance to 3% when outside the corridor"]
  kill_criteria:
    - {metric: te_vs_saa, above: 0.02, months: 12}       # rolling realised TE vs the SAA
    - {metric: active_return, below: -0.03, months: 24}  # rolling return minus the SAA's
    - {text: "Key-person event at the manager"}
  owner: "CIO office"
  review: 2027-06
```

**Frictions (optional).** `costs` are charged on every trade (the SAA path pays them too) and all
out-of-sample results and evidence use net returns. `liquidity` lets the candidate trade only on
dealing dates (the other assets are rescaled around its frozen weight), delays redemptions by
`notice_periods` dealing dates and caps each redemption at `gate` of the position. A
`threshold` rebalance fits at every date but trades only beyond `band` drift. `funding` in a
`saa_plus` entry can be a list, which makes the funding source a grid dimension. Specs without
these sections keep their `spec_hash`. See `specs/example_frictions.yaml`.

**Constraint-set keys.** `candidate_cap` is the maximum candidate weight. `asset_ranges` applies
the SAA's per-asset ranges. `class_limits` bounds summed class weights. `band` is a per-asset band
`|w − SAA| ≤ band`. `te_annual` is the annual tracking-error limit vs the SAA. Risk caps
carry their units in the name: `max_vol_annual`, `max_cvar_period` (CVaR 95% loss per return
period), `max_cdar` (CDaR 95% of uncompounded returns), `min_return_annual`, and
`candidate_max_risk_share` (the candidate's share of portfolio variance). Caps are post-checked
on the window's sample moments for every allocator. `{sweep: [...]}` on any key expands the set
into one set per value (cartesian over several keys), named `base[key=value]`; the report shows
the corridor and realised out-of-sample TE per value.

Mean-risk allocators enforce all of these keys. HC enforces asset bounds and the band only.
**Every allocator's output is post-checked against the full constraint set**: a breach records
the cell as `infeasible` with the violations in `diagnostics_json`.

**Grid expansion order** is deterministic: constraint set → allocator entry → parameter product
(spec order) → estimator. Each configuration gets a content-hashed `config_id` that is stable
under reordering. In walk-forward mode every rebalance date is a cell.

**Data variants.** If the candidate has backfilled observations, every experiment also runs a
`live_only` variant. The report always shows it and flags out-of-sample samples shorter than 36
periods.

**Risk budgets.** `riskfolio_risk_budget` / `skfolio_risk_budget` (`candidate_share`, `rm`,
`rest: saa | equal`) give the candidate a target share of risk; the report shows the capital
each target implies per lens and the realised share. Downside lenses `MSV`, `FLPM`, `SLPM` work
in `rm` and `risk_lenses`; denoised covariances `fixed`, `spectral`, `shrink` in `method_cov`.
Add `SCS` to `solvers` for risk budgets. See `specs/example_risk_budgets.yaml`.

**Return assumptions (CMA).** `method_mu: cma` in `grid.estimators` takes expected returns from
the spec's `cma` section instead of the window's history, so CMA vs historical means is a grid
dimension. `cma: {version: placeholder}` is the synthetic truth (an oracle, for testing); a real
version is written inline as dated vectors of expected annual total returns for every asset,
candidate included:

```yaml
cma:
  version: house_2026q3
  vectors:
    - {effective: 2016-01, returns_annual: {SE_EQ: 0.065, GL_EQ: 0.06, CAND: 0.05}}  # all assets
    - {effective: 2021-01, returns_annual: {SE_EQ: 0.07, GL_EQ: 0.065, CAND: 0.045}}
```

Each fit uses the latest vector effective on or before its date; a fit before the first vector
is an error (set `backtest.start`). The resolved values are part of `spec_hash`. Values are
expected annual (arithmetic) returns, converted with `(1 + r)^(1/n) − 1`.

**Black–Litterman breakeven.** `riskfolio_bl` / `skfolio_bl` use the SAA as the prior: the
equilibrium excess returns that make the SAA optimal, scaled by `prior_sharpe` (the SAA's
assumed annual Sharpe ratio, default 0.3). One view on the candidate's expected excess return
over rf. Two modes, exactly one per entry:
`view_annual` + `confidence` gives the weight a stated view earns; `target_weight` root-finds,
at every date, the expected excess return the candidate needs for that weight (an unreachable
target, e.g. above a band, is `infeasible` and counted). `obj: Sharpe | Utility`, `method_cov` as
a parameter; mean-variance only. The report adds "What would it have to earn?" with the required
excess return, the equilibrium, the premium, the required Sharpe and the unconstrained closed
form. See `specs/example_black_litterman.yaml`.

**Stress (optional).** `stress:` stresses the SAA and the SAA plus the candidate at the
corridor's P25 / median / P75 (latest date, full variant) and at any listed `weights`, funded per
`funding`, as fixed weights rebalanced every period. Crisis `windows` (`default`: GFC
2007-11..2009-02, COVID 2020-02..2020-03, rates 2022-01..2022-09) are evaluated on these policy
portfolios and on the stored walk-forward paths (when the window lies in the out-of-sample
period). The bootstrap draws `n_paths` stationary block-bootstrap paths of `horizon_years` from
each variant's history (mean block ceil(T^(1/3)) unless `block` is set), the same paths for every
weight, and reports max drawdown, CDaR, CVaR, return and volatility, paired with the SAA. Results
are disclosures stored in `evidence`. On synthetic data the windows carry no real crisis. See
`specs/example_stress.yaml`.

**IC memo.** `wb memo <experiment>` writes `out/<experiment>/memo.md`, a short SCQA document
built from the registry alone: the recommendation (from `decision`, written by people; the
workbench never recommends), a checks table flagging where the proposal and the evidence disagree
(corridor, significance net of search, DSR, PBO, spanning, stress, BL breakeven vs CMA, library
agreement, failures, backfill), the candidate's standalone profile, the corridor, evidence, risk
and stress at the proposal, conditions, and kill criteria replayed on the walk-forward path of the
`saa_plus` configuration at the proposed weight ("fired in 2 of 226 windows"). Flags are
information, not gates. `decision` is not part of `spec_hash`: edit it after a run and pass the
revised file with `wb memo <experiment> --spec specs/x.yaml` (refused if anything else changed).
Without a `decision` block the memo is a DRAFT around the corridor median. Add `saa_plus` at the
proposed weight to the grid (and to `stress.weights`) so the memo has evidence at that weight.

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

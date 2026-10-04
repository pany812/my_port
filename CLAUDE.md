# Portfolio construction workbench

Tests whether a candidate strategy improves the strategic portfolio (SAA), and at what size,
by running it through a grid of estimators, construction methods, risk measures and constraint
sets, and judging it on out-of-sample evidence. Riskfolio-Lib does estimation and construction.
Backtesting, statistics, the registry and reporting are our own code.

- Concept and context map: https://claude.ai/artifact/5QJ6JZvKZ85LLdvMuBxCg7
- Current build scope and milestones: @docs/PHASE1.md

## The one output that matters

An **allocation corridor** for the candidate: the distribution of its capital weight and risk
share across all grid cells (median, P25–P75, P10–P90, share of cells at zero). Failed cells are
counted and reported, never silently dropped. A single "optimal" weight is never the answer.

## Experiment grammar

Universe × Estimator × Method × Risk measure × Constraints × Rebalance policy → Evidence.
Every run is fully described by an `ExperimentSpec` (YAML) plus a data vintage.
Same spec + same data vintage ⇒ same `spec_hash` ⇒ identical weights.

## Layout (target)

```
src/workbench/
  data/         loaders (PostgreSQL fed by Bloomberg), synthetic generator, history alignment
  policy/       SAA object, policy compiler -> Riskfolio-Lib constraint tables
  allocators/   Allocator protocol + implementations (static, naive, riskfolio_*)
  grid/         spec parsing, deterministic grid expansion, runner with failure capture
  backtest/     walk-forward engine (reuse riskbench where it fits)
  evaluation/   corridor statistics, risk contributions, ex-post risk table
  registry/     SQLAlchemy models; SQLite in tests, PostgreSQL in production
  units.py      the ONLY place annual <-> per-period conversions happen
  cli.py        `wb run specs/<name>.yaml`, `wb report <experiment>`
specs/          experiment specs (YAML)
tests/          pytest, synthetic fixtures only
```

## Core abstractions (ask before changing these signatures)

- `Allocator.fit(returns: pd.DataFrame, ctx: FitContext) -> AllocationResult`.
  Deterministic given inputs and seed. No I/O inside `fit`.
- `AllocationResult`: `weights: pd.Series | None`, `status` in
  {`ok`, `infeasible`, `solver_error`, `exception`}, `message`, `elapsed_s`, `diagnostics`.
- `FitContext`: `as_of`, `saa` (pd.Series, candidate at 0), `candidate` id,
  `policy` (compiled constraints, already in per-period units), optional `mu_override`.
- Weights are a `pd.Series` indexed by asset id, summing to 1, long-only unless the spec says
  otherwise. Convert Riskfolio-Lib output (one-column DataFrame named `weights`) at the
  allocator boundary.

## Conventions

- One return frequency per experiment (default monthly simple total returns). Currency and
  hedging convention are set in the spec, never assumed.
- Riskfolio-Lib reads `rf`, `TE`, `lowerret` and CVaR-type limits in the period of the returns.
  Convert annual inputs only in `workbench.units` (e.g. TE: annual / sqrt(12) for monthly).
- No look-ahead: an allocator fitted at t sees only returns dated ≤ t; its weights apply from
  the next period.
- Backfilled or proxied candidate observations carry a flag column; every report also shows
  a live-only variant.
- Randomness: explicit seed in the spec, stored in the registry.
- Use `logging`, never `print`. Riskfolio-Lib prints on infeasibility: wrap every solve in
  `contextlib.redirect_stdout` and store the captured text in `message`.
- Docstrings state units and frequency for every numeric input and output.

## Riskfolio-Lib 7.4.0: verified behaviour and traps

Pinned: `riskfolio-lib==7.4.0`. Each item below was checked against the installed source
or by running it on synthetic data (October 2026).

- `port.optimization(...)` returns **None** when infeasible (prints a message, raises nothing).
  Map to `status="infeasible"`.
- `port.solvers` is a list tried in order until one returns a solution. Default CLARABEL.
  MOSEK is recommended for EVaR, RLVaR, GMD, Tail Gini and even moments. `card` and integer
  constraints need a MIP-capable solver.
- `allowTO` / `turnover` is an element-wise band |wᵢ − benchweightsᵢ| ≤ turnover, not total
  portfolio turnover. TE and the band both use `benchweights`.
- TE vs the SAA: `kindbench=True`, `benchweights` = one-column DataFrame indexed by asset,
  `allowTE=True`, `TE` in return frequency.
- Custom expected returns: call `assets_stats(...)` first, then set
  `port.mu = mu_series.to_frame().T[returns.columns]` (1×N, per period).
- `model="FM"` uses the historical covariance unless `hist=False`.
- The docstring says `model="BLFM"`; the code branch is `"BL_FM"` (`"BLFM"` fails).
  Pass `P_f` / `Q_f` as numpy arrays.
- `rp.entropy_pooling_views` fails when the view table contains only inequality views;
  include at least one equality view.
- `rp_optimization(b=...)`: `b` is an (n, 1) array. Check realised shares with
  `rp.Risk_Contribution` because linear constraints can stop a budget from being met.
- `rp.hrp_constraints(...)` returns `(w_max, w_min)`; set `hc.w_max`, `hc.w_min`.
- `rp.loadings_matrix(X=factors, Y=assets)` with stepwise selection adds a `const` column;
  drop it before setting `port.B` for factor risk-contribution constraints.
- Riskfolio-Lib has no backtester. Its single backtesting tutorial uses vectorbt.

## Testing

- `pytest -q` must pass before every commit. Tests use `tests/fixtures/synthetic.py` only:
  never Bloomberg, never the production database.
- Every allocator has a smoke test on synthetic data and a forced-infeasible case that must be
  recorded with `status="infeasible"`.
- Golden test: fixed spec + seed ⇒ identical `spec_hash` and weights.
- No-look-ahead test for the walk-forward engine.

## Integration points

- `riskbench` (existing volatility-targeting backtest package).
  TODO(Patrik): location and import name. Inspect it before building `backtest/`; prefer
  extending or wrapping it over duplicating it.
- Mandate-adherence analyzer. TODO(Patrik): location. Source of policy rules for `policy/`.
- Data: Bloomberg (blpapi) → PostgreSQL. TODO(Patrik): schema and tables for building-block and
  candidate returns. Loaders read from PostgreSQL only; no live Bloomberg calls in this package.
- UI: Streamlit over the registry. Phase 2, not now.

## Working style

- For anything touching more than one module, propose a plan first and wait for approval.
- One milestone at a time (see docs/PHASE1.md). Stop and show results after each.
- Ask before adding dependencies, changing core signatures or changing the registry schema.

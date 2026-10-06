# Portfolio construction workbench

Tests whether a candidate strategy improves the strategic portfolio (SAA), and at what size,
by running it through a grid of estimators, construction methods, risk measures and constraint
sets, and judging it on out-of-sample evidence. Riskfolio-Lib does estimation and construction,
with skfolio as a second backend (same configurations, compared in the library-agreement
section). Backtesting, statistics, the registry and reporting are our own code.

- Concept and context map: https://claude.ai/artifact/5QJ6JZvKZ85LLdvMuBxCg7
- Phase 1 (complete): @docs/PHASE1.md
- Phase 2 (current: evidence; PBO/DSR are disclosures, not gates): @docs/PHASE2.md

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
  ui/           read-only Streamlit UI over the registry (`wb ui`, optional extra `ui`)
  units.py      the ONLY place annual <-> per-period conversions happen
  cli.py        `wb run`, `wb report`, `wb memo`, `wb migrate`, `wb ui`
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
- Frictions (P2-M2) live in the walk-forward engine only; allocators do not see costs (no
  `current_weights` in `FitContext`). Without `costs`/`liquidity`/threshold the path must stay
  byte-identical to Phase 1 (golden test). Cells store the allocator's fitted weights; executed
  weights after liquidity/threshold rules are on the path (`RebalanceFit.executed`).
- Registry changes are additive only: a new column needs an entry in `registry.store.BACKFILL`
  with a value that is exact for older rows, so `wb migrate` can upgrade existing databases.
- Risk caps and the candidate risk-share cap (P2-M3) are post-checked on the fitting window's
  **sample** moments (vol, mean, variance share) and Riskfolio's CVaR/CDaR definitions: one
  definition for every allocator. Optimisers using shrunk estimators enforce caps under their
  own estimates and may breach the sample version; such cells are recorded `infeasible`.
- Constraint-set keys and sections added after Phase 1 enter the canonical form only when set
  (existing `spec_hash`/`config_id` unchanged); `sweep` metadata is never hashed, so a sweep and
  the same sets listed by hand are identical.
- CMAs (P2-M4, `data/cma.py`): `method_mu: cma` in `grid.estimators` takes the mean from the
  spec's `cma` (versioned, dated vectors of expected annual arithmetic total returns, candidate
  included; `(1 + r)^(1/n) - 1`). Point in time: a fit uses the latest vector effective on or
  before its date; a fit before the first vector is a `SpecError`, never a silent look-ahead.
  The resolved values enter the canonical form, so `spec_hash` changes when a CMA does.
  `placeholder` is the synthetic truth (an oracle). SAA weights are still hashed by version
  name only (fix with real SAA versions, P2-M8).
- Black–Litterman (P2-M4, `allocators/_bl.py`): the SAA is the prior, `pi = delta Sigma w_SAA`
  with `delta = prior_sharpe / sigma_SAA` (positive, frequency-free); one view on the candidate;
  our posterior formula, verified against both libraries. BL changes mu only (covariance stays
  the estimator's, so tau cancels), MV only, `obj` Sharpe or Utility with `l = delta / 2` (the
  only value for which no view returns the SAA). `target_weight` root-finds the posterior
  premium per date (bracket from the closed form, bisect to 1e-4 weight); a target above the
  policy's maximum feasible candidate weight, or needing a Sharpe > 3, is `infeasible`
  ("unreachable"). BL allocators take `method_cov` as a parameter, not from `grid.estimators`.
- Stress (P2-M5, `evaluation/stress.py`) is disclosure, stored as `evidence` rows
  (`stress_window:<name>`, `stress_bootstrap`; subject `x=<weight>`). Policy portfolios: the SAA
  and SAA + candidate at the full variant's latest corridor P25/median/P75 plus spec weights,
  funded per `funding` (`allocators.naive.funded_weights`, shared with `saa_plus`), rebalanced
  every period, no costs. Crisis windows are inclusive ("YYYY-MM" ends at month end; wealth
  starts at 1 at the window start) and also read off the stored walk-forward paths at report
  time. Bootstrap: stationary (Politis–Romano) over whole rows, mean block ceil(T^(1/3)), seeded
  per variant, the same paths for every weight (paired). Meaningful crisis results need real
  data; synthetic reports say so.
- IC memo (P2-M6, `evaluation/memo.py`, `wb memo`): built from the registry alone. The spec's
  `decision` block is governance, **excluded from `spec_hash`**, read from the stored spec or a
  revised `--spec` file whose hash must match. The workbench never writes a recommendation; the
  memo's checks flag disagreements (information, not gates; thresholds are module constants,
  significance 5% pending house conventions). Kill criteria (`te_vs_saa`, `active_return` over
  n months) are replayed on the stored path of `saa_plus` at the proposed weight and funding.
  Candidate and SAA profiles are `evidence` rows (subjects `profile:candidate`, `profile:saa`).
- UI (P2-M7, `ui/`): read-only by construction (`Registry(read_only=True)`: SQLite `mode=ro`,
  PostgreSQL `default_transaction_read_only`); a test checks the database file is byte-identical
  after a session. `ui/data.py` is plain functions (no Streamlit) over the same evaluation code as
  the report and memo; `ui/app.py` only lays out. `wb ui` binds to localhost with usage statistics
  off (Streamlit's defaults are all interfaces and telemetry on) and no Deploy button;
  `.streamlit/config.toml` repeats this. `streamlit==1.65.0` is the optional extra `ui` and in
  the dev group; UI tests skip without it.
- Max-Sharpe with no feasible positive expected excess return over rf is undefined: both
  mean-risk allocators record `infeasible` with the reason (`policy.skfolio.sharpe_undefined`).
- Evidence (P2-M1, `evaluation/inference.py`, `evaluation/evidence.py`) is disclosure, never a
  gate, and never changes a cell's status. Sharpe ratios are on returns in excess of the SAA's
  cash asset (or the policy rf): raw-return Sharpe inflates cash-heavy portfolios. Spanning
  with a cash asset runs in excess returns and tests alpha only (a near-constant benchmark makes
  the HK/KZ-F2 legs degenerate). The robust spanning line is HC3 with F scaling (Newey-West
  chi2 over-rejects at T~130). Path tests need >= 24 OOS periods. Every statistic has a
  known-answer test in `tests/test_inference.py`; extend it before changing a formula.

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
- Riskfolio's TE is **not demeaned**: `‖R(w − b)‖_F / sqrt(T − 1)` over the fitting window
  (root-mean-square active return), not the std of active returns. Our policy post-check
  uses the same definition.
- `HCPortfolio` raises `NameError` (not None) when `w_max` sums below 1 or `w_min > w_max`;
  pre-check bounds and record `status="infeasible"`.
- Custom expected returns: call `assets_stats(...)` first, then set
  `port.mu = mu_series.to_frame().T[returns.columns]` (1×N, per period).
- `method_mu="JS"` returns a **complex128** mu (eigenvalues via `np.linalg.eig`, zero imaginary
  part) and cvxpy then raises "Inequality constraints cannot be complex". Affects Classic and
  NCO. Cast to real at the allocator boundary (`allocators/_estimates.py`); for HC compute mu
  ourselves and pass `method_mu="custom_mu"`.
- `model="FM"` uses the historical covariance unless `hist=False`.
- The docstring says `model="BLFM"`; the code branch is `"BL_FM"` (`"BLFM"` fails).
  Pass `P_f` / `Q_f` as numpy arrays.
- `rp.entropy_pooling_views` fails when the view table contains only inequality views;
  include at least one equality view.
- `rp_optimization(b=...)`: `b` is an (n, 1) array. Check realised shares with
  `rp.Risk_Contribution` because linear constraints can stop a budget from being met.
- `rp.hrp_constraints(...)` returns `(w_max, w_min)`; set `hc.w_max`, `hc.w_min`.
- `rp.hrp_constraints` tests `data.loc[i, "Disabled"] is False`: with a numpy-bool `Disabled`
  column (pandas' default) every row is **silently ignored**. Build the table with
  `Disabled` as `object` dtype (Python bools).
- `rp.loadings_matrix(X=factors, Y=assets)` with stepwise selection adds a `const` column;
  drop it before setting `port.B` for factor risk-contribution constraints.
- Riskfolio-Lib has no backtester. Its single backtesting tutorial uses vectorbt.
- `arcinequality`/`brcinequality` (risk-contribution limits) are only meaningful with
  `rm="MV"`: Riskfolio compares variance contributions against the objective's risk measure,
  so under CVaR a 5% candidate cap pushed its variance share from 0.1% to 70%. Pass them for
  MV only; everything else relies on the post-check.
- **Bug: `upperdev` together with `arcinequality` (MV) raises `UnboundLocalError: 'g'`**: the
  risk-contribution constraint switches MV to the SDP form, which never defines `g`, and the
  `upperdev` branch only checks `network_sdp`/`cluster_sdp`. We drop `upperdev` in that case
  and the post-check holds the vol cap (`diagnostics["vol_cap"]`).
- `upperdev`, `upperCVaR`, `upperCDaR`, `lowerret` bind exactly as documented (per period;
  CDaR on uncompounded cumulative returns); skfolio's `max_standard_deviation`, `max_cvar`,
  `max_cdar`, `min_return` use the same definitions.
- `rp_optimization` returns None for numerical failures too (an FLPM budget fails in CLARABEL
  and ECOS, solves in SCS). Positive budgets are always feasible unless the linear constraints
  are, so classify with `policy.skfolio.linear_infeasibility`; add SCS to `solvers`.
- Risk budgets are exact (realised share = target) for smooth measures (MV, MSV). For CVaR and
  CDaR on historical scenarios contributions are not unique: the realised share differs from
  the target at the optimum (Riskfolio and skfolio give the same weights), and on short windows
  (36 months, ~2 tail scenarios) different targets can give nearly the same weight.
- `black_litterman` / `blacklitterman_stats` fix tau = 1/T and Omega = diag(P tau Sigma P'), i.e.
  confidence 0.5 (not configurable). The default delta is `(w' mu_hist - rf) / (w' Sigma w)` on the
  window: negative in about a third of 36-month synthetic windows, which flips the prior. With
  `model="BL"`, `hist=True` uses the sample covariance, not `cov_bl`. We use our own posterior.
- Classic `obj="Utility"` with MV maximises `mu'w - l w' Sigma w` (variance, no 1/2).
- Max-Sharpe returns None when no feasible portfolio has a positive expected excess return over
  rf, and sometimes when the best one is barely positive (<= 0.2% p.a.; skfolio solves those:
  4 of 172 pairs in the BL example, visible in the agreement section).
- Reusing one `Portfolio` and changing only `port.mu` between `optimization` calls gives the
  same weights as a fresh object (the BL root-finder relies on it).
- `assets_stats` has no `detone` argument (use `dict_cov`). Denoising `fixed`/`spectral`/`shrink`
  works in Classic and HC; **detoned covariances are not PSD** (min eigenvalue ~ -7e-4): keep
  detoning out of optimisers.

## skfolio 1.4.11: verified behaviour and traps

Pinned: `skfolio==1.4.11` (co-installs with riskfolio-lib 7.4.0; scipy 1.18.1). Adapters in
`allocators/skfolio_*.py`, names in `allocators/_skfolio_map.py`, policy in `policy/skfolio.py`.
Checked on synthetic data (October 2026):

- Estimators are numerically identical to Riskfolio's: mu `hist` -> `EmpiricalMu`,
  `JS` -> `ShrunkMu(method=JAMES_STEIN)` (real-valued, no complex trap); cov `hist`,
  `ledoit` -> `LedoitWolf`, `gerber1` -> `GerberCovariance`. Unmapped names fail loudly.
- CVaR/CDaR MinRisk and Sharpe match Riskfolio to ~1e-8. MV: use `STANDARD_DEVIATION` for
  `MAXIMIZE_RATIO` (Riskfolio maximises mean/std), `VARIANCE` otherwise; cash-dominated
  min-variance shows a ~5e-5 relative volatility gap (solver tolerance, flat region).
- Utility: Riskfolio `l` == skfolio `risk_aversion` (checked on interior solutions).
- HRP is identical. HERC/NCO are identical (NCO ~5e-4) only when the cluster count matches:
  Riskfolio picks k by the two-difference gap statistic, skfolio by its own rule
  (`HierarchicalClustering.max_clusters=None`). `SkfolioHC.max_clusters` forces it.
- `max_turnover` + `previous_weights` is an element-wise band (same as `allowTO`).
  `max_tracking_error` with `y = R @ SAA` is the same non-demeaned RMS/(T-1) TE as Riskfolio.
  Class limits: `groups` + string `linear_constraints` (class names must be identifiers).
- Failure: raises `cvxpy.error.SolverError` for both infeasible and numerically failed problems
  and keeps no `problem_`. We classify with `policy.skfolio.linear_infeasibility` (SAA is a
  feasible point when it meets the linear constraints; else an LP decides). No stdout output.
- `Portfolio.contribution` is not Euler-additive (variance sums to 2x, CDaR ~1.5% off): risk
  shares stay on Riskfolio-Lib/our code.
- NCO takes no composite weight bounds here; the post-check catches breaches.
- Downside measures: `MSV` -> `SEMI_DEVIATION`; `FLPM` -> `FIRST_LOWER_PARTIAL_MOMENT` and
  `SLPM` -> `SEMI_DEVIATION`, both with `min_acceptable_return=rf` (a **scalar**: an array raises
  "unhashable type"). Denoised `fixed` -> `DenoiseCovariance` (identical); `spectral`/`shrink`
  have no skfolio equivalent. `RiskBudgeting` gives Riskfolio's risk-budget weights to ~1e-5.
- `BlackLitterman` (views as strings: format numbers as `float(x)!r`, since a numpy 2 scalar's
  repr is `np.float64(...)`) equals our posterior for every tau and confidence; with one view
  `view_confidences=[k]` moves the asset exactly k of the way to the view (Idzorek). Its
  posterior covariance is not used. `EquilibriumMu(risk_aversion, weights)` gives `delta Sigma w`.
- MeanRisk max-Sharpe raises `SolverError` when no feasible portfolio has a positive excess
  return (classified with `sharpe_undefined`). Max-Sharpe is flat along the cash direction:
  weights carry ~1e-4 noise (cash up to 0.2pp), so the BL root-finder can step over a target by
  up to ~3e-4 (`diagnostics["jump"]`).
- **Bounded HRP/HERC returns NaN weights** when lower bounds bind: `_hrp.py:486` divides by
  `weights[cluster[0]]`, which is 0 once an asset is pushed to zero (0/0). Upper bounds alone
  are fine. Riskfolio solves the same bounds. `SkfolioHC` maps non-finite weights to
  `solver_error`. Unconstrained HRP is identical to Riskfolio for every estimator; with binding
  bounds the two libraries' bound-handling in the bisection differs (~0.7% on the example).
- skfolio HERC can overshoot `max_weights` by ~1e-6 (seen once in 2,064 cells: CASH 0.100001
  vs 0.10). Our post-check tolerance stays at 1e-6, so such a cell is recorded `infeasible`.

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
- UI: Streamlit over the registry (P2-M7, `wb ui`). Local only for now; a shared server against
  PostgreSQL needs authentication in front (SSO via a reverse proxy) and a read-only DB role.

## Working style

- For anything touching more than one module, propose a plan first and wait for approval.
- One milestone at a time (see docs/PHASE1.md). Stop and show results after each.
- Ask before adding dependencies, changing core signatures or changing the registry schema.

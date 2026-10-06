# Phase 2: evidence

Approved 2026-10-04. PBO and the deflated Sharpe ratio are **disclosures**, not gates. Source: the concept map's Phase 2 ("Costs and funding rules,
stress scenarios, spanning and overfitting statistics, factor models, Black–Litterman, risk
budgeting, NCO and HERC, the UI and the IC memo"), the Phase 1 out-of-scope list, and what
Phase 1 (M0–M7) actually delivered.

## Goal

Phase 1 answers "what range of weights do reasonable methods give?". Phase 2 answers "is that
range **evidence**?": out-of-sample, net of costs and of how much we searched, robust to
stress and to the return assumptions, and written up as a decision an IC can govern.

## Already delivered (do not rebuild)

- HERC and NCO, HC bounds (Riskfolio-Lib and skfolio).
- Walk-forward OOS paths per configuration in `oos_returns`: the input for every
  statistical test below.
- skfolio as a second backend. Its `WalkForward`, `CombinatorialPurgedCV`, `SyntheticData` and
  `RiskBudgeting` are now available without a new dependency.

## Definition of done

1. For an experiment, the report shows per configuration a **Sharpe-difference test vs the
   SAA** (bootstrap p-value) and the **deflated Sharpe ratio**, plus experiment-level **PBO**
   and a **spanning test** for the candidate. Every statistic is validated on synthetic data
   with a known answer (size under the null, power under an alternative).
2. Walk-forward results are **net of transaction costs**, with the candidate's **liquidity
   terms** (dealing frequency, notice, gates) and **funding from a named source** honoured.
3. The **risk-budget** ("how much of our risk should it carry?") and **TE-sweep** ("what fits
   inside a TE budget?") experiments run from a spec and have report views.
4. A **Black–Litterman breakeven** experiment reports the view on the candidate's excess return
   needed for a given allocation, with the SAA as the prior. Versioned CMAs can replace mu.
5. A **stress** section: crisis windows on real data plus block-bootstrap paths.
6. `wb memo <experiment>` generates an **IC memo** (SCQA: recommendation, corridor, risk share,
   evidence net of search, conditions, kill criteria) from the registry alone.
7. A read-only **Streamlit UI** over the registry.
8. `pytest -q` green; golden files updated deliberately and reviewed.

## Milestones (one at a time; stop and show results after each)

Ordered by evidence value per unit of work. Milestones marked **data-gated** wait for answers
to the open questions; they can be pulled forward as soon as data arrives.

**P2-M1 Statistical evidence.** *Built 2026-10-04.* New `evaluation/inference.py`, our own
code on numpy/scipy, run on the stored paths (no refitting). Decisions made during the build:
Sharpe on returns in excess of the cash asset; spanning in excess returns over cash (alpha
test) when the SAA has one; HC3-F as the robust line; path tests skipped below 24 OOS periods.
- Sharpe difference vs the SAA path: Ledoit–Wolf (2008) studentised circular-block bootstrap,
  block length by a stated rule.
- Deflated Sharpe ratio (Bailey & López de Prado 2014), with the number of trials equal to the
  number of configurations actually run (including failed ones) and the variance of their
  Sharpe ratios.
- Probability of backtest overfitting via CSCV over the configuration × period OOS matrix.
- Spanning: Huberman–Kandel regression test and the Kan–Zhou step-down (tangency then GMV),
  for the candidate against the SAA building blocks, on full and live-only histories.
- Validation: rejection rates near 5% on synthetic null data, PBO ≈ 0.5 on pure-noise
  configurations, the DSR worked example from the paper.
- Report: an "Evidence net of search" section.

**P2-M2 Costs, funding and liquidity.** *Built 2026-10-04* (own engine, costs ex post only,
liquidity by freeze-and-rescale; plus `wb migrate` for additive registry upgrades).
- Spec `costs:` gives one-way transaction costs per building block in bps. Paths are stored
  gross and net.
- `funding:` gains named sources (asset or class) beyond `pro_rata` for `saa_plus` and the
  engine.
- Candidate `liquidity:` covers dealing frequency, notice period and gate. The engine can only
  trade the candidate on dealing dates, and a gate blocks reductions.
- Threshold rebalancing (`rebalance: {kind: threshold, band: ...}`) next to calendar.
- Cost-aware optimisers need the current holdings, so `FitContext` would gain an optional
  `current_weights` field. That is a core-signature change; ask before making it.

**P2-M3 Risk budgets, risk limits and sweeps.** *Built 2026-10-05* (cap keys carry units;
post-check on sample moments; `rest: saa` default; CVaR/CDaR realised shares disclosed).
- Risk-budget allocators `riskfolio_risk_budget(candidate_share, rm)` and a skfolio
  `RiskBudgeting` twin. The realised share is checked with `rp.Risk_Contribution` because
  linear constraints can stop a budget from being met.
- Constraint-set keys for risk-measure caps (`max_cvar`, `max_cdar`, `min_return`; annual in
  the spec, converted in `units`) and a variance risk-contribution cap
  (`candidate_max_risk_share` → `arcinequality`).
- Sweep syntax for constraint sets (e.g. `te_annual: {sweep: [0.005, 0.01, 0.02, 0.03]}`),
  expanding into named sets. Report: candidate weight vs TE budget.
- Downside lenses (`MSV`, `FLPM`, `SLPM`) and denoising estimators (`fixed`, `spectral`,
  `shrink`, detoning), verified in both libraries where both exist.

**P2-M4 Return assumptions and Black–Litterman.** *Built 2026-10-06* (CMA as an estimator,
`method_mu: cma`; dated vectors used point in time and hashed resolved; own BL posterior in both
libraries, skfolio twin via its native `BlackLitterman`; BL changes mu only, MV only; prior
strength as an SAA Sharpe; view mode and breakeven mode with per-date root-finding).
- `cma:` in the spec: a versioned expected-return vector (annual, converted in `units`) passed
  as `mu_override`, which already exists end to end. Its version goes into the provenance.
- `riskfolio_bl` allocator with the SAA as the prior, plus a view sweep on the candidate's
  excess return. Report: allocation vs view, and the breakeven premium for a target weight.

**P2-M5 Stress scenarios.** *Built 2026-10-06* (crisis windows built now and tested with an
injected crash, meaningful on real data; policy portfolios at the corridor's P25/median/P75 plus
spec weights; stationary block bootstrap with the Sharpe-test block rule; results in `evidence`,
no schema change; vine copula deferred: ~10 s per fit and i.i.d. draws understate drawdown
persistence).
- Named crisis windows (2008, 2020, 2022) evaluated on stored paths: the candidate's
  contribution to SAA drawdowns in each window. **Data-gated**: synthetic data has no real
  crises.
- Simulated paths from a stationary block bootstrap of the building blocks plus candidate, and
  optionally skfolio `SyntheticData` (vine copula). Report: distribution of max drawdown and
  CDaR with and without the candidate at the corridor's median weight.

**P2-M6 IC memo.** *Built 2026-10-06* (decision block unhashed, revisions via `--spec`;
people write the recommendation, the memo flags disagreements; kill criteria replayed on the
proposal's walk-forward path; candidate/SAA profiles stored as evidence; default SCQA layout until
the house template exists). `wb memo <experiment>`: an SCQA markdown memo from the registry. A spec
`decision:` block holds target corridor, conditions and kill criteria (e.g. "revisit if
realised TE > x for n months"). Format to follow the house template (open question).

**P2-M7 Workbench UI.** *Built 2026-10-06* (`wb ui`; streamlit==1.65.0 as optional extra
`ui`; read-only registry mode enforced by the database; localhost, telemetry off; pages:
experiments, corridor, cells, paths, evidence, libraries, memo). Read-only Streamlit over the registry: pick an experiment, read the
corridor, drill into any cell (weights, diagnostics, path), compare libraries. New
dependency: `streamlit`.

**P2-M8 Real data (data-gated).** *P2-M8a built 2026-10-06*: data contract
(docs/DATA_CONTRACT.md) and `SqlLoader` (`data.source: sql`), point-in-time vintages, proxy
backfill, file-based SAA versions hashed into the experiment id, `data_source` /
`data_vintage_tag` registry columns, `wb data check` / `wb data demo`, PostgreSQL extra with an
opt-in integration suite (passed against postgres:17). *P2-M8b* (map the house tables, real SAA
and candidate, registry on the house server) waits for the answers below. `PostgresLoader` behind the existing `Loader` protocol (no
live Bloomberg); the real SAA as an `SAA` version; the first real candidate; the registry on
PostgreSQL; data vintage tags next to the content hash.

**P2-M9 Factor panel and factor models (data-gated).** Factor returns from PostgreSQL;
`model="FM"` estimators (`hist=False`, see CLAUDE.md); factor risk contributions per
rebalance date; a factor-mimicking backfill proxy via `rp.loadings_matrix` (drop the `const`
column), with flagged observations and the live-only variant kept.

## Interfaces and schema (to approve when each milestone is planned)

- **Config-level results table** (P2-M1): `config_evidence(experiment_id, config_id,
  data_variant, test, statistic, p_value, extra_json)` plus experiment-level rows (PBO,
  spanning). `metrics` is keyed by cell (one date), so it does not fit.
- **`oos_returns`** gains `portfolio_return_net` and `cost` (P2-M2).
- **`FitContext.current_weights: pd.Series | None = None`** (P2-M2): core signature.
- **Spec additions**: `costs`, `funding` sources, `candidate.liquidity`, threshold rebalance,
  constraint-set sweeps, risk caps, `cma`, `decision`. All strict and all in `spec_hash`.
- **Dependencies**: `streamlit` (P2-M7). Bootstraps are our own code; `arch` is installed through
  riskfolio-lib, but we won't rely on a transitive dependency without declaring it.

## Out of scope for Phase 2 (Phase 3 per the concept map)

Entropy pooling, Black–Litterman on factors, uncertainty sets and worst-case optimisation,
higher-moment models (MVSK, kurtosis), relaxed risk parity, OWA, factor risk budgets, network
and integer constraints, regime analysis, Brinson attribution, combinatorial purged CV as the
main OOS engine. Live Bloomberg calls stay out permanently.

## Open questions for Patrik

Still open from Phase 1:
1. riskbench: location and entry point. P2-M2 changes the engine, so this is the last good
   moment to extend riskbench instead.
2. Mandate-adherence analyzer: location (source of policy rules and risk limits for P2-M3).
3. PostgreSQL: schema and tables for building-block and candidate returns, and where the
   registry lives.
4. Real SAA (blocks, weights, ranges, version), frequency, base currency, hedging convention.
5. The first real candidate, with its fees, dealing terms, notice and gates.

New for Phase 2:
6. Transaction-cost assumptions per building block (bps, one-way). Is there a house table?
7. House CMAs: source, horizon, update cadence and versioning; arithmetic (expected annual return, assumed now) or geometric (needs a variance-drag adjustment).
8. Factor panel: which factors, source table and history length.
9. IC memo: house template, and who sets the kill criteria.
10. Statistical conventions: significance level, block-length rule, and whether PBO/DSR are
    gates (must pass) or disclosures (must be shown).

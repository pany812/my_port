# Data contract: what the workbench reads from the database

The workbench never calls Bloomberg. It reads returns from PostgreSQL through **two views** (or
tables) with the columns below. Map them onto the Bloomberg-fed tables with SQL views; the
workbench needs nothing else from the database. A demo database with these tables:
`wb data demo` (SQLite, synthetic data).

## `wb_assets`: one row per series

| column | type | meaning |
|---|---|---|
| `asset_id` | text, unique | the id used in SAA files and specs (e.g. `SE_EQ`, `CAND`) |
| `kind` | text | `block` (SAA building block), `cash`, `candidate`, or `proxy` (backfill series) |
| `currency` | text | ISO code of the series (informational; returns must already be in base currency) |
| `hedged` | boolean | whether the series is currency-hedged to the base currency (informational) |
| `description` | text | free text (informational) |

## `wb_returns`: one row per series, period and load

| column | type | meaning |
|---|---|---|
| `asset_id` | text | references `wb_assets.asset_id` |
| `period_end` | date | the last day of the period (any day inside the period works; data are aligned by period) |
| `frequency` | text | `D` (business days), `W`, `M`, `Q` or `A` |
| `simple_return` | double | simple total return over the period, decimal (0.012 = 1.2%) |
| `vintage` | text | load tag that **sorts in load order**, e.g. an ISO timestamp `2026-10-01T06:00` |

Conventions the views must deliver (the workbench does not convert):

- **Total returns** (income reinvested), **simple** (not log), per period.
- **In the base currency of the experiment**, following its **hedging convention** (e.g. fixed
  income SEK-hedged, equities unhedged). If both versions exist, expose them as separate
  `asset_id`s (e.g. `GL_EQ` and `GL_EQ_H`).
- **Net of fees** for the candidate (and for building blocks where that is the house convention).
- **Gaps are not filled.** A missing period stays missing; the workbench reports it
  (`wb data check`) and refuses to run with gaps after the common start.

## Vintages (point in time)

Each row carries the load that produced it. A spec reads either `vintage: latest` or a tag; for
every (`asset_id`, `period_end`, `frequency`) the workbench takes the row with the **greatest
vintage ≤ the tag**. Revisions therefore never change an old experiment: re-running with its tag
reproduces it exactly, and the content hash (`data_vintage`) proves it. If the database keeps no
load history, expose a constant vintage (e.g. `'0'`).

## Frequencies

Data at the experiment's frequency are used as they are. Finer data (e.g. daily) are compounded:
`(1 + r1)(1 + r2)… − 1` per period; a first or last period with fewer than half the usual number
of observations is dropped as partial. Coarser data are an error.

## The candidate's history

The spec's `data.sql.live_start` is the candidate's first live period. Earlier periods are
flagged as backfilled and, when `candidate_proxy` names a `proxy` series, filled from it. Every
report then shows a live-only variant alongside the full history.

## Access

- Connection: a SQLAlchemy URL in an environment variable (`WB_DATA_URL` by default, see
  `data.sql.url_env`), e.g. `postgresql+psycopg://wb_reader@host:5432/analytics`. Credentials
  never go into specs, the registry or logs (messages redact passwords).
- The workbench opens the connection **read-only** (`default_transaction_read_only`); please
  also give it a role with `SELECT` on the two views only.
- Optional `data.sql.schema` if the views live outside the role's default schema.

## Spec section

```yaml
data:
  source: sql
  frequency: M
  start: 2006-01
  end: 2026-09
  base_currency: SEK
  hedging: "fixed income SEK-hedged, equity unhedged"
  candidate: CAND
  sql:
    url_env: WB_DATA_URL
    schema: workbench_data        # optional
    vintage: latest               # or a tag
    live_start: 2016-01           # optional
    candidate_proxy: CAND_PROXY   # optional, needs live_start
saa: {version: house_2026}        # saa/house_2026.yaml
```

## Example views (to adapt)

```sql
create view workbench_data.wb_assets as
select series_code as asset_id, role as kind, ccy as currency, is_hedged as hedged,
       long_name as description
from analytics.series_master where in_workbench;

create view workbench_data.wb_returns as
select s.series_code as asset_id, r.as_of_date as period_end, 'M' as frequency,
       r.tr_return_sek as simple_return, to_char(r.loaded_at, 'YYYY-MM-DD"T"HH24:MI') as vintage
from analytics.monthly_returns r join analytics.series_master s using (series_id)
where s.in_workbench;
```

-- Data contract views over the house tables (P2-M8b; docs/DATA_CONTRACT.md).
-- Decisions 1a / 2a: the house holds monthly total returns already in SEK per the hedging
-- convention, and every row carries its load timestamp. Run after roles.sql, as an
-- administrator. Idempotent.
--
-- EDIT the house names marked "EDIT" (house.series, house.monthly_returns and their columns).
-- Everything else is fixed by the contract.

-- Which house series the workbench uses, under which asset id and role. Maintained by the
-- workbench team (wb_writer); the house tables are never changed.
CREATE TABLE IF NOT EXISTS workbench_data.asset_map (
    asset_id        text PRIMARY KEY,   -- the id used in saa/*.yaml and specs, e.g. SE_EQ, CAND
    house_series_id text NOT NULL,      -- EDIT if the house key is not text
    kind            text NOT NULL
        CHECK (kind IN ('block', 'cash', 'candidate', 'proxy', 'benchmark')),
    description     text
);

CREATE OR REPLACE VIEW workbench_data.wb_assets AS
SELECT m.asset_id,
       m.kind,
       s.currency,                                  -- EDIT: ISO currency column
       s.is_hedged AS hedged,                       -- EDIT: hedged flag column
       coalesce(m.description, s.name) AS description   -- EDIT: series name column
FROM workbench_data.asset_map m
JOIN house.series s ON s.series_id = m.house_series_id;          -- EDIT: series master

CREATE OR REPLACE VIEW workbench_data.wb_returns AS
SELECT m.asset_id,
       r.period_end,                                -- EDIT: month-end date column
       'M'::text AS frequency,
       r.total_return_sek AS simple_return,         -- EDIT: simple total return in SEK, decimal
       to_char(r.loaded_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS') AS vintage  -- EDIT
FROM workbench_data.asset_map m
JOIN house.monthly_returns r ON r.series_id = m.house_series_id; -- EDIT: monthly returns

GRANT SELECT ON workbench_data.wb_assets, workbench_data.wb_returns TO wb_data_reader;
GRANT SELECT, INSERT, UPDATE, DELETE ON workbench_data.asset_map TO wb_writer;
-- the views read the house tables with the view owner's rights: wb_data_reader needs no grant
-- on the house schema, and gets none

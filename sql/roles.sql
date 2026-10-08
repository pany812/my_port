-- Workbench roles and schemas on the house PostgreSQL server (P2-M8b, decision 3a).
-- Run once as a database administrator, before contract_views.sql. Idempotent.
--
--   workbench_data  the data contract: asset map + views wb_assets, wb_returns
--   workbench       the registry (experiments, cells, weights, metrics, oos_returns, evidence)
--
--   wb_data_reader  `WB_DATA_URL`  reads the two contract views, nothing else
--   wb_writer       `WB_REGISTRY` for `wb run`: owns the registry schema, maintains the asset map
--   wb_ui           `WB_REGISTRY` for `wb ui` / `wb report` / `wb memo`: reads the registry
--
-- Authentication (passwords, LDAP, ...) is set by the DBA outside this file; no secrets here.

CREATE SCHEMA IF NOT EXISTS workbench_data;
CREATE SCHEMA IF NOT EXISTS workbench;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'wb_data_reader') THEN
        CREATE ROLE wb_data_reader LOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'wb_writer') THEN
        CREATE ROLE wb_writer LOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'wb_ui') THEN
        CREATE ROLE wb_ui LOGIN;
    END IF;
END
$$;

-- the registry: wb_writer creates and owns its tables; wb_ui reads them, including tables that
-- later versions add (`wb migrate` runs as wb_writer)
ALTER SCHEMA workbench OWNER TO wb_writer;
ALTER ROLE wb_writer SET search_path = workbench;
ALTER ROLE wb_ui SET search_path = workbench;
GRANT USAGE ON SCHEMA workbench TO wb_ui;
GRANT SELECT ON ALL TABLES IN SCHEMA workbench TO wb_ui;
ALTER DEFAULT PRIVILEGES FOR ROLE wb_writer IN SCHEMA workbench GRANT SELECT ON TABLES TO wb_ui;
ALTER ROLE wb_ui SET default_transaction_read_only = on;

-- the data contract: read-only for everyone in the workbench
ALTER ROLE wb_data_reader SET default_transaction_read_only = on;
GRANT USAGE ON SCHEMA workbench_data TO wb_data_reader, wb_writer;

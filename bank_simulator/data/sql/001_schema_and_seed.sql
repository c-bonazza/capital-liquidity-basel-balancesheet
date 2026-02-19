-- =============================================================================
-- Integrated Bank Balance Sheet Simulator & Capital Optimizer (ICAAP/ILAAP) — Data Mart DDL & Seed
-- =============================================================================
-- Purpose : Structure raw banking data into a clean analytical Data Mart
--           using complex CTEs, window functions, and CASE expressions.
-- Engine  : SQLite (compatible with DuckDB / PostgreSQL with minor changes)
-- =============================================================================

-- ─── 1. REFERENCE / DIMENSION TABLES ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_rating_grades (
    grade_id        INTEGER PRIMARY KEY,
    grade_label     TEXT    NOT NULL,        -- e.g. 'AAA', 'A', 'BBB', 'Sub-IG', 'Default'
    pd_floor        REAL    NOT NULL,
    pd_ceiling      REAL    NOT NULL,
    is_default      INTEGER NOT NULL DEFAULT 0
);

INSERT INTO dim_rating_grades (grade_id, grade_label, pd_floor, pd_ceiling, is_default) VALUES
    (1, 'AAA',    0.0000, 0.0010, 0),
    (2, 'A',      0.0010, 0.0050, 0),
    (3, 'BBB',    0.0050, 0.0200, 0),
    (4, 'Sub-IG', 0.0200, 0.1500, 0),
    (5, 'Default', 1.0000, 1.0000, 1);


CREATE TABLE IF NOT EXISTS dim_asset_class (
    asset_class_id   INTEGER PRIMARY KEY,
    asset_class_name TEXT    NOT NULL,
    risk_weight_sa   REAL    NOT NULL,   -- Standardised-Approach risk weight
    avg_lgd          REAL    NOT NULL
);

INSERT INTO dim_asset_class (asset_class_id, asset_class_name, risk_weight_sa, avg_lgd) VALUES
    (1, 'Residential Mortgages', 0.35, 0.15),
    (2, 'SME Loans',             0.75, 0.40),
    (3, 'Consumer Credit',       0.75, 0.55),
    (4, 'Corporate Loans',       0.65, 0.35);


CREATE TABLE IF NOT EXISTS dim_liability_type (
    liability_type_id  INTEGER PRIMARY KEY,
    liability_name     TEXT    NOT NULL,
    run_off_rate       REAL    NOT NULL,   -- Basel III 30-day run-off
    is_retail          INTEGER NOT NULL DEFAULT 0
);

INSERT INTO dim_liability_type (liability_type_id, liability_name, run_off_rate, is_retail) VALUES
    (1, 'Retail Deposits — Stable',       0.05, 1),
    (2, 'Retail Deposits — Less Stable',  0.10, 1),
    (3, 'Wholesale Operational',          0.25, 0),
    (4, 'Wholesale Non-Operational',      0.40, 0),
    (5, 'Wholesale Unsecured',            0.75, 0);


-- ─── 2. FACT TABLES ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_loan_book (
    loan_id          INTEGER PRIMARY KEY,
    asset_class_id   INTEGER NOT NULL REFERENCES dim_asset_class(asset_class_id),
    grade_id         INTEGER NOT NULL REFERENCES dim_rating_grades(grade_id),
    origination_date TEXT    NOT NULL,
    maturity_date    TEXT    NOT NULL,
    ead              REAL    NOT NULL,     -- Exposure at Default (€m)
    interest_rate    REAL    NOT NULL,     -- Contractual rate
    is_fixed_rate    INTEGER NOT NULL DEFAULT 1,
    ifrs9_stage      INTEGER NOT NULL DEFAULT 1   -- 1, 2, or 3
);

CREATE TABLE IF NOT EXISTS fact_deposits (
    deposit_id        INTEGER PRIMARY KEY,
    liability_type_id INTEGER NOT NULL REFERENCES dim_liability_type(liability_type_id),
    balance           REAL    NOT NULL,    -- €m
    interest_rate     REAL    NOT NULL,
    maturity_bucket   TEXT    NOT NULL     -- 'demand', '< 1M', '1-3M', '3-6M', '6-12M', '> 1Y'
);

CREATE TABLE IF NOT EXISTS fact_hqla_portfolio (
    hqla_id      INTEGER PRIMARY KEY,
    hqla_level   TEXT    NOT NULL,        -- 'L1', 'L2A', 'L2B'
    instrument   TEXT    NOT NULL,
    face_value   REAL    NOT NULL,        -- €m
    market_value REAL    NOT NULL,
    haircut      REAL    NOT NULL,
    duration     REAL    NOT NULL         -- Macaulay duration (years)
);

CREATE TABLE IF NOT EXISTS fact_macro_scenarios (
    scenario_id   INTEGER PRIMARY KEY,
    scenario_name TEXT    NOT NULL,
    year_offset   INTEGER NOT NULL,       -- 0, 1, 2 (projection year)
    gdp_growth    REAL    NOT NULL,
    ir_shock_bps  REAL    NOT NULL,
    unemployment  REAL    NOT NULL,
    hpi_change    REAL    NOT NULL,
    pd_multiplier REAL    NOT NULL
);


-- ─── 3. SEED DATA — Synthetic Loan Book (200 representative facilities) ─────

-- Residential Mortgages
INSERT INTO fact_loan_book (loan_id, asset_class_id, grade_id, origination_date, maturity_date, ead, interest_rate, is_fixed_rate, ifrs9_stage)
SELECT
    row_number() OVER () AS loan_id,
    1 AS asset_class_id,
    CASE
        WHEN abs(random()) % 100 < 60 THEN 1
        WHEN abs(random()) % 100 < 85 THEN 2
        WHEN abs(random()) % 100 < 95 THEN 3
        ELSE 4
    END AS grade_id,
    date('2018-01-01', '+' || (abs(random()) % 2000) || ' days') AS origination_date,
    date('2035-01-01', '+' || (abs(random()) % 3650) || ' days') AS maturity_date,
    round(0.15 + (abs(random()) % 300) / 100.0, 2) AS ead,
    round(0.015 + (abs(random()) % 200) / 10000.0, 4) AS interest_rate,
    CASE WHEN abs(random()) % 100 < 70 THEN 1 ELSE 0 END AS is_fixed_rate,
    CASE
        WHEN abs(random()) % 100 < 85 THEN 1
        WHEN abs(random()) % 100 < 95 THEN 2
        ELSE 3
    END AS ifrs9_stage
FROM (
    -- Generate 80 rows for mortgages
    WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 80)
    SELECT x FROM cnt
);

-- SME Loans
INSERT INTO fact_loan_book (loan_id, asset_class_id, grade_id, origination_date, maturity_date, ead, interest_rate, is_fixed_rate, ifrs9_stage)
SELECT
    80 + row_number() OVER () AS loan_id,
    2 AS asset_class_id,
    CASE
        WHEN abs(random()) % 100 < 30 THEN 1
        WHEN abs(random()) % 100 < 65 THEN 2
        WHEN abs(random()) % 100 < 90 THEN 3
        ELSE 4
    END AS grade_id,
    date('2019-01-01', '+' || (abs(random()) % 1500) || ' days') AS origination_date,
    date('2027-01-01', '+' || (abs(random()) % 2500) || ' days') AS maturity_date,
    round(0.5 + (abs(random()) % 500) / 100.0, 2) AS ead,
    round(0.025 + (abs(random()) % 400) / 10000.0, 4) AS interest_rate,
    CASE WHEN abs(random()) % 100 < 40 THEN 1 ELSE 0 END AS is_fixed_rate,
    CASE
        WHEN abs(random()) % 100 < 75 THEN 1
        WHEN abs(random()) % 100 < 92 THEN 2
        ELSE 3
    END AS ifrs9_stage
FROM (
    WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 60)
    SELECT x FROM cnt
);

-- Consumer Credit
INSERT INTO fact_loan_book (loan_id, asset_class_id, grade_id, origination_date, maturity_date, ead, interest_rate, is_fixed_rate, ifrs9_stage)
SELECT
    140 + row_number() OVER () AS loan_id,
    3 AS asset_class_id,
    CASE
        WHEN abs(random()) % 100 < 25 THEN 1
        WHEN abs(random()) % 100 < 60 THEN 2
        WHEN abs(random()) % 100 < 88 THEN 3
        ELSE 4
    END AS grade_id,
    date('2020-01-01', '+' || (abs(random()) % 1200) || ' days') AS origination_date,
    date('2026-01-01', '+' || (abs(random()) % 1800) || ' days') AS maturity_date,
    round(0.01 + (abs(random()) % 100) / 100.0, 2) AS ead,
    round(0.045 + (abs(random()) % 600) / 10000.0, 4) AS interest_rate,
    1 AS is_fixed_rate,
    CASE
        WHEN abs(random()) % 100 < 70 THEN 1
        WHEN abs(random()) % 100 < 88 THEN 2
        ELSE 3
    END AS ifrs9_stage
FROM (
    WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 30)
    SELECT x FROM cnt
);

-- Corporate Loans
INSERT INTO fact_loan_book (loan_id, asset_class_id, grade_id, origination_date, maturity_date, ead, interest_rate, is_fixed_rate, ifrs9_stage)
SELECT
    170 + row_number() OVER () AS loan_id,
    4 AS asset_class_id,
    CASE
        WHEN abs(random()) % 100 < 40 THEN 1
        WHEN abs(random()) % 100 < 75 THEN 2
        WHEN abs(random()) % 100 < 93 THEN 3
        ELSE 4
    END AS grade_id,
    date('2019-06-01', '+' || (abs(random()) % 1800) || ' days') AS origination_date,
    date('2028-01-01', '+' || (abs(random()) % 3000) || ' days') AS maturity_date,
    round(1.0 + (abs(random()) % 1000) / 100.0, 2) AS ead,
    round(0.020 + (abs(random()) % 350) / 10000.0, 4) AS interest_rate,
    CASE WHEN abs(random()) % 100 < 50 THEN 1 ELSE 0 END AS is_fixed_rate,
    CASE
        WHEN abs(random()) % 100 < 80 THEN 1
        WHEN abs(random()) % 100 < 94 THEN 2
        ELSE 3
    END AS ifrs9_stage
FROM (
    WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 30)
    SELECT x FROM cnt
);


-- ─── 4. SEED DATA — Deposits ────────────────────────────────────────────────
INSERT INTO fact_deposits (deposit_id, liability_type_id, balance, interest_rate, maturity_bucket) VALUES
    (1,  1, 12000.0, 0.005, 'demand'),
    (2,  1,  5000.0, 0.008, '> 1Y'),
    (3,  1,  5000.0, 0.006, '3-6M'),
    (4,  2,  4000.0, 0.012, 'demand'),
    (5,  2,  2000.0, 0.015, '< 1M'),
    (6,  2,  2000.0, 0.018, '1-3M'),
    (7,  3,  3000.0, 0.020, '1-3M'),
    (8,  3,  3000.0, 0.022, '3-6M'),
    (9,  4,  2500.0, 0.030, '< 1M'),
    (10, 4,  2500.0, 0.035, '1-3M'),
    (11, 5,  2000.0, 0.040, 'demand'),
    (12, 5,  2000.0, 0.045, '< 1M');


-- ─── 5. SEED DATA — HQLA Portfolio ─────────────────────────────────────────
INSERT INTO fact_hqla_portfolio (hqla_id, hqla_level, instrument, face_value, market_value, haircut, duration) VALUES
    (1, 'L1', 'Central Bank Reserves',    2000.0, 2000.0, 0.00, 0.0),
    (2, 'L1', 'German Bunds 2Y',          2000.0, 1980.0, 0.00, 1.9),
    (3, 'L1', 'French OAT 5Y',            1500.0, 1460.0, 0.00, 4.6),
    (4, 'L1', 'Luxembourg Govt 3Y',       1500.0, 1490.0, 0.00, 2.8),
    (5, 'L2A', 'Covered Bond AAA 3Y',     1200.0, 1190.0, 0.15, 2.7),
    (6, 'L2A', 'Covered Bond AA 5Y',       800.0,  780.0, 0.15, 4.3),
    (7, 'L2B', 'Corporate Bond A 3Y',      600.0,  580.0, 0.50, 2.6),
    (8, 'L2B', 'RMBS AAA 5Y',              400.0,  370.0, 0.50, 3.8);


-- ─── 6. SEED DATA — Macro Scenarios ────────────────────────────────────────
INSERT INTO fact_macro_scenarios VALUES
    (1, 'Baseline',          0,  1.8,   0, 6.5,   2.0, 1.0),
    (2, 'Baseline',          1,  2.0,   0, 6.3,   2.5, 1.0),
    (3, 'Baseline',          2,  2.1,   0, 6.1,   3.0, 1.0),
    (4, 'Adverse',           0, -1.5, 200, 9.0,  -8.0, 2.0),
    (5, 'Adverse',           1,  0.2, 250, 10.5, -12.0, 3.0),
    (6, 'Adverse',           2,  1.0, 200, 9.8,  -5.0, 2.5),
    (7, 'Severely Adverse',  0, -4.0, 400, 12.0, -20.0, 3.5),
    (8, 'Severely Adverse',  1, -1.5, 450, 14.0, -25.0, 5.0),
    (9, 'Severely Adverse',  2,  0.5, 350, 13.0, -10.0, 4.0);


-- =============================================================================
-- 7. ANALYTICAL VIEWS (Complex CTEs for Data Mart consumption)
-- =============================================================================

-- ─── VIEW: Loan-level RWA & Expected Loss ───────────────────────────────────
CREATE VIEW IF NOT EXISTS v_loan_rwa_el AS
WITH loan_params AS (
    SELECT
        l.loan_id,
        l.asset_class_id,
        a.asset_class_name,
        g.grade_label,
        l.ead,
        g.pd_ceiling                          AS pd_pit,
        a.avg_lgd                             AS lgd,
        a.risk_weight_sa                      AS rw,
        l.ifrs9_stage,
        julianday(l.maturity_date) - julianday(l.origination_date) AS days_to_maturity,
        l.interest_rate
    FROM fact_loan_book l
    JOIN dim_asset_class a     ON l.asset_class_id = a.asset_class_id
    JOIN dim_rating_grades g   ON l.grade_id       = g.grade_id
),
rwa_calc AS (
    SELECT
        *,
        ead * rw                              AS rwa,
        ead * pd_pit * lgd                    AS expected_loss_12m,
        CASE
            WHEN ifrs9_stage = 1 THEN ead * pd_pit * lgd
            WHEN ifrs9_stage = 2 THEN ead * pd_pit * lgd * (days_to_maturity / 365.25)
            WHEN ifrs9_stage = 3 THEN ead * lgd
            ELSE 0
        END                                   AS ecl_provision
    FROM loan_params
)
SELECT
    loan_id,
    asset_class_name,
    grade_label,
    ifrs9_stage,
    round(ead, 2)               AS ead,
    round(rwa, 2)               AS rwa,
    round(expected_loss_12m, 4) AS expected_loss_12m,
    round(ecl_provision, 4)     AS ecl_provision,
    round(interest_rate, 4)     AS interest_rate
FROM rwa_calc;


-- ─── VIEW: Aggregated RWA by Asset Class ────────────────────────────────────
CREATE VIEW IF NOT EXISTS v_portfolio_summary AS
WITH base AS (
    SELECT * FROM v_loan_rwa_el
)
SELECT
    asset_class_name,
    count(*)                    AS num_facilities,
    round(sum(ead), 2)          AS total_ead,
    round(sum(rwa), 2)          AS total_rwa,
    round(sum(expected_loss_12m), 4)  AS total_el,
    round(sum(ecl_provision), 4)      AS total_ecl,
    round(avg(interest_rate), 4)      AS avg_rate,
    round(sum(rwa) / nullif(sum(ead), 0), 4) AS avg_risk_weight
FROM base
GROUP BY asset_class_name;


-- ─── VIEW: LCR Components ──────────────────────────────────────────────────
CREATE VIEW IF NOT EXISTS v_lcr_components AS
WITH hqla AS (
    SELECT
        hqla_level,
        sum(market_value * (1 - haircut))  AS adjusted_value,
        sum(market_value * (1 - haircut) * duration) / 
            nullif(sum(market_value * (1 - haircut)), 0)  AS weighted_duration
    FROM fact_hqla_portfolio
    GROUP BY hqla_level
),
outflows AS (
    SELECT
        lt.liability_name,
        lt.run_off_rate,
        d.balance,
        d.balance * lt.run_off_rate  AS stressed_outflow
    FROM fact_deposits d
    JOIN dim_liability_type lt ON d.liability_type_id = lt.liability_type_id
)
SELECT
    'HQLA'           AS component,
    hqla_level       AS detail,
    adjusted_value   AS amount,
    weighted_duration AS metric
FROM hqla
UNION ALL
SELECT
    'Outflow'        AS component,
    liability_name   AS detail,
    stressed_outflow AS amount,
    run_off_rate     AS metric
FROM outflows;


-- ─── VIEW: IFRS 9 Stage Distribution ───────────────────────────────────────
CREATE VIEW IF NOT EXISTS v_ifrs9_stage_summary AS
SELECT
    asset_class_name,
    ifrs9_stage,
    count(*)                        AS num_loans,
    round(sum(ead), 2)              AS total_ead,
    round(sum(ecl_provision), 4)    AS total_provision,
    round(
        sum(ecl_provision) / nullif(sum(ead), 0) * 100, 4
    )                               AS coverage_ratio_pct
FROM v_loan_rwa_el
GROUP BY asset_class_name, ifrs9_stage
ORDER BY asset_class_name, ifrs9_stage;

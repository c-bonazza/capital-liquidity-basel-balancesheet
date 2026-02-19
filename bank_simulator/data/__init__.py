"""
Data ingestion layer — SQLite-based Data Mart builder.
Reads the SQL DDL/seed script, initialises an in-memory (or file-backed)
database, and exposes DataFrames for downstream consumption.
"""

import sqlite3
import pathlib
import pandas as pd
from typing import Optional, Dict


SQL_DIR = pathlib.Path(__file__).parent / "sql"


class DataMart:
    """Builds and manages the analytical Data Mart from raw SQL scripts."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._initialised = False

    # ── Bootstrap ────────────────────────────────────────────────────────
    def initialise(self) -> "DataMart":
        """Run DDL + seed scripts to populate the Data Mart."""
        sql_file = SQL_DIR / "001_schema_and_seed.sql"
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL seed file not found: {sql_file}")
        script = sql_file.read_text(encoding="utf-8")
        self.conn.executescript(script)
        self._initialised = True
        return self

    # ── Generic query ────────────────────────────────────────────────────
    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame."""
        return pd.read_sql_query(sql, self.conn)

    # ── Pre-built extracts ───────────────────────────────────────────────
    def get_loan_book(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_loan_rwa_el")

    def get_portfolio_summary(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_portfolio_summary")

    def get_lcr_components(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_lcr_components")

    def get_ifrs9_summary(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_ifrs9_stage_summary")

    def get_hqla_portfolio(self) -> pd.DataFrame:
        return self.query("SELECT * FROM fact_hqla_portfolio")

    def get_deposits(self) -> pd.DataFrame:
        return self.query("""
            SELECT d.*, lt.liability_name, lt.run_off_rate, lt.is_retail
            FROM fact_deposits d
            JOIN dim_liability_type lt ON d.liability_type_id = lt.liability_type_id
        """)

    def get_macro_scenarios(self) -> pd.DataFrame:
        return self.query("SELECT * FROM fact_macro_scenarios")

    # ── Aggregate helpers ────────────────────────────────────────────────
    def get_total_rwa(self) -> float:
        row = self.query("SELECT sum(rwa) AS total FROM v_loan_rwa_el")
        return float(row["total"].iloc[0])

    def get_total_el(self) -> float:
        row = self.query("SELECT sum(expected_loss_12m) AS total FROM v_loan_rwa_el")
        return float(row["total"].iloc[0])

    def get_total_ecl(self) -> float:
        row = self.query("SELECT sum(ecl_provision) AS total FROM v_loan_rwa_el")
        return float(row["total"].iloc[0])

    # ── Clean up ─────────────────────────────────────────────────────────
    def close(self):
        self.conn.close()

    def __enter__(self):
        self.initialise()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

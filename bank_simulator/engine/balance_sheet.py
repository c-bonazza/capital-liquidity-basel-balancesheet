"""
Core Balance Sheet Engine
=========================
Central object-oriented representation of the bank's balance sheet.
Supports multi-year projection under macro scenarios with dynamic
P&L, capital, and RWA recomputation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bank_simulator.config import (
    InitialBalanceSheet,
    MacroScenario,
    PROJECTION_HORIZON_YEARS,
    DEFAULT_CREDIT_SEGMENTS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Balance Sheet Snapshot (one point in time)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BalanceSheetSnapshot:
    """Immutable snapshot of the bank at a single point in time."""
    year: int

    # Assets (€m)
    cash_and_reserves: float = 0.0
    hqla_l1: float = 0.0
    hqla_l2a: float = 0.0
    hqla_l2b: float = 0.0
    residential_mortgages: float = 0.0
    sme_loans: float = 0.0
    consumer_credit: float = 0.0
    corporate_loans: float = 0.0
    trading_book: float = 0.0
    other_assets: float = 0.0

    # Liabilities (€m)
    retail_deposits_stable: float = 0.0
    retail_deposits_less_stable: float = 0.0
    wholesale_operational: float = 0.0
    wholesale_non_operational: float = 0.0
    wholesale_unsecured: float = 0.0
    subordinated_debt: float = 0.0
    other_liabilities: float = 0.0

    # Equity (€m)
    cet1_capital: float = 0.0
    at1_capital: float = 0.0
    tier2_capital: float = 0.0

    # P&L (€m) — annual
    net_interest_income: float = 0.0
    fee_income: float = 0.0
    trading_income: float = 0.0
    operating_costs: float = 0.0
    credit_losses: float = 0.0
    op_risk_losses: float = 0.0
    market_risk_pnl: float = 0.0

    # Computed risk metrics
    total_rwa: float = 0.0
    ecl_provision: float = 0.0

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def total_assets(self) -> float:
        return (self.cash_and_reserves + self.hqla_l1 + self.hqla_l2a +
                self.hqla_l2b + self.residential_mortgages + self.sme_loans +
                self.consumer_credit + self.corporate_loans +
                self.trading_book + self.other_assets)

    @property
    def total_loan_book(self) -> float:
        return (self.residential_mortgages + self.sme_loans +
                self.consumer_credit + self.corporate_loans)

    @property
    def total_liabilities(self) -> float:
        return (self.retail_deposits_stable + self.retail_deposits_less_stable +
                self.wholesale_operational + self.wholesale_non_operational +
                self.wholesale_unsecured + self.subordinated_debt +
                self.other_liabilities)

    @property
    def total_equity(self) -> float:
        return self.cet1_capital + self.at1_capital + self.tier2_capital

    @property
    def total_hqla(self) -> float:
        return (self.cash_and_reserves + self.hqla_l1 +
                self.hqla_l2a * 0.85 + self.hqla_l2b * 0.50)

    @property
    def net_income(self) -> float:
        return (self.net_interest_income + self.fee_income +
                self.trading_income + self.operating_costs -  # costs are negative
                self.credit_losses - self.op_risk_losses + self.market_risk_pnl)

    @property
    def cet1_ratio(self) -> float:
        if self.total_rwa <= 0:
            return 0.0
        return self.cet1_capital / self.total_rwa

    @property
    def total_capital_ratio(self) -> float:
        if self.total_rwa <= 0:
            return 0.0
        return self.total_equity / self.total_rwa

    @property
    def leverage_ratio(self) -> float:
        if self.total_assets <= 0:
            return 0.0
        return self.tier1_capital / self.total_assets

    @property
    def tier1_capital(self) -> float:
        return self.cet1_capital + self.at1_capital

    @property
    def roe(self) -> float:
        if self.total_equity <= 0:
            return 0.0
        return self.net_income / self.total_equity


# ═══════════════════════════════════════════════════════════════════════════════
#  Balance Sheet Projector
# ═══════════════════════════════════════════════════════════════════════════════

class BalanceSheetProjector:
    """
    Projects the bank balance sheet forward over a multi-year horizon.
    
    Acts as a coordinator that:
    1. Takes a t=0 snapshot
    2. Applies macro-driven P&L impacts from each risk module
    3. Recomputes capital, RWA, and key ratios at each time step
    """

    def __init__(self, initial_bs: InitialBalanceSheet):
        self.initial_bs = initial_bs
        self.snapshots: List[BalanceSheetSnapshot] = []

    def build_t0_snapshot(self) -> BalanceSheetSnapshot:
        """Construct the starting balance sheet snapshot."""
        bs = self.initial_bs
        snap = BalanceSheetSnapshot(
            year=0,
            cash_and_reserves=bs.cash_and_reserves,
            hqla_l1=bs.govt_bonds_hqla_l1,
            hqla_l2a=bs.covered_bonds_hqla_l2a,
            hqla_l2b=bs.corporate_bonds_hqla_l2b,
            residential_mortgages=bs.residential_mortgages,
            sme_loans=bs.sme_loans,
            consumer_credit=bs.consumer_credit,
            corporate_loans=bs.corporate_loans,
            trading_book=bs.trading_book,
            other_assets=bs.other_assets,
            retail_deposits_stable=bs.retail_deposits_stable,
            retail_deposits_less_stable=bs.retail_deposits_less_stable,
            wholesale_operational=bs.wholesale_operational,
            wholesale_non_operational=bs.wholesale_non_operational,
            wholesale_unsecured=bs.wholesale_unsecured,
            subordinated_debt=bs.subordinated_debt,
            other_liabilities=bs.other_liabilities,
            cet1_capital=bs.cet1_capital,
            at1_capital=bs.at1_capital,
            tier2_capital=bs.tier2_capital,
            net_interest_income=bs.net_interest_income,
            fee_income=bs.fee_income,
            trading_income=bs.trading_income,
            operating_costs=bs.operating_costs,
        )
        # Compute t=0 RWA from standardised approach
        snap.total_rwa = self._compute_rwa(snap)
        return snap

    def _compute_rwa(self, snap: BalanceSheetSnapshot) -> float:
        """Compute total RWA using standardised risk weights."""
        rwa = 0.0
        # By segment mapping: (balance, RW)
        segments = [
            (snap.cash_and_reserves, 0.00),
            (snap.hqla_l1, 0.00),
            (snap.hqla_l2a, 0.20),
            (snap.hqla_l2b, 0.50),
            (snap.residential_mortgages, 0.35),
            (snap.sme_loans, 0.75),
            (snap.consumer_credit, 0.75),
            (snap.corporate_loans, 0.65),
            (snap.trading_book, 1.00),       # Market risk RWA simplified
            (snap.other_assets, 1.00),
        ]
        for balance, rw in segments:
            rwa += balance * rw
        return rwa

    def project(
        self,
        scenario: MacroScenario,
        credit_losses_by_year: List[float],
        oprisk_losses_by_year: List[float],
        market_pnl_by_year: List[float],
        nii_adjustment_by_year: Optional[List[float]] = None,
        deposit_outflow_by_year: Optional[List[float]] = None,
    ) -> List[BalanceSheetSnapshot]:
        """
        Project the balance sheet forward under a given scenario.

        Parameters
        ----------
        scenario : MacroScenario
        credit_losses_by_year : losses from credit risk module
        oprisk_losses_by_year : losses from op risk module
        market_pnl_by_year : P&L impact from market risk (EVE delta)
        nii_adjustment_by_year : change in NII from IRRBB
        deposit_outflow_by_year : deposit erosion from liquidity module
        """
        self.snapshots = []
        prev = self.build_t0_snapshot()
        self.snapshots.append(prev)

        n_years = min(len(scenario.gdp_growth), PROJECTION_HORIZON_YEARS)

        for yr in range(n_years):
            snap = copy.deepcopy(prev)
            snap.year = yr + 1

            # ── 1. Interest income adjustment (IRRBB) ───────────────────
            if nii_adjustment_by_year:
                snap.net_interest_income = prev.net_interest_income + nii_adjustment_by_year[yr]

            # ── 2. Fee & cost scaling (GDP-linked) ───────────────────────
            gdp_factor = 1 + scenario.gdp_growth[yr] / 100.0
            snap.fee_income = prev.fee_income * max(gdp_factor, 0.85)
            snap.trading_income = prev.trading_income * max(gdp_factor, 0.70)
            snap.operating_costs = prev.operating_costs  # Costs are sticky

            # ── 3. Credit losses ─────────────────────────────────────────
            snap.credit_losses = credit_losses_by_year[yr]

            # ── 4. Operational risk losses ───────────────────────────────
            snap.op_risk_losses = oprisk_losses_by_year[yr]

            # ── 5. Market risk P&L ───────────────────────────────────────
            snap.market_risk_pnl = market_pnl_by_year[yr]

            # ── 6. Update capital (retained earnings) ────────────────────
            net_pl = snap.net_income
            snap.cet1_capital = prev.cet1_capital + net_pl * 0.70  # 30% dividend
            snap.cet1_capital = max(snap.cet1_capital, 0)  # Floor at zero

            # ── 7. Loan book growth (GDP-linked, constrained) ────────────
            loan_growth = max(scenario.gdp_growth[yr] / 100.0, -0.05)
            snap.residential_mortgages *= (1 + loan_growth * 0.5)
            snap.sme_loans *= (1 + loan_growth)
            snap.consumer_credit *= (1 + loan_growth * 0.7)
            snap.corporate_loans *= (1 + loan_growth * 0.8)

            # Write down loans by credit losses
            total_loans = snap.total_loan_book
            if total_loans > 0:
                loss_share = snap.credit_losses / total_loans
                snap.residential_mortgages *= max(1 - loss_share * 0.45 / 0.35, 0.90)
                snap.sme_loans *= max(1 - loss_share * 0.30 / 0.75, 0.85)
                snap.consumer_credit *= max(1 - loss_share * 0.15 / 0.75, 0.85)
                snap.corporate_loans *= max(1 - loss_share * 0.10 / 0.65, 0.85)

            # ── 8. Deposit dynamics ──────────────────────────────────────
            if deposit_outflow_by_year and deposit_outflow_by_year[yr] > 0:
                outflow = deposit_outflow_by_year[yr]
                # Distribute outflow proportionally
                total_dep = (snap.retail_deposits_stable +
                             snap.retail_deposits_less_stable +
                             snap.wholesale_operational +
                             snap.wholesale_non_operational +
                             snap.wholesale_unsecured)
                if total_dep > 0:
                    factor = max(1 - outflow / total_dep, 0.70)
                    snap.retail_deposits_stable *= factor
                    snap.retail_deposits_less_stable *= factor
                    snap.wholesale_operational *= factor
                    snap.wholesale_non_operational *= factor
                    snap.wholesale_unsecured *= factor
                    # Cash buffer absorbs the outflow
                    snap.cash_and_reserves = max(
                        snap.cash_and_reserves - outflow * 0.6, 0
                    )

            # ── 9. Recompute RWA ─────────────────────────────────────────
            snap.total_rwa = self._compute_rwa(snap)

            # ── 10. Balance check ────────────────────────────────────────
            # Plug any A=L+E gap through 'other_liabilities'
            gap = snap.total_assets - snap.total_liabilities - snap.total_equity
            snap.other_liabilities += gap

            self.snapshots.append(snap)
            prev = snap

        return self.snapshots

    def get_projection_summary(self) -> List[Dict]:
        """Return a list of dicts summarising each year's key metrics."""
        summary = []
        for s in self.snapshots:
            summary.append({
                "Year": s.year,
                "Total Assets (€m)": round(s.total_assets, 1),
                "Total Loans (€m)": round(s.total_loan_book, 1),
                "CET1 Capital (€m)": round(s.cet1_capital, 1),
                "Total RWA (€m)": round(s.total_rwa, 1),
                "CET1 Ratio (%)": round(s.cet1_ratio * 100, 2),
                "Total Capital Ratio (%)": round(s.total_capital_ratio * 100, 2),
                "Leverage Ratio (%)": round(s.leverage_ratio * 100, 2),
                "NII (€m)": round(s.net_interest_income, 1),
                "Credit Losses (€m)": round(s.credit_losses, 1),
                "Op Risk Losses (€m)": round(s.op_risk_losses, 1),
                "Net Income (€m)": round(s.net_income, 1),
                "ROE (%)": round(s.roe * 100, 2),
                "HQLA (€m)": round(s.total_hqla, 1),
            })
        return summary

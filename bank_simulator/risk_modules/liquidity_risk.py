"""
Pillar B — Liquidity Risk (ILAAP)
==================================
Implements:
  - Liquidity Coverage Ratio (LCR) per Basel III
  - Net Stable Funding Ratio (NSFR) — simplified
  - Bank Run simulation (survival horizon in days)
  - Deposit run-off modelling with Basel III coefficients

References: Basel III LCR (BCBS 238), NSFR (BCBS 295), EBA RTS
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from bank_simulator.config import (
    InitialBalanceSheet,
    RUN_OFF_RATES,
    INFLOW_RATES,
    HQLA_HAIRCUTS,
    MacroScenario,
)


@dataclass
class LCRComponents:
    """Decomposition of the Liquidity Coverage Ratio."""
    hqla_l1: float
    hqla_l2a: float
    hqla_l2b: float
    total_hqla: float
    retail_stable_outflow: float
    retail_less_stable_outflow: float
    wholesale_operational_outflow: float
    wholesale_non_operational_outflow: float
    wholesale_unsecured_outflow: float
    total_outflows: float
    total_inflows: float
    net_cash_outflows: float
    lcr_ratio: float


@dataclass
class BankRunResult:
    """Output of a bank-run survival simulation."""
    survival_days: int
    daily_hqla: List[float]
    daily_cumulative_outflows: List[float]
    initial_hqla: float
    peak_daily_outflow: float
    scenario_name: str


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Liquidity Risk Engine
# ═══════════════════════════════════════════════════════════════════════════════

class LiquidityRiskEngine:
    """
    ILAAP-compliant liquidity risk module.
    
    Computes LCR, simulates bank runs, and provides deposit outflow
    projections for the balance sheet projector.
    """

    def __init__(self, bs: InitialBalanceSheet):
        self.bs = bs

    # ── LCR Calculation ──────────────────────────────────────────────────

    def compute_lcr(self, stress_multiplier: float = 1.0) -> LCRComponents:
        """
        Compute the Liquidity Coverage Ratio.
        
        LCR = HQLA / Net Cash Outflows (30 days)
        
        Parameters
        ----------
        stress_multiplier : amplifies run-off rates (>1 for stressed)
        """
        # ── HQLA (after haircuts) ────────────────────────────────────────
        hqla_l1 = (self.bs.cash_and_reserves +
                    self.bs.govt_bonds_hqla_l1 * (1 - HQLA_HAIRCUTS["level_1"]))
        hqla_l2a = self.bs.covered_bonds_hqla_l2a * (1 - HQLA_HAIRCUTS["level_2a"])
        hqla_l2b = self.bs.corporate_bonds_hqla_l2b * (1 - HQLA_HAIRCUTS["level_2b"])

        # L2 cap: max 40 % of total HQLA; L2B cap: max 15 %
        total_hqla_raw = hqla_l1 + hqla_l2a + hqla_l2b
        l2_cap = total_hqla_raw * 0.40
        l2b_cap = total_hqla_raw * 0.15
        hqla_l2b = min(hqla_l2b, l2b_cap)
        hqla_l2a = min(hqla_l2a, l2_cap - hqla_l2b)
        total_hqla = hqla_l1 + hqla_l2a + hqla_l2b

        # ── Cash Outflows (30-day horizon) ───────────────────────────────
        sm = stress_multiplier
        ret_stable_out = self.bs.retail_deposits_stable * RUN_OFF_RATES["retail_stable"] * sm
        ret_less_out = self.bs.retail_deposits_less_stable * RUN_OFF_RATES["retail_less_stable"] * sm
        ws_op_out = self.bs.wholesale_operational * RUN_OFF_RATES["wholesale_operational"] * sm
        ws_nop_out = self.bs.wholesale_non_operational * RUN_OFF_RATES["wholesale_non_operational"] * sm
        ws_unsec_out = self.bs.wholesale_unsecured * RUN_OFF_RATES["wholesale_unsecured"] * sm

        total_outflows = (ret_stable_out + ret_less_out +
                          ws_op_out + ws_nop_out + ws_unsec_out)

        # ── Cash Inflows (capped at 75% of outflows) ────────────────────
        loan_inflows = self.bs.total_loan_book * 0.02 * INFLOW_RATES["performing_loans"]
        total_inflows = min(loan_inflows, total_outflows * 0.75)

        net_outflows = max(total_outflows - total_inflows, total_outflows * 0.25)

        lcr = total_hqla / net_outflows if net_outflows > 0 else float('inf')

        return LCRComponents(
            hqla_l1=round(hqla_l1, 2),
            hqla_l2a=round(hqla_l2a, 2),
            hqla_l2b=round(hqla_l2b, 2),
            total_hqla=round(total_hqla, 2),
            retail_stable_outflow=round(ret_stable_out, 2),
            retail_less_stable_outflow=round(ret_less_out, 2),
            wholesale_operational_outflow=round(ws_op_out, 2),
            wholesale_non_operational_outflow=round(ws_nop_out, 2),
            wholesale_unsecured_outflow=round(ws_unsec_out, 2),
            total_outflows=round(total_outflows, 2),
            total_inflows=round(total_inflows, 2),
            net_cash_outflows=round(net_outflows, 2),
            lcr_ratio=round(lcr, 4),
        )

    # ── Bank Run Simulation ──────────────────────────────────────────────

    def simulate_bank_run(
        self,
        daily_retail_run_rate: float = 0.02,
        daily_wholesale_run_rate: float = 0.05,
        panic_acceleration: float = 1.05,
        max_days: int = 90,
        scenario_name: str = "Bank Run",
    ) -> BankRunResult:
        """
        Simulate a bank run and determine survival in days.
        
        Models accelerating deposit outflows until HQLA is depleted.
        
        Parameters
        ----------
        daily_retail_run_rate : daily % of retail deposits withdrawn
        daily_wholesale_run_rate : daily % of wholesale deposits withdrawn
        panic_acceleration : daily multiplier on run rates (panic effect)
        max_days : simulation cutoff
        scenario_name : label for the scenario
        
        Returns
        -------
        BankRunResult with survival metrics
        """
        hqla = self.bs.total_hqla
        initial_hqla = hqla

        retail_remaining = (self.bs.retail_deposits_stable +
                            self.bs.retail_deposits_less_stable)
        wholesale_remaining = (self.bs.wholesale_operational +
                               self.bs.wholesale_non_operational +
                               self.bs.wholesale_unsecured)

        daily_hqla = [hqla]
        daily_cum_outflows = [0.0]
        cumulative_outflow = 0.0
        peak_outflow = 0.0

        retail_rate = daily_retail_run_rate
        wholesale_rate = daily_wholesale_run_rate

        for day in range(1, max_days + 1):
            # Daily outflows
            retail_outflow = retail_remaining * retail_rate
            wholesale_outflow = wholesale_remaining * wholesale_rate

            daily_outflow = retail_outflow + wholesale_outflow

            # Reduce remaining deposits
            retail_remaining = max(retail_remaining - retail_outflow, 0)
            wholesale_remaining = max(wholesale_remaining - wholesale_outflow, 0)

            # HQLA absorbs outflows
            hqla -= daily_outflow
            cumulative_outflow += daily_outflow
            peak_outflow = max(peak_outflow, daily_outflow)

            daily_hqla.append(max(hqla, 0))
            daily_cum_outflows.append(cumulative_outflow)

            # Check survival
            if hqla <= 0:
                return BankRunResult(
                    survival_days=day,
                    daily_hqla=daily_hqla,
                    daily_cumulative_outflows=daily_cum_outflows,
                    initial_hqla=initial_hqla,
                    peak_daily_outflow=round(peak_outflow, 2),
                    scenario_name=scenario_name,
                )

            # Accelerate panic
            retail_rate *= panic_acceleration
            wholesale_rate *= panic_acceleration

        return BankRunResult(
            survival_days=max_days,
            daily_hqla=daily_hqla,
            daily_cumulative_outflows=daily_cum_outflows,
            initial_hqla=initial_hqla,
            peak_daily_outflow=round(peak_outflow, 2),
            scenario_name=scenario_name,
        )

    def simulate_multiple_runs(self) -> List[BankRunResult]:
        """Run bank-run simulations under different severity assumptions."""
        scenarios = [
            ("Moderate Run", 0.015, 0.03, 1.03),
            ("Severe Run", 0.025, 0.06, 1.06),
            ("Systemic Panic", 0.04, 0.10, 1.10),
        ]
        results = []
        for name, retail, wholesale, accel in scenarios:
            results.append(self.simulate_bank_run(
                daily_retail_run_rate=retail,
                daily_wholesale_run_rate=wholesale,
                panic_acceleration=accel,
                scenario_name=name,
            ))
        return results

    # ── Projection Helper ────────────────────────────────────────────────

    def compute_yearly_deposit_outflows(
        self, scenario: MacroScenario
    ) -> List[float]:
        """
        Estimate annual deposit erosion under a macro scenario.
        
        Higher unemployment and rate hikes trigger deposit flight.
        
        Returns
        -------
        List of annual deposit outflows (€m) for each projection year.
        """
        outflows = []
        base_deposits = (self.bs.retail_deposits_stable +
                         self.bs.retail_deposits_less_stable +
                         self.bs.wholesale_operational +
                         self.bs.wholesale_non_operational +
                         self.bs.wholesale_unsecured)

        for yr in range(len(scenario.gdp_growth)):
            # Stress intensity: driven by GDP contraction and rate shocks
            gdp_stress = max(-scenario.gdp_growth[yr] / 100.0, 0)
            rate_stress = abs(scenario.interest_rate_shock_bps[yr]) / 1000.0
            unemployment_stress = max(scenario.unemployment_rate[yr] - 7.0, 0) / 100.0

            outflow_pct = (gdp_stress * 0.02 + rate_stress * 0.01 +
                           unemployment_stress * 0.005)
            outflows.append(base_deposits * outflow_pct)

        return outflows

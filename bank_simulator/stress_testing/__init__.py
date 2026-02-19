"""
Monte Carlo Stress Testing Engine
==================================
Implements:
  - Full-bank Monte Carlo simulation across all risk pillars
  - Scenario-based macro variable generation (correlated draws)
  - Multi-year projection with stochastic risk factor overlay
  - Distribution of CET1, LCR, RAROC at horizon
  - Probability of breaching regulatory thresholds

Orchestrates: MarketRisk × CreditRisk × LiquidityRisk × OpRisk
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bank_simulator.config import (
    InitialBalanceSheet,
    MacroScenario,
    ALL_SCENARIOS,
    PROJECTION_HORIZON_YEARS,
    MIN_CET1_RATIO,
    MIN_LCR_RATIO,
)
from bank_simulator.engine.balance_sheet import BalanceSheetProjector
from bank_simulator.risk_modules.market_risk import MarketRiskALM
from bank_simulator.risk_modules.liquidity_risk import LiquidityRiskEngine
from bank_simulator.risk_modules.credit_risk import CreditRiskEngine
from bank_simulator.risk_modules.operational_risk import OperationalRiskEngine


@dataclass
class StressTestPath:
    """Single Monte Carlo path result."""
    path_id: int
    scenario_name: str
    yearly_cet1_ratio: List[float]
    yearly_lcr: List[float]
    yearly_raroc: List[float]
    yearly_net_income: List[float]
    terminal_cet1_ratio: float
    terminal_lcr: float
    cet1_breach: bool
    lcr_breach: bool


@dataclass
class StressTestResult:
    """Aggregate results from the Monte Carlo stress test."""
    scenario_name: str
    n_paths: int
    n_years: int

    # Terminal distributions
    cet1_mean: float
    cet1_std: float
    cet1_p5: float
    cet1_p50: float
    cet1_p95: float
    cet1_breach_probability: float

    lcr_mean: float
    lcr_std: float
    lcr_p5: float
    lcr_p50: float
    lcr_p95: float
    lcr_breach_probability: float

    raroc_mean: float
    raroc_std: float
    raroc_p5: float
    raroc_p50: float
    raroc_p95: float

    net_income_mean: float
    net_income_std: float

    # Year-by-year averages
    yearly_cet1_mean: List[float]
    yearly_lcr_mean: List[float]
    yearly_net_income_mean: List[float]

    # All paths for distribution plots
    all_terminal_cet1: List[float] = field(default_factory=list)
    all_terminal_lcr: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stochastic Macro Variable Generator
# ═══════════════════════════════════════════════════════════════════════════════

class MacroVariableGenerator:
    """
    Generates correlated macro variables around a base scenario
    using Cholesky decomposition of the correlation matrix.
    """

    # Correlation matrix between GDP, rates, unemployment, HPI
    CORRELATION = np.array([
        [ 1.00, -0.30,  -0.70,  0.60],  # GDP
        [-0.30,  1.00,   0.20, -0.40],  # Interest rates
        [-0.70,  0.20,   1.00, -0.50],  # Unemployment
        [ 0.60, -0.40,  -0.50,  1.00],  # HPI
    ])

    # Volatility of shocks around scenario central path
    VOLATILITIES = np.array([0.015, 50.0, 0.01, 0.05])  # GDP%, bps, unemp%, hpi%

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.chol = np.linalg.cholesky(self.CORRELATION)

    def generate_paths(
        self,
        scenario: MacroScenario,
        n_paths: int,
        n_years: int = PROJECTION_HORIZON_YEARS,
    ) -> Dict[str, np.ndarray]:
        """
        Generate n_paths of correlated macro variables.
        
        Returns
        -------
        Dict with keys 'gdp', 'rates', 'unemployment', 'hpi', 'pd_multiplier'
        Each value is (n_paths, n_years) array.
        """
        result = {
            "gdp": np.zeros((n_paths, n_years)),
            "rates": np.zeros((n_paths, n_years)),
            "unemployment": np.zeros((n_paths, n_years)),
            "hpi": np.zeros((n_paths, n_years)),
            "pd_multiplier": np.zeros((n_paths, n_years)),
        }

        for yr in range(n_years):
            # Correlated standard normals
            z = self.rng.standard_normal((n_paths, 4))
            correlated = z @ self.chol.T

            # Add shocks to scenario central path
            result["gdp"][:, yr] = (scenario.gdp_growth[yr] +
                                     correlated[:, 0] * self.VOLATILITIES[0] * 100)
            result["rates"][:, yr] = (scenario.interest_rate_shock_bps[yr] +
                                       correlated[:, 1] * self.VOLATILITIES[1])
            result["unemployment"][:, yr] = (scenario.unemployment_rate[yr] +
                                              correlated[:, 2] * self.VOLATILITIES[2] * 100)
            result["hpi"][:, yr] = (scenario.house_price_change[yr] +
                                     correlated[:, 3] * self.VOLATILITIES[3] * 100)

            # PD multiplier: linked to GDP shock (inverse) and unemployment
            base_mult = scenario.pd_multiplier[yr]
            gdp_shock = (scenario.gdp_growth[yr] - result["gdp"][:, yr]) / 100.0
            result["pd_multiplier"][:, yr] = np.maximum(
                base_mult * (1 + gdp_shock * 2), 0.5
            )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo Stress Test Engine
# ═══════════════════════════════════════════════════════════════════════════════

class MonteCarloStressEngine:
    """
    Full-bank Monte Carlo stress test integrating all risk pillars.
    
    For each simulation path:
      1. Generate stochastic macro variables
      2. Compute credit losses (stressed PD)
      3. Compute market risk P&L (rate shocks)
      4. Compute op risk losses (Poisson-Lognormal)
      5. Compute NII sensitivity
      6. Project balance sheet forward
      7. Record terminal CET1, LCR, RAROC
    """

    def __init__(
        self,
        bs: Optional[InitialBalanceSheet] = None,
        seed: int = 42,
    ):
        self.bs = bs or InitialBalanceSheet()
        self.seed = seed
        self.macro_gen = MacroVariableGenerator(seed=seed)

        # Initialise risk engines
        self.market_engine = MarketRiskALM(self.bs)
        self.liquidity_engine = LiquidityRiskEngine(self.bs)
        self.credit_engine = CreditRiskEngine(self.bs)
        self.op_risk_engine = OperationalRiskEngine(seed=seed)

    def run(
        self,
        scenario: MacroScenario,
        n_paths: int = 1_000,
        n_years: int = PROJECTION_HORIZON_YEARS,
    ) -> StressTestResult:
        """
        Execute the Monte Carlo stress test for a given scenario.
        
        Parameters
        ----------
        scenario : base macro scenario (with stochastic overlay)
        n_paths : number of simulation paths
        n_years : projection horizon
        """
        # Generate macro paths
        macro = self.macro_gen.generate_paths(scenario, n_paths, n_years)

        # Pre-simulate op risk losses (n_paths × n_years)
        op_losses = self.op_risk_engine.simulate_yearly_losses(n_years, n_paths)

        # Storage
        all_cet1 = np.zeros((n_paths, n_years + 1))  # Include t=0
        all_lcr = np.zeros((n_paths, n_years))
        all_raroc = np.zeros((n_paths, n_years))
        all_ni = np.zeros((n_paths, n_years))

        for path in range(n_paths):
            # Build path-specific scenario
            path_scenario = MacroScenario(
                name=f"{scenario.name}_path_{path}",
                gdp_growth=[float(macro["gdp"][path, yr]) for yr in range(n_years)],
                interest_rate_shock_bps=[float(macro["rates"][path, yr]) for yr in range(n_years)],
                unemployment_rate=[float(macro["unemployment"][path, yr]) for yr in range(n_years)],
                house_price_change=[float(macro["hpi"][path, yr]) for yr in range(n_years)],
                pd_multiplier=[float(macro["pd_multiplier"][path, yr]) for yr in range(n_years)],
            )

            # Credit losses
            credit_losses = self.credit_engine.compute_yearly_credit_losses(path_scenario)

            # Market risk
            nii_adj, mkt_pnl = self.market_engine.compute_yearly_impacts(path_scenario)

            # Op risk losses for this path
            oprisk = [float(op_losses[path, yr]) for yr in range(n_years)]

            # Deposit outflows
            dep_outflows = self.liquidity_engine.compute_yearly_deposit_outflows(path_scenario)

            # Project balance sheet
            projector = BalanceSheetProjector(self.bs)
            snapshots = projector.project(
                scenario=path_scenario,
                credit_losses_by_year=credit_losses,
                oprisk_losses_by_year=oprisk,
                market_pnl_by_year=mkt_pnl,
                nii_adjustment_by_year=nii_adj,
                deposit_outflow_by_year=dep_outflows,
            )

            # Record metrics
            for yr_idx, snap in enumerate(snapshots):
                all_cet1[path, yr_idx] = snap.cet1_ratio
                if yr_idx > 0:
                    all_ni[path, yr_idx - 1] = snap.net_income
                    # Simplified LCR = HQLA / (deposits * avg_run_off_rate)
                    total_dep = (snap.retail_deposits_stable +
                                 snap.retail_deposits_less_stable +
                                 snap.wholesale_operational +
                                 snap.wholesale_non_operational +
                                 snap.wholesale_unsecured)
                    net_outflow = total_dep * 0.15
                    all_lcr[path, yr_idx - 1] = (
                        snap.total_hqla / net_outflow if net_outflow > 0 else 999
                    )
                    # RAROC
                    eco_cap = snap.total_rwa * MIN_CET1_RATIO
                    all_raroc[path, yr_idx - 1] = (
                        snap.net_income / eco_cap if eco_cap > 0 else 0
                    )

        # ── Aggregate Statistics ─────────────────────────────────────────
        terminal_cet1 = all_cet1[:, -1]
        terminal_lcr = all_lcr[:, -1]
        terminal_raroc = all_raroc[:, -1]
        terminal_ni = all_ni[:, -1]

        return StressTestResult(
            scenario_name=scenario.name,
            n_paths=n_paths,
            n_years=n_years,
            # CET1
            cet1_mean=round(float(np.mean(terminal_cet1)), 4),
            cet1_std=round(float(np.std(terminal_cet1)), 4),
            cet1_p5=round(float(np.percentile(terminal_cet1, 5)), 4),
            cet1_p50=round(float(np.percentile(terminal_cet1, 50)), 4),
            cet1_p95=round(float(np.percentile(terminal_cet1, 95)), 4),
            cet1_breach_probability=round(
                float(np.mean(terminal_cet1 < MIN_CET1_RATIO)), 4
            ),
            # LCR
            lcr_mean=round(float(np.mean(terminal_lcr)), 4),
            lcr_std=round(float(np.std(terminal_lcr)), 4),
            lcr_p5=round(float(np.percentile(terminal_lcr, 5)), 4),
            lcr_p50=round(float(np.percentile(terminal_lcr, 50)), 4),
            lcr_p95=round(float(np.percentile(terminal_lcr, 95)), 4),
            lcr_breach_probability=round(
                float(np.mean(terminal_lcr < MIN_LCR_RATIO)), 4
            ),
            # RAROC
            raroc_mean=round(float(np.mean(terminal_raroc)), 4),
            raroc_std=round(float(np.std(terminal_raroc)), 4),
            raroc_p5=round(float(np.percentile(terminal_raroc, 5)), 4),
            raroc_p50=round(float(np.percentile(terminal_raroc, 50)), 4),
            raroc_p95=round(float(np.percentile(terminal_raroc, 95)), 4),
            # Net Income
            net_income_mean=round(float(np.mean(terminal_ni)), 2),
            net_income_std=round(float(np.std(terminal_ni)), 2),
            # Yearly means
            yearly_cet1_mean=[round(float(np.mean(all_cet1[:, yr])), 4)
                              for yr in range(n_years + 1)],
            yearly_lcr_mean=[round(float(np.mean(all_lcr[:, yr])), 4)
                             for yr in range(n_years)],
            yearly_net_income_mean=[round(float(np.mean(all_ni[:, yr])), 2)
                                   for yr in range(n_years)],
            # Distributions
            all_terminal_cet1=terminal_cet1.tolist(),
            all_terminal_lcr=terminal_lcr.tolist(),
        )

    def run_all_scenarios(
        self,
        n_paths: int = 1_000,
        scenarios: Optional[List[MacroScenario]] = None,
    ) -> List[StressTestResult]:
        """Run stress tests across all predefined scenarios."""
        scenarios = scenarios or ALL_SCENARIOS
        return [self.run(s, n_paths) for s in scenarios]


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience function
# ═══════════════════════════════════════════════════════════════════════════════

def run_stress_test(
    n_paths: int = 1_000,
    bs: Optional[InitialBalanceSheet] = None,
) -> List[StressTestResult]:
    """Run the full Monte Carlo stress test suite."""
    engine = MonteCarloStressEngine(bs=bs)
    return engine.run_all_scenarios(n_paths=n_paths)

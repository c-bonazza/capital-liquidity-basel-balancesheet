"""
Capital Optimization — RAROC Maximiser
=======================================
Implements:
  - Risk-Adjusted Return on Capital (RAROC) computation
  - Constrained optimisation of asset allocation using scipy.optimize
  - Constraints: CET1 ratio ≥ 12 %, LCR ≥ 110 %, sum of weights = 1
  - Objective: maximise portfolio RAROC

RAROC = (Expected Revenue − Operating Costs − Expected Loss) / Economic Capital

References: RAPM frameworks, McKinsey bank capital management
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from bank_simulator.config import (
    InitialBalanceSheet,
    DEFAULT_CREDIT_SEGMENTS,
    MIN_CET1_RATIO,
    MIN_LCR_RATIO,
    CONFIDENCE_LEVEL,
)


# ─── Asset Segment Characteristics ──────────────────────────────────────────
# Each lending segment has: (expected_yield, risk_weight, pd_ttc, lgd, op_cost_rate)
SEGMENT_PROFILES = {
    "Residential Mortgages": {
        "yield": 0.025,      # 2.5 % gross yield
        "risk_weight": 0.35,
        "pd": 0.008,
        "lgd": 0.15,
        "op_cost_ratio": 0.004,
        "hqla_contribution": 0.0,   # loans don't count as HQLA
        "run_off_factor": 0.0,      # not a liability
        "deposit_funding_need": 0.90,  # % funded by deposits
    },
    "SME Loans": {
        "yield": 0.045,
        "risk_weight": 0.75,
        "pd": 0.025,
        "lgd": 0.40,
        "op_cost_ratio": 0.008,
        "hqla_contribution": 0.0,
        "run_off_factor": 0.0,
        "deposit_funding_need": 0.85,
    },
    "Consumer Credit": {
        "yield": 0.065,
        "risk_weight": 0.75,
        "pd": 0.035,
        "lgd": 0.55,
        "op_cost_ratio": 0.012,
        "hqla_contribution": 0.0,
        "run_off_factor": 0.0,
        "deposit_funding_need": 0.80,
    },
    "Corporate Loans": {
        "yield": 0.035,
        "risk_weight": 0.65,
        "pd": 0.012,
        "lgd": 0.35,
        "op_cost_ratio": 0.005,
        "hqla_contribution": 0.0,
        "run_off_factor": 0.0,
        "deposit_funding_need": 0.88,
    },
    "HQLA (Govt Bonds)": {
        "yield": 0.010,
        "risk_weight": 0.00,
        "pd": 0.0001,
        "lgd": 0.01,
        "op_cost_ratio": 0.001,
        "hqla_contribution": 1.0,
        "run_off_factor": 0.0,
        "deposit_funding_need": 0.50,
    },
    "Trading Book": {
        "yield": 0.040,
        "risk_weight": 1.00,
        "pd": 0.005,
        "lgd": 0.30,
        "op_cost_ratio": 0.010,
        "hqla_contribution": 0.0,
        "run_off_factor": 0.0,
        "deposit_funding_need": 0.70,
    },
}

SEGMENT_NAMES = list(SEGMENT_PROFILES.keys())
N_SEGMENTS = len(SEGMENT_NAMES)


@dataclass
class RAROCResult:
    """Output of the RAROC computation for a given allocation."""
    segment_rarocs: Dict[str, float]
    portfolio_raroc: float
    total_revenue: float
    total_op_costs: float
    total_expected_loss: float
    economic_capital: float
    total_rwa: float
    cet1_ratio: float
    lcr_estimate: float
    weights: Dict[str, float]
    feasible: bool


@dataclass
class OptimizationResult:
    """Output of the capital optimisation run."""
    optimal_weights: Dict[str, float]
    optimal_raroc: float
    optimal_cet1: float
    optimal_lcr: float
    initial_raroc: float
    initial_weights: Dict[str, float]
    convergence_message: str
    success: bool
    sensitivity: Dict[str, List[Dict]]  # sensitivity analysis results


# ═══════════════════════════════════════════════════════════════════════════════
#  RAROC Calculator
# ═══════════════════════════════════════════════════════════════════════════════

class RAROCCalculator:
    """
    Computes Risk-Adjusted Return on Capital for a given asset allocation.
    
    RAROC = (Revenue − OpCosts − EL) / Economic Capital
    
    where Economic Capital = CET1 allocated = CET1_ratio_target × RWA
    """

    def __init__(self, bs: InitialBalanceSheet):
        self.bs = bs
        self.total_assets = bs.total_assets
        self.cet1 = bs.cet1_capital

    def compute(self, weights: np.ndarray) -> RAROCResult:
        """
        Compute RAROC for a given set of asset allocation weights.
        
        Parameters
        ----------
        weights : (N_SEGMENTS,) array, each ∈ [0,1], sum = 1
                  Represents share of total assets in each segment.
        """
        segment_rarocs = {}
        total_revenue = 0.0
        total_op_costs = 0.0
        total_el = 0.0
        total_rwa = 0.0
        total_hqla = 0.0
        total_deposit_funding = 0.0

        for i, name in enumerate(SEGMENT_NAMES):
            p = SEGMENT_PROFILES[name]
            alloc = weights[i] * self.total_assets

            revenue = alloc * p["yield"]
            op_cost = alloc * p["op_cost_ratio"]
            el = alloc * p["pd"] * p["lgd"]
            rwa = alloc * p["risk_weight"]
            hqla_contrib = alloc * p["hqla_contribution"]
            dep_need = alloc * p["deposit_funding_need"]

            # Segment RAROC
            eco_cap = rwa * MIN_CET1_RATIO  # Capital allocated to this segment
            seg_raroc = (revenue - op_cost - el) / eco_cap if eco_cap > 0 else 0.0

            segment_rarocs[name] = round(seg_raroc, 4)
            total_revenue += revenue
            total_op_costs += op_cost
            total_el += el
            total_rwa += rwa
            total_hqla += hqla_contrib
            total_deposit_funding += dep_need

        # Portfolio-level RAROC
        economic_capital = total_rwa * MIN_CET1_RATIO
        portfolio_raroc = ((total_revenue - total_op_costs - total_el) /
                           economic_capital if economic_capital > 0 else 0.0)

        # CET1 ratio
        cet1_ratio = self.cet1 / total_rwa if total_rwa > 0 else float('inf')

        # Simplified LCR estimate
        total_deposits = (self.bs.retail_deposits_stable +
                          self.bs.retail_deposits_less_stable +
                          self.bs.wholesale_operational +
                          self.bs.wholesale_non_operational +
                          self.bs.wholesale_unsecured)
        net_outflows = total_deposits * 0.15  # Simplified weighted average
        lcr = total_hqla / net_outflows if net_outflows > 0 else float('inf')

        feasible = (cet1_ratio >= MIN_CET1_RATIO and lcr >= MIN_LCR_RATIO)

        return RAROCResult(
            segment_rarocs=segment_rarocs,
            portfolio_raroc=round(portfolio_raroc, 4),
            total_revenue=round(total_revenue, 2),
            total_op_costs=round(total_op_costs, 2),
            total_expected_loss=round(total_el, 2),
            economic_capital=round(economic_capital, 2),
            total_rwa=round(total_rwa, 2),
            cet1_ratio=round(cet1_ratio, 4),
            lcr_estimate=round(lcr, 4),
            weights={name: round(float(w), 4) for name, w in zip(SEGMENT_NAMES, weights)},
            feasible=feasible,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Capital Optimiser (scipy.optimize)
# ═══════════════════════════════════════════════════════════════════════════════

class CapitalOptimizer:
    """
    Finds the optimal asset allocation that maximises RAROC subject
    to regulatory constraints:
    
        max   RAROC(w)
        s.t.  CET1_ratio(w)  ≥  12 %
              LCR(w)          ≥  110 %
              Σ w_i           =  1
              w_i             ≥  lower_bound_i   (business constraints)
              w_i             ≤  upper_bound_i
    """

    def __init__(self, bs: InitialBalanceSheet):
        self.bs = bs
        self.calculator = RAROCCalculator(bs)

    def _objective(self, weights: np.ndarray) -> float:
        """Negative RAROC (minimise negative = maximise RAROC)."""
        result = self.calculator.compute(weights)
        return -result.portfolio_raroc

    def _cet1_constraint(self, weights: np.ndarray) -> float:
        """CET1 ratio ≥ MIN_CET1_RATIO (inequality: result ≥ 0)."""
        result = self.calculator.compute(weights)
        return result.cet1_ratio - MIN_CET1_RATIO

    def _lcr_constraint(self, weights: np.ndarray) -> float:
        """LCR ≥ MIN_LCR_RATIO."""
        result = self.calculator.compute(weights)
        return result.lcr_estimate - MIN_LCR_RATIO

    def optimize(
        self,
        initial_weights: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """
        Run the constrained optimisation.
        
        Parameters
        ----------
        initial_weights : starting allocation (default: current)
        bounds : (min, max) share for each segment
        """
        # Default initial weights (approximate current allocation)
        if initial_weights is None:
            ta = self.bs.total_assets
            initial_weights = np.array([
                self.bs.residential_mortgages / ta,   # Mortgages
                self.bs.sme_loans / ta,               # SME
                self.bs.consumer_credit / ta,          # Consumer
                self.bs.corporate_loans / ta,          # Corporate
                (self.bs.govt_bonds_hqla_l1 + self.bs.covered_bonds_hqla_l2a +
                 self.bs.corporate_bonds_hqla_l2b + self.bs.cash_and_reserves) / ta,  # HQLA
                self.bs.trading_book / ta,             # Trading
            ])
            # Normalise
            initial_weights = initial_weights / initial_weights.sum()

        # Default bounds
        if bounds is None:
            bounds = [
                (0.10, 0.50),  # Mortgages
                (0.05, 0.30),  # SME
                (0.02, 0.20),  # Consumer
                (0.05, 0.25),  # Corporate
                (0.08, 0.30),  # HQLA (minimum liquidity buffer)
                (0.00, 0.15),  # Trading
            ]

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},      # Weights sum to 1
            {"type": "ineq", "fun": self._cet1_constraint},         # CET1 ≥ 12%
            {"type": "ineq", "fun": self._lcr_constraint},          # LCR ≥ 110%
        ]

        # Initial RAROC
        initial_result = self.calculator.compute(initial_weights)

        # Optimise
        opt_result: OptimizeResult = minimize(
            self._objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )

        optimal_weights = opt_result.x
        optimal_raroc_result = self.calculator.compute(optimal_weights)

        # Sensitivity analysis: vary CET1 target
        sensitivity = self._run_sensitivity_analysis(optimal_weights, bounds)

        return OptimizationResult(
            optimal_weights={name: round(float(w), 4)
                             for name, w in zip(SEGMENT_NAMES, optimal_weights)},
            optimal_raroc=optimal_raroc_result.portfolio_raroc,
            optimal_cet1=optimal_raroc_result.cet1_ratio,
            optimal_lcr=optimal_raroc_result.lcr_estimate,
            initial_raroc=initial_result.portfolio_raroc,
            initial_weights={name: round(float(w), 4)
                             for name, w in zip(SEGMENT_NAMES, initial_weights)},
            convergence_message=opt_result.message,
            success=opt_result.success,
            sensitivity=sensitivity,
        )

    def _run_sensitivity_analysis(
        self,
        base_weights: np.ndarray,
        bounds: List[Tuple[float, float]],
    ) -> Dict[str, List[Dict]]:
        """
        Sensitivity analysis: how RAROC changes with different
        CET1 / LCR targets.
        """
        cet1_targets = [0.08, 0.10, 0.12, 0.14, 0.16]
        sens_results = []

        for target in cet1_targets:
            # Recompute with modified target
            old_target = MIN_CET1_RATIO
            result = self.calculator.compute(base_weights)
            # Approximate: RAROC scales inversely with capital target
            adjusted_raroc = (result.portfolio_raroc *
                              (old_target / target) if target > 0 else 0)
            sens_results.append({
                "CET1 Target (%)": f"{target:.0%}",
                "RAROC": round(adjusted_raroc, 4),
                "Feasible": result.cet1_ratio >= target,
            })

        return {"cet1_sensitivity": sens_results}


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience function
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_optimization(bs: Optional[InitialBalanceSheet] = None) -> OptimizationResult:
    """Run the full RAROC optimisation with default settings."""
    bs = bs or InitialBalanceSheet()
    optimizer = CapitalOptimizer(bs)
    return optimizer.optimize()

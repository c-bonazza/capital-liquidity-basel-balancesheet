"""
Pillar A — Market Risk & ALM (IRRBB)
=====================================
Implements:
  - Repricing gap analysis (sensitivity gaps by maturity bucket)
  - Modified Duration of equity
  - Economic Value of Equity (EVE) under parallel rate shocks
  - NII sensitivity under rate scenarios

References: Basel IRRBB Standards (BCBS 368), EBA Guidelines 2022
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from bank_simulator.config import MacroScenario, InitialBalanceSheet


# ─── Maturity Buckets ────────────────────────────────────────────────────────
BUCKET_MIDPOINTS_YEARS = {
    "< 3M":   0.125,
    "3M-6M":  0.375,
    "6M-1Y":  0.75,
    "1Y-2Y":  1.5,
    "2Y-5Y":  3.5,
    "5Y-10Y": 7.5,
    "10Y+":   15.0,
}


@dataclass
class GapBucket:
    """Repricing gap for a single maturity bucket."""
    bucket: str
    midpoint_years: float
    rate_sensitive_assets: float = 0.0
    rate_sensitive_liabilities: float = 0.0

    @property
    def gap(self) -> float:
        """Positive gap = asset-sensitive; negative = liability-sensitive."""
        return self.rate_sensitive_assets - self.rate_sensitive_liabilities

    @property
    def duration_weighted_gap(self) -> float:
        return self.gap * self.midpoint_years


# ═══════════════════════════════════════════════════════════════════════════════
#  Main ALM Engine
# ═══════════════════════════════════════════════════════════════════════════════

class MarketRiskALM:
    """
    Interest Rate Risk in the Banking Book (IRRBB) engine.
    
    Supports:
    - Repricing gap construction from the initial balance sheet
    - Modified Duration computation for EVE sensitivity
    - Δ-EVE and Δ-NII under parallel & non-parallel shocks
    """

    def __init__(self, bs: InitialBalanceSheet):
        self.bs = bs
        self.gap_schedule: List[GapBucket] = []
        self._build_gap_schedule()

    # ── Gap Schedule Construction ────────────────────────────────────────

    def _build_gap_schedule(self):
        """
        Allocate assets and liabilities into repricing buckets.
        Uses simplified heuristic allocation based on average maturities.
        """
        self.gap_schedule = []

        # Asset allocation to buckets (€m)
        asset_alloc = {
            "< 3M":   self.bs.cash_and_reserves + self.bs.trading_book * 0.6,
            "3M-6M":  self.bs.consumer_credit * 0.3 + self.bs.trading_book * 0.4,
            "6M-1Y":  self.bs.consumer_credit * 0.4 + self.bs.corporate_loans * 0.15,
            "1Y-2Y":  self.bs.sme_loans * 0.25 + self.bs.corporate_loans * 0.30,
            "2Y-5Y":  (self.bs.sme_loans * 0.50 + self.bs.corporate_loans * 0.40 +
                       self.bs.covered_bonds_hqla_l2a * 0.5 +
                       self.bs.corporate_bonds_hqla_l2b * 0.6),
            "5Y-10Y": (self.bs.residential_mortgages * 0.30 +
                       self.bs.govt_bonds_hqla_l1 * 0.40 +
                       self.bs.covered_bonds_hqla_l2a * 0.5),
            "10Y+":   (self.bs.residential_mortgages * 0.70 +
                       self.bs.consumer_credit * 0.3 +
                       self.bs.sme_loans * 0.25 +
                       self.bs.corporate_loans * 0.15 +
                       self.bs.govt_bonds_hqla_l1 * 0.60 +
                       self.bs.corporate_bonds_hqla_l2b * 0.4 +
                       self.bs.other_assets),
        }

        # Liability allocation to buckets
        liability_alloc = {
            "< 3M":   (self.bs.retail_deposits_stable * 0.10 +
                       self.bs.retail_deposits_less_stable * 0.40 +
                       self.bs.wholesale_non_operational * 0.50 +
                       self.bs.wholesale_unsecured * 0.60),
            "3M-6M":  (self.bs.retail_deposits_less_stable * 0.30 +
                       self.bs.wholesale_operational * 0.25 +
                       self.bs.wholesale_non_operational * 0.30),
            "6M-1Y":  (self.bs.retail_deposits_less_stable * 0.20 +
                       self.bs.wholesale_operational * 0.25 +
                       self.bs.wholesale_unsecured * 0.25),
            "1Y-2Y":  (self.bs.retail_deposits_stable * 0.15 +
                       self.bs.wholesale_operational * 0.25 +
                       self.bs.wholesale_non_operational * 0.20),
            "2Y-5Y":  (self.bs.retail_deposits_stable * 0.30 +
                       self.bs.retail_deposits_less_stable * 0.10 +
                       self.bs.wholesale_operational * 0.25 +
                       self.bs.subordinated_debt * 0.30),
            "5Y-10Y": (self.bs.retail_deposits_stable * 0.25 +
                       self.bs.subordinated_debt * 0.50 +
                       self.bs.wholesale_unsecured * 0.15 +
                       self.bs.other_liabilities * 0.50),
            "10Y+":   (self.bs.retail_deposits_stable * 0.20 +
                       self.bs.subordinated_debt * 0.20 +
                       self.bs.other_liabilities * 0.50),
        }

        for bucket, midpoint in BUCKET_MIDPOINTS_YEARS.items():
            gb = GapBucket(
                bucket=bucket,
                midpoint_years=midpoint,
                rate_sensitive_assets=asset_alloc.get(bucket, 0.0),
                rate_sensitive_liabilities=liability_alloc.get(bucket, 0.0),
            )
            self.gap_schedule.append(gb)

    # ── Core Analytics ───────────────────────────────────────────────────

    def get_gap_table(self) -> List[Dict]:
        """Return the repricing gap schedule as a list of dicts."""
        result = []
        cumulative = 0.0
        for gb in self.gap_schedule:
            cumulative += gb.gap
            result.append({
                "Bucket": gb.bucket,
                "Midpoint (yrs)": gb.midpoint_years,
                "RSA (€m)": round(gb.rate_sensitive_assets, 1),
                "RSL (€m)": round(gb.rate_sensitive_liabilities, 1),
                "Gap (€m)": round(gb.gap, 1),
                "Cumulative Gap (€m)": round(cumulative, 1),
                "Duration-Weighted Gap (€m·yr)": round(gb.duration_weighted_gap, 1),
            })
        return result

    def compute_modified_duration_equity(self, yield_level: float = 0.03) -> float:
        """
        Compute the Modified Duration of Equity (proxy).
        
        Uses duration-weighted gap approach:
            D_equity ≈ Σ(gap_i × midpoint_i) / Equity
        
        Parameters
        ----------
        yield_level : current yield for Macaulay-to-Modified conversion
        """
        dw_gap_total = sum(gb.duration_weighted_gap for gb in self.gap_schedule)
        equity = self.bs.cet1_capital + self.bs.at1_capital + self.bs.tier2_capital
        if equity <= 0:
            return 0.0
        mac_duration = dw_gap_total / equity
        modified_duration = mac_duration / (1 + yield_level)
        return modified_duration

    def compute_eve_delta(self, shock_bps: float) -> float:
        """
        Compute the change in Economic Value of Equity for a parallel
        rate shock.
        
        Δ-EVE ≈ -Modified_Duration × Equity × ΔRate
        
        Parameters
        ----------
        shock_bps : parallel rate shock in basis points (+200 = +2%)
        
        Returns
        -------
        Δ-EVE in €m (negative = loss of value)
        """
        mod_dur = self.compute_modified_duration_equity()
        equity = self.bs.cet1_capital + self.bs.at1_capital + self.bs.tier2_capital
        delta_rate = shock_bps / 10_000.0
        delta_eve = -mod_dur * equity * delta_rate
        return delta_eve

    def compute_nii_sensitivity(self, shock_bps: float, horizon_years: float = 1.0) -> float:
        """
        Estimate the impact on Net Interest Income from a rateshock.
        
        Only repricing within the horizon generates NII impact:
            Δ-NII ≈ Σ(gap_i × Δrate)  for buckets with midpoint ≤ horizon
        
        Parameters
        ----------
        shock_bps : parallel rate shock in basis points
        horizon_years : NII sensitivity horizon (default 1 year)
        """
        delta_rate = shock_bps / 10_000.0
        delta_nii = 0.0
        for gb in self.gap_schedule:
            if gb.midpoint_years <= horizon_years:
                delta_nii += gb.gap * delta_rate
            else:
                # Partial repricing for the bucket straddling the horizon
                overlap = max(0, horizon_years - (gb.midpoint_years - 0.5))
                if overlap > 0:
                    delta_nii += gb.gap * delta_rate * (overlap / 1.0)
        return delta_nii

    def compute_eve_six_scenarios(self) -> Dict[str, float]:
        """
        Basel IRRBB standard six supervisory scenarios for EVE.
        Returns dict of scenario_name → Δ-EVE.
        """
        scenarios = {
            "Parallel Up +200bp": 200,
            "Parallel Down -200bp": -200,
            "Short Up +300bp": 100,      # Simplified: use weighted average
            "Short Down -300bp": -100,
            "Steepener": 150,
            "Flattener": -50,
        }
        return {name: round(self.compute_eve_delta(shock), 2)
                for name, shock in scenarios.items()}

    def compute_yearly_impacts(
        self, scenario: MacroScenario
    ) -> Tuple[List[float], List[float]]:
        """
        Compute year-by-year NII adjustments and EVE P&L impacts.
        
        Returns
        -------
        (nii_adjustments, market_pnl) : each a list of length = projection years
        """
        nii_adj = []
        mkt_pnl = []
        cumulative_shock = 0.0
        for yr, shock in enumerate(scenario.interest_rate_shock_bps):
            incremental = shock - cumulative_shock
            nii_adj.append(self.compute_nii_sensitivity(incremental))
            mkt_pnl.append(self.compute_eve_delta(incremental) * 0.1)  # Amortised
            cumulative_shock = shock
        return nii_adj, mkt_pnl

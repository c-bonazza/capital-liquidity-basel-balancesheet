"""
Pillar C — Credit Risk (IFRS 9 & RWA)
======================================
Implements:
  - Expected Credit Loss (ECL) by IFRS 9 stage (1, 2, 3)
  - Rating Migration via Transition Matrices
  - Risk-Weighted Assets under Standardised Approach
  - Through-the-cycle → Point-in-time PD conversion
  - Macro-conditioned PD stress (TTC × scenario multiplier)

References: IFRS 9, Basel III/IV SA-CR, EBA GL/2017/06
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from bank_simulator.config import (
    CreditSegmentParams,
    DEFAULT_CREDIT_SEGMENTS,
    DEFAULT_TRANSITION_MATRIX,
    RATING_GRADES,
    MacroScenario,
    InitialBalanceSheet,
)


@dataclass
class ECLResult:
    """ECL computation result for a credit segment."""
    segment_name: str
    ead: float
    pd_12m: float
    lifetime_pd: float
    lgd: float
    ecl_stage1: float      # 12-month ECL
    ecl_stage2: float      # Lifetime ECL
    ecl_stage3: float      # LGD × EAD (defaulted)
    total_ecl: float
    rwa: float
    stage_distribution: Dict[int, float]  # stage → share of EAD


# ═══════════════════════════════════════════════════════════════════════════════
#  Transition Matrix Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TransitionMatrixEngine:
    """
    Applies multi-period rating migration using Markov chain transition
    matrices. Can stress the matrix under adverse macroeconomic scenarios.
    """

    def __init__(self, matrix: Optional[List[List[float]]] = None):
        self.matrix = np.array(matrix or DEFAULT_TRANSITION_MATRIX, dtype=np.float64)
        self._validate_matrix()

    def _validate_matrix(self):
        """Ensure rows sum to 1 (stochastic matrix)."""
        row_sums = self.matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            # Normalise rows
            self.matrix = self.matrix / row_sums[:, np.newaxis]

    def get_n_period_matrix(self, n: int) -> np.ndarray:
        """Compute the n-period transition matrix via matrix exponentiation."""
        return np.linalg.matrix_power(self.matrix, n)

    def stress_matrix(self, pd_multiplier: float) -> np.ndarray:
        """
        Produce a stressed transition matrix by increasing downgrade
        probabilities by the given multiplier.
        
        Approach: scale off-diagonal downgrade elements, then renormalise.
        """
        stressed = self.matrix.copy()
        n = stressed.shape[0]
        for i in range(n - 1):  # Don't touch the absorbing default state
            for j in range(n):
                if j > i:  # Below-diagonal = downgrade
                    stressed[i, j] *= pd_multiplier
            # Renormalise
            row_sum = stressed[i, :].sum()
            if row_sum > 0:
                stressed[i, :] /= row_sum
        return stressed

    def compute_cumulative_pd(
        self,
        initial_grade_idx: int,
        horizon_years: int,
        pd_multiplier: float = 1.0,
    ) -> float:
        """
        Compute the cumulative probability of default from a given
        starting grade over a multi-year horizon.
        
        Parameters
        ----------
        initial_grade_idx : 0-based index of current rating grade
        horizon_years : projection horizon
        pd_multiplier : stress multiplier on the transition matrix
        """
        if pd_multiplier != 1.0:
            matrix = self.stress_matrix(pd_multiplier)
        else:
            matrix = self.matrix
        n_period = np.linalg.matrix_power(matrix, horizon_years)
        default_col = n_period.shape[1] - 1  # Last column = default
        return float(n_period[initial_grade_idx, default_col])

    def get_migration_profile(
        self, initial_distribution: np.ndarray, years: int = 1, pd_multiplier: float = 1.0
    ) -> np.ndarray:
        """
        Project a portfolio rating distribution forward.
        
        Parameters
        ----------
        initial_distribution : (n_grades,) array of share by rating
        years : projection horizon
        pd_multiplier : stress multiplier
        
        Returns
        -------
        (n_grades,) array with projected shares
        """
        if pd_multiplier != 1.0:
            matrix = self.stress_matrix(pd_multiplier)
        else:
            matrix = self.matrix
        n_period = np.linalg.matrix_power(matrix, years)
        return initial_distribution @ n_period


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Credit Risk Engine
# ═══════════════════════════════════════════════════════════════════════════════

class CreditRiskEngine:
    """
    IFRS 9 & RWA calculator for the full loan book.
    
    Integrates:
    - Segment-level ECL (Stages 1/2/3)
    - Transition matrix-based lifetime PD
    - Macro-conditioned PD stress
    - Standardised Approach RWA
    """

    def __init__(
        self,
        bs: InitialBalanceSheet,
        segments: Optional[List[CreditSegmentParams]] = None,
        transition_engine: Optional[TransitionMatrixEngine] = None,
    ):
        self.bs = bs
        self.segments = segments or DEFAULT_CREDIT_SEGMENTS
        self.tm_engine = transition_engine or TransitionMatrixEngine()
        self.total_ead = bs.total_loan_book

    # ── Stage Distribution ───────────────────────────────────────────────

    @staticmethod
    def _assign_stage_distribution(
        pd_12m: float, pd_multiplier: float = 1.0
    ) -> Dict[int, float]:
        """
        Assign IFRS 9 stage distribution based on current PD level.
        
        Stage 2 trigger: PD has increased significantly from origination.
        Stage 3: PD exceeds 20% (proxy for default/impaired).
        """
        stressed_pd = pd_12m * pd_multiplier
        if stressed_pd >= 0.20:
            return {1: 0.10, 2: 0.20, 3: 0.70}
        elif stressed_pd >= 0.05:
            return {1: 0.30, 2: 0.55, 3: 0.15}
        elif stressed_pd >= 0.02:
            return {1: 0.60, 2: 0.30, 3: 0.10}
        else:
            return {1: 0.85, 2: 0.12, 3: 0.03}

    # ── ECL Computation ──────────────────────────────────────────────────

    def compute_ecl(
        self,
        pd_multiplier: float = 1.0,
    ) -> List[ECLResult]:
        """
        Compute IFRS 9 Expected Credit Losses for each segment.
        
        Stage 1 : ECL = PD_12m × LGD × EAD
        Stage 2 : ECL = PD_lifetime × LGD × EAD
        Stage 3 : ECL = LGD × EAD (fully impaired)
        """
        results = []
        for seg in self.segments:
            ead = self.total_ead * seg.ead_share
            pd_stressed = min(seg.pd_12m * pd_multiplier, 1.0)

            # Lifetime PD from transition matrix
            # Map segment PD to approximate rating grade
            grade_idx = self._map_pd_to_grade(seg.pd_12m)
            lifetime_pd = self.tm_engine.compute_cumulative_pd(
                grade_idx, int(seg.avg_maturity), pd_multiplier
            )
            lifetime_pd = min(lifetime_pd, 1.0)

            # Stage distribution
            stage_dist = self._assign_stage_distribution(seg.pd_12m, pd_multiplier)

            # ECL by stage
            ead_s1 = ead * stage_dist[1]
            ead_s2 = ead * stage_dist[2]
            ead_s3 = ead * stage_dist[3]

            ecl_s1 = ead_s1 * pd_stressed * seg.lgd
            ecl_s2 = ead_s2 * lifetime_pd * seg.lgd
            ecl_s3 = ead_s3 * seg.lgd  # Already defaulted

            total_ecl = ecl_s1 + ecl_s2 + ecl_s3

            # RWA (Standardised Approach)
            rwa = ead * seg.risk_weight

            results.append(ECLResult(
                segment_name=seg.name,
                ead=round(ead, 2),
                pd_12m=round(pd_stressed, 6),
                lifetime_pd=round(lifetime_pd, 6),
                lgd=seg.lgd,
                ecl_stage1=round(ecl_s1, 2),
                ecl_stage2=round(ecl_s2, 2),
                ecl_stage3=round(ecl_s3, 2),
                total_ecl=round(total_ecl, 2),
                rwa=round(rwa, 2),
                stage_distribution=stage_dist,
            ))
        return results

    def compute_total_ecl(self, pd_multiplier: float = 1.0) -> float:
        """Sum of ECL across all segments."""
        return sum(r.total_ecl for r in self.compute_ecl(pd_multiplier))

    def compute_total_rwa(self) -> float:
        """Total RWA across all segments (Standardised)."""
        return sum(r.rwa for r in self.compute_ecl())

    # ── Projection Helper ────────────────────────────────────────────────

    def compute_yearly_credit_losses(
        self, scenario: MacroScenario
    ) -> List[float]:
        """
        Compute yearly credit losses from the macro scenario.
        Uses PD multiplier from the scenario to stress ECLs.
        
        Returns
        -------
        List of annual credit loss amounts (€m)
        """
        losses = []
        for yr in range(len(scenario.pd_multiplier)):
            mult = scenario.pd_multiplier[yr]
            ecl = self.compute_total_ecl(pd_multiplier=mult)
            losses.append(ecl)
        return losses

    def get_ecl_summary_table(self, pd_multiplier: float = 1.0) -> List[Dict]:
        """Return a summary table of ECL by segment."""
        results = self.compute_ecl(pd_multiplier)
        table = []
        for r in results:
            table.append({
                "Segment": r.segment_name,
                "EAD (€m)": r.ead,
                "PD 12m": f"{r.pd_12m:.4%}",
                "Lifetime PD": f"{r.lifetime_pd:.4%}",
                "LGD": f"{r.lgd:.0%}",
                "ECL Stage 1 (€m)": r.ecl_stage1,
                "ECL Stage 2 (€m)": r.ecl_stage2,
                "ECL Stage 3 (€m)": r.ecl_stage3,
                "Total ECL (€m)": r.total_ecl,
                "RWA (€m)": r.rwa,
                "Stage 1 %": f"{r.stage_distribution[1]:.0%}",
                "Stage 2 %": f"{r.stage_distribution[2]:.0%}",
                "Stage 3 %": f"{r.stage_distribution[3]:.0%}",
            })
        return table

    def get_migration_timeline(
        self, scenario: MacroScenario, initial_dist: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Project the portfolio rating distribution year by year under a scenario.
        """
        if initial_dist is None:
            initial_dist = np.array([0.30, 0.40, 0.20, 0.08, 0.02])
        timeline = [{"Year": 0, **dict(zip(RATING_GRADES, initial_dist.tolist()))}]

        dist = initial_dist.copy()
        for yr in range(len(scenario.pd_multiplier)):
            dist = self.tm_engine.get_migration_profile(
                dist, years=1, pd_multiplier=scenario.pd_multiplier[yr]
            )
            timeline.append({
                "Year": yr + 1,
                **{g: round(v, 4) for g, v in zip(RATING_GRADES, dist.tolist())}
            })
        return timeline

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _map_pd_to_grade(pd_12m: float) -> int:
        """Map a PD to the closest rating grade index."""
        if pd_12m <= 0.001:
            return 0  # AAA
        elif pd_12m <= 0.005:
            return 1  # A
        elif pd_12m <= 0.020:
            return 2  # BBB
        else:
            return 3  # Sub-IG

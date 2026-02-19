"""
Pillar D — Operational Risk (Operational VaR)
=================================================
Implements:
  - Loss Distribution Approach (LDA) using Poisson–Lognormal compound model
  - Frequency: Poisson(λ)  —  models discrete event count per year
  - Severity: LogNormal(μ, σ)  —  models individual loss size
  - Aggregate Loss Distribution via Monte Carlo convolution
  - VaR and Expected Shortfall at configurable confidence level

References: Basel II Pillar I AMA, Basel III SMA (Standardised Measurement Approach)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from bank_simulator.config import OpRiskParams, CONFIDENCE_LEVEL


@dataclass
class OpRiskResult:
    """Output of the operational risk VaR computation."""
    expected_loss: float            # Mean of aggregate loss distribution
    var_99_5: float                 # VaR at 99.5 % confidence
    expected_shortfall: float       # ES (CVaR) at 99.5 %
    capital_charge: float           # Max(VaR - EL, 0) — unexpected loss
    n_simulations: int
    mean_frequency: float
    mean_severity: float
    percentiles: Dict[str, float]   # p50, p75, p90, p95, p99, p99.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Operational Risk Engine — Poisson–Lognormal LDA
# ═══════════════════════════════════════════════════════════════════════════════

class OperationalRiskEngine:
    """
    Monte Carlo Loss Distribution Approach for operational risk.
    
    Each simulation year:
      1. Draw event count ~ Poisson(λ)
      2. For each event, draw severity ~ LogNormal(μ, σ)
      3. Aggregate annual loss = Σ severity
    
    Repeat N times to build the aggregate loss distribution.
    """

    def __init__(self, params: Optional[OpRiskParams] = None, seed: int = 42):
        self.params = params or OpRiskParams()
        self.rng = np.random.default_rng(seed)
        self._simulated_losses: Optional[np.ndarray] = None

    # ── Core Simulation ──────────────────────────────────────────────────

    def simulate(self, n_simulations: int = 100_000) -> np.ndarray:
        """
        Run Monte Carlo simulation of the aggregate annual loss distribution.
        
        Returns
        -------
        Array of simulated annual aggregate losses (€ millions).
        """
        p = self.params
        aggregate_losses = np.zeros(n_simulations)

        # Vectorised Poisson draw for all simulations
        event_counts = self.rng.poisson(p.lambda_events_per_year, n_simulations)

        for i in range(n_simulations):
            n_events = event_counts[i]
            if n_events > 0:
                # Draw individual severities (in € millions)
                severities = self.rng.lognormal(
                    p.mu_log_severity, p.sigma_log_severity, n_events
                )
                # Convert from raw € to €m
                aggregate_losses[i] = severities.sum() / 1e6
            else:
                aggregate_losses[i] = 0.0

        self._simulated_losses = aggregate_losses
        return aggregate_losses

    # ── Risk Metrics ─────────────────────────────────────────────────────

    def compute_var(
        self,
        confidence: Optional[float] = None,
        n_simulations: int = 100_000,
    ) -> OpRiskResult:
        """
        Compute VaR and Expected Shortfall from simulated losses.
        
        Parameters
        ----------
        confidence : confidence level (default from config)
        n_simulations : number of simulations if not yet run
        """
        if self._simulated_losses is None or len(self._simulated_losses) != n_simulations:
            self.simulate(n_simulations)

        losses = self._simulated_losses
        conf = confidence or self.params.confidence
        p = self.params

        # VaR
        var = float(np.percentile(losses, conf * 100))

        # Expected Shortfall (mean of losses beyond VaR)
        tail = losses[losses >= var]
        es = float(tail.mean()) if len(tail) > 0 else var

        # Expected loss
        el = float(losses.mean())

        # Capital charge = unexpected loss
        capital = max(var - el, 0)

        # Percentile table
        pctiles = {
            "p50": float(np.percentile(losses, 50)),
            "p75": float(np.percentile(losses, 75)),
            "p90": float(np.percentile(losses, 90)),
            "p95": float(np.percentile(losses, 95)),
            "p99": float(np.percentile(losses, 99)),
            "p99.5": float(np.percentile(losses, 99.5)),
        }

        # Mean severity (analytical)
        mean_sev = np.exp(p.mu_log_severity + p.sigma_log_severity ** 2 / 2) / 1e6

        return OpRiskResult(
            expected_loss=round(el, 2),
            var_99_5=round(var, 2),
            expected_shortfall=round(es, 2),
            capital_charge=round(capital, 2),
            n_simulations=n_simulations,
            mean_frequency=p.lambda_events_per_year,
            mean_severity=round(mean_sev, 2),
            percentiles={k: round(v, 2) for k, v in pctiles.items()},
        )

    # ── Yearly Losses for Projection ─────────────────────────────────────

    def simulate_yearly_losses(self, n_years: int = 3, n_paths: int = 10_000) -> np.ndarray:
        """
        Simulate a panel of yearly operational risk losses.
        
        Returns
        -------
        (n_paths, n_years) array of annual aggregate losses in €m
        """
        p = self.params
        panel = np.zeros((n_paths, n_years))
        for yr in range(n_years):
            counts = self.rng.poisson(p.lambda_events_per_year, n_paths)
            for i in range(n_paths):
                if counts[i] > 0:
                    sevs = self.rng.lognormal(p.mu_log_severity, p.sigma_log_severity, counts[i])
                    panel[i, yr] = sevs.sum() / 1e6
        return panel

    def get_expected_yearly_losses(self, n_years: int = 3) -> List[float]:
        """
        Return expected (mean) annual op-risk losses for each projection year.
        Uses Monte Carlo with 10,000 paths.
        """
        panel = self.simulate_yearly_losses(n_years=n_years, n_paths=10_000)
        return [round(float(panel[:, yr].mean()), 2) for yr in range(n_years)]

    def get_loss_distribution_histogram(
        self, n_bins: int = 100, n_simulations: int = 100_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return histogram data for the loss distribution."""
        if self._simulated_losses is None:
            self.simulate(n_simulations)
        counts, edges = np.histogram(self._simulated_losses, bins=n_bins)
        return counts, edges

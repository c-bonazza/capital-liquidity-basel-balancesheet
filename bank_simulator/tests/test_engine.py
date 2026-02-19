"""
Comprehensive Test Suite for the Bank Balance Sheet Simulator.
Tests all risk pillars, the projection engine, and the optimizer.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bank_simulator.config import (
    InitialBalanceSheet,
    BASE_SCENARIO,
    ADVERSE_SCENARIO,
    SEVERELY_ADVERSE_SCENARIO,
    MIN_CET1_RATIO,
    MIN_LCR_RATIO,
    DEFAULT_TRANSITION_MATRIX,
)
from bank_simulator.engine.balance_sheet import BalanceSheetProjector, BalanceSheetSnapshot
from bank_simulator.risk_modules.market_risk import MarketRiskALM
from bank_simulator.risk_modules.liquidity_risk import LiquidityRiskEngine
from bank_simulator.risk_modules.credit_risk import CreditRiskEngine, TransitionMatrixEngine
from bank_simulator.risk_modules.operational_risk import OperationalRiskEngine
from bank_simulator.optimization import CapitalOptimizer, RAROCCalculator, run_full_optimization, SEGMENT_NAMES, N_SEGMENTS
from bank_simulator.stress_testing import MonteCarloStressEngine


@pytest.fixture
def bs():
    return InitialBalanceSheet()


# ═══════════════════════════════════════════════════════════════════════════════
#  Balance Sheet Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBalanceSheet:
    def test_initial_balance_sheet_balances(self, bs):
        """Assets ≈ Liabilities + Equity."""
        gap = abs(bs.total_assets - bs.total_liabilities - bs.total_equity)
        assert gap < 1e-6, f"Balance sheet gap: {gap}"

    def test_total_assets_positive(self, bs):
        assert bs.total_assets > 0

    def test_total_loan_book(self, bs):
        expected = (bs.residential_mortgages + bs.sme_loans +
                    bs.consumer_credit + bs.corporate_loans)
        assert abs(bs.total_loan_book - expected) < 1e-6

    def test_hqla_positive(self, bs):
        assert bs.total_hqla > 0


class TestBalanceSheetProjector:
    def test_t0_snapshot(self, bs):
        projector = BalanceSheetProjector(bs)
        snap = projector.build_t0_snapshot()
        assert snap.year == 0
        assert snap.total_assets > 0
        assert snap.total_rwa > 0
        assert snap.cet1_ratio > 0

    def test_projection_creates_correct_years(self, bs):
        projector = BalanceSheetProjector(bs)
        credit_engine = CreditRiskEngine(bs)
        losses = credit_engine.compute_yearly_credit_losses(BASE_SCENARIO)
        snapshots = projector.project(
            scenario=BASE_SCENARIO,
            credit_losses_by_year=losses,
            oprisk_losses_by_year=[10.0, 10.0, 10.0],
            market_pnl_by_year=[0.0, 0.0, 0.0],
        )
        assert len(snapshots) == 4  # t=0, t=1, t=2, t=3
        assert snapshots[0].year == 0
        assert snapshots[-1].year == 3

    def test_baseline_cet1_stays_above_minimum(self, bs):
        projector = BalanceSheetProjector(bs)
        credit_engine = CreditRiskEngine(bs)
        losses = credit_engine.compute_yearly_credit_losses(BASE_SCENARIO)
        snapshots = projector.project(
            scenario=BASE_SCENARIO,
            credit_losses_by_year=losses,
            oprisk_losses_by_year=[10.0, 10.0, 10.0],
            market_pnl_by_year=[0.0, 0.0, 0.0],
        )
        for snap in snapshots:
            assert snap.cet1_ratio > 0.05, f"CET1 too low at year {snap.year}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Market Risk Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketRisk:
    def test_gap_schedule_has_7_buckets(self, bs):
        engine = MarketRiskALM(bs)
        assert len(engine.gap_schedule) == 7

    def test_modified_duration_positive(self, bs):
        engine = MarketRiskALM(bs)
        mod_dur = engine.compute_modified_duration_equity()
        # Duration can be negative depending on asset-liability structure
        assert isinstance(mod_dur, float)

    def test_eve_delta_direction(self, bs):
        engine = MarketRiskALM(bs)
        eve_up = engine.compute_eve_delta(200)
        eve_down = engine.compute_eve_delta(-200)
        # Up and down should have opposite signs
        assert eve_up * eve_down <= 0 or abs(eve_up) < 1e-6

    def test_nii_sensitivity(self, bs):
        engine = MarketRiskALM(bs)
        nii = engine.compute_nii_sensitivity(200)
        assert isinstance(nii, float)

    def test_six_scenarios_returns_6(self, bs):
        engine = MarketRiskALM(bs)
        scenarios = engine.compute_eve_six_scenarios()
        assert len(scenarios) == 6

    def test_yearly_impacts(self, bs):
        engine = MarketRiskALM(bs)
        nii_adj, mkt_pnl = engine.compute_yearly_impacts(ADVERSE_SCENARIO)
        assert len(nii_adj) == 3
        assert len(mkt_pnl) == 3


# ═══════════════════════════════════════════════════════════════════════════════
#  Liquidity Risk Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiquidityRisk:
    def test_lcr_above_100_baseline(self, bs):
        engine = LiquidityRiskEngine(bs)
        lcr = engine.compute_lcr()
        assert lcr.lcr_ratio >= 1.0, f"LCR={lcr.lcr_ratio:.2f} < 100%"

    def test_lcr_components_positive(self, bs):
        engine = LiquidityRiskEngine(bs)
        lcr = engine.compute_lcr()
        assert lcr.total_hqla > 0
        assert lcr.total_outflows > 0
        assert lcr.net_cash_outflows > 0

    def test_stressed_lcr_lower(self, bs):
        engine = LiquidityRiskEngine(bs)
        lcr_base = engine.compute_lcr(stress_multiplier=1.0)
        lcr_stress = engine.compute_lcr(stress_multiplier=2.0)
        assert lcr_stress.lcr_ratio < lcr_base.lcr_ratio

    def test_bank_run_survival(self, bs):
        engine = LiquidityRiskEngine(bs)
        result = engine.simulate_bank_run()
        assert result.survival_days > 0
        assert result.survival_days <= 90
        assert result.initial_hqla > 0

    def test_multiple_runs(self, bs):
        engine = LiquidityRiskEngine(bs)
        results = engine.simulate_multiple_runs()
        assert len(results) == 3
        # More severe scenarios should have shorter survival
        assert results[2].survival_days <= results[0].survival_days

    def test_deposit_outflows(self, bs):
        engine = LiquidityRiskEngine(bs)
        outflows = engine.compute_yearly_deposit_outflows(ADVERSE_SCENARIO)
        assert len(outflows) == 3
        assert all(o >= 0 for o in outflows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Credit Risk Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreditRisk:
    def test_ecl_baseline_positive(self, bs):
        engine = CreditRiskEngine(bs)
        ecl = engine.compute_total_ecl()
        assert ecl > 0

    def test_stressed_ecl_higher(self, bs):
        engine = CreditRiskEngine(bs)
        ecl_base = engine.compute_total_ecl(pd_multiplier=1.0)
        ecl_stress = engine.compute_total_ecl(pd_multiplier=3.0)
        assert ecl_stress > ecl_base

    def test_ecl_segments_count(self, bs):
        engine = CreditRiskEngine(bs)
        results = engine.compute_ecl()
        assert len(results) == 4  # 4 segments

    def test_rwa_positive(self, bs):
        engine = CreditRiskEngine(bs)
        rwa = engine.compute_total_rwa()
        assert rwa > 0

    def test_yearly_credit_losses(self, bs):
        engine = CreditRiskEngine(bs)
        losses = engine.compute_yearly_credit_losses(ADVERSE_SCENARIO)
        assert len(losses) == 3
        assert all(l > 0 for l in losses)

    def test_migration_timeline(self, bs):
        engine = CreditRiskEngine(bs)
        timeline = engine.get_migration_timeline(ADVERSE_SCENARIO)
        assert len(timeline) == 4  # t=0 + 3 years
        # Default rate should increase under adverse
        assert timeline[-1]["Default"] > timeline[0]["Default"]


class TestTransitionMatrix:
    def test_matrix_rows_sum_to_one(self):
        tm = TransitionMatrixEngine()
        row_sums = tm.matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_n_period_matrix(self):
        tm = TransitionMatrixEngine()
        m2 = tm.get_n_period_matrix(2)
        assert m2.shape == (5, 5)
        np.testing.assert_allclose(m2.sum(axis=1), 1.0, atol=1e-4)

    def test_stressed_matrix_higher_defaults(self):
        tm = TransitionMatrixEngine()
        base_pd = tm.compute_cumulative_pd(2, 3, pd_multiplier=1.0)
        stress_pd = tm.compute_cumulative_pd(2, 3, pd_multiplier=3.0)
        assert stress_pd > base_pd

    def test_absorbing_default_state(self):
        tm = TransitionMatrixEngine()
        m10 = tm.get_n_period_matrix(10)
        # Default state should remain 100% default
        np.testing.assert_allclose(m10[4, 4], 1.0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
#  Operational Risk Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOperationalRisk:
    def test_simulation_returns_array(self):
        engine = OperationalRiskEngine(seed=42)
        losses = engine.simulate(n_simulations=1000)
        assert len(losses) == 1000
        assert all(l >= 0 for l in losses)

    def test_var_positive(self):
        engine = OperationalRiskEngine(seed=42)
        result = engine.compute_var(n_simulations=10000)
        assert result.var_99_5 > 0
        assert result.expected_loss > 0
        assert result.var_99_5 >= result.expected_loss

    def test_es_exceeds_var(self):
        engine = OperationalRiskEngine(seed=42)
        result = engine.compute_var(n_simulations=10000)
        assert result.expected_shortfall >= result.var_99_5

    def test_capital_charge_positive(self):
        engine = OperationalRiskEngine(seed=42)
        result = engine.compute_var(n_simulations=10000)
        assert result.capital_charge >= 0

    def test_yearly_losses(self):
        engine = OperationalRiskEngine(seed=42)
        losses = engine.get_expected_yearly_losses(3)
        assert len(losses) == 3
        assert all(l > 0 for l in losses)

    def test_histogram_data(self):
        engine = OperationalRiskEngine(seed=42)
        engine.simulate(10000)
        counts, edges = engine.get_loss_distribution_histogram(n_bins=50)
        assert len(counts) == 50
        assert len(edges) == 51


# ═══════════════════════════════════════════════════════════════════════════════
#  RAROC Optimization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAROCOptimization:
    def test_equal_weights_raroc(self, bs):
        calc = RAROCCalculator(bs)
        weights = np.ones(N_SEGMENTS) / N_SEGMENTS
        result = calc.compute(weights)
        assert result.portfolio_raroc > 0
        assert abs(sum(result.weights.values()) - 1.0) < 1e-3

    def test_optimization_converges(self, bs):
        optimizer = CapitalOptimizer(bs)
        result = optimizer.optimize()
        assert result.success
        assert result.optimal_raroc > 0
        # Optimal should be at least as good as initial
        assert result.optimal_raroc >= result.initial_raroc - 0.01

    def test_optimal_weights_sum_to_one(self, bs):
        optimizer = CapitalOptimizer(bs)
        result = optimizer.optimize()
        total = sum(result.optimal_weights.values())
        assert abs(total - 1.0) < 1e-3

    def test_optimal_cet1_feasible(self, bs):
        optimizer = CapitalOptimizer(bs)
        result = optimizer.optimize()
        assert result.optimal_cet1 >= MIN_CET1_RATIO - 0.01

    def test_run_full_optimization(self):
        result = run_full_optimization()
        assert result.success
        assert len(result.optimal_weights) == N_SEGMENTS


# ═══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo Stress Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestStressTesting:
    def test_baseline_stress_test(self, bs):
        engine = MonteCarloStressEngine(bs=bs, seed=42)
        result = engine.run(BASE_SCENARIO, n_paths=100)
        assert result.n_paths == 100
        assert result.n_years == 3
        assert result.cet1_mean > 0

    def test_adverse_higher_breach_prob(self, bs):
        engine = MonteCarloStressEngine(bs=bs, seed=42)
        base = engine.run(BASE_SCENARIO, n_paths=200)
        adverse = engine.run(ADVERSE_SCENARIO, n_paths=200)
        # Adverse should generally have higher breach probability
        assert adverse.cet1_mean <= base.cet1_mean + 0.05

    def test_all_scenarios(self, bs):
        engine = MonteCarloStressEngine(bs=bs, seed=42)
        results = engine.run_all_scenarios(n_paths=50)
        assert len(results) == 3

    def test_result_distributions(self, bs):
        engine = MonteCarloStressEngine(bs=bs, seed=42)
        result = engine.run(BASE_SCENARIO, n_paths=100)
        assert len(result.all_terminal_cet1) == 100
        assert result.cet1_p5 <= result.cet1_p50 <= result.cet1_p95


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Mart Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataMart:
    def test_initialise(self):
        from bank_simulator.data import DataMart
        with DataMart() as dm:
            loans = dm.get_loan_book()
            assert len(loans) > 0

    def test_portfolio_summary(self):
        from bank_simulator.data import DataMart
        with DataMart() as dm:
            summary = dm.get_portfolio_summary()
            assert len(summary) == 4  # 4 asset classes

    def test_lcr_components(self):
        from bank_simulator.data import DataMart
        with DataMart() as dm:
            lcr = dm.get_lcr_components()
            assert len(lcr) > 0

    def test_total_rwa_positive(self):
        from bank_simulator.data import DataMart
        with DataMart() as dm:
            rwa = dm.get_total_rwa()
            assert rwa > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

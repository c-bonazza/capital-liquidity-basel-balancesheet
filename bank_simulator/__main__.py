"""
Main entry point — run all risk modules and display summary.
Usage: python -m bank_simulator
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bank_simulator.config import (
    InitialBalanceSheet, BASE_SCENARIO, ADVERSE_SCENARIO,
    SEVERELY_ADVERSE_SCENARIO, MIN_CET1_RATIO, MIN_LCR_RATIO,
)
from bank_simulator.engine.balance_sheet import BalanceSheetProjector
from bank_simulator.risk_modules.market_risk import MarketRiskALM
from bank_simulator.risk_modules.liquidity_risk import LiquidityRiskEngine
from bank_simulator.risk_modules.credit_risk import CreditRiskEngine
from bank_simulator.risk_modules.operational_risk import OperationalRiskEngine
from bank_simulator.optimization import run_full_optimization


def main():
    bs = InitialBalanceSheet()
    print("=" * 72)
    print("  INTEGRATED BANK BALANCE SHEET SIMULATOR & CAPITAL OPTIMIZER (ICAAP/ILAAP)")
    print("=" * 72)

    # ── Balance Sheet ────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("INITIAL BALANCE SHEET (€m)")
    print(f"{'─' * 40}")
    print(f"  Total Assets:      €{bs.total_assets:>10,.1f}m")
    print(f"  Total Liabilities: €{bs.total_liabilities:>10,.1f}m")
    print(f"  Total Equity:      €{bs.total_equity:>10,.1f}m")
    print(f"  Loan Book:         €{bs.total_loan_book:>10,.1f}m")
    print(f"  HQLA:              €{bs.total_hqla:>10,.1f}m")

    # ── Market Risk ──────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("MARKET RISK / ALM (IRRBB)")
    print(f"{'─' * 40}")
    market = MarketRiskALM(bs)
    mod_dur = market.compute_modified_duration_equity()
    eve_200 = market.compute_eve_delta(200)
    nii_200 = market.compute_nii_sensitivity(200)
    print(f"  Modified Duration of Equity: {mod_dur:.2f} years")
    print(f"  Δ-EVE (+200bp):   €{eve_200:>10,.1f}m")
    print(f"  Δ-NII (+200bp):   €{nii_200:>10,.1f}m")
    print(f"\n  EVE — 6 Basel Scenarios:")
    for name, delta in market.compute_eve_six_scenarios().items():
        print(f"    {name:30s}: €{delta:>10,.1f}m")

    # ── Liquidity Risk ───────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("LIQUIDITY RISK (ILAAP)")
    print(f"{'─' * 40}")
    liquidity = LiquidityRiskEngine(bs)
    lcr = liquidity.compute_lcr()
    print(f"  LCR:               {lcr.lcr_ratio:>10.0%}")
    print(f"  HQLA:              €{lcr.total_hqla:>10,.1f}m")
    print(f"  Net Outflows (30d):€{lcr.net_cash_outflows:>10,.1f}m")
    print(f"\n  Bank Run Simulations:")
    for r in liquidity.simulate_multiple_runs():
        print(f"    {r.scenario_name:20s}: {r.survival_days:3d} days survival")

    # ── Credit Risk ──────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("CREDIT RISK (IFRS 9 & RWA)")
    print(f"{'─' * 40}")
    credit = CreditRiskEngine(bs)
    ecl_base = credit.compute_total_ecl(pd_multiplier=1.0)
    ecl_adv = credit.compute_total_ecl(pd_multiplier=3.0)
    rwa = credit.compute_total_rwa()
    print(f"  Total ECL (baseline): €{ecl_base:>10,.1f}m")
    print(f"  Total ECL (stressed): €{ecl_adv:>10,.1f}m")
    print(f"  Total RWA:            €{rwa:>10,.1f}m")
    print(f"  CET1 Ratio (t=0):     {bs.cet1_capital / rwa:>10.1%}")

    # ── Operational Risk ─────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("OPERATIONAL RISK (Poisson–Lognormal VaR)")
    print(f"{'─' * 40}")
    op = OperationalRiskEngine(seed=42)
    op_result = op.compute_var(n_simulations=50_000)
    print(f"  Expected Loss:    €{op_result.expected_loss:>10,.1f}m")
    print(f"  VaR (99.5%):      €{op_result.var_99_5:>10,.1f}m")
    print(f"  Expected Shortfall:€{op_result.expected_shortfall:>10,.1f}m")
    print(f"  Capital Charge:   €{op_result.capital_charge:>10,.1f}m")

    # ── RAROC Optimisation ───────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("RAROC CAPITAL OPTIMISATION")
    print(f"{'─' * 40}")
    opt = run_full_optimization(bs)
    print(f"  Initial RAROC:  {opt.initial_raroc:>10.2%}")
    print(f"  Optimal RAROC:  {opt.optimal_raroc:>10.2%}")
    print(f"  Optimal CET1:   {opt.optimal_cet1:>10.1%}")
    print(f"  Converged:      {opt.success}")
    print(f"\n  Optimal Allocation:")
    for seg, w in opt.optimal_weights.items():
        initial_w = opt.initial_weights.get(seg, 0)
        delta = (w - initial_w) * 100
        print(f"    {seg:25s}: {w:6.1%}  ({delta:+5.1f}pp)")

    # ── Multi-Year Projection ────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print(f"3-YEAR PROJECTION — {ADVERSE_SCENARIO.name}")
    print(f"{'─' * 40}")
    credit_losses = credit.compute_yearly_credit_losses(ADVERSE_SCENARIO)
    oprisk_losses = op.get_expected_yearly_losses(3)
    nii_adj, mkt_pnl = market.compute_yearly_impacts(ADVERSE_SCENARIO)
    dep_outflows = liquidity.compute_yearly_deposit_outflows(ADVERSE_SCENARIO)

    projector = BalanceSheetProjector(bs)
    snapshots = projector.project(
        scenario=ADVERSE_SCENARIO,
        credit_losses_by_year=credit_losses,
        oprisk_losses_by_year=oprisk_losses,
        market_pnl_by_year=mkt_pnl,
        nii_adjustment_by_year=nii_adj,
        deposit_outflow_by_year=dep_outflows,
    )
    for s in projector.get_projection_summary():
        print(f"  Year {s['Year']}: CET1={s['CET1 Ratio (%)']:6.2f}% | "
              f"RWA=€{s['Total RWA (€m)']:,.0f}m | "
              f"NI=€{s['Net Income (€m)']:,.0f}m | "
              f"ROE={s['ROE (%)']:5.2f}%")

    print(f"\n{'=' * 72}")
    print("  Run 'streamlit run bank_simulator/dashboard/app.py' for the dashboard")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

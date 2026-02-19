"""
Streamlit Executive Dashboard — "Bank-in-a-Box" Control Tower
==============================================================
A comprehensive, interactive dashboard that presents:
  1. Balance Sheet Overview & Key Ratios
  2. Market Risk / ALM (IRRBB) — Gap analysis, EVE sensitivity
  3. Liquidity Risk (ILAAP) — LCR, bank-run survival
  4. Credit Risk (IFRS 9) — ECL by stage, transition matrices
  5. Operational Risk — VaR distribution
  6. Monte Carlo Stress Testing — CET1/LCR distributions
  7. RAROC Capital Optimisation — Optimal allocation

Launch: streamlit run bank_simulator/dashboard/app.py
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bank_simulator.config import (
    InitialBalanceSheet,
    ALL_SCENARIOS,
    BASE_SCENARIO,
    ADVERSE_SCENARIO,
    SEVERELY_ADVERSE_SCENARIO,
    MIN_CET1_RATIO,
    MIN_LCR_RATIO,
    RATING_GRADES,
)
from bank_simulator.engine.balance_sheet import BalanceSheetProjector
from bank_simulator.risk_modules.market_risk import MarketRiskALM
from bank_simulator.risk_modules.liquidity_risk import LiquidityRiskEngine
from bank_simulator.risk_modules.credit_risk import CreditRiskEngine
from bank_simulator.risk_modules.operational_risk import OperationalRiskEngine
from bank_simulator.optimization import CapitalOptimizer, RAROCCalculator, SEGMENT_NAMES
from bank_simulator.stress_testing import MonteCarloStressEngine


# ══════════════════════════════════════════════════════════════════════════════
#  Page Configuration
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bank-in-a-Box — Risk 360 Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    h1 {color: #1a237e;}
    h2 {color: #283593; border-bottom: 2px solid #3949ab; padding-bottom: 0.3rem;}
    .metric-card {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar Controls
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🏦 Bank-in-a-Box")
st.sidebar.markdown("**Integrated Balance Sheet Simulator**")
st.sidebar.markdown("---")

scenario_choice = st.sidebar.selectbox(
    "📊 Macro Scenario",
    ["Baseline", "Adverse", "Severely Adverse"],
    index=0,
)

scenario_map = {
    "Baseline": BASE_SCENARIO,
    "Adverse": ADVERSE_SCENARIO,
    "Severely Adverse": SEVERELY_ADVERSE_SCENARIO,
}
selected_scenario = scenario_map[scenario_choice]

mc_paths = st.sidebar.slider("Monte Carlo Paths", 100, 5000, 500, step=100)

st.sidebar.markdown("---")
st.sidebar.markdown("### Balance Sheet Overrides (€m)")
cet1_override = st.sidebar.number_input("CET1 Capital", value=5500.0, step=100.0)
total_deposits = st.sidebar.number_input("Total Deposits", value=45000.0, step=1000.0)

# Build balance sheet with overrides
bs = InitialBalanceSheet()
bs.cet1_capital = cet1_override

st.sidebar.markdown("---")
st.sidebar.info("Built for ICAAP/ILAAP Integration — Risk 360 Framework")


# ══════════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏦 Integrated Bank Balance Sheet Simulator & Capital Optimizer (ICAAP/ILAAP)")
st.markdown(f"**Scenario: {selected_scenario.name}** — "
            f"Projection Horizon: 3 Years — "
            f"Monte Carlo: {mc_paths:,} paths")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab Layout
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📋 Balance Sheet",
    "📈 Market Risk / ALM",
    "💧 Liquidity Risk",
    "💳 Credit Risk (IFRS 9)",
    "⚡ Operational Risk",
    "🎲 Stress Testing",
    "🎯 RAROC Optimisation",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Balance Sheet Overview
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Balance Sheet Overview & Multi-Year Projection")

    # Initialise engines
    market_engine = MarketRiskALM(bs)
    liquidity_engine = LiquidityRiskEngine(bs)
    credit_engine = CreditRiskEngine(bs)
    op_engine = OperationalRiskEngine(seed=42)

    # Compute inputs for projection
    credit_losses = credit_engine.compute_yearly_credit_losses(selected_scenario)
    oprisk_losses = op_engine.get_expected_yearly_losses(3)
    nii_adj, mkt_pnl = market_engine.compute_yearly_impacts(selected_scenario)
    dep_outflows = liquidity_engine.compute_yearly_deposit_outflows(selected_scenario)

    projector = BalanceSheetProjector(bs)
    snapshots = projector.project(
        scenario=selected_scenario,
        credit_losses_by_year=credit_losses,
        oprisk_losses_by_year=oprisk_losses,
        market_pnl_by_year=mkt_pnl,
        nii_adjustment_by_year=nii_adj,
        deposit_outflow_by_year=dep_outflows,
    )
    summary = projector.get_projection_summary()

    # Key metrics at t=0
    t0 = snapshots[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Assets", f"€{t0.total_assets:,.0f}m")
    col2.metric("CET1 Ratio", f"{t0.cet1_ratio:.1%}",
                delta=f"{(snapshots[-1].cet1_ratio - t0.cet1_ratio)*100:+.1f}pp")
    col3.metric("Total RWA", f"€{t0.total_rwa:,.0f}m")
    col4.metric("HQLA", f"€{t0.total_hqla:,.0f}m")
    col5.metric("ROE", f"{t0.roe:.1%}" if t0.year > 0 else "N/A (t=0)")

    st.markdown("---")

    # Balance sheet composition
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Asset Composition (t=0)")
        asset_data = {
            "Cash & Reserves": t0.cash_and_reserves,
            "HQLA L1 (Govt Bonds)": t0.hqla_l1,
            "HQLA L2A (Covered)": t0.hqla_l2a,
            "HQLA L2B (Corporate)": t0.hqla_l2b,
            "Residential Mortgages": t0.residential_mortgages,
            "SME Loans": t0.sme_loans,
            "Consumer Credit": t0.consumer_credit,
            "Corporate Loans": t0.corporate_loans,
            "Trading Book": t0.trading_book,
            "Other": t0.other_assets,
        }
        fig_assets = px.pie(
            names=list(asset_data.keys()),
            values=list(asset_data.values()),
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4,
        )
        fig_assets.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_assets, use_container_width=True)

    with col_b:
        st.subheader("Liability & Equity Composition (t=0)")
        liab_data = {
            "Retail Stable": t0.retail_deposits_stable,
            "Retail Less Stable": t0.retail_deposits_less_stable,
            "Wholesale Op.": t0.wholesale_operational,
            "Wholesale Non-Op.": t0.wholesale_non_operational,
            "Wholesale Unsecured": t0.wholesale_unsecured,
            "Subordinated Debt": t0.subordinated_debt,
            "Other Liabil.": t0.other_liabilities,
            "CET1": t0.cet1_capital,
            "AT1": t0.at1_capital,
            "Tier 2": t0.tier2_capital,
        }
        fig_liab = px.pie(
            names=list(liab_data.keys()),
            values=list(liab_data.values()),
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4,
        )
        fig_liab.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_liab, use_container_width=True)

    # Projection table
    st.subheader("Multi-Year Projection")
    df_proj = pd.DataFrame(summary)
    st.dataframe(df_proj.style.format({
        "Total Assets (€m)": "{:,.1f}",
        "Total Loans (€m)": "{:,.1f}",
        "CET1 Capital (€m)": "{:,.1f}",
        "Total RWA (€m)": "{:,.1f}",
        "CET1 Ratio (%)": "{:.2f}%",
        "Total Capital Ratio (%)": "{:.2f}%",
        "Leverage Ratio (%)": "{:.2f}%",
        "NII (€m)": "{:,.1f}",
        "Credit Losses (€m)": "{:,.1f}",
        "Net Income (€m)": "{:,.1f}",
        "ROE (%)": "{:.2f}%",
    }), use_container_width=True)

    # CET1 ratio evolution chart
    fig_cet1 = go.Figure()
    years = [s["Year"] for s in summary]
    fig_cet1.add_trace(go.Scatter(
        x=years, y=[s["CET1 Ratio (%)"] for s in summary],
        mode="lines+markers", name="CET1 Ratio",
        line=dict(width=3, color="#1a237e"),
    ))
    fig_cet1.add_hline(y=MIN_CET1_RATIO * 100, line_dash="dash",
                        line_color="red", annotation_text="Min CET1 (12%)")
    fig_cet1.update_layout(
        title="CET1 Ratio Trajectory",
        xaxis_title="Year", yaxis_title="CET1 Ratio (%)",
        height=350,
    )
    st.plotly_chart(fig_cet1, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Market Risk / ALM (IRRBB)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Market Risk & ALM — Interest Rate Risk in the Banking Book")

    col1, col2, col3 = st.columns(3)
    mod_dur = market_engine.compute_modified_duration_equity()
    eve_200 = market_engine.compute_eve_delta(200)
    nii_200 = market_engine.compute_nii_sensitivity(200)

    col1.metric("Modified Duration of Equity", f"{mod_dur:.2f} years")
    col2.metric("Δ-EVE (+200bp)", f"€{eve_200:,.1f}m",
                delta="Loss" if eve_200 < 0 else "Gain")
    col3.metric("Δ-NII (+200bp, 1Y)", f"€{nii_200:,.1f}m",
                delta="Positive" if nii_200 > 0 else "Negative")

    st.markdown("---")

    # Gap Analysis Table
    st.subheader("Repricing Gap Schedule")
    gap_table = market_engine.get_gap_table()
    df_gap = pd.DataFrame(gap_table)
    st.dataframe(df_gap.style.format({
        "RSA (€m)": "{:,.1f}",
        "RSL (€m)": "{:,.1f}",
        "Gap (€m)": "{:,.1f}",
        "Cumulative Gap (€m)": "{:,.1f}",
        "Duration-Weighted Gap (€m·yr)": "{:,.1f}",
    }).background_gradient(subset=["Gap (€m)"], cmap="RdYlGn"),
    use_container_width=True)

    # Gap bar chart
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        x=df_gap["Bucket"], y=df_gap["RSA (€m)"],
        name="Rate-Sensitive Assets", marker_color="#1565c0",
    ))
    fig_gap.add_trace(go.Bar(
        x=df_gap["Bucket"], y=df_gap["RSL (€m)"],
        name="Rate-Sensitive Liabilities", marker_color="#c62828",
    ))
    fig_gap.add_trace(go.Scatter(
        x=df_gap["Bucket"], y=df_gap["Cumulative Gap (€m)"],
        mode="lines+markers", name="Cumulative Gap",
        line=dict(width=3, color="#ff6f00"),
        yaxis="y2",
    ))
    fig_gap.update_layout(
        barmode="group",
        title="Repricing Gap Analysis",
        yaxis_title="€m",
        yaxis2=dict(title="Cumulative Gap (€m)", overlaying="y", side="right"),
        height=450,
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # EVE scenarios
    st.subheader("EVE Sensitivity — Basel IRRBB 6 Scenarios")
    eve_scenarios = market_engine.compute_eve_six_scenarios()
    df_eve = pd.DataFrame([
        {"Scenario": k, "Δ-EVE (€m)": v} for k, v in eve_scenarios.items()
    ])
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(df_eve, use_container_width=True)
    with col_b:
        fig_eve = px.bar(df_eve, x="Scenario", y="Δ-EVE (€m)",
                         color="Δ-EVE (€m)",
                         color_continuous_scale="RdYlGn",
                         title="Δ-EVE by Scenario")
        fig_eve.update_layout(height=350)
        st.plotly_chart(fig_eve, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Liquidity Risk (ILAAP)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Liquidity Risk — ILAAP Framework")

    lcr = liquidity_engine.compute_lcr()
    lcr_stressed = liquidity_engine.compute_lcr(stress_multiplier=1.5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LCR (Baseline)", f"{lcr.lcr_ratio:.0%}",
                delta="✅ Compliant" if lcr.lcr_ratio >= 1.0 else "❌ Breach")
    col2.metric("LCR (Stressed ×1.5)", f"{lcr_stressed.lcr_ratio:.0%}")
    col3.metric("Total HQLA", f"€{lcr.total_hqla:,.0f}m")
    col4.metric("Net Cash Outflows (30d)", f"€{lcr.net_cash_outflows:,.0f}m")

    st.markdown("---")

    # LCR Decomposition
    st.subheader("LCR Decomposition")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**HQLA Buffer**")
        hqla_df = pd.DataFrame([
            {"Level": "Level 1", "Amount (€m)": lcr.hqla_l1},
            {"Level": "Level 2A", "Amount (€m)": lcr.hqla_l2a},
            {"Level": "Level 2B", "Amount (€m)": lcr.hqla_l2b},
            {"Level": "**Total HQLA**", "Amount (€m)": lcr.total_hqla},
        ])
        st.dataframe(hqla_df, use_container_width=True)

    with col_b:
        st.markdown("**Cash Outflows (30-day)**")
        outflow_df = pd.DataFrame([
            {"Category": "Retail Stable", "Outflow (€m)": lcr.retail_stable_outflow},
            {"Category": "Retail Less Stable", "Outflow (€m)": lcr.retail_less_stable_outflow},
            {"Category": "Wholesale Operational", "Outflow (€m)": lcr.wholesale_operational_outflow},
            {"Category": "Wholesale Non-Op.", "Outflow (€m)": lcr.wholesale_non_operational_outflow},
            {"Category": "Wholesale Unsecured", "Outflow (€m)": lcr.wholesale_unsecured_outflow},
            {"Category": "**Total Outflows**", "Outflow (€m)": lcr.total_outflows},
        ])
        st.dataframe(outflow_df, use_container_width=True)

    st.markdown("---")

    # Bank Run Simulation
    st.subheader("🏃 Bank Run Survival Simulation")
    run_results = liquidity_engine.simulate_multiple_runs()

    for result in run_results:
        severity_color = "🟢" if result.survival_days > 60 else ("🟡" if result.survival_days > 30 else "🔴")
        st.markdown(f"**{severity_color} {result.scenario_name}** — Survival: **{result.survival_days} days** | "
                    f"Peak Daily Outflow: €{result.peak_daily_outflow:,.0f}m")

    # Plot bank run
    fig_run = go.Figure()
    for result in run_results:
        days = list(range(len(result.daily_hqla)))
        fig_run.add_trace(go.Scatter(
            x=days, y=result.daily_hqla,
            mode="lines", name=f"{result.scenario_name} (HQLA)",
            line=dict(width=2),
        ))
    fig_run.add_hline(y=0, line_dash="dash", line_color="red",
                       annotation_text="HQLA Depletion")
    fig_run.update_layout(
        title="Bank Run Simulation — HQLA Depletion Path",
        xaxis_title="Day",
        yaxis_title="Remaining HQLA (€m)",
        height=400,
    )
    st.plotly_chart(fig_run, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Credit Risk (IFRS 9)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Credit Risk — IFRS 9 & RWA Analysis")

    # ECL summary
    ecl_baseline = credit_engine.compute_ecl(pd_multiplier=1.0)
    ecl_stressed = credit_engine.compute_ecl(
        pd_multiplier=selected_scenario.pd_multiplier[0]
    )

    total_ecl_base = sum(r.total_ecl for r in ecl_baseline)
    total_ecl_stress = sum(r.total_ecl for r in ecl_stressed)
    total_rwa = sum(r.rwa for r in ecl_baseline)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total ECL (Baseline)", f"€{total_ecl_base:,.1f}m")
    col2.metric("Total ECL (Stressed)", f"€{total_ecl_stress:,.1f}m",
                delta=f"+€{total_ecl_stress - total_ecl_base:,.1f}m")
    col3.metric("Total RWA", f"€{total_rwa:,.0f}m")
    col4.metric("ECL / Loan Book", f"{total_ecl_base / bs.total_loan_book * 100:.2f}%")

    st.markdown("---")

    # ECL by segment
    st.subheader("ECL by Segment & IFRS 9 Stage")
    ecl_table = credit_engine.get_ecl_summary_table(pd_multiplier=1.0)
    st.dataframe(pd.DataFrame(ecl_table), use_container_width=True)

    # ECL waterfall
    fig_ecl = go.Figure()
    for r in ecl_baseline:
        fig_ecl.add_trace(go.Bar(
            name=r.segment_name,
            x=["Stage 1", "Stage 2", "Stage 3"],
            y=[r.ecl_stage1, r.ecl_stage2, r.ecl_stage3],
        ))
    fig_ecl.update_layout(
        barmode="group",
        title="ECL by Segment & Stage (Baseline)",
        yaxis_title="ECL (€m)",
        height=400,
    )
    st.plotly_chart(fig_ecl, use_container_width=True)

    st.markdown("---")

    # Transition Matrix
    st.subheader("Rating Migration — Transition Matrix Analysis")
    timeline = credit_engine.get_migration_timeline(selected_scenario)
    df_migration = pd.DataFrame(timeline)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("**Portfolio Grade Distribution Over Time**")
        st.dataframe(df_migration.style.format(
            {g: "{:.2%}" for g in RATING_GRADES}
        ), use_container_width=True)

    with col_b:
        fig_mig = go.Figure()
        for grade in RATING_GRADES:
            fig_mig.add_trace(go.Scatter(
                x=df_migration["Year"],
                y=df_migration[grade],
                mode="lines+markers",
                name=grade,
                stackgroup="one",
            ))
        fig_mig.update_layout(
            title=f"Rating Migration — {selected_scenario.name}",
            yaxis_title="Portfolio Share",
            yaxis_tickformat=".0%",
            height=400,
        )
        st.plotly_chart(fig_mig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — Operational Risk
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("Operational Risk — Poisson–Lognormal VaR")

    op_result = op_engine.compute_var(n_simulations=50_000)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Loss", f"€{op_result.expected_loss:,.1f}m")
    col2.metric("VaR (99.5%)", f"€{op_result.var_99_5:,.1f}m")
    col3.metric("Expected Shortfall", f"€{op_result.expected_shortfall:,.1f}m")
    col4.metric("Capital Charge (UL)", f"€{op_result.capital_charge:,.1f}m")

    st.markdown("---")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Distribution Percentiles")
        perc_df = pd.DataFrame([
            {"Percentile": k, "Loss (€m)": v}
            for k, v in op_result.percentiles.items()
        ])
        st.dataframe(perc_df, use_container_width=True)

        st.markdown(f"""
        **Model Parameters:**
        - Frequency: Poisson(λ={op_result.mean_frequency})
        - Severity: LogNormal(μ=12.0, σ=2.0)
        - Mean Severity: €{op_result.mean_severity:,.1f}m
        - Simulations: {op_result.n_simulations:,}
        """)

    with col_b:
        st.subheader("Aggregate Loss Distribution")
        counts, edges = op_engine.get_loss_distribution_histogram(n_bins=80)
        midpoints = (edges[:-1] + edges[1:]) / 2
        fig_op = go.Figure()
        fig_op.add_trace(go.Bar(
            x=midpoints, y=counts,
            marker_color="#5c6bc0",
            name="Frequency",
        ))
        fig_op.add_vline(x=op_result.expected_loss, line_dash="dash",
                          line_color="green", annotation_text="EL")
        fig_op.add_vline(x=op_result.var_99_5, line_dash="dash",
                          line_color="red", annotation_text="VaR 99.5%")
        fig_op.update_layout(
            title="Operational Risk — Aggregate Annual Loss Distribution",
            xaxis_title="Annual Loss (€m)",
            yaxis_title="Frequency",
            height=400,
        )
        st.plotly_chart(fig_op, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — Monte Carlo Stress Testing
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("Monte Carlo Stress Testing")

    if st.button("🚀 Run Stress Test", type="primary"):
        with st.spinner(f"Running {mc_paths:,} paths across 3 scenarios..."):
            mc_engine = MonteCarloStressEngine(bs=bs, seed=42)
            results = mc_engine.run_all_scenarios(n_paths=mc_paths)
            st.session_state["mc_results"] = results

    if "mc_results" in st.session_state:
        results = st.session_state["mc_results"]

        for res in results:
            st.subheader(f"📊 {res.scenario_name}")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("CET1 Mean", f"{res.cet1_mean:.1%}")
            col2.metric("CET1 P5", f"{res.cet1_p5:.1%}",
                        delta="❌" if res.cet1_p5 < MIN_CET1_RATIO else "✅")
            col3.metric("P(CET1 Breach)", f"{res.cet1_breach_probability:.1%}")
            col4.metric("LCR Mean", f"{res.lcr_mean:.0%}")
            col5.metric("RAROC Mean", f"{res.raroc_mean:.1%}")

            col_a, col_b = st.columns(2)
            with col_a:
                fig_hist = px.histogram(
                    x=res.all_terminal_cet1,
                    nbins=50,
                    title=f"Terminal CET1 Distribution — {res.scenario_name}",
                    labels={"x": "CET1 Ratio"},
                    color_discrete_sequence=["#3949ab"],
                )
                fig_hist.add_vline(x=MIN_CET1_RATIO, line_dash="dash",
                                    line_color="red", annotation_text="Min 12%")
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_b:
                fig_lcr_hist = px.histogram(
                    x=[min(v, 5.0) for v in res.all_terminal_lcr],
                    nbins=50,
                    title=f"Terminal LCR Distribution — {res.scenario_name}",
                    labels={"x": "LCR Ratio"},
                    color_discrete_sequence=["#00897b"],
                )
                fig_lcr_hist.add_vline(x=MIN_LCR_RATIO, line_dash="dash",
                                        line_color="red", annotation_text="Min 110%")
                fig_lcr_hist.update_layout(height=300)
                st.plotly_chart(fig_lcr_hist, use_container_width=True)

            # CET1 trajectory
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=list(range(len(res.yearly_cet1_mean))),
                y=[v * 100 for v in res.yearly_cet1_mean],
                mode="lines+markers",
                name="Mean CET1",
                line=dict(width=3),
            ))
            fig_traj.add_hline(y=MIN_CET1_RATIO * 100, line_dash="dash",
                                line_color="red")
            fig_traj.update_layout(
                title=f"CET1 Trajectory (Mean) — {res.scenario_name}",
                xaxis_title="Year", yaxis_title="CET1 (%)",
                height=300,
            )
            st.plotly_chart(fig_traj, use_container_width=True)
            st.markdown("---")
    else:
        st.info("Click **Run Stress Test** to execute Monte Carlo simulations.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — RAROC Capital Optimisation
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.header("RAROC Capital Optimisation")
    st.markdown(r"""
    $$RAROC = \frac{\text{Expected Revenue} - \text{OpCosts} - \text{Expected Loss (EL)}}{\text{Economic Capital}}$$
    **Objective:** Maximise portfolio RAROC subject to CET1 ≥ 12% and LCR ≥ 110%.
    """)

    if st.button("🎯 Run Optimisation", type="primary"):
        with st.spinner("Optimising capital allocation..."):
            optimizer = CapitalOptimizer(bs)
            opt_result = optimizer.optimize()
            st.session_state["opt_result"] = opt_result

    if "opt_result" in st.session_state:
        opt = st.session_state["opt_result"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Initial RAROC", f"{opt.initial_raroc:.2%}")
        col2.metric("Optimal RAROC", f"{opt.optimal_raroc:.2%}",
                    delta=f"+{(opt.optimal_raroc - opt.initial_raroc)*100:.1f}pp")
        col3.metric("Optimal CET1", f"{opt.optimal_cet1:.1%}")
        col4.metric("Optimal LCR", f"{opt.optimal_lcr:.1%}")

        st.success(f"✅ Optimisation {'converged' if opt.success else 'failed'}: {opt.convergence_message}")

        st.markdown("---")

        # Allocation comparison
        st.subheader("Optimal vs Current Allocation")
        alloc_df = pd.DataFrame({
            "Segment": list(opt.initial_weights.keys()),
            "Current (%)": [v * 100 for v in opt.initial_weights.values()],
            "Optimal (%)": [v * 100 for v in opt.optimal_weights.values()],
        })
        alloc_df["Δ (pp)"] = alloc_df["Optimal (%)"] - alloc_df["Current (%)"]
        st.dataframe(alloc_df.style.format({
            "Current (%)": "{:.1f}%",
            "Optimal (%)": "{:.1f}%",
            "Δ (pp)": "{:+.1f}pp",
        }).background_gradient(subset=["Δ (pp)"], cmap="RdYlGn"),
        use_container_width=True)

        # Side-by-side pie charts
        col_a, col_b = st.columns(2)
        with col_a:
            fig_cur = px.pie(
                names=list(opt.initial_weights.keys()),
                values=list(opt.initial_weights.values()),
                title="Current Allocation",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.3,
            )
            fig_cur.update_layout(height=350)
            st.plotly_chart(fig_cur, use_container_width=True)

        with col_b:
            fig_opt = px.pie(
                names=list(opt.optimal_weights.keys()),
                values=list(opt.optimal_weights.values()),
                title="Optimal Allocation",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.3,
            )
            fig_opt.update_layout(height=350)
            st.plotly_chart(fig_opt, use_container_width=True)

        # Segment RAROC
        st.subheader("Segment-Level RAROC")
        calc = RAROCCalculator(bs)
        opt_weights = np.array(list(opt.optimal_weights.values()))
        raroc_result = calc.compute(opt_weights)
        seg_df = pd.DataFrame([
            {"Segment": k, "RAROC": f"{v:.2%}"}
            for k, v in raroc_result.segment_rarocs.items()
        ])
        st.dataframe(seg_df, use_container_width=True)

        # Sensitivity
        if "cet1_sensitivity" in opt.sensitivity:
            st.subheader("Sensitivity — RAROC vs CET1 Target")
            sens_df = pd.DataFrame(opt.sensitivity["cet1_sensitivity"])
            st.dataframe(sens_df, use_container_width=True)

    else:
        st.info("Click **Run Optimisation** to find the optimal capital allocation.")


# ══════════════════════════════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <strong>Integrated Bank Balance Sheet Simulator &amp; Capital Optimizer (ICAAP/ILAAP)</strong><br>
    Risk 360 Framework | Basel III/IV Compliant | Python OOP Architecture<br>
    Built with Streamlit • NumPy • SciPy • Plotly
</div>
""", unsafe_allow_html=True)

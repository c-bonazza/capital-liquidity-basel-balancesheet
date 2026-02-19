"""Configuration and macroeconomic scenario definitions."""

from dataclasses import dataclass, field
from typing import Dict, List
import copy


# ── Regulatory thresholds ────────────────────────────────────────────────────
MIN_CET1_RATIO = 0.12          # 12 % internal target (ICAAP)
MIN_LCR_RATIO = 1.10           # 110 % internal target (ILAAP)
PROJECTION_HORIZON_YEARS = 3   # 3-year forward projection
CONFIDENCE_LEVEL = 0.995       # 99.5 % VaR for economic capital


# ── Basel III / IV Run-off & Inflow Coefficients ─────────────────────────────
RUN_OFF_RATES = {
    "retail_stable": 0.05,     # Stable retail deposits
    "retail_less_stable": 0.10,
    "wholesale_operational": 0.25,
    "wholesale_non_operational": 0.40,
    "wholesale_unsecured": 0.75,
}

INFLOW_RATES = {
    "performing_loans": 0.50,
    "interbank_loans": 1.00,
}

# ── HQLA Haircuts ────────────────────────────────────────────────────────────
HQLA_HAIRCUTS = {
    "level_1": 0.00,   # Cash, central bank reserves, sovereigns 0 % RW
    "level_2a": 0.15,   # Sovereigns 20 % RW, covered bonds AA-
    "level_2b": 0.50,   # Corporate bonds A+ to BBB-, RMBS
}


# ── IFRS 9 PD / LGD Assumptions ─────────────────────────────────────────────
@dataclass
class CreditSegmentParams:
    """Parameters for a single credit segment."""
    name: str
    pd_12m: float          # 12-month Point-in-Time PD
    lgd: float             # Loss-Given-Default
    ead_share: float       # Share of total loan book EAD
    avg_maturity: float    # Average residual maturity (years)
    risk_weight: float     # Standardised RW (fallback)


DEFAULT_CREDIT_SEGMENTS = [
    CreditSegmentParams("Residential Mortgages", 0.008, 0.15, 0.45, 18.0, 0.35),
    CreditSegmentParams("SME Loans",             0.025, 0.40, 0.30, 4.5,  0.75),
    CreditSegmentParams("Consumer Credit",       0.035, 0.55, 0.15, 2.5,  0.75),
    CreditSegmentParams("Corporate Loans",       0.012, 0.35, 0.10, 5.0,  0.65),
]


# ── Transition Matrix (annual, 4 rating grades + Default) ────────────────────
# Rows = current grade, Cols = next grade  (AAA, A, BBB, Sub-IG, Default)
DEFAULT_TRANSITION_MATRIX = [
    # AAA     A       BBB     Sub-IG  Default
    [0.9200, 0.0600, 0.0150, 0.0040, 0.0010],   # AAA
    [0.0100, 0.9000, 0.0650, 0.0200, 0.0050],   # A
    [0.0020, 0.0300, 0.8800, 0.0700, 0.0180],   # BBB
    [0.0005, 0.0050, 0.0500, 0.8800, 0.0645],   # Sub-IG
    [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],   # Default (absorbing)
]

RATING_GRADES = ["AAA", "A", "BBB", "Sub-IG", "Default"]


# ── Operational Risk Assumptions ─────────────────────────────────────────────
@dataclass
class OpRiskParams:
    """Parameters for the Poisson–Lognormal operational risk model."""
    lambda_events_per_year: float = 5.0     # Average event frequency
    mu_log_severity: float = 12.0           # ln(severity) mean  ≈ €163k
    sigma_log_severity: float = 2.0         # ln(severity) std
    confidence: float = CONFIDENCE_LEVEL


# ── Macroeconomic Scenarios ──────────────────────────────────────────────────
@dataclass
class MacroScenario:
    """Multi-year macroeconomic scenario for stress testing."""
    name: str
    gdp_growth: List[float]         # Annual real GDP growth (%)
    interest_rate_shock_bps: List[float]  # Cumulative parallel shock (bps)
    unemployment_rate: List[float]   # Unemployment rate (%)
    house_price_change: List[float]  # Annual HPI change (%)
    pd_multiplier: List[float]       # Multiplier on through-the-cycle PDs
    description: str = ""


BASE_SCENARIO = MacroScenario(
    name="Baseline",
    gdp_growth=[1.8, 2.0, 2.1],
    interest_rate_shock_bps=[0, 0, 0],
    unemployment_rate=[6.5, 6.3, 6.1],
    house_price_change=[2.0, 2.5, 3.0],
    pd_multiplier=[1.0, 1.0, 1.0],
    description="Central economic projection — benign macro environment.",
)

ADVERSE_SCENARIO = MacroScenario(
    name="Adverse",
    gdp_growth=[-1.5, 0.2, 1.0],
    interest_rate_shock_bps=[200, 250, 200],
    unemployment_rate=[9.0, 10.5, 9.8],
    house_price_change=[-8.0, -12.0, -5.0],
    pd_multiplier=[2.0, 3.0, 2.5],
    description="Severe recession with rate hikes — EBA-style adverse.",
)

SEVERELY_ADVERSE_SCENARIO = MacroScenario(
    name="Severely Adverse",
    gdp_growth=[-4.0, -1.5, 0.5],
    interest_rate_shock_bps=[400, 450, 350],
    unemployment_rate=[12.0, 14.0, 13.0],
    house_price_change=[-20.0, -25.0, -10.0],
    pd_multiplier=[3.5, 5.0, 4.0],
    description="Systemic crisis — combined rate & credit shock.",
)

ALL_SCENARIOS = [BASE_SCENARIO, ADVERSE_SCENARIO, SEVERELY_ADVERSE_SCENARIO]


# ── Initial Synthetic Balance Sheet (€ millions) ─────────────────────────────
@dataclass
class InitialBalanceSheet:
    """Synthetic bank balance sheet used as t=0 starting point."""
    # ASSETS
    cash_and_reserves: float = 2_000.0
    govt_bonds_hqla_l1: float = 5_000.0
    covered_bonds_hqla_l2a: float = 2_000.0
    corporate_bonds_hqla_l2b: float = 1_000.0
    residential_mortgages: float = 25_000.0
    sme_loans: float = 12_000.0
    consumer_credit: float = 6_000.0
    corporate_loans: float = 8_000.0
    trading_book: float = 3_000.0
    other_assets: float = 2_000.0

    # LIABILITIES
    retail_deposits_stable: float = 22_000.0
    retail_deposits_less_stable: float = 8_000.0
    wholesale_operational: float = 6_000.0
    wholesale_non_operational: float = 5_000.0
    wholesale_unsecured: float = 4_000.0
    subordinated_debt: float = 2_000.0
    other_liabilities: float = 11_000.0  # Interbank, repo, derivatives, accruals

    # EQUITY
    cet1_capital: float = 5_500.0
    at1_capital: float = 500.0
    tier2_capital: float = 2_000.0

    # INCOME STATEMENT ASSUMPTIONS (annual, €m)
    net_interest_income: float = 1_800.0
    fee_income: float = 400.0
    trading_income: float = 200.0
    operating_costs: float = -1_400.0

    @property
    def total_assets(self) -> float:
        return (self.cash_and_reserves + self.govt_bonds_hqla_l1 +
                self.covered_bonds_hqla_l2a + self.corporate_bonds_hqla_l2b +
                self.residential_mortgages + self.sme_loans +
                self.consumer_credit + self.corporate_loans +
                self.trading_book + self.other_assets)

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
        return (self.cash_and_reserves +
                self.govt_bonds_hqla_l1 * (1 - HQLA_HAIRCUTS["level_1"]) +
                self.covered_bonds_hqla_l2a * (1 - HQLA_HAIRCUTS["level_2a"]) +
                self.corporate_bonds_hqla_l2b * (1 - HQLA_HAIRCUTS["level_2b"]))

    @property
    def total_loan_book(self) -> float:
        return (self.residential_mortgages + self.sme_loans +
                self.consumer_credit + self.corporate_loans)

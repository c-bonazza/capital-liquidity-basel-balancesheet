# 🏦 Integrated bank balance sheet simulator & Capital management solutions (ICAAP/ILAAP)

> **ICAAP / ILAAP Integration — Risk 360 Framework**

A comprehensive Python simulation engine that models the financial health of a bank over a 3-year horizon under multiple macroeconomic stress scenarios. Integrates all four risk pillars (Market, Credit, Liquidity, Operational) into a unified balance sheet projection with capital optimization capabilities.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![Basel III/IV](https://img.shields.io/badge/Basel-III%2FIV-green.svg)](#regulatory-framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


```
┌──────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT EXECUTIVE DASHBOARD                      │
│   Balance Sheet │ ALM │ Liquidity │ Credit │ OpRisk │ Stress │ RAROC │
        ext{PD}_{\text{cum}}(i, n) = \left[\mathbf{M}^n\right]_{i, \text{Default}}
            │
    	ext{PD}_{\text{cum}}(i, n) = \left[\mathbf{M}^n\right]_{i, \text{Default}}
│                   MONTE CARLO STRESS ENGINE                          │
│         Correlated Macro Variables × N Paths × 3 Years               │
└───────────┬──────────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────────┐
│                    BALANCE SHEET PROJECTOR                            │
│          BalanceSheetSnapshot → Project → [t₀, t₁, t₂, t₃]         │
└──┬──────────┬──────────┬──────────┬──────────────────────────────────┘
   │          │          │          │
┌──▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼─────┐  ┌─────────────────────┐
│Market│ │Liquid.│ │Credit │ │ OpRisk  │  │  RAROC OPTIMIZER    │
│ Risk │ │ Risk  │ │ Risk  │ │         │  │  (scipy.optimize)   │
│ ALM  │ │ ILAAP │ │ IFRS9 │ │Poisson- │  │  max RAROC s.t.     │
│IRRBB │ │ LCR   │ │ RWA   │ │LogNormal│  │  CET1≥12%, LCR≥110%│
└──────┘ └───────┘ └───────┘ └─────────┘  └─────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────────┐
│                    SQL DATA MART (SQLite)                             │
│   DDL + Seed │ Complex CTEs │ Analytical Views │ 200 Facilities      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Risk Pillars — Detailed Methodology

### A. Market Risk & ALM (IRRBB)

Models Interest Rate Risk in the Banking Book per **BCBS 368** standards.

**Repricing Gap Analysis:**

Assets and liabilities are allocated into maturity buckets. The gap for bucket $i$ is:

$$\text{Gap}_i = \text{RSA}_i - \text{RSL}_i$$

**Modified Duration of Equity:**

$$D_{\text{mod}} = \frac{\sum_{i} \text{Gap}_i \times t_i}{\text{Equity} \times (1 + y)}$$

where $t_i$ is the bucket midpoint and $y$ the current yield level.

**Economic Value of Equity (EVE) Sensitivity:**

$$\Delta\text{EVE} = -D_{\text{mod}} \times \text{Equity} \times \Delta r$$

**NII Sensitivity (1-year horizon):**

$$\Delta\text{NII} = \sum_{i: t_i \leq 1} \text{Gap}_i \times \Delta r$$

The six Basel IRRBB supervisory scenarios are computed: Parallel Up/Down (±200bp), Short Up/Down, Steepener, Flattener.

---

### B. Liquidity Risk (ILAAP)

Implements the **Liquidity Coverage Ratio** per Basel III (BCBS 238):

$$LCR = \frac{\text{HQLA}}{\text{Net Cash Outflows}_{30d}} \geq 100\%$$

**HQLA Computation (with haircuts and caps):**

$$\text{HQLA} = L_1 + \min(L_{2A} \times 0.85, \, 0.40 \times \text{Total}) + \min(L_{2B} \times 0.50, \, 0.15 \times \text{Total})$$

**Run-off Rates (Basel III):**

| Deposit Category | Run-off Rate |
|:--|--:|
| Retail Stable | 5% |
| Retail Less Stable | 10% |
| Wholesale Operational | 25% |
| Wholesale Non-Operational | 40% |
| Wholesale Unsecured | 75% |

**Bank Run Simulation:**

Simulates accelerating deposit flight with daily outflows:

$$\text{Outflow}_d = \text{Deposits}_d \times r_d, \quad r_{d+1} = r_d \times \alpha$$

where $\alpha > 1$ is the panic acceleration factor. Survival horizon = day HQLA reaches zero.

---

### C. Credit Risk (IFRS 9 & RWA)

**Expected Credit Loss by Stage:**

| Stage | ECL Formula | Trigger |
|:--|:--|:--|
| Stage 1 | $\text{ECL} = \text{PD}_{12m} \times \text{LGD} \times \text{EAD}$ | No significant deterioration |
| Stage 2 | $\text{ECL} = \text{PD}_{\text{lifetime}} \\times \text{LGD} \\times \text{EAD}$ | Significant increase in credit risk |
| Stage 3 | $\text{ECL} = \text{LGD} \\times \text{EAD}$ | Credit-impaired (default) |

**Lifetime PD via Transition Matrices:**

The cumulative default probability over $n$ years starting from rating grade $i$:


$$
    ext{PD}_{\text{cum}}(i, n) = \left[\mathbf{M}^n\right]_{i, \text{Default}}
$$

where $\mathbf{M}$ is the annual transition matrix (Markov chain). Under stress:

$$
\mathbf{M}_{\text{stressed}} = f(\mathbf{M}, \, PD_{\text{multiplier}})
$$

Downgrade probabilities are scaled by the macro-conditioned multiplier and rows are renormalised.

**Risk-Weighted Assets (Standardised Approach):**

$$\text{RWA} = \sum_s \text{EAD}_s \times w_s$$

where $w_s$ is the regulatory risk weight for segment $s$.

---

### D. Operational Risk (Operational VaR)

Uses the **Loss Distribution Approach (LDA)** with a compound Poisson–Lognormal model:

**Frequency:** $N \sim \text{Poisson}(\lambda)$ — number of loss events per year

**Severity:** $X_i \sim \text{LogNormal}(\mu, \sigma)$ — individual loss amount

**Aggregate Loss:** 

$$L = \sum_{i=1}^{N} X_i$$

The aggregate distribution is computed via **Monte Carlo convolution** (100,000 simulations):

$$\text{VaR}_{99.5\\%} = F_L^{-1}(0.995)$$

$$\text{ES}_{99.5\\%} = \mathbb{E}[L \,|\, L \geq \text{VaR}_{99.5\\%}]$$

**Capital Charge (Unexpected Loss):**

$$\text{UL} = \text{VaR}_{99.5\\%} - \mathbb{E}[L]$$

---

## 🎯 Capital Optimization (RAROC)

The optimizer answers: *"Given regulatory constraints, what is the optimal asset mix to maximise risk-adjusted returns?"*

**Objective Function:**

$$\text{RAROC} = \frac{\text{Expected Revenue} - \text{Operating Costs} - \text{Expected Loss (EL)}}{\text{Economic Capital}}$$

where Economic Capital $= \text{RWA} \times \text{CET1 target}$.

**Constrained Optimization Problem:**

$$\max_{\mathbf{w}} \quad \text{RAROC}(\mathbf{w})$$

$$\text{s.t.} \quad \frac{\text{CET1}}{\text{RWA}(\mathbf{w})} \geq 12\%$$

$$\frac{\text{HQLA}(\mathbf{w})}{\text{Net Outflows}} \geq 110\%$$

$$\sum_i w_i = 1, \quad w_i^{\min} \leq w_i \leq w_i^{\max}$$

Solved using **Sequential Least Squares Programming (SLSQP)** via `scipy.optimize.minimize`.

---

## 📊 Monte Carlo Stress Testing

The engine generates correlated macroeconomic paths using **Cholesky decomposition**:

$$\mathbf{Z} = \mathbf{L} \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

where $\mathbf{L}$ is the lower-triangular Cholesky factor of the correlation matrix between GDP growth, interest rates, unemployment, and house prices.

**Three predefined scenarios:**

| Scenario | GDP Y1 | Rate Shock | Unemployment | HPI Change |
|:--|--:|--:|--:|--:|
| Baseline | +1.8% | 0bp | 6.5% | +2.0% |
| Adverse | −1.5% | +200bp | 9.0% | −8.0% |
| Severely Adverse | −4.0% | +400bp | 12.0% | −20.0% |

Each path propagates through all risk pillars → balance sheet projection → terminal CET1/LCR/RAROC distributions.

**Key outputs:**
- Probability of CET1 breach ($P[\text{CET1} < 12\%]$)
- Terminal CET1 distribution (mean, P5, P50, P95)
- LCR survival analysis
- RAROC distribution under stress

---

## 🛠️ Technical Stack

| Component | Technology | Demonstrates |
|:--|:--|:--|
| **Data Ingestion** | SQL (Complex CTEs, Window Functions) | Structuring raw data into analytical Data Marts |
| **Logic Engine** | Python OOP (Classes, Inheritance, Dataclasses) | Clean software architecture & design patterns |
| **Stress Testing** | Monte Carlo + Cholesky Decomposition | Advanced statistics & probability modelling |
| **Optimization** | `scipy.optimize` (SLSQP) | Constrained nonlinear programming |
| **Reporting** | Streamlit + Plotly | Executive-grade interactive dashboards |
| **Documentation** | README with LaTeX (KaTeX) | Academic rigour & regulatory precision |

---

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bank-balance-sheet-simulator.git
cd bank-balance-sheet-simulator

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run bank_simulator/dashboard/app.py
```

### Run as Python Module

```python
from bank_simulator.config import InitialBalanceSheet, ADVERSE_SCENARIO
from bank_simulator.engine import BalanceSheetProjector
from bank_simulator.risk_modules import (
    MarketRiskALM, LiquidityRiskEngine,
    CreditRiskEngine, OperationalRiskEngine,
)
from bank_simulator.optimization import run_full_optimization
from bank_simulator.stress_testing import run_stress_test
$$
	ext{RWA} = \sum_s \text{EAD}_s \\times w_s
# Initialize balance sheet
bs = InitialBalanceSheet()

# Run individual risk modules
market = MarketRiskALM(bs)
print(f"Modified Duration: {market.compute_modified_duration_equity():.2f}")
print(f"Δ-EVE (+200bp): €{market.compute_eve_delta(200):,.1f}m")

credit = CreditRiskEngine(bs)
print(f"Total ECL (baseline): €{credit.compute_total_ecl():,.1f}m")

# Run capital optimization
opt = run_full_optimization(bs)
print(f"Optimal RAROC: {opt.optimal_raroc:.2%}")

# Run full Monte Carlo stress test
results = run_stress_test(n_paths=1000)
for r in results:
    print(f"{r.scenario_name}: CET1 mean={r.cet1_mean:.1%}, "
          f"P(breach)={r.cet1_breach_probability:.1%}")
```

### Run Tests

```bash
python -m pytest bank_simulator/tests/ -v
```

---

## 📁 Project Structure

```
bank_simulator/
├── __init__.py                    # Package metadata
├── config.py                      # Configuration, scenarios, regulatory params
├── data/
│   ├── __init__.py                # DataMart class (SQL ingestion layer)
│   └── sql/
│       └── 001_schema_and_seed.sql  # DDL, seed data, complex CTEs & views
├── engine/
│   ├── __init__.py
│   └── balance_sheet.py           # Core BS snapshot & projector (OOP)
├── risk_modules/
│   ├── __init__.py
│   ├── market_risk.py             # IRRBB: Gap analysis, Duration, EVE/NII
│   ├── liquidity_risk.py          # ILAAP: LCR, bank-run simulation
│   ├── credit_risk.py             # IFRS 9: ECL stages, transition matrices, RWA
│   └── operational_risk.py        # VaR: Poisson–Lognormal LDA
├── optimization/
│   └── __init__.py                # RAROC optimizer (scipy.optimize)
├── stress_testing/
│   └── __init__.py                # Monte Carlo engine (Cholesky, multi-pillar)
├── dashboard/
│   ├── __init__.py
│   └── app.py                     # Streamlit 7-tab executive dashboard
├── utils/
│   └── __init__.py                # Formatting helpers
└── tests/
    └── test_engine.py             # Comprehensive test suite
```

---

## 📚 Regulatory References

- **Basel III (BCBS 238)** — Liquidity Coverage Ratio
- **Basel III (BCBS 295)** — Net Stable Funding Ratio
- **BCBS 368** — Interest Rate Risk in the Banking Book (IRRBB)
- **Basel III/IV** — Standardised Approach for Credit Risk (SA-CR)
- **IFRS 9** — Financial Instruments (Expected Credit Loss model)
- **EBA GL/2017/06** — Guidelines on IFRS 9 implementation
- **CRR/CRD V** — Capital Requirements Regulation (EU)
- **ICAAP** — Internal Capital Adequacy Assessment Process
- **ILAAP** — Internal Liquidity Adequacy Assessment Process

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a demonstration of integrated bank risk management capabilities — from individual risk pillar modelling to capital adequacy optimisation.*

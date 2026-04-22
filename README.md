# QuantRisk

A Python library for quantitative risk measurement, portfolio construction, options pricing, and statistical modeling.

## Overview

QuantRisk provides a robust suite of tools designed for quantitative analysts, risk managers, and financial engineers. The repository is structured into four core modules, each adhering to rigorous engineering and mathematical standards.

## Modules

### 1. `Options` (Options Pricing Module)

Provides pricing and risk metrics (Greeks) for European and American options.

* **European Options**: Generalized Black-Scholes-Merton (GBSM) model.
* **American Options**: Binomial Tree models, supporting both continuous dividend yields (cost of carry) and discrete escrowed dividends.
* **Key Functions**: `calculate_gbsm`, `calculate_american_metrics`, `calculate_american_discrete_metrics`.

### 2. `Portfolio` (Portfolio Construction and Attribution Module)

Contains portfolio optimization tools and ex-post performance tracking.

* **Optimization**: Parametric and non-parametric Risk Parity (Risk Parity and Custom Risk Budgeting), and Maximum Sharpe Ratio optimization with support for custom weight boundaries.
* **Attribution**: Ex-post return and risk attribution tracking, including factor model attribution and return linking using Cariño K.
* **Key Functions**: `risk_parity_normal`, `risk_parity_es`, `max_sharpe_ratio_normal`, `expost_attribution`, `expost_factor_attribution`.

### 3. `Risk` (Risk Measurement Module)

Implements risk measurement metrics.

* **Value at Risk (VaR)**: Parametric (Normal and Student's t).
* **Expected Shortfall (ES / CVaR)**: Parametric (Normal and Student's t).
* **Simulation**: N-asset Copula-based VaR and ES simulations.
* **Key Functions**: `calculate_normal_var`, `calculate_t_var`, `calculate_normal_es`, `calculate_t_es`, `calculate_copula_var_es`.

### 4. `Stats` (Statistics and Simulation Module)

Core mathematical utilities for quantitative modeling.

* **Time Series**: Simulation of MA, AR, ARMA, ARIMA, GARCH, GBM, and Ornstein-Uhlenbeck (OU) processes.
* **Normal Simulation**: Monte Carlo simulation of Multivariate Normal distributions with fallback SVD handling for non-PD matrices, plus matrix comparison utilities.
* **Covariance & Correlation**: Exponentially Weighted Moving Average (EWMA) covariance matrices.
* **Matrix Fixes**: Positive Semi-Definite (PSD) corrections using Higham's nearest correlation, Rebonato-Jackel, and customized robust Cholesky decomposition.
* **PCA**: Principal Component Analysis (exponentially weighted and standard).
* **Regression**: Linear regression with t-distributed errors.

## Installation

Ensure you have Python 3.8+ installed. Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/samfan-27/QuantRisk.git](https://github.com/samfan-27/QuantRisk.git)
cd QuantRisk
pip install -r requirements.txt
```

*Core Dependencies:* `numpy`, `pandas`, `scipy`, `scikit-learn`.

## Quick Start / Usage Examples

> **Note on Documentation:** Every function in this repository is thoroughly documented. Please read the docstring under each function (e.g., using `help(calculate_gbsm)` or by hovering over the function in your IDE) to understand its specific functionality, required inputs, and expected outputs.

The library is designed to easily integrate with standard data workflows using `pandas`. Below are examples of how to load CSV data and run the core modules.

**1. Calculating Value at Risk (VaR) from Historical Returns:**

```python
import pandas as pd
from Risk import calculate_normal_var, calculate_t_var

df = pd.read_csv('data/returns.csv')
returns_series = df['Daily_Returns']

norm_var = calculate_normal_var(returns_series, alpha=0.05)
print(f"Normal VaR (95%): {norm_var['VaR_Absolute']:.6f}")

t_var = calculate_t_var(returns_series, alpha=0.05)
print(f"Student's t VaR (95%): {t_var['VaR_Absolute']:.6f}")
```

**2. Portfolio Optimization (Risk Parity):**

```python
import pandas as pd
from Portfolio import risk_parity_normal

cov_df = pd.read_csv('data/asset_covariance.csv')

rp_weights = risk_parity_normal(cov_df)
print("Equal Risk Parity Weights:\n", rp_weights)

custom_budget = [1.0, 1.0, 1.0, 1.0, 0.5]
custom_rp_weights = risk_parity_normal(cov_df, risk_budget=custom_budget)
print("\nCustom Risk Budget Weights:\n", custom_rp_weights)
```

**3. Pricing a Batch of European Options:**

```python
import pandas as pd
from Options import calculate_gbsm

options_df = pd.read_csv('data/options_chain.csv')

for index, row in options_df.iterrows():
    T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    
    metrics = calculate_gbsm(
        S=float(row['Underlying']),
        K=float(row['Strike']),
        T=T,
        r=float(row['RiskFreeRate']),
        q=float(row['DividendRate']),
        sigma=float(row['ImpliedVol']),
        opt_type=str(row['Option Type'])
    )
    
    print(f"Option ID {row['ID']} ({row['Option Type']}) Value:", metrics['Value'])
```

## Running Built-in Tests

Many of the modules include built-in test blocks (`if __name__ == '__main__':`) that rely on sample CSV data. To run these tests directly, ensure that you have the appropriate sample data mapped in a `data/` directory at the root of the project (e.g., `../data/test12_1.csv` relative to the module files).

## Acknowledgements

Special thanks to **Professor Dominic Pazzula** for his guidance, insights, and foundational teachings that helped make this repository possible.

## License

This project is licensed under the MIT License.

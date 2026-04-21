"""
QuantRisk: Portfolio Construction and Attribution Module
--------------------------------------------------------
This module contains portfolio optimization tools, 
including parametric and non-parametric Risk Parity, Maximum Sharpe Ratio 
optimization, and ex-post return and risk attribution tracking.
"""

# Return Linking
from .return_linking import (
    calculate_carino_k
)

# Ex-Post Attribution
from .expost_attribution import (
    expost_attribution
)
from .expost_attribution_factors import (
    expost_factor_attribution
)

# Risk Parity & Budgeting
from .risk_parity import (
    risk_parity_normal,
    risk_parity_es,
    optimize_risk_parity,
    component_volatility,
    component_expected_shortfall
)

# Maximum Sharpe Optimization
from .max_sharpe_optimization import (
    calculate_sharpe_ratio,
    max_sharpe_ratio_normal
)

__all__ = [
    # Return Linking
    "calculate_carino_k",
    
    # Ex-Post Attribution
    "expost_attribution",
    "expost_factor_attribution",
    
    # Risk Parity & Budgeting
    "risk_parity_normal",
    "risk_parity_es",
    "optimize_risk_parity",
    "component_volatility",
    "component_expected_shortfall",
    
    # Maximum Sharpe Optimization
    "calculate_sharpe_ratio",
    "max_sharpe_ratio_normal"
]

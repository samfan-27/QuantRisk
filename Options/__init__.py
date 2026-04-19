"""
QuantRisk: Options Pricing Module
---------------------------------
This module provides industry-standard pricing and risk metrics (Greeks)
for European and American options. It includes the Generalized Black-Scholes-Merton (GBSM)
model for European options, and Cox-Ross-Rubinstein Binomial Tree models 
(with continuous yields and discrete escrowed dividends) for American options.
"""

# European Options (GBSM)
from .european_models import (
    calculate_gbsm
)

# American Options (Continuous Dividend Yield / Cost of Carry)
from .american_continuous_models import (
    calculate_american_metrics
)

# American Options (Discrete Escrowed Dividends)
from .american_discrete_models import (
    calculate_american_discrete_metrics
)

__all__ = [
    # European Pricing
    "calculate_gbsm",
    
    # American Pricing
    "calculate_american_metrics",
    "calculate_american_discrete_metrics"
]

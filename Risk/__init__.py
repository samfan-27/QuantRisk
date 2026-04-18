"""
QuantRisk: Risk Measurement Module
----------------------------------
This module contains industry-standard risk measurement tools, including 
parametric and non-parametric implementations of VaR, 
ES/CVaR, and N-asset Copula simulations.
"""

# VaR
from .var_models import (
    calculate_normal_var,
    calculate_t_var
)

# ES
from .es_models import (
    calculate_normal_es,
    calculate_t_es
)

# Copula-based VaR & ES Simulation
from .copula_var_es_simulation import (
    calculate_copula_var_es
)

__all__ = [
    # VaR
    "calculate_normal_var",
    "calculate_t_var",
    
    # ES
    "calculate_normal_es",
    "calculate_t_es",
    
    # Copulas
    "calculate_copula_var_es"
]

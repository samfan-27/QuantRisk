"""
QuantRisk: Statistics and Simulation Module
-------------------------------------------
This module contains core mathematical utilities for covariance matrix 
estimation, positive semi-definite (PSD) corrections, eigenvalue decomposition, 
and stochastic time-series simulations, etc.
"""

# Time Series & Stochastic Simulations
from .time_series_simulation import (
    simulate_ma,
    simulate_ar,
    simulate_arma,
    simulate_arima,
    simulate_garch,
    simulate_gbm,
    simulate_ou
)

# Multivariate Normal Simulation
from .normal_simulation import (
    simulate_multivariate_normal,
    compare_matrices
)

# Regression Models
from .fit_t_regression import (
    fit_t_regression
)

# Covariance & Correlation (EWMA)
from .ewma_covariance_correlation import (
    calculate_ew_covariance_matrix,
    calculate_mixed_ew_covariance
)

# Matrix Factorization & PSD Fixes
from .customized_cholesky import (
    chol_psd,
    simulate_with_cholesky
)
from .psd_rebonato_jackel import (
    near_psd
)
from .psd_higham import (
    higham_nearest_correlation
)

# Principal Component Analysis (PCA)
from .pca import (
    exp_weighted_pca,
    sklearn_weighted_pca,
    simulate_pca
)

__all__ = [
    # Time Series
    "simulate_ma",
    "simulate_ar",
    "simulate_arma",
    "simulate_arima",
    "simulate_garch",
    "simulate_gbm",
    "simulate_ou",
    
    # Normal Simulation
    "simulate_multivariate_normal",
    "compare_matrices",
    
    # Regression
    "fit_t_regression",
    
    # Covariance/Correlation
    "calculate_ew_covariance_matrix",
    "calculate_mixed_ew_covariance",
    
    # Matrix Fixes & Cholesky
    "chol_psd",
    "simulate_with_cholesky",
    "near_psd",
    "higham_nearest_correlation",
    
    # PCA
    "exp_weighted_pca",
    "sklearn_weighted_pca",
    "simulate_pca"
]

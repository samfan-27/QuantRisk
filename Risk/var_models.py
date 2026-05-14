import numpy as np
import pandas as pd
from scipy.stats import norm, t
import warnings
from typing import Callable, Optional, Union

from ..Stats.normal_simulation import simulate_multivariate_normal
from ..Stats.ewma_covariance_correlation import (
    calculate_mixed_ew_covariance,
    _calculate_ew_weights
)
from ..Stats.psd_higham import fix_non_psd_covariance

def _sanitize_returns(returns):
    """
    Validates and extracts a 1D array from the input returns data.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError('Input must be a single series of returns, not a multi-column DataFrame.')
        returns = returns.iloc[:, 0]
        
    returns = np.asarray(returns)
    if returns.size == 0:
        raise ValueError('Input returns cannot be empty.')
        
    return returns

def _calculate_weighted_percentile(data: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """
    Calculates the empirical weighted percentile for Weighted Historical Simulation (WHS).
    Uses linear interpolation on the empirical cumulative distribution function.
    """
    sort_indices = np.argsort(data)
    sorted_data = data[sort_indices]
    sorted_weights = weights[sort_indices]
    
    cumulative_weights = np.cumsum(sorted_weights)
    cutoff_idx = np.searchsorted(cumulative_weights, alpha)
    
    if cutoff_idx == 0:
        return sorted_data[0]
    if cutoff_idx >= len(sorted_data):
        return sorted_data[-1]
        
    weight_lower = cumulative_weights[cutoff_idx - 1]
    weight_upper = cumulative_weights[cutoff_idx]
    data_lower = sorted_data[cutoff_idx - 1]
    data_upper = sorted_data[cutoff_idx]
    
    fraction = (alpha - weight_lower) / (weight_upper - weight_lower)
    return float(data_lower + fraction * (data_upper - data_lower))

def _aggregate_returns_over_horizon(returns: np.ndarray, horizon: int, overlapping: bool = True) -> np.ndarray:
    """
    Aggregates daily returns into multi-day returns over a specified horizon.
    Uses log-returns for mathematically accurate and fast compounding.
    """
    if horizon <= 1:
        return returns
        
    df = pd.DataFrame(returns)
    log_ret = np.log1p(df)
    
    if overlapping:
        agg_log_ret = log_ret.rolling(window=horizon).sum().dropna()
    else:
        agg_log_ret = log_ret.groupby(np.arange(len(df)) // horizon).sum()
        if len(df) % horizon != 0:
            agg_log_ret = agg_log_ret.iloc[:-1]
            
    return np.expm1(agg_log_ret).values

# 1D Empirical / Parametric VaR Models
def calculate_normal_var(returns, alpha=0.05, horizon=1):
    """
    Calculates VaR using a Normal Distribution (Parametric).
    
    Parameters:
        returns (pd.Series or np.array): A 1D series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05).
        horizon (int): Time horizon in days (default 1).
        
    Returns:
        dict: Absolute VaR, Relative VaR, Scaled Mean, and Scaled Standard Deviation.
    """
    returns = _sanitize_returns(returns)
    
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)
    
    z_score = norm.ppf(alpha)
    var_cutoff = mu_h + (z_score * sigma_h)
    
    return {
        'VaR_Absolute': float(max(-var_cutoff, 0.0)),
        'VaR_Relative': float(mu_h - var_cutoff),
        'Mean': float(mu_h),
        'Std_Dev': float(sigma_h)
    }

def calculate_t_var(returns, alpha=0.05, horizon=1):
    """
    Calculates Value at Risk (VaR) by fitting a Student's t-distribution.
    
    Parameters:
        returns (pd.Series or np.array): A 1D series of asset or portfolio returns.
        alpha (float): The significance level.
        horizon (int): Time horizon in days.
        
    Returns:
        dict: Absolute VaR, Relative VaR, df, loc (mean), and scale (sigma).
    """
    returns = _sanitize_returns(returns)
    nu, mu, sigma = t.fit(returns)

    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)

    t_percentile = t.ppf(alpha, df=nu, loc=mu_h, scale=sigma_h)

    return {
        'VaR_Absolute': float(max(-t_percentile, 0.0)),
        'VaR_Relative': float(mu_h - t_percentile),
        'df': float(nu),
        'loc': float(mu_h),
        'scale': float(sigma_h)
    }

# Multi-Factor Parametric VaR (Delta-Normal)
def calculate_delta_normal_portfolio_var(
    weights, 
    cov_matrix, 
    deltas=None, 
    expected_returns=None, 
    alpha=0.05,
    horizon=1,
    portfolio_value=1.0,
    auto_fix_psd=True
):
    """
    Calculates Delta-Normal VaR for a multi-asset portfolio using Matrix Algebra.

    Parameters:
        weights (np.ndarray or list): Capital allocation weights for each asset.
        cov_matrix (np.ndarray or pd.DataFrame): The covariance matrix of the asset returns.
        deltas (np.ndarray or list, optional): Sensitivities of the assets. Defaults to 1.0 for all assets.
        expected_returns (np.ndarray or list, optional): Expected daily returns of the assets. Defaults to 0.0.
        alpha (float): The significance level (default 0.05).
        horizon (int): Time horizon in days (default 1).
        portfolio_value (float): Total monetary value of the portfolio (default 1.0).
        auto_fix_psd (bool): If True, applies Higham's algorithm to automatically fix non-PSD covariance matrices.

    Returns:
        dict: Absolute VaR, Relative VaR, Portfolio Mean, and Portfolio Standard Deviation (scaled by portfolio value).
    """
    weights = np.asarray(weights).flatten()
    cov_matrix = np.asarray(cov_matrix)
    n_assets = len(weights)
    
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(f'Covariance matrix shape mismatch with weights length.')
        
    if deltas is None:
        deltas = np.ones(n_assets)
    else:
        deltas = np.asarray(deltas).flatten()
            
    if expected_returns is None:
        expected_returns = np.zeros(n_assets)
    else:
        expected_returns = np.asarray(expected_returns).flatten()

    cov_h = cov_matrix * horizon
    exp_ret_h = expected_returns * horizon
    exposures = weights * deltas

    port_mean = np.dot(exposures, exp_ret_h)
    port_var = exposures.T @ cov_h @ exposures
    
    if port_var < 0:
        if port_var > -1e-10:
            port_var = 0.0
        elif auto_fix_psd:
            warnings.warn("Non-PSD Covariance detected (Variance < 0). Applying Higham's algo.")
            
            fixed_cov = fix_non_psd_covariance(cov_matrix)
            
            fixed_cov_h = fixed_cov * horizon
            port_var = exposures.T @ fixed_cov_h @ exposures
            
            port_var = max(port_var, 0.0)
        else:
            warnings.warn('Calculated portfolio variance is negative. Check PSD status. Setting to 0.')
            port_var = 0.0
        
    port_std = np.sqrt(port_var)
    z_score = norm.ppf(alpha) 
    
    var_cutoff = port_mean + (z_score * port_std)
    
    var_absolute = max(-var_cutoff, 0.0)
    var_relative = port_mean - var_cutoff 

    return {
        'VaR_Absolute': float(var_absolute * portfolio_value),
        'VaR_Relative': float(var_relative * portfolio_value),
        'Portfolio_Mean': float(port_mean),
        'Portfolio_Std_Dev': float(port_std)
    }

# Full Revaluation Simulation Models (Historical & Monte Carlo)
def calculate_historical_var(
    returns_matrix: Union[np.ndarray, pd.DataFrame], 
    weights: Optional[np.ndarray] = None, 
    portfolio_value: float = 1.0, 
    lambda_decay: Optional[float] = None, 
    alpha: float = 0.05,
    horizon: int = 1,
    overlapping: bool = True,
    pricing_func: Optional[Callable] = None
) -> dict:
    """
    Calculates Historical Value at Risk (VaR) using full revaluation.
    Supports standard historical simulation as well as Weighted Historical Simulation (WHS) via exponential decay.

    Parameters:
        returns_matrix (np.ndarray or pd.DataFrame): A 2D array or DataFrame of historical asset returns.
        weights (np.ndarray, optional): Capital allocation weights. Required if pricing_func is None.
        portfolio_value (float): Total monetary value of the portfolio (default 1.0).
        lambda_decay (float, optional): Decay parameter for EWMA weights (e.g., 0.94 or 0.99) to apply to historical scenarios. If None, uses equal weighting.
        alpha (float): The significance level (default 0.05).
        horizon (int): Time horizon in days (default 1).
        overlapping (bool): If True, aggregates multi-day returns using a rolling window. If False, uses non-overlapping blocks.
        pricing_func (Callable, optional): Custom P&L mapping function for non-linear portfolios or complex derivatives.

    Returns:
        dict: Absolute VaR, Relative VaR, Portfolio Mean, and Portfolio Standard Deviation.
    """
    returns = np.asarray(returns_matrix)
    
    if horizon > 1:
        returns = _aggregate_returns_over_horizon(returns, horizon, overlapping)
        
    t_periods, n_assets = returns.shape
    
    if pricing_func is not None:
        pnl_scenarios = pricing_func(returns)
    else:
        if weights is None:
            raise ValueError('Weights must be provided if no pricing_func is specified.')
        w = np.asarray(weights).flatten()
        exposures = w * portfolio_value
        pnl_scenarios = returns @ exposures

    # VaR Extraction
    port_mean = np.mean(pnl_scenarios)
    port_std = np.std(pnl_scenarios, ddof=1)
    
    if lambda_decay is not None:
        prob_weights = _calculate_ew_weights(t_periods, lambda_decay)[::-1]
        var_cutoff = _calculate_weighted_percentile(pnl_scenarios, prob_weights, alpha)
    else:
        var_cutoff = np.quantile(pnl_scenarios, alpha)
        
    return {
        'VaR_Absolute': float(max(-var_cutoff, 0.0)),
        'VaR_Relative': float(port_mean - var_cutoff),
        'Portfolio_Mean': float(port_mean),
        'Portfolio_Std_Dev': float(port_std)
    }

def calculate_mc_var(
    cov_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
    returns_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_ewma: bool = False,
    lambda_var: float = 0.97,
    lambda_corr: float = 0.94,
    weights: Optional[np.ndarray] = None, 
    portfolio_value: float = 1.0, 
    n_sims: int = 10000, 
    alpha: float = 0.05,
    horizon: int = 1,
    pricing_func: Optional[Callable] = None,
    seed: Optional[int] = None
) -> dict:
    """
    Calculates Monte Carlo VaR by generating scenarios from a covariance matrix.
    
    Parameters:
        cov_matrix: Static covariance matrix. Required if use_ewma=False.
        returns_matrix: Historical returns. Required if use_ewma=True.
        use_ewma (bool): If True, calculates EWMA covariance before simulation.
        lambda_var, lambda_corr: Decay parameters for EWMA covariance.
        weights: Capital allocation weights.
        portfolio_value (float): Total monetary value of the portfolio.
        n_sims (int): Number of simulated paths.
        alpha (float): Significance level.
        horizon (int): Time horizon in days.
        pricing_func: Custom P&L mapping function for non-linear portfolios.
        seed (int): Random seed for reproducibility.
    """
    if use_ewma:
        if returns_matrix is None:
            raise ValueError('returns_matrix must be provided when use_ewma=True.')
        cov = calculate_mixed_ew_covariance(returns_matrix, lambda_var, lambda_corr)
    else:
        if cov_matrix is None:
            raise ValueError('cov_matrix must be provided when use_ewma=False.')
        cov = np.asarray(cov_matrix)
        
    cov_h = cov * horizon
        
    # Scenario Generator
    simulated_returns = simulate_multivariate_normal(
        cov_matrix=cov_h, 
        n_sims=n_sims, 
        seed=seed
    )
    
    if pricing_func is not None:
        pnl_scenarios = pricing_func(simulated_returns)
    else:
        if weights is None:
            raise ValueError('Weights must be provided if no pricing_func is specified.')
        w = np.asarray(weights).flatten()
        exposures = w * portfolio_value
        pnl_scenarios = simulated_returns @ exposures

    # VaR Extraction
    port_mean = np.mean(pnl_scenarios)
    port_std = np.std(pnl_scenarios, ddof=1)
    
    var_cutoff = np.quantile(pnl_scenarios, alpha)
    
    return {
        'VaR_Absolute': float(max(-var_cutoff, 0.0)),
        'VaR_Relative': float(port_mean - var_cutoff),
        'Portfolio_Mean': float(port_mean),
        'Portfolio_Std_Dev': float(port_std)
    }

if __name__ == '__main__':
    try:
        data = pd.read_csv('../data/test7_1.csv')
        
        norm_results = calculate_normal_var(data['x1'], alpha=0.05)
        print('--- Normal VaR ---')
        print(f"Absolute VaR: {norm_results['VaR_Absolute']:.6f}")
        print(f"Relative VaR: {norm_results['VaR_Relative']:.6f}")
        
        t_results = calculate_t_var(data['x1'], alpha=0.05)
        print("\n--- Student\'s t VaR ---")
        print(f'Absolute VaR: {t_results["VaR_Absolute"]:.6f}')
        print(f'Relative VaR: {t_results["VaR_Relative"]:.6f}')

    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Risk/ directory.')

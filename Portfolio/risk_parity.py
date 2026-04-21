import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
from typing import Callable, Optional

def component_volatility(w: np.ndarray, cov_matrix: np.ndarray):
    """
    Calculates the CSD for each asset.
    To be used as the risk_func in normal distribution Risk Parity.
    
    Parameters:
        w (np.ndarray): Portfolio weights.
        cov_matrix (np.ndarray): Covariance matrix of the assets.
        
    Returns:
        np.ndarray: Component Standard Deviations.
    """
    port_var = w.T @ cov_matrix @ w
    port_vol = np.sqrt(port_var)
    
    if np.isclose(port_vol, 0.0):
        return np.zeros_like(w)
        
    marginal_risk = (cov_matrix @ w) / port_vol
    csd = w * marginal_risk
    return csd

def component_expected_shortfall(
    w: np.ndarray, 
    returns_matrix: np.ndarray, 
    alpha=0.05, 
    epsilon=1e-4
):
    """
    Calculates the CES using finite differences.
    Can accept either purely historical returns or simulated Copula returns.
    
    Parameters:
        w (np.ndarray): Portfolio weights.
        returns_matrix (np.ndarray): Return series (T observations x N assets).
        alpha (float): Significance tail level (default 0.05 for 95% confidence).
        epsilon (float): Bump size for the finite difference derivative.
        
    Returns:
        np.ndarray: CES.
    """
    n = len(w)
    ces = np.zeros(n)
    
    port_returns = returns_matrix @ w
    var_cutoff = np.quantile(port_returns, alpha)
    base_es = -np.mean(port_returns[port_returns <= var_cutoff])
    
    # Marginal ES
    for i in range(n):
        w_bumped = w.copy()
        w_bumped[i] += epsilon
        
        bumped_returns = returns_matrix @ w_bumped
        bumped_var_cutoff = np.quantile(bumped_returns, alpha)
        bumped_es = -np.mean(bumped_returns[bumped_returns <= bumped_var_cutoff])
        
        marginal_es = (bumped_es - base_es) / epsilon
        ces[i] = w[i] * marginal_es
        
    return ces

def _risk_parity_objective(
    w: np.ndarray, 
    risk_func: Callable, 
    risk_budget: Optional[np.ndarray], 
    *args
) -> float:
    """
    Generalized objective function for Risk Parity and Risk Budgeting.
    """
    component_risk = risk_func(w, *args)
    
    if risk_budget is None:
        mean_component_risk = np.mean(component_risk)
        sse = np.sum((component_risk - mean_component_risk)**2)
    else:
        # Component risk = Total Risk * b_i
        total_risk = np.sum(component_risk)
        target_risk = total_risk * risk_budget
        sse = np.sum((component_risk - target_risk)**2)
        
    return float(sse)

def optimize_risk_parity(
    w0: np.ndarray, 
    risk_func: Callable, 
    risk_budget_normalized: Optional[np.ndarray] = None,
    bounds: Optional[tuple] = None,
    *args
) -> np.ndarray:
    """Universal Optimizer for Equal Risk Parity and Custom Risk Budgeting."""
    n = len(w0)
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
        
    opt_result = minimize(
        _risk_parity_objective, 
        w0, 
        args=(risk_func, risk_budget_normalized, *args), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    if not opt_result.success:
        warnings.warn(f'Risk optimization did not converge: {opt_result.message}')
        
    return opt_result.x

def risk_parity_normal(cov_df: pd.DataFrame, risk_budget: Optional[list] = None) -> pd.DataFrame:
    """Wrapper for Normal Risk Parity / Budgeting."""
    cov_matrix = cov_df.values
    
    vols = np.sqrt(np.diag(cov_matrix))
    
    if risk_budget is None:
        inv_vols = 1.0 / vols
        w0 = inv_vols / np.sum(inv_vols)
        b_normalized = None
    else:
        b_array = np.array(risk_budget)
        b_normalized = b_array / np.sum(b_array)
        
        w_guess = b_array / vols
        w0 = w_guess / np.sum(w_guess)
    
    weights = optimize_risk_parity(w0, component_volatility, b_normalized, None, cov_matrix)
    
    result_df = pd.DataFrame(weights, columns=['W'])
    if isinstance(cov_df.columns, pd.Index):
        result_df.index = cov_df.columns
    return result_df

def risk_parity_es(returns_df: pd.DataFrame, risk_budget: Optional[list] = None, alpha: float = 0.05) -> pd.DataFrame:
    """Wrapper for Expected Shortfall Risk Parity / Budgeting."""
    returns_matrix = returns_df.values
    n = returns_matrix.shape[1]
    
    if risk_budget is None:
        w0 = np.ones(n) / n
        b_normalized = None
    else:
        b_array = np.array(risk_budget)
        b_normalized = b_array / np.sum(b_array)
        w0 = b_normalized
    
    weights = optimize_risk_parity(
        w0, component_expected_shortfall, b_normalized, None, returns_matrix, alpha
    )
    
    result_df = pd.DataFrame(weights, columns=['W'])
    if isinstance(returns_df.columns, pd.Index):
        result_df.index = returns_df.columns
    return result_df

if __name__ == '__main__':
    try:
        cov_df = pd.read_csv('../data/test5_2.csv')
        
        print('--- Equal Risk Parity ---')
        rp_weights_df = risk_parity_normal(cov_df)
        print(rp_weights_df.to_csv(index=False))
        
        print('\n--- Custom Risk Budget ---')
        b = [1.0, 1.0, 1.0, 1.0, 0.5]
        custom_rp_weights_df = risk_parity_normal(cov_df, risk_budget=b)
        print(custom_rp_weights_df.to_csv(index=False))
        
    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Portfolio/ directory.')
    
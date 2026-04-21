import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
from typing import Optional, Tuple

def calculate_sharpe_ratio(
    w: np.ndarray, 
    mean_returns: np.ndarray, 
    cov_matrix: np.ndarray, 
    rf: float = 0.0
) -> float:
    """
    Calculates the Sharpe Ratio for a given set of weights.
    
    Parameters:
        w (np.ndarray): Portfolio weights.
        mean_returns (np.ndarray): Expected returns for the assets.
        cov_matrix (np.ndarray): Covariance matrix of the assets.
        rf (float): Risk-free rate. Defaults to 0.0.
        
    Returns:
        float: The portfolio Sharpe Ratio.
    """
    port_return = w.T @ mean_returns
    port_vol = np.sqrt(w.T @ cov_matrix @ w)
    
    if np.isclose(port_vol, 0.0):
        return 0.0
        
    sharpe_ratio = (port_return - rf) / port_vol
    return float(sharpe_ratio)

def _neg_sharpe_ratio(
    w: np.ndarray, 
    mean_returns: np.ndarray, 
    cov_matrix: np.ndarray, 
    rf: float
) -> float:
    """
    Minimizing the negative Sharpe Ratio maximizes the true Sharpe Ratio.
    """
    return -calculate_sharpe_ratio(w, mean_returns, cov_matrix, rf)

def max_sharpe_ratio_normal(
    mean_df: pd.DataFrame, 
    cov_df: pd.DataFrame, 
    rf: float = 0.0,
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None
) -> pd.DataFrame:
    """
    Calculates the weights of the Maximum Sharpe Ratio portfolio.
    
    Parameters:
        mean_df (pd.DataFrame): Expected returns for the assets.
        cov_df (pd.DataFrame): Covariance matrix of the assets.
        rf (float): Risk-free rate. Defaults to 0.0.
        bounds (tuple, optional): A tuple of (min, max) pairs for each asset. 
                                  Defaults to (0.0, 1.0) for long-only.
        
    Returns:
        pd.DataFrame: Portfolio weights that maximize the Sharpe Ratio.
    """
    mean_returns = mean_df.values.flatten()
    cov_matrix = cov_df.values
    n = len(mean_returns)
    
    w0 = np.ones(n) / n
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    elif len(bounds) != n:
        raise ValueError(f'Length of bounds ({len(bounds)}) must match the number of assets ({n}).')
    
    w0 = np.clip(w0, [b[0] for b in bounds], [b[1] for b in bounds])
    w0 = w0 / np.sum(w0) 
    
    opt_result = minimize(
        _neg_sharpe_ratio, 
        w0, 
        args=(mean_returns, cov_matrix, rf), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    if not opt_result.success:
        warnings.warn(f'Max Sharpe optimization did not converge: {opt_result.message}')
        
    sr_weights = opt_result.x
    
    result_df = pd.DataFrame(sr_weights, columns=['W'])
    if isinstance(cov_df.columns, pd.Index):
        result_df.index = cov_df.columns
    return result_df

if __name__ == '__main__':
    try:
        test_mean_df = pd.read_csv('../data/test10_3_means.csv')
        test_cov_df = pd.read_csv('../data/test10_3_cov.csv')
        
        weights = max_sharpe_ratio_normal(test_mean_df, test_cov_df, rf=0.04)
        print(weights)
        
        n_assets = test_cov_df.shape[1]
        custom_bounds = tuple((0.1, 0.5) for _ in range(n_assets))
        weights_constrained = max_sharpe_ratio_normal(
            test_mean_df, 
            test_cov_df, 
            rf=0.04, 
            bounds=custom_bounds
        )
        print('--- Constrained Max Sharpe (0.1 to 0.5) ---')
        print(weights_constrained)
    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Portfolio/ directory.')
        
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def neg_sharpe_ratio(w, mean_returns, cov_matrix, rf):
    """
    Calculates the negative Sharpe Ratio for a given set of weights.
    We use the negative SR because scipy.optimize minimizes the objective function
    """
    port_return = w.T @ mean_returns
    port_vol = np.sqrt(w.T @ cov_matrix @ w)
    sharpe_ratio = (port_return - rf) / port_vol
    
    return -sharpe_ratio

def max_sharpe_ratio_normal(mean_df, cov_df, rf):
    """
    Calculates the weights of the Maximum Sharpe Ratio portfolio given 
    mean returns and a covariance matrix, subject to long-only constraints
    
    Parameters:
    mean_df : pd.DataFrame
        Expected returns for the assets.
    cov_df : pd.DataFrame
        Covariance matrix of the assets.
    rf : float
        Risk-free rate.
        
    Returns:
    pd.DataFrame
        Portfolio weights that maximize the Sharpe Ratio.
    """
    mean_returns = mean_df.values.flatten()
    cov_matrix = cov_df.values
    n = len(mean_returns)
    
    w0 = np.ones(n) / n
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    opt_result = minimize(
        neg_sharpe_ratio, 
        w0, 
        args=(mean_returns, cov_matrix, rf), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    if not opt_result.success:
        print('Warning: Optimization did not converge:', opt_result.message)
        
    sr_weights = opt_result.x
    
    result_df = pd.DataFrame(sr_weights, columns=['W'])
    return result_df

if __name__ == "__main__":
    cov_df = pd.read_csv("data/test5_2.csv")
    mean_df = pd.read_csv("data/test10_3_means.csv")
    rf = 0.04
    
    sr_weights_df = max_sharpe_ratio_normal(mean_df, cov_df, rf)
    print(sr_weights_df.to_csv(index=False))
    
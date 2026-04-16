import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_csd(w, cov_matrix):
    """
    Calculates the Component Standard Deviation (CSD) for each asset
    """
    port_var = w.T @ cov_matrix @ w
    port_vol = np.sqrt(port_var)
    
    # CSD = w * (Sigma * w) / sigma_p
    marginal_risk = (cov_matrix @ w) / port_vol
    csd = w * marginal_risk
    return csd

def risk_parity_objective(w, cov_matrix):
    """
    Objective function to minimize the Sum of Squared Errors (SSE) 
    between individual CSDs and the average CSD
    """
    csd = calculate_csd(w, cov_matrix)
    csd_mean = np.mean(csd)
    
    # SSE of CSDs
    sse = np.sum((csd - csd_mean)**2)
    return sse

def risk_parity_normal(cov_df):
    """
    Calculates the Risk Parity weights given a covariance matrix 
    under the normal distribution assumption
    
    Parameters:
    cov_df : pd.DataFrame
        Covariance matrix of the assets.
        
    Returns:
    pd.DataFrame
        Risk Parity weights matching the testout format.
    """
    cov_matrix = cov_df.values
    n = cov_matrix.shape[0]
    
    # Starting values: Normalized inverse volatility
    # w_i = (1 / sigma_i) / sum(1 / sigma_j)
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vols = 1.0 / vols
    w0 = inv_vols / np.sum(inv_vols)
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    opt_result = minimize(
        risk_parity_objective, 
        w0, 
        args=(cov_matrix,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    if not opt_result.success:
        print('Warning: Optimization did not converge:', opt_result.message)
        
    rp_weights = opt_result.x
    
    result_df = pd.DataFrame(rp_weights, columns=['W'])
    return result_df

if __name__ == "__main__":
    cov_df = pd.read_csv("data/test5_2.csv")
    rp_weights_df = risk_parity_normal(cov_df)
    print(rp_weights_df.to_csv(index=False))
    
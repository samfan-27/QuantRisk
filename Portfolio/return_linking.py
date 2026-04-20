import numpy as np

def calculate_carino_k(portfolio_returns, total_portfolio_return):
    """
    Calculates Cariño K scaling factors for multi-period return attribution.
    Handles the edge case where returns are exactly zero to avoid division by zero.
    
    Parameters:
        portfolio_returns (np.ndarray): Time series of portfolio returns.
        total_portfolio_return (float): The total geometric return of the portfolio.
        
    Returns:
        np.ndarray: Vector of Cariño K factors for each time period.
    """
    if np.isclose(total_portfolio_return, 0.0):
        K = 1.0
    else:
        K = np.log(1 + total_portfolio_return) / total_portfolio_return
        
    k_t = np.zeros_like(portfolio_returns)
    mask = ~np.isclose(portfolio_returns, 0.0)
    
    k_t[mask] = np.log(1 + portfolio_returns[mask]) / (K * portfolio_returns[mask])
    k_t[~mask] = 1.0 / K
    
    return k_t

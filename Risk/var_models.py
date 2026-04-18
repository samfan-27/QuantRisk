import numpy as np
import pandas as pd
from scipy.stats import norm, t

def _sanitize_returns(returns):
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError('Input must be a single series of returns, not a multi-column DataFrame.')
        returns = returns.iloc[:, 0]
        
    returns = np.asarray(returns)
    if returns.size == 0:
        raise ValueError('Input returns cannot be empty.')
        
    return returns

def calculate_normal_var(returns, alpha=0.05):
    """
    Calculates Value at Risk (VaR) using a Normal Distribution (Parametric).
    
    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).
        
    Returns:
        dict: Absolute VaR, Relative VaR, Mean, and Standard Deviation.
    """
    returns = _sanitize_returns(returns)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    z_score = norm.ppf(alpha)
    var_cutoff = mu + (z_score * sigma)
    
    return {
        'VaR_Absolute': max(-var_cutoff, 0.0),
        'VaR_Relative': mu - var_cutoff,
        'Mean': mu,
        'Std_Dev': sigma
    }

def calculate_t_var(returns, alpha=0.05):
    """
    Calculates Value at Risk (VaR) by fitting a Student's t-distribution.
    
    This method is often preferred over Normal VaR for financial data 
    because it captures 'fat tails' (leptokurtosis) better.
    
    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).
        
    Returns:
        dict: Absolute VaR, Relative VaR, df, loc (mean), and scale (sigma).
    """
    returns = _sanitize_returns(returns)
    nu, mu, sigma = t.fit(returns)

    t_percentile = t.ppf(alpha, df=nu, loc=mu, scale=sigma)

    return {
        'VaR_Absolute': max(-t_percentile, 0.0),
        'VaR_Relative': mu - t_percentile,
        'df': nu,
        'loc': mu,
        'scale': sigma
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
        
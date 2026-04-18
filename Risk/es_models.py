import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.integrate import quad

def _sanitize_returns(returns):
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError('Input must be a single series of returns, not a multi-column DataFrame.')
        returns = returns.iloc[:, 0]
        
    returns = np.asarray(returns)
    if returns.size == 0:
        raise ValueError('Input returns cannot be empty.')
        
    return returns

def calculate_normal_es(returns, alpha=0.05):
    """
    Calculates Expected Shortfall (ES) using a Normal Distribution assumption.
    Also known as Conditional VaR (CVaR).

    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).

    Returns:
        dict: Absolute ES, Relative ES, VaR Breakpoint, Mean, and Standard Deviation.
    """
    returns = _sanitize_returns(returns)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    z_cutoff = norm.ppf(alpha)
    # mu - sigma * (pdf(z_cutoff) / alpha)
    pdf_at_cutoff = norm.pdf(z_cutoff)
    es_value = mu - sigma * (pdf_at_cutoff / alpha)
    
    # VaR for reference
    var_value = mu + (z_cutoff * sigma)

    return {
        'ES_Absolute': max(-es_value, 0.0),
        'ES_Relative': mu - es_value,
        'VaR_Breakpoint': var_value,
        'Mean': mu,
        'Std_Dev': sigma
    }

def calculate_t_es(returns, alpha=0.05):
    """
    Calculates Expected Shortfall (ES) by fitting a Student's t-distribution 
    to the data and integrating the tail.

    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).

    Returns:
        dict: Absolute ES, Relative ES, VaR Breakpoint, df, loc, and scale.
    """
    returns = _sanitize_returns(returns)
    nu, mu, sigma = t.fit(returns)

    var_cutoff = t.ppf(alpha, df=nu, loc=mu, scale=sigma)

    def integrand(x):
        return x * t.pdf(x, df=nu, loc=mu, scale=sigma)

    # Integrate from negative infinity to the VaR cutoff
    integral_result, _ = quad(integrand, -np.inf, var_cutoff)
    
    expected_tail_loss = integral_result / alpha

    return {
        'ES_Absolute': max(-expected_tail_loss, 0.0),
        'ES_Relative': mu - expected_tail_loss,
        'VaR_Breakpoint': var_cutoff,
        'df': nu,
        'loc': mu,
        'scale': sigma
    }

if __name__ == '__main__':
    try:
        data = pd.read_csv('../data/test7_2.csv')
        
        norm_results = calculate_normal_es(data['x1'], alpha=0.05)
        print('--- Normal Expected Shortfall ---')
        print(f'Absolute ES:     {norm_results["ES_Absolute"]:.6f}')
        print(f'Relative ES:     {norm_results["ES_Relative"]:.6f}')
        print(f'VaR Breakpoint:  {norm_results["VaR_Breakpoint"]:.6f}')
        
        t_results = calculate_t_es(data['x1'], alpha=0.05)
        print("\n--- Student's t Expected Shortfall ---")
        print(f'Absolute ES:     {t_results["ES_Absolute"]:.6f}')
        print(f'Relative ES:     {t_results["ES_Relative"]:.6f}')
        print(f'VaR Breakpoint:  {t_results["VaR_Breakpoint"]:.6f}')
        print(f'Degrees of Free: {t_results["df"]:.4f}')

    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Risk/ directory.')
        
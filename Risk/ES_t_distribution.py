import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.integrate import quad

def calculate_t_es(returns, alpha=0.05):
    """
    Calculates Expected Shortfall (ES), also known as Conditional VaR (CVaR),
    by fitting a Student's t-distribution to the data and integrating the tail.

    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).

    Returns:
        dict: A dictionary containing:
            - 'ES_Absolute': The expected loss magnitude given that the VaR threshold is breached.
            - 'ES_Relative': The difference between the fitted mean and the ES.
            - 'VaR_Breakpoint': The VaR threshold used for integration.
            - 'df': The estimated degrees of freedom.
            - 'loc': The estimated location parameter (mean).
            - 'scale': The estimated scale parameter.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError("Input must be a single series of returns.")
        returns = returns.iloc[:, 0]
    
    returns = np.asarray(returns)
    if len(returns) == 0:
        raise ValueError("Input returns cannot be empty.")

    # 1. Fit the Student's t-distribution
    nu, mu, sigma = t.fit(returns)

    # 2. Calculate VaR
    var_cutoff = t.ppf(alpha, df=nu, loc=mu, scale=sigma)

    # 3. Calculate Expected Shortfall via Integration
    # Definition: E[x | x <= VaR] = (1 / alpha) * Integral(-inf to VaR) of x * pdf(x) dx
    
    # Define the integrand: x * f(x)
    def integrand(x):
        return x * t.pdf(x, df=nu, loc=mu, scale=sigma)

    # Integrate from negative infinity to the VaR cutoff
    # scipy.integrate.quad handles -np.inf gracefully
    integral_result, error = quad(integrand, -np.inf, var_cutoff)
    
    # The raw expected value in the tail (a negative number for losses)
    expected_tail_loss = integral_result / alpha

    # 4. Standardize Metrics
    es_absolute = -expected_tail_loss if expected_tail_loss < 0 else 0
    
    # Formula: Mean - (Negative Tail Value)
    es_relative = mu - expected_tail_loss

    return {
        "ES_Absolute": es_absolute,
        "ES_Relative": es_relative,
        "VaR_Breakpoint": var_cutoff,
        "df": nu,
        "loc": mu,
        "scale": sigma
    }

# Example usage for testing
if __name__ == "__main__":
    try:
        data = pd.read_csv('data/test7_2.csv')
        
        results = calculate_t_es(data['x1'], alpha=0.05)

        print("--- Student's t Expected Shortfall (ES) ---")
        print(f"Fitted DF:         {results['df']:.4f}")
        print(f"Fitted Mean:       {results['loc']:.6f}")
        print(f"Fitted Scale:      {results['scale']:.6f}")
        print("-" * 40)
        print(f"ES (Absolute):     {results['ES_Absolute']:.6f}")
        print(f"ES (Relative):     {results['ES_Relative']:.6f}")
        print(f"VaR Breakpoint:    {results['VaR_Breakpoint']:.6f}")

    except FileNotFoundError:
        print("Data file not found. Please check the path.")

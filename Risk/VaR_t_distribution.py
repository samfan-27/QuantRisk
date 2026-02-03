import numpy as np
import pandas as pd
from scipy.stats import t

def calculate_t_var(returns, alpha=0.05):
    """
    Calculates Value at Risk (VaR) by fitting a Student's t-distribution 
    to the provided returns data.

    This method is often preferred over Normal VaR for financial data 
    because it captures 'fat tails' (leptokurtosis) better.

    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).

    Returns:
        dict: A dictionary containing:
            - 'VaR_Absolute': The loss level at the alpha percentile (magnitude).
            - 'VaR_Relative': The difference between the fitted location (mean) and VaR.
            - 'df': The estimated degrees of freedom (nu).
            - 'loc': The estimated location parameter (mean).
            - 'scale': The estimated scale parameter (sigma).
    """
    # Ensure input is a valid 1D array-like structure
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            # If a multi-column DataFrame is passed, warn or select first column
            # For safety, we raise an error to ensure the user knows which data they are using.
            raise ValueError("Input must be a single series of returns, not a multi-column DataFrame.")
        returns = returns.iloc[:, 0]
        
    returns = np.asarray(returns)
    if len(returns) == 0:
        raise ValueError("Input returns cannot be empty.")

    # Fit the Student's t-distribution (MLE)
    # Returns: df (degrees of freedom), loc (mean), scale (std dev-like)
    nu, mu, sigma = t.fit(returns)

    # Calculate the percentile (quantile) for the given alpha
    t_percentile = t.ppf(alpha, df=nu, loc=mu, scale=sigma)

    # Absolute VaR
    var_absolute = -t_percentile if t_percentile < 0 else 0

    # Relative VaR
    var_relative = mu - t_percentile

    return {
        "VaR_Absolute": var_absolute,
        "VaR_Relative": var_relative,
        "df": nu,
        "loc": mu,
        "scale": sigma
    }

# Example usage for testing
if __name__ == "__main__":
    try:
        data = pd.read_csv('test7_2.csv')
        
        results = calculate_t_var(data['x1'], alpha=0.05)

        print("--- Student's t VaR Results ---")
        print(f"Fitted Degrees of Freedom: {results['df']:.4f}")
        print(f"Fitted Location (Mean):    {results['loc']:.6f}")
        print(f"Fitted Scale:              {results['scale']:.6f}")
        print("-" * 30)
        print(f"VaR (Absolute):            {results['VaR_Absolute']:.6f}")
        print(f"VaR (Relative):            {results['VaR_Relative']:.6f}")

    except FileNotFoundError:
        print("Data file not found. Please check the path.")
        
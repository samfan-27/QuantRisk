import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_parametric_es(returns, alpha=0.05):
    """
    Calculates Expected Shortfall (ES), also known as Conditional VaR (CVaR),
    using a Normal Distribution assumption.

    ES measures the average loss in the tail cases where losses exceed the VaR.

    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).

    Returns:
        dict: A dictionary containing:
            - 'ES_Absolute': The expected loss magnitude given that the threshold is breached.
            - 'ES_Relative': The difference between the mean return and the ES.
            - 'VaR_Breakpoint': The VaR threshold used to define the tail (for reference).
            - 'Mean': The mean return.
            - 'Std_Dev': The standard deviation of returns.
    """
    # Ensure input is a valid 1D array-like structure
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError("Input must be a single series of returns, not a multi-column DataFrame.")
        returns = returns.iloc[:, 0]
        
    returns = np.asarray(returns)
    if len(returns) == 0:
        raise ValueError("Input returns cannot be empty.")

    # Calculate statistics
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    # Calculate Z-score for the VaR cutoff
    z_cutoff = norm.ppf(alpha)
    
    # Calculate Expected Shortfall (Normal Distribution Formula)
    # Formula: ES = mu - sigma * (pdf(z) / alpha)
    # The term (pdf(z) / alpha) represents the expected value of the standard normal variable 
    # conditional on it being less than z.
    pdf_at_cutoff = norm.pdf(z_cutoff)
    es_value = mu - sigma * (pdf_at_cutoff / alpha)
    
    # Calculate VaR for reference (the starting point of the tail)
    var_value = mu + (z_cutoff * sigma)

    # Absolute ES: The average loss magnitude (represented as a positive number).
    es_absolute = -es_value if es_value < 0 else 0
    
    # Relative ES: The distance from the mean to the ES value.
    es_relative = mu - es_value

    return {
        "ES_Absolute": es_absolute,
        "ES_Relative": es_relative,
        "VaR_Breakpoint": var_value,
        "Mean": mu,
        "Std_Dev": sigma
    }

# Example usage for testing
if __name__ == "__main__":
    try:
        data = pd.read_csv('test7_1.csv')
        
        # Calculate ES using the 'x1' column
        results = calculate_parametric_es(data['x1'], alpha=0.05)

        print("--- Parametric Expected Shortfall (Normal) ---")
        print(f"Mean Return:       {results['Mean']:.6f}")
        print(f"Std Dev:           {results['Std_Dev']:.6f}")
        print("-" * 40)
        print(f"ES (Absolute):     {results['ES_Absolute']:.6f}")
        print(f"ES (Relative):     {results['ES_Relative']:.6f}")
        print(f"VaR Breakpoint:    {results['VaR_Breakpoint']:.6f}")

    except FileNotFoundError:
        print("Data file not found. Please check the path.")

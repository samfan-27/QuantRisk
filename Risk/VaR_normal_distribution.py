import numpy as np
from scipy.stats import norm

def calculate_parametric_var(returns, alpha=0.05):
    """
    Calculates the Value at Risk (VaR) using a Normal Distribution (Parametric).
    
    Parameters:
        returns (pd.Series or np.array): A series of asset or portfolio returns.
        alpha (float): The significance level (default 0.05 for 95% confidence).
        
    Returns:
        dict: A dictionary containing:
            - 'VaR_Absolute': The loss level at the alpha percentile (returned as a positive magnitude of loss).
            - 'VaR_Relative': The difference between the mean return and the VaR percentile (measure of volatility risk).
            - 'Mean': The mean return.
            - 'Std_Dev': The standard deviation of returns.
    """
    if len(returns) == 0:
        raise ValueError("Input returns cannot be empty.")

    # Calculate statistics
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    # Calculate Z-score
    z_score = norm.ppf(alpha)
    
    # Calculate the return at the cutoff point
    var_cutoff = mu + (z_score * sigma)
    
    # Absolute VaR
    var_absolute = -var_cutoff if var_cutoff < 0 else 0
    
    # Relative VaR
    var_relative = mu - var_cutoff
    
    return {
        "VaR_Absolute": var_absolute,
        "VaR_Relative": var_relative,
        "Mean": mu,
        "Std_Dev": sigma
    }

# Example usage (only runs if script is executed directly, not imported)
if __name__ == "__main__":
    import pandas as pd
    
    # Simulating data for demonstration or loading your own file
    try:
        data = pd.read_csv('test7_1.csv')
        results = calculate_parametric_var(data['x1'], alpha=0.05)
        
        print(f"Mean Return:      {results['Mean']:.6f}")
        print(f"Std Dev:          {results['Std_Dev']:.6f}")
        print(f"VaR (Absolute):   {results['VaR_Absolute']:.6f}")
        print(f"VaR (Relative):   {results['VaR_Relative']:.6f}")
    except FileNotFoundError:
        print("Data file not found. Please check the path.")

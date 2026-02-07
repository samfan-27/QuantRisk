import numpy as np
import pandas as pd

def calculate_ew_weights(n, lambd):
    """
    Generates exponentially decaying weights.
    
    Parameters:
        n (int): Number of observations.
        lambd (float): The decay factor (lambda).
        
    Returns:
        np.array: An array of normalized weights summing to 1.
    """
    # Generate weights: (1-lambda) * lambda^i for i in 0 to n-1
    # use vectorization for efficiency over large datasets
    i = np.arange(n)
    weights = (1 - lambd) * np.power(lambd, i)
    
    # Normalize weights to ensure they sum to exactly 1
    weights /= np.sum(weights)
    return weights

def calculate_ew_covariance_matrix(returns, lambd):
    """
    Calculates the Exponentially Weighted Covariance Matrix.
    
    This implementation uses a biased estimator (sum of weights = 1), 
    consistent with standard RiskMetrics methodologies.
    
    Parameters:
        returns (pd.DataFrame or np.array): Return series (rows=time, cols=assets).
        lambd (float): The decay factor (0 < lambda < 1).
        
    Returns:
        np.array: The covariance matrix.
    """
    # Ensure input is a numpy array
    X = np.asarray(returns)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_samples, n_assets = X.shape
    
    # 1. Reverse data so the most recent observation is at index 0
    #    (Standard convention for generating weights: w_0 applies to t, w_1 to t-1...)
    X_reversed = X[::-1]
    
    # 2. Generate and shape weights
    weights = calculate_ew_weights(n_samples, lambd)
    
    # Reshape weights
    w_reshaped = weights[:, np.newaxis]
    
    # 3. Calculate Weighted Mean
    weighted_mean = np.sum(X_reversed * w_reshaped, axis=0)
    
    # 4. Calculate Deviations
    deviations = X_reversed - weighted_mean
    
    # 5. Calculate Covariance: (X - mu).T @ (W * (X - mu))
    #    multiply deviations by sqrt(weights) or weights directly in the matrix mult.
    #    Cov = Deviations.T @ (weights * Deviations)
    cov_matrix = deviations.T @ (w_reshaped * deviations)
    
    return cov_matrix

def calculate_mixed_ew_covariance(returns, lambda_var=0.97, lambda_corr=0.94):
    """
    Calculates a covariance matrix using different decay factors for variance and correlation.
    
    Industry standard often calculates volatility with a smoother lambda (e.g., 0.97)
    and correlation with a more reactive lambda (e.g., 0.94) to capture changing relationships
    without overreacting to volatility spikes.
    
    Parameters:
        returns (pd.DataFrame): Asset returns.
        lambda_var (float): Lambda used for volatility (variance) calculation.
        lambda_corr (float): Lambda used for correlation calculation.
        
    Returns:
        np.array: The combined covariance matrix.
    """
    # 1. Calculate the two base covariance matrices
    cov_var_matrix = calculate_ew_covariance_matrix(returns, lambda_var)
    cov_corr_matrix = calculate_ew_covariance_matrix(returns, lambda_corr)
    
    # 2. Extract Volatilities from the "Variance Lambda" matrix
    #    The diagonal contains variances; sqrt gives standard deviation (volatility)
    var_variances = np.diag(cov_var_matrix)
    volatilities = np.sqrt(var_variances)
    
    # 3. Extract Correlations from the Correlation Lambda matrix
    #    Corr_ij = Cov_ij / (Vol_i * Vol_j)
    corr_variances = np.diag(cov_corr_matrix)
    corr_vol_basis = np.sqrt(corr_variances)
    
    #    Outer product creates a matrix where element (i, j) is vol_i * vol_j
    outer_vol_corr = np.outer(corr_vol_basis, corr_vol_basis)
    
    #    Avoid division by zero if flat lines exist (though unlikely in returns)
    correlation_matrix = np.divide(cov_corr_matrix, outer_vol_corr, 
                                   out=np.zeros_like(cov_corr_matrix), 
                                   where=outer_vol_corr!=0)
    
    # 4. Recombine: Final Cov = Correlation(0.94) * (Vol(0.97) * Vol(0.97))
    outer_vol_final = np.outer(volatilities, volatilities)
    final_cov_matrix = correlation_matrix * outer_vol_final
    
    return final_cov_matrix

# Example usage
if __name__ == "__main__":
    try:
        df = pd.read_csv('data/test2.csv')
        
        lam_var = 0.97
        lam_corr = 0.94
        
        final_cov = calculate_mixed_ew_covariance(df, lambda_var=lam_var, lambda_corr=lam_corr)
        
        print(f"--- Mixed EWMA Covariance Matrix ---")
        print(f"Lambda Var: {lam_var} | Lambda Corr: {lam_corr}")
        print("-" * 40)
        
        final_cov_df = pd.DataFrame(final_cov, index=df.columns, columns=df.columns)
        print(final_cov_df)

    except FileNotFoundError:
        print("Data file not found. Please check the path.")
        
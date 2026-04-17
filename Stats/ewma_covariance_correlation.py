import numpy as np
import pandas as pd

def _calculate_ew_weights(n, lambd):
    """
    Generates exponentially decaying weights.
    """
    i = np.arange(n)
    weights = (1 - lambd) * np.power(lambd, i)
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
    X = np.asarray(returns)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_samples, n_assets = X.shape
    
    X_reversed = X[::-1]
    
    weights = _calculate_ew_weights(n_samples, lambd)
    w_reshaped = weights[:, np.newaxis]
    
    weighted_mean = np.sum(X_reversed * w_reshaped, axis=0)
    deviations = X_reversed - weighted_mean
    
    # Cov = Deviations.T @ (weights * Deviations)
    cov_matrix = deviations.T @ (w_reshaped * deviations)
    
    return cov_matrix

def calculate_mixed_ew_covariance(returns, lambda_var=0.97, lambda_corr=0.94):
    """
    Calculates a covariance matrix using different decay factors for variance and correlation.
    
    Industry standard often calculates volatility with a smoother lambda (e.g., 0.97)
    and correlation with a more reactive lambda (e.g., 0.94) to capture changing relationships
    without overreacting to volatility spikes.
    
    Parameters:
        returns (pd.DataFrame or np.array): Asset returns.
        lambda_var (float): Lambda used for volatility (variance) calculation.
        lambda_corr (float): Lambda used for correlation calculation.
        
    Returns:
        np.array: The combined covariance matrix.
    """
    cov_var_matrix = calculate_ew_covariance_matrix(returns, lambda_var)
    cov_corr_matrix = calculate_ew_covariance_matrix(returns, lambda_corr)
    
    # Extract volatilities
    var_variances = np.diag(cov_var_matrix)
    volatilities = np.sqrt(var_variances)
    
    # Extract corr from the corr lambda matrix
    corr_variances = np.diag(cov_corr_matrix)
    corr_vol_basis = np.sqrt(corr_variances)
    outer_vol_corr = np.outer(corr_vol_basis, corr_vol_basis)
    
    correlation_matrix = np.divide(cov_corr_matrix, outer_vol_corr, 
                                   out=np.zeros_like(cov_corr_matrix), 
                                   where=outer_vol_corr!=0)
    
    # Final Cov = corr(0.94) * (Vol(0.97) * Vol(0.97))
    outer_vol_final = np.outer(volatilities, volatilities)
    final_cov_matrix = correlation_matrix * outer_vol_final
    
    return final_cov_matrix

if __name__ == "__main__":
    try:
        df = pd.read_csv('../data/test2.csv')
        
        lam_var = 0.97
        lam_corr = 0.94
        
        final_cov = calculate_mixed_ew_covariance(df, lambda_var=lam_var, lambda_corr=lam_corr)
        
        print('--- Mixed EWMA Covariance Matrix ---')
        print(f'Lambda Var: {lam_var} | Lambda Corr: {lam_corr}')
        
        final_cov_df = pd.DataFrame(final_cov, index=df.columns, columns=df.columns)
        print(final_cov_df)

    except FileNotFoundError:
        print('Data file not found. Please check the path and ensure you are running this from the Stats/ directory.')
        
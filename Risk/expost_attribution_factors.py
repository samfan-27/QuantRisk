import numpy as np
import pandas as pd

def expost_factor_attribution(stock_returns_df, factor_returns_df, betas_df, weights_df):
    """
    Ex-Post Return and Volatility Attribution to a Factor Model.
    
    Parameters:
    stock_returns_df : pd.DataFrame
        Time series of asset returns.
    factor_returns_df : pd.DataFrame
        Time series of factor returns.
    betas_df : pd.DataFrame
        Factor exposures (betas) for each asset.
    weights_df : pd.DataFrame
        Initial weights for the assets.
        
    Returns:
    pd.DataFrame
        Attribution table matching the testout11_2.csv format.
    """
    stock_returns = stock_returns_df.values
    factor_returns = factor_returns_df.values
    betas = betas_df.iloc[:, 1:].values  
    weights = weights_df.values.flatten()
    
    T, N = stock_returns.shape
    M = factor_returns.shape[1]
    W = np.zeros((T, N))
    W[0] = weights
    
    R_p = np.zeros(T)
    factor_contributions = np.zeros((T, M))
    alpha_contributions = np.zeros(T)
    
    for t in range(T):
        w_start = W[t]
        r_t = stock_returns[t]
        F_t = factor_returns[t]
        
        R_t = np.sum(w_start * r_t)
        R_p[t] = R_t
        
        # factor weights: w_j = sum(w_i * beta_{i,j})
        w_j = w_start @ betas
        
        c_j = w_j * F_t
        factor_contributions[t] = c_j
        
        alpha_t = R_t - np.sum(c_j)
        alpha_contributions[t] = alpha_t
        
        if t < T - 1:
            w_star = w_start * (1 + r_t)
            W[t+1] = w_star / (1 + R_t)
            
    factor_total_returns = np.prod(1 + factor_returns, axis=0) - 1
    alpha_total_return = np.prod(1 + alpha_contributions) - 1
    portfolio_total_return = np.prod(1 + R_p) - 1
    
    # Cariño K
    if portfolio_total_return == 0.0:
        K = 1.0
    else:
        GR = np.log(1 + portfolio_total_return)
        K = GR / portfolio_total_return
        
    k_t = np.zeros(T)
    for t in range(T):
        if R_p[t] == 0.0:
            k_t[t] = 1.0 / K
        else:
            k_t[t] = np.log(1 + R_p[t]) / (K * R_p[t])
            
    # Return Attribution
    # Multiply k_t element-wise across factor and alpha components
    factor_return_attr = np.sum(k_t[:, np.newaxis] * factor_contributions, axis=0)
    alpha_return_attr = np.sum(k_t * alpha_contributions)
    
    # Volatility Attribution
    sigma_p = np.std(R_p, ddof=1)
    factor_vol_attr = np.zeros(M)
    
    for j in range(M):
        # Risk attribution = Cov(factor contribution, portfolio return) / portfolio volatility
        cov = np.cov(factor_contributions[:, j], R_p, ddof=1)[0, 1]
        factor_vol_attr[j] = cov / sigma_p
        
    alpha_vol_attr = np.cov(alpha_contributions, R_p, ddof=1)[0, 1] / sigma_p
    
    columns = list(factor_returns_df.columns) + ['Alpha', 'Portfolio']
    
    out_data = [
        list(factor_total_returns) + [alpha_total_return, portfolio_total_return],
        list(factor_return_attr) + [alpha_return_attr, portfolio_total_return],
        list(factor_vol_attr) + [alpha_vol_attr, sigma_p]
    ]
    
    result_df = pd.DataFrame(out_data, 
                             columns=columns, 
                             index=['TotalReturn', 'Return Attribution', 'Vol Attribution'])
    result_df.index.name = 'Value'
    return result_df


if __name__ == "__main__":
    stock_returns_df = pd.read_csv("data/test11_2_stock_returns.csv")
    factor_returns_df = pd.read_csv("data/test11_2_factor_returns.csv")
    betas_df = pd.read_csv("data/test11_2_beta.csv")
    weights_df = pd.read_csv("data/test11_2_weights.csv")
    
    attribution_results = expost_factor_attribution(stock_returns_df, factor_returns_df, betas_df, weights_df)
    print(attribution_results.to_csv())

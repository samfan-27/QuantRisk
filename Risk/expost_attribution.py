import numpy as np
import pandas as pd

def expost_attribution(returns_df, weights_df):
    """
    Ex-Post Return and Volatility Attribution
    
    Parameters:
    returns_df : pd.DataFrame
        Time series of asset returns.
    weights_df : pd.DataFrame
        Initial weights for the assets.
        
    Returns:
    pd.DataFrame
        Attribution table matching the testout output.
    """
    returns = returns_df.values
    weights = weights_df.values.flatten()
    T, N = returns.shape
    
    W = np.zeros((T, N))
    W[0] = weights
    R_p = np.zeros(T)
    weighted_returns = np.zeros((T, N))
    
    for t in range(T):
        w_start = W[t]
        r_t = returns[t]
        wr = w_start * r_t
        weighted_returns[t] = wr
        
        R_t = np.sum(wr)
        R_p[t] = R_t
        
        if t < T - 1:
            w_star = w_start * (1 + r_t)
            W[t+1] = w_star / (1 + R_t)
            
    asset_total_returns = np.prod(1 + returns, axis=0) - 1
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
    return_attribution = np.sum(k_t[:, np.newaxis] * weighted_returns, axis=0)
    
    # Volatility Attribution
    vol_attribution = np.zeros(N)
    sigma_p = np.std(R_p, ddof=1)
    
    for i in range(N):
        # Risk attribution is the covariance of weighted returns with portfolio return 
        # divided by portfolio volatility
        cov = np.cov(weighted_returns[:, i], R_p, ddof=1)[0, 1]
        vol_attribution[i] = cov / sigma_p
        
    columns = list(returns_df.columns) + ['Portfolio']
    
    out_data = [
        list(asset_total_returns) + [portfolio_total_return],
        list(return_attribution) + [portfolio_total_return],
        list(vol_attribution) + [sigma_p]
    ]
    
    result_df = pd.DataFrame(out_data, 
                             columns=columns, 
                             index=['TotalReturn', 'Return Attribution', 'Vol Attribution'])
    result_df.index.name = 'Value'
    
    return result_df

if __name__ == "__main__":
    returns_df = pd.read_csv("data/test11_1_returns.csv")
    weights_df = pd.read_csv("data/test11_1_weights.csv")
    
    attribution_results = expost_attribution(returns_df, weights_df)
    print(attribution_results.to_csv())
    
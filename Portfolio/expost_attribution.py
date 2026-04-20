import numpy as np
import pandas as pd
from return_linking import calculate_carino_k

def expost_attribution(returns_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ex-Post Return and Volatility Attribution.
    
    Calculates the exact return and risk contributions of individual assets 
    to a portfolio over time, accounting for weight drift.
    
    Parameters:
        returns_df (pd.DataFrame): Time series of asset returns (T x N).
        weights_df (pd.DataFrame): Initial weights for the assets (1 x N).
        
    Returns:
        pd.DataFrame: Attribution table containing Total Return, 
                      Return Attribution, and Volatility Attribution.
    """
    returns = returns_df.values
    weights = weights_df.values.flatten()
    T, N = returns.shape
    
    # Asset value = W_0 * cumprod(1 + r_t)
    cum_asset_returns = np.cumprod(1 + returns, axis=0)
    asset_values = weights * cum_asset_returns
    port_values = np.sum(asset_values, axis=1)
    
    port_values_shifted = np.concatenate(([1.0], port_values[:-1]))
    cum_asset_returns_shifted = np.concatenate((np.ones((1, N)), cum_asset_returns[:-1]))
    
    # W_t = Asset_Value_{t-1} / Portfolio_Value_{t-1}
    W = (weights * cum_asset_returns_shifted) / port_values_shifted[:, np.newaxis]
    
    weighted_returns = W * returns
    R_p = np.sum(weighted_returns, axis=1)
    
    asset_total_returns = cum_asset_returns[-1] - 1
    portfolio_total_return = port_values[-1] - 1
    
    # Cariño K
    k_t = calculate_carino_k(R_p, portfolio_total_return)
    return_attribution = np.sum(k_t[:, np.newaxis] * weighted_returns, axis=0)
    
    # Volatility Attribution
    sigma_p = np.std(R_p, ddof=1)
    
    # cov(x, y) = sum((x - mean_x) * (y - mean_y)) / (T - 1)
    R_p_centered = R_p - np.mean(R_p)
    wr_centered = weighted_returns - np.mean(weighted_returns, axis=0)
    cov_vector = np.sum(wr_centered * R_p_centered[:, np.newaxis], axis=0) / (T - 1)
    
    vol_attribution = cov_vector / sigma_p

    columns = list(returns_df.columns) + ['Portfolio']
    out_data = [
        list(asset_total_returns) + [portfolio_total_return],
        list(return_attribution) + [portfolio_total_return],
        list(vol_attribution) + [sigma_p]
    ]
    
    result_df = pd.DataFrame(
        out_data, 
        columns=columns, 
        index=['TotalReturn', 'Return Attribution', 'Vol Attribution']
    )
    result_df.index.name = 'Value'
    return result_df

if __name__ == '__main__':
    try:
        ret_df = pd.read_csv('../data/test11_1_returns.csv')
        w_df = pd.read_csv('../data/test11_1_weights.csv')
    
        attribution_results = expost_attribution(ret_df, w_df)
        print(attribution_results)
    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Portfolio/ directory.')
        
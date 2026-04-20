import numpy as np
import pandas as pd
from return_linking import calculate_carino_k

def expost_factor_attribution(
    stock_returns_df: pd.DataFrame, 
    factor_returns_df: pd.DataFrame, 
    betas_df: pd.DataFrame, 
    weights_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ex-Post Return and Volatility Attribution to a Factor Model.
    
    Calculates attribution metrics for systematic risk factors and 
    isolates the residual portfolio Alpha.
    
    Parameters:
        stock_returns_df (pd.DataFrame): Time series of asset returns.
        factor_returns_df (pd.DataFrame): Time series of factor returns.
        betas_df (pd.DataFrame): Factor exposures (betas) for each asset.
        weights_df (pd.DataFrame): Initial weights for the assets.
        
    Returns:
        pd.DataFrame: Attribution table containing Alpha and Factor metrics.
    """
    stock_returns = stock_returns_df.values
    factor_returns = factor_returns_df.values
    betas = betas_df.iloc[:, 1:].values  
    weights = weights_df.values.flatten()
    
    T, N = stock_returns.shape
    
    cum_stock_returns = np.cumprod(1 + stock_returns, axis=0)
    asset_values = weights * cum_stock_returns
    port_values = np.sum(asset_values, axis=1)
    
    port_values_shifted = np.concatenate(([1.0], port_values[:-1]))
    cum_stock_returns_shifted = np.concatenate((np.ones((1, N)), cum_stock_returns[:-1]))
    
    W = (weights * cum_stock_returns_shifted) / port_values_shifted[:, np.newaxis]
    
    weighted_returns = W * stock_returns
    R_p = np.sum(weighted_returns, axis=1)
    
    # w_j = sum(w_i * beta_{i,j}) via matrix multiplication (T, N) @ (N, M) -> (T, M)
    w_j = W @ betas
    factor_contributions = w_j * factor_returns
    alpha_contributions = R_p - np.sum(factor_contributions, axis=1)
    
    factor_total_returns = np.prod(1 + factor_returns, axis=0) - 1
    alpha_total_return = np.prod(1 + alpha_contributions) - 1
    portfolio_total_return = port_values[-1] - 1
    
    # Cariño K
    k_t = calculate_carino_k(R_p, portfolio_total_return)
    factor_return_attr = np.sum(k_t[:, np.newaxis] * factor_contributions, axis=0)
    alpha_return_attr = np.sum(k_t * alpha_contributions)
    
    # Volatility Attribution
    sigma_p = np.std(R_p, ddof=1)
    R_p_centered = R_p - np.mean(R_p)
    
    fc_centered = factor_contributions - np.mean(factor_contributions, axis=0)
    factor_cov_vector = np.sum(fc_centered * R_p_centered[:, np.newaxis], axis=0) / (T - 1)
    factor_vol_attr = factor_cov_vector / sigma_p
    
    alpha_centered = alpha_contributions - np.mean(alpha_contributions)
    alpha_cov = np.sum(alpha_centered * R_p_centered) / (T - 1)
    alpha_vol_attr = alpha_cov / sigma_p
    
    columns = list(factor_returns_df.columns) + ['Alpha', 'Portfolio']
    out_data = [
        list(factor_total_returns) + [alpha_total_return, portfolio_total_return],
        list(factor_return_attr) + [alpha_return_attr, portfolio_total_return],
        list(factor_vol_attr) + [alpha_vol_attr, sigma_p]
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
        s_ret_df = pd.read_csv('../data/test11_2_stock_returns.csv')
        f_ret_df = pd.read_csv('../data/test11_2_factor_returns.csv')
        b_df = pd.read_csv('../data/test11_2_beta.csv')
        w_df = pd.read_csv('../data/test11_2_weights.csv')
    
        attribution_results = expost_factor_attribution(s_ret_df, f_ret_df, b_df, w_df)
        print(attribution_results)
    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Portfolio/ directory.')
    
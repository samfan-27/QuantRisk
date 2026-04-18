import numpy as np
import pandas as pd
from scipy.stats import norm, t

def calculate_copula_var_es(
    portfolio: pd.DataFrame, 
    returns: pd.DataFrame, 
    marginal_dist: str = 't',
    alpha: float = 0.95, 
    n_sim: int = 100000, 
    seed: int = 42
) -> pd.DataFrame:
    """
    Calculates VaR and ES for an N-asset portfolio using a Gaussian Copula.

    Parameters:
        portfolio (pd.DataFrame): 'Stock', 'Holding', and 'Starting Price'.
        returns (pd.DataFrame): Historical returns.
        marginal_dist (str): 't' for Parametric Student's t, 'empirical' for Historical.
        alpha (float): Confidence level for loss distribution (default 0.95).
        n_sim (int): Number of Monte Carlo simulations.
        seed (int): Random seed.
    """
    if marginal_dist not in ['t', 'empirical']:
        raise ValueError("marginal_dist must be either 't' or 'empirical'.")

    try:
        portfolio_df = portfolio.set_index('Stock')
        initial_values = (portfolio_df['Holding'] * portfolio_df['Starting Price']).to_dict()
        assets = list(initial_values.keys())
    except KeyError as e:
        raise ValueError(f'Missing required column in portfolio data: {e}')

    returns = returns[assets]
    n_assets = len(assets)

    # fit marginals and transform to Uniform [0, 1]
    marginals = {}
    u_data = np.zeros_like(returns.values)

    for i, asset in enumerate(assets):
        asset_returns = returns[asset].dropna()
        
        if marginal_dist == 't':
            # parametric
            df, loc, scale = t.fit(asset_returns)
            marginals[asset] = {'type': 't', 'df': df, 'loc': loc, 'scale': scale}
            u_data[:, i] = t.cdf(asset_returns, df=df, loc=loc, scale=scale)
            
        elif marginal_dist == 'empirical':
            # non-parametric: Empirical CDF (Rank / N+1)
            n_obs = len(asset_returns)
            ranks = asset_returns.rank()
            u_data[:, i] = ranks / (n_obs + 1)
            marginals[asset] = {'type': 'empirical', 'data': asset_returns.values}

    # Calculate corr matrix for the Gaussian Copula
    # Map uniform data to std normal to compute Pearson corr
    norm_data = norm.ppf(u_data)
    corr_matrix = np.corrcoef(norm_data, rowvar=False)

    rng = np.random.default_rng(seed)
    sim_norm = rng.multivariate_normal(mean=np.zeros(n_assets), cov=corr_matrix, size=n_sim)
    sim_u = norm.cdf(sim_norm)

    # Transform back to marginals & calculate losses
    sim_losses = np.zeros((n_sim, n_assets))
    
    for i, asset in enumerate(assets):
        params = marginals[asset]
        
        if params['type'] == 't':
            sim_returns = t.ppf(sim_u[:, i], df=params['df'], loc=params['loc'], scale=params['scale'])
        elif params['type'] == 'empirical':
            sim_returns = np.quantile(params['data'], sim_u[:, i])
            
        sim_losses[:, i] = -sim_returns * initial_values[asset]

    total_loss = np.sum(sim_losses, axis=1)

    def calc_metrics(losses):
        var_val = np.quantile(losses, alpha)
        es_val = losses[losses > var_val].mean()
        return var_val, es_val

    results = []
    total_portfolio_value = sum(initial_values.values())

    for i, asset in enumerate(assets):
        var_val, es_val = calc_metrics(sim_losses[:, i])
        results.append({
            'Stock': asset,
            f'VaR_{int(alpha*100)}': var_val,
            f'ES_{int(alpha*100)}': es_val,
            f'VaR_{int(alpha*100)}_Pct': var_val / initial_values[asset],
            f'ES_{int(alpha*100)}_Pct': es_val / initial_values[asset]
        })

    var_total, es_total = calc_metrics(total_loss)
    results.append({
        'Stock': 'Total',
        f'VaR_{int(alpha*100)}': var_total,
        f'ES_{int(alpha*100)}': es_total,
        f'VaR_{int(alpha*100)}_Pct': var_total / total_portfolio_value,
        f'ES_{int(alpha*100)}_Pct': es_total / total_portfolio_value
    })

    return pd.DataFrame(results)

if __name__ == '__main__':
    import time
    
    try:
        portfolio_file = '../data/test9_1_portfolio.csv'
        returns_file = '../data/test9_1_returns.csv'
        
        portfolio_data = pd.read_csv(portfolio_file)
        returns_data = pd.read_csv(returns_file)
        
        start_time = time.time()
        
        results_df = calculate_copula_var_es(
            portfolio=portfolio_data, 
            returns=returns_data, 
            marginal_dist='empirical',
            alpha=0.95, 
            n_sim=100000, 
            seed=0
        )
        
        elapsed = time.time() - start_time
        print(f'--- Copula VaR/ES Simulation (Empirical Marginals) ---')
        print(f'Completed in {elapsed:.2f}s\n')
        print(results_df.to_string(index=False))

    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Risk/ directory.')
        
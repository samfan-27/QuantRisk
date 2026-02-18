import pandas as pd
import numpy as np
from scipy.stats import norm, t, multivariate_normal
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_copula_var_es(
    portfolio: pd.DataFrame, 
    returns: pd.DataFrame, 
    alpha: float = 0.95, 
    n_sim: int = 100000, 
    seed: int = 0
) -> pd.DataFrame:
    """
    Calculates Value at Risk (VaR) and Expected Shortfall (ES) for a portfolio 
    using a Gaussian Copula via Monte Carlo simulation.

    Parameters:
        portfolio (pd.DataFrame): DataFrame containing 'Stock', 'Holding', and 'Starting Price'.
        returns (pd.DataFrame): DataFrame of historical returns for the assets.
        alpha (float): Confidence level for VaR and ES (default 0.95).
        n_sim (int): Number of Monte Carlo simulations (default 100,000).
        seed (int): Random seed for reproducibility (default 0).

    Returns:
        pd.DataFrame: Summary table of Absolute and Percentage VaR and ES.
    """
    try:
        holdings = dict(zip(portfolio['Stock'], portfolio['Holding']))
        prices = dict(zip(portfolio['Stock'], portfolio['Starting Price']))
        initial_values = {s: holdings[s] * prices[s] for s in portfolio['Stock']}
    except KeyError as e:
        raise ValueError(f"Missing required column in portfolio data: {e}")

    # Marginal Distributions
    logging.info("Fitting marginal distributions (Normal for A, Student-t for B)...")
    mu_A, sigma_A = norm.fit(returns['A'])
    df_B, loc_B, scale_B = t.fit(returns['B'])

    # Transform to Uniform and then to std normal Z-scores
    u_A = norm.cdf(returns['A'], loc=mu_A, scale=sigma_A)
    u_B = t.cdf(returns['B'], df=df_B, loc=loc_B, scale=scale_B)
    z = norm.ppf(np.column_stack([u_A, u_B]))

    # Copula Correlation Mat
    corr = np.corrcoef(z, rowvar=False)

    logging.info(f"Running {n_sim} Monte Carlo simulations (seed={seed})...")
    np.random.seed(seed)
    sim_z = multivariate_normal.rvs(mean=[0, 0], cov=corr, size=n_sim)
    sim_u = norm.cdf(sim_z)

    # Inverse Transform back to simulated returns
    sim_A = norm.ppf(sim_u[:, 0], loc=mu_A, scale=sigma_A)
    sim_B = t.ppf(sim_u[:, 1], df=df_B, loc=loc_B, scale=scale_B)

    loss_A = -sim_A * initial_values['A']
    loss_B = -sim_B * initial_values['B']
    total_loss = loss_A + loss_B

    def calc_metrics(losses):
        var = np.quantile(losses, alpha)
        es = losses[losses > var].mean()
        return var, es

    var_A, es_A = calc_metrics(loss_A)
    var_B, es_B = calc_metrics(loss_B)
    var_total, es_total = calc_metrics(total_loss)

    total_value = sum(initial_values.values())

    results = pd.DataFrame({
        'Stock': ['A', 'B', 'Total'],
        'VaR95': [var_A, var_B, var_total],
        'ES95': [es_A, es_B, es_total],
        'VaR95_Pct': [var_A/initial_values['A'], var_B/initial_values['B'], var_total/total_value],
        'ES95_Pct': [es_A/initial_values['A'], es_B/initial_values['B'], es_total/total_value]
    })
    
    return results

def main():
    portfolio_file = 'data/test9_1_portfolio.csv'
    returns_file = 'data/test9_1_returns.csv'

    try:
        portfolio_df = pd.read_csv(portfolio_file)
        returns_df = pd.read_csv(returns_file)
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e.filename}. Please check your paths.")
        return

    results = calculate_copula_var_es(portfolio_df, returns_df, alpha=0.95, n_sim=100000, seed=0)
    
    print("\n--- Copula VaR and ES Results ---")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
    
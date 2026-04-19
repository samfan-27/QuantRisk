import numpy as np
import pandas as pd
from .finite_difference import approx_greek_fd

def _american_binomial_tree(S, K, T, vol, r, b, N, option_type='call'):
    option_type = str(option_type).strip().lower()

    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    payoff_side = 1 if option_type == 'call' else -1

    num_nodes = lambda N: int((N + 1) * (N + 2) / 2)
    total_nodes = num_nodes(N)
    get_idx = lambda j, i: num_nodes(j - 1) + i

    option_values = [0.0] * total_nodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = get_idx(j, i)
            option_values[idx] = max(0.0, payoff_side * (S * (u**i) * (d**(j - i)) - K))

            if j < N:
                option_values[idx] = max(
                    option_values[idx], 
                    df * (pu * option_values[get_idx(j + 1, i + 1)] + pd * option_values[get_idx(j + 1, i)])
                )
    
    return option_values[0]

def calculate_american_metrics(S, K, T, r, q, sigma, N=200, option_type='call', richardson=True):
    """
    Calculates the Value and Greeks of an American option using a Binomial Tree.
    
    Parameters:
        S, K, T: Spot, Strike, Time to Maturity (years).
        r, q: Risk-free rate and Continuous dividend yield.
        sigma: Implied volatility.
        N (int): Number of steps in the tree.
        option_type (str): 'call' or 'put'.
        richardson (bool): Apply Richardson Extrapolation for Greek stability.
        
    Returns:
        dict: Value, Delta, Gamma, Vega, Rho, Theta.
    """
    b = r - q
    
    # base value
    val = _american_binomial_tree(S, K, T, sigma, r, b, N, option_type)

    h_S = S * 0.01
    h_vol = 0.001
    h_r = 0.001
    h_T = 1.0 / 365.0
    
    delta = approx_greek_fd(lambda h: _american_binomial_tree(S+h, K, T, sigma, r, b, N, option_type), val, h_S, richardson=richardson)
    gamma = approx_greek_fd(lambda h: _american_binomial_tree(S+h, K, T, sigma, r, b, N, option_type), val, h_S, is_gamma=True, richardson=richardson)
    vega  = approx_greek_fd(lambda h: _american_binomial_tree(S, K, T, sigma+h, r, b, N, option_type), val, h_vol, richardson=richardson)
    rho = approx_greek_fd(lambda h: _american_binomial_tree(S, K, T, sigma, r+h, b, N, option_type), val, h_r, richardson=richardson)
    # bumping r also bumps the cost of carry (b = r - q)
    # OR rho = approx_greek_fd(lambda h: _american_binomial_tree(S, K, T, sigma, r+h, b+h, N, option_type), val, h_r, richardson=richardson)
    
    if T > h_T:
        theta = approx_greek_fd(lambda h: _american_binomial_tree(S, K, T+h, sigma, r, b, N, option_type), val, h_T, richardson=richardson)
    else:
        theta = 0.0
        
    return {
        'Value': val,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }

if __name__ == "__main__":
    try:
        input_file = '../data/test12_1.csv'
        df = pd.read_csv(input_file)
        df.dropna(how='all', inplace=True) 
        
        results = []
        for index, row in df.iterrows():
            S = float(row['Underlying'])
            K = float(row['Strike'])
            T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
            r = float(row['RiskFreeRate'])
            q = float(row['DividendRate'])
            sigma = float(row['ImpliedVol'])
            opt_type = str(row['Option Type'])
            
            metrics = calculate_american_metrics(S, K, T, r, q, sigma, N=200, option_type=opt_type)
            
            row_result = {'ID': int(row['ID'])}
            row_result.update(metrics)
            results.append(row_result)
            
        results_df = pd.DataFrame(results)
        print('--- American Binomial Options Data ---')
        print(results_df.head(10).to_string(index=False))

    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Options/ directory.')
        
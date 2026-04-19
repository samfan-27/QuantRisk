import numpy as np
import pandas as pd
from .finite_difference import approx_greek_fd

def _american_discrete_tree(S, K, T, sigma, r, div_times, div_amounts, N, option_type='call'):
    option_type = str(option_type).strip().lower()

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    
    pu = (np.exp(r * dt) - d) / (u - d) 
    pd = 1.0 - pu
    df_discount = np.exp(-r * dt)
    
    payoff_side = 1 if option_type == "call" else -1
    
    # PV of all dividends
    pv_divs_total = sum(D * np.exp(-r * t) for t, D in zip(div_times, div_amounts))
    S_adj = S - pv_divs_total 

    num_nodes = lambda N: int((N + 1) * (N + 2) / 2)
    total_nodes = num_nodes(N)
    get_idx = lambda j, i: num_nodes(j - 1) + i

    option_values = [0.0] * total_nodes
    
    pv_D_array = np.zeros(N + 1)
    for j in range(N + 1):
        t = j * dt
        pv_D_array[j] = sum(D * np.exp(-r * (t_k - t)) for t_k, D in zip(div_times, div_amounts) if t_k > t)

    for j in range(N, -1, -1):
        pv_D = pv_D_array[j]
        
        for i in range(j, -1, -1):
            idx = get_idx(j, i)
            
            # actual stock price at this node = risky component + PV of remaining dividends
            actual_S = S_adj * (u**i) * (d**(j - i)) + pv_D
            
            option_values[idx] = max(0.0, payoff_side * (actual_S - K))

            if j < N:
                option_values[idx] = max(
                    option_values[idx], 
                    df_discount * (pu * option_values[get_idx(j + 1, i + 1)] + pd * option_values[get_idx(j + 1, i)])
                )
    
    return option_values[0]

def calculate_american_discrete_metrics(S, K, T, r, sigma, div_times, div_amounts, N=200, option_type='call', richardson=True):
    """
    Calculates the Value and Greeks of an American option with discrete dividends.
    
    Parameters:
        S, K, T: Spot, Strike, Time to Maturity (years).
        r, sigma: Risk-free rate and Implied volatility.
        div_times (list): Times to each discrete dividend (in years).
        div_amounts (list): Amounts of each discrete dividend.
        N (int): Number of steps in the tree.
        option_type (str): 'call' or 'put'.
        richardson (bool): Apply Richardson Extrapolation for Greek stability.
        
    Returns:
        dict: Value, Delta, Gamma, Vega. (Theta and Rho omitted due to discrete jump instability).
    """
    val = _american_discrete_tree(S, K, T, sigma, r, div_times, div_amounts, N, option_type)
    
    h_S = S * 0.01
    h_vol = 0.001
    
    delta = approx_greek_fd(lambda h: _american_discrete_tree(S+h, K, T, sigma, r, div_times, div_amounts, N, option_type), val, h_S, richardson=richardson)
    gamma = approx_greek_fd(lambda h: _american_discrete_tree(S+h, K, T, sigma, r, div_times, div_amounts, N, option_type), val, h_S, is_gamma=True, richardson=richardson)
    vega  = approx_greek_fd(lambda h: _american_discrete_tree(S, K, T, sigma+h, r, div_times, div_amounts, N, option_type), val, h_vol, richardson=richardson)
    
    return {
        'Value': val,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega
    }

if __name__ == '__main__':
    try:
        input_file = 'test12_3.csv'
        df = pd.read_csv(input_file)
        df.dropna(how='all', inplace=True) 
        
        results = []
        for index, row in df.iterrows():
            S = float(row['Underlying'])
            K = float(row['Strike'])
            day_per_year = float(row['DayPerYear'])
            T = float(row['DaysToMaturity']) / day_per_year
            r = float(row['RiskFreeRate'])
            sigma = float(row['ImpliedVol'])
            opt_type = str(row['Option Type'])
            
            div_dates_str = str(row['DividendDates']).strip()
            div_amts_str = str(row['DividendAmts']).strip()
            
            if div_dates_str and div_dates_str.lower() != 'nan':
                div_times = [float(x) / day_per_year for x in div_dates_str.split(',')]
                div_amounts = [float(x) for x in div_amts_str.split(',')]
            else:
                div_times, div_amounts = [], []
            
            metrics = calculate_american_discrete_metrics(S, K, T, r, sigma, div_times, div_amounts, N=800, option_type=opt_type)
            
            row_result = {'ID': int(row['ID'])}
            row_result.update(metrics)
            results.append(row_result)
            
        results_df = pd.DataFrame(results)
        print('--- American Discrete Dividend Options Data ---')
        print(results_df.head(10).to_string(index=False))

    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Options/ directory.')
        
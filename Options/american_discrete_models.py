import numpy as np
import pandas as pd

def American_Binary_Tree_Discrete(S, X, T, vol, r, div_times, div_amounts, N, option_type='call'):
    """
    Calculates the price of an American option with discrete dividends using a 
    Binomial Tree model (Escrowed Dividend Model).
    
    Parameters:
        S (float): Current underlying asset price.
        X (float): Strike price of the option.
        T (float): Time to maturity in years.
        vol (float): Implied volatility of the underlying asset.
        r (float): Risk-free interest rate.
        div_times (list of float): Times to each discrete dividend (in years).
        div_amounts (list of float): Amounts of each discrete dividend.
        N (int): Number of steps in the binomial tree.
        option_type (str): 'call' or 'put' (default is 'call').
        
    Returns:
        float: The estimated present value of the American option.
    """
    option_type = str.lower(option_type)

    dt = T/N
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    
    # Since we extract the PV of dividends, the remaining "risky" component 
    # grows at the risk-free rate. Therefore, cost of carry (b) = r.
    pu = (np.exp(r*dt)-d)/(u-d) 
    pd = 1.0-pu
    df = np.exp(-r*dt)
    
    payoff_side = 1 if option_type == "call" else -1
    pv_divs = sum(D * np.exp(-r * t) for t, D in zip(div_times, div_amounts))
    S_adj = S - pv_divs 

    num_nodes = lambda N: int((N+1)*(N+2)/2)
    total_nodes = num_nodes(N)
    get_idx = lambda j, i: num_nodes(j-1)+i

    option_values = [0.0]*total_nodes
    
    def pv_div_at_t(t):
        """Calculates the PV of dividends that occur strictly AFTER time t."""
        return sum(D * np.exp(-r * (t_k - t)) for t_k, D in zip(div_times, div_amounts) if t_k > t)

    for j in range(N, -1, -1):
        t = j * dt
        pv_D = pv_div_at_t(t)
        
        for i in range(j, -1, -1):
            idx = get_idx(j, i)
            
            # Actual stock price at this node = risky component + PV of remaining dividends
            actual_S = S_adj * (u**i) * (d**(j-i)) + pv_D
            
            option_values[idx] = max(0.0, payoff_side * (actual_S - X))

            if j < N:
                option_values[idx] = max(
                    option_values[idx], 
                    df * (pu*option_values[get_idx(j+1, i+1)] + pd*option_values[get_idx(j+1, i)])
                )
    
    return option_values[0]


def process_discrete_american_options(input_file, N=200):
    """
    Reads the discrete dividend options universe, parses dividend schedules and
    calculates American values iteratively.
    """
    df = pd.read_csv(input_file)
    df.dropna(how='all', inplace=True) 
    
    results = []
    
    for index, row in df.iterrows():
        S = float(row['Underlying'])
        K = float(row['Strike'])
        day_per_year = float(row['DayPerYear'])
        T = float(row['DaysToMaturity']) / day_per_year
        r = float(row['RiskFreeRate'])
        vol = float(row['ImpliedVol'])
        opt_type = str(row['Option Type']).strip()
        
        div_dates_str = str(row['DividendDates']).strip()
        div_amts_str = str(row['DividendAmts']).strip()
        
        if div_dates_str and div_dates_str.lower() != 'nan':
            div_times = [float(x) / day_per_year for x in div_dates_str.split(',')]
            div_amounts = [float(x) for x in div_amts_str.split(',')]
        else:
            div_times = []
            div_amounts = []
        
        val = American_Binary_Tree_Discrete(
            S, K, T, vol, r, div_times, div_amounts, N, opt_type
        )
        
        results.append({
            'ID': int(row['ID']),
            'Value': val
        })
        
    results_df = pd.DataFrame(results)
    
    print(results_df)
    
if __name__ == "__main__":
    try:
        INPUT_FILE = 'data/test12_3.csv'
        process_discrete_american_options(INPUT_FILE, N=800)
        
    except FileNotFoundError:
        print('Data file not found. Please check the path.')
        
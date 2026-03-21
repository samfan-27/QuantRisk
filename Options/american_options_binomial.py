import numpy as np
import pandas as pd

def American_Binary_Tree(S, X, T, vol, r, b, N, option_type='call'):
    """
    Calculates the price of an American option using a Binomial Tree model.
    
    Parameters:
        S (float): Current underlying asset price.
        X (float): Strike price of the option.
        T (float): Time to maturity in years.
        vol (float): Implied volatility of the underlying asset.
        r (float): Risk-free interest rate.
        b (float): Cost of carry (r - q, where q is the continuous dividend yield).
        N (int): Number of steps in the binomial tree.
        option_type (str): 'call' or 'put' (default is 'call').
        
    Returns:
        float: The estimated present value of the American option.
    """
    option_type = str.lower(option_type)

    dt = T/N
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = np.exp(-r*dt)
    payoff_side = 1 if option_type=='call' else -1

    num_nodes = lambda N: int((N+1)*(N+2)/2)
    total_nodes = num_nodes(N)
    get_idx = lambda j, i: num_nodes(j-1)+i

    option_values = [0.0]*total_nodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = get_idx(j, i)
            option_values[idx] = max(0.0, payoff_side*(S*(u**i)*(d**(j-i)) - X))

            if j < N:
                option_values[idx] = max(
                    option_values[idx], 
                    df * (pu*option_values[get_idx(j+1, i+1)] + pd*option_values[get_idx(j+1, i)])
                )
    
    return option_values[0]

def approx_delta(h, S, X, T, vol, r, b, N, option_type, richardson=True):
    """
    Approximates the Delta (sensitivity to underlying price) using finite differences.
    Optionally applies Richardson Extrapolation to improve accuracy.
    
    Parameters:
        h (float): The step size/bump amount for the underlying price.
        S, X, T, vol, r, b, N, option_type: Standard option pricing parameters.
        richardson (bool): If True, applies Richardson Extrapolation.
        
    Returns:
        float: The approximated Delta value.
    """
    def D1(h):
        v_up  = American_Binary_Tree(S + h, X, T, vol, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S - h, X, T, vol, r, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        return (4*D_h2 - D_h) / 3
    else: 
        return D1(h)

def approx_gamma(h, S, X, T, vol, r, b, N, option_type, richardson=True):
    """
    Approximates the Gamma (second-order sensitivity to underlying price) using finite differences.
    Optionally applies Richardson Extrapolation to improve accuracy.
    
    Parameters:
        h (float): The step size/bump amount for the underlying price.
        S, X, T, vol, r, b, N, option_type: Standard option pricing parameters.
        richardson (bool): If True, applies Richardson Extrapolation.
        
    Returns:
        float: The approximated Gamma value.
    """
    def D2(h):
        v_up  = American_Binary_Tree(S + h, X, T, vol, r, b, N=N, option_type=option_type)
        v_mid = American_Binary_Tree(S,     X, T, vol, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S - h, X, T, vol, r, b, N=N, option_type=option_type)
        return (v_up - 2*v_mid + v_dn) / (h**2)
    
    if richardson:
        D_h   = D2(h)
        D_h2  = D2(h/2)
        return (4*D_h2 - D_h) / 3
    else:
        return D2(h)

def approx_theta(h, S, X, T, vol, r, b, N, option_type, richardson=True):
    """
    Approximates the Theta (sensitivity to time to maturity) using finite differences.
    Optionally applies Richardson Extrapolation to improve accuracy.
    
    Parameters:
        h (float): The step size/bump amount for time to maturity (T).
        S, X, T, vol, r, b, N, option_type: Standard option pricing parameters.
        richardson (bool): If True, applies Richardson Extrapolation.
        
    Returns:
        float: The approximated Theta value.
    """
    def D1(h):
        v_up  = American_Binary_Tree(S, X, T+h, vol, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S, X, T-h, vol, r, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        return (4*D_h2 - D_h) / 3
    else: 
        return D1(h)

def approx_vega(h, S, X, T, vol, r, b, N, option_type, richardson=True):
    """
    Approximates the Vega (sensitivity to implied volatility) using finite differences.
    Optionally applies Richardson Extrapolation to improve accuracy.
    
    Parameters:
        h (float): The step size/bump amount for the volatility.
        S, X, T, vol, r, b, N, option_type: Standard option pricing parameters.
        richardson (bool): If True, applies Richardson Extrapolation.
        
    Returns:
        float: The approximated Vega value.
    """
    def D1(h):
        v_up  = American_Binary_Tree(S, X, T, vol+h, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S, X, T, vol-h, r, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        return (4*D_h2 - D_h) / 3
    else: 
        return D1(h)

def approx_rho(h, S, X, T, vol, r, b, N, option_type, richardson=True):
    def D1(h):
        v_up  = American_Binary_Tree(S, X, T, vol, r+h, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S, X, T, vol, r-h, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        return (4*D_h2 - D_h) / 3
    else: 
        return D1(h)


def process_american_options(input_file, N=200):
    """
    Reads the options universe, processes American metrics iteratively, and writes the output.
    """
    df = pd.read_csv(input_file)
    df.dropna(how='all', inplace=True) 
    
    results = []
    
    for index, row in df.iterrows():
        S = float(row['Underlying'])
        K = float(row['Strike'])
        T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
        r = float(row['RiskFreeRate'])
        q = float(row['DividendRate'])
        vol = float(row['ImpliedVol'])
        opt_type = str(row['Option Type']).strip()
        
        # cost of carry
        b = r - q
        
        # base Value
        val = American_Binary_Tree(S, K, T, vol, r, b, N, opt_type)
        
        # Define step sizes (h) for the finite difference bumping
        h_S = S * 0.01
        h_vol = 0.001
        h_r = 0.001
        h_T = 1.0 / 365.0
        
        delta = approx_delta(h_S, S, K, T, vol, r, b, N, opt_type)
        gamma = approx_gamma(h_S, S, K, T, vol, r, b, N, opt_type)
        vega  = approx_vega(h_vol, S, K, T, vol, r, b, N, opt_type)
        rho   = approx_rho(h_r, S, K, T, vol, r, b, N, opt_type)
        
        if T > h_T:
            theta = approx_theta(h_T, S, K, T, vol, r, b, N, opt_type)
        else:
            theta = 0.0
            
        results.append({
            'ID': int(row['ID']),
            'Value': val,
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega,
            'Rho': rho,
            'Theta': theta
        })
        
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    try:
        INPUT_FILE = 'data/test12_1.csv'
        process_american_options(INPUT_FILE, N=200)
        
    except FileNotFoundError:
        print('Data file not found. Please check the path.')
        
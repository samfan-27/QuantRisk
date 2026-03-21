import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_gbsm_metrics(S, K, T, r, q, sigma, opt_type):
    """
    Calculates the GBSM Option Value and Greeks.
    
    Parameters:
        S (float): Underlying asset price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        q (float): Continuous dividend yield.
        sigma (float): Implied volatility.
        opt_type (str): 'Call' or 'Put'.
        
    Returns:
        tuple: (Value, Delta, Gamma, Vega, Rho, Theta)
    """
    # Cost of Carry
    b = r - q
    
    if T <= 0:
        if opt_type.lower() == 'call':
            return max(0.0, S - K), 1.0 if S > K else 0.0, 0.0, 0.0, 0.0, 0.0
        elif opt_type.lower() == 'put':
            return max(0.0, K - S), -1.0 if S < K else 0.0, 0.0, 0.0, 0.0, 0.0
            
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)
    pdf_d1 = norm.pdf(d1)
    
    # Pre-compute recurring exponential components
    exp_br_T = np.exp((b - r) * T)
    exp_r_T = np.exp(-r * T)
    
    gamma = (exp_br_T * pdf_d1) / (S * sigma * np.sqrt(T))
    vega = S * exp_br_T * pdf_d1 * np.sqrt(T)
    
    if opt_type.lower() == 'call':
        value = S * exp_br_T * N_d1 - K * exp_r_T * N_d2
        delta = exp_br_T * N_d1
        rho = T * K * exp_r_T * N_d2
        theta = (-(S * exp_br_T * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                 - (b - r) * S * exp_br_T * N_d1 
                 - r * K * exp_r_T * N_d2)
                 
    elif opt_type.lower() == 'put':
        value = K * exp_r_T * N_minus_d2 - S * exp_br_T * N_minus_d1
        delta = exp_br_T * (N_d1 - 1)
        rho = -T * K * exp_r_T * N_minus_d2
        theta = (-(S * exp_br_T * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                 + (b - r) * S * exp_br_T * N_minus_d1 
                 + r * K * exp_r_T * N_minus_d2)
    else:
        value, delta, rho, theta = np.nan, np.nan, np.nan, np.nan 
    return value, delta, gamma, vega, rho, theta

def process_options_data(input_file, output_file):
    """
    Reads the options universe, processes GBSM metrics iteratively.
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
        sigma = float(row['ImpliedVol'])
        opt_type = str(row['Option Type']).strip()
        
        val, delta, gamma, vega, rho, theta = calculate_gbsm_metrics(
            S, K, T, r, q, sigma, opt_type
        )
        
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
        INPUT_FILE = "test12_1.csv"
        OUTPUT_FILE = "testout_12.1.csv"
        process_options_data(INPUT_FILE, OUTPUT_FILE)
        
    except FileNotFoundError:
        print("Data file not found. Please check the path.")
        
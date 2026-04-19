import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_gbsm(S, K, T, r, q, sigma, opt_type):
    """
    Calculates the Generalized Black-Scholes-Merton (GBSM) Option Value and Greeks.
    
    Parameters:
        S (float or np.array): Underlying asset price.
        K (float or np.array): Strike price.
        T (float or np.array): Time to maturity in years.
        r (float or np.array): Risk-free interest rate.
        q (float or np.array): Continuous dividend yield.
        sigma (float or np.array): Implied volatility.
        opt_type (str): 'Call' or 'Put'.
        
    Returns:
        dict: A dictionary containing Value, Delta, Gamma, Vega, Rho, and Theta.
    """
    # Cost of Carry
    b = r - q
    opt_type = str(opt_type).strip().lower()
    
    if T <= 0:
        if opt_type == 'call':
            val = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
        elif opt_type == 'put':
            val = max(0.0, K - S)
            delta = -1.0 if S < K else 0.0
        else:
            raise ValueError("opt_type must be 'call' or 'put'")
            
        return {'Value': val, 'Delta': delta, 'Gamma': 0.0, 'Vega': 0.0, 'Rho': 0.0, 'Theta': 0.0}
            
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)
    pdf_d1 = norm.pdf(d1)
    
    # pre-compute recurring exponential components
    exp_br_T = np.exp((b - r) * T)
    exp_r_T = np.exp(-r * T)
    
    gamma = (exp_br_T * pdf_d1) / (S * sigma * np.sqrt(T))
    vega = S * exp_br_T * pdf_d1 * np.sqrt(T)
    
    if opt_type == 'call':
        value = S * exp_br_T * N_d1 - K * exp_r_T * N_d2
        delta = exp_br_T * N_d1
        rho = T * K * exp_r_T * N_d2
        theta = (-(S * exp_br_T * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                 - (b - r) * S * exp_br_T * N_d1 
                 - r * K * exp_r_T * N_d2)
                 
    elif opt_type == 'put':
        value = K * exp_r_T * N_minus_d2 - S * exp_br_T * N_minus_d1
        delta = exp_br_T * (N_d1 - 1)
        rho = -T * K * exp_r_T * N_minus_d2
        theta = (-(S * exp_br_T * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                 + (b - r) * S * exp_br_T * N_minus_d1 
                 + r * K * exp_r_T * N_minus_d2)
    else:
        raise ValueError("opt_type must be 'call' or 'put'")
        
    return {
        'Value': value,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }

if __name__ == '__main__':
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
            
            metrics = calculate_gbsm(S, K, T, r, q, sigma, opt_type)
            
            row_result = {'ID': int(row['ID'])}
            row_result.update(metrics)
            results.append(row_result)
            
        results_df = pd.DataFrame(results)
        print('--- GBSM European Options Data ---')
        print(results_df.head(10).to_string(index=False))

    except FileNotFoundError:
        print('Data file not found. Ensure you are running from the Options/ directory.')
        
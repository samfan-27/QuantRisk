import numpy as np
from scipy.signal import lfilter

def simulate_ma(theta, n=1000, seed=None):
    """
    Simulates a Moving Average MA(q) process using an FIR filter.
    
    Parameters:
        theta (list or np.array): The MA coefficients [theta_1, theta_2, ..., theta_q].
        n (int): Number of observations to simulate.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.array: Simulated MA process of length n.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    
    # MA process is a FIR filter
    # y[t] = noise[t] + theta_1*noise[t-1] + ... + theta_q*noise[t-q]
    # Filter takes (b, a, x) where b are numerator coeffs, a are denominator coeffs
    b = np.r_[1, theta] 
    a = [1]
    
    ma_process = lfilter(b, a, noise)
    return ma_process

def simulate_ar(phi, n=1000, burn_in=100, seed=None):
    """
    Simulates an Autoregressive AR(p) process using an IIR filter.
    
    Parameters:
        phi (list or np.array): The AR coefficients [phi_1, phi_2, ..., phi_p].
        n (int): Number of observations to return.
        burn_in (int): Number of initial observations to discard to achieve stationarity.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.array: Simulated AR process of length n.
    """
    rng = np.random.default_rng(seed)
    
    total_n = n + burn_in
    noise = rng.standard_normal(total_n)
    
    # AR process is an IIR filter
    # y[t] - phi_1*y[t-1] - ... - phi_p*y[t-p] = noise[t]
    b = [1]
    a = np.r_[1, -np.array(phi)]
    
    ar_process = lfilter(b, a, noise)
    
    return ar_process[burn_in:]

def simulate_arma(ar_coeffs, ma_coeffs, n=1000, burn_in=100, seed=None):
    """
    Simulates an ARMA(p, q) process using an IIR filter.
    
    Parameters:
        ar_coeffs (list or np.array): The AR coefficients [phi_1, ..., phi_p].
        ma_coeffs (list or np.array): The MA coefficients [theta_1, ..., theta_q].
        n (int): Number of observations to return.
        burn_in (int): Number of initial observations to discard.
        seed (int, optional): Random seed.
        
    Returns:
        np.array: Simulated ARMA process of length n.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n + burn_in)
    
    # Numerator coefficients (MA part): [1, theta_1, theta_2, ...]
    b = np.r_[1, np.array(ma_coeffs)] if ma_coeffs is not None else [1]
    
    # Denominator coefficients (AR part): [1, -phi_1, -phi_2, ...]
    a = np.r_[1, -np.array(ar_coeffs)] if ar_coeffs is not None else [1]
    
    arma_process = lfilter(b, a, noise)
    return arma_process[burn_in:]

def simulate_arima(ar_coeffs, d, ma_coeffs, n=1000, burn_in=100, seed=None):
    """
    Simulates an ARIMA(p, d, q) process.
    
    Parameters:
        ar_coeffs (list or np.array): The AR coefficients.
        d (int): The degree of differencing (integration order).
        ma_coeffs (list or np.array): The MA coefficients.
        n (int): Number of observations to return.
        burn_in (int): Number of initial observations to discard.
        seed (int, optional): Random seed.
        
    Returns:
        np.array: Simulated ARIMA process.
    """
    process = simulate_arma(ar_coeffs, ma_coeffs, n=n, burn_in=burn_in, seed=seed)
    
    for _ in range(d):
        process = np.cumsum(process)
        
    return process

def simulate_garch(omega, alpha, beta, n=1000, burn_in=100, seed=None):
    """
    Simulates a GARCH(1,1) volatility process.
    
    Parameters:
        omega (float): Constant term (must be > 0).
        alpha (float): ARCH coefficient (reaction to past shocks).
        beta (float): GARCH coefficient (persistence of past volatility).
        n (int): Number of observations to return.
        burn_in (int): Number of initial observations to discard.
        seed (int, optional): Random seed.
        
    Returns:
        tuple: (returns_array, volatilities_array)
    """
    if alpha + beta >= 1.0:
        raise ValueError('alpha + beta must be strictly less than 1 for stationarity.')
        
    rng = np.random.default_rng(seed)
    total_n = n + burn_in
    z = rng.standard_normal(total_n)
    
    returns = np.zeros(total_n)
    sigma2 = np.zeros(total_n)
    
    sigma2[0] = omega / (1.0 - alpha - beta)
    returns[0] = np.sqrt(sigma2[0]) * z[0]
    
    for t in range(1, total_n):
        sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * z[t]
        
    return returns[burn_in:], np.sqrt(sigma2[burn_in:])

def simulate_gbm(s0, mu, sigma, t=1.0, n_steps=1000, seed=None):
    """
    Simulates a Geometric Brownian Motion (GBM) path for an asset price.
    
    Parameters:
        s0 (float): Initial stock price.
        mu (float): Expected annual return (drift).
        sigma (float): Annual volatility.
        t (float): Time horizon in years (e.g., 1.0 for 1 year).
        n_steps (int): Number of time steps.
        seed (int, optional): Random seed.
        
    Returns:
        np.array: Simulated price path of length (n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    dw = rng.standard_normal(n_steps) * np.sqrt(dt)
    w = np.cumsum(dw)
    
    time_array = np.linspace(dt, t, n_steps)
    
    # solution to the GBM SDE
    prices = s0 * np.exp((mu - 0.5 * sigma**2) * time_array + sigma * w)
    return np.r_[s0, prices]

def simulate_ou(x0, theta, mu, sigma, t=1.0, n_steps=1000, seed=None):
    """
    Simulates an Ornstein-Uhlenbeck (Mean-Reverting) process.
    
    Parameters:
        x0 (float): Initial value (e.g., initial interest rate or spread).
        theta (float): Speed of mean reversion.
        mu (float): Long-term mean.
        sigma (float): Volatility.
        t (float): Time horizon.
        n_steps (int): Number of time steps.
        seed (int, optional): Random seed.
        
    Returns:
        np.array: Simulated path of length (n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    x = np.zeros(n_steps + 1)
    x[0] = x0
    
    z = rng.standard_normal(n_steps)
    
    # Euler-Maruyama approximation for the SDE: dx_t = theta*(mu - x_t)dt + sigma*dW_t
    for i in range(n_steps):
        x[i+1] = x[i] + theta * (mu - x[i]) * dt + sigma * np.sqrt(dt) * z[i]
        
    return x

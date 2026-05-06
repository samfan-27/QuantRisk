import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from typing import Dict, Any, Optional, Union

def estimate_arima(
    data: Union[np.ndarray, pd.Series], 
    p: int, 
    d: int, 
    q: int, 
    exog: Optional[Union[np.ndarray, pd.DataFrame]] = None
) -> Dict[str, Any]:
    """
    Estimates AR, MA, ARMA, ARIMA, or ARIMAX parameters.
    
    Parameters:
        data (np.ndarray or pd.Series): The endog. variable (e.g., asset returns or prices).
        p (int): Autoregressive order.
        d (int): Degree of differencing.
        q (int): Moving average order.
        exog (np.ndarray or pd.DataFrame, optional): Exog. variables for ARIMAX.
        
    Returns:
        dict: Fitted parameters including AR, MA, Exog. coefficients, and variance.
    """
    model = sm.tsa.ARIMA(endog=data, exog=exog, order=(p, d, q))
    fitted_model = model.fit()
    
    result = {
        'ar_coeffs': fitted_model.arparams.tolist() if p > 0 else [],
        'ma_coeffs': fitted_model.maparams.tolist() if q > 0 else [],
        'sigma2': float(fitted_model.sigma2)
    }
    
    if exog is not None:
        if isinstance(exog, pd.DataFrame):
            exog_names = exog.columns
        else:
            exog_names = [f'exog_{i}' for i in range(np.shape(exog)[1])]
            
        result['exog_coeffs'] = {
            name: float(fitted_model.params[name]) 
            for name in exog_names if name in fitted_model.params
        }
        
    return result

def estimate_garch(returns: Union[np.ndarray, pd.Series], p: int = 1, q: int = 1) -> Dict[str, float]:
    """
    Estimates GARCH(p,q) parameters from historical returns using Max Likelihood.
    
    Parameters:
        returns (np.ndarray or pd.Series): Historical asset returns.
        p (int): Lag order of the symmetric innovation (ARCH term).
        q (int): Lag order of lagged volatility (GARCH term).
        
    Returns:
        dict: Fitted parameters (omega, alpha, beta).
    """
    # Rescale returns for optimization stability
    rescaled_returns = returns * 100
    
    model = arch_model(rescaled_returns, vol='Garch', p=p, q=q, mean='Zero')
    fitted_model = model.fit(disp='off')
    
    return {
        'omega': float(fitted_model.params['omega'] / 10000),
        'alpha': float(fitted_model.params['alpha[1]']),
        'beta': float(fitted_model.params['beta[1]'])
    }

def estimate_gbm(prices: Union[np.ndarray, pd.Series], dt: float = 1/252) -> Dict[str, float]:
    """
    Estimates GBM parameters.
    
    Parameters:
        prices (np.ndarray or pd.Series): Historical price series.
        dt (float): Time step in years (default assumes daily data, 252 trading days).
        
    Returns:
        dict: Estimated annual drift (mu) and annual volatility (sigma).
    """
    prices_array = np.array(prices)
    log_returns = np.diff(np.log(prices_array))
    
    mean_log_ret = np.mean(log_returns)
    var_log_ret = np.var(log_returns, ddof=1)
    
    sigma = np.sqrt(var_log_ret / dt)
    mu = (mean_log_ret / dt) + (0.5 * sigma**2)
    
    return {
        'mu': float(mu),
        'sigma': float(sigma)
    }

def estimate_ou(data: Union[np.ndarray, pd.Series], dt: float = 1/252) -> Dict[str, float]:
    """
    Estimates OU parameters using OLS on the discretized Euler-Maruyama equation.
    
    Parameters:
        data (np.ndarray or pd.Series): Historical time series (e.g., interest rates, spreads).
        dt (float): Time step in years.
        
    Returns:
        dict: Estimated parameters (theta, mu, sigma).
    """
    # OU: x[t] - x[t-1] = theta * (mu - x[t-1]) * dt + error
    # -> linear regression: x[t] = a + b * x[t-1] + error
    # where b = 1 - theta * dt, and a = theta * mu * dt
    
    data_array = np.array(data)
    x_t = data_array[1:]
    x_t_minus_1 = data_array[:-1]
    
    X = sm.add_constant(x_t_minus_1)
    model = sm.OLS(x_t, X).fit()
    
    a, b = model.params
    residuals = model.resid
    theta = (1 - b) / dt
    
    if np.isclose(theta, 0.0):
        mu = 0.0
    else:
        mu = a / (theta * dt)
    
    sigma = np.std(residuals, ddof=1) / np.sqrt(dt)
    
    return {
        'theta': float(theta),
        'mu': float(mu),
        'sigma': float(sigma)
    }
    
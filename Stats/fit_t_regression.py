import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

def fit_t_regression(X, y, nu_init=5.0):
    """
    Fits a linear regression model assuming t-distributed errors.
    
    Parameters:
    X : np.array or pd.DataFrame
        Predictor variables (n_samples, n_features).
    y : np.array or pd.Series
        Target variable (n_samples,).
    nu_init : float
        Initial guess for the degrees of freedom.
        
    Returns:
    dict
        Dictionary containing optimized Sigma, Nu, Alpha, and Betas.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    X_bias = np.c_[np.ones(X.shape[0]), X]
    beta_ols = np.linalg.lstsq(X_bias, y, rcond=None)[0]
    residuals = y - X_bias.dot(beta_ols)
    sigma_init = np.std(residuals)
    
    initial_params = np.concatenate([beta_ols, [np.log(sigma_init), np.log(nu_init)]])
    
    def t_regression_nll(params):
        alpha = params[0]
        betas = params[1:-2]
        log_sigma = params[-2]
        log_nu = params[-1]
        
        sigma = np.exp(log_sigma)
        nu = np.exp(log_nu)
        
        y_pred = alpha + X.dot(betas)
        res = y - y_pred
        
        n = len(y)
        term1 = gammaln((nu+1)/2) - gammaln(nu/2)
        term2 = -0.5 * np.log(np.pi * nu) - np.log(sigma)
        term3 = - ((nu+1)/2) * np.log(1 + (res**2) / (nu * sigma**2))
        
        ll = n * term1 + n * term2 + np.sum(term3)
        return -ll

    result = minimize(t_regression_nll, initial_params, method='BFGS')
    
    if not result.success:
        print('Warning: Optimization failed.')
        
    opt_params = result.x
    alpha_opt = opt_params[0]
    betas_opt = opt_params[1:-2]
    sigma_opt = np.exp(opt_params[-2])
    nu_opt = np.exp(opt_params[-1])
    
    return {
        'sigma': sigma_opt,
        'nu': nu_opt,
        'alpha': alpha_opt,
        'betas': betas_opt
    }

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('../data/test7_3.csv')
    X_data = df[['x1', 'x2', 'x3']].values
    y_data = df['y'].values
    
    results = fit_t_regression(X_data, y_data)
    print('Fitted Parameters:', results)
    
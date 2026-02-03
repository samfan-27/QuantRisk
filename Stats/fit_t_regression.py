import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

df = pd.read_csv('data/test7_3.csv')
X = df[['x1', 'x2', 'x3']].values
y = df['y'].values

# Initial guesses using OLS
X_bias = np.c_[np.ones(X.shape[0]), X]
beta_ols = np.linalg.lstsq(X_bias, y, rcond=None)[0]
residuals = y - X_bias.dot(beta_ols)
sigma_init = np.std(residuals)
nu_init = 5.0

# Initial parameter vector: [Alpha, B1, B2, B3, log(sigma), log(nu)]
initial_params = np.concatenate([beta_ols, [np.log(sigma_init), np.log(nu_init)]])

def t_regression_nll(params):
    alpha, b1, b2, b3, log_sigma, log_nu = params
    sigma = np.exp(log_sigma)
    nu = np.exp(log_nu)
    
    y_pred = alpha + b1 * X[:, 0] + b2 * X[:, 1] + b3 * X[:, 2]
    res = y - y_pred
    
    n = len(y)
    # Log-likelihood of t-distribution
    term1 = gammaln((nu+1)/2) - gammaln(nu/2)
    term2 = -0.5 * np.log(np.pi * nu) - np.log(sigma)
    term3 = - (nu+1)/2 * np.log(1 + (res/sigma)**2 / nu)
    
    log_lik = n * (term1 + term2) + np.sum(term3)
    return -log_lik

result = minimize(t_regression_nll, initial_params, method='BFGS', options={'gtol': 1e-9})

params = result.x
alpha_est, b1_est, b2_est, b3_est = params[:4]
sigma_est = np.exp(params[4])
nu_est = np.exp(params[5])

print(f"Alpha: {alpha_est}")
print(f"B1: {b1_est}")
print(f"B2: {b2_est}")
print(f"B3: {b3_est}")
print(f"sigma: {sigma_est}")
print(f"nu: {nu_est}")


import statsmodels.miscmodels.tmodel as tmodel

model = tmodel.TLinearModel(y, X_bias)
result = model.fit()
print(result.summary())

import numpy as np


def simulate_ma(q, theta, n=1000):
    noise = np.random.normal(size=n)
    ma_process = noise.copy()
    
    for t in range(q, n):
        ma_process[t] = noise[t] + sum(theta[i] * noise[t - i - 1] for i in range(q))
        
    return ma_process


def simulate_ar(p, phi, n=1000):
    noise = np.random.normal(size=n)
    ar_process = np.zeros(n)
    for t in range(p, n):
        ar_process[t] = noise[t] + sum(phi[i] * ar_process[t - i - 1] for i in range(p))
    return ar_process

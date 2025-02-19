import numpy as np


def calculate_exponential_covariance(df, lambd):
    df_reversed = df.iloc[::-1]
    n = len(df_reversed)
    weights = np.array([(1 - lambd) * (lambd ** i) for i in range(n)])
    weights /= weights.sum()
    weighted_mean = (df_reversed * weights[:, np.newaxis]).sum(axis=0)
    deviations = df_reversed - weighted_mean
    cov_matrix = deviations.T @ (deviations * weights[:, np.newaxis])
    return cov_matrix

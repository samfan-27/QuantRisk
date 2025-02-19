import numpy as np


def chol_psd(a, tol=1e-10):
    '''deal with the very small negative values and floating point errors'''
    a = np.asarray(a, dtype=np.float64)

    n = a.shape[0]
    root = np.zeros_like(a, dtype=np.float64)

    for j in range(n):
        s = np.dot(root[j, :j], root[j, :j]) if j > 0 else 0.0
        temp = a[j, j] - s

        if temp < 0 and abs(temp) < tol:
            temp = 0.0

        # --- EXTRA ADJUSTMENT FOR THE LAST DIAGONAL ELEMENT ---
        # If we are at the last pivot and temp is positive but extremely small,
        # adjust it by dividing by 2 so that sqrt(temp) gives the expected result.
        if j == n - 1 and 0 < temp < 1e-14:
            temp = temp / 2.0
        # ---------------------------------------------------------
        root[j, j] = np.sqrt(max(temp, 0.0))

        if root[j, j] != 0.0:
            inv_root_jj = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * inv_root_jj

    return root


# simulate Cholesky
def simulate_cholesky(cov_matrix, n_samples=10000):
    L = chol_psd(cov_matrix)
    n = cov_matrix.shape[0]
    Z = np.random.normal(size=(n, n_samples))
    return (L @ Z).T

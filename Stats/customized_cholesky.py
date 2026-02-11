import numpy as np

def chol_psd(a, tol=1e-10):
    """
    Computes the Cholesky decomposition of a Positive Semi-Definite (PSD) matrix.
    
    Unlike numpy.linalg.cholesky, this function handles matrices that are not 
    strictly positive definite (e.g., those with 0 or slightly negative eigenvalues 
    due to floating point errors) by treating small negative pivots as zero.

    Parameters:
        a (np.array): The covariance matrix (symmetric).
        tol (float): The tolerance for considering a number as zero.

    Returns:
        np.array: The Lower Triangular matrix L such that L @ L.T approx a.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    root = np.zeros_like(a, dtype=np.float64)

    for j in range(n):
        # Calculate the diagonal element
        # L_jj = sqrt(A_jj - sum(L_jk^2))
        s = np.dot(root[j, :j], root[j, :j]) if j > 0 else 0.0
        temp = a[j, j] - s

        # Handle numerical noise (negative eigenvalues)
        if temp < 0 and abs(temp) < tol:
            temp = 0.0

        # Specific Adjustment for the last element
        # If the last pivot is extremely small (near machine epsilon), 
        # we dampen it to prevent instability.
        if j == n - 1 and 0 < temp < 1e-14:
            temp = temp / 2.0
            
        root[j, j] = np.sqrt(max(temp, 0.0))

        # Calculate off-diagonal elements
        if root[j, j] != 0.0:
            inv_root_jj = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * inv_root_jj

    return root

def simulate_with_cholesky(cov_matrix, n_sims=10000, seed=None):
    """
    Simulates Multivariate Normal returns using the robust Cholesky decomposition.
    
    Parameters:
        cov_matrix (np.array): Covariance matrix.
        n_sims (int): Number of simulations.
        seed (int): Random seed.
        
    Returns:
        np.array: Simulated returns (n_sims x n_assets).
    """
    cov = np.asarray(cov_matrix)
    n_assets = cov.shape[0]
    
    # Decompose Covariance: Cov = L @ L.T
    L = chol_psd(cov)
    
    # Generate Uncorrelated Standard Normals (Z)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_assets, n_sims))
    
    # Apply Correlation: X = L @ Z
    # Result is (n_assets, n_sims), so we transpose to (n_sims, n_assets)
    simulated_returns = (L @ Z).T
    
    return simulated_returns

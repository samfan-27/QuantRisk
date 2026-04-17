import numpy as np

def near_psd(a, epsilon=0.0):
    """
    Finds the nearest PSD matrix using the Rebonato and Jackel method.
    
    Parameters:
        a (np.array): The input square matrix (typically a correlation matrix).
        epsilon (float): Minimum bound for negative eigenvalues to ensure PSD.
        
    Returns:
        np.array: The nearest PSD matrix.
    """
    a = np.asarray(a)
    n = a.shape[0]
    
    eigval, eigvec = np.linalg.eigh(a)
    val = np.maximum(eigval, epsilon)
    
    # Rebonato-Jackel scaling matrix T
    # Vectorized: T_i = 1 / sum(eigvec_{ij}^2 * val_j)
    t_inv = np.sum((eigvec ** 2) * val, axis=1)
    t = np.diag(np.sqrt(1.0 / t_inv))
    
    b = t @ eigvec @ np.diag(np.sqrt(val))
    
    out = b @ b.T
    return out

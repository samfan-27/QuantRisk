import numpy as np

# Higham’s method
# Find the nearest positive semi-definite matrix using Higham’s method
# Adapt from https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
# https://blogs.sas.com/content/iml/2012/11/28/computing-the-nearest-correlation-matrix.html
def _proj_s(x):
    """Projection onto the PSD cone."""
    eigvals, eigvecs = np.linalg.eigh(x)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def _proj_u(x):
    """Projection onto matrices with unit diagonals."""
    y = x.copy()
    np.fill_diagonal(y, 1.0)
    return y

def _mat_inf_norm(a):
    """Calculates the infinity norm of a matrix."""
    return np.max(np.abs(a).sum(axis=1))

def higham_nearest_correlation(a, max_iter=100, tol=1e-8):
    """
    Finds the nearest PSD correlation matrix 
    using Higham's alternating projections method.
    """
    y_old, x_old, ds = a.copy(), a.copy(), np.zeros_like(a)
    
    for _ in range(max_iter):
        r = y_old - ds
        x = _proj_s(r)
        ds = x - r
        y = _proj_u(x)
        
        # check convergence
        max_d = max(_mat_inf_norm(x - x_old) / _mat_inf_norm(x),
                    _mat_inf_norm(y - y_old) / _mat_inf_norm(y),
                    _mat_inf_norm(y - x) / _mat_inf_norm(y))
        
        if max_d <= tol:
            break
            
        x_old, y_old = x.copy(), y.copy()
        
    return x

def fix_non_psd_covariance(cov_matrix, max_iter=100, tol=1e-8):
    """
    Fixes a non-PSD Covariance matrix
    by applying Higham's method to its underlying Correlation matrix.
    
    Parameters:
        cov_matrix (np.ndarray): The input covariance matrix (potentially non-PSD).
        max_iter (int): Maximum iterations for Higham's method.
        tol (float): Convergence tolerance.
        
    Returns:
        np.ndarray: The nearest PSD covariance matrix.
    """
    cov_matrix = np.asarray(cov_matrix)
    
    # Extract variances and calculate volatilities
    variances = np.diag(cov_matrix)
    volatilities = np.sqrt(np.maximum(variances, 0))
    
    # Convert Covariance to Correlation Matrix
    outer_vols = np.outer(volatilities, volatilities)
    corr_matrix = np.divide(cov_matrix, outer_vols, 
                            out=np.zeros_like(cov_matrix), 
                            where=outer_vols!=0)
    
    np.fill_diagonal(corr_matrix, 1.0)
    
    psd_corr = higham_nearest_correlation(corr_matrix, max_iter=max_iter, tol=tol)
    
    psd_cov = psd_corr * outer_vols
    return psd_cov

def is_psd(matrix, tol=1e-8):
    """Utility to check if a matrix is Positive Semi-Definite."""
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= -tol)

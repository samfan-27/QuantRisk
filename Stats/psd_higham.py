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

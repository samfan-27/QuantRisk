import numpy as np


# Higham’s method
# Find the nearest positive semi-definite matrix using Higham’s method
# Adapt from https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
# https://blogs.sas.com/content/iml/2012/11/28/computing-the-nearest-correlation-matrix.html
def ProjS(X):
    eigvals, eigvecs = np.linalg.eigh(X)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def ProjU(X):
    Y = X.copy()
    np.fill_diagonal(Y, 1)
    return Y

def MatInfNorm(A):
    return np.max(np.abs(A).sum(axis=1))

def Higham_nearest_correlation(A, max_iter=100, tol=1e-8):
    Yold, Xold, dS = A.copy(), A.copy(), np.zeros_like(A)
    for _ in range(max_iter):
        R = Yold - dS
        X = ProjS(R)
        dS = X - R
        Y = ProjU(X)
        # Check convergence
        maxd = max(MatInfNorm(X-Xold)/MatInfNorm(X),
                   MatInfNorm(Y-Yold)/MatInfNorm(Y),
                   MatInfNorm(Y-X)/MatInfNorm(Y))
        if maxd <= tol:
            break
        Xold, Yold = X.copy(), Y.copy()
    return X

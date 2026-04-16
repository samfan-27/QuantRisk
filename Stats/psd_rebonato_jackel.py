import numpy as np


# Find the nearest positive semi-definite matrix using Rebenato and Jackel method
def nearPSD(A, epsilon=0):
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    
    out = B * B.T
    return out

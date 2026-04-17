import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .ewma_covariance_correlation import calculate_ew_covariance_matrix

def exp_weighted_pca(returns, lambdas, plot=False):
    """
    Performs PCA on an exponentially weighted covariance matrix using manual eigendecomposition.
    """
    results = {}
    
    for lambd in lambdas:
        cov_matrix = calculate_ew_covariance_matrix(returns, lambd)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        
        eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        
        cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
        results[lambd] = cumulative_variance
        
        if plot:
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=f'Manual λ={lambd}')
    
    if plot:
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return results

def sklearn_weighted_pca(returns, lambdas, plot=False):
    """
    Performs PCA on an exponentially weighted dataset using scikit-learn.
    """
    results = {}
    for lambd in lambdas:
        n = len(returns)
        
        
        weights = (1 - lambd) * (lambd ** np.arange(n))
        weights = weights[::-1]
        weights /= weights.sum()
        
        weighted_mean = np.sum(returns.values * weights[:, np.newaxis], axis=0)
        centered_data = returns.values - weighted_mean
        weighted_data = centered_data * np.sqrt(weights[:, np.newaxis])
        
        pca = PCA()
        pca.fit(weighted_data)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        results[lambd] = cumulative_variance
        
        if plot:
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=f'Sklearn λ={lambd:.2f}')

    if plot:
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return results

def simulate_pca(cov_matrix, n_samples=10000, var_explained=0.75):
    """
    Simulates returns based on the principal components that explain a target variance.
    """
    const = 1
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vals, eigen_vecs = eigen_vals[idx], eigen_vecs[:, idx]
    
    cumulative_var = np.cumsum(eigen_vals) / np.sum(eigen_vals)
    k = np.argmax(cumulative_var >= var_explained) + const
    
    B = eigen_vecs[:, :k] @ np.diag(np.sqrt(eigen_vals[:k]))
    Z = np.random.normal(size=(k, n_samples))
    return (B @ Z).T

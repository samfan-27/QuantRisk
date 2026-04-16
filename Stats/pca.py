import numpy as np
import matplotlib.pyplot as plt
from exponential_covariance_mat import calculate_exponential_covariance


# Manual PCA
def manualPCA(df, lambdas):
    for lambd in lambdas:
        cov_matrix = calculate_exponential_covariance(df, lambd)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        
        # Ensure non-negative eigenvalues (clip negative values to 0)
        eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)
        
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
        
        # Plot the cumulative variance for the current lambda
        plt.plot(cumulative_variance, label=f'λ={lambd}')
    
    # Display the legend and plot
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
# sklearn PCA
from sklearn.decomposition import PCA

def sklearnPCA(lambdas, returns):
    for lambd in lambdas:
        n = len(returns)
        
        # Compute the weights for this lambda
        weights = (1 - lambd) * (lambd ** np.arange(n))
        weights = weights[::-1]
        weights /= weights.sum()  # Normalize the weights
        
        # Prepare weighted data matrix
        weighted_mean = np.sum(returns.values * weights[:, np.newaxis], axis=0)
        centered_data = returns.values - weighted_mean
        weighted_data = centered_data * np.sqrt(weights[:, np.newaxis])
        
        # Apply PCA
        pca = PCA()
        pca.fit(weighted_data)
        
        # Compute cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Plot the cumulative variance
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=f'λ={lambd:.2f}')

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.show()
    

# simulate PCA
def simulate_pca(cov_matrix, n_samples=10000, var_explained=0.75):
    const = 1
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vals, eigen_vecs = eigen_vals[idx], eigen_vecs[:, idx]
    
    cumulative_var = np.cumsum(eigen_vals) / np.sum(eigen_vals)
    k = np.argmax(cumulative_var >= var_explained) + const
    
    B = eigen_vecs[:, :k] @ np.diag(np.sqrt(eigen_vals[:k]))
    Z = np.random.normal(size=(k, n_samples))
    return (B @ Z).T

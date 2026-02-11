import numpy as np
import pandas as pd
import time

def simulate_multivariate_normal(cov_matrix, n_sims=100000, seed=None):
    """
    Performs a Monte Carlo simulation of a Multivariate Normal distribution.

    Assumes a mean of 0 for all variables, which is standard for 
    return-based risk modeling (random walk hypothesis).

    Parameters:
        cov_matrix (pd.DataFrame or np.array): The covariance matrix (Positive Definite).
        n_sims (int): Number of simulations to perform (default 100,000).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.array: A (n_sims x n_assets) array of simulated returns.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.values
    else:
        cov = np.asarray(cov_matrix)

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    rng = np.random.default_rng(seed)
    
    mean = np.zeros(cov.shape[0])

    # Simulation
    # method='cholesky' is generally faster and stable for PD matrices
    # but numpy's default (SVD-based) is more robust to slightly non-PD matrices.
    # We stick to the default for robustness.
    try:
        simulated_data = rng.multivariate_normal(mean, cov, size=n_sims, method='cholesky')
    except np.linalg.LinAlgError:
        print("Warning: Matrix is not Positive Definite. Falling back to SVD method.")
        simulated_data = rng.multivariate_normal(mean, cov, size=n_sims, method='svd')

    return simulated_data

def compare_matrices(matrix_a, matrix_b, title="Comparison"):
    """
    Calculates the difference between two matrices.
    """
    if isinstance(matrix_a, pd.DataFrame):
        matrix_a = matrix_a.values
    if isinstance(matrix_b, pd.DataFrame):
        matrix_b = matrix_b.values

    diff = matrix_a - matrix_b
    frobenius_norm = np.linalg.norm(diff)
    
    print(f"\n--- {title} ---")
    print(f"Frobenius Norm (Total Difference): {frobenius_norm:.6f}")
    print(f"Max Absolute Element Difference:   {np.max(np.abs(diff)):.6f}")

def main():
    try:
        input_file = 'data/test5_1.csv'
        input_cov_df = pd.read_csv(input_file)
        
        start_time = time.time()
        
        n_sims = 100000
        print(f"Running {n_sims} simulations based on {input_file}...")
        simulated_returns = simulate_multivariate_normal(input_cov_df, n_sims=n_sims, seed=3)
        
        # Calculate Output Covariance from Simulation
        # Rowvar=False because rows are observations (simulations), cols are assets
        simulated_cov = np.cov(simulated_returns, rowvar=False)
        
        simulated_cov_df = pd.DataFrame(
            simulated_cov, 
            index=input_cov_df.columns, 
            columns=input_cov_df.columns
        )
        
        elapsed = time.time() - start_time
        print(f"Simulation complete in {elapsed:.4f} seconds.")

        print("\nInput Covariance")
        print(input_cov_df)
        
        print("\nSimulated Covariance")
        print(simulated_cov_df)

        compare_matrices(input_cov_df, simulated_cov_df, title="Input vs. Simulated Covariance")

    except FileNotFoundError:
        print("Data file not found. Please check the path.")

if __name__ == "__main__":
    main()
    
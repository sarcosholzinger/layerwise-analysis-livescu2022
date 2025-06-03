import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute CKA (Centered Kernel Alignment) between two representations.
    
    Args:
        X: (n_samples, n_features1)
        Y: (n_samples, n_features2)
    
    Returns:
        CKA similarity score
    """
    # Center the matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # Compute Gram matrices (linear kernel)
    K = X @ X.T  # (n_samples, n_samples)
    L = Y @ Y.T  # (n_samples, n_samples)
    
    # Center the Gram matrices
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = np.trace(K_centered @ L_centered) / (n - 1)**2
    
    # Compute normalization
    var_K = np.trace(K_centered @ K_centered) / (n - 1)**2
    var_L = np.trace(L_centered @ L_centered) / (n - 1)**2
    
    # Compute CKA
    cka = hsic / np.sqrt(var_K * var_L + 1e-8)
    
    return cka


def compute_cka_without_padding(features1: np.ndarray, features2: np.ndarray, 
                               orig_lens1: List[int], orig_lens2: List[int]) -> float:
    """
    Compute CKA excluding padded time steps.
    
    Args:
        features1, features2: Padded features (batch, time, dim)
        orig_lens1, orig_lens2: Lists of original lengths for each batch item
    """
    all_X = []
    all_Y = []
    
    batch_size = features1.shape[0]
    
    for b in range(batch_size):
        # Get original length for this batch item
        orig_len1 = orig_lens1[b] if b < len(orig_lens1) else features1.shape[1]
        orig_len2 = orig_lens2[b] if b < len(orig_lens2) else features2.shape[1]
        orig_len = min(orig_len1, orig_len2)  # Use minimum to ensure both are valid
        
        # Extract only non-padded time steps
        X = features1[b, :orig_len, :]  # (time, dim)
        Y = features2[b, :orig_len, :]
        
        all_X.append(X)
        all_Y.append(Y)
    
    # Concatenate all valid samples
    X = np.vstack(all_X)  # (total_valid_samples, dim)
    Y = np.vstack(all_Y)
    
    # Ensure same number of samples
    min_samples = min(X.shape[0], Y.shape[0])
    X = X[:min_samples]
    Y = Y[:min_samples]
    
    return compute_cka(X, Y)


def compute_partial_correlation(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """
    Compute partial correlation between X and Y given Z.
    
    Partial correlation r_{XY|Z} = correlation between X and Y after removing 
    the linear effect of Z from both X and Y.
    
    Args:
        X: (n_samples, n_features_X)
        Y: (n_samples, n_features_Y)
        Z: (n_samples, n_features_Z) - conditioning variable (CNN output)
    
    Returns:
        Partial correlation coefficient
    """
    # Ensure 2D arrays
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    
    # Regress out Z from X
    reg_X = LinearRegression()
    reg_X.fit(Z, X)
    residuals_X = X - reg_X.predict(Z)
    
    # Regress out Z from Y
    reg_Y = LinearRegression()
    reg_Y.fit(Z, Y)
    residuals_Y = Y - reg_Y.predict(Z)
    
    # Compute correlation between residuals
    # Flatten residuals for correlation computation
    residuals_X_flat = residuals_X.flatten()
    residuals_Y_flat = residuals_Y.flatten()
    
    if len(residuals_X_flat) > 1 and len(residuals_Y_flat) > 1:
        partial_corr = np.corrcoef(residuals_X_flat, residuals_Y_flat)[0, 1]
    else:
        partial_corr = 0.0
    
    return partial_corr


def compute_conditional_cka(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                           method: str = 'residual') -> float:
    """
    Compute CKA between X and Y conditioned on Z (CNN output).
    
    Two methods available:
    1. 'residual': Compute CKA on residuals after regressing out Z
    2. 'partial': Compute partial CKA using kernel matrices
    
    Args:
        X: (n_samples, n_features_X)
        Y: (n_samples, n_features_Y)
        Z: (n_samples, n_features_Z) - CNN output features
        method: 'residual' or 'partial'
    
    Returns:
        Conditional CKA score
    """
    if method == 'residual':
        # Method 1: Regress out Z from both X and Y, then compute CKA on residuals
        
        # Regress out Z from X
        reg_X = LinearRegression()
        reg_X.fit(Z, X)
        X_residual = X - reg_X.predict(Z)
        
        # Regress out Z from Y
        reg_Y = LinearRegression()
        reg_Y.fit(Z, Y)
        Y_residual = Y - reg_Y.predict(Z)
        
        # Compute CKA on residuals
        return compute_cka(X_residual, Y_residual)
    
    elif method == 'partial':
        # Method 2: Partial CKA using kernel regression
        
        # Center all matrices
        X_c = X - X.mean(axis=0, keepdims=True)
        Y_c = Y - Y.mean(axis=0, keepdims=True)
        Z_c = Z - Z.mean(axis=0, keepdims=True)
        
        # Compute Gram matrices
        K_X = X_c @ X_c.T
        K_Y = Y_c @ Y_c.T
        K_Z = Z_c @ Z_c.T
        
        # Center Gram matrices
        n = K_X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_X_centered = H @ K_X @ H
        K_Y_centered = H @ K_Y @ H
        K_Z_centered = H @ K_Z @ H
        
        # Compute partial kernel matrices (remove effect of Z)
        # Using the formula: K_{X|Z} = K_X - K_{XZ} K_Z^{-1} K_{ZX}
        # We use regularized inverse for numerical stability
        epsilon = 1e-5
        K_Z_inv = np.linalg.inv(K_Z_centered + epsilon * np.eye(n))
        
        K_X_partial = K_X_centered - K_X_centered @ K_Z_inv @ K_Z_centered
        K_Y_partial = K_Y_centered - K_Y_centered @ K_Z_inv @ K_Z_centered
        
        # Compute CKA on partial kernels
        hsic_XY = np.trace(K_X_partial @ K_Y_partial) / (n - 1)**2
        hsic_XX = np.trace(K_X_partial @ K_X_partial) / (n - 1)**2
        hsic_YY = np.trace(K_Y_partial @ K_Y_partial) / (n - 1)**2
        
        # Compute conditional CKA
        conditional_cka = hsic_XY / np.sqrt(hsic_XX * hsic_YY + 1e-8)
        
        return conditional_cka
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)


def compute_r_squared(X: np.ndarray, Z: np.ndarray) -> float:
    """
    Compute R² showing how much variance in X is explained by Z.
    
    Args:
        X: Target features (n_samples, n_features_X)
        Z: Predictor features (n_samples, n_features_Z)
    
    Returns:
        Average R² across all dimensions of X
    """
    r2_per_dim = []
    for dim in range(X.shape[1]):
        reg = LinearRegression()
        reg.fit(Z, X[:, dim])
        r2 = reg.score(Z, X[:, dim])
        r2_per_dim.append(max(0, r2))  # Ensure non-negative
    
    return np.mean(r2_per_dim) 
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Optional
# import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    
    # Set optimal GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
except ImportError:
    TORCH_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Import the proper sorting function
from _utils.data_utils import get_layer_number


def _setup_gpu_device(gpu_id: int = 0, memory_fraction: float = 0.9) -> str:
    """Setup GPU device with memory optimization"""
    if not TORCH_AVAILABLE:
        return 'cpu'
    
    if torch.cuda.is_available():
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=gpu_id)
        return device
    return 'cpu'


def compute_cka_gpu(X: np.ndarray, Y: np.ndarray, device: str = 'cuda', 
                   memory_efficient: bool = True, chunk_size: int = 1000) -> float:
    """
    GPU-accelerated CKA computation optimized for V100s.
    
    Args:
        X: (n_samples, n_features1)
        Y: (n_samples, n_features2)
        device: Device to use ('cuda', 'cuda:0', etc.)
        memory_efficient: Use chunked computation for large matrices
        chunk_size: Chunk size for memory-efficient computation
    
    Returns:
        CKA similarity score
    """
    if not TORCH_AVAILABLE:
        return compute_cka(X, Y)
    
    try:
        # Convert to tensors
        X_torch = torch.tensor(X, device=device, dtype=torch.float32)
        Y_torch = torch.tensor(Y, device=device, dtype=torch.float32)
        
        n_samples = X_torch.shape[0]
        
        # Use chunked computation for large matrices to avoid OOM
        if n_samples > chunk_size and memory_efficient:
            result = _compute_cka_chunked_gpu(X_torch, Y_torch, chunk_size)
        else:
            result = _compute_cka_direct_gpu(X_torch, Y_torch)
        
        # Clear GPU memory
        del X_torch, Y_torch
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        # Fallback to CPU
        return compute_cka(X, Y)


def _compute_cka_direct_gpu(X_torch: torch.Tensor, Y_torch: torch.Tensor) -> float:
    """Direct GPU CKA computation."""
    n = X_torch.shape[0]
    
    # Center the data
    X_centered = X_torch - X_torch.mean(dim=0, keepdim=True)
    Y_centered = Y_torch - Y_torch.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices
    K = torch.mm(X_centered, X_centered.t())
    L = torch.mm(Y_centered, Y_centered.t())
    
    # Center Gram matrices
    H = torch.eye(n, device=X_torch.device) - torch.ones(n, n, device=X_torch.device) / n
    K_centered = torch.mm(torch.mm(H, K), H)
    L_centered = torch.mm(torch.mm(H, L), H)
    
    # Compute CKA
    hsic = torch.trace(torch.mm(K_centered, L_centered)) / (n - 1)**2
    var_K = torch.trace(torch.mm(K_centered, K_centered)) / (n - 1)**2
    var_L = torch.trace(torch.mm(L_centered, L_centered)) / (n - 1)**2
    
    cka = hsic / torch.sqrt(var_K * var_L + 1e-8)
    
    return cka.cpu().item()


def _compute_cka_chunked_gpu(X_torch: torch.Tensor, Y_torch: torch.Tensor, 
                            chunk_size: int) -> float:
    """Memory-efficient chunked CKA computation."""
    n = X_torch.shape[0]
    device = X_torch.device
    
    # Center the data
    X_centered = X_torch - X_torch.mean(dim=0, keepdim=True)
    Y_centered = Y_torch - Y_torch.mean(dim=0, keepdim=True)
    
    # Compute Gram matrix elements in chunks
    K = torch.zeros(n, n, device=device)
    L = torch.zeros(n, n, device=device)
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            
            # Compute chunk of Gram matrix
            K[i:end_i, j:end_j] = torch.mm(
                X_centered[i:end_i], X_centered[j:end_j].t()
            )
            L[i:end_i, j:end_j] = torch.mm(
                Y_centered[i:end_i], Y_centered[j:end_j].t()
            )
    
    # Center and compute CKA
    H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
    K_centered = torch.mm(torch.mm(H, K), H)
    L_centered = torch.mm(torch.mm(H, L), H)
    
    hsic = torch.trace(torch.mm(K_centered, L_centered)) / (n - 1)**2
    var_K = torch.trace(torch.mm(K_centered, K_centered)) / (n - 1)**2
    var_L = torch.trace(torch.mm(L_centered, L_centered)) / (n - 1)**2
    
    cka = hsic / torch.sqrt(var_K * var_L + 1e-8)
    
    return cka.cpu().item()


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    TEST 4 (CPU)
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
    Test 5 (CPU): Compute CKA excluding padded time steps.
    
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


def compute_partial_correlation_gpu(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                   device: str = 'cuda') -> float:
    """
    TEST 0 (GPU)
    GPU-accelerated partial correlation computation.
    """
    if not TORCH_AVAILABLE:
        return compute_partial_correlation(X, Y, Z)
    
    try:
        X_torch = torch.tensor(X, device=device, dtype=torch.float32)
        Y_torch = torch.tensor(Y, device=device, dtype=torch.float32)
        Z_torch = torch.tensor(Z, device=device, dtype=torch.float32)
        
        # Regress out Z from X and Y using GPU-accelerated linear regression
        X_residual = _regress_out_gpu(Z_torch, X_torch)
        Y_residual = _regress_out_gpu(Z_torch, Y_torch)
        
        # Compute correlation between residuals
        X_flat = X_residual.flatten() #TODO: Check if this is correct!
        Y_flat = Y_residual.flatten() #TODO: Check if this is correct!
        
        correlation = torch.corrcoef(torch.stack([X_flat, Y_flat]))[0, 1]
        result = correlation.cpu().item()
        
        # Clear GPU memory
        del X_torch, Y_torch, Z_torch, X_residual, Y_residual, X_flat, Y_flat, correlation
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        return compute_partial_correlation(X, Y, Z)


def compute_partial_correlation_multi_gpu(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                        n_gpus: int = None) -> float:
    """ 
    TEST 1 (multi-GPU) #TODO: May be removed in the future!
    Multi-GPU accelerated partial correlation computation.
    """ #TODO: Check if this is working as expected! New function with multi-GPU parallelization hasn't been tested yet!
    
    if not TORCH_AVAILABLE:
        return compute_partial_correlation(X, Y, Z)
    
    try:
        import torch
        if not torch.cuda.is_available():
            return compute_partial_correlation(X, Y, Z)
        
        available_gpus = torch.cuda.device_count()
        if n_gpus is None:
            n_gpus = available_gpus
        else:
            n_gpus = min(n_gpus, available_gpus)
        
        if n_gpus <= 1:
            # Fallback to single GPU
            return compute_partial_correlation_gpu(X, Y, Z)
        
        # Multi-GPU strategy: Split the regression operations across GPUs
        # GPU 0: Regress Z from X
        # GPU 1: Regress Z from Y  
        # GPU 2+: Assist with matrix operations if needed
        
        results = {}
        
        def regress_worker(gpu_id, target, predictor, result_key):
            """Worker function for regression on specific GPU."""
            try:
                device = f'cuda:{gpu_id}'
                target_torch = torch.tensor(target, device=device, dtype=torch.float32)
                predictor_torch = torch.tensor(predictor, device=device, dtype=torch.float32)
                
                residual = _regress_out_gpu(predictor_torch, target_torch)
                results[result_key] = residual.cpu().numpy()
                
                # Clear GPU memory
                del target_torch, predictor_torch, residual
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"GPU {gpu_id} regression failed: {e}")
                results[result_key] = None
        
        # Parallel regression on multiple GPUs
        from multiprocessing import Process, Manager
        
        manager = Manager()
        results = manager.dict()
        processes = []
        
        # GPU 0: Regress Z from X
        p1 = Process(target=regress_worker, args=(0, X, Z, 'X_residual'))
        processes.append(p1)
        p1.start()
        
        # GPU 1: Regress Z from Y (if available)
        if n_gpus > 1:
            p2 = Process(target=regress_worker, args=(1, Y, Z, 'Y_residual'))
            processes.append(p2)
            p2.start()
        
        # Wait for completion
        for p in processes:
            p.join()
        
        # Check results and compute correlation
        X_residual = results.get('X_residual')
        Y_residual = results.get('Y_residual')
        
        if X_residual is None:
            # Fallback to CPU for X regression
            try:
                # Try CPU-based PyTorch regression first
                X_torch = torch.tensor(X, dtype=torch.float32)
                Z_torch = torch.tensor(Z, dtype=torch.float32)
                X_residual = _regress_out_gpu(Z_torch, X_torch).numpy()
            except Exception as e:
                # Complete fallback to pure CPU (scikit-learn)
                print(f"PyTorch CPU fallback failed for X regression: {e}, using scikit-learn")
                from sklearn.linear_model import LinearRegression
                reg_X = LinearRegression() #TODO: This may be a problem for large matrices!! Possibly remove this fallback!
                reg_X.fit(Z, X)
                X_residual = X - reg_X.predict(Z)
        
        if Y_residual is None:
            # Fallback to CPU for Y regression  
            try:
                # Try CPU-based PyTorch regression first
                Y_torch = torch.tensor(Y, dtype=torch.float32)
                Z_torch = torch.tensor(Z, dtype=torch.float32)
                Y_residual = _regress_out_gpu(Z_torch, Y_torch).numpy()
            except Exception as e:
                # Complete fallback to pure CPU (scikit-learn)
                print(f"PyTorch CPU fallback failed for Y regression: {e}, using scikit-learn")
                from sklearn.linear_model import LinearRegression 
                reg_Y = LinearRegression()  #TODO: This may be a problem for large matrices!! Possibly remove this fallback!
                reg_Y.fit(Z, Y)
                Y_residual = Y - reg_Y.predict(Z)
        
        # Compute final correlation (on GPU 0)
        try:
            device = 'cuda:0'
            X_flat = torch.tensor(X_residual.flatten(), device=device, dtype=torch.float32)     #TODO: Check if this is correct!
            Y_flat = torch.tensor(Y_residual.flatten(), device=device, dtype=torch.float32)     #TODO: Check if this is correct!
            
            correlation = torch.corrcoef(torch.stack([X_flat, Y_flat]))[0, 1]
            result = correlation.cpu().item()
            
            # Clear GPU memory
            del X_flat, Y_flat, correlation
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            # Final fallback to CPU correlation
            X_flat = X_residual.flatten() #TODO: Check if this is correct!  
            Y_flat = Y_residual.flatten() #TODO: Check if this is correct!  
            if len(X_flat) > 1 and len(Y_flat) > 1:
                return np.corrcoef(X_flat, Y_flat)[0, 1]
            else:
                return 0.0
                
    except Exception as e:
        print(f"Multi-GPU partial correlation failed: {e}, falling back to single GPU")
        return compute_partial_correlation_gpu(X, Y, Z)


def _regress_out_gpu(Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated regression using torch.linalg.lstsq."""
    # Add bias term
    Z_with_bias = torch.cat([Z, torch.ones(Z.shape[0], 1, device=Z.device)], dim=1)
    
    # Solve least squares: Z_with_bias @ beta = X
    try:
        beta = torch.linalg.lstsq(Z_with_bias, X).solution
    except:
        # Fallback to pinverse if lstsq fails
        beta = torch.pinverse(Z_with_bias) @ X
    
    # Compute residuals
    X_pred = torch.mm(Z_with_bias, beta)
    residuals = X - X_pred
    
    return residuals


def compute_partial_correlation(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """
    TEST 0 (CPU)
    Compute partial correlation between X and Y given Z.
    
    This computes the correlation between X and Y after removing the linear 
    influence of Z from both X and Y. The result shows how X and Y correlate 
    when the effect of Z is "partialed out" or controlled for.
    
    Formula: r_{XY|Z} = correlation(residuals_X, residuals_Y)
    where residuals_X = X - predict(X|Z) and residuals_Y = Y - predict(Y|Z)
    
    NOTE: This is NOT measuring how Z correlates with X or Y individually.
    This measures the remaining correlation between X and Y after accounting for Z.
    
    Args:
        X: (n_samples, n_features_X) - First variable
        Y: (n_samples, n_features_Y) - Second variable  
        Z: (n_samples, n_features_Z) - Conditioning variable (e.g., CNN output)
    
    Returns:
        Partial correlation coefficient between X and Y given Z
    """
    # Ensure 2D arrays
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    
    # Regress out Z from X
    reg_X = LinearRegression()  # Linear regression is standard for partial correlation
    reg_X.fit(Z, X)
    residuals_X = X - reg_X.predict(Z)
    
    # Regress out Z from Y
    reg_Y = LinearRegression()
    reg_Y.fit(Z, Y)
    residuals_Y = Y - reg_Y.predict(Z)
    
    # Compute correlation between residuals
    # Flatten residuals for correlation computation
    residuals_X_flat = residuals_X.flatten() #TODO: Check if this is correct!
    residuals_Y_flat = residuals_Y.flatten() #TODO: Check if this is correct!
    
    if len(residuals_X_flat) > 1 and len(residuals_Y_flat) > 1:
        partial_corr = np.corrcoef(residuals_X_flat, residuals_Y_flat)[0, 1]
    else:
        partial_corr = 0.0
    
    return partial_corr


def _compute_layer_correlation_worker(args):
    """Worker function for parallel layer correlation computation."""
    layer_name, layer_features, input_features = args
    
    # Ensure same number of samples
    min_samples = min(input_features.shape[0], layer_features.shape[0])
    input_batch = input_features[:min_samples]
    layer_batch = layer_features[:min_samples]
    
    # Flatten for correlation
    input_flat = input_batch.flatten() #TODO: Check if this is correct!
    layer_flat = layer_batch.flatten() #TODO: Check if this is correct!
    
    # Ensure same length for correlation #TODO: Check for edge cases and add warnings catch_warnings
    min_length = min(len(input_flat), len(layer_flat))
    input_truncated = input_flat[:min_length]
    layer_truncated = layer_flat[:min_length]
    
    # Compute correlation #TODO: Check for edge cases
    if len(input_truncated) > 1 and len(layer_truncated) > 1:
        try:
            correlation = np.corrcoef(input_truncated, layer_truncated)[0, 1]
            return (layer_name, correlation)
        except:
            return (layer_name, 0.0)
    else:
        return (layer_name, 0.0)


def _compute_layer_correlation_gpu_worker(args):
    """GPU worker function for layer correlation computation."""
    gpu_id, layer_names, layer_features_list, input_features = args
    
    results = {}
    
    try:
        import torch
        device = f'cuda:{gpu_id}'
        
        # Move input features to GPU once
        input_tensor = torch.tensor(input_features, device=device, dtype=torch.float32)
        input_flat = input_tensor.flatten() #TODO: Check if this is correct!
        
        for layer_name, layer_features in zip(layer_names, layer_features_list):
            try:
                # Ensure same number of samples
                min_samples = min(input_features.shape[0], layer_features.shape[0])
                layer_batch = layer_features[:min_samples]
                
                # Move to GPU and flatten
                layer_tensor = torch.tensor(layer_batch, device=device, dtype=torch.float32)
                layer_flat = layer_tensor.flatten() #TODO: Check if this is correct!
                
                # Ensure same length
                min_length = min(input_flat.shape[0], layer_flat.shape[0])
                input_truncated = input_flat[:min_length]
                layer_truncated = layer_flat[:min_length]
                
                # Compute correlation on GPU
                if min_length > 1:
                    correlation = torch.corrcoef(torch.stack([input_truncated, layer_truncated]))[0, 1]
                    results[layer_name] = correlation.cpu().item()
                else:
                    results[layer_name] = 0.0
                    
                # Clear GPU memory
                del layer_tensor, layer_flat
                
            except Exception as e:
                print(f"Error computing correlation for {layer_name} on GPU {gpu_id}: {e}")
                results[layer_name] = 0.0
        
        # Clear GPU memory
        del input_tensor, input_flat
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"GPU {gpu_id} failed, falling back to CPU: {e}")
        # Fallback to CPU computation
        for layer_name, layer_features in zip(layer_names, layer_features_list):
            min_samples = min(input_features.shape[0], layer_features.shape[0])
            input_batch = input_features[:min_samples]
            layer_batch = layer_features[:min_samples]
            
            input_flat = input_batch.flatten()  #TODO: Check if this is correct!
            layer_flat = layer_batch.flatten()  #TODO: Check if this is correct!
            
            min_length = min(len(input_flat), len(layer_flat))
            if min_length > 1:
                try:
                    correlation = np.corrcoef(input_flat[:min_length], layer_flat[:min_length])[0, 1]
                    results[layer_name] = correlation
                except:
                    results[layer_name] = 0.0
            else:
                results[layer_name] = 0.0
    
    return results


def compute_input_layer_correlations(input_features: np.ndarray, 
                                   layer_features_dict: Dict[str, np.ndarray],
                                   layer_order: List[str] = None,
                                   show_progress: bool = False,
                                   n_jobs: int = -1,
                                   use_gpu: bool = True,
                                   n_gpus: int = None) -> Dict[str, float]:
    """
    TEST 1 (multi-GPU): Compute correlation between original input and each layer's output with multi-GPU parallelization.
    TODO: Check if this is working as expected! New function with multi-GPU may not be needed or working as expected!
    
    This tracks how the original input signal correlates with representations
    as information propagates through the network layers. Useful for understanding
    information retention/transformation across layers.
    
    Args
        input_features: (n_samples, n_input_features) - Original input features
        layer_features_dict: Dictionary mapping layer names to their features
        layer_order: Optional list specifying the order of layers for analysis
        show_progress: Whether to show progress bar with tqdm
        n_jobs: Number of parallel jobs (-1 for all cores, ignored if use_gpu=True)
        use_gpu: Whether to use GPU acceleration
        n_gpus: Number of GPUs to use (None for all available)
    
    Returns:
        Dictionary mapping layer names to correlation with input
    """
    if layer_order is None:
        # fix string sorting vs numerical sorting of layer names
        layer_order = sorted(layer_features_dict.keys(), key=get_layer_number)
    
    correlations = {}
    
    # GPU Implementation
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                if n_gpus is None:
                    n_gpus = available_gpus
                else:
                    n_gpus = min(n_gpus, available_gpus)
                
                if n_gpus > 0:
                    if show_progress:
                        print(f"Computing correlations using {n_gpus} GPUs...")
                    
                    # Distribute layers across GPUs
                    layers_per_gpu = len(layer_order) // n_gpus
                    gpu_tasks = []
                    
                    for gpu_id in range(n_gpus):
                        start_idx = gpu_id * layers_per_gpu
                        if gpu_id == n_gpus - 1:  # Last GPU gets remaining layers
                            end_idx = len(layer_order)
                        else:
                            end_idx = (gpu_id + 1) * layers_per_gpu
                        
                        gpu_layers = layer_order[start_idx:end_idx]
                        if gpu_layers:  # Only create task if there are layers to process
                            gpu_layer_features = [layer_features_dict[layer] for layer in gpu_layers]
                            gpu_tasks.append((gpu_id, gpu_layers, gpu_layer_features, input_features))
                    
                    # Parallel execution across GPUs
                    if len(gpu_tasks) > 1:
                        from multiprocessing import Pool
                        with Pool(processes=len(gpu_tasks)) as pool:
                            gpu_results = pool.map(_compute_layer_correlation_gpu_worker, gpu_tasks)
                    else:
                        # Single GPU or single task
                        gpu_results = [_compute_layer_correlation_gpu_worker(gpu_tasks[0])]
                    
                    # Combine results from all GPUs
                    for gpu_result in gpu_results:
                        correlations.update(gpu_result)
                    
                    return correlations
                else:
                    print("No GPUs available, falling back to CPU")
                    use_gpu = False
            else:
                print("CUDA not available, falling back to CPU")
                use_gpu = False
        except ImportError:
            print("PyTorch not available, falling back to CPU")
            use_gpu = False
    
    # CPU Implementation (fallback or explicitly requested)
    if JOBLIB_AVAILABLE and n_jobs != 1:
        # Use joblib for parallel processing
        if show_progress:
            print(f"Computing correlations in parallel with {n_jobs} CPU jobs...")
        
        # Prepare tasks
        tasks = [(layer_name, layer_features_dict[layer_name], input_features) 
                for layer_name in layer_order]
        
        # Parallel execution
        results = Parallel(n_jobs=n_jobs, verbose=1 if show_progress else 0)(
            delayed(_compute_layer_correlation_worker)(task) for task in tasks
        )
        
        # Collect results
        for layer_name, correlation in results:
            correlations[layer_name] = correlation
            
    else:
        # Sequential computation with optional progress bar
        layer_iter = tqdm(layer_order, desc="Simple correlations") if show_progress else layer_order
        
        for layer_name in layer_iter:
            layer_features = layer_features_dict[layer_name]
            
            # Ensure same number of samples
            min_samples = min(input_features.shape[0], layer_features.shape[0])
            input_batch = input_features[:min_samples]
            layer_batch = layer_features[:min_samples]
            
            # Flatten for correlation
            input_flat = input_batch.flatten() #TODO: Check if this is correct!
            layer_flat = layer_batch.flatten() #TODO: Check if this     is correct!
            
            # Ensure same length for correlation
            min_length = min(len(input_flat), len(layer_flat))
            input_truncated = input_flat[:min_length]
            layer_truncated = layer_flat[:min_length]
            
            # Compute correlation
            if len(input_truncated) > 1 and len(layer_truncated) > 1:
                try:
                    correlation = np.corrcoef(input_truncated, layer_truncated)[0, 1]
                    correlations[layer_name] = correlation
                except:
                    correlations[layer_name] = 0.0
            else:
                correlations[layer_name] = 0.0
    
    return correlations




def compute_progressive_partial_correlations(input_features: np.ndarray,
                                           layer_features_dict: Dict[str, np.ndarray],
                                           layer_order: List[str] = None,
                                           show_progress: bool = False,
                                           n_jobs: int = 1,
                                           use_gpu: bool = True,
                                           n_gpus: int = None) -> Dict[str, float]:
    """
     TEST 2: Compute partial correlations between input and each layer, controlling for previous layers.
    
    This computes:
    - r(input, layer_0): Simple correlation
    - r(input, layer_1 | layer_0): Partial correlation controlling for layer_0
    - r(input, layer_2 | layer_0, layer_1): Partial correlation controlling for layer_0 and layer_1
    - etc.
    
    This shows how much NEW correlation each layer adds beyond what previous layers explain.
    
    Args:
        input_features: (n_samples, n_input_features) - Original input features
        layer_features_dict: Dictionary mapping layer names to their features  
        layer_order: Optional list specifying the order of layers for analysis
        show_progress: Whether to show progress bar with tqdm
        n_jobs: Number of parallel jobs (1 for sequential due to dependencies)
        use_gpu: Whether to use GPU acceleration
        n_gpus: Number of GPUs to use for internal computations (None for all available)
    
    Returns:
        Dictionary mapping layer names to progressive partial correlations
    """
    if layer_order is None:
        #fix string sorting vs numerical sorting of layer names
        layer_order = sorted(layer_features_dict.keys(), key=get_layer_number)
    
    partial_correlations = {}
    
    # Check multi-GPU setup and fallback logic
    gpu_available = False
    actual_gpus = 0
    
    if use_gpu and TORCH_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                if available_gpus > 0:
                    gpu_available = True
                    if n_gpus is not None and n_gpus > 1:
                        actual_gpus = min(n_gpus, available_gpus)
                        if show_progress:
                            print(f"Using {actual_gpus} GPUs for internal partial correlation computations")
                    else:
                        actual_gpus = 1
                        if show_progress:
                            print("Using single GPU for partial correlation computations")
                else:
                    if show_progress:
                        print("No GPUs detected, falling back to CPU")
            else:
                if show_progress:
                    print("CUDA not available, falling back to CPU")
        except Exception as e:
            if show_progress:
                print(f"GPU initialization failed: {e}, falling back to CPU")
    
    if not gpu_available and show_progress:
        print("Using CPU for all partial correlation computations")
    
    # Note: This computation has dependencies, so parallel processing is limited
    # We can parallelize the heavy computation parts but need sequential structure
    
    layer_iter = tqdm(layer_order, desc="Partial correlations") if show_progress else enumerate(layer_order)
    if show_progress:
        layer_iter = enumerate(layer_iter)
    else:
        layer_iter = enumerate(layer_order)
    
    for i, current_layer in layer_iter:
        if show_progress:
            print(f"Processing layer {i}: {current_layer}")
        current_features = layer_features_dict[current_layer]
        
        # Ensure same number of samples
        min_samples = min(input_features.shape[0], current_features.shape[0])
        input_batch = input_features[:min_samples]
        current_batch = current_features[:min_samples]
        
        # Reshape to 2D for regression
        X = input_batch.reshape(-1, input_batch.shape[-1])  # Input
        Y = current_batch.reshape(-1, current_batch.shape[-1])  # Current layer
        
        if i == 0:
            # First layer: simple correlation
            try:
                correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1] #TODO: Check if this is correct!
                partial_correlations[current_layer] = correlation
            except:
                partial_correlations[current_layer] = 0.0
        else:
            # Later layers: partial correlation controlling for all previous layers
            previous_layers = [l for l in layer_order[:i] if l.startswith('transformer_layer_')]
            
            # Update progress description if showing progress
            if show_progress:
                tqdm.write(f"  Processing {current_layer} (controlling for {len(previous_layers)} previous layers)")
            
            # Concatenate all previous layer features as conditioning variables
            Z_list = []
            for prev_layer in previous_layers:
                prev_features = layer_features_dict[prev_layer][:min_samples]
                Z_list.append(prev_features.reshape(-1, prev_features.shape[-1]))
            
            if Z_list:
                Z = np.hstack(Z_list)  # Concatenate all previous layers
                
                # Ensure same number of samples
                min_samples_all = min(X.shape[0], Y.shape[0], Z.shape[0])
                X = X[:min_samples_all]
                Y = Y[:min_samples_all]
                Z = Z[:min_samples_all]
                
                try:
                    if gpu_available:
                        if actual_gpus > 1:
                            if show_progress:
                                tqdm.write(f"    Using {actual_gpus} GPUs for {current_layer}")
                            partial_corr = compute_partial_correlation_multi_gpu(X, Y, Z, actual_gpus)
                        else:
                            if show_progress:
                                tqdm.write(f"    Using single GPU for {current_layer}")
                            partial_corr = compute_partial_correlation_gpu(X, Y, Z)
                    else:
                        if show_progress:
                            tqdm.write(f"    Using CPU for {current_layer}")
                        partial_corr = compute_partial_correlation(X, Y, Z)
                    partial_correlations[current_layer] = partial_corr
                except Exception as e:
                    if show_progress:
                        tqdm.write(f"    Warning: Error computing partial correlation for {current_layer}: {e}")
                    partial_correlations[current_layer] = 0.0
            else:
                partial_correlations[current_layer] = 0.0
    
    return partial_correlations


# WARNING: This function is currently commented out due to potential issues with the regression step.
# It is still being used in:
# - _analysis/temporal_analysis.py
# - _analysis/similarity_analysis.py
# - gpu_layer_analysis.py
# - visualize_features.py
# Please update these files to use an alternative implementation or fix the regression step before uncommenting.
# def compute_conditional_cka(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
#                           method: str = 'residual') -> float:
#     """
#     Compute conditional CKA between X and Y given Z.
#     
#     Args:
#         X: First feature matrix
#         Y: Second feature matrix
#         Z: Conditioning variable matrix
#         method: 'residual' or 'partial' method
#     
#     Returns:
#         Conditional CKA value
#     """
#     if method == 'residual':
#         # Residual method: regress out Z from both X and Y
#         if USE_GPU and torch.cuda.is_available():
#             # GPU implementation
#             X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
#             Y_tensor = torch.tensor(Y, dtype=torch.float32, device='cuda')
#             Z_tensor = torch.tensor(Z, dtype=torch.float32, device='cuda')
#             
#             # Regress out Z from X
#             Z_pinv = torch.linalg.pinv(Z_tensor)
#             beta_X = torch.matmul(Z_pinv, X_tensor)
#             X_resid = X_tensor - torch.matmul(Z_tensor, beta_X)
#             
#             # Regress out Z from Y
#             beta_Y = torch.matmul(Z_pinv, Y_tensor)
#             Y_resid = Y_tensor - torch.matmul(Z_tensor, beta_Y)
#             
#             # Compute CKA on residuals
#             return compute_cka_gpu(X_resid, Y_resid)
#         else:
#             # CPU fallback
#             # Regress out Z from X #TODO: This may be wrong! Uncomment function after checking this condition on CKA!
#             reg_X = LinearRegression()
#             reg_X.fit(Z, X)
#             X_resid = X - reg_X.predict(Z)
#             
#             # Regress out Z from Y
#             reg_Y = LinearRegression()
#             reg_Y.fit(Z, Y)
#             Y_resid = Y - reg_Y.predict(Z)
#             
#             # Compute CKA on residuals
#             return compute_cka(X_resid, Y_resid)
#     else:
#         # Partial method: compute partial CKA
#         if USE_GPU and torch.cuda.is_available():
#             return compute_partial_cka_gpu(X, Y, Z)
#         else:
#             return compute_partial_cka(X, Y, Z)


# Commented out as these functions are being refactored
# def compute_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
#     """Compute cosine similarity between two vectors."""
#     return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
# 
# def compute_cosine_similarity_gpu(X: np.ndarray, Y: np.ndarray,
#                                 device: str = 'cuda') -> float:
#     """
#     Compute cosine similarity between two matrices using GPU acceleration.
#     
#     Args:
#         X: First matrix
#         Y: Second matrix
#         device: GPU device to use
#     
#     Returns:
#         Average cosine similarity
#     """
#     if USE_GPU and torch.cuda.is_available():
#         try:
#             # Convert to GPU tensors
#             X_torch = torch.tensor(X, device=device, dtype=torch.float32)
#             Y_torch = torch.tensor(Y, device=device, dtype=torch.float32)
#             
#             # Compute cosine similarity for each pair
#             cos_sims = []
#             for i in range(X.shape[0]):
#                 cos_sim = torch.dot(X_torch[i], Y_torch[i]) / (
#                     torch.norm(X_torch[i]) * torch.norm(Y_torch[i]) + 1e-8
#                 )
#                 cos_sims.append(cos_sim.item())
#             
#             # Clear GPU memory
#             del X_torch, Y_torch
#             torch.cuda.empty_cache()
#             
#             return np.mean(cos_sims)
#         except Exception as e:
#             # Fallback to CPU
#             pass
#     
#     # CPU fallback
#     cos_sims = []
#     for i in range(X.shape[0]):
#         cos_sim = compute_cosine_similarity(X[i], Y[i])
#         cos_sims.append(cos_sim)
#     return np.mean(cos_sims)


def _compute_r_squared_worker(args):
    """Worker function for parallel R² computation."""
    layer_name, layer_features, input_features = args
    
    # Ensure same number of samples
    min_samples = min(input_features.shape[0], layer_features.shape[0])
    input_batch = input_features[:min_samples]
    layer_batch = layer_features[:min_samples]
    
    # Reshape to 2D
    X = input_batch.reshape(-1, input_batch.shape[-1])  # Input as target
    Z = layer_batch.reshape(-1, layer_batch.shape[-1])  # Layer as predictor
    
    # Ensure same number of samples
    min_samples_all = min(X.shape[0], Z.shape[0])
    X = X[:min_samples_all]
    Z = Z[:min_samples_all]
    
    try:
        r2 = compute_r_squared(X, Z)  # How much variance in input is explained by layer
        return (layer_name, r2)
    except:
        return (layer_name, 0.0)


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


def analyze_input_propagation(input_features: np.ndarray,
                            layer_features_dict: Dict[str, np.ndarray],
                            layer_order: List[str] = None,
                            show_progress: bool = False,
                            n_jobs: int = -1,
                            use_gpu: bool = True,
                            n_gpus: int = None) -> Dict[str, Dict[str, float]]:
    """
    Complete analysis of how input information propagates through layers with parallelization.
    
    This combines three complementary analyses:
    1. Simple correlations: How much original input signal remains in each layer
    2. Progressive partial correlations: How much NEW signal each layer captures
    3. R² values: How much variance each layer explains in the input
    
    Args:
        input_features: (n_samples, n_input_features) - Original input features
        layer_features_dict: Dictionary mapping layer names to their features
        layer_order: Optional list specifying the order of layers for analysis
        show_progress: Whether to show progress bar with tqdm
        n_jobs: Number of parallel jobs (-1 for all cores)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dictionary containing all three types of analysis results
        
    Example:
        >>> # Assuming you have transformer_input (CNN output) and transformer layers
        >>> original_input = layer_features['transformer_input']  # CNN output
        >>> transformer_layers = {k: v for k, v in layer_features.items() 
        ...                      if k.startswith('transformer_layer_')}
        >>> results = analyze_input_propagation(original_input, transformer_layers, 
        ...                                    show_progress=True, n_jobs=-1, use_gpu=True)
        >>> 
        >>> # Plot how input correlation decays
        >>> import matplotlib.pyplot as plt
        >>> layers = results['layer_order']
        >>> simple_corrs = [results['simple_correlations'][l] for l in layers]
        >>> partial_corrs = [results['progressive_partial_correlations'][l] for l in layers]
        >>> r2_values = [results['r_squared_values'][l] for l in layers]
        >>> 
        >>> plt.figure(figsize=(12, 4))
        >>> plt.subplot(1, 3, 1)
        >>> plt.plot(simple_corrs, 'o-', label='Simple Correlation')
        >>> plt.title('Input Signal Retention')
        >>> plt.xlabel('Layer'); plt.ylabel('Correlation with Input')
        >>> 
        >>> plt.subplot(1, 3, 2) 
        >>> plt.plot(partial_corrs, 'o-', label='Partial Correlation', color='orange')
        >>> plt.title('New Information per Layer')
        >>> plt.xlabel('Layer'); plt.ylabel('Partial Correlation')
        >>> 
        >>> plt.subplot(1, 3, 3)
        >>> plt.plot(r2_values, 'o-', label='R²', color='green')
        >>> plt.title('Variance Explained')
        >>> plt.xlabel('Layer'); plt.ylabel('R²')
    """
    if layer_order is None:
        #fix string sorting vs numerical sorting of layer names
        layer_order = sorted(layer_features_dict.keys(), key=get_layer_number)
    
    if show_progress:
        print("Computing input propagation analysis...")
        if use_gpu and TORCH_AVAILABLE:
            gpu_count = torch.cuda.device_count()
            print(f"Using GPU acceleration with {gpu_count} GPU(s)")
        if n_jobs != 1:
            print(f"Using {n_jobs} parallel jobs")
    
    # 1. Simple correlations between input and each layer
    if show_progress:
        print("Step 1/3: Computing simple correlations...")
    simple_correlations = compute_input_layer_correlations(
        input_features, layer_features_dict, layer_order, show_progress, n_jobs, use_gpu
    )
    
    # 2. Progressive partial correlations 
    if show_progress:
        print("Step 2/3: Computing progressive partial correlations...")
    progressive_partial_correlations = compute_progressive_partial_correlations(
        input_features, layer_features_dict, layer_order, show_progress, 1, use_gpu, n_gpus  # Sequential due to dependencies
    )
    
    # 3. R² values (how much variance each layer explains in the input)
    if show_progress:
        print("Step 3/3: Computing R² values...")
    
    if JOBLIB_AVAILABLE and n_jobs != 1:
        # Parallel R² computation
        tasks = [(layer_name, layer_features_dict[layer_name], input_features) 
                for layer_name in layer_order]
        
        results = Parallel(n_jobs=n_jobs, verbose=1 if show_progress else 0)(
            delayed(_compute_r_squared_worker)(task) for task in tasks
        )
        
        r2_values = {}
        for layer_name, r2 in results:
            r2_values[layer_name] = r2
    else:
        # Sequential R² computation
        r2_values = {}
        layer_iter = tqdm(layer_order, desc="R² computation") if show_progress else layer_order
        
        for layer_name in layer_iter:
            layer_features = layer_features_dict[layer_name]
            
            # Ensure same number of samples
            min_samples = min(input_features.shape[0], layer_features.shape[0])
            input_batch = input_features[:min_samples]
            layer_batch = layer_features[:min_samples]
            
            # Reshape to 2D
            X = input_batch.reshape(-1, input_batch.shape[-1])  # Input as target
            Z = layer_batch.reshape(-1, layer_batch.shape[-1])  # Layer as predictor
            
            # Ensure same number of samples
            min_samples_all = min(X.shape[0], Z.shape[0])
            X = X[:min_samples_all]
            Z = Z[:min_samples_all]
            
            try:
                r2 = compute_r_squared(X, Z)  # How much variance in input is explained by layer
                r2_values[layer_name] = r2
            except:
                r2_values[layer_name] = 0.0
    
    if show_progress:
        print("✓ Input propagation analysis completed!")
    
    return {
        'simple_correlations': simple_correlations,
        'progressive_partial_correlations': progressive_partial_correlations,
        'r_squared_values': r2_values,
        'layer_order': layer_order,
        'interpretation': {
            'simple_correlations': 'Direct correlation between input and each layer (signal retention)',
            'progressive_partial_correlations': 'New correlation each layer adds beyond previous layers',
            'r_squared_values': 'Fraction of input variance explained by each layer'
        },
        'performance_info': {
            'gpu_acceleration': use_gpu and TORCH_AVAILABLE,
            'parallel_jobs': n_jobs,
            'num_layers': len(layer_order)
        }
    }


class CorrelationAnalyzer:
    """
    Unified class for all types of correlation analysis with GPU acceleration and parallelization.
    
    This consolidates:
    - Simple input-layer correlations
    - Progressive partial correlations
    - Conditional correlations (existing implementation)
    - R² analysis
    - CNN influence analysis
    """
    
    def __init__(self, layer_features: Dict[str, np.ndarray], 
                 original_lengths: Optional[Dict[str, List[int]]] = None,
                 cnn_layer: str = 'transformer_input',
                 max_layer: int = 11,
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 n_jobs: int = -1):
        """
        Initialize the analyzer with layer features.
        
        Args:
            layer_features: Dictionary of layer features
            original_lengths: Optional dictionary of original sequence lengths
            cnn_layer: Name of CNN output layer
            max_layer: Maximum transformer layer to include
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU ID to use (for multi-GPU systems)
            n_jobs: Number of parallel CPU jobs
        """
        from _utils.data_utils import create_layer_analysis_config, preprocess_feature_pairs
        
        self.layer_features = layer_features
        self.original_lengths = original_lengths
        self.cnn_layer = cnn_layer
        self.max_layer = max_layer
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.device = _setup_gpu_device(gpu_id) if self.use_gpu else 'cpu'
        self.n_jobs = n_jobs
        
        # Create analysis configuration
        self.config = create_layer_analysis_config(layer_features, True, max_layer)
        self.preprocess_feature_pairs = preprocess_feature_pairs
        
        # Validate CNN layer exists
        if self.config['cnn_features'] is None:
            raise ValueError(f"CNN layer '{cnn_layer}' not found in layer features")
        
        if self.use_gpu:
            print(f"CorrelationAnalyzer: Using GPU acceleration on {self.device}")
        if self.n_jobs != 1:
            print(f"CorrelationAnalyzer: Using {self.n_jobs} parallel CPU jobs")
    
    def compute_simple_input_correlations(self, show_progress: bool = False, n_gpus: int = None) -> Dict[str, float]:
        """ TEST 1: Compute simple correlations between input and each transformer layer."""
        return compute_input_layer_correlations(
            self.config['cnn_features'], 
            self.config['transformer_layers'],
            show_progress=show_progress,
            n_jobs=self.n_jobs,
            use_gpu=self.use_gpu,
            n_gpus=n_gpus
        )
    
    def compute_progressive_partial_correlations(self, show_progress: bool = False, n_gpus: int = None) -> Dict[str, float]:
        """Compute progressive partial correlations controlling for previous layers."""
        return compute_progressive_partial_correlations(
            self.config['cnn_features'],
            self.config['transformer_layers'],
            layer_order=self.config['transformer_layer_names'],  # Pass the layer order explicitly
            show_progress=show_progress,
            n_jobs=1,  # Sequential due to dependencies
            use_gpu=self.use_gpu,
            n_gpus=n_gpus
        )
    
    def compute_r_squared_analysis(self, show_progress: bool = False) -> Dict[str, float]:
        """Compute R² showing variance explained by each layer."""
        if JOBLIB_AVAILABLE and self.n_jobs != 1:
            # Parallel computation
            tasks = []
            for layer_name in self.config['transformer_layer_names']:
                layer_features = self.config['transformer_layers'][layer_name]
                tasks.append((layer_name, layer_features, self.config['cnn_features']))
            
            results = Parallel(n_jobs=self.n_jobs, verbose=1 if show_progress else 0)(
                delayed(_compute_r_squared_worker)(task) for task in tasks
            )
            
            r2_results = {}
            for layer_name, r2 in results:
                r2_results[layer_name] = r2
                
            return r2_results
        
        else:
            # Sequential computation
            r2_results = {}
            layer_iter = tqdm(self.config['transformer_layer_names'], desc="R² analysis") if show_progress else self.config['transformer_layer_names']
            
            for layer_name in layer_iter:
                layer_features = self.config['transformer_layers'][layer_name]
                
                # Preprocess features
                X, _, Z = self.preprocess_feature_pairs(
                    self.config['cnn_features'], None, layer_features,
                    self.original_lengths, self.cnn_layer, layer_name, self.cnn_layer
                )
                
                if X is not None and Z is not None:
                    r2 = compute_r_squared(X, Z)
                    r2_results[layer_name] = r2
                else:
                    r2_results[layer_name] = 0.0
            
            return r2_results
    
    # Commented out as this method is not currently being used in the codebase
    # def compute_layer_to_layer_correlations(self, layer1: str, layer2: str, 
    #                                       conditional: bool = False) -> float:
    #     """
    #     Compute correlation between two layers, optionally conditioned on CNN output.
    #     
    #     Args:
    #         layer1: First layer name
    #         layer2: Second layer name
    #         conditional: Whether to compute partial correlation conditioned on CNN
    #     
    #     Returns:
    #         Correlation coefficient
    #     """
    #     if layer1 not in self.layer_features or layer2 not in self.layer_features:
    #         return 0.0
    #     
    #     features1 = self.layer_features[layer1]
    #     features2 = self.layer_features[layer2]
    #     
    #     if conditional:
    #         # Conditional correlation (partial correlation given CNN output)
    #         X, Y, Z = self.preprocess_feature_pairs(
    #             features1, features2, self.config['cnn_features'],
    #             self.original_lengths, layer1, layer2, self.cnn_layer
    #         )
    #         
    #         if X is not None and Y is not None and Z is not None:
    #             if self.use_gpu:
    #                 return compute_partial_correlation_gpu(X, Y, Z, device=self.device)
    #             else:
    #                 return compute_partial_correlation(X, Y, Z)
    #     else:
    #         # Simple correlation
    #         X, Y, _ = self.preprocess_feature_pairs(
    #             features1, features2, None,
    #             self.original_lengths, layer1, layer2
    #         )
    #         
    #         if X is not None and Y is not None:
    #             try:
    #                 return np.corrcoef(X.flatten(), Y.flatten())[0, 1]
    #             except:
    #                 return 0.0
    #     
    #     return 0.0
    
    def compute_all_analyses(self, show_progress: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Compute all types of correlation analysis in one call with optimizations.
        
        Args:
            show_progress: Whether to show progress bars with tqdm
        
        Returns:
            Dictionary containing all analysis results
        """
        if show_progress:
            print("Computing all correlation analyses with optimizations...")
            print(f"Device: {self.device}, Parallel jobs: {self.n_jobs}")
        
        return {
            'simple_correlations': self.compute_simple_input_correlations(show_progress),
            'progressive_partial_correlations': self.compute_progressive_partial_correlations(show_progress),
            'r_squared_values': self.compute_r_squared_analysis(show_progress),
            'layer_order': self.config['transformer_layer_names'],
            'cnn_layer': self.cnn_layer,
            'interpretation': {
                'simple_correlations (Test 1)': 'Direct correlation between input and each layer (signal retention)',
                'progressive_partial_correlations (Test 2)': 'New correlation each layer adds beyond previous layers',
                'r_squared_values (Test 3)': 'Fraction of input variance explained by each layer'
            },
            'performance_info': { #TODO: This may not be needed as we're defaulting to CPU
                'gpu_acceleration': self.use_gpu,
                'device': self.device,
                'parallel_jobs': self.n_jobs,
                'num_layers': len(self.config['transformer_layer_names'])
            }
        }
    
    def get_layer_config(self) -> Dict[str, any]:
        """Get the layer analysis configuration."""
        return self.config 


# Utility functions for batch processing and memory management
def estimate_memory_usage(n_samples: int, n_features: int, dtype=np.float32) -> float:
    """Estimate memory usage in GB for a matrix."""
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = n_samples * n_features * bytes_per_element
    return total_bytes / (1024**3)  # Convert to GB


def optimize_chunk_size(n_samples: int, n_features: int, available_memory_gb: float = 8.0) -> int:
    """Optimize chunk size based on available memory."""
    # Estimate memory for Gram matrix (n_samples x n_samples)
    gram_memory_gb = estimate_memory_usage(n_samples, n_samples)
    
    if gram_memory_gb <= available_memory_gb:
        return n_samples  # No chunking needed
    
    # Calculate optimal chunk size
    optimal_chunk = int(np.sqrt(available_memory_gb / gram_memory_gb) * n_samples)
    return max(100, optimal_chunk)  # Minimum chunk size of 100


def get_optimal_batch_config(data_shapes: Dict[str, tuple], 
                           gpu_memory_gb: float = 16.0) -> Dict[str, int]:
    """Get optimal batch configuration for given data shapes and GPU memory."""
    config = {}
    
    for layer_name, shape in data_shapes.items():
        if len(shape) >= 2:
            n_samples, n_features = shape[0], np.prod(shape[1:])
            chunk_size = optimize_chunk_size(n_samples, n_features, gpu_memory_gb * 0.8)  # Use 80% of GPU memory
            config[layer_name] = chunk_size
    
    return config 


# Export all key functions for easy import
__all__ = [
    # Core correlation functions
    'compute_partial_correlation',
    'compute_partial_correlation_gpu',
    'compute_input_layer_correlations', 
    'compute_progressive_partial_correlations',
    'compute_r_squared',
    'analyze_input_propagation',
    
    # CKA functions (for potential future use)
    'compute_cka',
    'compute_cka_gpu',
    'compute_cka_without_padding',
    'compute_conditional_cka',
    
    # Similarity functions
    'compute_cosine_similarity',
    'compute_cosine_similarity_gpu',
    
    # Unified analyzer
    'CorrelationAnalyzer',
    
    # Utility functions
    'estimate_memory_usage',
    'get_optimal_batch_config',
    'optimize_chunk_size'
]


def verify_imports():
    """Verify that all key functions for visualization are accessible."""
    print("Verifying math_utils imports for visualization...")
    
    # Test basic functionality of each uncommented import
    test_functions = [
        'compute_partial_correlation',
        'compute_input_layer_correlations', 
        'compute_progressive_partial_correlations',
        'compute_r_squared',
        'analyze_input_propagation'
    ]
    
    for func_name in test_functions:
        func = globals().get(func_name)
        if func is None:
            print(f"  ✗ {func_name} not found")
        else:
            print(f"  ✓ {func_name} available")
    
    # Test CorrelationAnalyzer
    if CorrelationAnalyzer is not None:
        print(f"  ✓ CorrelationAnalyzer class available")
    else:
        print(f"  ✗ CorrelationAnalyzer class not found")
    
    print("Import verification complete!")


if __name__ == "__main__":
    # Only run verification if this file is executed directly
    verify_imports() 
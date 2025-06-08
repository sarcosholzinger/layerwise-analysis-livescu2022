"""
Layerwise Analysis Pipeline - GPU-Accelerated Analysis Module

This module provides high-performance, GPU-accelerated similarity analysis for layerwise
representations in HuBERT models. It implements memory-efficient parallel processing
and comprehensive visualization capabilities.

PURPOSE:
    Enable efficient analysis of large-scale layerwise representations through:
    - GPU-accelerated similarity computations
    - Memory-efficient processing of large matrices
    - Parallel processing across multiple GPUs
    - Comprehensive visualization and reporting

FUNCTIONALITY:
    1. Similarity Computations:
       - Correlation analysis
       - CKA (Centered Kernel Alignment)
       - Partial correlation
       - Memory-efficient chunked processing
    
    2. Parallel Processing:
       - Multi-GPU support
       - CPU fallback mechanisms
       - Adaptive batch sizing
       - Memory management
    
    3. Visualization:
       - Similarity matrices
       - Conditional vs unconditional analysis
       - CNN influence analysis
       - Summary reports

USAGE:
    python gpu_layer_analysis.py \
        --features_dir /path/to/features \
        --output_dir /path/to/output \
        --num_files 3 \
        --model_name HuBERT_Base \
        [--n_gpus 2] \
        [--memory_efficient]

STATUS:
    ACTIVE - This is a core performance-optimized module.
    
    INTEGRATION:
    - Used by main_layerwise_analysis.py for GPU-accelerated computations
    - Provides backend for similarity analysis in the pipeline
    - Supports both standalone and integrated usage
    
    DEPENDENCIES:
    - PyTorch (for GPU operations)
    - NumPy (for CPU fallback)
    - Matplotlib/Seaborn (for visualization)
    - CUDA-capable GPU(s)
    
    TODO:
    - Add support for more similarity metrics
    - Implement distributed processing across nodes
    - Add progress tracking for long-running analyses
    - Optimize memory usage further for very large models
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from matplotlib.animation import FuncAnimation, PillowWriter
import pickle
import time
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUParallelSimilarity:
    """GPU-accelerated similarity computation class."""
    
    def __init__(self, device: str = 'cuda', batch_size: int = 32, memory_efficient: bool = True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.memory_efficient = memory_efficient
        
        if self.device.startswith('cuda'):
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        logger.info(f"Initialized similarity computer on device: {self.device}")

    # Commented out as this function is being refactored
    # def compute_cosine_similarity_gpu(self, X: np.ndarray, Y: np.ndarray) -> float:
    #     """Compute cosine similarity using GPU acceleration."""
    #     try:
    #         return self._compute_cosine_similarity_gpu(X, Y)
    #     except Exception as e:
    #         logger.warning(f"GPU cosine similarity failed: {e}")
    #         return self._compute_cosine_similarity_cpu(X, Y)
    
    # def _compute_cosine_similarity_cpu(self, X: np.ndarray, Y: np.ndarray) -> float:
    #     """CPU fallback for cosine similarity."""
    #     # This function should probably also be commented out since the main cosine similarity is disabled
    #     if len(X.shape) == 3:
    #         X = X.mean(axis=1)
    #         Y = Y.mean(axis=1)
    #     
    #     cosine_sims = []
    #     for i in range(X.shape[0]):
    #         cos_sim = np.dot(X[i], Y[i]) / (np.linalg.norm(X[i]) * np.linalg.norm(Y[i]) + 1e-8)
    #         cosine_sims.append(cos_sim)
    #     
    #     return np.mean(cosine_sims)
    
    # def compute_cosine_similarity_gpu(self, X: np.ndarray, Y: np.ndarray) -> float: #TODO: This function is being refactored
    #     """GPU-accelerated cosine similarity computation."""
    #     try:
    #         X_torch = torch.tensor(X, device=self.device, dtype=torch.float32)
    #         Y_torch = torch.tensor(Y, device=self.device, dtype=torch.float32)
            
    #         # Time-average if 3D
    #         if len(X_torch.shape) == 3:
    #             X_torch = X_torch.mean(dim=1)  # (batch, features)
    #             Y_torch = Y_torch.mean(dim=1)
            
    #         # Normalize
    #         X_norm = F.normalize(X_torch, p=2, dim=1)
    #         Y_norm = F.normalize(Y_torch, p=2, dim=1)
            
    #         # Compute cosine similarity
    #         cosine_sim = torch.sum(X_norm * Y_norm, dim=1)
    #         result = cosine_sim.mean().cpu().item()
            
    #         # Clear GPU memory
    #         del X_torch, Y_torch, X_norm, Y_norm, cosine_sim
    #         if self.device.startswith('cuda'):
    #             torch.cuda.empty_cache()
            
    #         return result
            
    #     except Exception as e:
    #         logger.warning(f"GPU cosine similarity failed, falling back to CPU: {e}")
    #         return self._compute_cosine_similarity_cpu(X, Y)
    
    # def _compute_cosine_similarity_cpu(self, X: np.ndarray, Y: np.ndarray) -> float:
    #     """CPU fallback for cosine similarity."""
    #     if len(X.shape) == 3:
    #         X = X.mean(axis=1)
    #         Y = Y.mean(axis=1)
        
    #     cosine_sims = []
    #     for i in range(X.shape[0]):
    #         cos_sim = np.dot(X[i], Y[i]) / (np.linalg.norm(X[i]) * np.linalg.norm(Y[i]) + 1e-8)
    #         cosine_sims.append(cos_sim)
        
    #     return np.mean(cosine_sims)
    

    
    def compute_correlation_gpu(self, X: np.ndarray, Y: np.ndarray) -> float:
        """GPU-accelerated correlation computation."""
        try:
            X_torch = torch.tensor(X, device=self.device, dtype=torch.float32)
            Y_torch = torch.tensor(Y, device=self.device, dtype=torch.float32)
            
            # Flatten tensors
            X_flat = X_torch.flatten()
            Y_flat = Y_torch.flatten()
            
            # Compute correlation using torch
            correlation = torch.corrcoef(torch.stack([X_flat, Y_flat]))[0, 1]
            result = correlation.cpu().item()
            
            # Clear GPU memory
            del X_torch, Y_torch, X_flat, Y_flat, correlation
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU correlation failed, falling back to CPU: {e}")
            return np.corrcoef(X.flatten(), Y.flatten())[0, 1]
    
    def compute_cka_gpu_optimized(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Memory-optimized GPU CKA computation."""
        try:
            X_torch = torch.tensor(X, device=self.device, dtype=torch.float32)
            Y_torch = torch.tensor(Y, device=self.device, dtype=torch.float32)
            
            # Reshape if needed
            if len(X_torch.shape) == 3:
                batch_size, time_steps, features = X_torch.shape
                X_torch = X_torch.reshape(batch_size * time_steps, features)
                Y_torch = Y_torch.reshape(batch_size * time_steps, features)
            
            n_samples = X_torch.shape[0]
            
            # Memory-efficient computation for large matrices
            if n_samples > 1000 and self.memory_efficient:
                result = self._compute_cka_chunked_gpu(X_torch, Y_torch)
            else:
                result = self._compute_cka_direct_gpu(X_torch, Y_torch)
            
            # Clear GPU memory
            del X_torch, Y_torch
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU CKA failed, falling back to CPU: {e}")
            return self._compute_cka_cpu(X, Y)
    
    def _compute_cka_direct_gpu(self, X_torch: torch.Tensor, Y_torch: torch.Tensor) -> float:
        """Direct GPU CKA computation for smaller matrices."""
        n = X_torch.shape[0]
        
        # Center the data
        X_centered = X_torch - X_torch.mean(dim=0, keepdim=True)
        Y_centered = Y_torch - Y_torch.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        K = torch.mm(X_centered, X_centered.t())
        L = torch.mm(Y_centered, Y_centered.t())
        
        # Center Gram matrices
        H = torch.eye(n, device=self.device) - torch.ones(n, n, device=self.device) / n
        K_centered = torch.mm(torch.mm(H, K), H)
        L_centered = torch.mm(torch.mm(H, L), H)
        
        # Compute CKA
        hsic = torch.trace(torch.mm(K_centered, L_centered)) / (n - 1)**2
        var_K = torch.trace(torch.mm(K_centered, K_centered)) / (n - 1)**2
        var_L = torch.trace(torch.mm(L_centered, L_centered)) / (n - 1)**2
        
        cka = hsic / torch.sqrt(var_K * var_L + 1e-8)
        
        return cka.cpu().item()
    
    def _compute_cka_chunked_gpu(self, X_torch: torch.Tensor, Y_torch: torch.Tensor) -> float:
        """Memory-efficient chunked CKA computation."""
        n = X_torch.shape[0]
        chunk_size = min(500, n // 4)  # Adaptive chunk size
        
        # Center the data
        X_centered = X_torch - X_torch.mean(dim=0, keepdim=True)
        Y_centered = Y_torch - Y_torch.mean(dim=0, keepdim=True)
        
        # Compute Gram matrix elements in chunks
        K = torch.zeros(n, n, device=self.device)
        L = torch.zeros(n, n, device=self.device)
        
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
        H = torch.eye(n, device=self.device) - torch.ones(n, n, device=self.device) / n
        K_centered = torch.mm(torch.mm(H, K), H)
        L_centered = torch.mm(torch.mm(H, L), H)
        
        hsic = torch.trace(torch.mm(K_centered, L_centered)) / (n - 1)**2
        var_K = torch.trace(torch.mm(K_centered, K_centered)) / (n - 1)**2
        var_L = torch.trace(torch.mm(L_centered, L_centered)) / (n - 1)**2
        
        cka = hsic / torch.sqrt(var_K * var_L + 1e-8)
        
        return cka.cpu().item()
    
    def _compute_cka_cpu(self, X: np.ndarray, Y: np.ndarray) -> float:
        """CPU fallback for CKA computation."""
        if len(X.shape) == 3:
            X = X.reshape(-1, X.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])
        
        n = X.shape[0]
        
        # Center the data
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        
        # Compute Gram matrices
        K = X_centered @ X_centered.T
        L = Y_centered @ Y_centered.T
        
        # Center Gram matrices
        H = np.eye(n) - np.ones((n, n)) / n
        K_centered = H @ K @ H
        L_centered = H @ L @ H
        
        # Compute CKA
        hsic = np.trace(K_centered @ L_centered) / (n - 1)**2
        var_K = np.trace(K_centered @ K_centered) / (n - 1)**2
        var_L = np.trace(L_centered @ L_centered) / (n - 1)**2
        
        cka = hsic / np.sqrt(var_K * var_L + 1e-8)
        
        return cka
    
    def compute_partial_correlation_gpu(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """GPU-accelerated partial correlation computation."""
        try:
            X_torch = torch.tensor(X, device=self.device, dtype=torch.float32)
            Y_torch = torch.tensor(Y, device=self.device, dtype=torch.float32)
            Z_torch = torch.tensor(Z, device=self.device, dtype=torch.float32)
            
            # Regress out Z from X and Y
            X_residual = self._regress_out_gpu(Z_torch, X_torch)
            Y_residual = self._regress_out_gpu(Z_torch, Y_torch)
            
            # Compute correlation between residuals
            X_flat = X_residual.flatten()
            Y_flat = Y_residual.flatten()
            
            correlation = torch.corrcoef(torch.stack([X_flat, Y_flat]))[0, 1]
            result = correlation.cpu().item()
            
            # Clear GPU memory
            del X_torch, Y_torch, Z_torch, X_residual, Y_residual, X_flat, Y_flat, correlation
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU partial correlation failed, falling back to CPU: {e}")
            return self._compute_partial_correlation_cpu(X, Y, Z)
    
    def _regress_out_gpu(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated regression using torch.linalg.lstsq."""
        # Add bias term
        Z_with_bias = torch.cat([Z, torch.ones(Z.shape[0], 1, device=self.device)], dim=1)
        
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
    
    def _compute_partial_correlation_cpu(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """CPU fallback for partial correlation."""
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
        residuals_X_flat = residuals_X.flatten()
        residuals_Y_flat = residuals_Y.flatten()
        
        if len(residuals_X_flat) > 1 and len(residuals_Y_flat) > 1:
            partial_corr = np.corrcoef(residuals_X_flat, residuals_Y_flat)[0, 1]
        else:
            partial_corr = 0.0
        
        return partial_corr
    
    # Commented out due to regression step issues in the base implementation
    # def compute_conditional_cka_gpu(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    #     """Compute conditional CKA using GPU acceleration."""
    #     try:
    #         return self._compute_conditional_cka_gpu(X, Y, Z)
    #     except Exception as e:
    #         logger.warning(f"GPU conditional CKA failed: {e}")
    #         return self._compute_conditional_cka_cpu(X, Y, Z)
    # 
    # def _compute_conditional_cka_cpu(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    #     """Fallback CPU implementation of conditional CKA."""
    #     return compute_conditional_cka(X, Y, Z, method='residual', use_gpu=False)


class FeatureLoader:
    """Efficient feature loading with memory management."""
    
    def __init__(self, features_dir: str, num_files: int = 3):
        self.features_dir = Path(features_dir)
        self.num_files = num_files
        
    def load_features(self) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
        """Load features from .npz files with memory management."""
        feature_files = list(self.features_dir.glob("*_complete_features.npz"))
        
        if len(feature_files) == 0:
            raise ValueError(f"No feature files found in {self.features_dir}")
        
        # Take only the specified number of files
        feature_files = feature_files[:self.num_files]
        
        # Dictionary to store features for each layer
        layer_features = {}
        max_lengths = {}
        original_lengths = {}
        
        logger.info(f"Loading features from {len(feature_files)} files...")
        
        for file_idx, chkpnt_file_path in enumerate(feature_files):
            logger.info(f"Processing file {file_idx + 1}/{len(feature_files)}: {chkpnt_file_path.name}")
            
            try:
                layer_features_contextualized = np.load(chkpnt_file_path)
                
                for layer in layer_features_contextualized.files:
                    # Filter layers: transformer_input and transformer_layer_0 to 11
                    if (layer == 'transformer_input' or 
                        (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)):
                        
                        if layer not in layer_features:
                            layer_features[layer] = []
                            max_lengths[layer] = 0
                            original_lengths[layer] = []
                        
                        features = layer_features_contextualized[layer]
                        
                        # Get the time dimension
                        time_dim = features.shape[1] if len(features.shape) == 3 else features.shape[0]
                        max_lengths[layer] = max(max_lengths[layer], time_dim)
                        original_lengths[layer].append(time_dim)
                        layer_features[layer].append(features)
                        
                        logger.debug(f"Layer {layer}: shape {features.shape}, max_length {max_lengths[layer]}")
                
            except Exception as e:
                logger.error(f"Error loading file {chkpnt_file_path}: {e}")
                continue
        
        # Pad and concatenate features
        for layer in layer_features:
            logger.info(f"Padding layer {layer} to length {max_lengths[layer]}")
            padded_features = [self._pad_features(f, max_lengths[layer]) for f in layer_features[layer]]
            layer_features[layer] = np.concatenate(padded_features, axis=0)
            logger.info(f"Final shape for layer {layer}: {layer_features[layer].shape}")
        
        return layer_features, original_lengths
    
    def _pad_features(self, features: np.ndarray, max_length: int) -> np.ndarray:
        """Pad features to a consistent length."""
        if len(features.shape) == 3:  # [batch, time, dim]
            batch_size, time_steps, dim = features.shape
            padded = np.zeros((batch_size, max_length, dim))
            padded[:, :time_steps, :] = features
        else:  # [time, dim]
            time_steps, dim = features.shape
            padded = np.zeros((max_length, dim))
            padded[:time_steps, :] = features
        return padded


class ParallelSimilarityComputer:
    """Main class for parallel similarity computation."""
    
    def __init__(self, n_gpus: Optional[int] = None, memory_efficient: bool = True):
        self.n_gpus = n_gpus if n_gpus is not None else torch.cuda.device_count()
        self.memory_efficient = memory_efficient
        
        if self.n_gpus == 0:
            logger.warning("No GPUs available, will use CPU with multiprocessing")
        else:
            logger.info(f"Using {self.n_gpus} GPUs for parallel computation")
    
    def get_layer_number(self, layer_name: str) -> int:
        """Get layer number for sorting."""
        if layer_name == 'transformer_input':
            return -1
        elif layer_name.startswith('transformer_layer_'):
            layer_num = int(layer_name.split('_')[-1])
            if layer_num <= 11:
                return layer_num
        return float('inf')
    
    def compute_all_similarities(self, layer_features: Dict[str, np.ndarray], 
                               original_lengths: Dict[str, List[int]],
                               metrics: List[str] = None) -> Dict[str, np.ndarray]:
        """Compute all similarity matrices between layers."""
        if metrics is None:
            # metrics = ['cosine', 'correlation', 'cka', 'partial_correlation', 'conditional_cka']  # Commented out cosine similarity and conditionals
            metrics = ['correlation', 'cka']  # Only metrics actually implemented in worker functions
        
        # Filter and sort layers
        layers = sorted([layer for layer in layer_features.keys() 
                        if layer == 'transformer_input' or 
                        (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                       key=self.get_layer_number)
        
        n_layers = len(layers)
        logger.info(f"Computing similarities for {n_layers} layers: {layers}")
        
        # Initialize result matrices
        results = {}
        for metric in metrics:
            results[metric] = np.zeros((n_layers, n_layers))
        
        # Create tasks (only upper triangle to avoid redundant computation)
        tasks = []
        for i in range(n_layers):
            for j in range(i, n_layers):
                tasks.append((i, j, layers[i], layers[j]))
        
        logger.info(f"Total tasks to compute: {len(tasks)}")
        
        if self.n_gpus > 0:
            # GPU parallel computation
            results = self._compute_similarities_gpu_parallel(
                tasks, layer_features, original_lengths, metrics, results
            )
        else:
            # CPU parallel computation
            results = self._compute_similarities_cpu_parallel(
                tasks, layer_features, original_lengths, metrics, results
            )
        
        # Fill lower triangle (symmetric matrices)
        for metric in results:
            for i in range(n_layers):
                for j in range(i):
                    results[metric][i, j] = results[metric][j, i]
        
        return results, layers
    
    def _compute_similarities_gpu_parallel(self, tasks: List[Tuple], 
                                         layer_features: Dict[str, np.ndarray],
                                         original_lengths: Dict[str, List[int]],
                                         metrics: List[str],
                                         results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """GPU parallel computation of similarities."""
        # Distribute tasks across GPUs
        gpu_tasks = []
        for task_id, task in enumerate(tasks):
            gpu_id = task_id % self.n_gpus
            gpu_tasks.append((gpu_id, task))
        
        # Group tasks by GPU
        gpu_task_groups = {}
        for gpu_id, task in gpu_tasks:
            if gpu_id not in gpu_task_groups:
                gpu_task_groups[gpu_id] = []
            gpu_task_groups[gpu_id].append(task)
        
        # Process each GPU group in parallel
        with mp.Pool(processes=self.n_gpus) as pool:
            worker_func = partial(
                self._gpu_worker,
                layer_features=layer_features,
                original_lengths=original_lengths,
                metrics=metrics,
                memory_efficient=self.memory_efficient
            )
            
            gpu_results = pool.starmap(worker_func, 
                                     [(gpu_id, tasks) for gpu_id, tasks in gpu_task_groups.items()])
        
        # Combine results
        for gpu_result in gpu_results:
            for metric, matrix_updates in gpu_result.items():
                for (i, j), value in matrix_updates.items():
                    results[metric][i, j] = value
        
        return results
    
    def _compute_similarities_cpu_parallel(self, tasks: List[Tuple],
                                         layer_features: Dict[str, np.ndarray],
                                         original_lengths: Dict[str, List[int]],
                                         metrics: List[str],
                                         results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """CPU parallel computation of similarities."""
        n_cores = min(mp.cpu_count(), len(tasks))
        logger.info(f"Using {n_cores} CPU cores for parallel computation")
        
        with mp.Pool(processes=n_cores) as pool:
            worker_func = partial(
                self._cpu_worker,
                layer_features=layer_features,
                original_lengths=original_lengths,
                metrics=metrics
            )
            
            cpu_results = pool.map(worker_func, tasks)
        
        # Combine results
        for result in cpu_results:
            i, j = result['indices']
            for metric in metrics:
                if metric in result:
                    results[metric][i, j] = result[metric]
        
        return results
    
    @staticmethod
    def _gpu_worker(gpu_id: int, tasks: List[Tuple], 
                   layer_features: Dict[str, np.ndarray],
                   original_lengths: Dict[str, List[int]],
                   metrics: List[str],
                   memory_efficient: bool) -> Dict[str, Dict[Tuple, float]]:
        """GPU worker function for parallel computation."""
        results = {metric: {} for metric in metrics}
        similarity_computer = GPUParallelSimilarity(device=f'cuda:{gpu_id}', memory_efficient=memory_efficient)
        
        for i, j, layer1, layer2 in tqdm(tasks, desc=f"GPU {gpu_id}"):
            try:
                # Get features
                features1 = layer_features[layer1]
                features2 = layer_features[layer2]
                
                # Ensure same batch size
                min_batch = min(features1.shape[0], features2.shape[0])
                features1 = features1[:min_batch]
                features2 = features2[:min_batch]
                
                # Instead of simple averaging, use padding-aware approach
                if layer1 in original_lengths and layer2 in original_lengths:
                    all_X = []
                    all_Y = []
                    
                    for b in range(min_batch): #TODO: Is this needed? can we call the function _extract_valid_features instead?
                        # Use only valid (non-padded) portions
                        valid_len1 = original_lengths[layer1][b] if b < len(original_lengths[layer1]) else features1.shape[1]
                        valid_len2 = original_lengths[layer2][b] if b < len(original_lengths[layer2]) else features2.shape[1]
                        valid_len = min(valid_len1, valid_len2)
                        
                        if valid_len > 0:
                            all_X.append(features1[b, :valid_len, :].mean(axis=0))  # Average only valid timesteps
                            all_Y.append(features2[b, :valid_len, :].mean(axis=0))
                    
                    if all_X and all_Y:
                        f1_avg = np.vstack(all_X)
                        f2_avg = np.vstack(all_Y)
                    else:
                        # Fallback to simple averaging
                        f1_avg = features1.mean(axis=1)
                        f2_avg = features2.mean(axis=1)
                else:
                    # Fallback when original_lengths not available
                    f1_avg = features1.mean(axis=1)
                    f2_avg = features2.mean(axis=1)
                
                # Ensure same feature dimension
                min_features = min(f1_avg.shape[1], f2_avg.shape[1])
                f1_avg = f1_avg[:, :min_features]
                f2_avg = f2_avg[:, :min_features]
                
                # Compute similarities
                # if 'cosine' in metrics:  # Commented out as cosine similarity is being refactored
                #     try:
                #         results['cosine'][(i, j)] = similarity_computer.compute_cosine_similarity_gpu(
                #             f1_avg, f2_avg
                #         )
                #     except Exception as e:
                #         logger.error(f"Error computing cosine similarity for {layer1}-{layer2} on GPU {gpu_id}: {e}")
                #         results['cosine'][(i, j)] = 0.0
                
                if 'correlation' in metrics:
                    results['correlation'][(i, j)] = similarity_computer.compute_correlation_gpu(
                        f1_avg, f2_avg
                    )
                
                if 'cka' in metrics:
                    results['cka'][(i, j)] = similarity_computer.compute_cka_gpu_optimized(
                        f1_avg, f2_avg
                    )
                
                # Conditional metrics require CNN features
                if ('partial_correlation' in metrics or 'conditional_cka' in metrics) and 'transformer_input' in layer_features:
                    cnn_features = layer_features['transformer_input'][:min_batch]
                    
                    # Prepare data for conditional analysis
                    X = f1_avg.reshape(-1, f1_avg.shape[-1])
                    Y = f2_avg.reshape(-1, f2_avg.shape[-1])
                    Z = cnn_features.reshape(-1, cnn_features.shape[-1])
                    
                    min_samples = min(X.shape[0], Y.shape[0], Z.shape[0])
                    X, Y, Z = X[:min_samples], Y[:min_samples], Z[:min_samples]
                    
                    # Compute conditional metrics
                    # if 'partial_correlation' in metrics:  # Commented out as partial correlation is being refactored
                    #     try:
                    #         results['partial_correlation'][(i, j)] = similarity_computer.compute_partial_correlation_gpu(
                    #             X, Y, Z
                    #         )
                    #     except Exception as e:
                    #         logger.error(f"Error computing partial correlation for {layer1}-{layer2} on GPU {gpu_id}: {e}")
                    #         results['partial_correlation'][(i, j)] = 0.0
                    #
                    # if 'conditional_cka' in metrics:  # Commented out as conditional CKA is being refactored
                    #     try:
                    #         results['conditional_cka'][(i, j)] = similarity_computer.compute_conditional_cka_gpu(
                    #             X, Y, Z
                    #         )
                    #     except Exception as e:
                    #         logger.error(f"Error computing conditional CKA for {layer1}-{layer2} on GPU {gpu_id}: {e}")
                    #         results['conditional_cka'][(i, j)] = 0.0
                
            except Exception as e:
                logger.error(f"Error computing similarity for {layer1}-{layer2} on GPU {gpu_id}: {e}")
                # Set default values for failed computations
                for metric in metrics:
                    results[metric][(i, j)] = 0.0
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    @staticmethod
    def _cpu_worker(task: Tuple, layer_features: Dict[str, np.ndarray],
                   original_lengths: Dict[str, List[int]],
                   metrics: List[str]) -> Dict:
        """Worker function for CPU-based computation."""
        i, j, layer1, layer2 = task
        similarity_computer = GPUParallelSimilarity(device='cpu')
        
        result = {'indices': (i, j)}
        
        try:
            # Get features
            features1 = layer_features[layer1]
            features2 = layer_features[layer2]
            
            # Ensure same batch size
            min_batch = min(features1.shape[0], features2.shape[0])
            features1 = features1[:min_batch]
            features2 = features2[:min_batch]
            
            # Instead of simple averaging, use padding-aware approach
            if layer1 in original_lengths and layer2 in original_lengths: #TODO: Is this needed? can we call the function _extract_valid_features instead?
                all_X = []
                all_Y = []
                
                for b in range(min_batch):
                    # Use only valid (non-padded) portions
                    valid_len1 = original_lengths[layer1][b] if b < len(original_lengths[layer1]) else features1.shape[1]
                    valid_len2 = original_lengths[layer2][b] if b < len(original_lengths[layer2]) else features2.shape[1]
                    valid_len = min(valid_len1, valid_len2)
                    
                    if valid_len > 0:
                        all_X.append(features1[b, :valid_len, :].mean(axis=0))  # Average only valid timesteps
                        all_Y.append(features2[b, :valid_len, :].mean(axis=0))
                
                if all_X and all_Y:
                    f1_avg = np.vstack(all_X)
                    f2_avg = np.vstack(all_Y)
                else:
                    # Fallback to simple averaging
                    f1_avg = features1.mean(axis=1)
                    f2_avg = features2.mean(axis=1)
            else:
                # Fallback when original_lengths not available
                f1_avg = features1.mean(axis=1) #TODO: This may introduce errors! Confirm this is needed as an else statement
                f2_avg = features2.mean(axis=1)
            
            # Ensure same feature dimension
            min_features = min(f1_avg.shape[1], f2_avg.shape[1])
            f1_avg = f1_avg[:, :min_features]
            f2_avg = f2_avg[:, :min_features]
            
            # Compute similarities
            # if 'cosine' in metrics:  # Commented out as cosine similarity is being refactored
            #     try:
            #         result['cosine'] = similarity_computer.compute_cosine_similarity_gpu(
            #             f1_avg, f2_avg
            #         )
            #     except Exception as e:
            #         logger.error(f"Error computing cosine similarity for {layer1}-{layer2} on CPU: {e}")
            #         result['cosine'] = 0.0
            
            if 'correlation' in metrics:
                result['correlation'] = similarity_computer.compute_correlation_gpu(
                    f1_avg, f2_avg
                )
            
            if 'cka' in metrics:
                result['cka'] = similarity_computer.compute_cka_gpu_optimized(
                    f1_avg, f2_avg
                )
            
            # Conditional metrics require CNN features
            if ('partial_correlation' in metrics or 'conditional_cka' in metrics) and 'transformer_input' in layer_features:
                cnn_features = layer_features['transformer_input'][:min_batch]
                
                # Prepare data for conditional analysis
                X = f1_avg.reshape(-1, f1_avg.shape[-1])
                Y = f2_avg.reshape(-1, f2_avg.shape[-1])
                Z = cnn_features.reshape(-1, cnn_features.shape[-1])
                
                min_samples = min(X.shape[0], Y.shape[0], Z.shape[0])
                X, Y, Z = X[:min_samples], Y[:min_samples], Z[:min_samples]
                
                # Compute conditional metrics
                # if 'partial_correlation' in metrics:  # Commented out as partial correlation is being refactored
                #     try:
                #         result['partial_correlation'] = similarity_computer.compute_partial_correlation_gpu(
                #             X, Y, Z
                #         )
                #     except Exception as e:
                #         logger.error(f"Error computing partial correlation for {layer1}-{layer2} on CPU: {e}")
                #         result['partial_correlation'] = 0.0
                #
                # if 'conditional_cka' in metrics:  # Commented out as conditional CKA is being refactored
                #     try:
                #         result['conditional_cka'] = similarity_computer.compute_conditional_cka_gpu(
                #             X, Y, Z
                #         )
                #     except Exception as e:
                #         logger.error(f"Error computing conditional CKA for {layer1}-{layer2} on CPU: {e}")
                #         result['conditional_cka'] = 0.0
            
        except Exception as e:
            logger.error(f"Error computing similarity for {layer1}-{layer2} on CPU: {e}")
            # Set default values for failed computations
            for metric in metrics:
                result[metric] = 0.0
        
        return result
    
    @staticmethod
    def _extract_valid_features(features1: np.ndarray, features2: np.ndarray,
                               orig_lens1: List[int], orig_lens2: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract valid (non-padded) features."""
        all_X = []
        all_Y = []
        
        batch_size = features1.shape[0]
        
        for b in range(batch_size):
            # Get original length for this batch item
            orig_len1 = orig_lens1[b] if b < len(orig_lens1) else features1.shape[1]
            orig_len2 = orig_lens2[b] if b < len(orig_lens2) else features2.shape[1]
            orig_len = min(orig_len1, orig_len2)  # Use minimum to ensure both are valid
            
            if orig_len > 0:
                # Extract only non-padded time steps
                X = features1[b, :orig_len, :]  # (time, dim)
                Y = features2[b, :orig_len, :]
                
                all_X.append(X)
                all_Y.append(Y)
        
        if all_X and all_Y:
            # Concatenate all valid samples
            X = np.vstack(all_X)  # (total_valid_samples, dim)
            Y = np.vstack(all_Y)
            
            # Ensure same number of samples
            min_samples = min(X.shape[0], Y.shape[0])
            return X[:min_samples], Y[:min_samples]
        else:
            # Fallback to original features if extraction fails
            return features1, features2


class VisualizationManager:
    """Manages all visualization and plotting functions."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_similarity_matrices(self, similarity_results: Dict[str, np.ndarray], 
                                layers: List[str], model_name: str, num_files: int):
        """Plot similarity matrices."""
        n_metrics = len(similarity_results)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        metric_configs = {
            # 'cosine': {'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'title': 'Cosine Similarity'},  # Commented out as cosine similarity is being refactored
            'correlation': {'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1, 'title': 'Correlation'},
            'cka': {'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'title': 'CKA'},
            'partial_correlation': {'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1, 'title': 'Partial Correlation'},
            'conditional_cka': {'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'title': 'Conditional CKA'}
        }
        
        for idx, metric in enumerate(similarity_results):
            ax = axes[idx]
            config = metric_configs.get(metric, {'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'title': metric.title()})
            
            # Create heatmap
            sns.heatmap(similarity_results[metric], ax=ax,
                       xticklabels=layers, yticklabels=layers,
                       cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'],
                       annot=True, fmt='.2f', annot_kws={'size': 8},
                       cbar_kws={'label': 'Similarity'})
            
            ax.set_title(config['title'], fontsize=12)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Layer')
        
        plt.suptitle(f'Layer Similarity Analysis - {model_name} (n={num_files} files)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'all_similarities_{model_name}_n{num_files}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity matrices saved to {self.output_dir}")
    
    def plot_conditional_vs_unconditional(self, similarity_results: Dict[str, np.ndarray],
                                        layers: List[str], model_name: str, num_files: int):
        """Plot comparison between conditional and unconditional similarities."""
        if 'correlation' not in similarity_results or 'partial_correlation' not in similarity_results:
            logger.warning("Cannot create conditional vs unconditional plot: missing required metrics")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Unconditional correlation
        sns.heatmap(similarity_results['correlation'], ax=axes[0, 0],
                   xticklabels=layers, yticklabels=layers,
                   cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Correlation'})
        axes[0, 0].set_title('Unconditional Correlation')
        
        # Conditional correlation (partial)
        sns.heatmap(similarity_results['partial_correlation'], ax=axes[0, 1],
                   xticklabels=layers, yticklabels=layers,
                   cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Partial Correlation'})
        axes[0, 1].set_title('Partial Correlation | CNN Output')
        
        # Difference plot
        diff_correlation = similarity_results['correlation'] - similarity_results['partial_correlation']
        sns.heatmap(diff_correlation, ax=axes[1, 0],
                   xticklabels=layers, yticklabels=layers,
                   cmap='coolwarm', center=0, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Difference'})
        axes[1, 0].set_title('Difference (Unconditional - Conditional)')
        
        # CKA comparison if available
        if 'cka' in similarity_results and 'conditional_cka' in similarity_results:
            diff_cka = similarity_results['cka'] - similarity_results['conditional_cka']
            sns.heatmap(diff_cka, ax=axes[1, 1],
                       xticklabels=layers, yticklabels=layers,
                       cmap='coolwarm', center=0, annot=True, fmt='.2f',
                       cbar_kws={'label': 'CKA Difference'})
            axes[1, 1].set_title('CKA Difference (Unconditional - Conditional)')
        else:
            axes[1, 1].set_visible(False)
        
        plt.suptitle(f'Conditional vs Unconditional Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'conditional_comparison_{model_name}_n{num_files}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Conditional comparison plot saved to {self.output_dir}")
    
    def plot_cnn_influence_analysis(self, layer_features: Dict[str, np.ndarray],
                                  layers: List[str], model_name: str, num_files: int):
        """Plot CNN influence (R²) analysis."""
        if 'transformer_input' not in layer_features:
            logger.warning("Cannot create CNN influence plot: transformer_input not found")
            return
        
        cnn_features = layer_features['transformer_input']
        transformer_layers = [layer for layer in layers if layer.startswith('transformer_layer_')]
        
        r2_scores = []
        layer_names = []
        
        similarity_computer = GPUParallelSimilarity(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        for layer in transformer_layers:
            features = layer_features[layer]
            
            # Ensure same batch size
            min_batch = min(features.shape[0], cnn_features.shape[0])
            features = features[:min_batch]
            cnn_batch = cnn_features[:min_batch]
            
            # Reshape to 2D
            X = features.reshape(-1, features.shape[-1])
            Z = cnn_batch.reshape(-1, cnn_batch.shape[-1])
            
            # Ensure same number of samples
            min_samples = min(X.shape[0], Z.shape[0])
            X = X[:min_samples]
            Z = Z[:min_samples]
            
            # Compute R² for each dimension
            r2_per_dim = []
            for dim in range(min(X.shape[1], 100)):  # Limit to 100 dims for speed
                try:
                    if torch.cuda.is_available():
                        # GPU computation
                        X_torch = torch.tensor(X[:, dim:dim+1], device='cuda', dtype=torch.float32)
                        Z_torch = torch.tensor(Z, device='cuda', dtype=torch.float32)
                        
                        # Add bias term and solve
                        Z_with_bias = torch.cat([Z_torch, torch.ones(Z_torch.shape[0], 1, device='cuda')], dim=1)
                        beta = torch.linalg.lstsq(Z_with_bias, X_torch).solution
                        
                        # Compute R²
                        X_pred = torch.mm(Z_with_bias, beta)
                        ss_res = torch.sum((X_torch - X_pred) ** 2)
                        ss_tot = torch.sum((X_torch - torch.mean(X_torch)) ** 2)
                        r2 = 1 - ss_res / ss_tot
                        r2_per_dim.append(r2.cpu().item())
                        
                        # Clear GPU memory
                        del X_torch, Z_torch, Z_with_bias, beta, X_pred
                        torch.cuda.empty_cache()
                    else:
                        # CPU computation
                        reg = LinearRegression()
                        reg.fit(Z, X[:, dim])
                        r2 = reg.score(Z, X[:, dim])
                        r2_per_dim.append(max(0, r2))  # Ensure non-negative
                except:
                    r2_per_dim.append(0.0)
            
            # Average R² across dimensions
            avg_r2 = np.mean(r2_per_dim)
            r2_scores.append(avg_r2)
            layer_names.append(layer)
        
        # Plot R² decay
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(transformer_layers)), r2_scores, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Layer')
        plt.ylabel('R² (Variance Explained by CNN Output)')
        plt.title(f'CNN Influence Decay Across Transformer Layers - {model_name}')
        plt.xticks(range(len(transformer_layers)), layer_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add text annotations for R² values
        for i, r2 in enumerate(r2_scores):
            plt.annotate(f'{r2:.3f}', xy=(i, r2), xytext=(0, 5), 
                        textcoords='offset points', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'cnn_influence_decay_{model_name}_n{num_files}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CNN influence analysis saved to {self.output_dir}")
        
        return r2_scores
    
    def create_summary_report(self, similarity_results: Dict[str, np.ndarray],
                            layers: List[str], model_name: str, num_files: int,
                            r2_scores: Optional[List[float]] = None):
        """Create a summary report of the analysis."""
        report_path = self.output_dir / f'analysis_report_{model_name}_n{num_files}.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"Layer Similarity Analysis Report\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Number of files analyzed: {num_files}\n")
            f.write(f"Number of layers: {len(layers)}\n")
            f.write(f"Layers analyzed: {', '.join(layers)}\n\n")
            
            # Similarity metrics summary
            f.write("Similarity Metrics Computed:\n")
            f.write("-" * 30 + "\n")
            for metric in similarity_results.keys():
                matrix = similarity_results[metric]
                # Exclude diagonal for statistics
                off_diagonal = matrix[np.triu_indices_from(matrix, k=1)]
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {np.mean(off_diagonal):.3f}\n")
                f.write(f"  Std:  {np.std(off_diagonal):.3f}\n")
                f.write(f"  Min:  {np.min(off_diagonal):.3f}\n")
                f.write(f"  Max:  {np.max(off_diagonal):.3f}\n\n")
            
            # CNN influence summary
            if r2_scores is not None:
                f.write("CNN Influence Analysis:\n")
                f.write("-" * 30 + "\n")
                transformer_layers = [layer for layer in layers if layer.startswith('transformer_layer_')]
                for i, (layer, r2) in enumerate(zip(transformer_layers, r2_scores)):
                    independence = (1 - r2) * 100
                    f.write(f"{layer}: R² = {r2:.3f} ({independence:.1f}% independent of CNN)\n")
                
                if len(r2_scores) > 1:
                    decay = r2_scores[0] - r2_scores[-1]
                    f.write(f"\nOverall CNN influence decay: {decay:.3f}\n")
            
            f.write(f"\nAnalysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Summary report saved to {report_path}")


class CompleteSimilarityAnalysis:
    """Main class orchestrating the complete analysis pipeline."""
    
    def __init__(self, features_dir: str, output_dir: str, num_files: int = 3,
                 model_name: str = "HuBERT", n_gpus: Optional[int] = None,
                 memory_efficient: bool = True):
        self.features_dir = features_dir
        self.output_dir = output_dir
        self.num_files = num_files
        self.model_name = model_name
        self.n_gpus = n_gpus
        self.memory_efficient = memory_efficient
        
        # Initialize components
        self.feature_loader = FeatureLoader(features_dir, num_files)
        self.similarity_computer = ParallelSimilarityComputer(n_gpus, memory_efficient)
        self.visualizer = VisualizationManager(output_dir)
        
        logger.info(f"Initialized analysis pipeline for {model_name}")
    
    def run_complete_analysis(self, metrics: List[str] = None) -> Dict:
        """Run the complete similarity analysis pipeline."""
        if metrics is None:
            # metrics = ['cosine', 'correlation', 'cka', 'partial_correlation', 'conditional_cka']  # Commented out cosine similarity and conditionals
            metrics = ['correlation', 'cka']  # Only metrics actually implemented in worker functions
        
        logger.info("Starting complete similarity analysis...")
        start_time = time.time()
        
        # Step 1: Load features
        logger.info("Step 1: Loading features...")
        layer_features, original_lengths = self.feature_loader.load_features()
        
        # Step 2: Compute similarities
        logger.info("Step 2: Computing similarities...")
        similarity_results, layers = self.similarity_computer.compute_all_similarities(
            layer_features, original_lengths, metrics
        )
        
        # Step 3: Generate visualizations
        logger.info("Step 3: Generating visualizations...")
        self.visualizer.plot_similarity_matrices(
            similarity_results, layers, self.model_name, self.num_files
        )
        
        if 'partial_correlation' in similarity_results:
            self.visualizer.plot_conditional_vs_unconditional(
                similarity_results, layers, self.model_name, self.num_files
            )
        
        # Step 4: CNN influence analysis
        logger.info("Step 4: Analyzing CNN influence...")
        r2_scores = self.visualizer.plot_cnn_influence_analysis(
            layer_features, layers, self.model_name, self.num_files
        )
        
        # Step 5: Generate summary report
        logger.info("Step 5: Generating summary report...")
        self.visualizer.create_summary_report(
            similarity_results, layers, self.model_name, self.num_files, r2_scores
        )
        
        # Save results
        results_path = Path(self.output_dir) / f'similarity_results_{self.model_name}_n{self.num_files}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump({
                'similarity_results': similarity_results,
                'layers': layers,
                'r2_scores': r2_scores,
                'model_name': self.model_name,
                'num_files': self.num_files
            }, f)
        
        total_time = time.time() - start_time
        logger.info(f"Complete analysis finished in {total_time:.2f} seconds")
        
        return {
            'similarity_results': similarity_results,
            'layers': layers,
            'r2_scores': r2_scores,
            'total_time': total_time
        }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='GPU-accelerated layer similarity analysis')
    parser.add_argument('--features_dir', type=str, required=True,
                      help='Directory containing layer features')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    parser.add_argument('--num_files', type=int, default=3,
                      help='Number of files to process')
    parser.add_argument('--model_name', type=str, default='HuBERT',
                      help='Name of the model')
    parser.add_argument('--n_gpus', type=int, default=None,
                      help='Number of GPUs to use (default: all available)')
    parser.add_argument('--memory_efficient', action='store_true',
                      help='Use memory-efficient computation')
    parser.add_argument('--metrics', nargs='+',
                      # default=['cosine', 'correlation', 'cka', 'partial_correlation', 'conditional_cka'],  # Commented out cosine similarity and conditionals
                      default=['correlation', 'cka'],  # Only metrics actually implemented in worker functions
                      help='Metrics to compute')
    parser.add_argument('--cpu_only', action='store_true', default=False,
                      help='Force CPU-only computation')
    parser.add_argument('--verbose', action='store_true', default=False,
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Override GPU detection if CPU-only requested
    n_gpus = 0 if args.cpu_only else args.n_gpus
    
    # Create analysis pipeline
    analysis = CompleteSimilarityAnalysis(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        num_files=args.num_files,
        model_name=args.model_name,
        n_gpus=n_gpus,
        memory_efficient=args.memory_efficient
    )
    
    # Run analysis
    try:
        results = analysis.run_complete_analysis(metrics=args.metrics)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model: {args.model_name}")
        print(f"Files analyzed: {args.num_files}")
        print(f"Layers analyzed: {len(results['layers'])}")
        print(f"Metrics computed: {', '.join(args.metrics)}")
        print(f"Total computation time: {results['total_time']:.2f} seconds")
        
        if results['r2_scores']:
            print(f"\nCNN Influence Decay:")
            transformer_layers = [l for l in results['layers'] if l.startswith('transformer_layer_')]
            print(f"  First layer R²: {results['r2_scores'][0]:.3f}")
            print(f"  Last layer R²:  {results['r2_scores'][-1]:.3f}")
            print(f"  Decay: {results['r2_scores'][0] - results['r2_scores'][-1]:.3f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
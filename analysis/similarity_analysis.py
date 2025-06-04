import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
from pathlib import Path

from utils.data_utils import filter_and_sort_layers
from utils.math_utils import (
    compute_cka, compute_cka_without_padding, 
    compute_partial_correlation, compute_conditional_cka,
    compute_cosine_similarity
)
from utils.visualization_utils import save_figure


def compute_layer_similarities(layer_features: Dict[str, np.ndarray], 
                              original_lengths: Dict[str, List[int]],
                              include_conditional: bool = False,
                              cnn_layer: str = 'transformer_input') -> Dict[str, np.ndarray]:
    """
    Compute similarity matrices between layers.
    
    Args:
        layer_features: Dictionary of layer features
        original_lengths: Dictionary of original sequence lengths
        include_conditional: Whether to compute conditional similarities
        cnn_layer: Name of CNN output layer for conditioning
    
    Returns:
        Dictionary containing similarity matrices
    """
    # Filter and sort layers
    layers = filter_and_sort_layers(layer_features)
    n_layers = len(layers)
    
    # Initialize matrices
    cosine_matrix = np.zeros((n_layers, n_layers))
    correlation_matrix = np.zeros((n_layers, n_layers))
    cka_matrix = np.zeros((n_layers, n_layers))
    
    # Conditional matrices (if requested)
    if include_conditional and cnn_layer in layer_features:
        conditional_correlation_matrix = np.zeros((n_layers, n_layers))
        conditional_cka_matrix = np.zeros((n_layers, n_layers))
        cnn_features = layer_features[cnn_layer]
    else:
        conditional_correlation_matrix = None
        conditional_cka_matrix = None
        cnn_features = None
    
    print("Computing layer similarities...")
    total_pairs = n_layers * n_layers
    
    with tqdm(total=total_pairs, desc="Computing similarities") as pbar:
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                pbar.set_description(f"Processing {layer1} vs {layer2}")
                
                features1 = layer_features[layer1]
                features2 = layer_features[layer2]
                
                # Ensure same batch size
                min_batch = min(features1.shape[0], features2.shape[0])
                features1 = features1[:min_batch]
                features2 = features2[:min_batch]
                
                # Method 1: Average cosine similarity
                f1_avg = features1.mean(axis=1)  # (batch, features)
                f2_avg = features2.mean(axis=1)  # (batch, features)
                
                # Ensure same feature dimension for cosine/correlation
                min_features = min(f1_avg.shape[1], f2_avg.shape[1])
                f1_avg_truncated = f1_avg[:, :min_features]
                f2_avg_truncated = f2_avg[:, :min_features]
                
                # Compute cosine similarity
                cos_sims = []
                for b in range(min_batch):
                    cos_sim = compute_cosine_similarity(f1_avg_truncated[b], f2_avg_truncated[b])
                    cos_sims.append(cos_sim)
                cosine_matrix[i, j] = np.mean(cos_sims)
                
                # Method 2: Correlation
                try:
                    correlation_matrix[i, j] = np.corrcoef(
                        f1_avg_truncated.flatten(), f2_avg_truncated.flatten()
                    )[0, 1]
                except:
                    correlation_matrix[i, j] = 0.0
                
                # Method 3: CKA
                if layer1 in original_lengths and layer2 in original_lengths:
                    cka_matrix[i, j] = compute_cka_without_padding(
                        features1, features2, 
                        original_lengths[layer1], 
                        original_lengths[layer2]
                    )
                else:
                    # Fallback to original method if lengths not available
                    X = features1.reshape(features1.shape[0] * features1.shape[1], -1)
                    Y = features2.reshape(features2.shape[0] * features2.shape[1], -1)
                    min_samples = min(X.shape[0], Y.shape[0])
                    X = X[:min_samples]
                    Y = Y[:min_samples]
                    cka_matrix[i, j] = compute_cka(X, Y)
                
                # Conditional similarities (if requested)
                if include_conditional and cnn_features is not None:
                    # Prepare features for conditional analysis
                    cnn_batch = cnn_features[:min_batch]
                    
                    # Handle padding-aware approach
                    if (layer1 in original_lengths and layer2 in original_lengths 
                        and cnn_layer in original_lengths):
                        all_X, all_Y, all_Z = [], [], []
                        
                        for b in range(min_batch):
                            valid_len = min(
                                original_lengths[layer1][b] if b < len(original_lengths[layer1]) else features1.shape[1],
                                original_lengths[layer2][b] if b < len(original_lengths[layer2]) else features2.shape[1],
                                original_lengths[cnn_layer][b] if b < len(original_lengths[cnn_layer]) else cnn_features.shape[1]
                            )
                            
                            if valid_len > 0:
                                all_X.append(features1[b, :valid_len, :])
                                all_Y.append(features2[b, :valid_len, :])
                                all_Z.append(cnn_batch[b, :valid_len, :])
                        
                        if all_X and all_Y and all_Z:
                            X = np.vstack(all_X)
                            Y = np.vstack(all_Y)
                            Z = np.vstack(all_Z)
                        else:
                            pbar.update(1)
                            continue
                    else:
                        # Reshape to 2D
                        X = features1.reshape(-1, features1.shape[-1])
                        Y = features2.reshape(-1, features2.shape[-1])
                        Z = cnn_batch.reshape(-1, cnn_batch.shape[-1])
                    
                    # Ensure same number of samples
                    min_samples = min(X.shape[0], Y.shape[0], Z.shape[0])
                    if min_samples >= 2:
                        X = X[:min_samples]
                        Y = Y[:min_samples]
                        Z = Z[:min_samples]
                        
                        try:
                            conditional_correlation_matrix[i, j] = compute_partial_correlation(X, Y, Z)
                            conditional_cka_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='residual')
                        except:
                            conditional_correlation_matrix[i, j] = 0.0
                            conditional_cka_matrix[i, j] = 0.0
                
                pbar.update(1)
    
    # Prepare results
    results = {
        'cosine': cosine_matrix,
        'correlation': correlation_matrix,
        'cka': cka_matrix,
        'layers': layers
    }
    
    if include_conditional:
        results['conditional_correlation'] = conditional_correlation_matrix
        results['conditional_cka'] = conditional_cka_matrix
        results['cnn_layer'] = cnn_layer
    
    return results


def plot_similarity_matrices(similarity_results: Dict[str, np.ndarray], 
                            output_dir: str, model_name: str, num_files: int,
                            include_conditional: bool = False):
    """
    Plot similarity matrices with multiple metrics.
    
    Args:
        similarity_results: Results from compute_layer_similarities
        output_dir: Output directory for plots
        model_name: Model name for titles
        num_files: Number of files processed
        include_conditional: Whether to include conditional plots
    """
    layers = similarity_results['layers']
    
    if include_conditional and 'conditional_correlation' in similarity_results:
        # Create comparison plot with conditional vs unconditional
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # Row 1: Unconditional metrics
        sns.heatmap(similarity_results['correlation'], ax=axes[0, 0],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Correlation'})
        axes[0, 0].set_title('Unconditional Correlation', fontsize=14)
        
        sns.heatmap(similarity_results['cka'], ax=axes[0, 1],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'CKA'})
        axes[0, 1].set_title('Unconditional CKA', fontsize=14)
        
        # Difference plot
        diff_correlation = similarity_results['correlation'] - similarity_results['conditional_correlation']
        sns.heatmap(diff_correlation, ax=axes[0, 2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='coolwarm', center=0, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Difference'})
        axes[0, 2].set_title('Correlation Difference\n(Unconditional - Conditional)', fontsize=14)
        
        # Row 2: Conditional metrics
        cnn_layer = similarity_results['cnn_layer']
        sns.heatmap(similarity_results['conditional_correlation'], ax=axes[1, 0],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Partial Correlation'})
        axes[1, 0].set_title(f'Partial Correlation | {cnn_layer}', fontsize=14)
        
        sns.heatmap(similarity_results['conditional_cka'], ax=axes[1, 1],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Conditional CKA'})
        axes[1, 1].set_title(f'Conditional CKA | {cnn_layer}', fontsize=14)
        
        sns.heatmap(similarity_results['cosine'], ax=axes[1, 2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Similarity'})
        axes[1, 2].set_title('Cosine Similarity (Time-Averaged)', fontsize=14)
        
        plt.suptitle(f'Conditional vs Unconditional Layer Similarity - {model_name} (n={num_files} files)', fontsize=16)
        plt.tight_layout()
        save_figure(fig, f'{output_dir}/conditional_layer_similarity_{model_name}_n{num_files}.png')
        
    else:
        # Create standard similarity plot
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        # Cosine similarity
        sns.heatmap(similarity_results['cosine'], ax=axes[0], 
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Similarity'})
        axes[0].set_title('Cosine Similarity (Time-Averaged)', fontsize=14)
        
        # Correlation
        sns.heatmap(similarity_results['correlation'], ax=axes[1],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Correlation (Time-Averaged)', fontsize=14)
        
        # CKA
        sns.heatmap(similarity_results['cka'], ax=axes[2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'CKA Similarity'})
        axes[2].set_title('CKA (Centered Kernel Alignment)', fontsize=14)
        
        plt.suptitle(f'Layer Similarity Analysis - {model_name} (n={num_files} files)', fontsize=16)
        plt.tight_layout()
        save_figure(fig, f'{output_dir}/layer_similarity_{model_name}_n{num_files}.png')
    
    # Save matrices for later use
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for metric, matrix in similarity_results.items():
        if isinstance(matrix, np.ndarray):
            np.save(f'{output_dir}/{metric}_matrix.npy', matrix)


def analyze_feature_divergence(layer_features: Dict[str, np.ndarray], 
                              output_dir: str, model_name: str, num_files: int,
                              reference_layer: str = None) -> Dict[str, List[float]]:
    """
    Analyze how features diverge from a reference layer.
    
    Args:
        layer_features: Dictionary of layer features
        output_dir: Output directory
        model_name: Model name
        num_files: Number of files
        reference_layer: Reference layer (defaults to first layer)
    
    Returns:
        Dictionary of divergence metrics
    """
    # Filter and sort layers
    layers = filter_and_sort_layers(layer_features)
    
    # Use the first layer as reference if not specified
    if reference_layer is None:
        reference_layer = layers[0]
    
    print(f"Using {reference_layer} as reference layer for divergence analysis")
    
    reference_features = layer_features[reference_layer]
    if len(reference_features.shape) > 2:
        reference_features = reference_features.reshape(reference_features.shape[0], -1)
    
    # Compute divergence metrics for each layer
    divergence_metrics = {'cosine_distance': [], 'l2_distance': [], 'correlation': []}
    layer_names = []
    
    for layer in layers:
        if layer == reference_layer:
            continue
            
        features = layer_features[layer]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Ensure same shape for comparison
        min_samples = min(reference_features.shape[0], features.shape[0])
        ref_subset = reference_features[:min_samples]
        feat_subset = features[:min_samples]
        
        # Compute various divergence metrics
        # 1. Average cosine distance
        cosine_sims = []
        for i in range(min_samples):
            cos_sim = compute_cosine_similarity(ref_subset[i], feat_subset[i])
            cosine_sims.append(cos_sim)
        avg_cosine_distance = 1 - np.mean(cosine_sims)
        
        # 2. Average L2 distance
        l2_distances = np.linalg.norm(feat_subset - ref_subset, axis=1)
        avg_l2_distance = np.mean(l2_distances)
        
        # 3. Correlation coefficient
        try:
            correlation = np.corrcoef(ref_subset.flatten(), feat_subset.flatten())[0, 1]
        except:
            correlation = 0.0
        
        divergence_metrics['cosine_distance'].append(avg_cosine_distance)
        divergence_metrics['l2_distance'].append(avg_l2_distance)
        divergence_metrics['correlation'].append(correlation)
        layer_names.append(layer)
    
    # Plot divergence metrics
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Cosine distance from reference
    axes[0].plot(range(len(layer_names)), divergence_metrics['cosine_distance'], 'o-', color='blue')
    axes[0].set_xticks(range(len(layer_names)))
    axes[0].set_xticklabels(layer_names, rotation=45)
    axes[0].set_title(f'Cosine Distance from {reference_layer}')
    axes[0].set_ylabel('Cosine Distance')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: L2 distance from reference
    axes[1].plot(range(len(layer_names)), divergence_metrics['l2_distance'], 'o-', color='red')
    axes[1].set_xticks(range(len(layer_names)))
    axes[1].set_xticklabels(layer_names, rotation=45)
    axes[1].set_title(f'L2 Distance from {reference_layer}')
    axes[1].set_ylabel('L2 Distance')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation with reference
    axes[2].plot(range(len(layer_names)), divergence_metrics['correlation'], 'o-', color='green')
    axes[2].set_xticks(range(len(layer_names)))
    axes[2].set_xticklabels(layer_names, rotation=45)
    axes[2].set_title(f'Correlation with {reference_layer}')
    axes[2].set_ylabel('Correlation Coefficient')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/feature_divergence_{model_name}_n{num_files}.png')
    
    divergence_metrics['layers'] = layer_names
    return divergence_metrics 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from utils.data_utils import filter_and_sort_layers
from utils.math_utils import (
    compute_cka, compute_cka_without_padding, compute_cka_gpu,
    compute_partial_correlation, compute_partial_correlation_gpu,
    # compute_conditional_cka,  # Commented out due to regression step issues
    # compute_cosine_similarity, compute_cosine_similarity_gpu,  # Commented out as these functions are being refactored
    # NEW: Import the enhanced correlation analyses
    compute_input_layer_correlations,
    compute_progressive_partial_correlations,
    compute_r_squared,
    analyze_input_propagation,
    CorrelationAnalyzer
)
from utils.visualization_utils import save_figure


def compute_layer_similarities(layer_features: Dict[str, np.ndarray], 
                              original_lengths: Dict[str, List[int]],
                              include_conditional: bool = False,
                              cnn_layer: str = 'transformer_input',
                              use_gpu: bool = True,
                              n_jobs: int = -1) -> Dict[str, np.ndarray]:
    """
    Compute similarity matrices between layers with GPU acceleration.
    
    Args:
        layer_features: Dictionary of layer features
        original_lengths: Dictionary of original sequence lengths
        include_conditional: Whether to compute conditional similarities
        cnn_layer: Name of CNN output layer for conditioning
        use_gpu: Whether to use GPU acceleration
        n_jobs: Number of parallel jobs for CPU operations
    
    Returns:
        Dictionary containing similarity matrices
    """
    # Filter and sort layers
    layers = filter_and_sort_layers(layer_features)
    n_layers = len(layers)
    
    # Initialize matrices (cosine, correlation, cka)
    # cosine_matrix = np.zeros((n_layers, n_layers))  # Commented out as cosine similarity is being refactored
    correlation_matrix = np.zeros((n_layers, n_layers))
    cka_matrix = np.zeros((n_layers, n_layers))
    
    if include_conditional:
        conditional_correlation_matrix = np.zeros((n_layers, n_layers))
        # conditional_cka_matrix = np.zeros((n_layers, n_layers))  # Commented out as conditional CKA is being refactored
    
    # Conditional matrices (if requested)
    if include_conditional and cnn_layer in layer_features:
        # conditional_correlation_matrix = np.zeros((n_layers, n_layers)) #TODO: Check if this is needed
        # conditional_cka_matrix = np.zeros((n_layers, n_layers)) #TODO: conditional CKA is being refactored
        cnn_features = layer_features[cnn_layer]
    else:
        conditional_correlation_matrix = None
        # conditional_cka_matrix = None
        cnn_features = None
    
    print("Computing layer similarities...")
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"Using GPU acceleration on {torch.cuda.get_device_name(0)}")
            else:
                print("GPU not available, falling back to CPU")
                use_gpu = False
        except ImportError:
            print("PyTorch not available, falling back to CPU")
            use_gpu = False
    
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
                
                # # Method 1: Average cosine similarity (non-signifcant TODO: will be removed)
                # f1_avg = features1.mean(axis=1)  # (batch, features)
                # f2_avg = features2.mean(axis=1)  # (batch, features)
                
                # Average over time dimension if 3D
                if len(features1.shape) == 3:
                    f1_avg = features1.mean(axis=1)  # (batch, features) #TODO: Taking average over time dimension needs to be checked -- may dampen the signal
                    f2_avg = features2.mean(axis=1)  # (batch, features) #TODO: Taking average over time dimension needs to be checked -- may dampen the signal
                else:
                    f1_avg = features1
                    f2_avg = features2
                
                # Ensure same feature dimension
                min_features = min(f1_avg.shape[1], f2_avg.shape[1])
                f1_avg_truncated = f1_avg[:, :min_features]
                f2_avg_truncated = f2_avg[:, :min_features]
                
                # Compute cosine similarity (with GPU option)
                # if use_gpu:
                #     try:
                #         # cosine_matrix[i, j] = compute_cosine_similarity_gpu(  # Commented out as this function is being refactored
                #         #     f1_avg_truncated, f2_avg_truncated, device='cuda'
                #         # )
                #         # Fallback to CPU
                #         cos_sims = []
                #         for b in range(min_batch):
                #             # cos_sim = compute_cosine_similarity(f1_avg_truncated[b], f2_avg_truncated[b])  # Commented out as this function is being refactored
                #             cos_sims.append(cos_sim)
                #         cosine_matrix[i, j] = np.mean(cos_sims)
                #     except:
                #         # Fallback to CPU
                #         cos_sims = []
                #         for b in range(min_batch):
                #             # cos_sim = compute_cosine_similarity(f1_avg_truncated[b], f2_avg_truncated[b])  # Commented out as this function is being refactored
                #             cos_sims.append(cos_sim)
                #         cosine_matrix[i, j] = np.mean(cos_sims)
                # else:
                #     cos_sims = []
                #     for b in range(min_batch):
                #         # cos_sim = compute_cosine_similarity(f1_avg_truncated[b], f2_avg_truncated[b])  # Commented out as this function is being refactored
                #         cos_sims.append(cos_sim)
                #     cosine_matrix[i, j] = np.mean(cos_sims)
                
                # Method 2: Correlation
                try:
                    correlation_matrix[i, j] = np.corrcoef(
                        f1_avg_truncated.flatten(), f2_avg_truncated.flatten() #TODO: check if this is correct
                    )[0, 1]
                except:
                    correlation_matrix[i, j] = 0.0
                
                # Method 3: CKA (TODO: check if CKA implementation is correct)
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
                    
                    if use_gpu:
                        try:
                            cka_matrix[i, j] = compute_cka_gpu(X, Y, device='cuda')
                        except:
                            cka_matrix[i, j] = compute_cka(X, Y)
                    else:
                        cka_matrix[i, j] = compute_cka(X, Y)
                
                # Conditional similarities (if requested) with GPU acceleration
                if include_conditional and cnn_features is not None:
                    # Prepare features for conditional analysis
                    cnn_batch = cnn_features[:min_batch]
                    
                    # Ensure same batch size
                    min_batch = min(features1.shape[0], features2.shape[0], cnn_features.shape[0])
                    features1 = features1[:min_batch]
                    features2 = features2[:min_batch]
                    cnn_features = cnn_features[:min_batch]
                    
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
                            # Use GPU-accelerated versions if available
                            if use_gpu:
                                try:
                                    # Partial correlation
                                    conditional_correlation_matrix[i, j] = compute_partial_correlation(X, Y, Z)
                                    
                                    # Conditional CKA (residual method)
                                    # conditional_cka_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='residual')  # Commented out due to regression step issues
                                    
                                    # Conditional CKA (partial method)
                                    # conditional_cka_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='partial')  # Commented out due to regression step issues
                                    
                                except:
                                    # Fallback to CPU
                                    conditional_correlation_matrix[i, j] = compute_partial_correlation(X, Y, Z)
                                    # conditional_cka_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='residual')  # Commented out due to regression step issues
                            else:
                                conditional_correlation_matrix[i, j] = compute_partial_correlation(X, Y, Z)
                                # conditional_cka_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='residual')  # Commented out due to regression step issues
                        except:
                            conditional_correlation_matrix[i, j] = 0.0
                            # conditional_cka_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='residual')  # Commented out due to regression step issues
                
                pbar.update(1)
    
    # Prepare results
    results = {
        # 'cosine': cosine_matrix,  # Commented out as cosine similarity is being refactored
        'correlation': correlation_matrix,
        'cka': cka_matrix,
        'layers': layers
    }
    
    if include_conditional:
        results['conditional_correlation'] = conditional_correlation_matrix
        # results['conditional_cka'] = conditional_cka_matrix  # Commented out as conditional CKA is being refactored
        results['cnn_layer'] = cnn_layer
    
    return results


def compute_input_propagation_similarities(layer_features: Dict[str, np.ndarray],
                                         original_lengths: Dict[str, List[int]],
                                         cnn_layer: str = 'transformer_input',
                                         use_gpu: bool = True,
                                         n_jobs: int = -1,
                                         show_progress: bool = True) -> Dict[str, Dict[str, float]]:
    """
    NEW: Compute the three types of input propagation correlations.
    
    This computes:
    1. Simple Input-Layer Correlations (signal retention)
    2. Progressive Partial Correlations (new information per layer)  
    3. R² Analysis (variance explained by each layer)
    
    Args:
        layer_features: Dictionary of layer features
        original_lengths: Dictionary of original sequence lengths
        cnn_layer: Name of CNN output layer (transformer input)
        use_gpu: Whether to use GPU acceleration
        n_jobs: Number of parallel jobs
        show_progress: Whether to show progress bars
    
    Returns:
        Dictionary containing all three correlation analyses
    """
    print("Computing input propagation similarities...")
    
    if cnn_layer not in layer_features:
        print(f"Warning: CNN layer '{cnn_layer}' not found. Skipping input propagation analysis.")
        return {}
    
    # Use the unified analyzer for efficiency
    try:
        analyzer = CorrelationAnalyzer(
            layer_features=layer_features,
            original_lengths=original_lengths,
            cnn_layer=cnn_layer,
            max_layer=11,
            use_gpu=use_gpu,
            n_jobs=n_jobs
        )
        
        # Compute all analyses
        results = analyzer.compute_all_analyses(show_progress=show_progress)
        
        print(f"✓ Input propagation analysis completed")
        print(f"  Performance info: {results.get('performance_info', {})}")
        
        return results
        
    except Exception as e:
        print(f"Error in input propagation analysis: {e}")
        
        # Fallback to individual function calls
        print("Falling back to individual function calls...")
        
        input_features = layer_features[cnn_layer]
        transformer_layers = {k: v for k, v in layer_features.items() 
                            if k.startswith('transformer_layer_')}
        
        try:
            fallback_results = analyze_input_propagation(
                input_features, transformer_layers,
                show_progress=show_progress,
                n_jobs=n_jobs,
                use_gpu=use_gpu
            )
            return fallback_results
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return {}


def plot_input_propagation_correlations(propagation_results: Dict[str, Dict[str, float]],
                                       output_dir: str, model_name: str, num_files: int,
                                       show_performance_info: bool = True):
    """
    NEW: Plot the three types of input propagation correlations.
    
    Creates visualizations for:
    1. Simple Input-Layer Correlations (signal retention)
    2. Progressive Partial Correlations (new information per layer)
    3. R² Analysis (variance explained)
    
    Args:
        propagation_results: Results from compute_input_propagation_similarities
        output_dir: Output directory for plots
        model_name: Model name for titles
        num_files: Number of files processed
        show_performance_info: Whether to include performance information
    """
    if not propagation_results:
        print("No propagation results to plot")
        return
    
    # Extract data
    layer_order = propagation_results.get('layer_order', [])
    simple_corrs = propagation_results.get('simple_correlations', {})
    partial_corrs = propagation_results.get('progressive_partial_correlations', {})
    r2_values = propagation_results.get('r_squared_values', {})
    
    if not layer_order:
        print("No layer order information found")
        return
    
    # Prepare data for plotting
    layer_numbers = [int(layer.split('_')[-1]) for layer in layer_order]
    simple_values = [simple_corrs.get(layer, 0) for layer in layer_order]
    partial_values = [partial_corrs.get(layer, 0) for layer in layer_order]
    r2_vals = [r2_values.get(layer, 0) for layer in layer_order]
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Simple Input-Layer Correlations
    axes[0, 0].plot(layer_numbers, simple_values, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_title('Simple Input-Layer Correlations\n(Signal Retention)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Transformer Layer')
    axes[0, 0].set_ylabel('Correlation with Input')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-1, 1)
    
    # Add trend line
    z = np.polyfit(layer_numbers, simple_values, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(layer_numbers, p(layer_numbers), '--', alpha=0.7, color='blue')
    
    # Plot 2: Progressive Partial Correlations
    axes[0, 1].plot(layer_numbers, partial_values, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_title('Progressive Partial Correlations\n(New Information per Layer)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Transformer Layer')
    axes[0, 1].set_ylabel('Partial Correlation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-1, 1)
    
    # Add trend line
    z = np.polyfit(layer_numbers, partial_values, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(layer_numbers, p(layer_numbers), '--', alpha=0.7, color='orange')
    
    # Plot 3: R² Analysis
    axes[1, 0].plot(layer_numbers, r2_vals, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_title('R² Analysis\n(Variance Explained by Each Layer)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Transformer Layer')
    axes[1, 0].set_ylabel('R² (Variance Explained)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Add trend line
    z = np.polyfit(layer_numbers, r2_vals, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(layer_numbers, p(layer_numbers), '--', alpha=0.7, color='green')
    
    # Plot 4: Combined comparison
    axes[1, 1].plot(layer_numbers, simple_values, 'o-', label='Simple Correlations', linewidth=2, color='blue')
    axes[1, 1].plot(layer_numbers, partial_values, 's-', label='Partial Correlations', linewidth=2, color='orange')
    axes[1, 1].plot(layer_numbers, r2_vals, '^-', label='R² Values', linewidth=2, color='green')
    axes[1, 1].set_title('Combined Analysis\n(All Three Correlation Types)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Transformer Layer')
    axes[1, 1].set_ylabel('Correlation/R² Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-1, 1)
    
    # Add overall title
    fig.suptitle(f'Input Propagation Analysis - {model_name} (n={num_files})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add performance information if available
    if show_performance_info and 'performance_info' in propagation_results:
        perf_info = propagation_results['performance_info']
        info_text = f"GPU: {perf_info.get('gpu_acceleration', False)}, "
        info_text += f"Jobs: {perf_info.get('parallel_jobs', 1)}, "
        info_text += f"Layers: {perf_info.get('num_layers', len(layer_order))}"
        
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save plot
    output_path = f"{output_dir}/input_propagation_analysis_{model_name}_n{num_files}.png"
    save_figure(fig, output_path)
    print(f"Saved input propagation analysis to {output_path}")


def plot_enhanced_similarity_matrices(similarity_results: Dict[str, np.ndarray], 
                                     propagation_results: Optional[Dict[str, Dict[str, float]]] = None,
                                     output_dir: str = "", model_name: str = "", num_files: int = 0,
                                     include_conditional: bool = False,
                                     use_gpu_info: bool = True):
    """
    Enhanced plotting function that combines layer-to-layer similarities with input propagation analyses.
    
    Args:
        similarity_results: Results from compute_layer_similarities  
        propagation_results: Results from compute_input_propagation_similarities
        output_dir: Output directory for plots
        model_name: Model name for titles
        num_files: Number of files processed
        include_conditional: Whether to include conditional plots
        use_gpu_info: Whether to include GPU performance information
    """
    layers = similarity_results['layers']
    
    if propagation_results and include_conditional and 'conditional_correlation' in similarity_results:
        # Create large comprehensive plot (layer similarities + input propagation + conditional)
        fig, axes = plt.subplots(3, 3, figsize=(24, 20))
        
        # Row 1: Layer-to-layer similarities
        sns.heatmap(similarity_results['correlation'], ax=axes[0, 0],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Correlation'})
        axes[0, 0].set_title('Layer-to-Layer Correlation', fontsize=14)
        
        sns.heatmap(similarity_results['cka'], ax=axes[0, 1],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'CKA'})
        axes[0, 1].set_title('Layer-to-Layer CKA', fontsize=14)
        
        sns.heatmap(similarity_results['cosine'], ax=axes[0, 2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Cosine Similarity'})
        axes[0, 2].set_title('Layer-to-Layer Cosine Similarity', fontsize=14)
        
        # Row 2: Conditional similarities
        sns.heatmap(similarity_results['conditional_correlation'], ax=axes[1, 0],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Partial Correlation'})
        axes[1, 0].set_title('Conditional Correlation\n(CNN Influence Removed)', fontsize=14)
        
        sns.heatmap(similarity_results['conditional_cka'], ax=axes[1, 1],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Conditional CKA'})
        axes[1, 1].set_title('Conditional CKA\n(CNN Influence Removed)', fontsize=14)
        
        # Difference plot
        diff_matrix = similarity_results['correlation'] - similarity_results['conditional_correlation']
        sns.heatmap(diff_matrix, ax=axes[1, 2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Difference'})
        axes[1, 2].set_title('Correlation Difference\n(Unconditional - Conditional)', fontsize=14)
        
        # Row 3: Input propagation analyses
        if propagation_results:
            layer_order = propagation_results.get('layer_order', [])
            simple_corrs = propagation_results.get('simple_correlations', {})
            partial_corrs = propagation_results.get('progressive_partial_correlations', {})
            r2_values = propagation_results.get('r_squared_values', {})
            
            if layer_order:
                layer_numbers = [int(layer.split('_')[-1]) for layer in layer_order]
                simple_values = [simple_corrs.get(layer, 0) for layer in layer_order]
                partial_values = [partial_corrs.get(layer, 0) for layer in layer_order]
                r2_vals = [r2_values.get(layer, 0) for layer in layer_order]
                
                # Simple correlations
                axes[2, 0].plot(layer_numbers, simple_values, 'o-', linewidth=2, color='blue')
                axes[2, 0].set_title('Simple Input-Layer Correlations', fontsize=14)
                axes[2, 0].set_xlabel('Layer')
                axes[2, 0].set_ylabel('Correlation')
                axes[2, 0].grid(True, alpha=0.3)
                
                # Progressive partial correlations
                axes[2, 1].plot(layer_numbers, partial_values, 'o-', linewidth=2, color='orange')
                axes[2, 1].set_title('Progressive Partial Correlations', fontsize=14)
                axes[2, 1].set_xlabel('Layer')
                axes[2, 1].set_ylabel('Partial Correlation')
                axes[2, 1].grid(True, alpha=0.3)
                
                # R² analysis
                axes[2, 2].plot(layer_numbers, r2_vals, 'o-', linewidth=2, color='green')
                axes[2, 2].set_title('R² Analysis', fontsize=14)
                axes[2, 2].set_xlabel('Layer')
                axes[2, 2].set_ylabel('R²')
                axes[2, 2].grid(True, alpha=0.3)
        
    elif include_conditional and 'conditional_correlation' in similarity_results:
        # Create comparison plot with conditional vs unconditional (original functionality)
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
        
        sns.heatmap(similarity_results['cosine'], ax=axes[0, 2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Cosine Similarity'})
        axes[0, 2].set_title('Cosine Similarity', fontsize=14)
        
        # Row 2: Conditional metrics
        sns.heatmap(similarity_results['conditional_correlation'], ax=axes[1, 0],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Partial Correlation'})
        axes[1, 0].set_title(f'Conditional Correlation\n(Conditioned on {similarity_results.get("cnn_layer", "CNN")})', fontsize=14)
        
        sns.heatmap(similarity_results['conditional_cka'], ax=axes[1, 1],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Conditional CKA'})
        axes[1, 1].set_title(f'Conditional CKA\n(Conditioned on {similarity_results.get("cnn_layer", "CNN")})', fontsize=14)
        
        # Difference plot
        diff_matrix = similarity_results['correlation'] - similarity_results['conditional_correlation']
        sns.heatmap(diff_matrix, ax=axes[1, 2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Difference'})
        axes[1, 2].set_title('Correlation Difference\n(Unconditional - Conditional)', fontsize=14)
        
    else:
        # Simple plot with just basic similarities (original functionality)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        sns.heatmap(similarity_results['correlation'], ax=axes[0],
                    xticklabels=layers, yticklabels=layers,
                    cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Layer Correlation', fontsize=14)
        
        sns.heatmap(similarity_results['cka'], ax=axes[1],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'CKA'})
        axes[1].set_title('Layer CKA', fontsize=14)
        
        sns.heatmap(similarity_results['cosine'], ax=axes[2],
                    xticklabels=layers, yticklabels=layers,
                    cmap='viridis', vmin=-1, vmax=1, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Cosine Similarity'})
        axes[2].set_title('Cosine Similarity', fontsize=14)
    
    # Add overall title and performance info
    title = f'Layer Similarity Analysis - {model_name} (n={num_files})'
    if use_gpu_info and propagation_results and 'performance_info' in propagation_results:
        perf_info = propagation_results['performance_info']
        if perf_info.get('gpu_acceleration', False):
            title += f" [GPU Accelerated]"
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    suffix = "_conditional" if include_conditional else ""
    suffix += "_with_propagation" if propagation_results else ""
    output_path = f"{output_dir}/enhanced_similarity_analysis{suffix}_{model_name}_n{num_files}.png"
    save_figure(fig, output_path)
    print(f"Saved enhanced similarity analysis to {output_path}")


# Keep the original function for backward compatibility
def plot_similarity_matrices(similarity_results: Dict[str, np.ndarray], 
                            output_dir: str, model_name: str, num_files: int,
                            include_conditional: bool = False):
    """
    Original plotting function for layer similarity matrices (backward compatibility).
    
    For enhanced functionality with new correlation types, use plot_enhanced_similarity_matrices.
    """
    return plot_enhanced_similarity_matrices(
        similarity_results, None, output_dir, model_name, num_files, 
        include_conditional, use_gpu_info=False
    )


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
            # cos_sim = compute_cosine_similarity(ref_subset[i], feat_subset[i])  # Commented out as this function is being refactored
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
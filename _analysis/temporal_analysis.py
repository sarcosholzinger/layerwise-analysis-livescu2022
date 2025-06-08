import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, List, Optional

from _utils.data_utils import filter_and_sort_layers, get_layer_number
from _utils.math_utils import (
    compute_cka, compute_cka_without_padding, compute_cka_gpu,
    compute_partial_correlation, 
    # compute_conditional_cka,  # Commented out due to regression step issues
    # compute_cosine_similarity, compute_cosine_similarity_gpu,  # Commented out as these functions are being refactored
)
from _utils.visualization_utils import save_figure


def compute_temporal_similarities(layer_features: Dict[str, np.ndarray], 
                                 original_lengths: Dict[str, List[int]], 
                                 window_size: int = 10, 
                                 stride: int = 5) -> Dict[str, List[np.ndarray]]:
    """
    Compute layer similarities for sliding windows across time.
    
    Args:
        layer_features: Dictionary of layer features
        original_lengths: Dictionary of original sequence lengths
        window_size: Size of sliding window
        stride: Stride for sliding window
    
    Returns:
        Dictionary containing temporal similarity matrices
    """
    # Filter and sort layers
    layers = filter_and_sort_layers(layer_features)
    n_layers = len(layers)
    
    # Get the minimum time steps across all layers
    min_time_steps = min([features.shape[1] for features in layer_features.values()])
    
    # Calculate number of windows
    n_windows = (min_time_steps - window_size) // stride + 1
    
    # Store similarity matrices for each time window
    # cosine_matrices = []  # Commented out as cosine similarity is being refactored
    correlation_matrices = []
    cka_matrices = []
    
    print(f"Computing similarities for {n_windows} time windows...")
    
    for window_idx in tqdm(range(n_windows)):
        start_time = window_idx * stride
        end_time = start_time + window_size
        
        # Compute similarities for this time window
        # cosine_matrix = np.zeros((n_layers, n_layers))  # Commented out as cosine similarity is being refactored
        correlation_matrix = np.zeros((n_layers, n_layers))
        cka_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                # Extract features for this time window
                features1 = layer_features[layer1][:, start_time:end_time, :]
                features2 = layer_features[layer2][:, start_time:end_time, :]
                
                # Ensure same batch size
                min_batch = min(features1.shape[0], features2.shape[0])
                features1 = features1[:min_batch]
                features2 = features2[:min_batch]
                
                # Average over time window
                f1_avg = features1.mean(axis=1)
                f2_avg = features2.mean(axis=1)
                
                # Ensure same feature dimension
                min_features = min(f1_avg.shape[1], f2_avg.shape[1])
                f1_avg = f1_avg[:, :min_features]
                f2_avg = f2_avg[:, :min_features]
                
                # Compute cosine similarity
                # cos_sims = []
                # for b in range(min_batch):
                #     # cos_sim = compute_cosine_similarity(f1_avg[b], f2_avg[b])  # Commented out as this function is being refactored
                #     cos_sims.append(cos_sim)
                # cosine_matrix[i, j] = np.mean(cos_sims)
                
                # Compute correlation
                if f1_avg.size > 0 and f2_avg.size > 0:
                    try:
                        correlation_matrix[i, j] = np.corrcoef(
                            f1_avg.flatten(), f2_avg.flatten()
                        )[0, 1]
                    except:
                        correlation_matrix[i, j] = 0.0
                
                # Compute CKA
                if layer1 in original_lengths and layer2 in original_lengths:
                    # Create temporary lengths for this window
                    temp_lens1 = []
                    temp_lens2 = []
                    
                    for orig_len1, orig_len2 in zip(original_lengths[layer1], original_lengths[layer2]):
                        # Adjust lengths for this window
                        valid_len1 = max(0, min(orig_len1 - start_time, window_size))
                        valid_len2 = max(0, min(orig_len2 - start_time, window_size))
                        temp_lens1.append(valid_len1)
                        temp_lens2.append(valid_len2)
                    
                    if any(l > 0 for l in temp_lens1) and any(l > 0 for l in temp_lens2):
                        cka_matrix[i, j] = compute_cka_without_padding(
                            features1, features2,
                            temp_lens1, temp_lens2
                        )
                    else:
                        cka_matrix[i, j] = 0.0
                else:
                    # Fallback to original method
                    X = features1.reshape(-1, features1.shape[-1])
                    Y = features2.reshape(-1, features2.shape[-1])
                    min_samples = min(X.shape[0], Y.shape[0])
                    if min_samples > 1:
                        cka_matrix[i, j] = compute_cka(X[:min_samples], Y[:min_samples])
                    else:
                        cka_matrix[i, j] = 0.0
        
        # Append matrices
        # cosine_matrices.append(cosine_matrix)  # Commented out as cosine similarity is being refactored
        correlation_matrices.append(correlation_matrix)
        cka_matrices.append(cka_matrix)
    
    return {
        # 'cosine': cosine_matrices,  # Commented out as cosine similarity is being refactored
        'correlation': correlation_matrices,
        'cka': cka_matrices,
        'layers': layers,
        'window_size': window_size,
        'stride': stride
    }


def create_similarity_animation(temporal_similarities: Dict[str, List[np.ndarray]], 
                               output_dir: str, model_name: str, 
                               metric: str = 'correlation') -> str:
    """
    Create an animation showing how similarities evolve over time.
    
    Args:
        temporal_similarities: Output from compute_temporal_similarities
        output_dir: Directory to save animation
        model_name: Model name for filename
        metric: Which metric to animate ('correlation', 'cka')
    
    Returns:
        Path to saved animation
    """
    matrices = temporal_similarities[metric]
    layers = temporal_similarities['layers']
    window_size = temporal_similarities['window_size']
    stride = temporal_similarities['stride']
    
    # Set up metric-specific parameters
    if metric == 'correlation':
        vmin, vmax = -1, 1
        cmap = 'RdBu_r'
        metric_name = 'Correlation'
    else:
        vmin, vmax = 0, 1
        cmap = 'viridis'
        metric_name = metric.upper() if metric == 'cka' else metric.capitalize()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the initial heatmap
    im = ax.imshow(matrices[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric_name} Similarity')
    
    # Set ticks and labels
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_yticklabels(layers)
    
    # Add title
    title = ax.set_title(f'{metric_name} Similarity - {model_name}\nTime: 0-{window_size} steps')
    
    # Add text annotations for values
    text_annotations = []
    for i in range(len(layers)):
        text_row = []
        for j in range(len(layers)):
            text = ax.text(j, i, f'{matrices[0][i, j]:.2f}',
                         ha='center', va='center', color='white', fontsize=8)
            text_row.append(text)
        text_annotations.append(text_row)
    
    def update(frame):
        # Update the heatmap data
        im.set_array(matrices[frame])
        
        # Update title with current time window
        start_time = frame * stride
        end_time = start_time + window_size
        title.set_text(f'{metric_name} Similarity - {model_name}\nTime: {start_time}-{end_time} steps')
        
        # Update text annotations
        for i in range(len(layers)):
            for j in range(len(layers)):
                text_annotations[i][j].set_text(f'{matrices[frame][i, j]:.2f}')
        
        return [im, title] + [text for row in text_annotations for text in row]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(matrices), interval=200, blit=True)
    
    # Save as GIF
    output_path = f'{output_dir}/layer_similarity_{metric}_{model_name}_animation.gif'
    anim.save(output_path, writer=PillowWriter(fps=5))
    plt.close()
    
    print(f"Animation saved to {output_path}")
    return output_path


def compute_conditional_temporal_similarities(layer_features: Dict[str, np.ndarray], 
                                            original_lengths: Dict[str, List[int]], 
                                            cnn_layer: str = 'transformer_input',
                                            window_size: int = 10, 
                                            stride: int = 5) -> Optional[Dict[str, List[np.ndarray]]]:
    """
    Compute conditional layer similarities (conditioned on CNN output) for sliding windows across time.
    
    Args:
        layer_features: Dictionary of layer features
        original_lengths: Dictionary of original sequence lengths
        cnn_layer: CNN output layer name for conditioning
        window_size: Size of sliding window
        stride: Stride for sliding window
    
    Returns:
        Dictionary containing conditional temporal similarity matrices, or None if CNN layer not found
    """
    # Check if CNN output layer exists
    if cnn_layer not in layer_features:
        print(f"Warning: {cnn_layer} not found. Cannot compute conditional similarities.")
        return None
    
    cnn_features = layer_features[cnn_layer]
    
    # Filter and sort layers (excluding CNN output)
    layers = sorted([layer for layer in layer_features.keys() 
                    if layer != cnn_layer and
                    (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                   key=get_layer_number)
    
    n_layers = len(layers)
    
    # Get the minimum time steps across all layers
    min_time_steps = min([features.shape[1] for features in layer_features.values()])
    
    # Calculate number of windows
    n_windows = (min_time_steps - window_size) // stride + 1
    
    # Store similarity matrices for each time window
    uncond_correlation_matrices = []
    uncond_cka_matrices = []
    partial_correlation_matrices = []
    conditional_cka_matrices = []
    
    print(f"Computing conditional similarities for {n_windows} time windows...")
    
    for window_idx in tqdm(range(n_windows), desc="Processing time windows"):
        start_time = window_idx * stride
        end_time = start_time + window_size
        
        # Initialize matrices for this window
        uncond_corr_matrix = np.zeros((n_layers, n_layers))
        uncond_cka_matrix = np.zeros((n_layers, n_layers))
        partial_corr_matrix = np.zeros((n_layers, n_layers))
        cond_cka_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                # Extract features for this time window
                features1 = layer_features[layer1][:, start_time:end_time, :]
                features2 = layer_features[layer2][:, start_time:end_time, :]
                cnn_window = cnn_features[:, start_time:end_time, :]
                
                # Ensure same batch size
                min_batch = min(features1.shape[0], features2.shape[0], cnn_window.shape[0])
                features1 = features1[:min_batch]
                features2 = features2[:min_batch]
                cnn_window = cnn_window[:min_batch]
                
                # Handle padding if original lengths are available
                if (layer1 in original_lengths and layer2 in original_lengths and 
                    cnn_layer in original_lengths):
                    # Create temporary lengths for this window
                    all_X = []
                    all_Y = []
                    all_Z = []
                    
                    for b in range(min_batch):
                        # Adjust lengths for this window
                        valid_len1 = max(0, min(original_lengths[layer1][b] - start_time, window_size))
                        valid_len2 = max(0, min(original_lengths[layer2][b] - start_time, window_size))
                        valid_len_cnn = max(0, min(original_lengths[cnn_layer][b] - start_time, window_size))
                        valid_len = min(valid_len1, valid_len2, valid_len_cnn)
                        
                        if valid_len > 0:
                            all_X.append(features1[b, :valid_len, :])
                            all_Y.append(features2[b, :valid_len, :])
                            all_Z.append(cnn_window[b, :valid_len, :])
                    
                    if all_X and all_Y and all_Z:
                        X = np.vstack(all_X)
                        Y = np.vstack(all_Y)
                        Z = np.vstack(all_Z)
                    else:
                        continue
                else:
                    # Reshape to 2D
                    X = features1.reshape(-1, features1.shape[-1])
                    Y = features2.reshape(-1, features2.shape[-1])
                    Z = cnn_window.reshape(-1, cnn_window.shape[-1])
                
                # Ensure same number of samples
                min_samples = min(X.shape[0], Y.shape[0], Z.shape[0])
                if min_samples < 2:
                    continue
                
                X = X[:min_samples]
                Y = Y[:min_samples]
                Z = Z[:min_samples]
                
                try:
                    # Compute unconditional metrics
                    uncond_corr_matrix[i, j] = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
                    uncond_cka_matrix[i, j] = compute_cka(X, Y)
                    
                    # Compute conditional metrics
                    partial_corr_matrix[i, j] = compute_partial_correlation(X, Y, Z)
                    # cond_cka_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='residual')  # Commented out due to regression step issues
                    
                except Exception as e:
                    # Set to NaN for failed computations
                    uncond_corr_matrix[i, j] = np.nan
                    uncond_cka_matrix[i, j] = np.nan
                    partial_corr_matrix[i, j] = np.nan
                    cond_cka_matrix[i, j] = np.nan
        
        # Replace NaN with 0 for visualization
        uncond_corr_matrix = np.nan_to_num(uncond_corr_matrix, 0)
        uncond_cka_matrix = np.nan_to_num(uncond_cka_matrix, 0)
        partial_corr_matrix = np.nan_to_num(partial_corr_matrix, 0)
        cond_cka_matrix = np.nan_to_num(cond_cka_matrix, 0)
        
        # Append matrices
        uncond_correlation_matrices.append(uncond_corr_matrix)
        uncond_cka_matrices.append(uncond_cka_matrix)
        partial_correlation_matrices.append(partial_corr_matrix)
        conditional_cka_matrices.append(cond_cka_matrix)
    
    return {
        'unconditional_correlation': uncond_correlation_matrices,
        'unconditional_cka': uncond_cka_matrices,
        'partial_correlation': partial_correlation_matrices,
        'conditional_cka': conditional_cka_matrices,
        'layers': layers,
        'cnn_layer': cnn_layer,
        'window_size': window_size,
        'stride': stride
    }


def create_conditional_similarity_animation(temporal_similarities: Dict[str, List[np.ndarray]], 
                                          output_dir: str, model_name: str, 
                                          metric: str = 'partial_correlation', 
                                          comparison_mode: str = 'side_by_side') -> str:
    """
    Create animations showing conditional similarities over time.
    
    Args:
        temporal_similarities: Output from compute_conditional_temporal_similarities
        output_dir: Directory to save animations
        model_name: Model name for title
        metric: One of 'partial_correlation', 'conditional_cka'
        comparison_mode: 'side_by_side' to show uncond vs cond, 'difference' to show the difference
    
    Returns:
        Path to saved animation
    """
    if temporal_similarities is None:
        print("No temporal similarities data available.")
        return ""
    
    # Get the appropriate matrices
    if metric == 'partial_correlation':
        cond_matrices = temporal_similarities['partial_correlation']
        uncond_matrices = temporal_similarities['unconditional_correlation']
        metric_name = 'Partial Correlation'
        uncond_name = 'Correlation'
        vmin, vmax = -1, 1
        cmap = 'RdBu_r'
    elif metric == 'conditional_cka':
        cond_matrices = temporal_similarities['conditional_cka']
        uncond_matrices = temporal_similarities['unconditional_cka']
        metric_name = 'Conditional CKA'
        uncond_name = 'CKA'
        vmin, vmax = 0, 1
        cmap = 'viridis'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    layers = temporal_similarities['layers']
    cnn_layer = temporal_similarities['cnn_layer']
    window_size = temporal_similarities['window_size']
    stride = temporal_similarities['stride']
    
    if comparison_mode == 'side_by_side':
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Set up initial heatmaps
        im1 = ax1.imshow(uncond_matrices[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        im2 = ax2.imshow(cond_matrices[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # Add colorbars
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label(uncond_name)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label(metric_name)
        
        # Set ticks and labels
        for ax in [ax1, ax2]:
            ax.set_xticks(range(len(layers)))
            ax.set_yticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha='right')
            ax.set_yticklabels(layers)
        
        # Add titles
        title1 = ax1.set_title(f'Unconditional {uncond_name}')
        title2 = ax2.set_title(f'{metric_name} | {cnn_layer}')
        
        # Main title
        suptitle = fig.suptitle(f'{model_name} - Time: 0-{window_size} steps')
        
        def update(frame):
            # Update heatmap data
            im1.set_array(uncond_matrices[frame])
            im2.set_array(cond_matrices[frame])
            
            # Update title with current time window
            start_time = frame * stride
            end_time = start_time + window_size
            suptitle.set_text(f'{model_name} - Time: {start_time}-{end_time} steps')
            
            return [im1, im2, suptitle]
    
    elif comparison_mode == 'difference':
        # Show difference between unconditional and conditional
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compute initial difference
        diff_matrix = uncond_matrices[0] - cond_matrices[0]
        
        # Set up heatmap
        im = ax.imshow(diff_matrix, cmap='coolwarm', aspect='auto', 
                      vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'Difference ({uncond_name} - {metric_name})')
        
        # Set ticks and labels
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_yticklabels(layers)
        
        # Add title
        title = ax.set_title(f'Effect of CNN Conditioning - {model_name}\nTime: 0-{window_size} steps')
        
        def update(frame):
            # Compute difference for this frame
            diff_matrix = uncond_matrices[frame] - cond_matrices[frame]
            im.set_array(diff_matrix)
            
            # Update title
            start_time = frame * stride
            end_time = start_time + window_size
            title.set_text(f'Effect of CNN Conditioning - {model_name}\nTime: {start_time}-{end_time} steps')
            
            return [im, title]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(cond_matrices), interval=200, blit=True)
    
    # Save as GIF
    mode_suffix = 'comparison' if comparison_mode == 'side_by_side' else 'difference'
    output_path = f'{output_dir}/conditional_{metric}_{mode_suffix}_{model_name}_animation.gif'
    anim.save(output_path, writer=PillowWriter(fps=5))
    plt.close()
    
    print(f"Animation saved to {output_path}")
    return output_path 
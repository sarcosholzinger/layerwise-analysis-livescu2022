import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from matplotlib.animation import FuncAnimation, PillowWriter

# Set default plotting style
plt.style.use('default')
sns.set_palette("husl")


def setup_plot_style():
    """Set up consistent plotting style across all visualizations."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })


def save_figure(fig, output_path: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save figure with consistent settings.
    
    Args:
        fig: matplotlib figure object
        output_path: path to save the figure
        dpi: resolution
        bbox_inches: bbox setting for saving
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def plot_feature_distributions(layer_features: Dict[str, np.ndarray], 
                              output_dir: str, model_name: str, num_files: int):
    """Plot feature distributions for each layer."""
    from .data_utils import filter_and_sort_layers
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Sort layers numerically
    layers = filter_and_sort_layers(layer_features)
    
    # Plot mean and std of features for each layer
    means = []
    stds = []
    
    for layer in layers:
        features = layer_features[layer]
        # Reshape to 2D if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        means.append(np.mean(features))
        stds.append(np.std(features))
    
    ax.errorbar(range(len(layers)), means, yerr=stds, fmt='o-', capsize=5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_title('Feature Statistics Across Layers')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Feature Value')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/feature_distributions_{model_name}_n{num_files}.png')


def plot_layer_statistics(layer_features: Dict[str, np.ndarray], 
                         output_dir: str, model_name: str, num_files: int):
    """Plot statistics of activations across layers to show their progression."""
    from .data_utils import filter_and_sort_layers
    
    layers = filter_and_sort_layers(layer_features)
    
    # Analyze all dimensions instead of just the last one
    all_dim_means = []
    all_dim_stds = []
    activation_magnitude_means = []
    activation_magnitude_stds = []
    
    for layer in layers:
        features = layer_features[layer]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Compute statistics across all dimensions
        dim_means = np.mean(features, axis=0)  # Mean of each dimension
        dim_stds = np.std(features, axis=0)    # Std of each dimension
        
        all_dim_means.append(np.mean(dim_means))
        all_dim_stds.append(np.mean(dim_stds))
        
        # Also compute activation magnitudes (L2 norm)
        activation_magnitudes = np.linalg.norm(features, axis=1)
        activation_magnitude_means.append(np.mean(activation_magnitudes))
        activation_magnitude_stds.append(np.std(activation_magnitudes))
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Average feature values across dimensions
    axes[0].errorbar(range(len(layers)), all_dim_means, yerr=all_dim_stds, 
                    fmt='o-', capsize=5, label='Mean ± Std across dimensions')
    axes[0].set_xticks(range(len(layers)))
    axes[0].set_xticklabels(layers, rotation=45)
    axes[0].set_title('Average Feature Statistics Across All Dimensions')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Average Feature Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Activation magnitude progression
    axes[1].errorbar(range(len(layers)), activation_magnitude_means, 
                    yerr=activation_magnitude_stds, 
                    fmt='o-', capsize=5, color='green', label='Mean L2 Norm ± Std')
    axes[1].set_xticks(range(len(layers)))
    axes[1].set_xticklabels(layers, rotation=45)
    axes[1].set_title('Activation Magnitude (L2 Norm) Progression Across Layers')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('L2 Norm')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Dimension-wise variance distribution
    dim_variances = []
    for layer in layers:
        features = layer_features[layer]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        var_per_dim = np.var(features, axis=0)
        dim_variances.append(np.mean(var_per_dim))
    
    axes[2].plot(range(len(layers)), dim_variances, 'o-', color='purple')
    axes[2].set_xticks(range(len(layers)))
    axes[2].set_xticklabels(layers, rotation=45)
    axes[2].set_title('Average Variance Per Dimension Across Layers')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Average Variance')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/layer_statistics_{model_name}_n{num_files}.png')


def create_similarity_heatmap(similarity_matrix: np.ndarray, 
                             layers: List[str],
                             title: str,
                             cmap: str = 'viridis',
                             vmin: float = 0,
                             vmax: float = 1,
                             ax=None) -> plt.Axes:
    """
    Create a similarity heatmap with consistent styling.
    
    Args:
        similarity_matrix: Square similarity matrix
        layers: Layer names for labels
        title: Plot title
        cmap: Colormap
        vmin, vmax: Color scale limits
        ax: Optional existing axes
    
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_yticklabels(layers)
    
    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(layers)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                         ha='center', va='center', 
                         color='white' if similarity_matrix[i, j] < (vmax - vmin) / 2 + vmin else 'black',
                         fontsize=8)
    
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity')
    
    return ax


def plot_padding_ratios(layer_features: Dict[str, np.ndarray], 
                       original_lengths: Dict[str, List[int]], 
                       output_dir: str, model_name: str, num_files: int):
    """Plot the ratio of valid (non-padded) time steps across layers."""
    from .data_utils import filter_and_sort_layers
    
    # Get all layers
    layers = filter_and_sort_layers(layer_features)
    
    # Get max sequence length
    max_length = max(features.shape[1] for features in layer_features.values())
    
    # Initialize padding ratio matrices
    aggregate_padding_ratios = np.zeros((len(layers), max_length))
    
    # Compute padding ratios for each layer and time step
    for i, layer in enumerate(layers):
        if layer in original_lengths:
            for t in range(max_length):
                valid_count = sum(1 for length in original_lengths[layer] if t < length)
                aggregate_padding_ratios[i, t] = valid_count / len(original_lengths[layer])
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(aggregate_padding_ratios, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    
    # Set labels and ticks
    ax.set_xticks(range(0, max_length, 50))  # Show every 50th time step
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(range(0, max_length, 50))
    ax.set_yticklabels(layers)
    
    ax.set_title(f'Aggregate Padding Ratios Across Layers and Time Steps\n{model_name} (n={num_files} files)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Layer')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ratio of Valid Sequences')
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/padding_ratios_aggregate_{model_name}_n{num_files}.png')
    
    return aggregate_padding_ratios


def create_animation_figure(matrix: np.ndarray, layers: List[str], 
                           title: str, cmap: str = 'viridis', 
                           vmin: float = 0, vmax: float = 1) -> Tuple[plt.Figure, plt.Axes, plt.Image]:
    """
    Create a figure setup for animations.
    
    Returns:
        Tuple of (figure, axes, image) for use in animations
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the initial heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity')
    
    # Set ticks and labels
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_yticklabels(layers)
    
    # Add title
    ax.set_title(title)
    
    return fig, ax, im 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
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
                           vmin: float = 0, vmax: float = 1) -> Tuple[plt.Figure, plt.Axes, Any]:
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


def create_correlation_plot(values: List[float], layers: List[str], 
                          title: str, ylabel: str, color: str = 'blue',
                          ylim: Tuple[float, float] = (-1, 1),
                          figsize: Tuple[int, int] = (12, 6),
                          add_annotations: bool = True) -> plt.Figure:
    """
    Create a standardized correlation/R² line plot.
    
    Args:
        values: List of correlation/R² values
        layers: List of layer names
        title: Plot title
        ylabel: Y-axis label
        color: Line color
        ylim: Y-axis limits
        figsize: Figure size
        add_annotations: Whether to add value annotations
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(range(len(layers)), values, 'o-', linewidth=2, markersize=8, color=color)
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace('transformer_layer_', 'L') for l in layers], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ylim)
    
    # Add value annotations
    if add_annotations:
        for i, val in enumerate(values):
            ax.annotate(f'{val:.3f}', xy=(i, val), xytext=(0, 5), 
                       textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    return fig


def create_bar_plot(values: List[float], layers: List[str],
                   title: str, ylabel: str, color: str = 'blue',
                   ylim: Tuple[float, float] = (-1, 1),
                   figsize: Tuple[int, int] = (12, 6),
                   add_value_labels: bool = True) -> plt.Figure:
    """
    Create a standardized bar plot for correlation/R² values.
    
    Args:
        values: List of values to plot
        layers: List of layer names
        title: Plot title
        ylabel: Y-axis label
        color: Bar color
        ylim: Y-axis limits
        figsize: Figure size
        add_value_labels: Whether to add value labels on bars
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(len(layers)), values, alpha=0.7, color=color)
    ax.set_title(title)
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace('transformer_layer_', 'Layer ') for l in layers], 
                      rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(ylim)
    
    # Add value labels on bars
    if add_value_labels:
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.02 if height >= 0 else -0.05),
                   f'{val:.3f}', ha='center', 
                   va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_multi_metric_comparison_plot(metrics_dict: Dict[str, List[float]], 
                                      layers: List[str],
                                      title: str, 
                                      figsize: Tuple[int, int] = (12, 6),
                                      colors: List[str] = None) -> plt.Figure:
    """
    Create a comparison plot with multiple metrics on the same axes.
    
    Args:
        metrics_dict: Dictionary mapping metric names to value lists
        layers: List of layer names
        title: Plot title
        figsize: Figure size
        colors: List of colors for each metric
    
    Returns:
        matplotlib Figure object
    """
    if colors is None:
        colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(range(len(layers)), values, f'{marker}-', linewidth=2, 
               label=metric_name, color=color)
    
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation/R² Values')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace('transformer_layer_', 'L') for l in layers], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    return fig


def create_analysis_summary_plot(simple_corrs: List[float], 
                               partial_corrs: List[float],
                               r2_values: List[float],
                               layers: List[str],
                               model_name: str,
                               num_files: int,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create the comprehensive 2x2 input propagation analysis plot.
    
    This consolidates the plotting logic from run_input_propagation_analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Simple correlations (signal retention)
    axes[0, 0].plot(range(len(layers)), simple_corrs, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_title('Input Signal Retention Across Layers')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Correlation with Original Input')
    axes[0, 0].set_xticks(range(len(layers)))
    axes[0, 0].set_xticklabels([l.replace('transformer_layer_', 'L') for l in layers], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-1, 1)
    
    # Add value annotations
    for i, corr in enumerate(simple_corrs):
        axes[0, 0].annotate(f'{corr:.3f}', xy=(i, corr), xytext=(0, 5), 
                           textcoords='offset points', ha='center', fontsize=8)
    
    # Plot 2: Progressive partial correlations (new information)
    axes[0, 1].plot(range(len(layers)), partial_corrs, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_title('New Information per Layer')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Partial Correlation with Input')
    axes[0, 1].set_xticks(range(len(layers)))
    axes[0, 1].set_xticklabels([l.replace('transformer_layer_', 'L') for l in layers], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-1, 1)
    
    # Add value annotations
    for i, corr in enumerate(partial_corrs):
        axes[0, 1].annotate(f'{corr:.3f}', xy=(i, corr), xytext=(0, 5), 
                           textcoords='offset points', ha='center', fontsize=8)
    
    # Plot 3: R² values (variance explained)
    axes[1, 0].plot(range(len(layers)), r2_values, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_title('Input Variance Explained by Each Layer')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('R² (Fraction of Input Variance)')
    axes[1, 0].set_xticks(range(len(layers)))
    axes[1, 0].set_xticklabels([l.replace('transformer_layer_', 'L') for l in layers], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Add value annotations
    for i, r2 in enumerate(r2_values):
        axes[1, 0].annotate(f'{r2:.3f}', xy=(i, r2), xytext=(0, 5), 
                           textcoords='offset points', ha='center', fontsize=8)
    
    # Plot 4: Combined comparison
    axes[1, 1].plot(range(len(layers)), simple_corrs, 'o-', linewidth=2, label='Simple Correlation', color='blue')
    axes[1, 1].plot(range(len(layers)), partial_corrs, 's-', linewidth=2, label='Partial Correlation', color='orange')
    axes[1, 1].plot(range(len(layers)), r2_values, '^-', linewidth=2, label='R² Values', color='green')
    axes[1, 1].set_title('Combined Input Propagation Analysis')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Correlation/R² Values')
    axes[1, 1].set_xticks(range(len(layers)))
    axes[1, 1].set_xticklabels([l.replace('transformer_layer_', 'L') for l in layers], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(-1, 1)
    
    plt.suptitle(f'Input Propagation Analysis - {model_name} (n={num_files} files)', fontsize=14)
    plt.tight_layout()
    return fig 
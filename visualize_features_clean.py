#!/usr/bin/env python3
"""
Clean, modular feature visualization pipeline for HuBERT model analysis.

This script provides a comprehensive analysis pipeline with the following features:
1. Flexible data preprocessing (padding vs segmentation)
2. Multiple similarity metrics and visualizations
3. Conditional analysis (CNN influence)
4. Temporal dynamics analysis with animations
5. Modular, well-organized code structure

Author: [Your name]
Date: [Current date]
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add utils and analysis modules to path
sys.path.append('.')

from utils.data_utils import load_features
from utils.visualization_utils import setup_plot_style, plot_feature_distributions, plot_layer_statistics, plot_padding_ratios
from utils.math_utils import compute_r_squared
from analysis.similarity_analysis import compute_layer_similarities, plot_similarity_matrices, analyze_feature_divergence
from analysis.temporal_analysis import (
    compute_temporal_similarities, create_similarity_animation,
    compute_conditional_temporal_similarities, create_conditional_similarity_animation
)


def analyze_cnn_influence(layer_features, original_lengths, output_dir, model_name, num_files):
    """
    Analyze how much each layer's representation is influenced by the CNN output.
    """
    from utils.data_utils import filter_and_sort_layers
    from utils.visualization_utils import save_figure
    import matplotlib.pyplot as plt
    
    cnn_output_layer = 'transformer_input'
    if cnn_output_layer not in layer_features:
        print(f"Warning: {cnn_output_layer} not found. Skipping CNN influence analysis.")
        return None
    
    cnn_features = layer_features[cnn_output_layer]
    
    # Get all transformer layers
    layers = filter_and_sort_layers(layer_features)
    layers = [l for l in layers if l != cnn_output_layer]  # Exclude CNN layer itself
    
    # Compute R² values showing how much variance in each layer is explained by CNN output
    r2_scores = []
    
    for layer in layers:
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
        
        # Compute R²
        r2 = compute_r_squared(X, Z)
        r2_scores.append(r2)
    
    # Plot R² decay
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(layers)), r2_scores, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('R² (Variance Explained by CNN Output)')
    ax.set_title(f'CNN Influence Decay Across Transformer Layers - {model_name}')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations for R² values
    for i, r2 in enumerate(r2_scores):
        ax.annotate(f'{r2:.3f}', xy=(i, r2), xytext=(0, 5), 
                   textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/cnn_influence_decay_{model_name}_n{num_files}.png')
    
    return r2_scores


def run_basic_analysis(layer_features, original_lengths, args):
    """Run basic feature analysis and visualizations."""
    print("\n=== Running Basic Analysis ===")
    
    # Basic feature statistics
    print("Generating feature distributions...")
    plot_feature_distributions(layer_features, args.output_dir, args.model_name, args.num_files)
    
    print("Generating layer statistics...")
    plot_layer_statistics(layer_features, args.output_dir, args.model_name, args.num_files)
    
    # Padding analysis (only relevant for padding preprocessing)
    if args.preprocessing == 'pad':
        print("Generating padding ratio analysis...")
        plot_padding_ratios(layer_features, original_lengths, args.output_dir, 
                           args.model_name, args.num_files)


def run_similarity_analysis(layer_features, original_lengths, args):
    """Run layer similarity analysis."""
    print("\n=== Running Similarity Analysis ===")
    
    # Basic similarity analysis
    print("Computing layer similarities...")
    similarity_results = compute_layer_similarities(
        layer_features, original_lengths, 
        include_conditional=args.include_conditional
    )
    
    print("Plotting similarity matrices...")
    plot_similarity_matrices(
        similarity_results, args.output_dir, args.model_name, args.num_files,
        include_conditional=args.include_conditional
    )
    
    # Feature divergence analysis
    if args.include_divergence:
        print("Analyzing feature divergence...")
        divergence_results = analyze_feature_divergence(
            layer_features, args.output_dir, args.model_name, args.num_files
        )
    
    return similarity_results


def run_conditional_analysis(layer_features, original_lengths, args):
    """Run conditional analysis (CNN influence)."""
    print("\n=== Running CNN Influence Analysis ===")
    
    # Analyze CNN influence decay
    print("Analyzing CNN influence across layers...")
    r2_scores = analyze_cnn_influence(
        layer_features, original_lengths, args.output_dir, 
        args.model_name, args.num_files
    )
    
    return r2_scores


def run_temporal_analysis(layer_features, original_lengths, args):
    """Run temporal dynamics analysis with animations."""
    print("\n=== Running Temporal Analysis ===")
    
    # Basic temporal similarities
    print("Computing temporal similarities...")
    temporal_similarities = compute_temporal_similarities(
        layer_features, original_lengths,
        window_size=args.window_size, stride=args.stride
    )
    
    # Create animations for each metric
    print("Generating temporal animations...")
    for metric in ['cosine', 'correlation', 'cka']:
        create_similarity_animation(
            temporal_similarities, args.output_dir, args.model_name, metric
        )
    
    # Conditional temporal analysis (if requested)
    if args.include_conditional:
        print("Computing conditional temporal similarities...")
        conditional_temporal_sims = compute_conditional_temporal_similarities(
            layer_features, original_lengths,
            window_size=args.window_size, stride=args.stride
        )
        
        if conditional_temporal_sims is not None:
            print("Generating conditional temporal animations...")
            
            # Partial correlation animations
            create_conditional_similarity_animation(
                conditional_temporal_sims, args.output_dir, args.model_name,
                metric='partial_correlation', comparison_mode='side_by_side'
            )
            create_conditional_similarity_animation(
                conditional_temporal_sims, args.output_dir, args.model_name,
                metric='partial_correlation', comparison_mode='difference'
            )
            
            # Conditional CKA animations
            create_conditional_similarity_animation(
                conditional_temporal_sims, args.output_dir, args.model_name,
                metric='conditional_cka', comparison_mode='side_by_side'
            )
    
    return temporal_similarities


def print_summary(layer_features, r2_scores, args):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print(f"ANALYSIS SUMMARY - {args.model_name}")
    print("="*60)
    
    # Data summary
    print(f"Files processed: {args.num_files}")
    print(f"Preprocessing method: {args.preprocessing}")
    if args.preprocessing == 'segment':
        print(f"Segment length: {args.segment_length}")
        print(f"Segment strategy: {args.segment_strategy}")
    
    # Layer information
    from utils.data_utils import filter_and_sort_layers
    layers = filter_and_sort_layers(layer_features)
    print(f"Layers analyzed: {len(layers)} ({layers[0]} to {layers[-1]})")
    
    # Feature dimensions
    total_params = sum(np.prod(features.shape) for features in layer_features.values())
    print(f"Total feature parameters: {total_params:,}")
    
    # CNN influence summary
    if r2_scores is not None:
        cnn_layers = [l for l in layers if l != 'transformer_input']
        print(f"\nCNN Influence (R² values):")
        for layer, r2 in zip(cnn_layers, r2_scores):
            independence = (1 - r2) * 100
            print(f"  {layer}: {r2:.3f} ({independence:.1f}% independent)")
        
        if len(r2_scores) > 1:
            print(f"\nInfluence decay: {r2_scores[0]:.3f} → {r2_scores[-1]:.3f}")
            print(f"Total decay: {(r2_scores[0] - r2_scores[-1]):.3f}")
    
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("="*60)


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Comprehensive HuBERT feature analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data loading arguments
    parser.add_argument("--features_dir", type=str, required=True,
                       help="Directory containing feature .npz files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save visualizations")
    parser.add_argument("--num_files", type=int, default=3,
                       help="Number of audio files to analyze")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name for titles and filenames")
    
    # Preprocessing arguments
    parser.add_argument("--preprocessing", type=str, default='pad',
                       choices=['pad', 'segment'],
                       help="Preprocessing method: 'pad' (pad to max length) or 'segment' (extract segments)")
    parser.add_argument("--segment_length", type=int, default=None,
                       help="Length of segments (auto-determined if None)")
    parser.add_argument("--segment_strategy", type=str, default='beginning',
                       choices=['beginning', 'middle', 'end', 'random'],
                       help="Strategy for segment extraction")
    
    # Analysis options
    parser.add_argument("--skip_basic", action='store_true',
                       help="Skip basic feature analysis")
    parser.add_argument("--skip_similarity", action='store_true',
                       help="Skip similarity analysis")
    parser.add_argument("--skip_temporal", action='store_true',
                       help="Skip temporal analysis")
    parser.add_argument("--include_conditional", action='store_true',
                       help="Include conditional analysis (CNN influence)")
    parser.add_argument("--include_divergence", action='store_true',
                       help="Include feature divergence analysis")
    
    # Temporal analysis arguments
    parser.add_argument("--window_size", type=int, default=20,
                       help="Window size for temporal analysis")
    parser.add_argument("--stride", type=int, default=10,
                       help="Stride for sliding window")
    
    args = parser.parse_args()
    
    # Validation
    if args.num_files < 1:
        raise ValueError("Number of files must be at least 1")
    
    if args.preprocessing == 'segment' and args.segment_length is not None and args.segment_length <= 0:
        raise ValueError("Segment length must be positive")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    setup_plot_style()
    
    print(f"Starting analysis for {args.model_name}")
    print(f"Processing {args.num_files} files with '{args.preprocessing}' preprocessing")
    
    # Load features
    print("\nLoading features...")
    layer_features, original_lengths = load_features(
        args.features_dir, 
        num_files=args.num_files,
        preprocessing=args.preprocessing,
        segment_length=args.segment_length,
        segment_strategy=args.segment_strategy
    )
    
    # Check minimum requirements
    min_samples = 2
    first_layer = list(layer_features.keys())[0]
    if layer_features[first_layer].shape[0] < min_samples:
        print(f"Warning: Only {layer_features[first_layer].shape[0]} samples available.")
        print("Some analyses may be skipped or may not be meaningful.")
    
    # Run analyses based on arguments
    r2_scores = None
    
    if not args.skip_basic:
        run_basic_analysis(layer_features, original_lengths, args)
    
    if not args.skip_similarity:
        similarity_results = run_similarity_analysis(layer_features, original_lengths, args)
    
    if args.include_conditional:
        r2_scores = run_conditional_analysis(layer_features, original_lengths, args)
    
    if not args.skip_temporal:
        temporal_results = run_temporal_analysis(layer_features, original_lengths, args)
    
    # Print summary
    print_summary(layer_features, r2_scores, args)


if __name__ == "__main__":
    main() 
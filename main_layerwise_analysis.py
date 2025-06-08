#!/usr/bin/env python3
"""
Layerwise Analysis Pipeline - Main Analysis Script

This script provides a comprehensive analysis pipeline for studying layerwise representations
in HuBERT models, with a focus on similarity analysis, temporal dynamics, and feature propagation.

PURPOSE:
    Analyze and visualize layerwise representations from HuBERT models to understand:
    - Layer-to-layer similarity patterns
    - Temporal dynamics of representations
    - Feature propagation through the network
    - CNN influence on transformer layers (this is what we call "conditional analysis" or conditioned on the input to the transformer which is the CNN output)

FUNCTIONALITY:
    1. Data Preprocessing:
       - Flexible padding strategies
       - Segmentation options
       - Length normalization
    
    2. Similarity Analysis:
       - Layer-to-layer similarity computation
       - Multiple metrics (correlation, CKA)
       - GPU-accelerated computation
       - Conditional analysis
    
    3. Temporal Analysis:
       - Sliding window analysis
       - Dynamic similarity tracking
       - Animation generation
       - Time-series visualization
    
    4. Feature Propagation:
       - Input propagation analysis
       - Progressive partial correlations
       - R² analysis for variance explanation
       - CNN influence tracking

USAGE:
    python visualize_features_clean.py \
        --features_dir /path/to/features \
        --output_dir /path/to/output \
        --model_name HuBERT_Base \
        --num_files 3 \
        [additional options]

STATUS:
    ACTIVE - This is the primary analysis pipeline script.
    
    INTEGRATION:
    - Core analysis pipeline for layerwise studies
    - Used in conjunction with feature extraction
    - Outputs feed into visualization tools
    
    DEPENDENCIES:
    - _utils/data_utils.py (feature loading)
    - _utils/math_utils.py (similarity computations)
    - _utils/visualization_utils.py (plotting)
    - _analysis/similarity_analysis.py (layer analysis)
    - _analysis/temporal_analysis.py (temporal dynamics)
    
    TODO:
    - Add support for more similarity metrics (local instrinsic dimensionality, spatial correlation, etc.)
    - Implement batch processing for large datasets
    - Add interactive visualization options
    - Optimize memory usage for large models

Author: Sandra Arcos Holzinger
Date: June 7, 2025
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add utils and analysis modules to path
sys.path.append('.')

from _utils.data_utils import load_features
from _utils.visualization_utils import setup_plot_style, plot_feature_distributions, plot_layer_statistics, plot_padding_ratios

from _analysis.similarity_analysis import (
    compute_layer_similarities, plot_similarity_matrices, analyze_feature_divergence,
    # NEW: Import the enhanced correlation analysis functions
    compute_input_propagation_similarities, plot_input_propagation_correlations,
    plot_enhanced_similarity_matrices
)
from _analysis.temporal_analysis import (
    compute_temporal_similarities, create_similarity_animation,
    compute_conditional_temporal_similarities, create_conditional_similarity_animation
)


def analyze_cnn_influence(layer_features, original_lengths, output_dir, model_name, num_files):
    """
    Analyze how much each layer's representation is influenced by the CNN output.
    """
    try:
        # Use the new unified analyzer
        from _utils.math_utils import CorrelationAnalyzer
        from _utils.visualization_utils import create_correlation_plot, save_figure
        
        # Initialize analyzer  
        analyzer = CorrelationAnalyzer(
            layer_features, 
            original_lengths, 
            cnn_layer='transformer_input', 
            max_layer=11
        )
        
        # Compute R² analysis
        r2_results = analyzer.compute_r_squared_analysis()
        
        # Get layers and values in order
        layers = analyzer.config['transformer_layer_names']
        r2_scores = [r2_results[layer] for layer in layers]
        
        # Create plot using utility function
        fig = create_correlation_plot(
            r2_scores, layers,
            f'CNN Influence Decay Across Transformer Layers - {model_name}',
            'R² (Variance Explained by CNN Output)',
            color='green',
            ylim=(0, 1)
        )
        
        save_figure(fig, f'{output_dir}/cnn_influence_decay_{model_name}_n{num_files}.png')
        
        return r2_scores
        
    except Exception as e:
        print(f"Warning: {e}. Skipping CNN influence analysis.")
        return None


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
    """Enhanced layer similarity analysis with GPU acceleration and new correlation types."""
    print("\n=== Running Enhanced Similarity Analysis ===")
    
    # Check for GPU availability and user preferences
    use_gpu = getattr(args, 'use_gpu', True)
    n_jobs = getattr(args, 'n_jobs', -1)
    
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ Using GPU acceleration: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("GPU not available, using CPU with parallel processing")
                use_gpu = False
        except ImportError:
            print("PyTorch not available, using CPU with parallel processing")
            use_gpu = False
    
    print(f"  CPU jobs: {n_jobs} {'(all cores)' if n_jobs == -1 else ''}")
    
    # Basic layer-to-layer similarity analysis with acceleration
    print("Computing enhanced layer similarities...")
    similarity_results = compute_layer_similarities(
        layer_features, original_lengths, 
        include_conditional=args.include_conditional,
        use_gpu=use_gpu,
        n_jobs=n_jobs
    )
    
    # Input propagation analysis (NEW)
    propagation_results = None
    if args.include_input_propagation:
        print("Computing input propagation correlations...")
        propagation_results = compute_input_propagation_similarities(
            layer_features, original_lengths,
            cnn_layer='transformer_input',
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            show_progress=True
        )
        
        if propagation_results:
            print("Input propagation analysis completed successfully")
            # Create separate detailed plot for propagation
            plot_input_propagation_correlations(
                propagation_results, args.output_dir, args.model_name, args.num_files,
                show_performance_info=True
            )
        else:
            print("Input propagation analysis failed")
    
    # Enhanced plotting with both analyses
    print("Plotting enhanced similarity matrices...")
    plot_enhanced_similarity_matrices(
        similarity_results, propagation_results, 
        args.output_dir, args.model_name, args.num_files,
        include_conditional=args.include_conditional,
        use_gpu_info=True
    )
    
    # Keep backward compatibility with original plots
    if not args.include_input_propagation:
        print("Plotting traditional similarity matrices...")
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
    
    # Return combined results
    results = {'layer_similarities': similarity_results}
    if propagation_results:
        results['input_propagation'] = propagation_results
    
    return results


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
    for metric in ['correlation', 'cka']:
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
            create_conditional_similarity_animation(
                conditional_temporal_sims, args.output_dir, args.model_name,
                metric='conditional_cka', comparison_mode='difference'
            )
    
    return temporal_similarities


def run_input_propagation_analysis(layer_features, original_lengths, args):
    """Run input propagation analysis to track how original input correlates with each layer."""
    print("\n=== Running Input Propagation Analysis ===")
    
    try:
        # Use the new unified analyzer
        from _utils.math_utils import CorrelationAnalyzer
        from _utils.visualization_utils import (
            create_analysis_summary_plot, 
            create_correlation_plot, 
            create_bar_plot, 
            save_figure
        )
        
        # Initialize analyzer
        analyzer = CorrelationAnalyzer(
            layer_features, 
            original_lengths, 
            cnn_layer='transformer_input', 
            max_layer=11
        )
        
        print(f"Analyzing input propagation through {analyzer.config['transformer_count']} transformer layers...")
        
        # Compute all analyses at once
        results = analyzer.compute_all_analyses()
        
        layers = results['layer_order']
        simple_corrs = [results['simple_correlations'][l] for l in layers]
        partial_corrs = [results['progressive_partial_correlations'][l] for l in layers]
        r2_values = [results['r_squared_values'][l] for l in layers]
        
        # Create comprehensive plot using utility function
        fig = create_analysis_summary_plot(
            simple_corrs, partial_corrs, r2_values, 
            layers, args.model_name, args.num_files
        )
        save_figure(fig, f'{args.output_dir}/input_propagation_analysis_{args.model_name}_n{args.num_files}.png')
        
        # Create individual detailed plots using utility functions
        
        # Simple correlations plot
        fig = create_bar_plot(
            simple_corrs, layers,
            f'Input Signal Retention Across Layers - {args.model_name}',
            'Correlation with Original Input',
            color='blue'
        )
        save_figure(fig, f'{args.output_dir}/simple_correlations_{args.model_name}_n{args.num_files}.png')
        
        # Partial correlations plot
        fig = create_bar_plot(
            partial_corrs, layers,
            f'Progressive Partial Correlations - {args.model_name}',
            'Partial Correlation with Input (Controlling for Previous Layers)',
            color='orange'
        )
        save_figure(fig, f'{args.output_dir}/partial_correlations_{args.model_name}_n{args.num_files}.png')
        
        # Print summary to console
        print("\n=== Input Propagation Summary ===")
        print(f"Analysis of {results['cnn_layer']} → transformer layers:")
        print("\nSimple Correlations (signal retention):")
        for layer, corr in zip(layers, simple_corrs):
            layer_num = layer.replace('transformer_layer_', 'L')
            print(f"  {layer_num}: {corr:.3f}")
        
        print(f"\nSignal retention decay: {simple_corrs[0]:.3f} → {simple_corrs[-1]:.3f}")
        
        print("\nPartial Correlations (new information per layer):")
        for layer, corr in zip(layers, partial_corrs):
            layer_num = layer.replace('transformer_layer_', 'L')
            print(f"  {layer_num}: {corr:.3f}")
        
        print(f"\nMean absolute partial correlation: {np.mean(np.abs(partial_corrs)):.3f}")
        
        return results
        
    except Exception as e:
        print(f"Error in input propagation analysis: {e}")
        return None


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
    
    # Layer information using new utilities
    from _utils.data_utils import create_layer_analysis_config
    config = create_layer_analysis_config(layer_features)
    print(f"Layers analyzed: {config['layer_count']} ({config['all_layers'][0]} to {config['all_layers'][-1]})")
    
    # Feature dimensions
    total_params = sum(np.prod(features.shape) for features in layer_features.values())
    print(f"Total feature parameters: {total_params:,}")
    
    # CNN influence summary
    if r2_scores is not None:
        transformer_layers = config['transformer_layer_names']
        print(f"\nCNN Influence (R² values):")
        for layer, r2 in zip(transformer_layers, r2_scores):
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
    parser.add_argument("--include_input_propagation", action='store_true',
                       help="Include input propagation analysis (Simple, Progressive Partial, and R² correlations)")
    
    # Performance and parallelization options
    parser.add_argument("--use_gpu", action='store_true', default=True,
                       help="Use GPU acceleration for computations (default: True)")
    parser.add_argument("--no_gpu", action='store_false', dest='use_gpu',
                       help="Disable GPU acceleration (force CPU-only)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                       help="Number of parallel CPU jobs (-1 for all cores, 1 to disable)")
    
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
    propagation_results = None
    
    if not args.skip_basic:
        print("Running basic analysis... (feature distributions, layer statistics, padding analysis)")
        run_basic_analysis(layer_features, original_lengths, args)
    
    if not args.skip_similarity:
        print("Running similarity analysis... (layer-to-layer similarities, input propagation (CKA, correlation (old), partial correlation (old)), feature divergence)")
        similarity_results = run_similarity_analysis(layer_features, original_lengths, args)
    
    if args.include_conditional:
        print("Running conditional analysis... (CNN influence)")
        r2_scores = run_conditional_analysis(layer_features, original_lengths, args)
    
    if args.include_input_propagation:
        print("Running input propagation analysis... (simple, progressive partial (Test 1, 2, 3), and R² correlations) Excludes CCA")
        propagation_results = run_input_propagation_analysis(layer_features, original_lengths, args)
    
    if not args.skip_temporal:
        print("Running temporal analysis... (temporal similarities, animations) Excludes CCA, Test 1, 2, 3.")
        temporal_results = run_temporal_analysis(layer_features, original_lengths, args)
    
    # Print summary
    print_summary(layer_features, r2_scores, args)


if __name__ == "__main__":
    main() 
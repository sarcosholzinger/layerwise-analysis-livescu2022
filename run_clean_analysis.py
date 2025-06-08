#!/usr/bin/env python3
"""
Example script demonstrating how to use the clean, modular analysis pipeline.

This script shows various usage patterns for different types of analysis.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a shell command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True


def example_basic_analysis():
    """Example: Basic analysis with padding preprocessing."""
    print("\n=== Example 1: Basic Analysis (Padding) ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/basic_padding",
        "--model_name", "HuBERT_Base_Padding",
        "--num_files", "3",
        "--preprocessing", "pad",
        "--skip_temporal"  # Skip temporal analysis for speed
    ]
    
    return run_command(cmd)


def example_segment_analysis():
    """Example: Analysis with segmentation preprocessing."""
    print("\n=== Example 2: Segmentation Analysis ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/segment_middle",
        "--model_name", "HuBERT_Base_Segment",
        "--num_files", "5",
        "--preprocessing", "segment",
        "--segment_length", "100",
        "--segment_strategy", "middle",
        "--skip_temporal"
    ]
    
    return run_command(cmd)


def example_conditional_analysis():
    """Example: Full analysis with conditional (CNN influence) analysis."""
    print("\n=== Example 3: Conditional Analysis ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/conditional",
        "--model_name", "HuBERT_Base_Conditional",
        "--num_files", "3",
        "--preprocessing", "pad",
        "--include_conditional",
        "--include_divergence",
        "--skip_temporal"  # Skip temporal for speed in this example
    ]
    
    return run_command(cmd)


def example_temporal_analysis():
    """Example: Temporal dynamics analysis with animations."""
    print("\n=== Example 4: Temporal Analysis ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/temporal",
        "--model_name", "HuBERT_Base_Temporal",
        "--num_files", "3",
        "--preprocessing", "segment",
        "--segment_length", "150",
        "--skip_basic",
        "--skip_similarity",
        "--window_size", "15",
        "--stride", "5"
    ]
    
    return run_command(cmd)


def example_full_analysis():
    """Example: Complete analysis with all features."""
    print("\n=== Example 5: Full Analysis ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/full",
        "--model_name", "HuBERT_Base_Complete",
        "--num_files", "5",
        "--preprocessing", "pad",
        "--include_conditional",
        "--include_divergence",
        "--include_input_propagation",
        "--window_size", "20",
        "--stride", "10"
    ]
    
    return run_command(cmd)


def example_input_propagation_analysis():
    """Example: Input propagation analysis with GPU acceleration (Simple, Progressive Partial, and R² correlations)."""
    print("\n=== Example 6: Input Propagation Analysis (GPU Accelerated) ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/input_propagation",
        "--model_name", "HuBERT_Input_Propagation",
        "--num_files", "3",
        "--preprocessing", "pad",
        "--skip_basic",
        "--skip_similarity", 
        "--skip_temporal",
        "--include_input_propagation",
        "--use_gpu",
        "--n_jobs", "-1"  # Use all CPU cores for parallel processing
    ]
    
    return run_command(cmd)


def example_all_correlations():
    """Example: All correlation types with full GPU acceleration and performance optimization."""
    print("\n=== Example 7: All Correlation Types (Full GPU + CPU Parallel) ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/all_correlations",
        "--model_name", "HuBERT_All_Correlations",
        "--num_files", "3",
        "--preprocessing", "pad",
        "--skip_basic",
        "--skip_temporal",
        "--include_conditional",
        "--include_input_propagation",
        "--use_gpu",
        "--n_jobs", "-1"
    ]
    
    return run_command(cmd)


def example_performance_benchmark():
    """Example: Performance benchmark comparing GPU vs CPU-only processing."""
    print("\n=== Example 8: Performance Benchmark (GPU vs CPU) ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/performance_benchmark",
        "--model_name", "HuBERT_Performance_Test",
        "--num_files", "5",  # More files for better benchmarking
        "--preprocessing", "pad",
        "--skip_basic",
        "--skip_temporal",
        "--include_conditional",
        "--include_input_propagation",
        "--include_divergence",
        "--use_gpu",
        "--n_jobs", "-1"
    ]
    
    return run_command(cmd)


def example_cpu_only_comparison():
    """Example: CPU-only processing for performance comparison."""
    print("\n=== Example 9: CPU-Only Processing (For Comparison) ===")
    
    cmd = [
        "python", "visualize_features_clean.py",
        "--features_dir", "./output/hubert_complete/librispeech_dev-clean_sample1",
        "--output_dir", "./output/clean_analysis/cpu_only",
        "--model_name", "HuBERT_CPU_Only",
        "--num_files", "3",
        "--preprocessing", "pad",
        "--skip_basic",
        "--skip_temporal",
        "--include_conditional",
        "--include_input_propagation",
        "--no_gpu",  # Force CPU-only
        "--n_jobs", "-1"
    ]
    
    return run_command(cmd)


def main():
    """Run various analysis examples."""
    print("HuBERT Feature Analysis - Clean Pipeline Examples")
    print("=" * 60)
    
    # Create output directories
    Path("./output/clean_analysis").mkdir(parents=True, exist_ok=True)
    
    examples = [
        # ("Basic Analysis", example_basic_analysis),
        # ("Segmentation Analysis", example_segment_analysis),
        # ("Conditional Analysis", example_conditional_analysis),
        ("Temporal Analysis", example_temporal_analysis),
        # ("Full Analysis", example_full_analysis),
        ("Input Propagation Analysis", example_input_propagation_analysis),
        ("All Correlation Types", example_all_correlations),
        ("Performance Benchmark", example_performance_benchmark),
        # ("CPU-Only Comparison", example_cpu_only_comparison)
    ]
    
    success_count = 0
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            if example_func():
                success_count += 1
                print(f" {name} completed successfully")
            else:
                print(f" {name} failed")
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(examples)} examples completed successfully")
    
    if success_count == len(examples):
        print("(^^) All examples completed successfully!")
        print("\nTo view results, check the following directories:")
        print("- ./output/clean_analysis/basic_padding/")
        print("- ./output/clean_analysis/segment_middle/")
        print("- ./output/clean_analysis/conditional/")
        print("- ./output/clean_analysis/temporal/")
        print("- ./output/clean_analysis/full/")
        print("- ./output/clean_analysis/input_propagation/")
        print("- ./output/clean_analysis/all_correlations/")
        print("- ./output/clean_analysis/performance_benchmark/")
        print("- ./output/clean_analysis/cpu_only/")
    else:
        print("(!!) Some examples failed. Check the error messages above.")


if __name__ == "__main__":
    main() 
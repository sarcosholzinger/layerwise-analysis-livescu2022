#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_cca_scores(scores_file, output_dir=None):
    """
    Plot CCA scores across layers.
    
    Args:
        scores_file (str): Path to the JSON file containing CCA scores
        output_dir (str, optional): Directory to save plots. If None, uses the same directory as scores_file
    """
    # Load CCA scores
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    
    # Convert to numpy arrays for easier manipulation
    layers = np.array([int(k) for k in scores.keys()])
    values = np.array([float(v) for v in scores.values()])
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the scores with a nice color
    line = ax.plot(layers, values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    
    # Add labels and title
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('CCA Score', fontsize=12)
    ax.set_title('Layer-wise CCA Analysis', fontsize=14, pad=20)
    
    # Set x-axis ticks to show all layers
    ax.set_xticks(layers)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of each point
    for i, v in enumerate(values):
        ax.text(layers[i], v + 0.02, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plots if output directory is provided
    if output_dir is None:
        output_dir = Path(scores_file).parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the base name of the input file for the output files
    base_name = Path(scores_file).stem
    
    # Save as PNG
    plt.savefig(output_dir / f'{base_name}_plot.png', dpi=300, bbox_inches='tight')
    # Save as PDF
    plt.savefig(output_dir / f'{base_name}_plot.pdf', bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}")
    print(f"- PNG: {output_dir / f'{base_name}_plot.png'}")
    print(f"- PDF: {output_dir / f'{base_name}_plot.pdf'}")
    
    # Show the plot
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Plot CCA scores from a JSON file')
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Path to the JSON file containing CCA scores')
    parser.add_argument('--output', '-o', type=str,
                      help='Directory to save the plots (default: same directory as input file)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Plot the scores
    plot_cca_scores(args.input, args.output)

    ''' usage:
    python3 CCA_analysis/plot_cca_results.py --input cca_results/cca_scores.json --output cca_results/plots
    python3 CCA_analysis/plot_cca_results.py --input cca_results/old_cca_scores_mean.json --output cca_results/plots
    '''
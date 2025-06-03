# Layer-wise Feature Visualization Analysis

This document explains the visualization and analysis of features across different layers of the HuBERT model.

## Overview

The visualization suite provides multiple perspectives on how features evolve across layers:
1. Feature distributions and statistics
2. Layer-wise similarity analysis
3. Dimensionality reduction visualizations (PCA, t-SNE, UMAP)
4. Activation statistics and outlier detection

## Feature Distributions

The `plot_feature_distributions` function visualizes how feature statistics change across layers:

- **Mean and Standard Deviation**: Shows the central tendency and spread of features
- **Layer Progression**: Layers are sorted numerically to show proper progression
- **Interpretation**:
  - Lower layers: More consistent activations (lower std)
  - Higher layers: More diverse activations (higher std)

## Layer Similarity Analysis

The `plot_layer_similarity` function computes cosine similarity between layers:

```python
similarity = np.dot(features1.flatten(), features2.flatten()) / (
    np.linalg.norm(features1.flatten()) * np.linalg.norm(features2.flatten())
)
```

This helps understand:
- How features evolve across layers
- Which layers are most similar/different
- The overall architecture's information flow

## Dimensionality Reduction

### UMAP Analysis

UMAP (Uniform Manifold Approximation and Projection) is used with the following parameters:
```python
reducer = umap.UMAP(
    n_components=2,
    random_state=42,  # For reproducibility
    n_neighbors=15,   # Local structure focus
    min_dist=0.1,     # Cluster separation
    metric='euclidean',
    n_jobs=1,         # Single-threaded for reproducibility
    verbose=False
)
```

Key aspects:
- Preserves both local and global structure
- Shows clustering patterns in high-dimensional features
- Color-coded by hidden unit activations

### Activation Distribution Analysis

The visualization includes:
1. UMAP projection of features
2. Separate activation distribution plots
3. Statistical analysis of activations

## Activation Statistics and Outlier Detection

The `plot_layer_statistics` function provides comprehensive analysis of activations:

### Statistical Measures
- Mean activation values
- Standard deviation
- Percentage of high/low activations

### Outlier Detection
- High activations: > 2 standard deviations above mean
- Low activations: < 2 standard deviations below mean
- Percentage of extreme activations tracked across layers

### Layer Progression Analysis
- Shows how activation patterns evolve
- Tracks percentage of extreme activations
- Helps identify layers with unusual behavior

## Key Findings

1. **Layer-wise Progression**:
   - Lower layers: More uniform activations
   - Higher layers: More diverse and specialized activations
   - Increasing standard deviation indicates growing feature specialization

2. **Activation Patterns**:
   - Isolated high activations in higher layers suggest specialized feature detection
   - Distribution of activations becomes more spread out in deeper layers
   - Some layers show distinct clusters of high/low activations

3. **Architecture Insights**:
   - Feature space becomes more complex in higher layers
   - Clear progression from basic to abstract features
   - Some layers show distinct specialization patterns

## Usage

The visualization suite can be run with:
```bash
python visualize_features.py --features_dir <path> --output_dir <path> --model_name <name> --num_files <n>
```

## Output Files

1. `feature_distributions_{model_name}_n{num_files}.png`
   - Mean and standard deviation across layers
   - Properly sorted layer progression

2. `layer_similarity_{model_name}_n{num_files}.png`
   - Cosine similarity matrix between layers
   - Heatmap visualization

3. `umap_visualization_{model_name}_n{num_files}.png`
   - UMAP projection with activation coloring
   - Separate activation distribution plots

4. `layer_statistics_{model_name}_n{num_files}.png`
   - Activation statistics progression
   - Percentage of extreme activations
   - Layer-wise analysis

## Notes

- UMAP is run in single-threaded mode for reproducibility
- Layer numbers are sorted numerically for proper progression
- Activation thresholds are set at ±2 standard deviations
- All visualizations include proper error handling and data validation 
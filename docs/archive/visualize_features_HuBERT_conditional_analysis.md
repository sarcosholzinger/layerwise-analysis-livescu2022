# HuBERT Conditional Layer Analysis

This document describes the conditional analysis of HuBERT layer representations, focusing on how transformer layers relate to each other after accounting for the influence of the CNN output.

## Overview

The analysis examines layer relationships in two ways:
1. Unconditional analysis: Direct similarity between layers
2. Conditional analysis: Similarity after removing the influence of CNN features

## Key Components

### 1. Partial Correlation
- Removes linear influence of CNN output from both layers before computing correlation
- Formula: r(X,Y|Z) = corr(X - X̂, Y - Ŷ)
- Where X̂ and Ŷ are predictions from CNN features (Z)

### 2. Conditional CKA (Two Methods)
- **Residual Method**: Applies CKA to residuals after regressing out CNN influence
- **Partial Method**: Uses kernel regression to compute partial kernel matrices:
  - K_{X|Z} = K_X - K_{XZ} K_Z^{-1} K_{ZX}

### 3. CNN Influence Analysis
- Computes R² values showing variance explained by CNN output
- Reveals:
  - How quickly transformer "forgets" CNN representation
  - Which layers maintain strong ties to CNN features
  - Decay pattern of CNN influence

### 4. Visualizations
- Side-by-side comparison of unconditional vs conditional similarities
- Difference heatmaps showing effect of conditioning
- Bar charts showing average similarities with/without conditioning
- R² decay plot showing CNN influence across layers

## Insights

### True Layer Relationships
- Removes CNN influence to reveal intrinsic layer similarities
- Distinguishes between inherited and learned relationships

### Information Flow
- R² decay shows how CNN information propagates through transformer
- Reveals transformation patterns across layers

### Critical Transitions
- Layers with significant conditional similarity drops indicate major representational changes
- Helps identify key architectural boundaries

### Redundancy Analysis
- High conditional similarity suggests similar computations beyond CNN inheritance
- Useful for identifying potential layer pruning opportunities

## Usage

```bash
python visualize_features.py \
    --features_dir /path/to/features \
    --output_dir /path/to/output \
    --num_files 3 \
    --model_name "HuBERT Base"
```

## Output Files

1. `conditional_layer_similarity_{model_name}_n{num_files}.png`
   - Main visualization comparing conditional vs unconditional metrics
   - Includes correlation and CKA matrices

2. `conditioning_effect_summary_{model_name}_n{num_files}.png`
   - Summary plots showing impact of conditioning
   - Bar charts of average similarities

3. `cnn_influence_decay_{model_name}_n{num_files}.png`
   - R² decay plot showing CNN influence across layers
   - Helps identify critical transition points

4. Raw data files:
   - `conditional_correlation_matrix.npy`
   - `conditional_cka_residual_matrix.npy`
   - `conditional_cka_partial_matrix.npy`

## Interpretation Guide

1. **High Unconditional, Low Conditional Similarity**
   - Layers share similar CNN-inherited features
   - But perform different transformations

2. **High Both Unconditional and Conditional Similarity**
   - Layers maintain similar representations
   - Even after removing CNN influence

3. **Low Both Unconditional and Conditional Similarity**
   - Layers perform distinct computations
   - Independent of CNN features

4. **R² Decay Patterns**
   - Steep decay: Quick independence from CNN
   - Gradual decay: Maintains CNN influence longer
   - Plateaus: Layers with similar CNN dependence 
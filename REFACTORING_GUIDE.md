# HuBERT Feature Analysis - Code Restructuring Guide

## Overview

This guide documents the restructuring of the original `visualize_features.py` script into a clean, modular, and maintainable codebase. The refactoring addresses several issues:

1. **Monolithic code**: Original 2000+ line script was hard to maintain
2. **Mixed concerns**: Data loading, math, visualization all in one file
3. **Limited preprocessing options**: Only padding was supported
4. **Poor reusability**: Functions were tightly coupled
5. **Difficult testing**: Large functions with multiple responsibilities

## New Structure

```
layerwise-analysis/
├── utils/
│   ├── __init__.py
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── math_utils.py          # Mathematical computations (CKA, correlation, etc.)
│   └── visualization_utils.py  # Basic plotting utilities
├── analysis/
│   ├── __init__.py
│   ├── similarity_analysis.py  # Layer similarity analysis
│   └── temporal_analysis.py    # Temporal dynamics and animations
├── visualize_features_clean.py # Clean main pipeline
├── run_clean_analysis.py      # Usage examples
└── visualize_features.py      # Original (for reference)
```

## Key Improvements

### 1. **Modular Architecture**
- **Separation of concerns**: Each module has a single responsibility
- **Reusable components**: Functions can be imported and used independently
- **Easy testing**: Smaller, focused functions are easier to unit test
- **Clear dependencies**: Import structure shows relationships between modules

### 2. **Enhanced Preprocessing Options**
```python
# Original: Only padding
load_features(features_dir, num_files, max_length)

# New: Flexible preprocessing
load_features(
    features_dir, 
    num_files=3,
    preprocessing='pad',           # or 'segment'
    segment_length=100,           # for segmentation
    segment_strategy='middle'     # 'beginning', 'middle', 'end', 'random'
)
```

### 3. **Configurable Analysis Pipeline**
```bash
# Run only basic analysis
python visualize_features_clean.py --features_dir DATA --output_dir OUT --model_name MODEL --skip_temporal --skip_similarity

# Run conditional analysis
python visualize_features_clean.py --features_dir DATA --output_dir OUT --model_name MODEL --include_conditional

# Run with segmentation
python visualize_features_clean.py --features_dir DATA --output_dir OUT --model_name MODEL --preprocessing segment --segment_length 150

# See ALL examples
python run_clean_analysis.py
```

## Function Migration Guide

Here's how functions from the original script were reorganized:

### Data Loading & Preprocessing
| Original Function | New Location | Changes |
|------------------|--------------|---------|
| `pad_features()` | `utils/data_utils.py` | Enhanced with better type hints |
| `load_features()` | `utils/data_utils.py` | Added segmentation option |
| `get_layer_number()` | `utils/data_utils.py` | Moved to utils |
| NEW: `segment_features()` | `utils/data_utils.py` | New function for segmentation |
| NEW: `filter_and_sort_layers()` | `utils/data_utils.py` | Extracted common functionality |

### Mathematical Computations
| Original Function | New Location | Changes |
|------------------|--------------|---------|
| `compute_cka()` | `utils/math_utils.py` | Improved documentation |
| `compute_cka_without_padding()` | `utils/math_utils.py` | Enhanced error handling |
| `compute_partial_correlation()` | `utils/math_utils.py` | Extracted from larger function |
| `compute_conditional_cka()` | `utils/math_utils.py` | Standalone function |
| NEW: `compute_cosine_similarity()` | `utils/math_utils.py` | Extracted utility |
| NEW: `compute_r_squared()` | `utils/math_utils.py` | CNN influence computation |

### Visualization
| Original Function | New Location | Changes |
|------------------|--------------|---------|
| `plot_feature_distributions()` | `utils/visualization_utils.py` | Simplified |
| `plot_layer_statistics()` | `utils/visualization_utils.py` | Better structure |
| `plot_padding_ratios()` | `utils/visualization_utils.py` | Enhanced |
| NEW: `create_similarity_heatmap()` | `utils/visualization_utils.py` | Reusable heatmap |
| NEW: `save_figure()` | `utils/visualization_utils.py` | Consistent saving |

### Analysis Orchestrators
| Original Function | New Location | Changes |
|------------------|--------------|---------|
| `plot_layer_similarity_improved()` | `analysis/similarity_analysis.py` | Split into compute + plot |
| `analyze_feature_divergence()` | `analysis/similarity_analysis.py` | Enhanced |
| `compute_temporal_similarities()` | `analysis/temporal_analysis.py` | Improved structure |
| `create_similarity_animation()` | `analysis/temporal_analysis.py` | Better organization |
| NEW: `compute_layer_similarities()` | `analysis/similarity_analysis.py` | Core similarity computation |
| NEW: `plot_similarity_matrices()` | `analysis/similarity_analysis.py` | Flexible plotting |

### Functions Removed or Consolidated
| Original Function | Status | Reason |
|------------------|--------|--------|
| `plot_dimensionality_reduction()` | **REMOVED** | Complex, rarely used, can be added back if needed |
| `plot_cca_analysis()` | **REMOVED** | Specialized analysis, can be added back if needed |
| Large parts of `main()` | **RESTRUCTURED** | Split into focused functions |

## Usage Examples

### Basic Usage
```python
from utils.data_utils import load_features
from analysis.similarity_analysis import compute_layer_similarities, plot_similarity_matrices

# Load data
layer_features, original_lengths = load_features(
    "path/to/features", 
    num_files=5,
    preprocessing="segment",
    segment_length=100
)

# Analyze similarities
similarity_results = compute_layer_similarities(layer_features, original_lengths)
plot_similarity_matrices(similarity_results, "output_dir", "model_name", 5)
```

### Advanced Usage
```python
from analysis.temporal_analysis import compute_temporal_similarities, create_similarity_animation

# Temporal analysis
temporal_similarities = compute_temporal_similarities(
    layer_features, original_lengths,
    window_size=20, stride=10
)

# Create animations
for metric in ['cosine', 'correlation', 'cka']:
    create_similarity_animation(temporal_similarities, "output_dir", "model_name", metric)
```

### Command Line Usage
```bash
# Basic analysis with padding
python visualize_features_clean.py \
  --features_dir /path/to/features \
  --output_dir ./results \
  --model_name "HuBERT_Base" \
  --num_files 5 \
  --preprocessing pad

# Advanced analysis with segmentation and conditional analysis
python visualize_features_clean.py \
  --features_dir /path/to/features \
  --output_dir ./results \
  --model_name "HuBERT_Base" \
  --num_files 10 \
  --preprocessing segment \
  --segment_length 150 \
  --segment_strategy middle \
  --include_conditional \
  --include_divergence \
  --window_size 25 \
  --stride 10
```

### What Gets Saved in `output_dir`?

```python
our_output_dir/
├── feature_distributions_HuBERT_Base_n5.png
├── layer_statistics_HuBERT_Base_n5.png
├── layer_similarity_HuBERT_Base_n5.png
├── cnn_influence_decay_HuBERT_Base_n5.png
├── cosine_matrix.npy
├── correlation_matrix.npy
├── cka_matrix.npy
├── layer_similarity_cosine_HuBERT_Base_animation.gif
├── layer_similarity_correlation_HuBERT_Base_animation.gif
└── layer_similarity_cka_HuBERT_Base_animation.gif
```


## Key Benefits of New Structure

### 1. **Maintainability**
- **Single Responsibility**: Each function has one clear purpose
- **Easy to modify**: Changes to one aspect don't affect others
- **Clear documentation**: Each module is well-documented

### 2. **Flexibility**
- **Preprocessing options**: Choose between padding and segmentation
- **Configurable analysis**: Skip or include different analysis types
- **Parameter control**: Fine-tune analysis parameters

### 3. **Reusability**
- **Import specific functions**: Use only what you need
- **Build custom pipelines**: Combine functions in new ways
- **Extension friendly**: Easy to add new analysis methods

### 4. **Performance**
- **Skip unnecessary analyses**: Only run what you need
- **Better memory management**: Process data in focused chunks
- **Parallel processing ready**: Functions are designed for easy parallelization

## Migration Strategy

If you have existing code using the old `visualize_features.py`:

### 1. **Immediate Migration**
- Use `visualize_features_clean.py` with equivalent command-line arguments
- Check `run_clean_analysis.py` for examples

### 2. **Gradual Migration**
- Keep old script for reference
- Migrate one analysis type at a time
- Test new functions against old results

### 3. **Custom Analysis**
- Import specific functions from new modules
- Build custom analysis pipelines
- Extend with new functionality

## Future Enhancements

The new structure makes it easy to add:

1. **New preprocessing methods**: Add to `utils/data_utils.py`
2. **New similarity metrics**: Add to `utils/math_utils.py`
3. **New visualizations**: Add to `utils/visualization_utils.py`
4. **New analysis types**: Create new modules in `analysis/`
5. **Configuration files**: YAML/JSON configs for complex analyses
6. **Parallel processing**: Multi-GPU or distributed analysis
7. **Interactive visualization**: Jupyter widgets or web interfaces

## Best Practices

### 1. **Function Design**
- Keep functions small and focused
- Use type hints for all parameters
- Include comprehensive docstrings
- Handle edge cases gracefully

### 2. **Error Handling**
- Validate inputs early
- Provide informative error messages
- Use appropriate exception types
- Log important events

### 3. **Testing**
- Write unit tests for each function
- Test edge cases and error conditions
- Use fixtures for common test data
- Maintain high test coverage

### 4. **Documentation**
- Document all functions and modules
- Provide usage examples
- Keep documentation up to date
- Use consistent style

This restructuring provides a solid foundation for future development while maintaining all the functionality of the original script with improved usability and maintainability. 
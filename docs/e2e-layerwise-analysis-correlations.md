# Layerwise Analysis - Mathematics and Applications

The current code implementation computes partial correlations between different transformer layers. This is achieved using various approaches as an exploratory exercise. 

## Three Different Analyses:

1. Original Implementation 
Uses linear regression for partial correlation.
Removes CNN influence from both layers.

```python
compute_partial_correlation(layer_i, layer_j, CNN_output)
```
Purpose: Measures correlation between different layers after removing CNN influence
Use case: Understanding layer-to-layer relationships independent of input

2. Simple Input-Layer Correlations (new function)
Direct correlation between original input (Z) and each layer
Shows signal retention across layers. It aims to analyze how the original input correlates with each layer as information propagates through the network.

```python
compute_input_layer_correlations(original_input, layer_features_dict)
```
Purpose: Direct correlation between original input and each layer
Use case: How much of the original signal remains in each layer

3. Progressive Partial Correlations (new function)
Partial correlation between input and each layer, controlling for previous layers
Shows NEW information each layer captures.

```python
compute_progressive_partial_correlations(original_input, layer_features_dict)
```
Purpose: Correlation between input and each layer, controlling for previous layers
Use case: How much NEW information each layer captures beyond previous layers
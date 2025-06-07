# Layerwise Analysis - Mathematics and Applications

The current code implementation computes partial correlations between different transformer layers. This is achieved using various approaches as an exploratory exercise. 

## Three Different Analyses:

1. Original Implementation (Layer-to-Layer Partial Correlations)
Uses linear regression for partial correlation.
Removes CNN influence from both layers.

```python
compute_partial_correlation(layer_i, layer_j, CNN_output)
```
Purpose: Measures correlation between different layers after removing CNN influence.
Use case: Understanding layer-to-layer relationships independent of input signal.
Mathematical approach: Uses linear regression to "partial out" the CNN influence from both layers, then correlates the residuals.

2. Simple Input-Layer Correlations (new function)
Direct correlation between original input (Z) and each transformer layer.
Shows signal retention across layers. It aims to analyze how the original input correlates. with each layer as information propagates through the network.

Expected Prediction:
    1. Lower correlations between distant layers after removing CNN influence
    2. Higher correlations between adjacent layers
    3. Shows intrinsic layer relationships independent of input signal

```python
compute_input_layer_correlations(original_input, layer_features_dict)
```
Purpose: Direct correlation between original input and each layer. Track how much of the original signal is retained as it propagates through layers.
Use case: How much of the original signal remains in each layer.
Mathematical approach: Simple Pearson correlation between flattened input and layer features.

Expected Prediction:
    1. Decreasing correlation as layers get deeper (signal decay)
    2. Highest correlation at early layers (closer to input)
    3. Exponential decay pattern showing signal retention loss
    4. Shows how much original signal remains in each layer

3. Progressive Partial Correlations (new function)
Partial correlation between input and each layer, controlling for previous layers
Shows NEW information each layer captures.

```python
compute_progressive_partial_correlations(original_input, layer_features_dict)
```
Purpose: Correlation between input and each layer, controlling for previous layers. 
Shows what NEW information each layer captures beyond what previous layers already explain.
Use case: How much NEW information each layer captures beyond previous layers.
Mathematical approach:
    Layer 0: Simple correlation with input
    Layer 1: Partial correlation with input, controlling for Layer 0
    Layer 2: Partial correlation with input, controlling for Layers 0 & 1
    And so on...

Expected Prediction:
    1. Decreasing values as layers get deeper (diminishing new information)
    2. Positive values indicating each layer captures some unique information
    3. Near-zero values at deeper layers (little new information beyond previous layers)
    4. Shows incremental information gain per layer

4. Additionally, compute the $R^2$ values:
```python
compute_r_squared(X, Z)  # How much variance in X is explained by Z
```
Decreasing $R^2$ as layers get deeper
Shows variance explanation of input by each layer

Expected Overall Patterns: (To be confirmed!!)
Simple Correlations: 0.8 → 0.6 → 0.4 → 0.2 (steady decay)
Progressive Partial: 0.8 → 0.3 → 0.1 → 0.05 (rapid decay of new info)
$R^2$ Values: 0.6 → 0.4 → 0.2 → 0.1 (decreasing variance explained)
These predictions align with the theoretical expectation that:
    Early layers retain more input signal
    Later layers transform information more abstractly
    Each layer adds progressively less new information about the original input
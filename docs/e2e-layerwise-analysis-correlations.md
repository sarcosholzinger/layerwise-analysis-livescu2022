# Layerwise Analysis - Mathematics and Applications

The current code implementation computes partial correlations between different transformer layers. This is achieved using various approaches as an exploratory exercise. 

## Three Different Analyses:

1. Original Implementation (Layer-to-Layer Partial Correlations)
Uses linear regression for partial correlation.
Removes CNN influence from both layers.

```python
compute_partial_correlation(layer_i, layer_j, CNN_output)
```
Purpose: Measures correlation between different layers after removing CNN influence. \
Use case: Understanding layer-to-layer relationships independent of input signal. \
Mathematical approach: Uses linear regression to "partial out" the CNN influence from both layers, then correlates the residuals.

2. Simple Input-Layer Correlations (new function)
Direct correlation between original input (Z) and each transformer layer. \
Shows signal retention across layers. It aims to analyze how the original input correlates. with each layer as information propagates through the network. 

Expected Prediction: \
    1. Lower correlations between distant layers after removing CNN influence \
    2. Higher correlations between adjacent layers \
    3. Shows intrinsic layer relationships independent of input signal 

```python
compute_input_layer_correlations(original_input, layer_features_dict)
```
Purpose: Direct correlation between original input and each layer. Track how much of the original signal is retained as it propagates through layers. \
Use case: How much of the original signal remains in each layer. \
Mathematical approach: Simple Pearson correlation between flattened input and layer features. 

Expected Prediction: \
    1. Decreasing correlation as layers get deeper (signal decay) \
    2. Highest correlation at early layers (closer to input) \
    3. Exponential decay pattern showing signal retention loss \
    4. Shows how much original signal remains in each layer 

3. Progressive Partial Correlations (new function) 
Partial correlation between input and each layer, controlling for previous layers. \
Shows NEW information each layer captures. 

```python
compute_progressive_partial_correlations(original_input, layer_features_dict)
```
Purpose: Correlation between input and each layer, controlling for previous layers. \
Shows what NEW information each layer captures beyond what previous layers already explain. \
Use case: How much NEW information each layer captures beyond previous layers. \
Mathematical approach:\
    Layer 0: Simple correlation with input \
    Layer 1: Partial correlation with input, controlling for Layer 0 \
    Layer 2: Partial correlation with input, controlling for Layers 0 & 1 \
    And so on... 

Expected Prediction:\
    1. Decreasing values as layers get deeper (diminishing new information) \
    2. Positive values indicating each layer captures some unique information \
    3. Near-zero values at deeper layers (little new information beyond previous layers) \
    4. Shows incremental information gain per layer 

----
Additionally, compute the $R^2$ values:
```python
compute_r_squared(X, Z)  # How much variance in X is explained by Z
```
Decreasing $R^2$ as layers get deeper \
Shows variance explanation of input by each layer 

Expected Overall Patterns: (To be confirmed!!) \
Simple Correlations: 0.8 → 0.6 → 0.4 → 0.2 (steady decay) \
Progressive Partial: 0.8 → 0.3 → 0.1 → 0.05 (rapid decay of new info) \
$R^2$ Values: 0.6 → 0.4 → 0.2 → 0.1 (decreasing variance explained) \
These predictions align with the theoretical expectation that: \
    * Early layers retain more input signal \
    * Later layers transform information more abstractly \
    * Each layer adds progressively less new information about the original input 

## Temporal Analysis - Implementation (Needs review!!)

The temporal analysis extends the static layer similarity analysis to examine how similarities evolve over time through sliding window approaches. This provides insights into the temporal dynamics of information processing across transformer layers.

### 1. Regular Temporal Analysis

Computes layer similarities for sliding windows across time sequences.

```python
compute_temporal_similarities(layer_features, original_lengths, window_size=10, stride=5)
```

**Purpose:** Track how layer similarities change across different temporal windows. \
**Use case:** Understanding temporal dynamics of layer relationships. \
**Mathematical approach:** 
- Extract features for each time window (start_time:start_time+window_size)
- Compute similarity metrics (correlation, CKA) for each window
- Return temporal evolution of similarity matrices

**Key Features:**
- **Padding-aware computation:** Uses `original_lengths` to exclude padded timesteps
- **Window-specific length adjustment:** Dynamically calculates valid timesteps per window
- **Temporal windowing logic:** 
  ```python
  valid_len = max(0, min(orig_len - start_time, window_size))
  ```

**Expected Patterns:**
1. Similarity may vary across different temporal positions
2. Early timesteps might show different layer relationships than later ones
3. Temporal evolution reveals information flow dynamics

### 2. Conditional Temporal Analysis

Extends temporal analysis to include conditional similarities controlling for CNN output.

```python
compute_conditional_temporal_similarities(layer_features, original_lengths, 
                                        cnn_layer='transformer_input', 
                                        window_size=10, stride=5)
```

**Purpose:** Examine temporal layer relationships independent of CNN influence. \
**Use case:** Understanding intrinsic temporal dynamics after removing input signal effects. \
**Mathematical approach:**
- For each temporal window, compute both unconditional and conditional metrics
- Unconditional: Direct correlation and CKA between layers
- Conditional: Partial correlation and conditional CKA controlling for CNN output

**Computed Metrics per Window:**
1. **Unconditional correlation:** `corr(layer_i, layer_j)`
2. **Unconditional CKA:** `CKA(layer_i, layer_j)`
3. **Partial correlation:** `partial_corr(layer_i, layer_j | CNN_output)`
4. **Conditional CKA:** `conditional_CKA(layer_i, layer_j | CNN_output)` *(currently commented out)*

**Expected Insights:**
- How much of temporal similarity is due to shared input vs. intrinsic processing
- Temporal evolution of conditional vs. unconditional relationships
- Windows where layers operate more/less independently of input

### 3. Time Averaging Options: `time_average=True/False`

The temporal analysis implementation supports two different approaches for handling time dimension:

#### `time_average=True` (Sequence-Level Analysis)
**Research Question:** *"How similar are these transformer layers as sequence processors?"*

```python
_extract_valid_features(features1, features2, orig_lens1, orig_lens2, time_average=True)
```

**Behavior:**
- Extract valid timesteps for each sequence
- Average over time dimension per sequence: `features[b, :valid_len, :].mean(axis=0)`
- Result: One averaged vector per sequence `(batch_size, feature_dim)`

**Use Cases:**
- Layer-wise similarity analysis (current primary use case)
- Understanding sequence-level processing differences
- Comparing how layers transform entire sequences

**Interpretation:** "Layer X and Y process sequences similarly"

#### `time_average=False` (Timestep-Level Analysis)
**Research Question:** *"How do individual timesteps relate between layers?"*

```python
_extract_valid_features(features1, features2, orig_lens1, orig_lens2, time_average=False)
```

**Behavior:**
- Extract valid timesteps from all sequences
- Concatenate all timesteps: `np.vstack([valid_timesteps_batch_0, valid_timesteps_batch_1, ...])`
- Result: All individual timesteps `(total_valid_timesteps, feature_dim)`

**Use Cases:**
- Fine-grained temporal dynamics analysis
- Position-specific processing patterns
- Understanding timestep-level similarities

**Interpretation:** "Layer X and Y have similar timestep-level dynamics"

### 4. Implementation Details

#### Padding-Aware Windowing
```python
# Create temporary lengths for this window
temp_lens1 = []
temp_lens2 = []

for orig_len1, orig_len2 in zip(original_lengths[layer1], original_lengths[layer2]):
    # Adjust lengths for this window
    valid_len1 = max(0, min(orig_len1 - start_time, window_size))
    valid_len2 = max(0, min(orig_len2 - start_time, window_size))
    temp_lens1.append(valid_len1)
    temp_lens2.append(valid_len2)

if any(l > 0 for l in temp_lens1) and any(l > 0 for l in temp_lens2):
    cka_matrix[i, j] = compute_cka_without_padding(
        features1, features2, temp_lens1, temp_lens2
    )
```

**Why This Matters:**
- Excludes padded zeros from similarity calculations
- Prevents artificial similarity inflation/deflation
- Ensures accurate temporal boundary handling
- Maintains sequence integrity within windows

#### Dimension Handling for Temporal Windowing

**Critical Implementation Note:** When running windows across sequences, correct dimension indexing is essential.

**Feature Shape Convention:**
```python
features.shape = (batch_size, time_steps, feature_dim)
#                     ↑          ↑           ↑
#                  shape[0]   shape[1]    shape[2]
```

**COMMON MISTAKE: Using batch dimension for temporal operations**
```python
# WRONG: This gives batch_size, not sequence length!
window_size = features1.shape[0]  # Returns number of sequences in batch
max_time = features1.shape[0]     # Incorrect for temporal operations
```

**CORRECT: Using time dimension for temporal operations**
```python
# CORRECT: Use time dimension (shape[1]) for temporal calculations
max_time_steps = features1.shape[1]  # Returns sequence length
n_windows = (max_time_steps - window_size) // stride + 1

# CORRECT: Window extraction from time dimension (shape[1])
for window_idx in range(n_windows):
    start_time = window_idx * stride
    end_time = start_time + window_size
    
    # Extract window from time dimension (shape[1])
    window_features1 = features1[:, start_time:end_time, :]  # (batch, window_size, features)
    window_features2 = features2[:, start_time:end_time, :]
```

**Understanding Batches vs. Sequences:**

**Key Terminology:**
- **Sequence:** One individual sample (e.g., one audio file, one sentence)
- **Batch:** Collection of multiple sequences processed together for efficiency
- **Timestep:** Individual time points within each sequence

```python
# Example: 8 different audio files in one batch
audio_file_1: 30 timesteps → padded to 50
audio_file_2: 45 timesteps → padded to 50  
audio_file_3: 25 timesteps → padded to 50
...
audio_file_8: 40 timesteps → padded to 50

# Result: One batch containing 8 sequences
features.shape = (8, 50, 768)  # batch_size=8, max_time=50, features=768
```

**Batch and Sequences within a window:**
- **Across batches:** Process multiple sequences simultaneously within each window
- **Across time:** Slide window along the time dimension (shape[1]) for temporal analysis

**Temporal Windowing Process:**
- **Window sliding:** Move along time dimension (shape[1]) for temporal analysis
- **Batch processing:** Process all sequences in batch simultaneously within each window
- **Window size:** Should be a parameter (e.g., 10, 20), NOT derived from shape[0] or shape[1]

**Typical Windowing Example:**
```python
# Real data dimensions
features.shape = (8, 50, 768)  # 8 sequences, 50 timesteps each, 768 features

# CORRECT temporal windowing
window_size = 10    # Parameter: analyze 10 timesteps at a time
stride = 5          # Parameter: slide window by 5 timesteps
max_time_steps = features.shape[1]  # = 50 timesteps
n_windows = (50 - 10) // 5 + 1     # = 9 windows

# Each window processes all 8 sequences simultaneously:
# Window 0: timesteps 0-9   for all 8 sequences in the batch
# Window 1: timesteps 5-14  for all 8 sequences in the batch
# Window 2: timesteps 10-19 for all 8 sequences in the batch
# ...
```

**Impact on Similarity Calculations:**
- **Per window:** Compute similarities using all sequences within that time window
- **Temporal evolution:** Track how layer similarities change across windows
- **Padding awareness:** Each sequence may have different valid lengths within each window

#### Performance Considerations
- **`time_average=True`:** Faster computation, smaller matrices
- **`time_average=False`:** More data points, larger computational cost
- **Window size:** Larger windows = more stable estimates, less temporal resolution
- **Stride:** Smaller strides = more temporal detail, higher computational cost

### 5. Visualization and Animation

The temporal analysis includes animation capabilities:

```python
create_similarity_animation(temporal_similarities, output_dir, model_name, metric='correlation')
create_conditional_similarity_animation(temporal_similarities, output_dir, model_name, 
                                       metric='partial_correlation', comparison_mode='side_by_side')
```

**Outputs:**
- Animated heatmaps showing temporal evolution
- Side-by-side unconditional vs conditional comparisons
- Difference animations highlighting conditional effects

**Expected Temporal Patterns:**
1. **Stable similarities:** Consistent layer relationships across time
2. **Temporal drift:** Gradual changes in similarities over sequence length
3. **Position effects:** Different similarities at sequence beginning vs. end
4. **Conditional independence:** Varying CNN influence across temporal windows

## Current Implementation Status - Similarity Metrics

This section documents the current status of all similarity metric functions in the codebase, indicating which are active and which have been commented out during refactoring.

### **ACTIVE METRICS (Currently Implemented)**

#### **Core Similarity Metrics**

1. **`compute_partial_correlation_gpu()`** - GPU partial correlation
   - Location: `_utils/math_utils.py`, `gpu_layer_analysis.py`
   - Purpose: GPU-accelerated partial correlation (Test 0 - GPU based)

2. **`compute_partial_correlation()`** - CPU partial correlation
   - Location: `_utils/math_utils.py`
   - Purpose: Partial correlation controlling for third variable (Test 0 - CPU based)

3. **`compute_input_layer_correlations()`** - Simple input-layer correlations
   - Location: `_utils/math_utils.py`
   - Purpose: Direct correlation between input and each layer (Test 1 - Multi-GPU/CPU)


#### **Input Propagation Analysis**

4. **`compute_progressive_partial_correlations()`** - Progressive partial correlations
   - Location: `_utils/math_utils.py`
   - Purpose: Shows NEW information each layer captures (Test 2 - Multi-GPU)

5. **`compute_r_squared()`** - R² variance explanation
    - Location: `_utils/math_utils.py`
    - Purpose: Variance in X explained by Z

##### CKA
6. **`compute_cka()`** - Regular CKA computation (CPU)
   - Location: `_utils/math_utils.py`
   - Purpose: Centered Kernel Alignment between two representations
   - Implementation details:
     - Pure NumPy implementation
     - Single-threaded CPU computation
     - Straightforward implementation with no memory optimizations
     - Steps:
       1. Centers the input matrices (X, Y)
       2. Computes Gram matrices using matrix multiplication
       3. Centers the Gram matrices using centering matrix H
       4. Computes HSIC and normalization terms
       5. Returns final CKA score

7. **`compute_cka_gpu()`** - GPU-accelerated CKA
   - Location: `_utils/math_utils.py`
   - Purpose: GPU-optimized CKA computation
   - Implementation details:
     - PyTorch-based GPU implementation
     - Falls back to CPU if GPU unavailable
     - Memory-efficient option with chunked computation
     - Handles large matrices through adaptive chunking
     - Cleans up GPU memory after computation

8. **`compute_cka_without_padding()`** - Padding-aware CKA
   - Location: `_utils/math_utils.py`
   - Purpose: CKA computation excluding padded timesteps
   - Implementation details:
     - Handles batched sequence data
     - Respects original sequence lengths
     - Filters out padded timesteps before computation
     - Ensures valid sample alignment between sequences

9. **`compute_cka_gpu_optimized()`** - Memory-optimized GPU CKA
   - Location: `gpu_layer_analysis.py`
   - Purpose: Memory-efficient GPU CKA with chunking
   - Implementation details:
     - Part of GPUParallelSimilarity class
     - Adaptive chunk size based on input dimensions
     - Handles 3D input tensors (batch, time, features)
     - Includes CPU fallback for error cases
     - Optimized for V100 GPUs
     - Memory-efficient computation for large matrices








   

##### Other
10. **`compute_correlation_gpu()`** - GPU correlation
   - Location: `gpu_layer_analysis.py`
   - Purpose: Standard correlation with GPU acceleration (TODO: can probably remove from current analysis)


#### **Temporal Analysis Functions**
11. **`compute_temporal_similarities()`** - Regular temporal analysis
    - Location: `_analysis/temporal_analysis.py`, `visualize_features.py`
    - Purpose: Layer similarities across sliding windows

12. **`compute_conditional_temporal_similarities()`** - Conditional temporal analysis
    - Location: `_analysis/temporal_analysis.py`, `visualize_features.py`
    - Purpose: Temporal analysis controlling for CNN output

#### **High-Level Analysis Functions**
13. **`compute_layer_similarities()`** - Main similarity analysis
    - Location: `_analysis/similarity_analysis.py`
    - Purpose: Orchestrates multiple similarity computations

14. **`compute_input_propagation_similarities()`** - Input propagation wrapper
    - Location: `_analysis/similarity_analysis.py`
    - Purpose: Wrapper for all three input propagation analyses

15. **`compute_all_similarities()`** - GPU parallel computation
    - Location: `gpu_layer_analysis.py`
    - Purpose: Parallel GPU-accelerated similarity computation

### **!!COMMENTED OUT METRICS (Currently Disabled)**

#### **Cosine Similarity (Refactoring)**
1. **~~`compute_cosine_similarity()`~~** 
   - Location: `_utils/math_utils.py` (lines 664-667)
   - Reason: Not significant for our analysis

2. **~~`compute_cosine_similarity_gpu()`~~**
   - Location: `_utils/math_utils.py` (lines 668-736), `gpu_layer_analysis.py`
   - Reason: Not significant for our analysis

#### **Conditional CKA (Regression Issues)**
3. **~~`compute_conditional_cka()`~~**
   - Location: `_utils/math_utils.py` (lines 608-662), `visualize_features.py` (line 1046)
   - Reason: Regression step issues in base implementation

4. **~~`compute_conditional_cka_gpu()`~~**
   - Location: `gpu_layer_analysis.py` (lines 335-344)
   - Reason: Regression step issues in base implementation

#### **Layer-to-Layer Correlations**
5. **~~`compute_layer_to_layer_correlations()`~~**
   - Location: `_utils/math_utils.py` (lines 1021-1066)
   - Reason: Part of refactoring effort

### ⚠️ **INCONSISTENCY ISSUES**

#### **Default Metrics Lists vs. Worker Function Implementation**

**Problem:** Default metrics lists include functions that are commented out in worker functions.

**Example in `gpu_layer_analysis.py`:**
```python
# Default metrics list includes disabled functions:
metrics = ['correlation', 'cka', 'partial_correlation', 'conditional_cka']

# But worker functions have these commented out:
# if 'partial_correlation' in metrics:  # COMMENTED OUT!
# if 'conditional_cka' in metrics:      # COMMENTED OUT!
```

**Files Affected:**
- `gpu_layer_analysis.py`: Lines 452, 1084
- `gpu_temporal_layer_analysis.py`: Lines 51, 490, 672

**Status After Fix:**
- **Working metrics:** `['correlation', 'cka']`
- **Listed metrics:** `['correlation', 'cka']` ✅ **NOW CONSISTENT**

#### **Functional vs. Listed Metrics by File**

| File | Listed Metrics | Actually Working | Issue |
|------|----------------|------------------|-------|
| `gpu_layer_analysis.py` | `['correlation', 'cka']` | `['correlation', 'cka']` | ✅ **FIXED** |
| `gpu_temporal_layer_analysis.py` | `['correlation', 'cka']` | `['correlation', 'cka']` | ✅ Consistent |
| `_analysis/similarity_analysis.py` | Dynamic based on function | `['correlation', 'cka']` + conditionals | ✅ Working |

# Multi-GPU Support

The `compute_input_layer_correlations` function now supports multi-GPU acceleration for faster computation of input-layer correlations across large numbers of layers.

## Usage

```python
from _utils.math_utils import compute_input_layer_correlations, compute_progressive_partial_correlations

# INPUT-LAYER CORRELATIONS
# Use all available GPUs (default)
correlations = compute_input_layer_correlations(
    input_features, 
    layer_features_dict, 
    use_gpu=True
)

# Use specific number of GPUs
correlations = compute_input_layer_correlations(
    input_features, 
    layer_features_dict, 
    use_gpu=True,
    n_gpus=2  # Use only 2 GPUs
)

# PROGRESSIVE PARTIAL CORRELATIONS
# Multi-GPU accelerated internal computations
partial_correlations = compute_progressive_partial_correlations(
    input_features,
    layer_features_dict,
    use_gpu=True,
    n_gpus=2  # Each partial correlation uses 2 GPUs internally
)

# Disable GPU usage (fallback to CPU parallelization)
correlations = compute_input_layer_correlations(
    input_features, 
    layer_features_dict, 
    use_gpu=False,
    n_jobs=8  # Use 8 CPU cores instead
)
```

## How Multi-GPU Works

### Input-Layer Correlations
1. **Layer Distribution**: Layers are distributed evenly across available GPUs
2. **Parallel Processing**: Each GPU processes its assigned layers independently
3. **Memory Management**: Automatic GPU memory cleanup after each layer computation
4. **Fallback Strategy**: Graceful fallback to CPU if GPU computation fails

### Progressive Partial Correlations
1. **Sequential Structure**: Layers must be processed sequentially due to dependencies
2. **Internal Parallelization**: Each partial correlation computation is parallelized across GPUs:
   - GPU 0: Regresses conditioning variables from input features
   - GPU 1: Regresses conditioning variables from current layer features
   - GPU 0: Computes final correlation between residuals
3. **Memory Efficiency**: Each GPU only holds data for its specific regression task
4. **Robust Fallbacks**: Complete fallback chain with multiple levels:
   - **Level 1**: Multi-GPU computation (if n_gpus > 1)
   - **Level 2**: Single GPU computation (if multi-GPU fails)
   - **Level 3**: CPU computation (if all GPU methods fail)
   - **Automatic Detection**: Graceful fallback when GPUs unavailable

## Performance Benefits

- **Scalability**: Near-linear speedup with additional GPUs
- **Memory Efficiency**: Each GPU only loads data for its assigned layers
- **Robustness**: Individual GPU failures don't crash the entire computation

## Requirements

- PyTorch with CUDA support (optional - will fallback to CPU)
- Multiple CUDA-capable GPUs (optional - will use fewer or fallback)
- Sufficient GPU memory for layer features (will fallback if insufficient)

## Example Performance

```python
# Single GPU: ~30 seconds for 12 layers
# Dual GPU: ~16 seconds for 12 layers  
# Quad GPU: ~8 seconds for 12 layers
```

## Fallback Scenarios

The implementation automatically handles various failure scenarios:

### No GPU Available
```python
# Automatically detects and uses CPU
correlations = compute_progressive_partial_correlations(
    input_features, layer_features_dict, use_gpu=True  # Will fallback to CPU
)
# Output: "CUDA not available, falling back to CPU"
```

### Insufficient GPUs
```python
# Requests 4 GPUs but only 2 available
correlations = compute_progressive_partial_correlations(
    input_features, layer_features_dict, use_gpu=True, n_gpus=4
)
# Output: "Using 2 GPUs for internal partial correlation computations"
```

### GPU Memory Exhausted
```python
# Individual GPU operations fail, fallback occurs automatically
correlations = compute_progressive_partial_correlations(
    large_input_features, large_layer_features_dict, use_gpu=True, n_gpus=2
)
# Output: "GPU 1 regression failed: CUDA out of memory, using scikit-learn"
```

### PyTorch Not Available
```python
# Automatically uses pure CPU implementation
correlations = compute_progressive_partial_correlations(
    input_features, layer_features_dict, use_gpu=True
)
# Output: "Using CPU for all partial correlation computations"
```

## Testing Fallbacks

Run the test script to verify fallback behavior:
```bash
python test_cpu_fallback.py
```

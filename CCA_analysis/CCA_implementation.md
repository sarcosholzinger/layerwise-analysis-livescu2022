# CCA Implementation Overview

## What is Canonical Correlation Analysis (CCA)?

Canonical Correlation Analysis (CCA) is a statistical method for finding linear relationships between two sets of variables. In the context of neural network analysis, CCA measures the similarity between representations from different layers or models by finding maximally correlated linear projections.

**Key Concepts:**
- **Input**: Two matrices X and Y representing different neural network layers/models
- **Output**: Canonical correlation coefficients measuring linear similarity
- **Goal**: Find linear transformations that maximize correlation between projected representations

## CCA vs Pairwise CCA (PWCCA)

| Aspect | CCA | PWCCA |
|--------|-----|-------|
| **Purpose** | General similarity between any two representation sets | Specifically compares layers within the same model |
| **Input** | Any two matrices (X, Y) | Layer representations from the same model |
| **Use Case** | Inter-model comparison, model-to-mel comparison | Intra-model layer similarity analysis |
| **Base Layer** | Not applicable | Compares all layers against a reference layer (default: layer 0) |
| **Weighting** | All canonical directions weighted equally | Canonical correlations weighted by variance explained |
| **Score Calculation** | Mean of canonical correlations: $\frac{1}{k}\sum_{i=1}^k \rho_i$ | Weighted sum: $\sum_{i=1}^k w_i \times \rho_i$ |
| **Implementation** | `cca_inter`, `cca_mel`, `cca_embed` | `cca_intra` (uses `compute_pwcca`) |

### **Critical Implementation Note: This Repository Uses PWCCA, Not Standard CCA**

The implementation in this codebase computes **Projection Weighted CCA (PWCCA)**, which is different from standard CCA. This distinction is crucial for interpreting results correctly.

#### **Why PWCCA Instead of Standard CCA?**

**Standard CCA Problem:**
- Treats all canonical directions equally
- Can be dominated by high-correlation but low-variance directions
- Misleading when comparing neural representations

**Example Problem:**
```
Canonical correlations: [0.95, 0.02, 0.01]
Variance explained:     [5%,  40%, 55%]
Standard CCA mean = 0.33, but most data variance has low correlation!
```

**PWCCA Solution:**
- Weights canonical correlations by variance explained in original data
- Focuses on correlations in directions that actually matter
- More meaningful for neural network analysis

#### **Mathematical Formulation**

**Standard CCA Score:**
$$\text{CCA} = \frac{1}{k}\sum_{i=1}^k \rho_i$$

**PWCCA Score:**
$$\text{PWCCA} = \sum_{i=1}^k w_i \times \rho_i$$

Where:
- $\rho_i$: i-th canonical correlation coefficient  
- $w_i$: projection weight for i-th canonical direction
- $w_i = \frac{|P_i^T \times \text{data}|}{\sum_j |P_j^T \times \text{data}|}$

#### **Implementation Details in Code**

**In `compute_pwcca` method (`codes/tools/cca_core.py`):**

1. **Compute canonical directions**: Standard CCA to get projection matrices
2. **Project data**: `proj_1 = proj_mat_x × (view1 - mean1)`
3. **Compute correlations**: Element-wise correlations between projections
4. **Calculate weights**: Based on variance explained by each canonical direction  
5. **Return weighted sum**: Final PWCCA score

**Key Code Segment:**
```python
def compute_pwcca(self, proj_mat_x, proj_mat_y, x_idxs, y_idxs, mean_score=False):
    # Get canonical correlations
    corr_scores, view1_projected, view2_projected = self.get_cca_coefficients(...)
    
    # Compute projection weights for both views
    score_x = self.compute_weighted_sum(view1_data, view1_projected, corr_scores)
    score_y = self.compute_weighted_sum(view2_data, view2_projected, corr_scores)
    
    if mean_score:  # NOT part of original PWCCA - custom modification
        return (score_x + score_y) / 2
    elif len(proj_mat_x) < len(proj_mat_y):
        return score_x  # Use lower-dimensional view (standard PWCCA)
    else:
        return score_y
```

#### **Custom Modification: `mean_score=True`**

**Important:** This implementation includes a **custom modification** to standard PWCCA:

- **Standard PWCCA**: Uses weights from lower-dimensional view only
- **Modified version** (`mean_score=True`): Averages weights from both views
- **Our default**: Uses the modified version for more balanced scoring

This modification affects score interpretation and is **not part of the original PWCCA literature**.

## File Structure and Data Flow

### NPZ to NPY Conversion Pipeline

```
Original HuBERT Features (NPZ)
    │
    ├── Contains: Multiple layers, multiple speakers, frame-level data
    │
    ▼ 
convert_hubert_subset_npz.py
    │
    ├── Processes: Subset selection, feature extraction, format conversion
    │
    ▼
Individual Layer Files (NPY)
    │
    ├── Structure: prepare_features/converted_features/hubert_subset/
    │   ├── layer_0.npy  (shape: [n_frames, n_features])
    │   ├── layer_1.npy
    │   ├── ...
    │   └── layer_11.npy
    │
    ▼
CCA Analysis Ready
```

### Expected Directory Structure for CCA

The CCA implementation expects a specific directory structure:
```
{rep_dir}/
├── contextualized/
│   └── frame_level/
│       ├── layer_0.npy
│       ├── layer_1.npy
│       └── ...
└── local/  (for convolutional layers, if applicable)
```

**Note**: Our implementation automatically creates symbolic links if files are found directly in `rep_dir` to maintain compatibility with the expected structure.

## Main Implementation Changes

### 1. CCA_analysis Directory Changes

#### `run_cca.py` - Main Analysis Script
- **Auto-detection**: Automatically detects model type based on layer count and directory names
- **Directory handling**: Creates symbolic links to handle different file structures
- **Simplified interface**: Focuses specifically on pairwise CCA (PWCCA) analysis
- **Path flexibility**: Works from different working directories

#### `convert_hubert_subset_npz.py` - Data Preprocessing
- **NPZ processing**: Extracts features from HuBERT NPZ files
- **Feature selection**: Supports subsetting of features and frames
- **Format conversion**: Converts to individual NPY files per layer
- **Quality checks**: Validates output shapes and data integrity

### 2. codes/codes Directory Changes

#### `tools/get_scores.py` - Core Analysis Functions
- **Import fixes**: Resolved circular import issues with relative imports
- **Function signatures**: Added missing parameters for different CCA analysis types
- **Error handling**: Better parameter validation and error messages

#### `tools/tools.py` - Utility Functions  
- **Import resolution**: Fixed `utils.` function call conflicts
- **Function availability**: Added missing `write_to_file` function
- **Path handling**: Improved directory and file path management

#### `tools/tools_utils.py` - Model Configuration
- **Model registry**: Contains layer counts for different model architectures
- **Supported models**: HuBERT (small/large), Wav2Vec2, WavLM, AV-HuBERT, etc.

## Implementation Details

### Model Auto-Detection Algorithm

```python
def detect_model_type(rep_dir):
    # 1. Count layer files (layer_*.npy)
    # 2. Determine max layer number
    # 3. Check directory name for model hints
    # 4. Map to LAYER_CNT dictionary keys
    # 5. Return appropriate model name
```

**Detection Logic:**
- `hubert` + 12 layers → `hubert_small`
- `hubert` + 24 layers → `hubert_large`
- Similar logic for other model types

### PWCCA Analysis Flow

```python
def cca_intra():
    # 1. Load base layer (default: layer 0)
    # 2. For each subsequent layer:
    #    a. Load layer representations  
    #    b. Compute CCA between base and current layer
    #    c. Store similarity score
    # 3. Return dictionary of layer similarities
```

### Cross-Validation Strategy

The CCA implementation uses k-fold cross-validation:
- **Splits**: Data divided into 10 parts
- **Training**: 8 parts for CCA parameter learning
- **Validation**: 1 part for hyperparameter selection
- **Testing**: 1 part for final evaluation
- **Regularization**: Multiple epsilon values tested for stability

## Usage Example

```python
# Simple PWCCA analysis
run_cca_analysis(
    rep_dir="prepare_features/converted_features/hubert_subset",
    save_fn="results/cca_scores.json"
)

# Output: Dictionary mapping layer numbers to CCA scores
# {1: 0.85, 2: 0.78, 3: 0.72, ...}
```

## Key Files and Their Roles

| File | Purpose | Key Functions |
|------|---------|---------------|
| `run_cca.py` | Main entry point | `run_cca_analysis()`, `detect_model_type()` |
| `get_scores.py` | CCA orchestration | `evaluate_cca()`, `getCCA` class |
| `tools.py` | Core CCA implementation | `get_cca_score()`, `CCACrossVal` class |
| `cca_core.py` | Mathematical CCA | `CCA` class with fit/transform methods |
| `convert_hubert_subset_npz.py` | Data preprocessing | NPZ to NPY conversion |

## Output Format

CCA scores are saved as JSON files with the structure:
```json
{
    "1": 0.847,
    "2": 0.782, 
    "3": 0.721,
    ...
    "11": 0.234
}
```

Where keys are layer numbers and values are CCA similarity scores relative to the base layer.

## Interpreting PWCCA Scores

**Important:** These interpretations are specific to **PWCCA (Projection Weighted CCA)**, not standard CCA. PWCCA weights correlations by variance explained in the original data, making the scores more meaningful for neural network analysis.

### PWCCA Score Ranges and Meaning

**PWCCA scores reflect both correlation AND importance** - they weight correlations by variance explained in the original data.

#### **High Scores (0.8-1.0):**
- **Mathematical meaning**: Strong correlations in **high-variance directions**
- **Neural interpretation**: Layer preserves the most important aspects of representations
- **Practical meaning**: Conservative, information-preserving transformation
- **Common in**: Early layers that perform minimal transformation
- **Example**: Layers that mainly adjust positional encoding or perform normalization

#### **Medium Scores (0.4-0.8):**
- **Mathematical meaning**: Mixed pattern - some important directions preserved, others transformed
- **Neural interpretation**: Healthy balance of information preservation and feature learning
- **Practical meaning**: Layer selectively transforms while preserving key information
- **Common in**: Well-functioning middle layers in transformers
- **Example**: Attention layers that preserve semantic content while reorganizing syntactic structure

#### **Low Scores (0.0-0.4):**
- **Mathematical meaning**: Little correlation in high-variance directions
- **Neural interpretation**: Layer performs significant transformation of important features
- **Practical meaning**: Fundamental representational change
- **Common in**: Layers that perform major abstraction (e.g., attention → output projection)
- **Example**: Final layers that map semantic representations to task-specific outputs

### **Key Difference from Standard CCA:**
Unlike standard CCA which treats all canonical directions equally, PWCCA focuses on directions that explain the most variance in the original data. This makes low scores more meaningful - they indicate that the layer is transforming the **most important** representational dimensions, not just noisy ones.

### **Understanding the Custom `mean_score=True` Modification:**
Our implementation uses `mean_score=True`, which averages weights from both views instead of using the standard PWCCA approach (weights from lower-dimensional view only). This provides more balanced scoring but is **not part of the original PWCCA literature**.

### Practical Guidelines
- **Layer progression**: Expect generally decreasing scores as layer distance from base increases
- **Model comparison**: Lower correlations suggest architectural differences  
- **Feature evolution**: Track how representations transform across layers
- **Important transformations**: Low PWCCA scores highlight layers performing major representational changes
- **Information flow**: High scores indicate information preservation pathways

## Technical Implementation Details

### Regularization and Numerical Stability

The implementation includes several techniques to ensure stable PWCCA computation:

#### **Variance-Based Regularization**
```python
# From remove_small method in cca_core.py
x_diag = np.abs(np.diagonal(sigma_xx))
y_diag = np.abs(np.diagonal(sigma_yy))
x_idxs = x_diag >= epsilon_x  # Remove low-variance dimensions
y_idxs = y_diag >= epsilon_y
```

**Purpose**: Remove dimensions with very low variance that can cause numerical instability in matrix inversions.

#### **Cross-Validation for Epsilon Selection**
- **Multiple epsilon values** tested: typically 6 different regularization strengths
- **10-fold cross-validation** used to select optimal regularization
- **Prevents overfitting** to specific data splits

#### **SVCCA Preprocessing**
The implementation combines **SVCCA (Singular Vector CCA)** preprocessing with **PWCCA** scoring:

1. **SVD preprocessing** reduces dimensionality before CCA computation
2. **PWCCA weighting** focuses on important canonical directions
3. **Improved numerical stability** through dimension reduction
4. **Consistent results** across different data scales

### Why This Matters for Neural Network Analysis

**High-dimensional problem**: Neural representations often have `d >> N` (dimensions >> samples)
- Covariance matrices become **singular** or **ill-conditioned**
- Standard CCA fails without regularization
- PWCCA+SVCCA provides stable, interpretable results

**Reference**: Based on methods from [Raghu et al. (2017)](https://papers.nips.cc/paper/2017/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html) and the [Google Research CCA implementation](https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb).

## References

1. Hotelling, H. (1936). Relations between two sets of variates. *Biometrika*, 28(3/4), 321-377.

2. Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004). Canonical correlation analysis: An overview with application to learning methods. *Neural Computation*, 16(12), 2639-2664.

3. Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. *Advances in Neural Information Processing Systems*, 30.

---

# Appendix: CCA Mathematical Derivation and Tutorial

## Introduction and Motivation

### The Core Idea of CCA

**CCA extends dimensionality reduction to work with two views of the same data.** Sometimes we have **extra knowledge** about our data that can help us find better low-dimensional representations than PCA alone.

**The key insight:** When we have **two different views** of the same underlying phenomenon, the **correlations between these views represent the meaningful part** of the data.

Note (this short comments requires further attention and rewrite - draft): there are limitations posed by CCA. In a paper by [Hinton et.al](https://arxiv.org/abs/1905.00414), it notes that CCA may fail when representations have higher dimension than number of data points. It also notes that any similarity index invariant to invertible linear transformation gives the same result when p ≥ n (Theorem 1 in the paper). The issue is that CCA is sensitive to perturbations and can be unstable.

### Two Formulations of CCA

1. **Maximum correlation formulation**: Learn linear subspaces that maximize correlations between two datasets
2. **Subspace minimization formulation**: Learn two subspaces such that projections of both datasets into their respective subspaces best approximate each other

The tutorial uses the **subspace minimization** approach, analogous to PCA's reconstruction error formulation.

### Mathematical Setup

**Data:**

- **View 1**: $X \in \mathbb{R}^{d_x \times N}$ (e.g., images of objects)
- **View 2**: $Y \in \mathbb{R}^{d_y \times N}$ (e.g., text descriptions of the same objects)
- $N$: number of data samples
- Both views observe the same underlying phenomenon

**Transformation matrices:**

- $U \in \mathbb{R}^{d_x \times k}$: transformation matrix for view $X$
- $V \in \mathbb{R}^{d_y \times k}$: transformation matrix for view $Y$
- $k$: desired dimensionality of the common subspace

---

## Understanding the Core CCA Assumption

### "Correlations Between Views Represent the Meaningful Part of the Data"

This fundamental assumption means:

- **What correlates across views = what matters**
- **What doesn't correlate = noise or view-specific artifacts**

### Intuitive Examples

**Example 1: Images and Captions**

- **View X**: Image features (pixel intensities, edges, shapes)
- **View Y**: Text descriptions ("red car", "blue sky", "person walking")
- **Correlation insight**: Image features that correlate with text features represent **semantic content** (objects, actions, colors)
- **Non-correlating parts**: Image noise, lighting variations, text style differences

**Example 2: Audio and Video**

- **View X**: Audio features (frequency spectrum, phonemes)
- **View Y**: Video features (lip movements, facial expressions)
- **Correlation insight**: What correlates represents **speech content**
- **Non-correlating parts**: Background noise, lighting, camera angles

### Application to Transformer Neural Architecture

**Setup for Transformer Layer Analysis:**

- **View X**: Input representations to a transformer layer $h_{\text{in}} \in \mathbb{R}^{d}$
- **View Y**: Output representations from that layer $h_{\text{out}} \in \mathbb{R}^{d}$
- **Data**: Multiple tokens/sequences processed through this layer

#### What CCA Would Reveal About Transformer Layers

**High correlation directions reveal:**

1. **Information preservation**: What semantic content is being maintained through the layer
2. **Core transformations**: The fundamental computation the layer performs
3. **Stable features**: Representations that survive the nonlinear transformation

**Low correlation directions reveal:**

1. **Information discarding**: What the layer considers "noise" and filters out
2. **View-specific processing**:
    - Input-specific: Positional encoding artifacts, embedding noise
    - Output-specific: Attention pattern side effects, activation function artifacts

#### Specific Transformer Insights

**If we find high correlation in direction $u$ (input) and $v$ (output):** $$\text{high correlation} \Rightarrow u^T h_{\text{in}} \approx v^T h_{\text{out}}$$

**This tells us:**

- **Linear transformation component**: The layer has a strong linear component $u \rightarrow v$
- **Information bottleneck**: This direction represents information the layer considers essential to preserve
- **Computational invariant**: Despite attention, MLPs, and nonlinearities, this semantic direction remains stable

#### Practical Transformer Interpretations

**Scenario 1: Early layers show high correlation**

- Layer is primarily doing **position encoding integration** and **basic feature extraction**
- Semantic content flows through relatively unchanged
- Correlations reveal which input dimensions encode stable semantic information

**Scenario 2: Middle layers show selective correlation**

- Some directions highly correlated: **syntactic structure preservation** (grammar, word relationships)
- Other directions uncorrelated: **semantic feature creation** (building higher-level concepts)
- CCA reveals which aspects of meaning vs. structure the layer focuses on

**Scenario 3: Late layers show low correlation**

- Layer is doing substantial **semantic transformation**
- Input representations are being heavily reorganized for the task
- Few directions correlate: the layer is essentially "rewriting" the representation

#### Mathematical Insight for Transformers

**CCA assumption in transformer context:** $$\text{Meaningful computation} = \text{What correlates between } h_{\text{in}} \text{ and } h_{\text{out}}$$

**This means:**

- **Correlated directions**: Capture the layer's **core computational purpose**
- **Uncorrelated directions**: Represent **auxiliary processing** or **noise filtering**

**Key insight:** CCA doesn't just find correlations—it finds the **most informative correlations**. In a transformer layer, this reveals the core computational transformation while filtering out implementation details.

---

## Problem Formulation

### The CCA Optimization Problem

$$\min_{U \in \mathbb{R}^{d_x \times k}, V \in \mathbb{R}^{d_y \times k}} |U^T X - V^T Y|_F \quad \text{s.t.} \quad U^T X X^T U = I_k, \quad V^T Y Y^T V = I_k$$

### Understanding the Components

**Objective function**: $|U^T X - V^T Y|_F$

- $U^T X$: projection of $X$ data into $k$-dimensional subspace
- $V^T Y$: projection of $Y$ data into $k$-dimensional subspace
- $|U^T X - V^T Y|_F$: Frobenius norm measuring how different these projections are
- **Goal**: Make the projections as similar as possible

**Constraints**: $U^T X X^T U = I_k$ and $V^T Y Y^T V = I_k$

- These impose **whitening** on the projected data
- Projected data should have diagonal covariance with unit variance in all directions
- Prevents trivial solutions and ensures fair comparison between views

### Requirements for CCA

**For this formulation to work, we require:**

- $C_{XX} = X X^T$ and $C_{YY} = Y Y^T$ are **invertible**
- This means the covariance matrices must be **positive definite**
- **Challenge**: This often fails in high-dimensional settings

---

## Understanding Whitening and Frobenius Norm

### What is Whitening?

**Whitening** transforms data so that:

1. **Zero mean**: $\mathbb{E}[\text{whitened data}] = 0$
2. **Unit covariance**: $\text{Cov}(\text{whitened data}) = I$

**This means:**

- All dimensions have **unit variance** (variance = 1)
- All dimensions are **uncorrelated** (covariance = 0 between different dimensions)

### The CCA Whitening Constraints

$$U^T X X^T U = I_k \quad \text{and} \quad V^T Y Y^T V = I_k$$

**What this means:**

- $U^T X$ (projected $X$ data) has covariance matrix $I_k$
- $V^T Y$ (projected $Y$ data) has covariance matrix $I_k$
- Both projected datasets are **whitened**

### Why Whitening is Critical for CCA

**Problem without whitening:** Imagine we have:

- **View X**: Very high variance in first dimension, low variance elsewhere
- **View Y**: Very high variance in second dimension, low variance elsewhere

**Without whitening:**

- CCA would find correlation between X's first dimension and Y's second dimension
- This might be **spurious correlation** just due to both having high variance
- The "correlation" is really just **scale effects**, not meaningful relationship

**Example:**

```
X = [1000, 1, 2]  (first dimension dominates)
Y = [3, 500, 1]   (second dimension dominates)
```

Without whitening, any small correlation between dominant dimensions would appear artificially important.

**With whitening:**

- All dimensions are normalized to unit variance
- CCA finds correlations based on **actual linear relationships**, not scale
- The correlation structure becomes about **direction**, not magnitude

### Geometric Interpretation of Whitening

**Before whitening:**

- Data might lie in an elongated ellipsoid
- Correlations are confounded with the shape of the data distribution

**After whitening:**

- Data lies in a unit sphere
- Correlations reflect pure **directional relationships**

**Analogy:** Like comparing apples to apples—we normalize out the "size effects" to focus on the "shape relationships."

### Understanding the Frobenius Norm

#### Definition

For matrix $A$, the Frobenius norm is: $$|A|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

**Intuition:** It's like the "Euclidean norm" but for matrices—sum of squares of all elements.

#### Why Use Frobenius Norm in CCA?

**Our objective:** $|U^T X - V^T Y|_F$

Let's say:

- $U^T X = P \in \mathbb{R}^{k \times N}$ (projected X data)
- $V^T Y = Q \in \mathbb{R}^{k \times N}$ (projected Y data)

Then: $$|P - Q|_F^2 = \sum_{i=1}^k \sum_{j=1}^N (P_{ij} - Q_{ij})^2$$

**This measures:**

- **Total squared difference** between corresponding elements
- **Treats all dimensions equally** (no preferential weighting)
- **Treats all data points equally** (no preferential weighting)

#### Alternative Interpretation

$$|U^T X - V^T Y|_F^2 = \sum_{n=1}^N |U^T x_n - V^T y_n|_2^2$$

**This shows:** We're minimizing the sum of **Euclidean distances** between corresponding projected data points.

**Geometric meaning:** We want the projected point clouds to be as close as possible, point by point.

#### Connection to Correlation

**Key insight:** Minimizing $|U^T X - V^T Y|_F^2$ with whitening constraints is equivalent to **maximizing correlation**!

**Why?** With whitened data: $$|U^T X - V^T Y|_F^2 = 2k - 2\text{tr}(U^T X Y^T V) = 2k - 2 \times \text{correlation terms}$$

So minimizing the Frobenius norm = maximizing the correlation terms.

---

## Mathematical Derivation

### Step 1: Expanding the Frobenius Norm Objective

**Starting with:** $$|U^T X - V^T Y|_F$$

**Using the definition of Frobenius norm:** $$|U^T X - V^T Y|_F^2 = \text{tr}((U^T X - V^T Y)(U^T X - V^T Y)^T)$$

### Step 2: Expand the Product

$$(U^T X - V^T Y)(U^T X - V^T Y)^T = (U^T X - V^T Y)(X^T U - Y^T V)$$

**Expanding the matrix multiplication:** $$= U^T X X^T U + V^T Y Y^T V - V^T Y X^T U - U^T X Y^T V$$

### Step 3: Take the Trace

$$|U^T X - V^T Y|_F^2 = \text{tr}(U^T X X^T U) + \text{tr}(V^T Y Y^T V) - \text{tr}(V^T Y X^T U) - \text{tr}(U^T X Y^T V)$$

### Step 4: Apply the Constraints

**Using our whitening constraints:**

- $U^T X X^T U = I_k \Rightarrow \text{tr}(U^T X X^T U) = k$
- $V^T Y Y^T V = I_k \Rightarrow \text{tr}(V^T Y Y^T V) = k$

**So our expression becomes:** $$|U^T X - V^T Y|_F^2 = 2k - \text{tr}(V^T Y X^T U) - \text{tr}(U^T X Y^T V)$$

### Step 5: Simplify Using Trace Properties

**Key insight:** $\text{tr}(A) = \text{tr}(A^T)$ for any matrix.

$$\text{tr}(V^T Y X^T U) = \text{tr}((V^T Y X^T U)^T) = \text{tr}(U^T X Y^T V)$$

**Therefore:** $$|U^T X - V^T Y|_F^2 = 2k - 2\text{tr}(U^T X Y^T V)$$

### Step 6: Convert to Equivalent Optimization Problem

Since $2k$ is constant: $$\min_{U,V} |U^T X - V^T Y|_F^2 \equiv \max_{U,V} \text{tr}(U^T X Y^T V)$$

**Final CCA problem:** $$\max_{U,V} \text{tr}(U^T X Y^T V) \quad \text{s.t.} \quad U^T X X^T U = V^T Y Y^T V = I_k$$

**Beautiful result:** Minimizing Frobenius norm with whitening constraints is exactly equivalent to maximizing total correlation between projected datasets.

### Step 7: Lagrangian Setup

**Lagrangian:** $$L(\Lambda_X, \Lambda_Y) = \text{tr}(U^T X Y^T V) - \text{tr}((U^T X X^T U - I_k)\Lambda_X) - \text{tr}((V^T Y Y^T V - I_k)\Lambda_Y)$$

### Step 8: Taking Partial Derivatives

**With respect to $U$:** $$\frac{\partial L}{\partial U} = X Y^T V - 2X X^T U\Lambda_X = 0$$

**Using covariance notation** ($C_{XY} = X Y^T$, $C_{XX} = X X^T$): $$C_{XY} V = 2C_{XX} U\Lambda_X$$

**With respect to $V$:** $$\frac{\partial L}{\partial V} = Y X^T U - 2Y Y^T V\Lambda_Y = 0$$

**Using covariance notation** ($C_{YX} = Y X^T$, $C_{YY} = Y Y^T$): $$C_{YX} U = 2C_{YY} V\Lambda_Y$$

### Step 9: Finding Relationship Between Lagrange Multipliers

**Pre-multiply first equation by $U^T$ and second by $V^T$:**

**From first equation:** $$U^T C_{XY} V = 2U^T C_{XX} U\Lambda_X = 2\Lambda_X$$ (using constraint $U^T C_{XX} U = I_k$)

**From second equation:** $$V^T C_{YX} U = 2V^T C_{YY} V\Lambda_Y = 2\Lambda_Y$$ (using constraint $V^T C_{YY} V = I_k$)

**Key insight:** $C_{XY} = (C_{YX})^T$, so: $$U^T C_{XY} V = (V^T C_{YX} U)^T = V^T C_{YX} U$$

**Therefore:** $\Lambda_X = \Lambda_Y = \Lambda$

### Step 10: Diagonalization and Final Equations

**Since $\Lambda$ is symmetric, diagonalize it:** $$\Lambda = W D W^T$$

**Following same approach as PCA, define:** $$\tilde{U} = UW, \quad \tilde{V} = VW$$

**This gives us:** $$C_{YX} \tilde{U} = C_{YY} \tilde{V} D \quad \text{(1)}$$ $$C_{XY} \tilde{V} = C_{XX} \tilde{U} D \quad \text{(2)}$$

### Step 11: Isolating Variables

**From equation (1), solve for $\tilde{V}$:** $$\tilde{V} = C_{YY}^{-1} C_{YX} \tilde{U} D^{-1} \quad \text{(3)}$$

**Substitute into equation (2):** $$C_{XY} (C_{YY}^{-1} C_{YX} \tilde{U} D^{-1}) = C_{XX} \tilde{U} D$$

**Simplifying:** $$C_{XY} C_{YY}^{-1} C_{YX} \tilde{U} D^{-1} = C_{XX} \tilde{U} D$$

**Multiply both sides by $D$ on the right:** $$C_{XY} C_{YY}^{-1} C_{YX} \tilde{U} = C_{XX} \tilde{U} D^2$$

**Pre-multiply by $C_{XX}^{-1}$:** $$C_{XX}^{-1} C_{XY} C_{YY}^{-1} C_{YX} \tilde{U} = \tilde{U} D^2$$

### Final Result: Eigenvalue Problem

**We've derived the eigenvalue problem:** $$C_{XX}^{-1} C_{XY} C_{YY}^{-1} C_{YX} \tilde{U} = \tilde{U} D^2$$

**Similarly:** $$C_{YY}^{-1} C_{YX} C_{XX}^{-1} C_{XY} \tilde{V} = \tilde{V} D^2$$

**The solution:**

- Find eigenvectors $\tilde{U}$ of $C_{XX}^{-1} C_{XY} C_{YY}^{-1} C_{YX}$
- Find eigenvectors $\tilde{V}$ of $C_{YY}^{-1} C_{YX} C_{XX}^{-1} C_{XY}$
- Eigenvalues $D^2$ represent **squared canonical correlations**
- Choose $k$ largest eigenvalues for maximum correlation

**Relationship between solutions:** $$\tilde{V} = C_{YY}^{-1} C_{YX} \tilde{U} D^{-1}$$

---

## Generalized Eigenvalue Formulation

### Block Matrix Formulation

Instead of solving separate eigenvalue problems, we can solve for both bases simultaneously:

$$\begin{bmatrix} 0 & C_{XY} \\ C_{YX} & 0 \end{bmatrix} \begin{bmatrix} \tilde{U} \\ \tilde{V} \end{bmatrix} = \begin{bmatrix} C_{XX} & 0 \\ 0 & C_{YY} \end{bmatrix} \begin{bmatrix} \tilde{U} \\ \tilde{V} \end{bmatrix} D$$

### Understanding the Block Structure

**Define:** $$A = \begin{bmatrix} 0 & C_{XY} \\ C_{YX} & 0 \end{bmatrix}, \quad B = \begin{bmatrix} C_{XX} & 0 \\ 0 & C_{YY} \end{bmatrix}$$

**Then our problem becomes the generalized eigenvalue problem:** $$A \begin{bmatrix} \tilde{U} \\ \tilde{V} \end{bmatrix} = B \begin{bmatrix} \tilde{U} \\ \tilde{V} \end{bmatrix} D$$

### Expanding the Block Equations

**The block matrix equation gives us:** $$\begin{bmatrix} C_{XY} \tilde{V} \\ C_{YX} \tilde{U} \end{bmatrix} = \begin{bmatrix} C_{XX} \tilde{U} D \\ C_{YY} \tilde{V} D \end{bmatrix}$$

**This gives us the system:** $$C_{XY} \tilde{V} = C_{XX} \tilde{U} D$$ $$C_{YX} \tilde{U} = C_{YY} \tilde{V} D$$

**These are exactly the same equations we derived before!**

### Converting to Standard Eigenvalue Problem

**Assuming covariance matrices are invertible:** $$B^{-1} A \begin{bmatrix} \tilde{U} \\ \tilde{V} \end{bmatrix} = \begin{bmatrix} \tilde{U} \\ \tilde{V} \end{bmatrix} D$$

**Where:** $$B^{-1} A = \begin{bmatrix} 0 & C_{XX}^{-1} C_{XY} \\ C_{YY}^{-1} C_{YX} & 0 \end{bmatrix}$$

### Connection to Rayleigh Quotient

The generalized eigenvalue problem is equivalent to maximizing the **Rayleigh quotient**:

$$R(w) = \frac{w^T A w}{w^T B w}$$

**For our CCA case:** $$w = \begin{bmatrix} \tilde{U} \\ \tilde{V} \end{bmatrix}$$

**The Rayleigh quotient becomes:** $$R(w) = \frac{2\tilde{U}^T C_{XY} \tilde{V}}{\tilde{U}^T C_{XX} \tilde{U} + \tilde{V}^T C_{YY} \tilde{V}}$$

**This is exactly the correlation we want to maximize!**

### Comparison with PCA

**For PCA:** $$A = C_{XX}, \quad B = I$$

Since PCA uses only a single view of data, it makes intuitive sense that:

- Matrix $A$ is just a single covariance matrix (not a block matrix)
- Matrix $B$ is identity (no cross-view constraints)

**CCA is a natural generalization of PCA to the two-view setting!**

---

## Stability and Regularized CCA

### The Stability Problem

#### When Covariance Matrices Fail to be Invertible

**Common scenarios:**

1. **High-dimensional data**: $d_x$ or $d_y > N$ (more features than samples)
2. **Perfectly correlated features**: Some dimensions are linear combinations of others
3. **Numerical issues**: Matrix is theoretically invertible but computationally singular

**The problem:** Our derivation required computing $C_{XX}^{-1}$ and $C_{YY}^{-1}$, but these may not exist!

**Mathematical symptoms:**

- $C_{XX}$ or $C_{YY}$ are **positive semi-definite** but not **positive definite**
- Some eigenvalues are zero or very close to zero
- Matrix inversion becomes unstable or impossible

#### Example of the Problem

**High-dimensional case:**

- 1000-dimensional features, 100 samples
- $C_{XX} \in \mathbb{R}^{1000 \times 1000}$ but rank($C_{XX}$) ≤ 100
- 900 eigenvalues are exactly zero → matrix is singular

**Numerical instability case:**

- Some eigenvalues are $10^{-16}$ (machine precision)
- Inverting these gives eigenvalues of $10^{16}$ → numerical explosion
- Small noise gets amplified enormously

### Regularized CCA Solution

#### The Regularized Objective

Instead of the original constrained optimization:

$$\min_{U \in \mathbb{R}^{d_x \times k}, V \in \mathbb{R}^{d_y \times k}} |U^T X - V^T Y|_F^2 + \lambda_x |U|_F^2 + \lambda_y |V|_F^2$$

**Understanding the regularization terms:**

- $\lambda_x |U|_F^2$: **Penalty on complexity** of $U$ transformation
- $\lambda_y |V|_F^2$: **Penalty on complexity** of $V$ transformation
- **Effect**: Prefer **smaller, simpler** transformation matrices

**Intuition:** "Find correlations, but don't make the transformations too complex."

#### Converting to Constrained Form

This is equivalent to:

$$\max_{U, V} \text{tr}(U^T X Y^T V) \quad \text{s.t.} \begin{cases} U^T(X X^T + \lambda_x I)U = I_k \\ V^T(Y Y^T + \lambda_y I)V = I_k \end{cases}$$

**Key insight:** We've **replaced** the original covariance matrices:

- $C_{XX} = X X^T \rightarrow X X^T + \lambda_x I$
- $C_{YY} = Y Y^T \rightarrow Y Y^T + \lambda_y I$

#### Why Regularization Solves the Problem

**Eigenvalue perspective:**

- Original: $C_{XX}$ has eigenvalues $[\sigma_1^2, \sigma_2^2, \ldots, \sigma_r^2, 0, 0, \ldots, 0]$
- Regularized: $C_{XX} + \lambda_x I$ has eigenvalues $[\sigma_1^2 + \lambda_x, \sigma_2^2 + \lambda_x, \ldots, \sigma_r^2 + \lambda_x, \lambda_x, \lambda_x, \ldots, \lambda_x]$

**All eigenvalues are now ≥ $\lambda_x > 0$** → matrix is **positive definite** → invertible!

**Geometric interpretation:**

- Original covariance: data lies in a lower-dimensional subspace
- Regularized: we "inflate" the covariance in all directions by $\lambda_x$
- Result: data now spans the full space (with minimum variance $\lambda_x$ in each direction)

### Cholesky Decomposition Approach

#### Why Use Cholesky?

Since $(C_{XX} + \lambda_x I)$ is now **positive definite**, it has a **Cholesky decomposition**:

$$(C_{XX} + \lambda_x I) = L_{XX} L_{XX}^T$$

where $L_{XX}$ is **lower triangular**.

**This allows us to solve the eigenvalue problem more stably.**

#### The SVD Solution

**The document shows this can be written as:** $$L_{XX}^{-\frac{1}{2}} C_{XY} (C_{YY} + \lambda_y I)^{-1} C_{YX} (L_{XX}^{-\frac{1}{2}})^T \tilde{U} = \tilde{U} D^2$$

**Key insight:** Let $M = L_{XX}^{-\frac{1}{2}} C_{XY} (C_{YY} + \lambda_y I)^{-\frac{1}{2}}$

Then: $M M^T \tilde{U} = \tilde{U} D^2$

**The beautiful result:** We can solve this by **SVD of $M$**!

**If $M = U_M \Sigma_M V_M^T$**, then:

- $M M^T = U_M \Sigma_M^2 U_M^T$
- The eigenvectors we need are columns of $U_M$
- The eigenvalues are $\Sigma_M^2$

**Why this is better:**

1. **SVD is more numerically stable** than eigenvalue decomposition
2. **No matrix inversions** in the final computation
3. **Works even when matrices are nearly singular**

### Practical Regularized CCA Algorithm

1. **Add regularization:**
    
    - $\tilde{C}_{XX} = X X^T + \lambda_x I$
    - $\tilde{C}_{YY} = Y Y^T + \lambda_y I$
2. **Compute Cholesky decompositions:**
    
    - $\tilde{C}_{XX} = L_{XX} L_{XX}^T$
    - $\tilde{C}_{YY} = L_{YY} L_{YY}^T$
3. **Form the matrix:**
    
    - $M = L_{XX}^{-\frac{1}{2}} C_{XY} L_{YY}^{-\frac{1}{2}}$
4. **Compute SVD:**
    
    - $M = U_M \Sigma_M V_M^T$
5. **Extract solutions:**
    
    - $\tilde{U} = L_{XX}^{-\frac{1}{2}} U_M$
    - $\tilde{V} = L_{YY}^{-\frac{1}{2}} V_M$
    - Canonical correlations = $\Sigma_M$

### Choosing Regularization Parameters

**Cross-validation approach:**

- Try different values of $\lambda_x, \lambda_y$
- Evaluate on held-out data using correlation or downstream task performance

**Rule of thumb:**

- Start with $\lambda_x = \lambda_y = 0.01 \times \text{trace}(C_{XX})/d_x$
- Adjust based on stability and performance

**Theoretical guidance:**

- If $d_x, d_y >> N$: need larger regularization
- If data is noisy: need larger regularization
- If seeking interpretability: larger regularization gives simpler solutions

### Summary: Key Insight

**Regularization provides a principled way to trade off between:**

- **Correlation maximization** (what we want)
- **Solution complexity** (what we want to control)

This makes regularized CCA much more practical for real-world high-dimensional data!

---

## Matrix Calculus and Definitions

### Matrix Derivatives for CCA

**Scalar with respect to vector:**

- $\frac{\partial}{\partial x}[a^T x] = a$
- $\frac{\partial}{\partial x}[x^T a] = a$

**Trace derivatives:**

- $\frac{\partial}{\partial X}[\text{tr}(AX)] = A^T$
- $\frac{\partial}{\partial X}[\text{tr}(XA)] = A^T$
- $\frac{\partial}{\partial X}[\text{tr}(AXB)] = A^T B^T$
- $\frac{\partial}{\partial X}[\text{tr}(X^T A X)] = 2AX$ (when $A$ is symmetric)

**Matrix products:**

- $\frac{\partial}{\partial X}[\text{tr}(X^T X)] = 2X$
- $\frac{\partial}{\partial X}[\text{tr}(XX^T)] = 2X$

### Key Matrix Properties

**Trace properties:**

- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(AB) = \text{tr}(BA)$ (cyclic property)
- $\text{tr}(A) = \text{tr}(A^T)$
- $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$

**Matrix definiteness:**

- **Positive definite**: $x^T A x > 0$ for all $x \neq 0$
- **Positive semi-definite**: $x^T A x \geq 0$ for all $x$

### Definitions

**Frobenius Norm**: $|A|_F = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_{i,j} A_{ij}^2}$

**Whitening**: Transform data to have identity covariance matrix

**Canonical Correlations**: The correlations between optimal linear combinations of two datasets

**Generalized Eigenvalue Problem**: $Ax = \lambda Bx$ where both $A$ and $B$ are matrices

**Rayleigh Quotient**: $R(x) = \frac{x^T A x}{x^T B x}$, maximized by eigenvector of $B^{-1}A$

**Cholesky Decomposition**: For positive definite $A$, $A = LL^T$ where $L$ is lower triangular

**Regularization**: Adding penalty terms to prevent overfitting and ensure numerical stability

## Additional Resources

- **CCA and CKA notebook**: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
- **Kornblith et al. (2019)**: Similarity of Neural Network Representations Revisited
- **Morcos et al. (2018)**: Insights on representational similarity in neural networks with canonical correlation analysis
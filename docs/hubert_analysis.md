# HuBERT Layer-wise Analysis

This document outlines the process of extracting and analyzing features from the HuBERT model.

## 1. Feature Extraction

### Prerequisites
- Python environment with required packages:
  - torch
  - torchaudio
  - transformers
  - numpy
  - tqdm
  - matplotlib
  - seaborn
  - scikit-learn

### Directory Structure
```
layerwise-analysis/
├── content/
│   └── data/
│       └── LibriSpeech/
│           └── dev-clean-2/  # Audio files
├── output/
│   └── hubert/
│       └── librispeech_dev-clean_sample1/  # Extracted features
└── output/
    └── visualizations/  # Analysis plots
```

### Running Feature Extraction

1. The feature extraction script (`extract_features.py`) extracts frame-level features from all layers of the HuBERT model.

2. Run the extraction using the provided bash script:
```bash
./extract_hubert_features.sh
```

This will:
- Load the HuBERT base model from HuggingFace
- Process all audio files in the specified directory
- Extract features from all layers
- Save features as .npz files in the output directory

## 2. Feature Visualization and Analysis (simple)

### Key Features

1. **Adaptive Padding Mechanism**
   - Handles variable-length audio files
   - Automatically determines maximum sequence length per layer
   - Supports both 2D (time, dim) and 3D (batch, time, dim) feature arrays
   - Ensures consistent dimensions for analysis while preserving temporal information

2. **Layer-wise Analysis**
   - Analyzes features from multiple transformer layers (0, 6, 12)
   - Compares feature distributions across layers
   - Identifies layer-specific patterns and transformations
   - Helps understand how information evolves through the network

3. **Comprehensive Visualizations**
   - **Feature Distributions**
     - Mean and standard deviation across layers
     - Helps identify layer specialization
     - Shows how feature statistics evolve through the network
   
   - **Layer Similarity Analysis**
     - Cosine similarity matrix between layers
     - Identifies which layers are more similar
     - Helps understand layer grouping and specialization
   
   - **Dimensionality Reduction**
     - PCA visualization with explained variance
     - t-SNE visualization (when sample size permits)
     - Shows how features cluster and separate across layers
   
   - **Feature Variance Analysis**
     - Distribution of feature variances per layer
     - Helps identify which layers have more discriminative features
     - Useful for understanding layer specialization

### Running Visualization

1. Run the visualization script:
```bash
python visualize_features.py --num_files 3
```

Parameters:
- `--num_files`: Number of audio files to analyze (default: 3)
- `--features_dir`: Directory containing feature .npz files
- `--output_dir`: Directory to save visualizations

### Interpreting the Results

1. **Feature Distributions**
   - Look for trends in mean and variance across layers
   - Higher variance might indicate more discriminative features
   - Sudden changes might indicate layer specialization

2. **Layer Similarity**
   - Darker colors indicate higher similarity between layers
   - Look for patterns of layer grouping
   - Identify which layers are most distinct

3. **Dimensionality Reduction**
   - PCA shows linear relationships in the feature space
   - Explained variance ratio indicates how much information is preserved
   - t-SNE shows non-linear relationships (when available)
   - Compare patterns across layers to understand feature evolution

4. **Feature Variance**
   - Wider distributions indicate more diverse feature activations
   - Helps identify which layers are more discriminative
   - Useful for understanding layer specialization

## 3. Customizing the Analysis

### Feature Extraction Parameters
- `--model_name`: HuBERT model to use
- `--data_sample`: Sample identifier
- `--rep_type`: Type of representation ("local" or "contextualized")
- `--span`: Time span ("frame", "phone", or "word")
- `--subset_id`: Subset identifier for parallel processing
- `--dataset_split`: Dataset split to process
- `--save_dir`: Directory to save extracted features
- `--audio_dir`: Directory containing audio files

### Visualization Parameters
- `--features_dir`: Directory containing feature .npz files
- `--output_dir`: Directory to save visualizations
- `--num_files`: Number of audio files to analyze

## 4. Notes

- The analysis focuses on frame-level features from the entire audio files
- Features are extracted from all transformer layers
- The visualization tools help understand how representations evolve across layers
- The analysis can be extended to include more specific aspects (e.g., phone-level or word-level analysis)
- The padding mechanism ensures consistent analysis across variable-length inputs
- Layer-wise analysis provides insights into the model's hierarchical feature learning 

----

# Visualization analysis across 3, 10, 50 extracted features

## Analysis of HuBERT-Base Feature Distributions
![HuBERT Small Feature Distributions Plots][/home/sarcosh1/repos/layerwise-analysis/docs/images/hubert_plots_n3_10_50.png]
### Feature Statistics Across Layers (Images 1-3)

The first set of visualizations shows feature statistics with error bars across all layers (0-12) of the HuBERT-Base model:

- **Consistent Mean Values**: All visualizations show feature means consistently centered around zero across all layers
- **Layer 12 Distinctiveness**: Layer 12 shows notably larger variance with error bars extending to approximately ±0.4, compared to ±0.2 for most other layers
- **Scale Differences**: Image 3 has a compressed y-axis scale (-0.25 to 0.25) compared to Images 1-2 (-0.4 to 0.4)
- **Error Bar Pattern**: Layers 10-11 also show slightly larger variance than early layers (0-1)

This consistent zero-centered pattern with increasing variance in deeper layers suggests the model maintains normalized representations while allowing for greater feature specialization in deeper layers.

### Feature Variance Distributions (Images 4-6)

These histograms reveal how feature variance is distributed across different layers:

- **Images 4 & 6**: Similar visualizations showing:
  - Layer 0: Variance range up to 1.4
  - Layer 6: Variance range up to 5.0
  - Layer 12: Variance range up to 20.0

- **Image 5**: Shows similar distributions with narrower x-axis ranges:
  - Layer 0: Up to 0.8
  - Layer 6: Up to 3.0
  - Layer 12: Up to 12.0

- **Different Count Scales**:
  - Image 6 shows higher count scales (~600,000 max for layer 12)
  - Images 4-5 show lower but consistent count scales

All visualizations confirm the same fundamental pattern: most features have very low variance (strong peak near zero), but the range of possible variances increases dramatically in deeper layers, with layer 12 showing the broadest variance distribution.

### Layer-wise Similarity Matrices (Images 7-9)

Images 7-9 show identical layer-wise similarity matrices for the HuBERT-Base model:

- **Clear Model Identification**: All three include "Model: HuBERT-Base" in the title
- **Three Distinct Clusters** are visible:
  - Early layers (0-1)
  - Middle layers (2-6)
  - Later layers (7-12)
- **Strong Diagonal**: Perfect self-similarity (1.0) shown in yellow along the diagonal
- **Near-Neighbor Similarity**: Adjacent layers show higher similarity than distant ones
- **Layer 0 Distinction**: Layer 0 has very low similarity (dark purple) with layers 7-12

The hierarchical structure of the model is clearly visible with the same pattern of layer clusters across all similarity matrices.

### PCA Visualizations (Images 10-12)

The PCA plots show differences in data distribution and explained variance:

- **Image 10**: Shows only 3 data points per layer with 100% explained variance
  - The scale of principal components increases with layer depth
  - Layer 0: PC1 range ~ -60 to +30
  - Layer 6: PC1 range ~ -40 to +80
  - Layer 12: PC1 range ~ -75 to +125

- **Image 11**: Shows ~10 data points with ~32% explained variance
  - Points are more scattered
  - The scale continues to increase with layer depth

- **Image 12**: Shows many more data points with only ~15% explained variance
  - Contains visible outliers far from the main clusters
  - The scale range expands significantly

This progression across the three PCA visualizations demonstrates how the representations become more complex and harder to capture in low dimensions as we include more data points, while maintaining the pattern of increasing scale/separation in deeper layers.

## Key Insights from HuBERT-Base Analysis

1. **Hierarchical Processing Structure**: The HuBERT-Base model displays a clear three-stage processing hierarchy, with distinct layer clusters that likely perform different functions in speech representation.

2. **Feature Normalization with Increasing Specialization**: All layers maintain zero-centered feature distributions, but variance increases dramatically in deeper layers, suggesting specialized neurons that respond to specific linguistic patterns.

3. **Layer 12 Specialization**: The final layer consistently shows the largest variance range, reinforcing its role as the most specialized layer for high-level speech representation.

4. **Representational Complexity Growth**: The PCA visualizations reveal that deeper layers require more principal components to explain the same amount of variance, indicating a more complex and nuanced representation space.

5. **Speech Processing Pipeline**: The layer similarity patterns suggest HuBERT-Base processes speech through a pipeline: acoustic features in early layers (0-1), intermediate phonetic integration in middle layers (2-6), and abstract linguistic representation in later layers (7-12).

These visualizations collectively demonstrate how HuBERT-Base transforms raw speech features into increasingly abstract and specialized representations through its 13-layer architecture, with clear functional differentiation between early, middle, and late layers.
# Wav2Vec2 Layer-wise Analysis

This document outlines the process of extracting and analyzing features from the Wav2Vec2 model.

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
│   └── wav2vec2/
│       └── librispeech_dev-clean_sample1/  # Extracted features
└── output/
    └── visualizations/  # Analysis plots
```

### Running Feature Extraction

1. The feature extraction script (`extract_features-wav2vec2.py`) extracts frame-level features from all layers of the Wav2Vec2 model.

2. Run the extraction using the provided bash script:
```bash
./extract_wav2vec2_features.sh
```

This will:
- Load the Wav2Vec2 base model from HuggingFace
- Process all audio files in the specified directory
- Extract features from all layers
- Save features as .npz files in the output directory

## 2. Feature Visualization and Analysis

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
python visualize_features.py \
    --features_dir /home/sarcosh1/repos/layerwise-analysis/output/wav2vec2/librispeech_dev-clean_sample1 \
    --output_dir /home/sarcosh1/repos/layerwise-analysis/output/visualizations \
    --num_files 3 \
    --model_name "wav2vec2-base"
```

Parameters:
- `--num_files`: Number of audio files to analyze (default: 3)
- `--features_dir`: Directory containing feature .npz files
- `--output_dir`: Directory to save visualizations
- `--model_name`: Name of the model architecture

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
- `--model_name`: Wav2Vec2 model to use
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
- `--model_name`: Name of the model architecture

## 4. Notes

- The analysis focuses on frame-level features from the entire audio files
- Features are extracted from all transformer layers
- The visualization tools help understand how representations evolve across layers
- The analysis can be extended to include more specific aspects (e.g., phone-level or word-level analysis)
- The padding mechanism ensures consistent analysis across variable-length inputs
- Layer-wise analysis provides insights into the model's hierarchical feature learning
----
# Visualization Analysis (simple)

### t-SNE Visualizations with Different Perplexity Settings (Images 1 & 3)

The t-SNE visualizations reveal important differences in feature clustering at different perplexity settings:

- **High Perplexity (Image 1, perplexity=30)**:
  - Shows approximately 30-40 data points with nuanced clustering behavior
  - Layer 0: Points are primarily distributed in the range of 3.5-6.0 on t-SNE1 and 1.5-4.5 on t-SNE2
  - Layer 6: Shows a narrower negative range on t-SNE2 (-5.0 to -2.5)
  - Layer 12: Shows the most negative range on t-SNE2 (-9.5 to -6.5) and shifted leftward on t-SNE1

- **Low Perplexity (Image 3, perplexity=9)**:
  - Shows only about 9 points with wider spacing
  - Points are more evenly distributed across the coordinate space
  - Lacks the dense clustering seen in the high perplexity visualization

Higher perplexity settings reveal more global structure while preserving local relationships, suggesting that wav2vec2 forms meaningful feature clusters that become more distinct in deeper layers.

## PCA Visualizations with Different Data Samples (Images 2, 5 & 9)

The PCA plots show significant differences in explained variance and point distribution:

- **Image 2**: Shows ~10 data points with ~32% explained variance
  - Relatively consistent variance explanation across layers (0.32-0.34)
  - Points are scattered with some outliers
  
- **Image 5**: Shows many more data points with less explained variance (0.14-0.22)
  - Layer 12 explains more variance (0.22) than layers 0 (0.14) and 6 (0.16)
  - Layer 12 shows a distinct curved pattern suggesting non-linear relationships
  
- **Image 9**: Shows only 3 data points with 100% explained variance
  - This subset is perfectly explained in just 2 principal components
  - Scale ranges are consistent across layers

The increasing explained variance in deeper layers of wav2vec2-base suggests more efficient feature organization than in early layers.

### Feature Statistics Visualizations (Images 8, 10 & 11)

The error bar plots show feature means and variances across layers:

- **Images 8 & 10**: Show similar patterns with substantial difference in y-axis scale
  - All layers maintain zero-centered features
  - Layer 11 has the largest variance with error bars extending to approximately ±1.2
  - Other layers show relatively consistent variance ranges

- **Image 11**: Shows a compressed y-axis scale (-0.6 to 0.6) but maintains the same pattern
  - The relative difference between layer 11 and others is still clearly visible

These consistent zero-centered patterns with layer 11 showing the largest variance suggest that wav2vec2-base concentrates its most specialized features in the penultimate layer rather than the final layer.

## Feature Variance Distributions (Images 7, 12 & 13)

These histograms reveal the distribution of variance across features:

- **Image 7**: Shows similar distributions with relatively narrow x-axis ranges (0-1.0)
  - All layers show strong peak near zero

- **Image 12**: Similar distribution pattern with slightly different count scales
  - Layer 6 has the highest peak count (~125,000)
  - Layer 12 shows a slightly wider tail

- **Image 13**: Shows wider x-axis ranges, particularly for layer 6 (0-3.5)
  - All three layers maintain the same basic distribution shape
  - The majority of features still have very low variance

All visualizations confirm that most features in wav2vec2-base have low variance, but the range of possible variances is narrower than in HuBERT, suggesting different specialization patterns.

### Layer-wise Similarity Matrices for wav2vec2-base (Images 4, 6 & 14)

The identical similarity matrices reveal the hierarchical structure of the wav2vec2-base model:

- **Three Distinct Clusters**:
  - Early layers (0-1)
  - Middle layers (2-6)
  - Later layers (7-12), with subgroups at 7-9 and 10-12
  
- **Notable Features**:
  - Perfect self-similarity (1.0) along the diagonal
  - Layer 0 has extremely low similarity with layers 11-12 (darkest purple)
  - Layers 1-3 form a tighter cluster than in HuBERT
  - Layer 11 shows particularly high similarity with only layer 10

The wav2vec2-base model shows a clearer separation between late-stage layer groups (10-12) than HuBERT, suggesting more distinct functional specialization in these final layers.

### Comparative Analysis: wav2vec2-base vs. HuBERT-base

### Architectural Similarities

1. **Hierarchical Processing**: Both models display a three-stage processing hierarchy with distinct layer clusters for early acoustic processing, middle contextual integration, and late linguistic abstraction.

2. **Zero-Centered Features**: Both maintain normalized, zero-centered features across all layers while allowing for increasing variance in deeper layers.

3. **Low-Variance Dominance**: Both models show that the vast majority of features have very low variance, with only a small subset of features developing high variance for specialized pattern detection.

### Key Differences

1. **Peak Variance Location**:
   - **HuBERT-base**: Maximum variance occurs in layer 12 (final layer)
   - **wav2vec2-base**: Maximum variance occurs in layer 11 (penultimate layer)
   
2. **Variance Range**:
   - **HuBERT-base**: Shows wider variance ranges in deep layers (up to 20.0)
   - **wav2vec2-base**: Shows narrower variance ranges (generally under 3.5)

3. **Layer Similarity Patterns**:
   - **HuBERT-base**: Shows smoother transitions between adjacent layers
   - **wav2vec2-base**: Shows sharper boundaries between layer groups, especially between 9-10 and 10-11

4. **PCA Explained Variance**:
   - **HuBERT-base**: Later layers show decreasing explained variance with more points (15%)
   - **wav2vec2-base**: Layer 12 shows higher explained variance (22%) than earlier layers

5. **Feature Distribution Structure**:
   - **HuBERT-base**: PCA reveals outlier points with extreme values
   - **wav2vec2-base**: PCA shows more curved, continuous distribution patterns, suggesting more structured representation spaces

### Functional Implications

1. **Information Encoding Strategy**:
   - **HuBERT-base** appears to use a progressive refinement approach where each layer builds upon the previous one
   - **wav2vec2-base** seems to employ more distinct functional specialization between layer groups

2. **Feature Specialization**:
   - **HuBERT-base** gradually increases feature specialization throughout the network
   - **wav2vec2-base** concentrates specialized features in layer 11, with layer 12 potentially serving as a final integration layer

3. **Representation Efficiency**:
   - **wav2vec2-base**'s higher explained variance in layer 12 suggests it may achieve more efficient feature organization in its final representations

## Conclusion

While both HuBERT-base and wav2vec2-base follow similar overall architectural patterns with hierarchical processing and zero-centered features, they exhibit distinct specialization strategies. HuBERT-base displays more gradual feature refinement with maximum variance in its final layer, while wav2vec2-base shows sharper functional boundaries between layer groups with peak variance in its penultimate layer.

These differences likely reflect their different training objectives and methodologies: HuBERT uses discrete speech units with an iterative training approach, while wav2vec2 employs contrastive learning on masked segments. This comparative analysis reveals how architectural variations in self-supervised speech models lead to different internal representation patterns despite similar overall performance on downstream tasks.
# Audio File Extraction and Processing

This document outlines the audio file extraction and processing pipeline for both HuBERT and Wav2Vec2 models.

## Audio File Details

### Sample Files Used
The analysis uses 3 audio files from the LibriSpeech dev-clean-2 dataset:

1. **7850-281318-0010.flac**
   - Size: 210KB
   - Content: "O WISE MOTHER MAGPIE DEAR MOTHER MAGPIE THEY CRIED TEACH US HOW TO BUILD OUR NESTS LIKE YOURS FOR IT IS GROWING NIGHT AND WE ARE TIRED AND SLEEPY"
   - Duration: ~3-5 seconds
   - Samples: ~48,000-80,000 (at 16kHz)

2. **7850-281318-0011.flac**
   - Size: 134KB
   - Content: "THE MAGPIE SAID SHE WOULD TEACH THEM IF THEY WOULD BE A PATIENT DILIGENT OBEDIENT CLASS OF LITTLE BIRDS"
   - Duration: ~3-5 seconds
   - Samples: ~48,000-80,000 (at 16kHz)

3. **7850-281318-0012.flac**
   - Size: 79KB
   - Content: "AND WHERE EACH BIRD PERCHED THERE IT WAS TO BUILD ITS NEST"
   - Duration: ~3-5 seconds
   - Samples: ~48,000-80,000 (at 16kHz)

### Total Audio Statistics
- Total duration: 9-15 seconds
- Total samples: 144,000-240,000 (at 16kHz)
- Sampling rate: 16kHz (16,000 samples per second)
- Format: FLAC (lossless audio compression)

## Processing Pipeline

### 1. Audio Loading and Preprocessing
```python
# Load and preprocess audio
waveform, sample_rate = torchaudio.load(audio_path)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
```

### 2. Feature Extraction
- Features are extracted from all layers of the model
- Each layer produces high-dimensional features (768 dimensions for base models)
- Features are extracted at a frame rate of approximately 20ms per frame

### 3. Feature Storage
- Features are saved in NPZ (NumPy compressed) format
- Each file contains features from all model layers
- File naming convention: `[audio_file_name]_features.npz`

### Output File Sizes
1. **7850-281318-0010_features.npz**: 23MB
2. **7850-281318-0011_features.npz**: 14MB
3. **7850-281318-0012_features.npz**: 8.1MB

## Directory Structure 

## Model Layer Analysis

### HuBERT and Wav2Vec2 Layer Structure
Both models use a transformer architecture with multiple layers. The analysis probes different aspects of these layers:

1. **Comprehensive Layer Analysis**
   - All layers are analyzed for:
     - Feature distributions (mean and standard deviation)
     - Layer-wise similarity matrix
   - This provides a complete view of how features evolve through the network

2. **Representative Layer Analysis**
   Three key layers are selected for detailed analysis:
   - `layer_0`: First transformer layer
   - `layer_6`: Middle transformer layer
   - `layer_12`: Final transformer layer
   
   These layers are analyzed for:
   - PCA visualization
   - t-SNE visualization
   - Feature variance distribution

### Visualization Types and Layer Coverage

1. **Feature Distributions Plot**
   - Analyzes ALL layers
   - Shows mean and standard deviation of features
   - Helps understand feature evolution across the entire network

2. **Layer Similarity Matrix**
   - Analyzes ALL layers
   - Computes cosine similarity between every pair of layers
   - Shows how layers relate to each other

3. **Dimensionality Reduction (PCA and t-SNE)**
   - Analyzes three representative layers (0, 6, 12)
   - Shows how features cluster in 2D space
   - Helps understand feature organization at different network depths

4. **Feature Variance Analysis**
   - Analyzes three representative layers (0, 6, 12)
   - Shows distribution of feature variances
   - Helps understand feature specialization at different depths
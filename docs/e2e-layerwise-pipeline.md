# End-to-End Layerwise Analysis Pipeline

## Overview

This pipeline extracts and analyzes layerwise features from HuBERT models on LibriSpeech audio data (dev-clean).
It provides an initial analysis of neural speech representations across all model layers.


## Project Structure

```
layerwise-analysis-cca-vis/
├── _utils/
│   ├── data_utils.py          # Data loading & preprocessing (with segmentation option!)
│   ├── math_utils.py          # Mathematical computations (CKA, correlation, etc.)
│   └── visualization_utils.py # Basic plotting utilities
├── _analysis/
│   ├── similarity_analysis.py # Layer similarity analysis
│   └── temporal_analysis.py   # Temporal dynamics & animations
├── extract/
│   ├── extract_features-HuBERT.py # Multi-GPU feature extraction
│   ├── main_extract_features_chunked.py # Chunked processing for large datasets
│   └── convert_flac_to_wav.sh     # Convert FLAC to WAV
├── main_layerwise_analysis.py     # Primary analysis pipeline
├── main_gpu_analysis.py           # GPU-accelerated similarity analysis
├── main_gpu_temporal_analysis.py  # GPU-accelerated temporal analysis
├── main_layerwise_analysis_examples.py # Usage examples
├── tools/                         # Setup and utility scripts
├── _slurm/                        # SLURM job configurations
├── _tests_and_debug/             # Testing and debugging utilities
└── .vscode/launch.json           # Debugging configurations
```

## LibriSpeech Dataset

### Description
LibriSpeech is a large corpus of English speech derived from audiobooks that are part of the LibriVox project. It contains approximately 1,000 hours of speech data sampled at 16kHz.

Some of the dev-clean audio has also been converted from FLAC to WAV (speaker ids 422, 251, 174, 84, 2412). These have been chosen arbitrarily only to test the scripts and general pipeline, but are not required to extract features from models. 

### LibriSpeech Dataset Overview
For the purpose of layer-wise analysis we're only using portions of the dev-clean dataset. 
- **dev-clean**: Clean development set (~5.4 hours, 2,703 utterances)
- **test-clean**: Clean test set (~5.4 hours, 2,620 utterances)
- **train-clean-100**: Clean training subset (100 hours)

### Audio Characteristics
- **Sample Rate**: 16 kHz
- **Format**: FLAC
- **Language**: English
- **Content**: Read speech from audiobooks
- **Quality**: High-quality recordings with minimal background noise

### Directory Structure
```
LibriSpeech/
├── dev-clean/
│   ├── 84/
│   │   ├── 121123/
│   │   │   ├── 84-121123-0000.flac
│   │   │   ├── 84-121123-0000.wav
│   │   │   ├── 84-121123-0001.flac
│   │   │   ├── 84-121123-0001.wav
│   │   │   └── ...
│   │   └── ...
│   └── ...
```

## Feature Extraction Pipeline
Implemented to enable multi-GPU processing to speed up feature extraction. Although careful implementation has been maintained, it is worth noting this step may require further review (!!) 
Each audio file is processed while being distributed across different GPUs for parallel processing, an example of this is shown below:

`extract_features-HuBERT.py` log output:
```When Output directory: ./output/hubert_complete/librispeech_dev-clean_sample1

Found 2880 audio files
Available GPUs: 4, Using: 4
Using multi-GPU processing with 4 GPUs
Batch sizes: [720, 720, 720, 720]
```

## NPZ Files extracted with Features

### Current Processing Status (Updated)
- **Files Processed**: 1,655 audio files from LibriSpeech dev-clean
- **Output Location**: `output/hubert_complete/librispeech_dev-clean_sample1/`
- **Feature Format**: NPZ compressed format with all HuBERT layer features
- **Total Storage**: ~150GB of extracted features
- **Pattern**: All files follow the LibriSpeech naming convention --> SPEAKER_CHAPTER_SPEAKER-CHAPTER-UTTERANCE

#### Dataset Breakdown
**Unique Speakers (27 total):**
84, 174, 251, 422, 652, 1462, 1919, 1988, 2078, 2086, 2277, 2412, 2428, 3081, 3752, 3853, 5338, 5536, 5694, 5895, 6241, 6295, 6313, 6319, 6345, 7850, 8842

**Unique Chapters (64 total):**
4943, 4944, 24615, 24640, 24833, 34615, 34622, 34629, 43358, 43359, 43363, 50561, 57405, 61943, 61946, 64029, 64257, 64301, 64726, 66125, 66129, 66616, 73752, 83699, 84280, 93302, 93306, 111771, 118436, 121123, 121550, 122949, 129742, 130726, 130737, 136532, 137823, 142785, 142845, 147956, 148538, 149214, 149220, 149874, 149896, 149897, 153947, 153948, 153954, 163249, 166546, 168635, 170138, 170142, 170145, 244435, 281318, 284437, 286674, 302196, 302201, 302203, 304647

**Utterances:**
- **Range**: 0000 - 0089 (90 unique utterance IDs)
- **Coverage**: Comprehensive coverage of utterances per speaker-chapter combination

### Processed File IDs
Sample of processed files (total: 1,655):
```
251_118436_251-118436-0014
251_136532_251-136532-0000
3081_166546_3081-166546-0000
3752_4943_3752-4943-0000
...
8842_304647_8842-304647-0013
```

----
## Feature Extraction Pipeline

### HuBERT Architecture Coverage

#### CNN Features (8 total)
- `conv_0` through `conv_6`: Progressive downsampling with stride reductions
- `cnn_final_raw`: Final CNN output (same as conv_6)
- **Time Progression**: 53,120 samples → 166 time steps (for 3.32s audio)
- **Total Stride**: 320 (5 × 2^6)
- **Channel Dimension**: 512 channels throughout CNN layers

#### Projection Features (2 total)
- `after_transpose`: CNN output reshaped for transformer (batch, time, 512)
- `after_projection`: Linear projection 512→768 dimensions

#### Transformer Features (13 total)
- `transformer_input`: Input to first transformer layer (post-projection)
- `transformer_layer_0` through `transformer_layer_11`: All 12 transformer layer outputs
- **Dimensions**: 768 features with 12 attention heads per layer
- **Architecture**: Multi-Head Attention → Feed Forward (768→3072→768)

### Feature File Structure (NPZ Format)
Each `.npz` file contains all features for a single audio file:
```python
# Load features
data = np.load('output/hubert_complete/librispeech_dev-clean_sample1/251_118436_251-118436-0014.npz')

# Available features (23 total):
# CNN Features (8):
data['conv_0']        # Shape: (1, 512, time_0)
data['conv_1']        # Shape: (1, 512, time_1)
# ... conv_2 through conv_6
data['cnn_final_raw'] # Shape: (1, 512, time_final)

# Projection Features (2):
data['after_transpose']   # Shape: (1, time_final, 512)
data['after_projection']  # Shape: (1, time_final, 768)

# Transformer Features (13):
data['transformer_input']    # Shape: (1, time_final, 768)
data['transformer_layer_0']  # Shape: (1, time_final, 768)
# ... transformer_layer_1 through transformer_layer_11
```

### Feature Consistency Verification
Built-in verification system ensures identical features across processing methods:
- **Shape Validation**: Checks expected dimensions for each feature type
- **Time Step Verification**: Validates CNN stride calculations
- **Memory Consistency**: Ensures proper data type and memory layout


## Usage Examples

### Single GPU Processing
```bash
python extract/extract_features-HuBERT.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --save_dir /exp/sholzinger/output \
    --feature_type all \
    --dataset_split dev-clean \
    --single_gpu
```

### Multi-GPU Processing (Default)
```bash
python extract/extract_features-HuBERT.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --save_dir /exp/sholzinger/output \
    --feature_type all \
    --dataset_split dev-clean
```

### Available Feature Types
- `cnn`: Extract only CNN layer features (8 features)
- `projection`: Extract only projection features (2 features)
- `transformer`: Extract only transformer features (13 features)
- `all`: Extract all features (23 features total - default)

### Multi-GPU Parallel Processing
- **Enhanced Performance**: Added support for multi-GPU parallel processing across V100 GPUs
- **Feature Consistency**: Implemented deterministic processing to ensure identical features regardless of processing method
- **Error Handling**: Robust error handling with detailed logging and result aggregation
- **Verification System**: Optional hash-based verification of feature consistency between single and multi-GPU processing

#### Hardware Requirements
- **GPUs**: V100 or similar (16GB+ VRAM recommended)
- **RAM**: 128GB+ for large-scale processing
- **Storage**: Fast I/O for audio file access and feature storage

### Debug Configuration
Updated `.vscode/launch.json` with:
- Configurations for all main analysis scripts
- Remote debugging support for SLURM allocated resources
- Proper PYTHONPATH and environment setup

### Key Technical Improvements
1. **Process Isolation**: Each GPU loads its own model instance to avoid CUDA context conflicts
2. **Deterministic Operations**: Set `torch.backends.cudnn.deterministic = True` for reproducible results
3. **Memory Management**: Efficient memory handling with proper `.cpu()` calls and cleanup
4. **Load Balancing**: Automatic distribution of audio files across available GPUs

----
## Analysis Pipeline

### 1. Similarity Analysis
```bash
python analysis/similarity_analysis.py
```
- Layer-wise similarity metrics (CKA, correlation)
- Cross-layer comparisons
- Temporal dynamics analysis

### 2. Temporal Analysis
```bash
python analysis/temporal_analysis.py
```
- Feature evolution over time
- Animation generation
- Temporal correlation patterns

### 3. Visualization Pipeline
```bash
python main_layerwise_analysis.py
```
- Clean visualization pipeline
- Multiple plot types and formats
- Configurable analysis parameters

----
## Debugging & Development

### Remote Debugging Setup
1. Allocate SLURM resources:
   ```bash
   salloc -p gpu --mem 4G -c 2 --gres=gpu:v100:1 -t 04:00:00
   ```

2. Start debugpy server on allocated node:
   ```bash
   ./layerwise-analysis-cca-vis/e2e-layerwise/bin/python \
   -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
   ./layerwise-analysis-cca-vis/main_layerwise_analysis.py
   ```

3. Connect from VS Code using "Remote Debug (rack7n06)" configuration

### Available Debug Configurations
- **Python: Current File**: Debug any currently open file
- **Run Clean Analysis**: Debug main analysis pipeline
- **Visualize Features**: Debug visualization pipeline
- **Similarity Analysis**: Debug similarity analysis module
- **Temporal Analysis**: Debug temporal analysis module
- **Remote Debug (rack7n06)**: Connect to remote debugpy server

### Utility Scripts
- **`main_extract_features_chunked.py`**: Process large datasets in manageable chunks
- **`main_layerwise_analysis_examples.py`**: Contains example usage patterns (demo-like)
- **`debug.txt`**: Contains debugging commands and troubleshooting notes

## Output Structure

```
output/
├── hubert_complete/
│   └── librispeech_dev-clean_sample1/        # 1,655 processed files
│       ├── 251_118436_251-118436-0014.npz   # 23 features per file
│       ├── 251_136532_251-136532-0000.npz
│       ├── 3081_166546_3081-166546-0000.npz
│       └── ... (1,652 more files)
└── clean_analysis/                           # Analysis outputs
    ├── similarity_matrices/
    ├── temporal_analysis/
    └── visualizations/
```

## Dependencies

```bash
# Core dependencies
pip install torch torchaudio transformers numpy tqdm pathlib

# Additional utilities
pip install debugpy  # For remote debugging
```

For complete environment setup:
```bash
# Activate environment
source e2e-layerwise/bin/activate

# All dependencies should already be installed in the environment
```
----
## Adding Future Contributions

### Current Main Pipeline Files
1. Feature Extraction
File: main_extract_features_chunked.py (/home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/extract/main_extract_features_chunked.py)
Purpose: Chunked HuBERT feature extraction script for large datasets

2. Complete Analysis Pipeline
File: main_layerwise_analysis.py
Purpose: Primary analysis pipeline with multiple analysis types (similarity, temporal, conditional)

3. GPU-Accelerated Layer Analysis
File: main_gpu_analysis.py
Purpose: GPU-accelerated similarity computation between layers

4. GPU-Accelerated Temporal Analysis
File: main_gpu_temporal_analysis.py
Purpose: GPU-accelerated temporal dynamics analysis with sliding windows

5. Example Runner/Demo
File: main_layerwise_analysis_examples.py
Purpose: Example script demonstrating various analysis patterns

MAIN ENTRY POINTS:
├── extract/
│   └── main_extract_features_chunked.py     # 1. Extract features from audio
├── main_layerwise_analysis.py               # 2. Primary analysis pipeline  
├── main_gpu_analysis.py                     # 3. GPU similarity analysis
├── main_gpu_temporal_analysis.py            # 4. GPU temporal analysis
└── main_layerwise_analysis_examples.py      # 5. Usage examples

SUPPORT MODULES:
├── _utils/          # Utility functions
├── _analysis/       # Analysis modules  
└── tools/          # Setup/utility scripts

When adding new features or analysis methods:
1. Follow the existing project structure under `_utils/`, `_analysis/`, or `extract/`
2. Add appropriate debug configurations to `.vscode/launch.json`
3. Include comprehensive error handling and logging
4. Test both single and multi-GPU processing modes
5. Update this documentation with new capabilities
6. Ensure NPZ format compatibility for feature files

----
## Recent Project Restructuring (June 2025)

# End-to-End Layerwise Analysis Pipeline

## Overview

This pipeline extracts and analyzes layerwise features from HuBERT models on LibriSpeech audio data (dev-clean).
It provides an initial analysis of neural speech representations across all model layers.


## Project Structure

```
layerwise-analysis-cca-vis/
├── utils/
│   ├── data_utils.py          # Data loading & preprocessing (with segmentation option!)
│   ├── math_utils.py          # Mathematical computations (CKA, correlation, etc.)
│   └── visualization_utils.py # Basic plotting utilities
├── analysis/
│   ├── similarity_analysis.py # Layer similarity analysis
│   └── temporal_analysis.py   # Temporal dynamics & animations
├── extract/
│   └── extract_features-HuBERT.py # Multi-GPU feature extraction
│   └── convert_flac_to_wav.sh     # Convert FLAC to WAV
├── run_extract_features_chunked.py # Chunked processing for large datasets
├── visualize_features_clean.py    # Clean main pipeline
├── run_clean_analysis.py          # Usage examples
└── .vscode/launch.json            # Debugging configurations
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

----
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

### Limit Number of GPUs
```bash
python extract/extract_features-HuBERT.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --save_dir /exp/sholzinger/output \
    --feature_type all \
    --dataset_split dev-clean \
    --max_gpus 2
```

### Chunked Processing for Large Datasets
```bash
python run_extract_features_chunked.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --save_dir /exp/sholzinger/output \
    --chunk_size 100 \
    --feature_type all
```

### Performance Benchmarking
```bash
python benchmark_audio_loading.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --num_files 50 \
    --iterations 5
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
- Remote debugging support for SLURM allocated resources (rack7n06:5678)
- Proper PYTHONPATH and environment setup

### Key Technical Improvements
1. **Process Isolation**: Each GPU loads its own model instance to avoid CUDA context conflicts
2. **Deterministic Operations**: Set `torch.backends.cudnn.deterministic = True` for reproducible results
3. **Memory Management**: Efficient memory handling with proper `.cpu()` calls and cleanup
4. **Load Balancing**: Automatic distribution of audio files across available GPUs


#### Processing Performance
- **Single File**: ~2-5 seconds per file (varies with audio duration)
- **Multi-GPU Speedup**: Near-linear scaling (4 GPUs ≈ 4x speedup)
- **Memory Usage**: ~8GB VRAM per GPU for model + processing
- **1,655 Files Processed**: Successfully completed on dev-clean sample

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
python visualize_features_clean.py
```
- Clean visualization pipeline
- Multiple plot types and formats
- Configurable analysis parameters

----
## Debugging & Development

### Remote Debugging Setup
1. Allocate SLURM resources:
   ```bash
   salloc --gres=gpu:1 --time=4:00:00
   ```

2. Start debugpy server on allocated node:
   ```bash
   /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/e2e-layerwise/bin/python \
   -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
   /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/visualize_features_clean.py
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
- **`benchmark_audio_loading.py`**: Compare FLAC vs WAV loading performance
- **`run_extract_features_chunked.py`**: Process large datasets in manageable chunks
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

## Quality Assurance & Error Handling

### Feature Consistency Verification
- **Shape Validation**: Automatic verification of feature dimensions
- **Time Step Calculation**: Validates CNN stride computations (audio_length / 320)
- **Memory Layout**: Ensures consistent numpy array formats
- **Processing Resume**: Automatically skips already processed files

### Error Recovery Features
- **Per-File Error Logging**: Individual file failures don't stop processing
- **GPU-Specific Tracking**: Separate error handling per GPU process
- **Progress Monitoring**: Regular progress updates and completion summaries
- **Memory Management**: Aggressive GPU cache clearing and garbage collection

### Recent Improvements
1. **Multi-GPU Parallelization**: Complete rewrite for efficient parallel processing
2. **Feature Verification**: Added consistency checks for all feature types
3. **File Naming**: Safe filename generation preserving directory structure
4. **Debug Integration**: Enhanced VS Code debugging support with remote capabilities

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

The following changes have been implemented and committed (Note these names below were not changed in the refactoring. Actual names for main scripts are outlined in the MAIN ENTRY point -section above- ):

#### **Project Structure Update (New)**
```
layerwise-analysis-cca-vis/
├── _utils/                    # Reorganized utilities (underscore prefix)
│   ├── __init__.py
│   ├── data_utils.py          # Data loading & preprocessing (323 lines)
│   ├── math_utils.py          # Mathematical computations (1,508 lines)
│   └── visualization_utils.py # Plotting utilities (483 lines)
├── _analysis/                 # Reorganized analysis modules
│   ├── __init__.py
│   ├── similarity_analysis.py # Layer similarity analysis (741 lines)
│   └── temporal_analysis.py   # Temporal dynamics & animations (508 lines)
├── _slurm/                    # Reorganized SLURM job scripts
│   ├── extract_hubert_features.slurm (55 lines)
│   └── run_visualize_features.slurm (31 lines)
├── extract/
├── gpu_layer_analysis.py      # Updated with new imports and metric consistency
├── gpu_temporal_layer_analysis.py # Updated temporal analysis
├── visualize_features_clean.py    # Major enhancement with GPU acceleration
├── run_clean_analysis.py          # Enhanced with new analysis examples
└── .vscode/launch.json            # Updated debugging configurations
```

#### **Commit History Summary (13 commits)**

##### **Documentation Enhancement (Commit 1)**
**Hash**: `28754ca`
- **Changes**: 506 insertions, 2 deletions in markdown files
- **Details**: Comprehensive update to `e2e-layerwise-analysis-correlations.md` with:
  - Detailed mathematical explanations for three analysis approaches
  - Temporal analysis implementation details and windowing logic
  - Complete similarity metrics status documentation
  - Multi-GPU support documentation with usage examples
  - Performance considerations and fallback scenarios

##### **Directory Structure Cleanup (Commit 2)**
**Hash**: `02d6042`
- **Changes**: 1,648 deletions (9 files)
- **Details**: Removed old directory structure files:
  - Deleted `analysis/`, `utils/`, and `slurm/` directories
  - Prepared for reorganization with underscore-prefixed directories

##### **New Utilities Package (_utils/) (Commits 3-6)**

**Commit 3** - `_utils/__init__.py` (Hash: `c4071d0`)
- Initialize new utils package structure

**Commit 4** - `_utils/data_utils.py` (Hash: `b9aec95`)
- 323 lines of dataset handling, feature extraction, and batching functions

**Commit 5** - `_utils/math_utils.py` (Hash: `e4701b7`)
- **1,508 lines** of comprehensive mathematical utilities
- **ACTIVE metrics**: partial correlation (CPU/GPU), input-layer correlations, progressive partial correlations, robust CPU fallbacks
- **DISABLED metrics**: conditional CKA (regression issues), layer-to-layer correlations (refactoring), cosine similarity (not significant)
- Multi-GPU acceleration support with automatic fallbacks

**Commit 6** - `_utils/visualization_utils.py` (Hash: `8ebb46f`)
- 483 lines of heatmap generation, similarity matrix plotting, temporal animation creation

##### **New Analysis Package (_analysis/) (Commits 7-9)**

**Commit 7** - `_analysis/__init__.py` (Hash: `44db17c`)
- Initialize new analysis package structure

**Commit 8** - `_analysis/similarity_analysis.py` (Hash: `03510ee`)
- **741 lines** of layer-to-layer similarity computation
- Input propagation analysis and orchestration of correlation, CKA, and partial correlation metrics
- Excludes commented functions: conditional_cka, cosine_similarity variants, layer_to_layer_correlations

**Commit 9** - `_analysis/temporal_analysis.py` (Hash: `6d013f8`)
- **508 lines** of sliding window similarity computation
- **Padding-aware** temporal dynamics with corrected time steps dimension handling
- Conditional temporal analysis controlling for CNN output
- **ACTIVE metrics**: correlation, cka, partial_correlation
- **DISABLED**: conditional_cka (regression issues), cosine similarity (refactoring)

##### **SLURM Scripts Reorganization (Commit 10)**
**Hash**: `c875b2b`
- **Changes**: 85 insertions (2 files)
- Reorganized SLURM job scripts in `_slurm/` directory

##### **Core Analysis Updates (Commits 11-13)**

**Commit 11** - `gpu_layer_analysis.py` (Hash: `14b0be5`)
- **Changes**: 226 insertions, 160 deletions
- Fixed import paths to use `_utils` and `_analysis` packages
- Updated default metrics to consistent `['correlation', 'cka']` list
- Fixed inconsistency between listed and implemented metrics

**Commit 12** - `gpu_temporal_layer_analysis.py` (Hash: `5e0c6b7`)
- **Changes**: 98 insertions, 54 deletions
- Import path fixes and metric consistency updates

**Commit 13** - `run_clean_analysis.py` (Hash: `a5439f0`)
- **Changes**: 111 insertions, 13 deletions
- Fixed absolute paths to relative paths
- **Added 4 new analysis examples**:
  - `example_input_propagation_analysis()` - GPU-accelerated input propagation
  - `example_all_correlations()` - Full GPU + CPU parallel analysis
  - `example_performance_benchmark()` - GPU vs CPU performance comparison
  - `example_cpu_only_comparison()` - CPU-only processing
- **New CLI flags**: `--use_gpu`, `--no_gpu`, `--include_input_propagation`, `--n_jobs`

##### **Major Feature Enhancement (Commit 14)**

**Commit 14** - `visualize_features_clean.py` (Hash: `5141f01`)
- **Changes**: 228 insertions, 76 deletions
- **MAJOR ENHANCEMENT** with comprehensive new features:
  - **GPU acceleration support** with CUDA detection and memory reporting
  - **Input propagation analysis** with `CorrelationAnalyzer`
  - **Enhanced similarity analysis** with new correlation types
  - **Performance monitoring** and progress reporting
  - **New CLI flags**: `--use_gpu`, `--no_gpu`, `--include_input_propagation`, `--n_jobs`
  - Refactored CNN influence analysis with better error handling
  - Updated author information (Sandra Arcos Holzinger, June 7, 2025)

##### **Configuration Update (Commit 15)**
**Hash**: `937f82b`
- **Changes**: 24 insertions, 50 deletions
- Updated VS Code debug configuration for new project structure

### **Key Improvements Achieved**

#### **1. Similarity Metrics Consistency**
- **Fixed inconsistency** between listed metrics and actual implementations
- **Active metrics**: `['correlation', 'cka']` in all GPU analysis functions
- **Documented status** of disabled metrics with reasons

#### **2. Enhanced Mathematical Capabilities**
- **Multi-GPU acceleration** for input-layer correlations and progressive partial correlations
- **Robust fallback system**: GPU → Single GPU → CPU
- **Three analysis types**: layer-to-layer, input-layer, progressive partial correlations
- **Performance optimization** with parallel processing

#### **3. Temporal Analysis Improvements**
- **Padding-aware computation** with corrected dimension handling
- **Temporal windowing** fixed to use time steps dimension correctly
- **Conditional analysis** controlling for CNN output
- **Animation support** for temporal evolution visualization

#### **4. Development Experience**
- **Organized code structure** with underscore-prefixed packages
- **Enhanced debugging** configurations for new structure
- **Better error handling** and performance monitoring
- **Comprehensive documentation** with usage examples

#### **5. Performance & Usability**
- **GPU acceleration** with automatic CUDA detection
- **CPU parallelization** with configurable job counts
- **Progress reporting** and performance benchmarking
- **Multiple analysis workflows** with example scripts

### **Current Similarity Metrics Status**

#### **ACTIVE & WORKING**
1. **`correlation`** - Pearson correlation between layers
2. **`cka`** - Centered Kernel Alignment with padding support
3. **`partial_correlation`** - Partial correlation controlling for CNN output
4. **`input_layer_correlations`** - Direct correlation between input and layers
5. **`progressive_partial_correlations`** - Progressive partial correlations showing new information per layer
6. **`r_squared`** - Variance explanation analysis

#### **DISABLED/COMMENTED OUT**
1. ~~`conditional_cka`~~ - CKA controlling for variables (regression step issues)
2. ~~`cosine_similarity`~~ - Cosine similarity variants (not significant for analysis)
3. ~~`layer_to_layer_correlations`~~ - Layer-to-layer analysis (refactoring in progress)

### **Usage Examples with New Structure**

#### **GPU-Accelerated Input Propagation Analysis**
```bash
python visualize_features_clean.py \
    --features_dir ./output/hubert_complete/librispeech_dev-clean_sample1 \
    --output_dir ./output/clean_analysis/input_propagation \
    --model_name HuBERT_Input_Propagation \
    --num_files 5 \
    --include_input_propagation \
    --use_gpu \
    --n_jobs -1
```

#### **Performance Benchmark (GPU vs CPU)**
```bash
python run_clean_analysis.py  # Runs example_performance_benchmark()
```

#### **All Correlation Types Analysis**
```bash
python visualize_features_clean.py \
    --features_dir ./output/hubert_complete/librispeech_dev-clean_sample1 \
    --output_dir ./output/clean_analysis/all_correlations \
    --include_conditional \
    --include_input_propagation \
    --use_gpu
```
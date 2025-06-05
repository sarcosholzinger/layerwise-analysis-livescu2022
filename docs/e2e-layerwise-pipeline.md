# End-to-End Layerwise Analysis Pipeline

## Overview

This pipeline extracts and analyzes layerwise features from HuBERT models on LibriSpeech audio data (dev-clean).
It provides an initial analysis of neural speech representations across all model layers.
Some of the dev-clean audio has also been converted from FLAC to WAV (speaker ids 422, 251, 174, 84, 2412). These have been chosend arbitrarily only to test the scripts and general pipeline.

## LibriSpeech Dataset

### Description
LibriSpeech is a large corpus of English speech derived from audiobooks that are part of the LibriVox project. It contains approximately 1,000 hours of speech data sampled at 16kHz.

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
├── visualize_features_clean.py    # Clean main pipeline
├── run_clean_analysis.py          # Usage examples
└── .vscode/launch.json            # Debugging configurations
```


### Usage Examples

#### Single GPU Processing
```bash
python extract/extract_features-HuBERT.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --save_dir /exp/sholzinger/output \
    --feature_type all \
    --dataset_split dev-clean
```

#### Multi-GPU Processing
```bash
python extract/extract_features-HuBERT.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --save_dir /exp/sholzinger/output \
    --feature_type all \
    --dataset_split dev-clean \
    --use_multi_gpu
```

#### With Consistency Verification
```bash
python extract/extract_features-HuBERT.py \
    --audio_dir /exp/sholzinger/LibriSpeech/dev-clean \
    --save_dir /exp/sholzinger/output \
    --feature_type all \
    --dataset_split dev-clean \
    --use_multi_gpu \
    --verify_consistency
```

### Feature Types Available
- `cnn`: Extract only CNN layer features
- `projection`: Extract only projection layer features  
- `transformer`: Extract only transformer layer features
- `all`: Extract all features (default)

----
## Feature Extraction Pipeline

### HuBERT Architecture Coverage

#### CNN Features (7 layers)
- `conv_0` through `conv_6`: Progressive downsampling from audio to 512-dimensional features
- Total stride: 320 (5 × 2^6)
- Time resolution: ~20ms per final timestep

#### Projection Features
- `after_transpose`: CNN output reshaped for transformer processing
- `after_projection`: Linear projection from 512 to 768 dimensions

#### Transformer Features (13 layers)
- `transformer_input`: Input to first transformer layer
- `transformer_layer_0` through `transformer_layer_11`: All 12 transformer layer outputs
- Each layer: 768-dimensional features with 12 attention heads

## Recent Changes & Improvements

### Multi-GPU Parallel Processing
- **Enhanced Performance**: Added support for multi-GPU parallel processing across V100 GPUs
- **Feature Consistency**: Implemented deterministic processing to ensure identical features regardless of processing method
- **Error Handling**: Robust error handling with detailed logging and result aggregation
- **Verification System**: Optional hash-based verification of feature consistency between single and multi-GPU processing

### Key Technical Improvements
1. **Process Isolation**: Each GPU loads its own model instance to avoid CUDA context conflicts
2. **Deterministic Operations**: Set `torch.backends.cudnn.deterministic = True` for reproducible results
3. **Memory Management**: Efficient memory handling with proper `.cpu()` calls and cleanup
4. **Load Balancing**: Automatic distribution of audio files across available GPUs

### Debug Configuration
Updated `.vscode/launch.json` with:
- Configurations for all main analysis scripts
- Remote debugging support for SLURM allocated resources (rack7n06:5678)
- Proper PYTHONPATH and environment setup

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
   /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/run_clean_analysis.py
   ```

3. Connect from VS Code using "Remote Debug (rack7n06)" configuration

### Available Debug Configurations
- **Python: Current File**: Debug any currently open file
- **Run Clean Analysis**: Debug main analysis pipeline
- **Visualize Features**: Debug visualization pipeline
- **Similarity Analysis**: Debug similarity analysis module
- **Temporal Analysis**: Debug temporal analysis module
- **Remote Debug (rack7n06)**: Connect to remote debugpy server

## Performance Specifications

### Hardware Requirements
- **GPUs**: V100 or similar (16GB+ VRAM recommended)
- **RAM**: 128GB+ for large-scale processing
- **Storage**: Fast I/O for audio file access

### Expected Processing Times
- **Single Audio File**: ~2-5 seconds per file (depending on duration)
- **1000 Files**: ~1-2 hours on single V100
- **Multi-GPU Speedup**: Near-linear scaling with number of GPUs

## Output Structure

```
output/
├── hubert_complete/
│   ├── librispeech_dev-clean_sample1/
│   │   ├── audio_file_1_features.pkl
│   │   ├── audio_file_2_features.pkl
│   │   └── ...
│   ├── gpu_0_results.pkl
│   ├── gpu_1_results.pkl
│   └── final_results.pkl
```

### Feature File Format
Each `.pkl` file contains a dictionary with:
```python
{
    'conv_0': numpy.ndarray,           # Shape: (1, 512, time_steps)
    'conv_1': numpy.ndarray,           # Shape: (1, 512, time_steps)
    # ... all CNN layers
    'after_transpose': numpy.ndarray,   # Shape: (1, time_steps, 512)
    'after_projection': numpy.ndarray,  # Shape: (1, time_steps, 768)
    'transformer_input': numpy.ndarray, # Shape: (1, time_steps, 768)
    'transformer_layer_0': numpy.ndarray, # Shape: (1, time_steps, 768)
    # ... all transformer layers
}
```

## Verification & Quality Assurance

### Feature Consistency Checks
- Numerical verification between single and multi-GPU processing
- Hash-based feature verification
- Shape and dimension validation
- Time step calculation verification

### Error Handling
- Per-file error logging
- GPU-specific error tracking
- Graceful failure handling
- Result aggregation and summary

## Contributing

When adding new features or analysis methods:
1. Follow the existing project structure
2. Add appropriate debug configurations to `.vscode/launch.json`
3. Include error handling and logging
4. Test both single and multi-GPU processing modes
5. Update this documentation

## Dependencies

```bash
pip install torch torchaudio transformers numpy tqdm pathlib pickle hashlib
```

For the complete environment, use:
```bash
pip install -r requirements.txt  # If available
``` 
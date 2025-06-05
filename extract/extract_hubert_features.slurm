#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=logs/extract_features_%j.log
#SBATCH --error=logs/extract_features_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# # Activate conda environment
# source /opt/slurm_files/software/conda24/bin/activate e2e-layerwise

# Activate uv environment
source ./e2e-layerwise/bin/activate



# FEATURE_TYPE="local" bash extract_hubert_features.sh  # For CNN features only
# FEATURE_TYPE="contextualized" bash extract_hubert_features.sh  # For transformer features only
# FEATURE_TYPE="both" bash extract_hubert_features.sh  # For both types

# Default parameters
MODEL_NAME="facebook/hubert-base-ls960"
DATA_SAMPLE=1
FEATURE_TYPE="all"  # Options: "all", "cnn", "transformer"
SPAN="frame"
SUBSET_ID=0
DATASET_SPLIT="dev-clean"
SAVE_DIR="./output"
AUDIO_DIR="./content/data/LibriSpeech/dev-clean"

# Optional GPU control parameters
MAX_GPUS=4  # Use 4 GPUs for parallel processing
# SINGLE_GPU="--single_gpu"  # Commented out to enable multi-GPU

# Create output directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Run feature extraction
python ./extract/extract_features-HuBERT.py \
    --model_name "$MODEL_NAME" \
    --data_sample "$DATA_SAMPLE" \
    --feature_type "$FEATURE_TYPE" \
    --span "$SPAN" \
    --subset_id "$SUBSET_ID" \
    --dataset_split "$DATASET_SPLIT" \
    --save_dir "$SAVE_DIR" \
    --audio_dir "$AUDIO_DIR" \
    --max_gpus "$MAX_GPUS"


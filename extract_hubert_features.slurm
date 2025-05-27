#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --output=logs/extract_features_%j.log
#SBATCH --error=logs/extract_features_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu #gpu-a100 #queue

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /opt/slurm_files/software/conda24/bin/activate e2e-layerwise

# Default parameters
MODEL_NAME="facebook/hubert-base-ls960"
DATA_SAMPLE=1
FEATURE_TYPE="both"  # Options: "local", "contextualized", or "both"
SPAN="frame"
SUBSET_ID=0
DATASET_SPLIT="dev-clean"
SAVE_DIR="/home/sarcosh1/repos/layerwise-analysis/output"
AUDIO_DIR="/home/sarcosh1/repos/layerwise-analysis/content/data/LibriSpeech/dev-clean-2"

# Create output directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Run feature extraction
python extract_features-HuBERT.py \
    --model_name "$MODEL_NAME" \
    --data_sample "$DATA_SAMPLE" \
    --feature_type "$FEATURE_TYPE" \
    --span "$SPAN" \
    --subset_id "$SUBSET_ID" \
    --dataset_split "$DATASET_SPLIT" \
    --save_dir "$SAVE_DIR" \
    --audio_dir "$AUDIO_DIR"


# FEATURE_TYPE="local" bash extract_hubert_features.sh  # For CNN features only
# FEATURE_TYPE="contextualized" bash extract_hubert_features.sh  # For transformer features only
# FEATURE_TYPE="both" bash extract_hubert_features.sh  # For both types
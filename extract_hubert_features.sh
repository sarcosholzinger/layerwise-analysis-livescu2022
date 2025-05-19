#!/bin/bash

# Activate conda environment
source /opt/slurm_files/software/conda24/bin/activate e2e-layerwise

# Default parameters
MODEL_NAME="facebook/hubert-base-ls960"
DATA_SAMPLE=1
REP_TYPE="contextualized"
SPAN="frame"
SUBSET_ID=0
DATASET_SPLIT="dev-clean"
SAVE_DIR="/home/sarcosh1/repos/layerwise-analysis/output"
AUDIO_DIR="/home/sarcosh1/repos/layerwise-analysis/content/data/LibriSpeech/dev-clean-2"

# Create output directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Run feature extraction
python extract_features.py \
    --model_name "$MODEL_NAME" \
    --data_sample "$DATA_SAMPLE" \
    --rep_type "$REP_TYPE" \
    --span "$SPAN" \
    --subset_id "$SUBSET_ID" \
    --dataset_split "$DATASET_SPLIT" \
    --save_dir "$SAVE_DIR" \
    --audio_dir "$AUDIO_DIR" 
#!/bin/bash
#SBATCH --job-name=visualize_features
#SBATCH --output=logs/visualize_features_%j.log
#SBATCH --error=logs/visualize_features_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /opt/slurm_files/software/conda24/bin/activate e2e-layerwise

# Default parameters
FEATURES_DIR="/home/sarcosh1/repos/layerwise-analysis/output/hubert_complete/librispeech_dev-clean_sample1"
OUTPUT_DIR="/home/sarcosh1/repos/layerwise-analysis/output/visualizations"
NUM_FILES=10
MODEL_NAME="HuBERT Base"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run visualization
python visualize_features.py \
    --features_dir "$FEATURES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_files "$NUM_FILES" \
    --model_name "$MODEL_NAME" 
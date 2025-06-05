#!/bin/bash
#SBATCH --job-name=visualize_features
#SBATCH --output=logs/visualize_features_%j.log
#SBATCH --error=logs/visualize_features_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu #gpu-a100 #queue

# Activate conda environment
source /opt/slurm_files/software/conda24/bin/activate e2e-layerwise

# Create logs directory if it doesn't exist
mkdir -p logs

# Set parameters
FEATURES_DIR="/home/sarcosh1/repos/layerwise-analysis/output/hubert_complete/librispeech_dev-clean_sample1"
OUTPUT_DIR="/home/sarcosh1/repos/layerwise-analysis/output/visualizations"
MODEL_NAME="facebook_hubert-base-ls960"
NUM_FILES=10

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run visualization script
python visualize_features.py \
    --features_dir "$FEATURES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_files "$NUM_FILES" 
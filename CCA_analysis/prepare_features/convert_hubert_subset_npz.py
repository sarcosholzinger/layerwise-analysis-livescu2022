#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm
import glob
import random

def convert_npz_to_npy_subset(
    input_dir, 
    output_dir, 
    num_layers=12,
    num_files=5,  # Number of NPZ files to process
    random_seed=42
):
    """
    Convert a subset of HuBERT NPZ files to layer-wise NPY format.
    This script processes only a subset of NPZ files while keeping all features.
    
    Args:
        input_dir (str): Directory containing NPZ files
        output_dir (str): Directory to save NPY files
        num_layers (int): Number of layers in the model
        num_files (int): Number of NPZ files to process
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all NPZ files
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    print(f"Found {len(npz_files)} NPZ files")
    
    # Select subset of files
    if len(npz_files) > num_files:
        npz_files = random.sample(npz_files, num_files)
    print(f"Processing {len(npz_files)} files")
    
    # Initialize arrays for each layer
    layer_arrays = {i: [] for i in range(num_layers)}
    
    # Process each NPZ file
    for npz_file in tqdm(npz_files, desc="Converting NPZ files"):
        try:
            # Load NPZ file
            data = np.load(npz_file)
            
            # Process each transformer layer
            for layer in range(num_layers):
                layer_key = f'transformer_layer_{layer}'
                if layer_key in data:
                    # Get features for this layer [1, time_steps, features]
                    features = data[layer_key]
                    # Remove batch dimension
                    features = features[0, :, :]
                    layer_arrays[layer].append(features)
                else:
                    print(f"Warning: {layer_key} not found in {npz_file}")
                
        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
            continue
    
    # Concatenate and save each layer
    print("\nSaving layer-wise NPY files...")
    for layer in range(num_layers):
        if layer_arrays[layer]:
            # Concatenate all time steps for this layer
            layer_data = np.concatenate(layer_arrays[layer], axis=0)
            
            # Save as NPY
            output_file = os.path.join(output_dir, f"layer_{layer}.npy")
            np.save(output_file, layer_data)
            print(f"Saved layer {layer} with shape {layer_data.shape}")
        else:
            print(f"Warning: No data for layer {layer}")

if __name__ == "__main__":
    # Example usage
    input_dir = "/home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/output/hubert_complete/librispeech_dev-clean_sample1"
    output_dir = "/home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/CCA_analysis/prepare_features/converted_features/hubert_subset"
    
    convert_npz_to_npy_subset(
        input_dir=input_dir,
        output_dir=output_dir,
        num_files=10  # Process only 5 files
    ) 
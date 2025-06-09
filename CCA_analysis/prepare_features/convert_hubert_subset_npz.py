#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm
import glob
import random
import argparse

def convert_npz_to_npy_subset(
    input_dir, 
    output_dir, 
    num_layers=12,
    selection_mode="random",  # "random" or "speakers"
    num_files=30,  # Number of files to process in random mode
    target_speakers=[84, 174],  # Target speaker IDs for speaker mode
    random_seed=42
):
    """
    Convert HuBERT NPZ files to layer-wise NPY format.
    Supports both random selection and speaker-based selection.
    Saves both features and speaker IDs for each layer.
    
    Args:
        input_dir (str): Directory containing NPZ files
        output_dir (str): Directory to save NPY files
        num_layers (int): Number of layers in the model
        selection_mode (str): Either "random" or "speakers"
        num_files (int): Number of files to process in random mode
        target_speakers (list): List of speaker IDs to include in speaker mode
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all NPZ files
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    print(f"Found {len(npz_files)} total NPZ files")
    
    # Select files based on mode
    if selection_mode == "random":
        if len(npz_files) > num_files:
            selected_files = random.sample(npz_files, num_files)
        else:
            selected_files = npz_files
        print(f"Randomly selected {len(selected_files)} files")
    
    elif selection_mode == "speakers":
        selected_files = []
        for npz_file in npz_files:
            # Extract speaker ID from filename (assuming format: speakerID_*)
            try:
                speaker_id = int(os.path.basename(npz_file).split('_')[0])
                if speaker_id in target_speakers:
                    selected_files.append(npz_file)
            except (ValueError, IndexError):
                continue
        print(f"Found {len(selected_files)} files for speakers {target_speakers}")
    
    else:
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'random' or 'speakers'")
    
    # Initialize arrays for each layer
    layer_arrays = {i: [] for i in range(num_layers)}
    layer_speaker_ids = {i: [] for i in range(num_layers)}
    
    # Process each NPZ file
    for npz_file in tqdm(selected_files, desc="Converting NPZ files"):
        try:
            # Extract speaker ID from filename
            speaker_id = int(os.path.basename(npz_file).split('_')[0])
            
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
                    # Add speaker ID for each time step
                    layer_speaker_ids[layer].append(np.full(features.shape[0], speaker_id))
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
            speaker_ids = np.concatenate(layer_speaker_ids[layer], axis=0)
            
            # Save features and speaker IDs in separate files
            features_file = os.path.join(output_dir, f"layer_{layer}.npy")
            speaker_ids_file = os.path.join(output_dir, f"layer_{layer}_speaker_ids.npy")
            
            np.save(features_file, layer_data)
            np.save(speaker_ids_file, speaker_ids)
            
            print(f"Saved layer {layer} with features shape {layer_data.shape} and speaker IDs shape {speaker_ids.shape}")
        else:
            print(f"Warning: No data for layer {layer}")

def parse_args():
    parser = argparse.ArgumentParser(description='Convert HuBERT NPZ files to layer-wise NPY format')
    parser.add_argument('--mode', type=str, choices=['random', 'speakers'], default='random',
                      help='Selection mode: random or speakers')
    parser.add_argument('--num_files', type=int, default=30,
                      help='Number of files to process in random mode')
    parser.add_argument('--speakers', type=int, nargs='+', default=[84, 174],
                      help='Speaker IDs to include in speakers mode')
    parser.add_argument('--input_dir', type=str, 
                      default="/home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/output/hubert_complete/librispeech_dev-clean_sample1",
                      help='Directory containing NPZ files')
    parser.add_argument('--output_dir', type=str,
                      default="/home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/CCA_analysis/prepare_features/converted_features/hubert_subset/",
                      help='Directory to save NPY files')
    parser.add_argument('--num_layers', type=int, default=12,
                      help='Number of layers in the model')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    convert_npz_to_npy_subset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_layers=args.num_layers,
        selection_mode=args.mode,
        num_files=args.num_files,
        target_speakers=args.speakers,
        random_seed=args.seed
    ) 
    """
    Example usage:

    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py --mode speakers --speakers 84 174 123 --num_layers 12 
    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py --mode random --num_files 30 --num_layers 12 

    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py \
    --mode speakers \
    --speakers 84 174 123 \
    --num_layers 12 \
    --input_dir /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/output/hubert_complete/librispeech_dev-clean_sample1 \
    --output_dir /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/CCA_analysis/prepare_features/converted_features/hubert_subset \
    --seed 42
    ```
    bash
    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py \
    --mode random \
    --num_files 30 \
    --num_layers 12 \
    --input_dir /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/output/hubert_complete/librispeech_dev-clean_sample1 \
    --output_dir /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/CCA_analysis/prepare_features/converted_features/hubert_subset \
    --seed 42
    ```
    """
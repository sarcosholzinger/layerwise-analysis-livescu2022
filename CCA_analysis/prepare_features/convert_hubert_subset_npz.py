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
    random_seed=42,
    save_frame_level=True,  # Save concatenated frame-level data
    save_utterance_level=True  # Save individual utterance data
):
    """
    Convert HuBERT NPZ files to layer-wise NPY format.
    Supports both random selection and speaker-based selection.
    Saves both features and speaker IDs for each layer.
    Can save both frame-level (concatenated) and utterance-level (separate) data.
    
    Args:
        input_dir (str): Directory containing NPZ files
        output_dir (str): Directory to save NPY files
        num_layers (int): Number of layers in the model
        selection_mode (str): Either "random" or "speakers"
        num_files (int): Number of files to process in random mode
        target_speakers (list): List of speaker IDs to include in speaker mode
        random_seed (int): Random seed for reproducibility
        save_frame_level (bool): Whether to save frame-level data
        save_utterance_level (bool): Whether to save utterance-level data
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file for dimensions
    dim_log_file = os.path.join(output_dir, "feature_dimensions.txt")
    with open(dim_log_file, 'w') as f:
        f.write("File\tLayer\tShape\tFeature_Dim\n")
    
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
    
    # Initialize arrays for each layer (including CNN output)
    layer_arrays = {i: [] for i in range(num_layers + 1)}  # +1 for CNN output
    layer_speaker_ids = {i: [] for i in range(num_layers + 1)}
    
    # For utterance-level saving, also store metadata
    utterance_metadata = []  # Store (filename, speaker_id, layer_shapes)
    
    # Process each NPZ file
    for npz_file in tqdm(selected_files, desc="Converting NPZ files"):
        try:
            # Extract speaker ID from filename
            speaker_id = int(os.path.basename(npz_file).split('_')[0])
            
            # Load NPZ file
            data = np.load(npz_file)
            
            # Store utterance metadata
            filename = os.path.basename(npz_file)
            utt_shapes = {}
            
            # Log dimensions to file
            with open(dim_log_file, 'a') as f:
                # First handle CNN output if present
                if 'after_transpose' in data:
                    features = data['after_transpose']
                    features = features[0, :, :]  # Remove batch dimension
                    f.write(f"{filename}\tCNN\t{features.shape}\t{features.shape[1]}\n")
                    layer_arrays[0].append(features)
                    layer_speaker_ids[0].append(np.full(features.shape[0], speaker_id))
                    utt_shapes['CNN'] = features.shape
                    
                    # Save individual utterance if requested
                    if save_utterance_level:
                        utt_dir = os.path.join(output_dir, "utterances", filename.replace('.npz', ''))
                        os.makedirs(utt_dir, exist_ok=True)
                        np.save(os.path.join(utt_dir, "layer_0.npy"), features)
                        np.save(os.path.join(utt_dir, "layer_0_speaker_id.npy"), np.array([speaker_id]))
                
                # Then process transformer layers
                for layer in range(num_layers):
                    layer_key = f'transformer_layer_{layer}'
                    if layer_key in data:
                        features = data[layer_key]
                        features = features[0, :, :]  # Remove batch dimension
                        f.write(f"{filename}\t{layer}\t{features.shape}\t{features.shape[1]}\n")
                        layer_arrays[layer + 1].append(features)  # +1 because layer 0 is CNN
                        layer_speaker_ids[layer + 1].append(np.full(features.shape[0], speaker_id))
                        utt_shapes[f'layer_{layer}'] = features.shape
                        
                        # Save individual utterance if requested
                        if save_utterance_level:
                            utt_dir = os.path.join(output_dir, "utterances", filename.replace('.npz', ''))
                            os.makedirs(utt_dir, exist_ok=True)
                            np.save(os.path.join(utt_dir, f"layer_{layer + 1}.npy"), features)
                            np.save(os.path.join(utt_dir, f"layer_{layer + 1}_speaker_id.npy"), np.array([speaker_id]))
            
            # Store metadata for this utterance
            utterance_metadata.append({
                'filename': filename,
                'speaker_id': speaker_id,
                'shapes': utt_shapes
            })
            
        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
            continue
    print(f"Saved dimensions to {dim_log_file}")
    
    # Save frame-level data (concatenated across all utterances)
    if save_frame_level:
        print("\nSaving frame-level NPY files...")
        
        # Create the directory structure expected by get_scores.py
        frame_level_dir = os.path.join(output_dir, "contextualized", "frame_level")
        os.makedirs(frame_level_dir, exist_ok=True)
        
        for layer in range(num_layers + 1):
            if layer_arrays[layer]:
                try:
                    # Simple concatenation like the original paper - concatenate all frames from all utterances
                    layer_data = np.concatenate(layer_arrays[layer], axis=0)
                    speaker_ids_data = np.concatenate(layer_speaker_ids[layer], axis=0)
                    
                    # Save features and speaker IDs in the expected directory structure
                    features_file = os.path.join(frame_level_dir, f"layer_{layer}.npy")
                    speaker_ids_file = os.path.join(frame_level_dir, f"layer_{layer}_speaker_ids.npy")
                    
                    np.save(features_file, layer_data)
                    np.save(speaker_ids_file, speaker_ids_data)
                    
                    print(f"Saved frame-level layer {layer} with features shape {layer_data.shape} and speaker IDs shape {speaker_ids_data.shape}")
                except ValueError as e:
                    print(f"Error concatenating layer {layer}: {str(e)}")
                    print("This likely means different utterances have different feature dimensions.")
                    print("Consider using utterance-level saving instead.")
            else:
                print(f"Warning: No data for layer {layer}")
        
        print(f"Frame-level files saved in: {frame_level_dir}")
    
    # Save utterance metadata
    if save_utterance_level and utterance_metadata:
        import json
        metadata_file = os.path.join(output_dir, "utterance_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(utterance_metadata, f, indent=2)
        print(f"\nSaved utterance metadata to {metadata_file}")
        print(f"Individual utterances saved in {os.path.join(output_dir, 'utterances')}")
    
    print(f"\nProcessing complete!")
    print(f"Frame-level data: {'Saved' if save_frame_level else 'Skipped'}")
    print(f"Utterance-level data: {'Saved' if save_utterance_level else 'Skipped'}")
    print(f"Total utterances processed: {len(utterance_metadata)}")


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
    parser.add_argument('--save_frame_level', action='store_true', default=True,
                      help='Save frame-level (concatenated) data')
    parser.add_argument('--no_frame_level', dest='save_frame_level', action='store_false',
                      help='Skip saving frame-level data')
    parser.add_argument('--save_utterance_level', action='store_true', default=True,
                      help='Save utterance-level (individual) data')
    parser.add_argument('--no_utterance_level', dest='save_utterance_level', action='store_false',
                      help='Skip saving utterance-level data')
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
        random_seed=args.seed,
        save_frame_level=args.save_frame_level,
        save_utterance_level=args.save_utterance_level
    ) 
    """
    Example usage:

    # Save both frame-level and utterance-level data
    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py --mode speakers --speakers 84 174 123 --num_layers 12
    
    # Save only frame-level data (for CCA-intra and CCA-inter analysis)
    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py --mode speakers --speakers 84 174 123 --num_layers 12 --no_utterance_level
    
    # Save only utterance-level data (for individual analysis)
    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py --mode random --num_files 3 --num_layers 12 --no_frame_level --save_utterance_level 

    # Full command with all options
    python3 CCA_analysis/prepare_features/convert_hubert_subset_npz.py \
    --mode speakers \
    --speakers 84 174 123 \
    --num_layers 12 \
    --input_dir /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/output/hubert_complete/librispeech_dev-clean_sample1 \
    --output_dir /home/hltcoe/sholzinger/phd/git/layerwise-analysis-cca-vis/CCA_analysis/prepare_features/converted_features/hubert_subset \
    --seed 42
    
    Output structure:
    output_dir/ 
    ├── contextualized/frame_level/          # Frame-level data (for CCA analysis)
    │   ├── layer_0.npy                      # CNN output features
    │   ├── layer_1.npy                      # Transformer layer 0 features
    │   ├── ...
    │   ├── layer_12.npy                     # Transformer layer 11 features
    │   └── layer_*_speaker_ids.npy          # Speaker IDs for each layer
    ├── utterances/                          # Individual utterance data
    │   ├── filename1/
    │   │   ├── layer_0.npy
    │   │   └── layer_*_speaker_id.npy
    │   └── ...
    ├── feature_dimensions.txt               # Dimension analysis
    └── utterance_metadata.json             # Metadata for all utterances
  
    Example of feature dimensions:
    CONVOLUTIONAL LAYERS:
    conv_0: (1, 512, 23951)
    conv_1: (1, 512, 11975)
    conv_2: (1, 512, 5987)
    conv_3: (1, 512, 2993)
    conv_4: (1, 512, 1496)
    conv_5: (1, 512, 748)
    conv_6: (1, 512, 374)

    TRANSFORMER LAYERS:
    (1, 374, 768) is the number of frames in the utterance, 768 is the number of features per frame
    transformer_input: (1, 374, 768)
    transformer_layer_0: (1, 374, 768)
    transformer_layer_1: (1, 374, 768)
    transformer_layer_10: (1, 374, 768)
    transformer_layer_11: (1, 374, 768)
    transformer_layer_2: (1, 374, 768)
    transformer_layer_3: (1, 374, 768)
    transformer_layer_4: (1, 374, 768)
    transformer_layer_5: (1, 374, 768)
    transformer_layer_6: (1, 374, 768)
    transformer_layer_7: (1, 374, 768)
    transformer_layer_8: (1, 374, 768)
    transformer_layer_9: (1, 374, 768)
    
    OTHER KEYS:
    after_projection: (1, 374, 768)
    after_transpose: (1, 374, 512)
    cnn_final_raw: (1, 512, 374)
"""
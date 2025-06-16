#!/usr/bin/env python3

import os
import sys
import json
import glob
import numpy as np

# Add the codes directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
codes_dir = os.path.join(root_dir, 'codes', 'codes')
sys.path.append(codes_dir)

# Now import the CCA module
from tools.get_scores import evaluate_cca
from tools.tools_utils import LAYER_CNT

def detect_model_type(rep_dir):
    """
    Automatically detect model type based on the number of layer files and directory structure.
    
    Args:
        rep_dir (str): Directory containing model layer representations
        
    Returns:
        str: Detected model name that matches LAYER_CNT keys
    """
    # First try the standard structure: contextualized/frame_level
    ctx_dir = os.path.join(rep_dir, "contextualized", "frame_level")
    
    # If that doesn't exist, try the direct structure - this is for layerwise analysis (SAH)
    if not os.path.exists(ctx_dir):
        ctx_dir = rep_dir
    
    if not os.path.exists(ctx_dir):
        raise ValueError(f"Directory not found: {ctx_dir}")
    
    # Count layer files
    layer_files = glob.glob(os.path.join(ctx_dir, "layer_*.npy"))
    if not layer_files:
        raise ValueError(f"No layer files found in {ctx_dir}")
    
    # Extract layer numbers and find the maximum -- this is based on the assumption that files names correspond to the layers numbers   
    layer_nums = []
    for f in layer_files:
        try:
            layer_num = int(os.path.basename(f).replace("layer_", "").replace(".npy", ""))
            layer_nums.append(layer_num)
        except ValueError:
            continue
    
    if not layer_nums:
        raise ValueError("Could not parse layer numbers from files")
    
    max_layer = max(layer_nums)
    print(f"Detected {max_layer + 1} transformer layers (layers 0-{max_layer}) - based on file names at {rep_dir}")
    
    # Try to infer model type from directory name and layer count
    rep_dir_lower = rep_dir.lower()
    
    if "hubert" in rep_dir_lower:
        if max_layer >= 23:  # 24 layers (0-23)
            return "hubert_large"
        else:  # 12 layers (0-11)
            return "hubert_small"
    elif "wav2vec" in rep_dir_lower:
        if max_layer >= 23:  # 24 layers
            return "wav2vec_vox"
        else:  # 12 layers
            return "wav2vec_small"
    elif "wavlm" in rep_dir_lower:
        if max_layer >= 23:  # 24 layers
            return "wavlm_large"
        else:  # 12 layers
            return "wavlm_small"
    elif "avhubert" in rep_dir_lower:
        if max_layer >= 23:  # 24 layers
            return "avhubert_large_lrs3_vc2"
        else:  # 12 layers
            return "avhubert_small_lrs3"
    elif "xlsr" in rep_dir_lower:
        return "xlsr53_56"  # Both have 24 layers, default to this
    elif "fastvgs" in rep_dir_lower:
        if "plus" in rep_dir_lower:
            return "fastvgs_plus_coco"
        elif "places" in rep_dir_lower:
            return "fastvgs_places"
        else:
            return "fastvgs_coco"
    else:
        # Fallback: guess based on layer count alone
        if max_layer >= 23:  # 24 layers
            print("Warning: Could not identify model from directory name, defaulting to hubert_large")
            return "hubert_large"
        else:  # 12 layers
            print("Warning: Could not identify model from directory name, defaulting to hubert_small")
            return "hubert_small"

def run_cca_analysis(
    rep_dir="prepare_features/hubert_subset",
    model_name=None,  # Will be auto-detected if None
    save_fn="cca_results/cca_scores.json",
    base_layer=0,
    mean_score=False,
    eval_single_layer=False,
    layer_num=-1
):
    """
    Run pairwise CCA analysis between layers of the same model using only the features.
    
    Args:
        rep_dir (str): Directory containing model layer representations
        model_name (str, optional): Name of the model. If None, will be auto-detected from rep_dir
        save_fn (str): Path to save CCA scores
        base_layer (int): Reference layer for intra-model analysis
        mean_score (bool): Use mean CCA score across all canonical correlations
        eval_single_layer (bool): Whether to evaluate only a single layer
        layer_num (int): Layer number to evaluate if eval_single_layer is True
    """
    # Auto-detect model type if not provided
    if model_name is None:
        model_name = detect_model_type(rep_dir)
        print(f"Auto-detected model: {model_name}")
    
    # Validate model name
    if model_name not in LAYER_CNT:
        available_models = list(LAYER_CNT.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    
    # Check if files are directly in rep_dir or in contextualized/frame_level subdirectory
    direct_files = glob.glob(os.path.join(rep_dir, "layer_*.npy"))
    if direct_files:
        # Files are directly in rep_dir, so we need to adjust the path for cca_intra
        # cca_intra expects rep_dir/contextualized/frame_level/layer_*.npy
        # So we need to pass the parent directory
        adjusted_rep_dir = os.path.dirname(rep_dir)
        if not adjusted_rep_dir:
            adjusted_rep_dir = "."
        
        # Create the expected directory structure temporarily or use a different approach
        # Since cca_intra hardcodes the path, we need to work with its expectations
        print(f"Layer files found directly in {rep_dir}")
        print("Note: cca_intra expects files in contextualized/frame_level/ subdirectory")
        
        # We'll pass the parent directory and let cca_intra append contextualized/frame_level
        # This means your files should be in: parent_dir/contextualized/frame_level/
        expected_path = os.path.join(rep_dir, "contextualized", "frame_level")
        if not os.path.exists(expected_path):
            # Create symbolic link or suggest reorganizing
            print(f"Creating directory structure: {expected_path}")
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            os.makedirs(expected_path, exist_ok=True)
            
            # Create symbolic links to the actual files
            for layer_file in direct_files:
                link_path = os.path.join(expected_path, os.path.basename(layer_file))
                if not os.path.exists(link_path):
                    os.symlink(os.path.abspath(layer_file), link_path)
                    print(f"Created symlink: {link_path} -> {layer_file}")
        
        final_rep_dir = rep_dir
    else:
        # Files are in the expected structure
        final_rep_dir = rep_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    
    # Run CCA analysis
    scores = evaluate_cca(
        model_name=model_name,
        save_fn=save_fn,
        fbank_dir="dummy",  # Required by function signature but not used
        rep_dir=final_rep_dir,
        exp_name="cca_intra",
        base_layer=base_layer,
        rep_dir2=None,
        embed_dir="dummy",  # Required by constructor but not used for cca_intra
        sample_data_fn=None,
        span="phone",
        mean_score=mean_score,
        eval_single_layer=eval_single_layer,
        layer_num=layer_num
    )
    
    # Print results
    print(f"\nPairwise CCA Analysis Results ({model_name}):")
    print("=" * 50)
    for layer, score in scores.items():
        print(f"Layer {layer}: {score:.3f}")
    
    return scores

if __name__ == "__main__":
    codes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'codes'))
    # Example usage - model type will be auto-detected
    run_cca_analysis(
        rep_dir="CCA_analysis/prepare_features/converted_features/hubert_subset",
        save_fn="CCA_analysis/cca_results/cca_scores.json",
    ) 
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def get_layer_number(layer_name: str) -> Union[int, float]:
    """Get layer number for sorting, handling transformer input and layers."""
    if layer_name == 'transformer_input':
        return -1  # Put input layer first
    elif layer_name.startswith('transformer_layer_'):
        layer_num = int(layer_name.split('_')[-1])
        if layer_num <= 11:  # Only include up to layer 11
            return layer_num
    return float('inf')  # Put other layers at the end - ignore layers 12 and above


def pad_features(features: np.ndarray, max_length: int) -> np.ndarray:
    """Pad features to a consistent length."""
    if len(features.shape) == 3:  # [batch, time, dim]
        batch_size, time_steps, dim = features.shape
        padded = np.zeros((batch_size, max_length, dim))
        padded[:, :time_steps, :] = features
    else:  # [time, dim]
        time_steps, dim = features.shape
        padded = np.zeros((max_length, dim))
        padded[:time_steps, :] = features
    return padded


def segment_features(features: np.ndarray, segment_length: int, 
                    strategy: str = 'beginning') -> np.ndarray:
    """
    Extract a fixed-length segment from features instead of padding.
    
    Args:
        features: Input features with shape [batch, time, dim] or [time, dim]
        segment_length: Length of segment to extract
        strategy: 'beginning', 'middle', 'end', or 'random'
    """
    if len(features.shape) == 3:  # [batch, time, dim]
        batch_size, time_steps, dim = features.shape
        segmented = np.zeros((batch_size, segment_length, dim))
        
        for b in range(batch_size):
            if time_steps >= segment_length:
                if strategy == 'beginning':
                    start_idx = 0
                elif strategy == 'end':
                    start_idx = time_steps - segment_length
                elif strategy == 'middle':
                    start_idx = (time_steps - segment_length) // 2
                elif strategy == 'random':
                    start_idx = np.random.randint(0, time_steps - segment_length + 1)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                segmented[b] = features[b, start_idx:start_idx + segment_length, :]
            else:
                # If shorter than segment_length, pad it
                segmented[b, :time_steps, :] = features[b]
                
    else:  # [time, dim]
        time_steps, dim = features.shape
        segmented = np.zeros((segment_length, dim))
        
        if time_steps >= segment_length:
            if strategy == 'beginning':
                start_idx = 0
            elif strategy == 'end':
                start_idx = time_steps - segment_length
            elif strategy == 'middle':
                start_idx = (time_steps - segment_length) // 2
            elif strategy == 'random':
                start_idx = np.random.randint(0, time_steps - segment_length + 1)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            segmented = features[start_idx:start_idx + segment_length, :]
        else:
            # If shorter than segment_length, pad it
            segmented[:time_steps, :] = features
    
    return segmented


def load_features(features_dir: Union[str, Path], 
                 num_files: int = 3,
                 preprocessing: str = 'pad',
                 segment_length: Optional[int] = None,
                 segment_strategy: str = 'beginning') -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    """
    Load features from .npz files with flexible preprocessing options.
    
    Args:
        features_dir: Directory containing feature files
        num_files: Number of files to load
        preprocessing: 'pad' (pad to max length) or 'segment' (extract fixed segments)
        segment_length: Length of segments if using 'segment' preprocessing
        segment_strategy: Strategy for segment extraction ('beginning', 'middle', 'end', 'random')
    
    Returns:
        Tuple of (layer_features dict, original_lengths dict)
    """
    features_dir = Path(features_dir)
    feature_files = list(features_dir.glob("*_complete_features.npz"))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")
    
    # Take only the first num_files
    feature_files = feature_files[:num_files]
    
    # Dictionary to store features for each layer
    layer_features = {}
    max_lengths = {}  # Track max length for each layer (used for padding)
    original_lengths = {}  # Track original lengths
    
    print(f"Loading features from {len(feature_files)} files using '{preprocessing}' preprocessing...")
    
    # First pass: collect all features and determine max lengths
    all_layer_data = {}
    
    for chkpnt_file_path in feature_files:
        print(f"Processing {chkpnt_file_path}")
        layer_features_contextualized = np.load(chkpnt_file_path)
            
        for layer in layer_features_contextualized.files:
            # Only include transformer layers from input to layer 11
            if layer == 'transformer_input' or (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11):
                if layer not in all_layer_data:
                    all_layer_data[layer] = []
                    max_lengths[layer] = 0
                    original_lengths[layer] = []
                
                features = layer_features_contextualized[layer]
                # Get the time dimension (second dimension for 3D, first for 2D)
                time_dim = features.shape[1] if len(features.shape) == 3 else features.shape[0]
                max_lengths[layer] = max(max_lengths[layer], time_dim)
                original_lengths[layer].append(time_dim)
                all_layer_data[layer].append(features)
                print(f"Layer {layer}: shape {features.shape}, time_dim {time_dim}")
    
    # Second pass: apply preprocessing
    for layer in all_layer_data:
        if preprocessing == 'pad':
            print(f"Padding layer {layer} to length {max_lengths[layer]}")
            processed_features = [pad_features(f, max_lengths[layer]) for f in all_layer_data[layer]]
        
        elif preprocessing == 'segment':
            if segment_length is None:
                # Use the minimum length across all files for this layer
                segment_length = min(original_lengths[layer])
                print(f"Auto-determined segment length: {segment_length}")
            
            print(f"Segmenting layer {layer} to length {segment_length} using '{segment_strategy}' strategy")
            processed_features = [segment_features(f, segment_length, segment_strategy) for f in all_layer_data[layer]]
            
            # Update original lengths for segmented data
            original_lengths[layer] = [min(orig_len, segment_length) for orig_len in original_lengths[layer]]
        
        else:
            raise ValueError(f"Unknown preprocessing method: {preprocessing}")
        
        layer_features[layer] = np.concatenate(processed_features, axis=0)
        print(f"Final shape for layer {layer}: {layer_features[layer].shape}")
    
    return layer_features, original_lengths


def filter_and_sort_layers(layer_features: Dict[str, np.ndarray], 
                          max_layer: int = 11) -> List[str]:
    """Filter and sort transformer layers up to a maximum layer number."""
    return sorted([
        layer for layer in layer_features.keys() 
        if layer == 'transformer_input' or 
        (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= max_layer)
    ], key=get_layer_number) 
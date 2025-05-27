import os
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

def calculate_time_steps(audio_duration_seconds, sample_rate=16000):
    """
    Calculate final time steps after CNN processing.
    
    Args:
        audio_duration_seconds: Duration of audio in seconds
        sample_rate: Audio sample rate (default 16kHz for HuBERT)
    
    Returns:
        Number of time steps after CNN downsampling
        
    Examples:
        1.0 seconds → 50 time steps
        3.32 seconds → 166 time steps  
        10.0 seconds → 500 time steps
    """
    input_samples = int(audio_duration_seconds * sample_rate)
    total_stride = 5 * (2**6)  # 5 × 2^6 = 320
    time_steps = input_samples // total_stride
    return time_steps

def load_model(model_name="facebook/hubert-base-ls960"):
    """Load HuBERT model and processor from HuggingFace."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    return model, feature_extractor

def extract_all_cnn_features(model, inputs, device):
    """
    Extract features from all CNN layers (conv_0 through conv_6).
    
    HuBERT CNN Architecture:
    - Layer 0: (batch, 1, time) → (batch, 512, time//5)     [kernel=10, stride=5]
    - Layer 1: (batch, 512, time) → (batch, 512, time//2)   [kernel=3, stride=2]
    - Layer 2: (batch, 512, time) → (batch, 512, time//2)   [kernel=3, stride=2]
    - Layer 3: (batch, 512, time) → (batch, 512, time//2)   [kernel=3, stride=2]
    - Layer 4: (batch, 512, time) → (batch, 512, time//2)   [kernel=3, stride=2]
    - Layer 5: (batch, 512, time) → (batch, 512, time//2)   [kernel=2, stride=2]
    - Layer 6: (batch, 512, time) → (batch, 512, time//2)   [kernel=2, stride=2]
    
    Total stride: 5 × 2^6 = 320
    """
    cnn_features = {}
    
    # Hook function to capture intermediate CNN outputs
    def make_cnn_hook(layer_name):
        def hook_fn(module, input, output):
            # CNN outputs are in format (batch, channels, time)
            cnn_features[layer_name] = output.detach().cpu().numpy()
        return hook_fn
    
    # Register hooks on each CNN layer
    hooks = []
    for i, conv_layer in enumerate(model.feature_extractor.conv_layers):
        layer_name = f"conv_{i}"
        hook = conv_layer.register_forward_hook(make_cnn_hook(layer_name))
        hooks.append(hook)
    
    # Forward pass through CNN to trigger hooks
    with torch.no_grad():
        # This will populate cnn_features dict with all CNN layer outputs
        cnn_output = model.feature_extractor(inputs["input_values"])
        
        # Store final CNN output (same as conv_6 but explicitly captured)
        cnn_features["cnn_final_raw"] = cnn_output.cpu().numpy()  # Shape: (batch, channels=512, time)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return cnn_features, cnn_output

def extract_projection_features(model, cnn_output):
    """
    Extract features from transpose and projection operations.
    
    Process:
    1. CNN output: (batch, channels=512, time) - typical conv format
    2. Transpose: (batch, channels=512, time) → (batch, time, channels=512) - transformer format
    3. Projection: (batch, time, 512) → (batch, time, 768) - via linear layer
    """
    projection_features = {}
    
    with torch.no_grad():
        # Step 1: Transpose from CNN format to Transformer format
        # CNN outputs channels-first, Transformers expect time-first
        transposed = cnn_output.transpose(1, 2)  # (batch, time, channels=512)
        projection_features["after_transpose"] = transposed.cpu().numpy()
        
        # Step 2: Apply feature projection (512 → 768)
        # This uses model.feature_projection layer
        if hasattr(model, 'feature_projection'):
            projected = model.feature_projection(transposed)  # (batch, time, 768)
            projection_features["after_projection"] = projected.cpu().numpy()
        else:
            # Fallback: manually apply projection if attribute name differs
            # Look for projection layer in the model
            for name, module in model.named_modules():
                if 'projection' in name.lower() and hasattr(module, 'weight'):
                    if module.weight.shape == (768, 512):  # Correct projection dimensions
                        projected = module(transposed)
                        projection_features["after_projection"] = projected.cpu().numpy()
                        break
    
    return projection_features

def extract_all_transformer_features(model, inputs):
    """
    Extract features from all 12 Transformer layers.
    
    Each Transformer layer processes:
    Input: (batch, time, 768) → Multi-Head Attention → Feed Forward → Output: (batch, time, 768)
    
    Transformer details:
    - 12 layers total (layer_0 through layer_11)
    - Each layer has 12 attention heads (64 dimensions each = 768 total)
    - Feed forward: 768 → 3072 → 768
    - Layer normalization and residual connections
    """
    transformer_features = {}
    
    with torch.no_grad():
        # Get all hidden states from transformer layers
        outputs = model(**inputs, output_hidden_states=True)
        
        # outputs.hidden_states contains:
        # [0]: Input to first transformer layer (after CNN + projection)
        # [1]: Output of transformer layer 0
        # [2]: Output of transformer layer 1
        # ...
        # [12]: Output of transformer layer 11 (final output)
        
        for i, hidden_state in enumerate(outputs.hidden_states):
            if i == 0:
                # First element is input to transformer (post-projection CNN features)
                layer_name = "transformer_input"
            else:
                # Subsequent elements are outputs of transformer layers
                layer_name = f"transformer_layer_{i-1}"
            
            transformer_features[layer_name] = hidden_state.cpu().numpy()
    
    return transformer_features

def process_audio_file(audio_path, model, feature_extractor, feature_type="all", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Process a single audio file and extract all possible features.
    
    Args:
        audio_path: Path to audio file
        model: HuBERT model
        feature_extractor: Wav2Vec2 feature extractor
        feature_type: Type of features to extract ("cnn", "transformer", "projection", or "all")
        device: Device to run inference on
    
    Returns:
        Dictionary containing all extracted features with descriptive names
    """
    # Convert string path to Path object if needed
    audio_path = Path(audio_path)
    
    # Load and preprocess audio
    print(f"\nProcessing: {audio_path.name}")
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Prepare input
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Input audio shape: {inputs['input_values'].shape}")  # e.g., (1, 53120) for 3.32s audio
    
    # Calculate expected time steps after CNN processing
    audio_length = inputs['input_values'].shape[1]
    expected_time_steps = audio_length // 320  # Total CNN stride
    print(f"Expected final time steps: {expected_time_steps} (from {audio_length} samples)")
    print(f"Audio duration: {audio_length / 16000:.3f} seconds")
    
    all_features = {}
    
    # 1. Extract CNN features (conv_0 through conv_6)
    if feature_type in ["cnn", "all"]:
        print("Extracting CNN features...")
        cnn_features, cnn_output = extract_all_cnn_features(model, inputs, device)
        all_features.update(cnn_features)
        
        # Print CNN layer shapes for verification
        for layer_name, features in cnn_features.items():
            print(f"  {layer_name}: {features.shape}")
    else:
        # We need CNN output for projection step
        with torch.no_grad():
            cnn_output = model.feature_extractor(inputs["input_values"])
    
    # 2. Extract projection features (transpose + linear projection)
    if feature_type in ["projection", "all"]:
        print("Extracting projection features...")
        projection_features = extract_projection_features(model, cnn_output)
        all_features.update(projection_features)
        
        # Print projection shapes
        for layer_name, features in projection_features.items():
            print(f"  {layer_name}: {features.shape}")
    
    # 3. Extract Transformer features (all 12 layers)
    if feature_type in ["transformer", "all"]:
        print("Extracting Transformer features...")
        transformer_features = extract_all_transformer_features(model, inputs)
        all_features.update(transformer_features)
        
        # Print transformer layer shapes
        for layer_name, features in transformer_features.items():
            print(f"  {layer_name}: {features.shape}")
    
    # Summary of extracted features
    print(f"\nTotal features extracted: {len(all_features)}")
    print("Feature summary:")
    for name, features in all_features.items():
        print(f"  {name}: {features.shape}")
    
    # Verify time steps match expectation
    if 'after_projection' in all_features:
        actual_time_steps = all_features['after_projection'].shape[1]
        expected = audio_length // 320
        print(f"\nTime steps verification:")
        print(f"  Expected: {expected}, Actual: {actual_time_steps} ✓" if actual_time_steps == expected else f"  Expected: {expected}, Actual: {actual_time_steps} ✗")
    
    return all_features

def main():
    parser = argparse.ArgumentParser(description="Extract ALL HuBERT features from audio files")
    parser.add_argument("--model_name", type=str, default="facebook/hubert-base-ls960",
                      help="Name of the HuBERT model to use")
    parser.add_argument("--data_sample", type=int, default=1,
                      help="Sample identifier")
    parser.add_argument("--feature_type", type=str, default="all",
                      choices=["cnn", "projection", "transformer", "all"],
                      help="Type of features to extract")
    parser.add_argument("--span", type=str, default="frame",
                      choices=["frame", "phone", "word"],
                      help="Time span of the features")
    parser.add_argument("--subset_id", type=int, default=0,
                      help="Subset identifier for parallel processing")
    parser.add_argument("--dataset_split", type=str, default="dev-clean",
                      help="Dataset split to process")
    parser.add_argument("--save_dir", type=str, required=True,
                      help="Directory to save extracted features")
    parser.add_argument("--audio_dir", type=str, required=True,
                      help="Directory containing audio files")
    args = parser.parse_args()

    # Setup paths
    output_dir = os.path.join(args.save_dir, "hubert_complete", f"librispeech_{args.dataset_split}_sample{args.data_sample}")
    if args.span != "frame":
        output_dir = os.path.join(output_dir, str(args.subset_id))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading HuBERT model...")
    model, feature_extractor = load_model(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Print model architecture info
    print(f"\nModel info:")
    print(f"CNN layers: {len(model.feature_extractor.conv_layers)}")
    if hasattr(model, 'encoder'):
        print(f"Transformer layers: {len(model.encoder.layers)}")
    
    # Process all audio files recursively
    audio_files = list(Path(args.audio_dir).rglob("*.wav"))
    audio_files += list(Path(args.audio_dir).rglob("*.flac"))
    print(f"\nFound {len(audio_files)} audio files")
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Extract ALL features
            all_features = process_audio_file(audio_path, model, feature_extractor, args.feature_type, device)
            
            # Save features with comprehensive naming
            output_path = os.path.join(output_dir, f"{audio_path.stem}_complete_features.npz")
            np.savez_compressed(output_path, **all_features)  # Use compression for large files
            
            print(f"Saved features to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()

"""
FEATURE EXTRACTION SUMMARY:

CNN Features (7 layers):
- conv_0: (batch, 512, time_0) - First conv layer, 1→512 channels
- conv_1: (batch, 512, time_1) - Second conv layer, maintains 512 channels
- conv_2: (batch, 512, time_2) - Third conv layer, maintains 512 channels
- conv_3: (batch, 512, time_3) - Fourth conv layer, maintains 512 channels
- conv_4: (batch, 512, time_4) - Fifth conv layer, maintains 512 channels
- conv_5: (batch, 512, time_5) - Sixth conv layer, maintains 512 channels
- conv_6: (batch, 512, time_6) - Seventh conv layer, maintains 512 channels
- cnn_final_raw: Same as conv_6, final CNN output

Projection Features:
- after_transpose: (batch, time_final, 512) - CNN output transposed to transformer format
- after_projection: (batch, time_final, 768) - After linear projection 512→768

Transformer Features (13 features total):
- transformer_input: (batch, time_final, 768) - Input to first transformer layer
- transformer_layer_0: (batch, time_final, 768) - Output of transformer layer 0
- transformer_layer_1: (batch, time_final, 768) - Output of transformer layer 1
- ...
- transformer_layer_11: (batch, time_final, 768) - Output of final transformer layer

EXAMPLE: 3.32 seconds audio (typical use case)
Time dimension progression through CNN:
- Original audio: 53,120 samples (3.32 seconds at 16kHz)
- Layer 0: 53,120 ÷ 5 = 10,624 time steps
- Layer 1: 10,624 ÷ 2 = 5,312 time steps
- Layer 2: 5,312 ÷ 2 = 2,656 time steps
- Layer 3: 2,656 ÷ 2 = 1,328 time steps
- Layer 4: 1,328 ÷ 2 = 664 time steps
- Layer 5: 664 ÷ 2 = 332 time steps
- Layer 6: 332 ÷ 2 = 166 time steps (final)
- After CNN (total stride 320): 53,120/320 = 166 time steps
- Transformer maintains 166 time steps throughout
- Each final time step represents ~20ms of original audio

General formula: final_time_steps = int(audio_duration_seconds * 16000 / 320)

Channel/Feature dimension progression:
- CNN: 1 → 512 channels (conv format)
- Projection: 512 → 768 features (transformer format)
- Transformer: maintains 768 features throughout

Expected shapes for 3.32s audio:
- CNN layers: (1, 512, 166)
- After projection: (1, 166, 768)
- All transformer layers: (1, 166, 768)
"""
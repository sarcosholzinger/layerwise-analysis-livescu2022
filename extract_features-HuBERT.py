import os
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

def load_model(model_name="facebook/hubert-base-ls960"):
    """Load HuBERT model and processor from HuggingFace."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    return model, feature_extractor

def process_audio_file(audio_path, model, feature_extractor, rep_type="contextualized", device="cuda" if torch.cuda.is_available() else "cpu"):
    """Process a single audio file and extract features.
    
    Args:
        audio_path: Path to audio file
        model: HuBERT model
        feature_extractor: Wav2Vec2 feature extractor
        rep_type: Type of representation to extract ("local" or "contextualized")
        device: Device to run inference on
    """
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Prepare input
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    features = {}
    if rep_type == "local":
        # Get convolutional layer features
        for i, h in enumerate(outputs.hidden_states[:7]):  # First 7 layers are convolutional
            features[f"conv_{i}"] = h.cpu().numpy()
    else:  # contextualized
        # Get all transformer layer features
        for i, h in enumerate(outputs.hidden_states):
            features[f"layer_{i}"] = h.cpu().numpy()
    
    return features

def main():
    parser = argparse.ArgumentParser(description="Extract HuBERT features from audio files")
    parser.add_argument("--model_name", type=str, default="facebook/hubert-base-ls960",
                      help="Name of the HuBERT model to use")
    parser.add_argument("--data_sample", type=int, default=1,
                      help="Sample identifier")
    parser.add_argument("--rep_type", type=str, default="contextualized",
                      choices=["local", "contextualized"],
                      help="Type of representation to extract")
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
    output_dir = os.path.join(args.save_dir, "hubert", f"librispeech_{args.dataset_split}_sample{args.data_sample}")
    if args.span != "frame":
        output_dir = os.path.join(output_dir, str(args.subset_id))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading HuBERT model...")
    model, feature_extractor = load_model(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Process all audio files recursively
    audio_files = list(Path(args.audio_dir).rglob("*.wav"))
    audio_files += list(Path(args.audio_dir).rglob("*.flac"))
    print(f"Found {len(audio_files)} audio files")
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Extract features
            features = process_audio_file(str(audio_path), model, feature_extractor, args.rep_type, device)
            
            # Save features
            output_path = os.path.join(output_dir, f"{audio_path.stem}_features.npz")
            np.savez(output_path, **features)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
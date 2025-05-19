import json
import os
from pathlib import Path

def create_segments_json(audio_dir, output_file):
    """Create a segments JSON file from audio files."""
    segments = []
    
    # Get all audio files
    audio_files = list(Path(audio_dir).rglob("*.flac"))
    
    for audio_path in audio_files:
        # Create segment entry
        segment = {
            "utt_id": audio_path.stem,
            "path": str(audio_path),
            "start_time": 0.0,  # We don't have actual segment times, so using full file
            "end_time": -1.0    # -1 indicates use full file
        }
        segments.append(segment)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(segments, f, indent=2)

if __name__ == "__main__":
    audio_dir = "/home/sarcosh1/repos/layerwise-analysis/content/data/LibriSpeech/dev-clean-2"
    output_file = "/home/sarcosh1/repos/layerwise-analysis/data_samples/librispeech/frame_level/dev-clean_segments_sample1_0.json"
    
    create_segments_json(audio_dir, output_file) 
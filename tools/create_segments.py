"""
Audio Segments JSON Generator

This script creates a segments JSON file from audio files in a directory for speech processing pipelines.

PURPOSE:
    Converts a directory of audio files into a structured JSON format that can be used 
    for speech processing and layerwise analysis workflows.

FUNCTIONALITY:
    1. Recursively scans directories for .flac audio files
    2. Creates segment metadata for each audio file with:
       - utt_id: Filename (without extension) as unique utterance identifier  
       - path: Full file path to the audio file
       - start_time: Set to 0.0 (beginning of file)
       - end_time: Set to -1.0 (special value indicating "use entire file")
    3. Exports all segment information as a formatted JSON file

CONTEXT & USE CASE:
    - Dataset: Works with LibriSpeech (popular speech recognition dataset)
    - Integration: Part of layerwise analysis pipeline for speech models
    - Data preparation: Converts raw audio files into format expected by downstream tools

EXAMPLE OUTPUT:
    [
      {
        "utt_id": "speaker1_001",
        "path": "/audio/speaker1_001.flac", 
        "start_time": 0.0,
        "end_time": -1.0
      },
      ...
    ]

STATUS: 
    USAGE UNCLEAR - This file appears to be a utility script but its integration
    with the current layerwise analysis pipeline needs verification. The hardcoded
    paths suggest it may be legacy code from initial data preparation phases.
    
    TODO: Verify if this script or its outputs are actively used in:
    - Feature extraction pipeline
    - Layer similarity analysis
    - Current data preprocessing workflows
"""

import json
import os
from pathlib import Path

def create_segments_json(audio_dir, output_file):
    """
    Create a segments JSON file from audio files in a directory.
    This is commonly used in speech processing pipelines to create a segments JSON file from audio files in a directory.
    
    Args:
        audio_dir (str): Directory path containing audio files (.flac)
        output_file (str): Output path for the segments JSON file
        
    Returns:
        None: Writes JSON file to specified output path
    """
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
    audio_dir = "./content/data/LibriSpeech/dev-clean-2" #TODO: This is hardcoded, should be changed to use the project root directory!
    output_file = "./data_samples/librispeech/frame_level/dev-clean_segments_sample1_0.json" #TODO: This is hardcoded, should be changed to use the project root directory!
    create_segments_json(audio_dir, output_file) 
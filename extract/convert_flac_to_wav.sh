#!/bin/bash

# Convert all FLAC files to WAV in LibriSpeech dev-clean directory
# Usage: ./convert_flac_to_wav.sh [folder_name]
# Example: ./convert_flac_to_wav.sh 2412 (2412 is the speaker id aligned with the directory name)

# Load ffmpeg module
module load ffmpeg

# Parse command line arguments
SPECIFIC_FOLDER=""
if [[ $# -eq 1 ]]; then
    SPECIFIC_FOLDER="$1"
    echo "Converting FLAC files only in folder: $SPECIFIC_FOLDER"
fi

SOURCE_DIR="/exp/sholzinger/LibriSpeech/dev-clean"

# If specific folder is provided, update source directory
if [[ -n "$SPECIFIC_FOLDER" ]]; then
    SOURCE_DIR="$SOURCE_DIR/$SPECIFIC_FOLDER"
    if [[ ! -d "$SOURCE_DIR" ]]; then
        echo "Error: Directory $SOURCE_DIR does not exist!"
        exit 1
    fi
fi

echo "Converting FLAC files to WAV in: $SOURCE_DIR"
echo "This will create WAV files alongside the original FLAC files..."

# Find all FLAC files and convert them
find "$SOURCE_DIR" -name "*.flac" -type f -exec realpath {} \; | while read -r flac_file; do
    # Generate WAV filename by replacing .flac with .wav
    wav_file="${flac_file%.flac}.wav"
    
    # Check if WAV file already exists
    if [[ -f "$wav_file" ]]; then
        echo "Skipping (already exists): $wav_file"
        continue
    fi
    
    echo "Converting: $flac_file -> $wav_file"
    
    # Convert using ffmpeg (most common) or sox (alternative)
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -i "$flac_file" "$wav_file" -y -loglevel quiet
    elif command -v sox &> /dev/null; then
        sox "$flac_file" "$wav_file"
    else
        echo "Error: Neither ffmpeg nor sox found. Please install one of them."
        exit 1
    fi
    
    if [[ $? -eq 0 ]]; then
        echo " Converted successfully"
    else
        echo " Failed to convert: $flac_file"
    fi
done

echo "Conversion complete!" 
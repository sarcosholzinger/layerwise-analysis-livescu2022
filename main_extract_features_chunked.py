#!/usr/bin/env python3
"""
Run HuBERT feature extraction in smaller chunks to avoid SLURM limits.
This script processes files in batches and can be resumed if interrupted.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import time

def get_audio_files(audio_dir):
    """Get all audio files from directory."""
    audio_files = list(Path(audio_dir).rglob("*.wav"))
    audio_files += list(Path(audio_dir).rglob("*.flac"))
    return sorted(audio_files)

def create_file_chunks(audio_files, chunk_size):
    """Split audio files into chunks."""
    chunks = []
    for i in range(0, len(audio_files), chunk_size):
        chunks.append(audio_files[i:i + chunk_size])
    return chunks

def get_completed_files(output_dir):
    """Get list of already processed files to skip."""
    completed = set()
    if os.path.exists(output_dir):
        for npz_file in Path(output_dir).rglob("*.npz"):
            completed.add(npz_file.stem)
    return completed

def create_temp_audio_dir(chunk_files, temp_dir, base_audio_dir):
    """Create temporary directory with symlinks to chunk files."""
    temp_audio_path = Path(temp_dir) / "chunk_audio"
    temp_audio_path.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks preserving directory structure
    for audio_file in chunk_files:
        audio_path = Path(audio_file)
        rel_path = audio_path.relative_to(base_audio_dir)
        
        # Create subdirectories if needed
        link_path = temp_audio_path / rel_path
        link_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink if it doesn't exist
        if not link_path.exists():
            link_path.symlink_to(audio_path.absolute())
    
    return temp_audio_path

def run_chunk(chunk_files, chunk_id, args, base_audio_dir):
    """Run feature extraction on a single chunk."""
    print(f"\n{'='*60}")
    print(f"Processing Chunk {chunk_id + 1}: {len(chunk_files)} files")
    print(f"{'='*60}")
    
    # Create temporary directory for this chunk
    temp_dir = f"/tmp/extract_chunk_{chunk_id}_{os.getpid()}"
    temp_audio_dir = create_temp_audio_dir(chunk_files, temp_dir, base_audio_dir)
    
    try:
        # Run extraction on this chunk
        cmd = [
            sys.executable, "extract/extract_features-HuBERT.py",
            "--model_name", args.model_name,
            "--data_sample", str(args.data_sample),
            "--feature_type", args.feature_type,
            "--span", args.span,
            "--subset_id", str(args.subset_id),
            "--dataset_split", args.dataset_split,
            "--save_dir", args.save_dir,
            "--audio_dir", str(temp_audio_dir),
        ]
        
        # Add GPU options
        if args.single_gpu:
            cmd.append("--single_gpu")
        if args.max_gpus:
            cmd.extend(["--max_gpus", str(args.max_gpus)])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the extraction
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"Chunk {chunk_id + 1} completed in {end_time - start_time:.1f} seconds")
        
        if result.returncode != 0:
            print(f"ERROR in chunk {chunk_id + 1}:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        else:
            print(f"Chunk {chunk_id + 1} completed successfully")
            return True
            
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            subprocess.run(["rm", "-rf", temp_dir], check=False)

def main():
    parser = argparse.ArgumentParser(description="Run HuBERT feature extraction in chunks")
    parser.add_argument("--audio_dir", type=str, required=True,
                      help="Directory containing audio files")
    parser.add_argument("--save_dir", type=str, required=True,
                      help="Directory to save extracted features")
    parser.add_argument("--chunk_size", type=int, default=100,
                      help="Number of files per chunk (default: 100)")
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
    parser.add_argument("--max_gpus", type=int, default=None,
                      help="Maximum number of GPUs to use")
    parser.add_argument("--single_gpu", action="store_true",
                      help="Force single GPU processing")
    parser.add_argument("--start_chunk", type=int, default=0,
                      help="Start from this chunk (for resuming)")
    parser.add_argument("--max_chunks", type=int, default=None,
                      help="Maximum number of chunks to process")
    
    args = parser.parse_args()
    
    # Get all audio files
    print(f"Scanning for audio files in {args.audio_dir}...")
    audio_files = get_audio_files(args.audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return
    
    # Create chunks
    chunks = create_file_chunks(audio_files, args.chunk_size)
    print(f"Created {len(chunks)} chunks of size {args.chunk_size}")
    
    # Determine which chunks to process
    start_chunk = args.start_chunk
    end_chunk = len(chunks)
    if args.max_chunks:
        end_chunk = min(start_chunk + args.max_chunks, len(chunks))
    
    print(f"Processing chunks {start_chunk} to {end_chunk - 1}")
    
    # Process each chunk
    successful_chunks = 0
    failed_chunks = 0
    
    for i in range(start_chunk, end_chunk):
        chunk_files = chunks[i]
        print(f"\nChunk {i + 1}/{len(chunks)}: {len(chunk_files)} files")
        
        success = run_chunk(chunk_files, i, args, args.audio_dir)
        
        if success:
            successful_chunks += 1
        else:
            failed_chunks += 1
            print(f"Failed to process chunk {i + 1}")
            
            # Decide whether to continue or stop
            response = input("Continue with next chunk? (y/n): ")
            if response.lower() != 'y':
                break
    
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"Successful chunks: {successful_chunks}")
    print(f"Failed chunks: {failed_chunks}")
    print(f"Total chunks processed: {successful_chunks + failed_chunks}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

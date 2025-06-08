"""
HuBERT Model Download and Verification Script

This script downloads the HuBERT base model from HuggingFace and verifies its capability 
for layerwise feature extraction in speech analysis pipelines.

PURPOSE:
    Downloads and locally caches the HuBERT (Hidden-Unit BERT) base model for layerwise 
    analysis of speech representations. Verifies the model can extract hidden states 
    from all transformer layers.

FUNCTIONALITY:
    1. Creates local model checkpoint directory structure
    2. Downloads HuBERT base model from HuggingFace Hub (facebook/hubert-base-ls960)
    3. Saves model state dictionary to local checkpoint file
    4. Performs verification test with dummy audio input
    5. Confirms layerwise hidden state extraction capability
    6. Reports layer count and output tensor shapes

MODEL DETAILS:
    - Model: HuBERT Base (facebook/hubert-base-ls960)
    - Training Data: LibriSpeech 960h dataset
    - Architecture: Transformer-based speech representation model
    - Layers: Multiple transformer layers with hidden states accessible
    - Input: Raw audio waveform (16kHz sampling rate)
    - Output: Contextual speech representations per layer

VERIFICATION TEST:
    - Creates 1-second dummy audio input (16,000 samples at 16kHz)
    - Extracts hidden states from all layers
    - Validates tensor shapes and layer accessibility
    - Confirms model readiness for downstream analysis

STATUS:
    LEGACY/SETUP SCRIPT - This appears to be a one-time setup script with hardcoded 
    paths that may not align with current directory structure. The script downloads
    and verifies HuBERT model but may need path updates for current pipeline.
    
    TODO: Verify if this script is:
    - Still needed for model setup
    - Using correct paths for current environment  
    - Compatible with current feature extraction pipeline
    - Superseded by newer model loading utilities
"""

import os
import torch
from transformers import HubertModel, HubertConfig
import numpy as np

# Create directory for HuBERT model
os.makedirs('./model_checkpoints/hubert_base', exist_ok=True) #TODO: This is hardcoded, should be changed to use the project root directory!

# Download and load the model from HuggingFace
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

# Save the model state dict
model_path = './model_checkpoints/hubert_base/hubert_base.pt'
torch.save(model.state_dict(), model_path)

# Test the model with a dummy input to verify layerwise extraction
dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
with torch.no_grad():
    outputs = model(dummy_input, output_hidden_states=True)
    
# Verify we can access all layer representations
print(f"Number of layers: {len(outputs.hidden_states)}")
print(f"Shape of first layer output: {outputs.hidden_states[0].shape}")
print(f"Shape of last layer output: {outputs.hidden_states[-1].shape}")

print("HuBERT base model downloaded from HuggingFace and verified for layerwise extraction!") 
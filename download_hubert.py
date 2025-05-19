import os
import torch
from transformers import HubertModel, HubertConfig
import numpy as np

# Create directory for HuBERT model
os.makedirs('/home/sarcosh1/repos/layerwise-analysis/model_checkpoints/hubert_base', exist_ok=True)

# Download and load the model from HuggingFace
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

# Save the model state dict
model_path = '/home/sarcosh1/repos/layerwise-analysis/model_checkpoints/hubert_base/hubert_base.pt'
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
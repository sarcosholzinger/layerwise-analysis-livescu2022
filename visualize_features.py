import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

def pad_features(features, max_length):
    """Pad features to a consistent length."""
    if len(features.shape) == 3:  # [batch, time, dim]
        # Ensure we're using the correct dimensions
        batch_size, time_steps, dim = features.shape
        padded = np.zeros((batch_size, max_length, dim))
        padded[:, :time_steps, :] = features
    else:  # [time, dim]
        time_steps, dim = features.shape
        padded = np.zeros((max_length, dim))
        padded[:time_steps, :] = features
    return padded

def load_features(features_dir, num_files=3):
    """Load features from a subset of .npz files in the directory."""
    features_dir = Path(features_dir)
    feature_files = list(features_dir.glob("*_features.npz"))
    
    # Take only the first num_files
    feature_files = feature_files[:num_files]
    
    # Dictionary to store features for each layer
    layer_features = {}
    max_lengths = {}  # Track max length for each layer
    
    print(f"Loading features from {len(feature_files)} files...")
    for file_path in feature_files:
        print(f"Processing {file_path}")
        data = np.load(file_path)
        for key in data.files:
            if key not in layer_features:
                layer_features[key] = []
                max_lengths[key] = 0
            features = data[key]
            # Get the time dimension (second dimension for 3D, first for 2D)
            time_dim = features.shape[1] if len(features.shape) == 3 else features.shape[0]
            max_lengths[key] = max(max_lengths[key], time_dim)
            layer_features[key].append(features)
            print(f"Layer {key}: shape {features.shape}, max_length {max_lengths[key]}")
    
    # Pad and concatenate features
    for key in layer_features:
        print(f"Padding layer {key} to length {max_lengths[key]}")
        padded_features = [pad_features(f, max_lengths[key]) for f in layer_features[key]]
        layer_features[key] = np.concatenate(padded_features, axis=0)
        print(f"Final shape for layer {key}: {layer_features[key].shape}")
    
    return layer_features

def plot_feature_distributions(layer_features, output_dir, model_name, num_files):
    """Plot feature distributions for each layer."""
    plt.figure(figsize=(15, 10))
    
    # Plot mean and std of features for each layer
    means = []
    stds = []
    layers = sorted(layer_features.keys())
    
    for layer in layers:
        features = layer_features[layer]
        # Reshape to 2D if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        means.append(np.mean(features))
        stds.append(np.std(features))
    
    plt.errorbar(range(len(layers)), means, yerr=stds, fmt='o-', capsize=5)
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.title('Feature Statistics Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Feature Value')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions_{model_name}_n{num_files}.png')
    plt.close()

def plot_layer_similarity(layer_features, output_dir, model_name, num_files):
    """Plot similarity matrix between layers."""
    # Sort layers numerically instead of alphabetically
    def get_layer_number(layer_name):
        return int(layer_name.split('_')[1]) if layer_name.startswith('layer_') else int(layer_name.split('_')[1])
    
    # Get and sort layers
    layers = list(layer_features.keys())
    layers.sort(key=get_layer_number)
    print(f"Sorted layers: {layers}")  # Debug print
    
    n_layers = len(layers)
    similarity_matrix = np.zeros((n_layers, n_layers))
    
    # Compute cosine similarity between layers
    for i, layer1 in enumerate(layers):
        for j, layer2 in enumerate(layers):
            features1 = layer_features[layer1]
            features2 = layer_features[layer2]
            # Reshape to 2D if needed
            if len(features1.shape) > 2:
                features1 = features1.reshape(features1.shape[0], -1)
            if len(features2.shape) > 2:
                features2 = features2.reshape(features2.shape[0], -1)
            # Compute similarity
            similarity = np.dot(features1.flatten(), features2.flatten()) / (
                np.linalg.norm(features1.flatten()) * np.linalg.norm(features2.flatten())
            )
            similarity_matrix[i, j] = similarity
    
    plt.figure(figsize=(12, 10))
    # Create heatmap with explicit layer ordering
    sns.heatmap(similarity_matrix, 
                xticklabels=layers, 
                yticklabels=layers,
                cmap='viridis')
    
    # Ensure y-axis is in correct order (top to bottom)
    plt.gca().invert_yaxis()
    
    plt.title(f'Layer-wise Similarity Matrix\nModel: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_similarity_{model_name}_n{num_files}.png')
    plt.close()

def plot_dimensionality_reduction(layer_features, output_dir, model_name, num_files):
    """Plot PCA and t-SNE visualizations for selected layers."""
    # Select a few representative layers
    selected_layers = ['layer_0', 'layer_6', 'layer_12']  # First, middle, and last transformer layers
    
    # PCA
    plt.figure(figsize=(15, 5))
    for i, layer in enumerate(selected_layers, 1):
        features = layer_features[layer]
        # Reshape to 2D if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        
        plt.subplot(1, 3, i)
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
        plt.title(f'PCA - {layer}\nExplained variance: {pca.explained_variance_ratio_.sum():.2f}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_visualization_{model_name}_n{num_files}.png')
    plt.close()
    
    # t-SNE (only if we have enough samples)
    n_samples = features.shape[0]
    if n_samples > 5:  # Only do t-SNE if we have more than 5 samples
        plt.figure(figsize=(15, 5))
        for i, layer in enumerate(selected_layers, 1):
            features = layer_features[layer]
            # Reshape to 2D if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Apply t-SNE with adjusted perplexity
            perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than n_samples
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            reduced_features = tsne.fit_transform(features)
            
            plt.subplot(1, 3, i)
            plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
            plt.title(f't-SNE - {layer}\nPerplexity: {perplexity}')
            plt.xlabel('t-SNE1')
            plt.ylabel('t-SNE2')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsne_visualization_{model_name}_n{num_files}.png')
        plt.close()
    else:
        print(f"Skipping t-SNE visualization due to small sample size (n={n_samples})")
    
    # Additional visualization: Feature variance across layers
    plt.figure(figsize=(15, 5))
    for i, layer in enumerate(selected_layers, 1):
        features = layer_features[layer]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Calculate variance of each feature dimension
        feature_vars = np.var(features, axis=0)
        
        plt.subplot(1, 3, i)
        plt.hist(feature_vars, bins=50)
        plt.title(f'Feature Variance Distribution - {layer}')
        plt.xlabel('Variance')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_variance_{model_name}_n{num_files}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize HuBERT features")
    parser.add_argument("--features_dir", type=str, 
                      default="/home/sarcosh1/repos/layerwise-analysis/output/hubert/librispeech_dev-clean_sample1",
                      help="Directory containing feature .npz files")
    parser.add_argument("--output_dir", type=str,
                      default="/home/sarcosh1/repos/layerwise-analysis/output/visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--num_files", type=int, default=3,
                      help="Number of audio files to analyze")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the model architecture (e.g., 'HuBERT Base', 'HuBERT Large')")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load features
    layer_features = load_features(args.features_dir, args.num_files)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_feature_distributions(layer_features, args.output_dir, args.model_name, args.num_files)
    plot_layer_similarity(layer_features, args.output_dir, args.model_name, args.num_files)
    plot_dimensionality_reduction(layer_features, args.output_dir, args.model_name, args.num_files)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
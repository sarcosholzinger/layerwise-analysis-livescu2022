import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import argparse
import torch
from sklearn.cross_decomposition import CCA

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
    
    # Sort layers numerically
    def get_layer_number(layer_name):
        return int(layer_name.split('_')[1]) if layer_name.startswith('layer_') else int(layer_name.split('_')[1])
    
    layers = sorted(layer_features.keys(), key=get_layer_number)
    
    # Plot mean and std of features for each layer
    means = []
    stds = []
    
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
    """Plot PCA, t-SNE, and UMAP visualizations for selected layers."""
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
    print(f"Number of samples for t-SNE: {n_samples}")
    
    if n_samples > 5:  # Only do t-SNE if we have more than 5 samples
        plt.figure(figsize=(15, 5))
        for i, layer in enumerate(selected_layers, 1):
            features = layer_features[layer]
            # Reshape to 2D if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Calculate appropriate perplexity
            perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than n_samples
            print(f"Using perplexity: {perplexity} for layer {layer}")
            
            try:
                # Apply t-SNE with adjusted perplexity
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=perplexity,
                    n_iter=1000,  # Increase iterations for better convergence
                    learning_rate='auto'  # Let t-SNE choose the learning rate
                )
                reduced_features = tsne.fit_transform(features)
                
                plt.subplot(1, 3, i)
                plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
                plt.title(f't-SNE - {layer}\nPerplexity: {perplexity}')
                plt.xlabel('t-SNE1')
                plt.ylabel('t-SNE2')
            except Exception as e:
                print(f"Error in t-SNE for layer {layer}: {str(e)}")
                continue
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsne_visualization_{model_name}_n{num_files}.png')
        plt.close()
    else:
        print(f"Skipping t-SNE visualization due to small sample size (n={n_samples})")
        print("Consider increasing the number of files for better t-SNE visualization")
    
    # UMAP visualization with color coding
    plt.figure(figsize=(20, 10))  # Increased figure size to accommodate all plots
    
    # First row: UMAP visualizations
    for i, layer in enumerate(selected_layers, 1):
        features = layer_features[layer]
        # Reshape to 2D if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Adjust n_neighbors based on dataset size
        n_samples = features.shape[0]
        n_neighbors = min(15, n_samples - 1)  # Ensure n_neighbors is less than n_samples
        
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,  # Dynamic adjustment
            min_dist=0.1,
            metric='euclidean',
            n_jobs=1,
            verbose=False
        )
        reduced_features = reducer.fit_transform(features)
        
        # Analyze activation magnitude (L2 norm) across all dimensions
        activation_magnitudes = np.linalg.norm(features, axis=1)
        
        # Skip UMAP if dataset is too small
        if n_samples <= 2:
            print(f"Skipping UMAP visualization for {layer} due to small sample size (n={n_samples})")
            continue
        
        # UMAP plot colored by activation magnitude
        plt.subplot(2, 3, i)
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=activation_magnitudes, 
                            cmap='coolwarm',
                            alpha=0.5)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Activation Magnitude (L2 Norm)\n(Blue: Low, Red: High)')
        plt.title(f'UMAP - {layer}\nActivation Magnitude Distribution')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        
        # Add statistics
        mean_mag = np.mean(activation_magnitudes)
        std_mag = np.std(activation_magnitudes)
        high_mag_threshold = mean_mag + 2 * std_mag
        low_mag_threshold = mean_mag - 2 * std_mag
        
        # Count points with high and low magnitudes
        high_mag_count = np.sum(activation_magnitudes > high_mag_threshold)
        low_mag_count = np.sum(activation_magnitudes < low_mag_threshold)
        
        stats_text = (
            f'Mean: {mean_mag:.2f}\n'
            f'Std: {std_mag:.2f}\n'
            f'High magnitude: {high_mag_count} ({high_mag_count/len(activation_magnitudes)*100:.1f}%)\n'
            f'Low magnitude: {low_mag_count} ({low_mag_count/len(activation_magnitudes)*100:.1f}%)'
        )
        
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Activation magnitude distribution plot
        plt.subplot(2, 3, i + 3)
        plt.hist(activation_magnitudes, bins=50, color='gray', alpha=0.7)
        plt.axvline(mean_mag, color='red', linestyle='--', label='Mean')
        plt.axvline(high_mag_threshold, color='blue', linestyle=':', label='±2σ')
        plt.axvline(low_mag_threshold, color='blue', linestyle=':')
        plt.title(f'Activation Magnitude Distribution - {layer}')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/umap_visualization_{model_name}_n{num_files}.png')
    plt.close()
    
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

def plot_layer_statistics(layer_features, output_dir, model_name, num_files):
    """Plot statistics of activations across layers to show their progression."""
    # Sort layers numerically
    def get_layer_number(layer_name):
        return int(layer_name.split('_')[1]) if layer_name.startswith('layer_') else int(layer_name.split('_')[1])
    
    layers = sorted(layer_features.keys(), key=get_layer_number)
    # Analyze all dimensions instead of just the last one
    all_dim_means = []
    all_dim_stds = []
    activation_magnitude_means = []
    activation_magnitude_stds = []
    
    for layer in layers:
        features = layer_features[layer]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Compute statistics across all dimensions
        # Mean and std of each dimension, then average
        dim_means = np.mean(features, axis=0)  # Mean of each dimension
        dim_stds = np.std(features, axis=0)    # Std of each dimension
        
        all_dim_means.append(np.mean(dim_means))
        all_dim_stds.append(np.mean(dim_stds))
        
        # Also compute activation magnitudes (L2 norm)
        activation_magnitudes = np.linalg.norm(features, axis=1)
        activation_magnitude_means.append(np.mean(activation_magnitudes))
        activation_magnitude_stds.append(np.std(activation_magnitudes))
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Average feature values across dimensions
    plt.subplot(3, 1, 1)
    plt.errorbar(range(len(layers)), all_dim_means, yerr=all_dim_stds, fmt='o-', capsize=5, label='Mean ± Std across dimensions')
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.title('Average Feature Statistics Across All Dimensions')
    plt.xlabel('Layer')
    plt.ylabel('Average Feature Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Activation magnitude progression
    plt.subplot(3, 1, 2)
    plt.errorbar(range(len(layers)), activation_magnitude_means, yerr=activation_magnitude_stds, 
                 fmt='o-', capsize=5, color='green', label='Mean L2 Norm ± Std')
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.title('Activation Magnitude (L2 Norm) Progression Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Dimension-wise variance distribution
    plt.subplot(3, 1, 3)
    dim_variances = []
    for layer in layers:
        features = layer_features[layer]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        var_per_dim = np.var(features, axis=0)
        dim_variances.append(np.mean(var_per_dim))
    
    plt.plot(range(len(layers)), dim_variances, 'o-', color='purple')
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.title('Average Variance Per Dimension Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Average Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_statistics_{model_name}_n{num_files}.png')
    plt.close()

def analyze_feature_divergence(layer_features, output_dir, model_name, num_files):
    """Analyze how features diverge from the CNN-transformer boundary."""
    # Sort layers numerically
    def get_layer_number(layer_name):
        return int(layer_name.split('_')[1]) if layer_name.startswith('layer_') else int(layer_name.split('_')[1])
    
    layers = sorted(layer_features.keys(), key=get_layer_number)
    
    # Use layer 7 (CNN-to-transformer boundary) as reference
    reference_layer = 'layer_7'
    if reference_layer not in layer_features:
        print(f"Warning: {reference_layer} not found. Using layer_6 as reference.")
        reference_layer = 'layer_6'
    
    reference_features = layer_features[reference_layer]
    if len(reference_features.shape) > 2:
        reference_features = reference_features.reshape(reference_features.shape[0], -1)
    
    # Compute divergence metrics for each layer
    divergence_metrics = []
    layer_names = []
    
    for layer in layers:
        if layer == reference_layer:
            continue
            
        features = layer_features[layer]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Ensure same shape for comparison
        min_samples = min(reference_features.shape[0], features.shape[0])
        ref_subset = reference_features[:min_samples]
        feat_subset = features[:min_samples]
        
        # Compute various divergence metrics
        # 1. Average cosine distance
        cosine_sims = []
        for i in range(min_samples):
            cos_sim = np.dot(ref_subset[i], feat_subset[i]) / (
                np.linalg.norm(ref_subset[i]) * np.linalg.norm(feat_subset[i]) + 1e-8
            )
            cosine_sims.append(cos_sim)
        avg_cosine_distance = 1 - np.mean(cosine_sims)
        
        # 2. Average L2 distance
        l2_distances = np.linalg.norm(feat_subset - ref_subset, axis=1)
        avg_l2_distance = np.mean(l2_distances)
        
        # 3. Correlation coefficient
        correlation = np.corrcoef(ref_subset.flatten(), feat_subset.flatten())[0, 1]
        
        divergence_metrics.append({
            'cosine_distance': avg_cosine_distance,
            'l2_distance': avg_l2_distance,
            'correlation': correlation
        })
        layer_names.append(layer)
    
    # Plot divergence metrics
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Cosine distance from reference
    plt.subplot(3, 1, 1)
    cosine_distances = [m['cosine_distance'] for m in divergence_metrics]
    plt.plot(range(len(layer_names)), cosine_distances, 'o-', color='blue')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    plt.title(f'Cosine Distance from {reference_layer} (CNN-Transformer Boundary)')
    plt.ylabel('Cosine Distance')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: L2 distance from reference
    plt.subplot(3, 1, 2)
    l2_distances = [m['l2_distance'] for m in divergence_metrics]
    plt.plot(range(len(layer_names)), l2_distances, 'o-', color='red')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    plt.title(f'L2 Distance from {reference_layer}')
    plt.ylabel('L2 Distance')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Correlation with reference
    plt.subplot(3, 1, 3)
    correlations = [m['correlation'] for m in divergence_metrics]
    plt.plot(range(len(layer_names)), correlations, 'o-', color='green')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    plt.title(f'Correlation with {reference_layer}')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_divergence_{model_name}_n{num_files}.png')
    plt.close()
    
    return divergence_metrics

def plot_cca_analysis(layer_features, output_dir, model_name, num_files):
    """Plot CCA analysis between layers."""
    # Sort layers numerically
    def get_layer_number(layer_name):
        return int(layer_name.split('_')[1]) if layer_name.startswith('layer_') else int(layer_name.split('_')[1])
    
    layers = sorted(layer_features.keys(), key=get_layer_number)
    n_layers = len(layers)
    
    # Compute CCA correlations between consecutive layers
    cca_correlations = []
    for i in range(n_layers - 1):
        layer1 = layers[i]
        layer2 = layers[i + 1]
        
        # Get features for both layers
        features1 = layer_features[layer1]
        features2 = layer_features[layer2]
        
        # Reshape to 2D if needed
        if len(features1.shape) > 2:
            features1 = features1.reshape(features1.shape[0], -1)
        if len(features2.shape) > 2:
            features2 = features2.reshape(features2.shape[0], -1)
        
        # Ensure same number of samples
        min_samples = min(features1.shape[0], features2.shape[0])
        features1 = features1[:min_samples]
        features2 = features2[:min_samples]
        
        # Initialize CCA with appropriate number of components
        n_components = min(20, min_samples - 1)  # Ensure n_components is less than n_samples
        cca = CCA(n_components=n_components)
        
        try:
            # Fit CCA and get canonical correlations
            cca.fit(features1, features2)
            # Get the canonical correlations
            X_c, Y_c = cca.transform(features1, features2)
            # Compute correlations between canonical variates
            correlations = [np.corrcoef(X_c[:, j], Y_c[:, j])[0, 1] for j in range(X_c.shape[1])]
            cca_correlations.append(np.mean(correlations))
        except Exception as e:
            print(f"Warning: CCA failed for {layer1}-{layer2}: {e}")
            cca_correlations.append(0.0)
    
    # Plot CCA correlations
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_layers - 1), cca_correlations, 'o-')
    plt.xticks(range(n_layers - 1), [f"{layers[i]}-{layers[i+1]}" for i in range(n_layers - 1)], rotation=45)
    plt.title(f'CCA Correlations Between Consecutive Layers\nModel: {model_name}')
    plt.xlabel('Layer Pairs')
    plt.ylabel('Mean CCA Correlation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cca_correlations_{model_name}_n{num_files}.png')
    plt.close()
    
    # Create heatmap of CCA correlations between all layer pairs
    cca_matrix = np.zeros((n_layers, n_layers))
    np.fill_diagonal(cca_matrix, 1.0)  # Perfect correlation with itself
    
    for i in range(n_layers):
        for j in range(n_layers):
            if i != j:
                features1 = layer_features[layers[i]]
                features2 = layer_features[layers[j]]
                
                # Reshape to 2D if needed
                if len(features1.shape) > 2:
                    features1 = features1.reshape(features1.shape[0], -1)
                if len(features2.shape) > 2:
                    features2 = features2.reshape(features2.shape[0], -1)
                
                # Ensure same number of samples
                min_samples = min(features1.shape[0], features2.shape[0])
                features1 = features1[:min_samples]
                features2 = features2[:min_samples]
                
                # Initialize CCA with appropriate number of components
                n_components = min(20, min_samples - 1)
                cca = CCA(n_components=n_components)
                
                try:
                    # Fit CCA
                    cca.fit(features1, features2)
                    X_c, Y_c = cca.transform(features1, features2)
                    correlations = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(X_c.shape[1])]
                    cca_matrix[i, j] = np.mean(correlations)
                except Exception as e:
                    print(f"Warning: CCA failed for {layers[i]}-{layers[j]}: {e}")
                    cca_matrix[i, j] = 0.0
    
    # Plot CCA correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cca_matrix, 
                xticklabels=layers,
                yticklabels=layers,
                cmap='viridis',
                annot=True,
                fmt='.2f',
                vmin=0, vmax=1)
    plt.title(f'CCA Correlation Matrix Between All Layers\nModel: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cca_matrix_{model_name}_n{num_files}.png')
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
    
    # Check if we have enough samples
    min_samples = 5  # Minimum samples needed for meaningful analysis
    if args.num_files < min_samples:
        print(f"Warning: Number of files ({args.num_files}) is less than recommended minimum ({min_samples})")
        print("Some visualizations may be skipped or may not be meaningful")
    
    # Load features
    layer_features = load_features(args.features_dir, args.num_files)
    
    # Generate visualizations
    print("Generating visualizations...")
    # plot_feature_distributions(layer_features, args.output_dir, args.model_name, args.num_files)
    plot_layer_similarity(layer_features, args.output_dir, args.model_name, args.num_files)
    plot_dimensionality_reduction(layer_features, args.output_dir, args.model_name, args.num_files)
    # plot_layer_statistics(layer_features, args.output_dir, args.model_name, args.num_files)
    # plot_cca_analysis(layer_features, args.output_dir, args.model_name, args.num_files)
    analyze_feature_divergence(layer_features, args.output_dir, args.model_name, args.num_files)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
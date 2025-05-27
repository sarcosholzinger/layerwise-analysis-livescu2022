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
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from sklearn.linear_model import LinearRegression

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
    feature_files = list(features_dir.glob("*_complete_features.npz"))
    
    # Take only the first num_files
    feature_files = feature_files[:num_files]
    
    # Dictionary to store features for each layer
    layer_features = {}
    max_lengths = {}  # Track max length for each layer
    original_lengths = {}  # Track original lengths before padding
    
    print(f"Loading features from {len(feature_files)} files...")
    for chkpnt_file_path in feature_files:
        print(f"Processing {chkpnt_file_path}")
        layer_features_contextualized = np.load(chkpnt_file_path)
            
        for layer in layer_features_contextualized.files:
            # Only include transformer layers from input to layer 11
            if layer == 'transformer_input' or (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11):
                if layer not in layer_features:
                    layer_features[layer] = []
                    max_lengths[layer] = 0
                    original_lengths[layer] = []  # Initialize list for original lengths
                features = layer_features_contextualized[layer]
                # Get the time dimension (second dimension for 3D, first for 2D)
                time_dim = features.shape[1] if len(features.shape) == 3 else features.shape[0]
                max_lengths[layer] = max(max_lengths[layer], time_dim)
                original_lengths[layer].append(time_dim)  # Store original length
                layer_features[layer].append(features)
                print(f"Layer {layer}: shape {features.shape}, max_length {max_lengths[layer]}")
    
    # Pad and concatenate features
    for layer in layer_features:
        print(f"Padding layer {layer} to length {max_lengths[layer]}")
        padded_features = [pad_features(f, max_lengths[layer]) for f in layer_features[layer]]
        layer_features[layer] = np.concatenate(padded_features, axis=0)
        print(f"Final shape for layer {layer}: {layer_features[layer].shape}")
    
    return layer_features, original_lengths  # Return both features and original lengths

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

def get_layer_number(layer_name):
    """Get layer number for sorting, handling transformer input and layers."""
    if layer_name == 'transformer_input':
        return -1  # Put input layer first
    elif layer_name.startswith('transformer_layer_'):
        layer_num = int(layer_name.split('_')[-1])
        if layer_num <= 11:  # Only include up to layer 11
            return layer_num
    return float('inf')  # Put other layers at the end - ignore layers 12 and above and exclude from analysis

def compute_cka(X, Y):
    """
    Compute CKA (Centered Kernel Alignment) between two representations.
    
    Args:
        X: (n_samples, n_features1)
        Y: (n_samples, n_features2)
    
    Returns:
        CKA similarity score
    """
    # Center the matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # Compute Gram matrices (linear kernel)
    K = X @ X.T  # (n_samples, n_samples)
    L = Y @ Y.T  # (n_samples, n_samples)
    
    # Center the Gram matrices
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = np.trace(K_centered @ L_centered) / (n - 1)**2
    
    # Compute normalization
    var_K = np.trace(K_centered @ K_centered) / (n - 1)**2
    var_L = np.trace(L_centered @ L_centered) / (n - 1)**2
    
    # Compute CKA
    cka = hsic / np.sqrt(var_K * var_L + 1e-8)
    
    return cka

def compute_cka_without_padding(features1, features2, orig_lens1, orig_lens2):
    """
    Compute CKA excluding padded time steps.
    
    Args:
        features1, features2: Padded features (batch, time, dim)
        orig_lens1, orig_lens2: Lists of original lengths for each batch item
    """
    all_X = []
    all_Y = []
    
    batch_size = features1.shape[0]
    
    for b in range(batch_size):
        # Get original length for this batch item
        orig_len1 = orig_lens1[b] if b < len(orig_lens1) else features1.shape[1]
        orig_len2 = orig_lens2[b] if b < len(orig_lens2) else features2.shape[1]
        orig_len = min(orig_len1, orig_len2)  # Use minimum to ensure both are valid
        
        # Extract only non-padded time steps
        X = features1[b, :orig_len, :]  # (time, dim)
        Y = features2[b, :orig_len, :]
        
        all_X.append(X)
        all_Y.append(Y)
    
    # Concatenate all valid samples
    X = np.vstack(all_X)  # (total_valid_samples, dim)
    Y = np.vstack(all_Y)
    
    # Ensure same number of samples
    min_samples = min(X.shape[0], Y.shape[0])
    X = X[:min_samples]
    Y = Y[:min_samples]
    
    return compute_cka(X, Y)

def plot_layer_similarity_improved(layer_features, original_lengths, output_dir, model_name, num_files):
    """Plot improved similarity matrix between layers."""
    
    # Filter and sort layers (only include transformer input and layers 0-11)
    layers = sorted([layer for layer in layer_features.keys() 
                    if layer == 'transformer_input' or 
                    (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                   key=get_layer_number)
    n_layers = len(layers)
    
    # Multiple similarity metrics
    cosine_matrix = np.zeros((n_layers, n_layers))
    correlation_matrix = np.zeros((n_layers, n_layers))
    cka_matrix = np.zeros((n_layers, n_layers))
    
    for i, layer1 in enumerate(layers):
        for j, layer2 in enumerate(layers):
            features1 = layer_features[layer1]
            features2 = layer_features[layer2]
            
            # Ensure same batch size
            min_batch = min(features1.shape[0], features2.shape[0])
            features1 = features1[:min_batch]
            features2 = features2[:min_batch]
            
            # Method 1: Average cosine similarity
            f1_avg = features1.mean(axis=1)  # (batch, features)
            f2_avg = features2.mean(axis=1)  # (batch, features)
            
            # Ensure same feature dimension for cosine/correlation
            min_features = min(f1_avg.shape[1], f2_avg.shape[1])   # --??
            f1_avg_truncated = f1_avg[:, :min_features]
            f2_avg_truncated = f2_avg[:, :min_features]
            
            # Compute cosine similarity
            cos_sims = []
            for b in range(min_batch):
                cos_sim = np.dot(f1_avg_truncated[b], f2_avg_truncated[b]) / (
                    np.linalg.norm(f1_avg_truncated[b]) * np.linalg.norm(f2_avg_truncated[b]) + 1e-8
                )
                cos_sims.append(cos_sim)
            cosine_matrix[i, j] = np.mean(cos_sims)
            
            # Method 2: Correlation
            correlation_matrix[i, j] = np.corrcoef(
                f1_avg_truncated.flatten(), f2_avg_truncated.flatten()
            )[0, 1]
            
            # Method 3: CKA - works with different dimensions!
            if layer1 in original_lengths and layer2 in original_lengths:
                cka_matrix[i, j] = compute_cka_without_padding(
                    features1, features2, 
                    original_lengths[layer1], 
                    original_lengths[layer2]
                )
            else:
                # Fallback to original method if lengths not available
                X = features1.reshape(features1.shape[0] * features1.shape[1], -1)
                Y = features2.reshape(features2.shape[0] * features2.shape[1], -1)
                min_samples = min(X.shape[0], Y.shape[0])
                X = X[:min_samples]
                Y = Y[:min_samples]
                cka_matrix[i, j] = compute_cka(X, Y)
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Cosine similarity
    sns.heatmap(cosine_matrix, ax=axes[0], 
                xticklabels=layers, yticklabels=layers,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'Similarity'})
    axes[0].set_title('Cosine Similarity (Time-Averaged)', fontsize=14)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Layer')
    
    # Correlation
    sns.heatmap(correlation_matrix, ax=axes[1],
                xticklabels=layers, yticklabels=layers,
                cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Correlation (Time-Averaged)', fontsize=14)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Layer')
    
    # CKA
    sns.heatmap(cka_matrix, ax=axes[2],
                xticklabels=layers, yticklabels=layers,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'CKA Similarity'})
    axes[2].set_title('CKA (Centered Kernel Alignment)', fontsize=14)
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Layer')
    
    plt.suptitle(f'Layer Similarity Analysis - {model_name} (n={num_files} files)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_similarity_improved_{model_name}_n{num_files}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save individual metrics for detailed analysis
    np.save(f'{output_dir}/cosine_similarity_matrix.npy', cosine_matrix)
    np.save(f'{output_dir}/correlation_matrix.npy', correlation_matrix)
    np.save(f'{output_dir}/cka_matrix.npy', cka_matrix)
    
    return cosine_matrix, correlation_matrix, cka_matrix

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
        if layer_name == 'transformer_input':
            return -1  # Put input layer first
        elif layer_name.startswith('transformer_layer_'):
            layer_num = int(layer_name.split('_')[-1])
            if layer_num <= 11:  # Only include up to layer 11
                return layer_num
        return float('inf')  # Put other layers at the end
    
    # Filter and sort layers (only include transformer input and layers 0-11)
    layers = sorted([layer for layer in layer_features.keys() 
                    if layer == 'transformer_input' or 
                    (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                   key=get_layer_number)
    
    # Use the first layer as reference
    reference_layer = layers[0]
    print(f"Using {reference_layer} as reference layer for divergence analysis")
    
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

def compute_temporal_similarities(layer_features, original_lengths, window_size=10, stride=5):
    """
    Compute layer similarities for sliding windows across time.
    """
    # Filter and sort layers
    layers = sorted([layer for layer in layer_features.keys() 
                    if layer == 'transformer_input' or 
                    (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                   key=get_layer_number)
    
    n_layers = len(layers)
    
    # Get the minimum time steps across all layers
    min_time_steps = min([features.shape[1] for features in layer_features.values()])
    
    # Calculate number of windows
    n_windows = (min_time_steps - window_size) // stride + 1
    
    # Store similarity matrices for each time window
    cosine_matrices = []
    correlation_matrices = []
    cka_matrices = []
    
    print(f"Computing similarities for {n_windows} time windows...")
    
    for window_idx in tqdm(range(n_windows)):
        start_time = window_idx * stride
        end_time = start_time + window_size
        
        # Compute similarities for this time window
        cosine_matrix = np.zeros((n_layers, n_layers))
        correlation_matrix = np.zeros((n_layers, n_layers))
        cka_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                # Extract features for this time window
                features1 = layer_features[layer1][:, start_time:end_time, :]
                features2 = layer_features[layer2][:, start_time:end_time, :]
                
                # Ensure same batch size
                min_batch = min(features1.shape[0], features2.shape[0])
                features1 = features1[:min_batch]
                features2 = features2[:min_batch]
                
                # Average over time window
                f1_avg = features1.mean(axis=1)
                f2_avg = features2.mean(axis=1)
                
                # Ensure same feature dimension
                min_features = min(f1_avg.shape[1], f2_avg.shape[1])
                f1_avg = f1_avg[:, :min_features]
                f2_avg = f2_avg[:, :min_features]
                
                # Compute cosine similarity
                cos_sims = []
                for b in range(min_batch):
                    cos_sim = np.dot(f1_avg[b], f2_avg[b]) / (
                        np.linalg.norm(f1_avg[b]) * np.linalg.norm(f2_avg[b]) + 1e-8
                    )
                    cos_sims.append(cos_sim)
                cosine_matrix[i, j] = np.mean(cos_sims)
                
                # Compute correlation
                if f1_avg.size > 0 and f2_avg.size > 0:
                    correlation_matrix[i, j] = np.corrcoef(
                        f1_avg.flatten(), f2_avg.flatten()
                    )[0, 1]
                
                # Compute CKA for this window
                if layer1 in original_lengths and layer2 in original_lengths:
                    # Create temporary lengths for this window
                    temp_lens1 = []
                    temp_lens2 = []
                    
                    for orig_len1, orig_len2 in zip(original_lengths[layer1], original_lengths[layer2]):
                        # Adjust lengths for this window
                        valid_len1 = max(0, min(orig_len1 - start_time, window_size))
                        valid_len2 = max(0, min(orig_len2 - start_time, window_size))
                        temp_lens1.append(valid_len1)
                        temp_lens2.append(valid_len2)
                    
                    if any(l > 0 for l in temp_lens1) and any(l > 0 for l in temp_lens2):
                        cka_matrix[i, j] = compute_cka_without_padding(
                            features1, features2,
                            temp_lens1, temp_lens2
                        )
                    else:
                        cka_matrix[i, j] = 0.0
                else:
                    # Fallback to original method
                    X = features1.reshape(-1, features1.shape[-1])
                    Y = features2.reshape(-1, features2.shape[-1])
                    min_samples = min(X.shape[0], Y.shape[0])
                    if min_samples > 1:
                        cka_matrix[i, j] = compute_cka(X[:min_samples], Y[:min_samples])
                    else:
                        cka_matrix[i, j] = 0.0
        
        cosine_matrices.append(cosine_matrix)
        correlation_matrices.append(correlation_matrix)
        cka_matrices.append(cka_matrix)
    
    return {
        'cosine': cosine_matrices,
        'correlation': correlation_matrices,
        'cka': cka_matrices,
        'layers': layers,
        'window_size': window_size,
        'stride': stride
    }

def create_similarity_animation(temporal_similarities, output_dir, model_name, metric='cosine'):
    """
    Create an animation showing how similarities evolve over time.
    """
    matrices = temporal_similarities[metric]
    layers = temporal_similarities['layers']
    window_size = temporal_similarities['window_size']
    stride = temporal_similarities['stride']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the initial heatmap
    im = ax.imshow(matrices[0], cmap='viridis', vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric.capitalize()} Similarity')
    
    # Set ticks and labels
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_yticklabels(layers)
    
    # Add title
    title = ax.set_title(f'{metric.capitalize()} Similarity - {model_name}\nTime: 0-{window_size} steps')
    
    # Add text annotations for values
    text_annotations = []
    for i in range(len(layers)):
        text_row = []
        for j in range(len(layers)):
            text = ax.text(j, i, f'{matrices[0][i, j]:.2f}',
                         ha='center', va='center', color='white', fontsize=8)
            text_row.append(text)
        text_annotations.append(text_row)
    
    def update(frame):
        # Update the heatmap data
        im.set_array(matrices[frame])
        
        # Update title with current time window
        start_time = frame * stride
        end_time = start_time + window_size
        title.set_text(f'{metric.capitalize()} Similarity - {model_name}\nTime: {start_time}-{end_time} steps')
        
        # Update text annotations
        for i in range(len(layers)):
            for j in range(len(layers)):
                text_annotations[i][j].set_text(f'{matrices[frame][i, j]:.2f}')
        
        return [im, title] + [text for row in text_annotations for text in row]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(matrices), interval=200, blit=True)
    
    # Save as GIF
    output_path = f'{output_dir}/layer_similarity_{metric}_{model_name}_animation.gif'
    anim.save(output_path, writer=PillowWriter(fps=5))
    plt.close()
    
    print(f"Animation saved to {output_path}")
    
    return output_path

def plot_padding_ratios(layer_features, original_lengths, output_dir, model_name, num_files):
    """Plot the ratio of valid (non-padded) time steps across layers."""
    # Get all layers
    layers = sorted([layer for layer in layer_features.keys() 
                    if layer == 'transformer_input' or 
                    (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                   key=get_layer_number)
    
    # Get max sequence length
    max_length = max(features.shape[1] for features in layer_features.values())
    
    # Initialize padding ratio matrices
    # Aggregate across all files
    aggregate_padding_ratios = np.zeros((len(layers), max_length))
    # Per-file padding ratios
    per_file_padding_ratios = []
    
    # Compute padding ratios for each layer and time step
    for i, layer in enumerate(layers):
        if layer in original_lengths:
            # For aggregate
            for t in range(max_length):
                valid_count = sum(1 for length in original_lengths[layer] if t < length)
                aggregate_padding_ratios[i, t] = valid_count / len(original_lengths[layer])
            
            # For per-file
            file_ratios = np.zeros((len(original_lengths[layer]), max_length))
            for file_idx, length in enumerate(original_lengths[layer]):
                file_ratios[file_idx, :length] = 1.0  # Valid up to original length
            per_file_padding_ratios.append(file_ratios)
    
    # Create visualizations
    # 1. Aggregate heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(aggregate_padding_ratios, 
                xticklabels=50,  # Show every 50th time step
                yticklabels=layers,
                cmap='viridis',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Ratio of Valid Sequences'})
    plt.title(f'Aggregate Padding Ratios Across Layers and Time Steps\n{model_name} (n={num_files} files)')
    plt.xlabel('Time Step')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/padding_ratios_aggregate_{model_name}_n{num_files}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-file heatmaps
    for layer_idx, layer in enumerate(layers):
        if layer in original_lengths:
            plt.figure(figsize=(15, 8))
            sns.heatmap(per_file_padding_ratios[layer_idx], 
                        xticklabels=50,
                        yticklabels=[f'File {i+1}' for i in range(len(original_lengths[layer]))],
                        cmap='viridis',
                        vmin=0, vmax=1,
                        cbar_kws={'label': 'Valid Sequence'})
            plt.title(f'Per-File Padding Ratios for {layer}\n{model_name} (n={num_files} files)')
            plt.xlabel('Time Step')
            plt.ylabel('File')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/padding_ratios_per_file_{layer}_{model_name}_n{num_files}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Save the raw data
    np.save(f'{output_dir}/padding_ratios_aggregate_{model_name}_n{num_files}.npy', aggregate_padding_ratios)
    for layer_idx, layer in enumerate(layers):
        if layer in original_lengths:
            np.save(f'{output_dir}/padding_ratios_per_file_{layer}_{model_name}_n{num_files}.npy', 
                   per_file_padding_ratios[layer_idx])
    
    return aggregate_padding_ratios, per_file_padding_ratios

# New functions for partial correlation and kernel analysis
def compute_partial_correlation(X, Y, Z):
    """
    Compute partial correlation between X and Y given Z.
    
    Partial correlation r_{XY|Z} = correlation between X and Y after removing 
    the linear effect of Z from both X and Y.
    
    Args:
        X: (n_samples, n_features_X)
        Y: (n_samples, n_features_Y)
        Z: (n_samples, n_features_Z) - conditioning variable (CNN output)
    
    Returns:
        Partial correlation coefficient
    """
    # Ensure 2D arrays
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    
    # Regress out Z from X
    reg_X = LinearRegression()
    reg_X.fit(Z, X)
    residuals_X = X - reg_X.predict(Z)
    
    # Regress out Z from Y
    reg_Y = LinearRegression()
    reg_Y.fit(Z, Y)
    residuals_Y = Y - reg_Y.predict(Z)
    
    # Compute correlation between residuals
    # Flatten residuals for correlation computation
    residuals_X_flat = residuals_X.flatten()
    residuals_Y_flat = residuals_Y.flatten()
    
    if len(residuals_X_flat) > 1 and len(residuals_Y_flat) > 1:
        partial_corr = np.corrcoef(residuals_X_flat, residuals_Y_flat)[0, 1]
    else:
        partial_corr = 0.0
    
    return partial_corr

def compute_conditional_cka(X, Y, Z, method='residual'):
    """
    Compute CKA between X and Y conditioned on Z (CNN output).
    
    Two methods available:
    1. 'residual': Compute CKA on residuals after regressing out Z
    2. 'partial': Compute partial CKA using kernel matrices
    
    Args:
        X: (n_samples, n_features_X)
        Y: (n_samples, n_features_Y)
        Z: (n_samples, n_features_Z) - CNN output features
        method: 'residual' or 'partial'
    
    Returns:
        Conditional CKA score
    """
    if method == 'residual':
        # Method 1: Regress out Z from both X and Y, then compute CKA on residuals
        
        # Regress out Z from X
        reg_X = LinearRegression()
        reg_X.fit(Z, X)
        X_residual = X - reg_X.predict(Z)
        
        # Regress out Z from Y
        reg_Y = LinearRegression()
        reg_Y.fit(Z, Y)
        Y_residual = Y - reg_Y.predict(Z)
        
        # Compute CKA on residuals
        return compute_cka(X_residual, Y_residual)
    
    elif method == 'partial':
        # Method 2: Partial CKA using kernel regression
        
        # Center all matrices
        X_c = X - X.mean(axis=0, keepdims=True)
        Y_c = Y - Y.mean(axis=0, keepdims=True)
        Z_c = Z - Z.mean(axis=0, keepdims=True)
        
        # Compute Gram matrices
        K_X = X_c @ X_c.T
        K_Y = Y_c @ Y_c.T
        K_Z = Z_c @ Z_c.T
        
        # Center Gram matrices
        n = K_X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_X_centered = H @ K_X @ H
        K_Y_centered = H @ K_Y @ H
        K_Z_centered = H @ K_Z @ H
        
        # Compute partial kernel matrices (remove effect of Z)
        # Using the formula: K_{X|Z} = K_X - K_{XZ} K_Z^{-1} K_{ZX}
        # We use regularized inverse for numerical stability
        epsilon = 1e-5
        K_Z_inv = np.linalg.inv(K_Z_centered + epsilon * np.eye(n))
        
        K_X_partial = K_X_centered - K_X_centered @ K_Z_inv @ K_Z_centered
        K_Y_partial = K_Y_centered - K_Y_centered @ K_Z_inv @ K_Z_centered
        
        # Compute CKA on partial kernels
        hsic_XY = np.trace(K_X_partial @ K_Y_partial) / (n - 1)**2
        hsic_XX = np.trace(K_X_partial @ K_X_partial) / (n - 1)**2
        hsic_YY = np.trace(K_Y_partial @ K_Y_partial) / (n - 1)**2
        
        # Compute conditional CKA
        conditional_cka = hsic_XY / np.sqrt(hsic_XX * hsic_YY + 1e-8)
        
        return conditional_cka
    
    else:
        raise ValueError(f"Unknown method: {method}")

def plot_conditional_layer_similarity(layer_features, original_lengths, output_dir, model_name, num_files):
    """
    Plot similarity matrices conditioned on CNN output (transformer_input).
    """
    # Get CNN output layer
    cnn_output_layer = 'transformer_input'
    if cnn_output_layer not in layer_features:
        print(f"Warning: {cnn_output_layer} not found in features. Using first layer as reference.")
        cnn_output_layer = sorted(layer_features.keys())[0]
    
    # Get CNN output features
    cnn_features = layer_features[cnn_output_layer]
    
    # Filter and sort layers (excluding CNN output itself)
    layers = sorted([layer for layer in layer_features.keys() 
                    if layer != cnn_output_layer and
                    (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                   key=get_layer_number)
    n_layers = len(layers)
    
    # Initialize matrices
    conditional_correlation_matrix = np.zeros((n_layers, n_layers))
    conditional_cka_residual_matrix = np.zeros((n_layers, n_layers))
    conditional_cka_partial_matrix = np.zeros((n_layers, n_layers))
    
    # Also compute unconditional for comparison
    unconditional_correlation_matrix = np.zeros((n_layers, n_layers))
    unconditional_cka_matrix = np.zeros((n_layers, n_layers))
    
    print(f"Computing conditional similarities with CNN output ({cnn_output_layer}) as conditioning variable...")
    
    # Create progress bar for layer pairs
    total_pairs = n_layers * n_layers
    with tqdm(total=total_pairs, desc="Computing layer similarities") as pbar:
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                pbar.set_description(f"Processing {layer1} vs {layer2}")
                
                features1 = layer_features[layer1]
                features2 = layer_features[layer2]
                
                # Ensure same batch size
                min_batch = min(features1.shape[0], features2.shape[0], cnn_features.shape[0])
                features1 = features1[:min_batch]
                features2 = features2[:min_batch]
                cnn_features_batch = cnn_features[:min_batch]
                
                # Handle padding if original lengths are available
                if layer1 in original_lengths and layer2 in original_lengths and cnn_output_layer in original_lengths:
                    # Use padding-aware approach
                    all_X = []
                    all_Y = []
                    all_Z = []
                    
                    for b in range(min_batch):
                        # Get valid lengths
                        valid_len = min(
                            original_lengths[layer1][b] if b < len(original_lengths[layer1]) else features1.shape[1],
                            original_lengths[layer2][b] if b < len(original_lengths[layer2]) else features2.shape[1],
                            original_lengths[cnn_output_layer][b] if b < len(original_lengths[cnn_output_layer]) else cnn_features.shape[1]
                        )
                        
                        if valid_len > 0:
                            all_X.append(features1[b, :valid_len, :])
                            all_Y.append(features2[b, :valid_len, :])
                            all_Z.append(cnn_features_batch[b, :valid_len, :])
                    
                    # Stack all valid samples
                    if all_X and all_Y and all_Z:
                        X = np.vstack(all_X)
                        Y = np.vstack(all_Y)
                        Z = np.vstack(all_Z)
                    else:
                        pbar.update(1)
                        continue
                else:
                    # Reshape to 2D
                    X = features1.reshape(features1.shape[0] * features1.shape[1], -1)
                    Y = features2.reshape(features2.shape[0] * features2.shape[1], -1)
                    Z = cnn_features_batch.reshape(cnn_features_batch.shape[0] * cnn_features_batch.shape[1], -1)
                
                # Ensure same number of samples
                min_samples = min(X.shape[0], Y.shape[0], Z.shape[0])
                if min_samples < 2:
                    pbar.update(1)
                    continue
                    
                X = X[:min_samples]
                Y = Y[:min_samples]
                Z = Z[:min_samples]
                
                # Compute unconditional metrics for comparison
                try:
                    unconditional_correlation_matrix[i, j] = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
                    unconditional_cka_matrix[i, j] = compute_cka(X, Y)
                except:
                    unconditional_correlation_matrix[i, j] = 0.0
                    unconditional_cka_matrix[i, j] = 0.0
                
                # Compute conditional metrics
                try:
                    # Partial correlation
                    conditional_correlation_matrix[i, j] = compute_partial_correlation(X, Y, Z)
                    
                    # Conditional CKA (residual method)
                    conditional_cka_residual_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='residual')
                    
                    # Conditional CKA (partial method)
                    conditional_cka_partial_matrix[i, j] = compute_conditional_cka(X, Y, Z, method='partial')
                    
                except Exception as e:
                    print(f"\nWarning: Failed to compute conditional metrics for {layer1}-{layer2}: {e}")
                    conditional_correlation_matrix[i, j] = 0.0
                    conditional_cka_residual_matrix[i, j] = 0.0
                    conditional_cka_partial_matrix[i, j] = 0.0
                
                pbar.update(1)
    
    print("\nGenerating visualizations...")
    
    # Create visualization comparing conditional vs unconditional
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Row 1: Unconditional metrics
    sns.heatmap(unconditional_correlation_matrix, ax=axes[0, 0],
                xticklabels=layers, yticklabels=layers,
                cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    axes[0, 0].set_title('Unconditional Correlation', fontsize=14)
    
    sns.heatmap(unconditional_cka_matrix, ax=axes[0, 1],
                xticklabels=layers, yticklabels=layers,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'CKA'})
    axes[0, 1].set_title('Unconditional CKA', fontsize=14)
    
    # Difference plot (unconditional - conditional)
    diff_correlation = unconditional_correlation_matrix - conditional_correlation_matrix
    sns.heatmap(diff_correlation, ax=axes[0, 2],
                xticklabels=layers, yticklabels=layers,
                cmap='coolwarm', center=0, annot=True, fmt='.2f',
                cbar_kws={'label': 'Difference'})
    axes[0, 2].set_title('Correlation Difference\n(Unconditional - Conditional)', fontsize=14)
    
    # Row 2: Conditional metrics
    sns.heatmap(conditional_correlation_matrix, ax=axes[1, 0],
                xticklabels=layers, yticklabels=layers,
                cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'Partial Correlation'})
    axes[1, 0].set_title(f'Partial Correlation | {cnn_output_layer}', fontsize=14)
    
    sns.heatmap(conditional_cka_residual_matrix, ax=axes[1, 1],
                xticklabels=layers, yticklabels=layers,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'Conditional CKA'})
    axes[1, 1].set_title(f'Conditional CKA (Residual) | {cnn_output_layer}', fontsize=14)
    
    sns.heatmap(conditional_cka_partial_matrix, ax=axes[1, 2],
                xticklabels=layers, yticklabels=layers,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'Conditional CKA'})
    axes[1, 2].set_title(f'Conditional CKA (Partial) | {cnn_output_layer}', fontsize=14)
    
    plt.suptitle(f'Conditional vs Unconditional Layer Similarity - {model_name} (n={num_files} files)\nConditioned on {cnn_output_layer}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/conditional_layer_similarity_{model_name}_n{num_files}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save matrices
    np.save(f'{output_dir}/conditional_correlation_matrix.npy', conditional_correlation_matrix)
    np.save(f'{output_dir}/conditional_cka_residual_matrix.npy', conditional_cka_residual_matrix)
    np.save(f'{output_dir}/conditional_cka_partial_matrix.npy', conditional_cka_partial_matrix)
    
    # Create a summary plot showing the effect of conditioning
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average correlation with and without conditioning
    avg_uncond_corr = []
    avg_cond_corr = []
    for i in range(n_layers):
        # Average correlation with other layers (excluding self)
        mask = np.ones(n_layers, dtype=bool)
        mask[i] = False
        avg_uncond_corr.append(np.mean(np.abs(unconditional_correlation_matrix[i, mask])))
        avg_cond_corr.append(np.mean(np.abs(conditional_correlation_matrix[i, mask])))
    
    x = np.arange(n_layers)
    width = 0.35
    axes[0].bar(x - width/2, avg_uncond_corr, width, label='Unconditional', alpha=0.8)
    axes[0].bar(x + width/2, avg_cond_corr, width, label='Conditional', alpha=0.8)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Average Absolute Correlation')
    axes[0].set_title('Effect of CNN Conditioning on Correlation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Similar for CKA
    avg_uncond_cka = []
    avg_cond_cka = []
    for i in range(n_layers):
        mask = np.ones(n_layers, dtype=bool)
        mask[i] = False
        avg_uncond_cka.append(np.mean(unconditional_cka_matrix[i, mask]))
        avg_cond_cka.append(np.mean(conditional_cka_residual_matrix[i, mask]))
    
    axes[1].bar(x - width/2, avg_uncond_cka, width, label='Unconditional', alpha=0.8)
    axes[1].bar(x + width/2, avg_cond_cka, width, label='Conditional', alpha=0.8)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Average CKA')
    axes[1].set_title('Effect of CNN Conditioning on CKA')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Impact of CNN Conditioning on Layer Similarities - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/conditioning_effect_summary_{model_name}_n{num_files}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return conditional_correlation_matrix, conditional_cka_residual_matrix, conditional_cka_partial_matrix

# Add this function to be called from main()
def analyze_cnn_influence(layer_features, original_lengths, output_dir, model_name, num_files):
    """
    Analyze how much each layer's representation is influenced by the CNN output.
    """
    cnn_output_layer = 'transformer_input'
    if cnn_output_layer not in layer_features:
        print(f"Warning: {cnn_output_layer} not found. Skipping CNN influence analysis.")
        return
    
    cnn_features = layer_features[cnn_output_layer]
    
    # Get all transformer layers
    layers = sorted([layer for layer in layer_features.keys() 
                    if layer != cnn_output_layer and
                    (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                   key=get_layer_number)
    
    # Compute R² values showing how much variance in each layer is explained by CNN output
    r2_scores = []
    layer_names = []
    
    for layer in layers:
        features = layer_features[layer]
        
        # Ensure same batch size
        min_batch = min(features.shape[0], cnn_features.shape[0])
        features = features[:min_batch]
        cnn_batch = cnn_features[:min_batch]
        
        # Reshape to 2D
        X = features.reshape(-1, features.shape[-1])
        Z = cnn_batch.reshape(-1, cnn_batch.shape[-1])
        
        # Ensure same number of samples
        min_samples = min(X.shape[0], Z.shape[0])
        X = X[:min_samples]
        Z = Z[:min_samples]
        
        # Compute R² for each dimension
        r2_per_dim = []
        for dim in range(X.shape[1]):
            reg = LinearRegression()
            reg.fit(Z, X[:, dim])
            r2 = reg.score(Z, X[:, dim])
            r2_per_dim.append(r2)
        
        # Average R² across dimensions
        avg_r2 = np.mean(r2_per_dim)
        r2_scores.append(avg_r2)
        layer_names.append(layer)
    
    # Plot R² decay
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(layers)), r2_scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Layer')
    plt.ylabel('R² (Variance Explained by CNN Output)')
    plt.title(f'CNN Influence Decay Across Transformer Layers - {model_name}')
    plt.xticks(range(len(layers)), layer_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for R² values
    for i, r2 in enumerate(r2_scores):
        plt.annotate(f'{r2:.3f}', xy=(i, r2), xytext=(0, 5), 
                    textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cnn_influence_decay_{model_name}_n{num_files}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return r2_scores

def main():
    parser = argparse.ArgumentParser(description="Visualize HuBERT features")
    parser.add_argument("--features_dir", type=str, 
                      default="/home/sarcosh1/repos/layerwise-analysis/output/hubert-complete/librispeech_dev-clean_sample1",
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
    print("Loading features...")
    layer_features, original_lengths = load_features(args.features_dir, args.num_files)
    
    # Check if CNN output layer exists
    cnn_output_layer = 'transformer_input'
    if cnn_output_layer not in layer_features:
        print(f"Error: Required layer '{cnn_output_layer}' not found in features.")
        print("Available layers:", list(layer_features.keys()))
        return
    
    # Generate original visualizations
    print("\nGenerating visualizations...")
    
    # # Original unconditional similarity analysis
    # print("\nComputing unconditional layer similarities...")
    # plot_layer_similarity_improved(layer_features, original_lengths, args.output_dir, args.model_name, args.num_files)
    
    # NEW: Conditional similarity analysis
    print("\nComputing CNN-conditioned layer similarities...")
    try:
        plot_conditional_layer_similarity(layer_features, original_lengths, args.output_dir, args.model_name, args.num_files)
    except Exception as e:
        print(f"Error in conditional similarity analysis: {e}")
        print("Continuing with other analyses...")
    
    # NEW: Analyze CNN influence decay
    print("\nAnalyzing CNN influence across layers...")
    try:
        r2_scores = analyze_cnn_influence(layer_features, original_lengths, args.output_dir, args.model_name, args.num_files)
    except Exception as e:
        print(f"Error in CNN influence analysis: {e}")
        r2_scores = None
    
    # # Original divergence analysis (for comparison)
    # print("\nAnalyzing feature divergence from CNN boundary...")
    # analyze_feature_divergence(layer_features, args.output_dir, args.model_name, args.num_files)
    
    # # Add padding ratio visualization
    # print("\nGenerating padding ratio visualization...")
    # padding_ratios = plot_padding_ratios(layer_features, original_lengths, args.output_dir, args.model_name, args.num_files)

    # Print summary of CNN influence
    if r2_scores is not None:
        print("\n=== CNN Influence Summary ===")
        print(f"Average R² (CNN influence) by layer depth:")
        layers = sorted([layer for layer in layer_features.keys() 
                        if layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11],
                       key=get_layer_number)
        for i, (layer, r2) in enumerate(zip(layers, r2_scores)):
            print(f"  {layer}: {r2:.3f} ({(1-r2)*100:.1f}% independent of CNN)")
        print(f"\nOverall CNN influence decay: {r2_scores[0]:.3f} → {r2_scores[-1]:.3f}")
    
    print("\nGenerating temporal animations...")
    temporal_similarities = compute_temporal_similarities(
        layer_features, 
        original_lengths,
        window_size=20,
        stride=10
    )
    
    # Create animations for each metric
    for metric in ['cosine', 'correlation', 'cka']:
        create_similarity_animation(temporal_similarities, args.output_dir, args.model_name, metric)
    
 
    
    print(f"\nAll visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
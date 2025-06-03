import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import argparse
from tqdm import tqdm
import logging
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, List, Tuple, Optional
import pickle
import time
from functools import partial
import multiprocessing as mp

# Import from the main analysis module
from complete_parallel_analysis import GPUParallelSimilarity, FeatureLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalSimilarityAnalyzer:
    """GPU-accelerated temporal similarity analysis with sliding window approach."""
    
    def __init__(self, window_size: int = 20, stride: int = 10, n_gpus: Optional[int] = None):
        self.window_size = window_size
        self.stride = stride
        self.n_gpus = n_gpus if n_gpus is not None else torch.cuda.device_count()
        
        if self.n_gpus == 0:
            logger.warning("No GPUs available, using CPU computation")
        else:
            logger.info(f"Using {self.n_gpus} GPUs for temporal analysis")
    
    def get_layer_number(self, layer_name: str) -> int:
        """Get layer number for sorting."""
        if layer_name == 'transformer_input':
            return -1
        elif layer_name.startswith('transformer_layer_'):
            layer_num = int(layer_name.split('_')[-1])
            if layer_num <= 11:
                return layer_num
        return float('inf')
    
    def compute_temporal_similarities(self, layer_features: Dict[str, np.ndarray],
                                    original_lengths: Dict[str, List[int]],
                                    metrics: List[str] = None) -> Dict:
        """Compute similarities across sliding temporal windows."""
        if metrics is None:
            metrics = ['cosine', 'correlation', 'cka']
        
        # Filter and sort layers
        layers = sorted([layer for layer in layer_features.keys() 
                        if layer == 'transformer_input' or 
                        (layer.startswith('transformer_layer_') and int(layer.split('_')[-1]) <= 11)],
                       key=self.get_layer_number)
        
        n_layers = len(layers)
        
        # Get the minimum time steps across all layers
        min_time_steps = min([features.shape[1] for features in layer_features.values()])
        
        # Calculate number of windows
        n_windows = (min_time_steps - self.window_size) // self.stride + 1
        
        logger.info(f"Computing temporal similarities:")
        logger.info(f"  Layers: {len(layers)}")
        logger.info(f"  Time windows: {n_windows}")
        logger.info(f"  Window size: {self.window_size}")
        logger.info(f"  Stride: {self.stride}")
        
        # Store similarity matrices for each time window
        temporal_similarities = {metric: [] for metric in metrics}
        
        # Process each time window
        for window_idx in tqdm(range(n_windows), desc="Processing time windows"):
            start_time = window_idx * self.stride
            end_time = start_time + self.window_size
            
            # Extract features for this window
            window_features = {}
            for layer in layers:
                window_features[layer] = layer_features[layer][:, start_time:end_time, :]
            
            # Compute similarities for this window
            window_similarities = self._compute_window_similarities(
                window_features, original_lengths, metrics, start_time, end_time
            )
            
            # Store results
            for metric in metrics:
                temporal_similarities[metric].append(window_similarities[metric])
        
        return {
            'similarities': temporal_similarities,
            'layers': layers,
            'window_size': self.window_size,
            'stride': self.stride,
            'n_windows': n_windows,
            'metrics': metrics
        }
    
    def _compute_window_similarities(self, window_features: Dict[str, np.ndarray],
                                   original_lengths: Dict[str, List[int]],
                                   metrics: List[str], start_time: int, end_time: int) -> Dict:
        """Compute similarities for a single time window."""
        layers = list(window_features.keys())
        n_layers = len(layers)
        
        # Initialize similarity matrices for this window
        window_similarities = {metric: np.zeros((n_layers, n_layers)) for metric in metrics}
        
        # Create tasks for parallel processing
        tasks = []
        for i in range(n_layers):
            for j in range(i, n_layers):  # Only upper triangle
                tasks.append((i, j, layers[i], layers[j]))
        
        if self.n_gpus > 0:
            # GPU parallel processing
            results = self._process_window_gpu_parallel(
                tasks, window_features, original_lengths, metrics, start_time
            )
        else:
            # CPU parallel processing
            results = self._process_window_cpu_parallel(
                tasks, window_features, original_lengths, metrics, start_time
            )
        
        # Fill similarity matrices
        for result in results:
            i, j = result['indices']
            for metric in metrics:
                if metric in result:
                    window_similarities[metric][i, j] = result[metric]
                    if i != j:  # Mirror for symmetric matrix
                        window_similarities[metric][j, i] = result[metric]
        
        return window_similarities
    
    def _process_window_gpu_parallel(self, tasks: List[Tuple], 
                                   window_features: Dict[str, np.ndarray],
                                   original_lengths: Dict[str, List[int]],
                                   metrics: List[str], start_time: int) -> List[Dict]:
        """Process window tasks using GPU parallelization."""
        # Distribute tasks across GPUs
        gpu_tasks = []
        for task_id, task in enumerate(tasks):
            gpu_id = task_id % self.n_gpus
            gpu_tasks.append((gpu_id, task))
        
        # Group tasks by GPU
        gpu_task_groups = {}
        for gpu_id, task in gpu_tasks:
            if gpu_id not in gpu_task_groups:
                gpu_task_groups[gpu_id] = []
            gpu_task_groups[gpu_id].append(task)
        
        # Process each GPU group
        with mp.Pool(processes=self.n_gpus) as pool:
            worker_func = partial(
                self._window_gpu_worker,
                window_features=window_features,
                original_lengths=original_lengths,
                metrics=metrics,
                start_time=start_time
            )
            
            gpu_results = pool.starmap(worker_func, 
                                     [(gpu_id, tasks) for gpu_id, tasks in gpu_task_groups.items()])
        
        # Flatten results
        all_results = []
        for gpu_result in gpu_results:
            all_results.extend(gpu_result)
        
        return all_results
    
    def _process_window_cpu_parallel(self, tasks: List[Tuple],
                                   window_features: Dict[str, np.ndarray],
                                   original_lengths: Dict[str, List[int]],
                                   metrics: List[str], start_time: int) -> List[Dict]:
        """Process window tasks using CPU parallelization."""
        n_cores = min(mp.cpu_count(), len(tasks))
        
        with mp.Pool(processes=n_cores) as pool:
            worker_func = partial(
                self._window_cpu_worker,
                window_features=window_features,
                original_lengths=original_lengths,
                metrics=metrics,
                start_time=start_time
            )
            
            results = pool.map(worker_func, tasks)
        
        return results
    
    @staticmethod
    def _window_gpu_worker(gpu_id: int, tasks: List[Tuple],
                          window_features: Dict[str, np.ndarray],
                          original_lengths: Dict[str, List[int]],
                          metrics: List[str], start_time: int) -> List[Dict]:
        """GPU worker for processing window tasks."""
        device = f'cuda:{gpu_id}'
        similarity_computer = GPUParallelSimilarity(device=device, memory_efficient=True)
        
        results = []
        
        for i, j, layer1, layer2 in tasks:
            result = {'indices': (i, j)}
            
            try:
                # Get window features
                features1 = window_features[layer1]
                features2 = window_features[layer2]
                
                # Ensure same batch size
                min_batch = min(features1.shape[0], features2.shape[0])
                features1 = features1[:min_batch]
                features2 = features2[:min_batch]
                
                # Handle padding for this window
                if layer1 in original_lengths and layer2 in original_lengths:
                    features1_valid, features2_valid = TemporalSimilarityAnalyzer._extract_window_valid_features(
                        features1, features2, original_lengths[layer1], original_lengths[layer2], start_time
                    )
                else:
                    features1_valid, features2_valid = features1, features2
                
                # Compute similarities
                if 'cosine' in metrics:
                    result['cosine'] = similarity_computer.compute_cosine_similarity_gpu(
                        features1_valid, features2_valid
                    )
                
                if 'correlation' in metrics:
                    result['correlation'] = similarity_computer.compute_correlation_gpu(
                        features1_valid, features2_valid
                    )
                
                if 'cka' in metrics:
                    result['cka'] = similarity_computer.compute_cka_gpu_optimized(
                        features1_valid, features2_valid
                    )
                
            except Exception as e:
                logger.warning(f"Error in GPU worker {gpu_id} for {layer1}-{layer2}: {e}")
                for metric in metrics:
                    result[metric] = 0.0
            
            results.append(result)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    @staticmethod
    def _window_cpu_worker(task: Tuple, window_features: Dict[str, np.ndarray],
                          original_lengths: Dict[str, List[int]],
                          metrics: List[str], start_time: int) -> Dict:
        """CPU worker for processing window tasks."""
        i, j, layer1, layer2 = task
        similarity_computer = GPUParallelSimilarity(device='cpu')
        
        result = {'indices': (i, j)}
        
        try:
            # Get window features
            features1 = window_features[layer1]
            features2 = window_features[layer2]
            
            # Ensure same batch size
            min_batch = min(features1.shape[0], features2.shape[0])
            features1 = features1[:min_batch]
            features2 = features2[:min_batch]
            
            # Handle padding for this window
            if layer1 in original_lengths and layer2 in original_lengths:
                features1_valid, features2_valid = TemporalSimilarityAnalyzer._extract_window_valid_features(
                    features1, features2, original_lengths[layer1], original_lengths[layer2], start_time
                )
            else:
                features1_valid, features2_valid = features1, features2
            
            # Compute similarities
            if 'cosine' in metrics:
                result['cosine'] = similarity_computer.compute_cosine_similarity_gpu(
                    features1_valid, features2_valid
                )
            
            if 'correlation' in metrics:
                result['correlation'] = similarity_computer.compute_correlation_gpu(
                    features1_valid, features2_valid
                )
            
            if 'cka' in metrics:
                result['cka'] = similarity_computer.compute_cka_gpu_optimized(
                    features1_valid, features2_valid
                )
            
        except Exception as e:
            logger.warning(f"Error in CPU worker for {layer1}-{layer2}: {e}")
            for metric in metrics:
                result[metric] = 0.0
        
        return result
    
    @staticmethod
    def _extract_window_valid_features(features1: np.ndarray, features2: np.ndarray,
                                     orig_lens1: List[int], orig_lens2: List[int],
                                     start_time: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract valid features for a specific time window."""
        all_X = []
        all_Y = []
        
        batch_size = features1.shape[0]
        window_size = features1.shape[1]
        
        for b in range(batch_size):
            # Get original length for this batch item
            orig_len1 = orig_lens1[b] if b < len(orig_lens1) else float('inf')
            orig_len2 = orig_lens2[b] if b < len(orig_lens2) else float('inf')
            
            # Adjust lengths for this window
            valid_len1 = max(0, min(orig_len1 - start_time, window_size))
            valid_len2 = max(0, min(orig_len2 - start_time, window_size))
            valid_len = min(valid_len1, valid_len2)
            
            if valid_len > 0:
                # Extract only valid time steps for this window
                X = features1[b, :valid_len, :]
                Y = features2[b, :valid_len, :]
                
                all_X.append(X)
                all_Y.append(Y)
        
        if all_X and all_Y:
            # Concatenate all valid samples
            X = np.vstack(all_X)
            Y = np.vstack(all_Y)
            
            # Ensure same number of samples
            min_samples = min(X.shape[0], Y.shape[0])
            return X[:min_samples], Y[:min_samples]
        else:
            # Fallback to original features if extraction fails
            return features1, features2


class AnimationGenerator:
    """Generate animations from temporal similarity data."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_similarity_animation(self, temporal_results: Dict, model_name: str,
                                  metric: str = 'cosine', fps: int = 5) -> str:
        """Create animation showing similarity evolution over time."""
        if metric not in temporal_results['similarities']:
            raise ValueError(f"Metric {metric} not available in results")
        
        matrices = temporal_results['similarities'][metric]
        layers = temporal_results['layers']
        window_size = temporal_results['window_size']
        stride = temporal_results['stride']
        
        # Determine color map and range
        metric_configs = {
            'cosine': {'cmap': 'viridis', 'vmin': 0, 'vmax': 1},
            'correlation': {'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1},
            'cka': {'cmap': 'viridis', 'vmin': 0, 'vmax': 1}
        }
        
        config = metric_configs.get(metric, {'cmap': 'viridis', 'vmin': 0, 'vmax': 1})
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up the initial heatmap
        im = ax.imshow(matrices[0], cmap=config['cmap'], 
                      vmin=config['vmin'], vmax=config['vmax'], aspect='auto')
        
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
                             ha='center', va='center', 
                             color='white' if config['cmap'] == 'viridis' else 'black',
                             fontsize=8)
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
                    value = matrices[frame][i, j]
                    text_annotations[i][j].set_text(f'{value:.2f}')
                    # Adjust text color based on background
                    if config['cmap'] == 'viridis':
                        text_annotations[i][j].set_color('white' if value < 0.5 else 'black')
                    else:
                        text_annotations[i][j].set_color('black' if abs(value) < 0.5 else 'white')
            
            return [im, title] + [text for row in text_annotations for text in row]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(matrices), interval=1000//fps, blit=True)
        
        # Save as GIF
        output_path = self.output_dir / f'temporal_{metric}_similarity_{model_name}_animation.gif'
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close()
        
        logger.info(f"Animation saved to {output_path}")
        return str(output_path)
    
    def create_similarity_comparison_animation(self, temporal_results: Dict, model_name: str,
                                            metrics: List[str] = None, fps: int = 5) -> str:
        """Create side-by-side comparison animation of multiple metrics."""
        if metrics is None:
            metrics = ['cosine', 'correlation', 'cka']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in temporal_results['similarities']]
        n_metrics = len(available_metrics)
        
        if n_metrics == 0:
            raise ValueError("No requested metrics available in results")
        
        layers = temporal_results['layers']
        window_size = temporal_results['window_size']
        stride = temporal_results['stride']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]
        
        # Metric configurations
        metric_configs = {
            'cosine': {'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'title': 'Cosine Similarity'},
            'correlation': {'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1, 'title': 'Correlation'},
            'cka': {'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'title': 'CKA'}
        }
        
        # Set up initial heatmaps
        ims = []
        text_annotations_all = []
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            config = metric_configs[metric]
            matrices = temporal_results['similarities'][metric]
            
            # Create heatmap
            im = ax.imshow(matrices[0], cmap=config['cmap'], 
                          vmin=config['vmin'], vmax=config['vmax'], aspect='auto')
            ims.append(im)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Similarity')
            
            # Set ticks and labels
            ax.set_xticks(range(len(layers)))
            ax.set_yticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha='right')
            ax.set_yticklabels(layers)
            ax.set_title(config['title'])
            
            # Add text annotations
            text_annotations = []
            for i in range(len(layers)):
                text_row = []
                for j in range(len(layers)):
                    text = ax.text(j, i, f'{matrices[0][i, j]:.2f}',
                                 ha='center', va='center', 
                                 color='white' if config['cmap'] == 'viridis' else 'black',
                                 fontsize=7)
                    text_row.append(text)
                text_annotations.append(text_row)
            text_annotations_all.append(text_annotations)
        
        # Main title
        suptitle = fig.suptitle(f'Layer Similarity Comparison - {model_name}\nTime: 0-{window_size} steps')
        
        def update(frame):
            # Update all heatmaps
            for idx, metric in enumerate(available_metrics):
                matrices = temporal_results['similarities'][metric]
                config = metric_configs[metric]
                
                # Update heatmap data
                ims[idx].set_array(matrices[frame])
                
                # Update text annotations
                for i in range(len(layers)):
                    for j in range(len(layers)):
                        value = matrices[frame][i, j]
                        text_annotations_all[idx][i][j].set_text(f'{value:.2f}')
                        
                        # Adjust text color
                        if config['cmap'] == 'viridis':
                            color = 'white' if value < 0.5 else 'black'
                        else:
                            color = 'black' if abs(value) < 0.5 else 'white'
                        text_annotations_all[idx][i][j].set_color(color)
            
            # Update main title
            start_time = frame * stride
            end_time = start_time + window_size
            suptitle.set_text(f'Layer Similarity Comparison - {model_name}\nTime: {start_time}-{end_time} steps')
            
            return ims + [suptitle] + [text for annotations in text_annotations_all for row in annotations for text in row]
        
        # Create animation
        n_frames = len(temporal_results['similarities'][available_metrics[0]])
        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=True)
        
        # Save as GIF
        output_path = self.output_dir / f'temporal_comparison_{model_name}_animation.gif'
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close()
        
        logger.info(f"Comparison animation saved to {output_path}")
        return str(output_path)
    
    def create_summary_plots(self, temporal_results: Dict, model_name: str):
        """Create summary plots of temporal similarity trends."""
        layers = temporal_results['layers']
        metrics = temporal_results['metrics']
        n_windows = temporal_results['n_windows']
        window_size = temporal_results['window_size']
        stride = temporal_results['stride']
        
        # Create time axis
        time_axis = [i * stride + window_size/2 for i in range(n_windows)]
        
        # Plot average similarity over time for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            matrices = temporal_results['similarities'][metric]
            
            # Compute average off-diagonal similarity for each time window
            avg_similarities = []
            for matrix in matrices:
                # Get upper triangle (excluding diagonal)
                upper_tri = np.triu(matrix, k=1)
                avg_sim = np.mean(upper_tri[upper_tri != 0])
                avg_similarities.append(avg_sim)
            
            # Plot trend
            ax.plot(time_axis, avg_similarities, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel(f'Average {metric.capitalize()} Similarity')
            ax.set_title(f'{metric.capitalize()} Similarity Over Time - {model_name}')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(time_axis, avg_similarities, 1)
            p = np.poly1d(z)
            ax.plot(time_axis, p(time_axis), "r--", alpha=0.8, 
                   label=f'Trend: slope={z[0]:.4f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'temporal_trends_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Temporal trend plots saved to {self.output_dir}")


class CompleteTemporalAnalysis:
    """Complete temporal analysis pipeline."""
    
    def __init__(self, features_dir: str, output_dir: str, num_files: int = 3,
                 model_name: str = "HuBERT", window_size: int = 20, stride: int = 10,
                 n_gpus: Optional[int] = None):
        self.features_dir = features_dir
        self.output_dir = output_dir
        self.num_files = num_files
        self.model_name = model_name
        self.window_size = window_size
        self.stride = stride
        self.n_gpus = n_gpus
        
        # Initialize components
        self.feature_loader = FeatureLoader(features_dir, num_files)
        self.temporal_analyzer = TemporalSimilarityAnalyzer(window_size, stride, n_gpus)
        self.animation_generator = AnimationGenerator(output_dir)
        
        logger.info(f"Initialized temporal analysis pipeline for {model_name}")
    
    def run_temporal_analysis(self, metrics: List[str] = None, 
                            create_animations: bool = True) -> Dict:
        """Run complete temporal similarity analysis."""
        if metrics is None:
            metrics = ['cosine', 'correlation', 'cka']
        
        logger.info("Starting temporal similarity analysis...")
        start_time = time.time()
        
        # Step 1: Load features
        logger.info("Step 1: Loading features...")
        layer_features, original_lengths = self.feature_loader.load_features()
        
        # Step 2: Compute temporal similarities
        logger.info("Step 2: Computing temporal similarities...")
        temporal_results = self.temporal_analyzer.compute_temporal_similarities(
            layer_features, original_lengths, metrics
        )
        
        # Step 3: Create visualizations
        if create_animations:
            logger.info("Step 3: Creating animations...")
            animation_paths = []
            
            # Individual metric animations
            for metric in metrics:
                try:
                    path = self.animation_generator.create_similarity_animation(
                        temporal_results, self.model_name, metric
                    )
                    animation_paths.append(path)
                except Exception as e:
                    logger.error(f"Failed to create animation for {metric}: {e}")
            
            # Comparison animation
            try:
                comparison_path = self.animation_generator.create_similarity_comparison_animation(
                    temporal_results, self.model_name, metrics
                )
                animation_paths.append(comparison_path)
            except Exception as e:
                logger.error(f"Failed to create comparison animation: {e}")
            
            # Summary plots
            try:
                self.animation_generator.create_summary_plots(temporal_results, self.model_name)
            except Exception as e:
                logger.error(f"Failed to create summary plots: {e}")
        else:
            animation_paths = []
        
        # Step 4: Save results
        results_path = Path(self.output_dir) / f'temporal_results_{self.model_name}_n{self.num_files}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump({
                'temporal_results': temporal_results,
                'animation_paths': animation_paths,
                'model_name': self.model_name,
                'num_files': self.num_files,
                'window_size': self.window_size,
                'stride': self.stride
            }, f)
        
        total_time = time.time() - start_time
        logger.info(f"Temporal analysis completed in {total_time:.2f} seconds")
        
        return {
            'temporal_results': temporal_results,
            'animation_paths': animation_paths,
            'total_time': total_time
        }


def main():
    """Main function for temporal analysis."""
    parser = argparse.ArgumentParser(description="GPU-Accelerated Temporal Similarity Analysis")
    parser.add_argument("--features_dir", type=str, required=True,
                      help="Directory containing feature .npz files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save animations and results")
    parser.add_argument("--num_files", type=int, default=3,
                      help="Number of audio files to analyze")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the model (e.g., 'HuBERT Base')")
    parser.add_argument("--window_size", type=int, default=20,
                      help="Size of sliding window for temporal analysis")
    parser.add_argument("--stride", type=int, default=10,
                      help="Stride for sliding window")
    parser.add_argument("--n_gpus", type=int, default=None,
                      help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--metrics", nargs='+', default=['cosine', 'correlation', 'cka'],
                      help="Similarity metrics to compute")
    parser.add_argument("--no_animations", action='store_true', default=False,
                      help="Skip animation generation")
    parser.add_argument("--fps", type=int, default=5,
                      help="Frames per second for animations")
    parser.add_argument("--cpu_only", action='store_true', default=False,
                      help="Force CPU-only computation")
    
    args = parser.parse_args()
    
    # Override GPU detection if CPU-only requested
    n_gpus = 0 if args.cpu_only else args.n_gpus
    
    # Create temporal analysis pipeline
    analysis = CompleteTemporalAnalysis(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        num_files=args.num_files,
        model_name=args.model_name,
        window_size=args.window_size,
        stride=args.stride,
        n_gpus=n_gpus
    )
    
    # Run analysis
    try:
        results = analysis.run_temporal_analysis(
            metrics=args.metrics,
            create_animations=not args.no_animations
        )
        
        # Print summary
        print("\n" + "="*60)
        print("TEMPORAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model: {args.model_name}")
        print(f"Files analyzed: {args.num_files}")
        print(f"Window size: {args.window_size}")
        print(f"Stride: {args.stride}")
        print(f"Metrics: {', '.join(args.metrics)}")
        print(f"Total computation time: {results['total_time']:.2f} seconds")
        
        if results['animation_paths']:
            print(f"\nAnimations created:")
            for path in results['animation_paths']:
                print(f"  {path}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
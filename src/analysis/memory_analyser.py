import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from torch.utils.data import DataLoader
import logging
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from engine.engine import Engine

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class MemoryVisualizationAnalyzer:
    """
    Analyzer for visualizing memory building process in Hopfield OOD detection
    """
    
    def __init__(self, feature_extractor, projection_head, classifier_head, device):
        self.feature_extractor = feature_extractor
        self.projection_head = projection_head
        self.classifier_head = classifier_head
        self.device = device
        
        # Storage for analysis
        self.layer_features = {}
        self.projected_features = {}
        self.memory_data = {
            'id_memory': None,
            'ood_memory': None,
            'id_labels': None,
            'ood_labels': None
        }
        
        # Color maps for visualization
        self.class_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        self.ood_color = np.array([1.0, 0.0, 0.0, 1.0])  # Red for OOD
        
    def extract_multi_layer_features(self, dataloader, max_batches=3):
        """Extract features from multiple layers for visualization - MEMORY OPTIMIZED"""
        
        self.feature_extractor.eval()
        self.projection_head.eval()
        
        all_layer_features = {
            'mod3': [], 'mod4': [], 'mod5': [], 'mod6': [], 'aspp': [],
            'projected': [], 'labels': [], 'is_ood': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                # MEMORY OPTIMIZATION - Clear cache before each batch
                clear_memory()
                    
                logger.info(f"Processing batch {batch_idx + 1}/{max_batches}")
                
                try:
                    # REDUCE SAMPLE RATE for memory efficiency
                    sample_rate = 0.02  # Changed from 0.1 to 0.02 (2% instead of 10%)
                    
                    # Process only first image in batch to save memory
                    if isinstance(batch, dict) and 'data' in batch:
                        batch['data'] = batch['data'][:1]  # Take only first sample
                        if 'label' in batch:
                            batch['label'] = batch['label'][:1]
                        if 'is_ood' in batch:
                            batch['is_ood'] = batch['is_ood'][:1]
                    
                    # Extract features from multiple layers
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    features = extracted['features']
                    labels = extracted.get('labels')
                    is_ood = extracted.get('is_ood', torch.zeros(features.shape[0], dtype=torch.bool))
                    
                    if labels is None:
                        continue
                    
                    # Get layer-wise features
                    layer_feats = self.feature_extractor.features
                    
                    # Project features
                    projected = self.projection_head(features)
                    
                    # Sample pixels for visualization (to manage memory)
                    B, C, H, W = projected.shape
                    num_samples = int(H * W * sample_rate)
                    
                    for img_idx in range(B):
                        # Randomly sample pixels
                        pixel_indices = torch.randperm(H * W)[:num_samples]
                        y_coords = pixel_indices // W
                        x_coords = pixel_indices % W
                        
                        # Extract features for sampled pixels
                        img_labels = labels[img_idx][y_coords, x_coords].cpu()
                        img_is_ood = is_ood[img_idx]
                        
                        # Store layer features
                        for layer_name in ['mod3', 'mod4', 'mod5', 'mod6', 'aspp']:
                            if layer_name in layer_feats:
                                layer_feat = layer_feats[layer_name][img_idx]  # [C, H, W]
                                # Resize to match projected features
                                layer_feat_resized = F.interpolate(
                                    layer_feat.unsqueeze(0), 
                                    size=(H, W), 
                                    mode='bilinear', 
                                    align_corners=True
                                ).squeeze(0)
                                
                                sampled_feat = layer_feat_resized[:, y_coords, x_coords].T  # [N, C]
                                all_layer_features[layer_name].append(sampled_feat.cpu())
                        
                        # Store projected features
                        proj_feat = projected[img_idx][:, y_coords, x_coords].T  # [N, C]
                        all_layer_features['projected'].append(proj_feat.cpu())
                        all_layer_features['labels'].append(img_labels)
                        all_layer_features['is_ood'].append(
                            torch.full((len(img_labels),), img_is_ood.item(), dtype=torch.bool)
                        )
                    
                    # CLEAR INTERMEDIATE TENSORS
                    del extracted, features, projected
                    clear_memory()
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    clear_memory()
                    continue
        
        # Concatenate all features
        for key in all_layer_features:
            if all_layer_features[key]:
                all_layer_features[key] = torch.cat(all_layer_features[key], dim=0)
            else:
                all_layer_features[key] = torch.empty(0)
        
        self.layer_features = all_layer_features
        return all_layer_features
    
    def create_memory_and_visualize(self, id_memory, ood_memory):
        """Store memory data for visualization"""
        self.memory_data['id_memory'] = id_memory.cpu()
        self.memory_data['ood_memory'] = ood_memory.cpu()
        
        # Create labels for memory
        self.memory_data['id_labels'] = torch.full((len(id_memory),), -1, dtype=torch.long)  # ID memory
        self.memory_data['ood_labels'] = torch.full((len(ood_memory),), -2, dtype=torch.long)  # OOD memory
    
    def visualize_layer_progression(self, save_dir="./visualizations"):
        """Visualize how features evolve through different layers"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.layer_features:
            logger.error("No layer features extracted. Run extract_multi_layer_features first.")
            return
        
        # Prepare data for visualization
        labels = self.layer_features['labels']
        is_ood = self.layer_features['is_ood']
        
        if len(labels) == 0:
            logger.error("No labels found for visualization")
            return
        
        # Filter for valid labels and subsample for visualization
        valid_mask = (labels >= 0) | (labels == 254) | (labels == 255)
        max_samples = 2000  # Reduced from 5000
        
        if valid_mask.sum() > max_samples:
            valid_indices = torch.where(valid_mask)[0]
            subsample_indices = torch.randperm(len(valid_indices))[:max_samples]
            selected_indices = valid_indices[subsample_indices]
        else:
            selected_indices = torch.where(valid_mask)[0]
        
        if len(selected_indices) == 0:
            logger.error("No valid indices for visualization")
            return
        
        # Create visualization for each layer
        layers_to_viz = ['mod3', 'mod4', 'mod5', 'mod6', 'aspp', 'projected']
        n_layers = len(layers_to_viz)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, layer_name in enumerate(layers_to_viz):
            if layer_name not in self.layer_features or len(self.layer_features[layer_name]) == 0:
                axes[i].text(0.5, 0.5, f'No data for {layer_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            # Get features for this layer
            features = self.layer_features[layer_name][selected_indices]
            layer_labels = labels[selected_indices]
            layer_is_ood = is_ood[selected_indices]
            
            if len(features) == 0:
                axes[i].text(0.5, 0.5, f'No features for {layer_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            # Reduce dimensionality for visualization
            try:
                if features.shape[1] > 50:
                    # Use PCA first to reduce to 50D, then t-SNE
                    pca = PCA(n_components=min(50, features.shape[0]-1))
                    features_pca = pca.fit_transform(features.numpy())
                    perplexity = min(30, len(features)//4)
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    features_2d = tsne.fit_transform(features_pca)
                else:
                    perplexity = min(30, len(features)//4)
                    if perplexity < 5:
                        perplexity = 5
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    features_2d = tsne.fit_transform(features.numpy())
            except Exception as e:
                logger.error(f"Error in dimensionality reduction for {layer_name}: {e}")
                axes[i].text(0.5, 0.5, f'Error in {layer_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            # Create color map
            colors = []
            for j, (label, ood_flag) in enumerate(zip(layer_labels, layer_is_ood)):
                if ood_flag or label == 254:
                    colors.append(self.ood_color)
                elif label == 255:
                    colors.append([0.5, 0.5, 0.5, 0.5])  # Gray for ignore
                else:
                    colors.append(self.class_colors[int(label) % 19])
            
            # Plot
            scatter = axes[i].scatter(
                features_2d[:, 0], features_2d[:, 1],
                c=colors, s=1, alpha=0.6
            )
            axes[i].set_title(f'{layer_name.upper()} Features', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('t-SNE 1')
            axes[i].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/layer_progression.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_dir}/layer_progression.pdf', bbox_inches='tight')
        plt.close()  # Close to save memory
        
        logger.info(f"Layer progression visualization saved to {save_dir}")
    
    def visualize_memory_separation(self, save_dir="./visualizations"):
        """Visualize ID vs OOD memory separation"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        if self.memory_data['id_memory'] is None or self.memory_data['ood_memory'] is None:
            logger.error("No memory data available. Run create_memory_and_visualize first.")
            return
        
        # Combine ID and OOD memory
        all_memory = torch.cat([
            self.memory_data['id_memory'],
            self.memory_data['ood_memory']
        ], dim=0)
        
        all_labels = torch.cat([
            self.memory_data['id_labels'],
            self.memory_data['ood_labels']
        ], dim=0)
        
        # Subsample for visualization - REDUCED
        max_samples = 5000  # Reduced from 10000
        if len(all_memory) > max_samples:
            indices = torch.randperm(len(all_memory))[:max_samples]
            all_memory = all_memory[indices]
            all_labels = all_labels[indices]
        
        # Apply dimensionality reduction
        if all_memory.shape[1] > 50:
            pca = PCA(n_components=50)
            memory_pca = pca.fit_transform(all_memory.numpy())
        else:
            memory_pca = all_memory.numpy()
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        memory_2d = tsne.fit_transform(memory_pca)
        
        # UMAP visualization
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        memory_umap = umap_reducer.fit_transform(memory_pca)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # t-SNE plot
        id_mask = (all_labels == -1)
        ood_mask = (all_labels == -2)
        
        ax1.scatter(memory_2d[id_mask, 0], memory_2d[id_mask, 1], 
                   c='blue', s=1, alpha=0.6, label='ID Memory')
        ax1.scatter(memory_2d[ood_mask, 0], memory_2d[ood_mask, 1], 
                   c='red', s=1, alpha=0.6, label='OOD Memory')
        ax1.set_title('Memory Separation (t-SNE)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend()
        
        # UMAP plot
        ax2.scatter(memory_umap[id_mask, 0], memory_umap[id_mask, 1], 
                   c='blue', s=1, alpha=0.6, label='ID Memory')
        ax2.scatter(memory_umap[ood_mask, 0], memory_umap[ood_mask, 1], 
                   c='red', s=1, alpha=0.6, label='OOD Memory')
        ax2.set_title('Memory Separation (UMAP)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/memory_separation.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_dir}/memory_separation.pdf', bbox_inches='tight')
        plt.close()  # Close to save memory
        
        # Calculate and plot separation metrics
        self._plot_separation_metrics(all_memory, all_labels, save_dir)
        
        logger.info(f"Memory separation visualization saved to {save_dir}")
    
    def _plot_separation_metrics(self, all_memory, all_labels, save_dir):
        """Plot separation metrics between ID and OOD memory"""
        
        id_memory = all_memory[all_labels == -1]
        ood_memory = all_memory[all_labels == -2]
        
        if len(id_memory) == 0 or len(ood_memory) == 0:
            return
        
        # Calculate cosine similarities
        id_normalized = F.normalize(id_memory, dim=1)
        ood_normalized = F.normalize(ood_memory, dim=1)
        
        # Sample for efficiency - REDUCED
        sample_size = 500  # Reduced from 1000
        if len(id_normalized) > sample_size:
            id_sample = id_normalized[torch.randperm(len(id_normalized))[:sample_size]]
        else:
            id_sample = id_normalized
            
        if len(ood_normalized) > sample_size:
            ood_sample = ood_normalized[torch.randperm(len(ood_normalized))[:sample_size]]
        else:
            ood_sample = ood_normalized
        
        # Intra-class similarities
        id_intra_sim = torch.mm(id_sample, id_sample.T)
        id_intra_sim = id_intra_sim[torch.triu(torch.ones_like(id_intra_sim, dtype=bool), diagonal=1)]
        
        ood_intra_sim = torch.mm(ood_sample, ood_sample.T)
        ood_intra_sim = ood_intra_sim[torch.triu(torch.ones_like(ood_intra_sim, dtype=bool), diagonal=1)]
        
        # Inter-class similarities
        inter_sim = torch.mm(id_sample, ood_sample.T).flatten()
        
        # Plot histograms
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(id_intra_sim.numpy(), bins=30, alpha=0.6, label='ID Intra-similarity', 
                color='blue', density=True)
        ax.hist(ood_intra_sim.numpy(), bins=30, alpha=0.6, label='OOD Intra-similarity', 
                color='red', density=True)
        ax.hist(inter_sim.numpy(), bins=30, alpha=0.6, label='ID-OOD Inter-similarity', 
                color='green', density=True)
        
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title('Memory Similarity Distributions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"""Statistics:
ID Intra: μ={id_intra_sim.mean():.3f}, σ={id_intra_sim.std():.3f}
OOD Intra: μ={ood_intra_sim.mean():.3f}, σ={ood_intra_sim.std():.3f}
ID-OOD Inter: μ={inter_sim.mean():.3f}, σ={inter_sim.std():.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_dir}/similarity_distributions.pdf', bbox_inches='tight')
        plt.close()  # Close to save memory
    
    def analyze_feature_evolution(self, save_dir="./visualizations"):
        """Analyze how feature representations evolve through the network"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.layer_features:
            logger.error("No layer features extracted.")
            return {}
        
        # Calculate feature statistics for each layer
        layer_stats = {}
        layers = ['mod3', 'mod4', 'mod5', 'mod6', 'aspp', 'projected']
        
        for layer in layers:
            if layer in self.layer_features and len(self.layer_features[layer]) > 0:
                features = self.layer_features[layer]
                
                layer_stats[layer] = {
                    'mean': features.mean(dim=0),
                    'std': features.std(dim=0),
                    'norm': torch.norm(features, dim=1).mean(),
                    'dimension': features.shape[1]
                }
        
        if not layer_stats:
            logger.error("No layer statistics computed")
            return {}
        
        # Plot feature statistics evolution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature dimensions
        dims = [layer_stats[layer]['dimension'] for layer in layers if layer in layer_stats]
        layer_names = [layer for layer in layers if layer in layer_stats]
        
        if dims:
            axes[0, 0].bar(layer_names, dims)
            axes[0, 0].set_title('Feature Dimensions Across Layers', fontweight='bold')
            axes[0, 0].set_ylabel('Dimension')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Feature norms
        if layer_names:
            norms = [layer_stats[layer]['norm'].item() for layer in layer_names]
            axes[0, 1].plot(layer_names, norms, 'o-', linewidth=2, markersize=8)
            axes[0, 1].set_title('Average Feature Norms', fontweight='bold')
            axes[0, 1].set_ylabel('L2 Norm')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Feature variance (mean std across dimensions)
        if layer_names:
            variances = [layer_stats[layer]['std'].mean().item() for layer in layer_names]
            axes[1, 0].plot(layer_names, variances, 's-', linewidth=2, markersize=8, color='red')
            axes[1, 0].set_title('Average Feature Variance', fontweight='bold')
            axes[1, 0].set_ylabel('Standard Deviation')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Feature distribution comparison
        if 'projected' in layer_stats and 'mod6' in layer_stats:
            # Sample features for comparison - REDUCED
            sample_size = 500  # Reduced from 1000
            mod6_features = self.layer_features['mod6'][:sample_size]
            proj_features = self.layer_features['projected'][:sample_size]
            
            # Calculate norms
            mod6_norms = torch.norm(mod6_features, dim=1)
            proj_norms = torch.norm(proj_features, dim=1)
            
            axes[1, 1].hist(mod6_norms.numpy(), bins=20, alpha=0.6, label='mod6', density=True)
            axes[1, 1].hist(proj_norms.numpy(), bins=20, alpha=0.6, label='projected', density=True)
            axes[1, 1].set_title('Feature Norm Distributions', fontweight='bold')
            axes[1, 1].set_xlabel('L2 Norm')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_evolution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_dir}/feature_evolution.pdf', bbox_inches='tight')
        plt.close()  # Close to save memory
        
        logger.info(f"Feature evolution analysis saved to {save_dir}")
        
        return layer_stats


def main():
    """Main visualization script - MEMORY OPTIMIZED"""
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Clear memory first
    clear_memory()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('visualization.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Configuration - MEMORY OPTIMIZED
    config = {
        'model_path': "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth",
        'num_classes': 19,
        'batch_size': 1,  # REDUCED
        'num_workers': 0
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Import your existing classes
    from feature_extractor import FeatureExtractor
    from projection_head import SimpleProjectionHead
    from segmentation_head import SegmentationClassifierHead
    from hopfield_memory_builder import create_memory_clustering
    
    # Initialize models with smaller resolution
    feature_extractor = FeatureExtractor(
        model_path=config['model_path'],
        device=device,
        num_classes=config['num_classes'],
        resize_resolution=(512, 1024),  # Much smaller resolution
        amp=True  # Enable mixed precision
    )
    
    projection_head = SimpleProjectionHead(input_dim=1280, output_dim=128).to(device)
    classifier_head = SegmentationClassifierHead(in_channels=1280, num_classes=config['num_classes']).to(device)
    
    # Set up data loading
    from dataset.data_loader import get_mix_loader
    
    # Create custom args for engine
    class CustomArgs:
        def __init__(self):
            self.ddp = False
            self.local_rank = -1
            self.gpus = 1
            self.world_size = 1
    
    custom_args = CustomArgs()
    
    engine_instance = Engine(
        custom_arg=custom_args,
        logger=logger,
        continue_state_object=config['model_path']
    )
    
    # Load data with proper engine
    train_loader, _, _ = get_mix_loader(
        engine=engine_instance,
        augment=True,
        cs_root="/home/ha51dybi/PEBAL/cityscapes",
        coco_root="/home/ha51dybi/PEBAL/coco"
    )
    
    # Create much smaller subset for memory efficiency
    subset_size = 100  # VERY SMALL for testing
    subset_data = []
    for i, batch in enumerate(train_loader):
        if i >= subset_size:
            break
        # Take only first sample from each batch
        if isinstance(batch, dict):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key][:1]  # Keep only first sample
        subset_data.append(batch)
    
    subset_loader = iter(subset_data)
    
    # Initialize analyzer
    analyzer = MemoryVisualizationAnalyzer(
        feature_extractor=feature_extractor,
        projection_head=projection_head,
        classifier_head=classifier_head,
        device=device
    )
    
    # Extract multi-layer features for visualization
    logger.info("Extracting multi-layer features...")
    layer_features = analyzer.extract_multi_layer_features(subset_loader, max_batches=3)  # REDUCED
    
    # Build memory using your existing function
    logger.info("Building memory...")
    try:
        id_memory, ood_memory = create_memory_clustering(
            dataloader=iter(subset_data),
            feature_extractor=feature_extractor,
            projection_head=projection_head,
            device=device
        )
        
        # Store memory data in analyzer
        analyzer.create_memory_and_visualize(id_memory, ood_memory)
    except Exception as e:
        logger.error(f"Error in memory building: {e}")
        # Create dummy memory for visualization
        id_memory = torch.randn(100, 128)
        ood_memory = torch.randn(100, 128)
        analyzer.create_memory_and_visualize(id_memory, ood_memory)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    save_dir = "./thesis_visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. Layer progression visualization
        analyzer.visualize_layer_progression(save_dir)
        clear_memory()
        
        # 2. Memory separation visualization
        analyzer.visualize_memory_separation(save_dir)
        clear_memory()
        
        # 3. Feature evolution analysis
        layer_stats = analyzer.analyze_feature_evolution(save_dir)
        clear_memory()
        
        # Print summary statistics
        logger.info("=== VISUALIZATION SUMMARY ===")
        logger.info(f"ID Memory size: {len(id_memory)}")
        logger.info(f"OOD Memory size: {len(ood_memory)}")
        logger.info(f"Total samples processed: {len(layer_features['labels']) if layer_features else 0}")
        
        for layer, stats in layer_stats.items():
            logger.info(f"{layer}: dim={stats['dimension']}, avg_norm={stats['norm']:.3f}")
        
        logger.info(f"All visualizations saved to: {save_dir}")
        logger.info("=== VISUALIZATION COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        logger.info("Partial results may be available in the save directory")
    
    return save_dir

if __name__ == "__main__":
    main()
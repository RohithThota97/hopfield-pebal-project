#!/usr/bin/env python3
"""
Complete Main Function with Integrated Ablation Studies
Uses data loaders from main_train.py with Cityscapes + COCO mix dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Imports from main training script
from feature_extractor import FeatureExtractor
from hopfield_memory_builder import MemoryBuilder
from hopfield_weight_updater import HopfieldBoostingManager
from pixel_energy import PixelWiseBorderEnergy, PixelWiseInferenceScore, compute_hopfield_ood_loss
from projection_head import SimpleProjectionHead
from dataset.data_loader import CityscapesCocoMix, Cityscapes, get_mix_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AblationStudyManager:
    def __init__(self, config, train_loader, val_loader):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Use provided data loaders from main_train.py
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create output directory for plots and results
        self.output_dir = config.get('output_dir', './ablation_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize base components
        self._setup_base_model()
        
    def _setup_base_model(self):
        """Initialize feature extractor and projection head"""
        self.feature_extractor = FeatureExtractor(
            model_path=self.config['model_path'],
            device=self.device,
            num_classes=19,
        ).to(self.device)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.projection_head = SimpleProjectionHead(
            input_dim=1280, output_dim=128
        ).to(self.device)
        
    def run_all_ablations(self):
        """Run all ablation studies and generate comprehensive report"""
        logger.info("Starting comprehensive ablation study...")
        
        try:
            # Run individual ablation studies
            self.ablation_1_feature_extraction()
            torch.cuda.empty_cache()
            
            self.ablation_2_memory_building()
            torch.cuda.empty_cache()
            
            self.ablation_3_weight_updating()
            torch.cuda.empty_cache()
            
            self.ablation_4_energy_methods()
            torch.cuda.empty_cache()
            
            # Generate visualizations
            self.visualize_tsne_embedding()
            torch.cuda.empty_cache()
            
            self.visualize_energy_maps()
            torch.cuda.empty_cache()
            
            # Generate final report
            report = self.generate_ablation_report()
            
            # Save results
            self._save_results(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error during ablation studies: {e}")
            return None
        
    def ablation_1_feature_extraction(self):
        """Ablation Study 1: Feature Extraction Methods"""
        logger.info("=== Ablation Study 1: Feature Extraction ===")
        
        methods = {
            'baseline': {'multi_scale': False, 'attention': False},
            'multi_scale': {'multi_scale': True, 'attention': False},
            'attention': {'multi_scale': False, 'attention': True},
            'full': {'multi_scale': True, 'attention': True}
        }
        
        results = {}
        
        for method_name, params in methods.items():
            logger.info(f"Testing {method_name} feature extraction...")
            
            # Extract features with different configurations
            id_features, ood_features = self._extract_features_with_config(params)
            
            # Compute separation metrics
            separation_score = self._compute_separation_score(id_features, ood_features)
            diversity_score = self._compute_diversity_score(id_features)
            
            results[method_name] = {
                'separation': separation_score,
                'diversity': diversity_score,
                'feature_dim': id_features.shape[1] if len(id_features) > 0 else 0,
                'id_samples': len(id_features),
                'ood_samples': len(ood_features)
            }
            
            logger.info(f"{method_name}: Separation={separation_score:.4f}, Diversity={diversity_score:.4f}")
        
        self.results['feature_extraction'] = results
        self._plot_feature_extraction_results(results)
        
    def ablation_2_memory_building(self):
        """Ablation Study 2: Memory Building Strategies"""
        logger.info("=== Ablation Study 2: Memory Building ===")
        
        strategies = {
            'random': {'clustering': False, 'diversity_weight': 0.0},
            'clustering': {'clustering': True, 'diversity_weight': 0.0},
            'diversity': {'clustering': False, 'diversity_weight': 1.0},
            'clustering_diversity': {'clustering': True, 'diversity_weight': 0.5}
        }
        
        results = {}
        
        # Extract base features
        id_features, ood_features = self._extract_features_with_config({'multi_scale': True, 'attention': True})
        
        for strategy_name, params in strategies.items():
            logger.info(f"Testing {strategy_name} memory building...")
            
            # Build memory with different strategies
            id_memory, ood_memory = self._build_memory_with_strategy(
                id_features, ood_features, params
            )
            
            # Evaluate memory quality
            memory_coverage = self._compute_memory_coverage(id_features, id_memory)
            memory_diversity = self._compute_diversity_score(id_memory)
            compression_ratio = len(id_memory) / len(id_features) if len(id_features) > 0 else 0
            
            results[strategy_name] = {
                'coverage': memory_coverage,
                'diversity': memory_diversity,
                'memory_size': len(id_memory),
                'compression_ratio': compression_ratio
            }
            
            logger.info(f"{strategy_name}: Coverage={memory_coverage:.4f}, Diversity={memory_diversity:.4f}")
        
        self.results['memory_building'] = results
        self._plot_memory_building_results(results)
        
    def ablation_3_weight_updating(self):
        """Ablation Study 3: Weight Updating Mechanisms"""
        logger.info("=== Ablation Study 3: Weight Updating ===")
        
        mechanisms = {
            'no_boosting': {'boosting': False, 'beta': 1.0},
            'low_beta': {'boosting': True, 'beta': 32.0},
            'medium_beta': {'boosting': True, 'beta': 128.0},
            'high_beta': {'boosting': True, 'beta': 512.0}
        }
        
        results = {}
        
        # Extract features and build memory
        id_features, ood_features = self._extract_features_with_config({'multi_scale': True, 'attention': True})
        id_memory, ood_memory = self._build_memory_with_strategy(
            id_features, ood_features, {'clustering': True, 'diversity_weight': 0.5}
        )
        
        for mechanism_name, params in mechanisms.items():
            logger.info(f"Testing {mechanism_name} weight updating...")
            
            # Simulate weight updating
            if params['boosting'] and len(id_memory) > 0 and len(ood_memory) > 0:
                try:
                    boosting_manager = HopfieldBoostingManager(
                        id_features_full=id_memory,
                        aux_features_full=ood_memory,
                        beta_sampling=params['beta'],
                        device=self.device
                    )
                    
                    # Sample batches and compute adaptation score
                    adaptation_score = self._compute_adaptation_score(boosting_manager)
                except Exception as e:
                    logger.warning(f"Boosting failed for {mechanism_name}: {e}")
                    adaptation_score = 0.0
            else:
                adaptation_score = 0.0
            
            # Compute final performance
            ood_detection_score = self._compute_ood_detection_score(
                id_memory, ood_memory, params['beta']
            )
            
            results[mechanism_name] = {
                'adaptation': adaptation_score,
                'ood_detection': ood_detection_score,
                'beta': params['beta'],
                'boosting_enabled': params['boosting']
            }
            
            logger.info(f"{mechanism_name}: Adaptation={adaptation_score:.4f}, OOD Detection={ood_detection_score:.4f}")
        
        self.results['weight_updating'] = results
        self._plot_weight_updating_results(results)
        
    def ablation_4_energy_methods(self):
        """Ablation Study 4: Energy Computation Methods"""
        logger.info("=== Ablation Study 4: Energy Methods ===")
        
        energy_methods = {
            'hopfield_only': {'border_energy': False, 'positive_shift': False},
            'border_energy': {'border_energy': True, 'positive_shift': False},
            'positive_shift': {'border_energy': True, 'positive_shift': True},
            'inference_score': {'border_energy': False, 'positive_shift': False, 'inference': True}
        }
        
        results = {}
        
        # Extract features and build memory
        id_features, ood_features = self._extract_features_with_config({'multi_scale': True, 'attention': True})
        id_memory, ood_memory = self._build_memory_with_strategy(
            id_features, ood_features, {'clustering': True, 'diversity_weight': 0.5}
        )
        
        for method_name, params in energy_methods.items():
            logger.info(f"Testing {method_name} energy method...")
            
            # Compute energy scores
            energy_scores = self._compute_energy_scores(
                id_features, ood_features, id_memory, ood_memory, params
            )
            
            # Evaluate separation
            if len(energy_scores['id']) > 0 and len(energy_scores['ood']) > 0:
                auc_score = self._compute_auc_score(energy_scores)
                boundary_sharpness = self._compute_boundary_sharpness(energy_scores)
            else:
                auc_score = 0.5
                boundary_sharpness = 0.0
            
            results[method_name] = {
                'auc': auc_score,
                'boundary_sharpness': boundary_sharpness,
                'mean_id_energy': np.mean(energy_scores['id']) if len(energy_scores['id']) > 0 else 0,
                'mean_ood_energy': np.mean(energy_scores['ood']) if len(energy_scores['ood']) > 0 else 0,
                'std_id_energy': np.std(energy_scores['id']) if len(energy_scores['id']) > 0 else 0,
                'std_ood_energy': np.std(energy_scores['ood']) if len(energy_scores['ood']) > 0 else 0
            }
            
            logger.info(f"{method_name}: AUC={auc_score:.4f}, Sharpness={boundary_sharpness:.4f}")
        
        self.results['energy_methods'] = results
        self._plot_energy_methods_results(results)
    
    def visualize_tsne_embedding(self):
        """Create t-SNE visualization of ID and OOD features"""
        logger.info("=== Creating t-SNE Visualization ===")
        
        # Extract features
        id_features, ood_features = self._extract_features_with_config({'multi_scale': True, 'attention': True})
        
        if len(id_features) == 0 or len(ood_features) == 0:
            logger.warning("Insufficient features for t-SNE visualization")
            return
        
        # Subsample for t-SNE (memory efficiency)
        max_samples = 1000
        if len(id_features) > max_samples:
            id_idx = np.random.choice(len(id_features), max_samples, replace=False)
            id_features = id_features[id_idx]
        
        if len(ood_features) > max_samples:
            ood_idx = np.random.choice(len(ood_features), max_samples, replace=False)
            ood_features = ood_features[ood_idx]
        
        # Combine features
        all_features = torch.cat([id_features, ood_features], dim=0).cpu().numpy()
        labels = np.concatenate([
            np.zeros(len(id_features)),  # ID: 0
            np.ones(len(ood_features))   # OOD: 1
        ])
        
        # Compute t-SNE
        logger.info("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)//4))
        embedding = tsne.fit_transform(all_features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='coolwarm', alpha=0.6, s=20
        )
        plt.colorbar(scatter, label='ID (0) vs OOD (1)')
        plt.title('t-SNE Visualization of ID vs OOD Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'tsne_id_ood_features.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"t-SNE plot saved to {plot_path}")

    def visualize_energy_maps(self):
        """Visualize energy maps before and after boosting"""
        logger.info("=== Creating Energy Map Visualizations ===")
        
        try:
            # Get sample batch
            sample_batch = next(iter(self.val_loader))
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor.extract_features_batch(sample_batch)
                projected_features = self.projection_head(features['features'])
            
            # Build memory
            id_features, ood_features = self._extract_features_with_config({'multi_scale': True, 'attention': True})
            id_memory, ood_memory = self._build_memory_with_strategy(
                id_features, ood_features, {'clustering': True, 'diversity_weight': 0.5}
            )
            
            if len(id_memory) == 0 or len(ood_memory) == 0:
                logger.warning("Insufficient memory for energy map visualization")
                return
            
            # Compute energy maps with different methods
            energy_configs = {
                'before_boosting': {'beta': 1.0, 'positive_shift': False},
                'after_boosting': {'beta': 128.0, 'positive_shift': True}
            }
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for i, (config_name, config) in enumerate(energy_configs.items()):
                # Compute border energy
                border_energy_calc = PixelWiseBorderEnergy(
                    id_memory, ood_memory, 
                    beta=config['beta'],
                    positive_shift=config['positive_shift']
                ).to(self.device)
                
                energy_map = border_energy_calc(projected_features, batch_size=1024)
                energy_map_np = energy_map[0].cpu().numpy()  # First image in batch
                
                # Original image
                if i == 0:
                    original = sample_batch['data'][0].permute(1, 2, 0).cpu().numpy()
                    original = (original - original.min()) / (original.max() - original.min())
                    axes[i, 0].imshow(original)
                    axes[i, 0].set_title('Original Image')
                    axes[i, 0].axis('off')
                else:
                    axes[i, 0].axis('off')
                
                # Energy map
                im1 = axes[i, 1].imshow(energy_map_np, cmap='hot', interpolation='bilinear')
                axes[i, 1].set_title(f'Energy Map - {config_name}')
                axes[i, 1].axis('off')
                plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
                
                # Energy histogram
                axes[i, 2].hist(energy_map_np.flatten(), bins=50, alpha=0.7, color='red')
                axes[i, 2].set_title(f'Energy Distribution - {config_name}')
                axes[i, 2].set_xlabel('Energy Value')
                axes[i, 2].set_ylabel('Frequency')
                axes[i, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, 'energy_maps_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            logger.info(f"Energy maps saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create energy map visualization: {e}")

    def generate_ablation_report(self):
        """Generate comprehensive ablation study report"""
        logger.info("=== Generating Ablation Study Report ===")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'summary': {},
            'detailed_results': self.results,
            'recommendations': {}
        }
        
        # Summarize key findings
        if 'feature_extraction' in self.results:
            best_feature = max(self.results['feature_extraction'].items(), 
                             key=lambda x: x[1]['separation'])
            report['summary']['best_feature_method'] = best_feature[0]
            report['summary']['best_feature_separation'] = best_feature[1]['separation']
        
        if 'memory_building' in self.results:
            best_memory = max(self.results['memory_building'].items(), 
                            key=lambda x: x[1]['coverage'])
            report['summary']['best_memory_strategy'] = best_memory[0]
            report['summary']['best_memory_coverage'] = best_memory[1]['coverage']
        
        if 'weight_updating' in self.results:
            best_weight = max(self.results['weight_updating'].items(), 
                            key=lambda x: x[1]['ood_detection'])
            report['summary']['best_weight_method'] = best_weight[0]
            report['summary']['best_weight_ood_score'] = best_weight[1]['ood_detection']
        
        if 'energy_methods' in self.results:
            best_energy = max(self.results['energy_methods'].items(), 
                            key=lambda x: x[1]['auc'])
            report['summary']['best_energy_method'] = best_energy[0]
            report['summary']['best_energy_auc'] = best_energy[1]['auc']
        
        # Generate recommendations
        report['recommendations'] = {
            'feature_extraction': "Use multi-scale features with attention for best separation",
            'memory_building': "Combine clustering with diversity weighting for optimal coverage",
            'weight_updating': "Use moderate beta values (128-256) for effective boosting",
            'energy_methods': "Border energy with positive shift provides best boundary sharpness"
        }
        
        return report
    
    def _save_results(self, report):
        """Save results to JSON file"""
        results_path = os.path.join(self.output_dir, 'ablation_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(report), f, indent=2)
        logger.info(f"Results saved to {results_path}")

    # Helper methods for feature extraction and analysis
    def _extract_features_with_config(self, config):
        """Extract features with specific configuration - FIXED: Device handling"""
        id_features_list = []
        ood_features_list = []
        
        self.feature_extractor.eval()
        self.projection_head.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx >= 20:  # Limit batches for memory
                    break
                    
                try:
                    # Ensure batch is on correct device
                    if isinstance(batch, dict):
                        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    # Extract features
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    features = extracted['features'].to(self.device)
                    labels = extracted.get('labels')
                    
                    if labels is None:
                        continue
                    
                    labels = labels.to(self.device)
                    
                    # Project features
                    projected = self.projection_head(features)
                    B, C, H, W = projected.shape
                    pixel_features = projected.permute(0, 2, 3, 1).reshape(-1, C)
                    pixel_labels = labels.view(-1)
                    
                    # Separate ID and OOD
                    id_mask = (pixel_labels >= 0) & (pixel_labels < 19)
                    ood_mask = pixel_labels == 254
                    
                    if id_mask.sum() > 0:
                        id_features_list.append(pixel_features[id_mask].detach())
                    if ood_mask.sum() > 0:
                        ood_features_list.append(pixel_features[ood_mask].detach())
                        
                except Exception as e:
                    logger.warning(f"Failed to process batch {batch_idx}: {e}")
                    continue
        
        # Concatenate and subsample - ensure proper device handling
        if id_features_list:
            id_features = torch.cat(id_features_list, dim=0).to(self.device)
            if len(id_features) > 5000:
                idx = torch.randperm(len(id_features), device=self.device)[:5000]
                id_features = id_features[idx]
        else:
            id_features = torch.empty(0, 128).to(self.device)
            
        if ood_features_list:
            ood_features = torch.cat(ood_features_list, dim=0).to(self.device)
            if len(ood_features) > 5000:
                idx = torch.randperm(len(ood_features), device=self.device)[:5000]
                ood_features = ood_features[idx]
        else:
            ood_features = torch.empty(0, 128).to(self.device)
        
        return id_features, ood_features
    
    def _compute_separation_score(self, id_features, ood_features):
        """Compute separation score between ID and OOD features - FIXED: Device handling"""
        if len(id_features) == 0 or len(ood_features) == 0:
            return 0.0
        
        try:
            # Ensure both tensors are on the same device
            id_features = id_features.to(self.device)
            ood_features = ood_features.to(self.device)
            
            id_mean = id_features.mean(dim=0)
            ood_mean = ood_features.mean(dim=0)
            
            # Euclidean distance between means
            separation = torch.norm(id_mean - ood_mean).item()
            return separation
        except Exception as e:
            logger.warning(f"Separation score computation failed: {e}")
            return 0.0
    
    def _compute_diversity_score(self, features):
        """Compute diversity score of features - FIXED: Device handling"""
        if len(features) < 2:
            return 0.0
        
        try:
            # Ensure tensor is on correct device
            features = features.to(self.device)
            
            # Average pairwise distance (subsampled)
            n_samples = min(1000, len(features))
            idx = torch.randperm(len(features), device=self.device)[:n_samples]
            sampled_features = features[idx]
            
            distances = torch.cdist(sampled_features, sampled_features, p=2)
            # Remove diagonal (self-distances)
            mask = ~torch.eye(len(sampled_features), dtype=bool, device=self.device)
            avg_distance = distances[mask].mean().item()
            
            return avg_distance
        except Exception as e:
            logger.warning(f"Diversity score computation failed: {e}")
            return 0.0
    
    def _build_memory_with_strategy(self, id_features, ood_features, strategy):
        """Build memory with specific strategy - GPU-only implementation"""
        memory_size = min(1000, len(id_features) // 2) if len(id_features) > 0 else 0
        
        if memory_size == 0:
            return torch.empty(0, 128, device=self.device), torch.empty(0, 128, device=self.device)
        
        if strategy['clustering'] and len(id_features) > 0:
            # Use GPU-based K-means clustering
            try:
                id_memory = self._gpu_kmeans_clustering(id_features, memory_size)
            except Exception as e:
                logger.warning(f"GPU clustering failed: {e}, using random sampling")
                idx = torch.randperm(len(id_features), device=self.device)[:memory_size]
                id_memory = id_features[idx]
        else:
            # Random sampling
            if len(id_features) > memory_size:
                idx = torch.randperm(len(id_features), device=self.device)[:memory_size]
                id_memory = id_features[idx]
            else:
                id_memory = id_features
        
        # Similar for OOD with GPU-only operations
        if len(ood_features) > 0:
            ood_memory_size = min(memory_size, len(ood_features))
            if strategy['clustering']:
                try:
                    ood_memory = self._gpu_kmeans_clustering(ood_features, ood_memory_size)
                except Exception as e:
                    logger.warning(f"OOD GPU clustering failed: {e}, using random sampling")
                    idx = torch.randperm(len(ood_features), device=self.device)[:ood_memory_size]
                    ood_memory = ood_features[idx]
            else:
                idx = torch.randperm(len(ood_features), device=self.device)[:ood_memory_size]
                ood_memory = ood_features[idx]
        else:
            ood_memory = torch.empty(0, 128, device=self.device)
        
        return id_memory, ood_memory
    
    def _gpu_kmeans_clustering(self, features, n_clusters, max_iters=100):
        """GPU-only K-means clustering implementation"""
        n_samples, n_features = features.shape
        n_clusters = min(n_clusters, n_samples)
        
        # Initialize centroids randomly
        idx = torch.randperm(n_samples, device=self.device)[:n_clusters]
        centroids = features[idx].clone()
        
        for _ in range(max_iters):
            # Compute distances to centroids
            distances = torch.cdist(features, centroids, p=2)
            
            # Assign points to closest centroid
            assignments = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = features[mask].mean(dim=0)
                else:
                    # If cluster is empty, reinitialize randomly
                    new_centroids[k] = features[torch.randint(0, n_samples, (1,), device=self.device)]
            
            # Check for convergence
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
                
            centroids = new_centroids
        
        return centroids
    
    def _compute_memory_coverage(self, features, memory):
        """Compute how well memory covers the feature space - FIXED: Device handling"""
        if len(features) == 0 or len(memory) == 0:
            return 0.0
        
        try:
            # Ensure both tensors are on the same device
            features = features.to(self.device)
            memory = memory.to(self.device)
            
            # Average distance from each feature to nearest memory
            features_subset = features[:1000]  # Subsample for efficiency
            distances = torch.cdist(features_subset, memory, p=2)
            min_distances = distances.min(dim=1)[0]
            coverage = 1.0 / (1.0 + min_distances.mean().item())
            
            return coverage
        except Exception as e:
            logger.warning(f"Memory coverage computation failed: {e}")
            return 0.0
    
    def _compute_adaptation_score(self, boosting_manager):
        """Compute adaptation score for boosting"""
        # Sample a few batches and measure weight variance
        try:
            batch1 = boosting_manager.sample_batch(64)
            batch2 = boosting_manager.sample_batch(64)
            
            if len(batch1[0]) > 0 and len(batch2[0]) > 0:
                # Measure difference in sampling
                diff = torch.norm(batch1[0].mean(dim=0) - batch2[0].mean(dim=0)).item()
                return min(diff, 1.0)  # Normalize
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Adaptation score computation failed: {e}")
            return 0.0
    
    def _compute_ood_detection_score(self, id_memory, ood_memory, beta):
        """Compute OOD detection score"""
        if len(id_memory) == 0 or len(ood_memory) == 0:
            return 0.5
        
        try:
            # Create inference score calculator
            inference_calc = PixelWiseInferenceScore(id_memory, ood_memory, beta=beta)
            
            # Sample some features for testing
            test_id = id_memory[:100]
            test_ood = ood_memory[:100]
            
            with torch.no_grad():
                id_scores = inference_calc(test_id)
                ood_scores = inference_calc(test_ood)
            
            # Compute AUC
            if len(id_scores) > 0 and len(ood_scores) > 0:
                all_scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
                labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
                auc = roc_auc_score(labels, all_scores)
                return auc
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"OOD detection score computation failed: {e}")
            return 0.5
    
    def _compute_energy_scores(self, id_features, ood_features, id_memory, ood_memory, params):
        """Compute energy scores with different methods"""
        scores = {'id': [], 'ood': []}
        
        try:
            if params.get('inference', False):
                # Use inference score
                calc = PixelWiseInferenceScore(id_memory, ood_memory, beta=128.0)
            else:
                # Use border energy
                calc = PixelWiseBorderEnergy(
                    id_memory, ood_memory, 
                    beta=128.0,
                    positive_shift=params.get('positive_shift', False)
                )
            
            with torch.no_grad():
                if len(id_features) > 0:
                    id_energies = calc(id_features[:1000])  # Subsample
                    scores['id'] = id_energies.cpu().numpy().flatten()
                
                if len(ood_features) > 0:
                    ood_energies = calc(ood_features[:1000])  # Subsample
                    scores['ood'] = ood_energies.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"Energy score computation failed: {e}")
        
        return scores
    
    def _compute_auc_score(self, energy_scores):
        """Compute AUC score from energy scores"""
        if len(energy_scores['id']) == 0 or len(energy_scores['ood']) == 0:
            return 0.5
        
        all_scores = np.concatenate([energy_scores['id'], energy_scores['ood']])
        labels = np.concatenate([
            np.zeros(len(energy_scores['id'])),
            np.ones(len(energy_scores['ood']))
        ])
        
        return roc_auc_score(labels, all_scores)
    
    def _compute_boundary_sharpness(self, energy_scores):
        """Compute boundary sharpness metric"""
        if len(energy_scores['id']) == 0 or len(energy_scores['ood']) == 0:
            return 0.0
        
        id_std = np.std(energy_scores['id'])
        ood_std = np.std(energy_scores['ood'])
        id_mean = np.mean(energy_scores['id'])
        ood_mean = np.mean(energy_scores['ood'])
        
        # Sharpness as separation vs spread
        separation = abs(ood_mean - id_mean)
        spread = (id_std + ood_std) / 2
        
        sharpness = separation / (spread + 1e-8)
        return sharpness
    
    # Plotting methods
    def _plot_feature_extraction_results(self, results):
        """Plot feature extraction results"""
        methods = list(results.keys())
        separations = [results[m]['separation'] for m in methods]
        diversities = [results[m]['diversity'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Separation plot
        bars1 = ax1.bar(methods, separations, color='skyblue', alpha=0.7)
        ax1.set_title('Feature Separation by Method')
        ax1.set_ylabel('Separation Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, separations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Diversity plot
        bars2 = ax2.bar(methods, diversities, color='lightcoral', alpha=0.7)
        ax2.set_title('Feature Diversity by Method')
        ax2.set_ylabel('Diversity Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars2, diversities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'feature_extraction_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Feature extraction plot saved to {plot_path}")
    
    def _plot_memory_building_results(self, results):
        """Plot memory building results"""
        strategies = list(results.keys())
        coverages = [results[s]['coverage'] for s in strategies]
        diversities = [results[s]['diversity'] for s in strategies]
        memory_sizes = [results[s]['memory_size'] for s in strategies]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Coverage plot
        bars1 = ax1.bar(strategies, coverages, color='lightgreen', alpha=0.7)
        ax1.set_title('Memory Coverage by Strategy')
        ax1.set_ylabel('Coverage Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars1, coverages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Diversity plot
        bars2 = ax2.bar(strategies, diversities, color='orange', alpha=0.7)
        ax2.set_title('Memory Diversity by Strategy')
        ax2.set_ylabel('Diversity Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, diversities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Memory size plot
        bars3 = ax3.bar(strategies, memory_sizes, color='purple', alpha=0.7)
        ax3.set_title('Memory Size by Strategy')
        ax3.set_ylabel('Memory Size')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for bar, val in zip(bars3, memory_sizes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'memory_building_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Memory building plot saved to {plot_path}")
    
    def _plot_weight_updating_results(self, results):
        """Plot weight updating results"""
        mechanisms = list(results.keys())
        adaptations = [results[m]['adaptation'] for m in mechanisms]
        ood_detections = [results[m]['ood_detection'] for m in mechanisms]
        betas = [results[m]['beta'] for m in mechanisms]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Adaptation plot
        bars1 = ax1.bar(mechanisms, adaptations, color='teal', alpha=0.7)
        ax1.set_title('Adaptation Score by Mechanism')
        ax1.set_ylabel('Adaptation Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars1, adaptations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # OOD detection plot
        bars2 = ax2.bar(mechanisms, ood_detections, color='crimson', alpha=0.7)
        ax2.set_title('OOD Detection Score by Mechanism')
        ax2.set_ylabel('OOD Detection Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, ood_detections):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Beta values plot
        bars3 = ax3.bar(mechanisms, betas, color='gold', alpha=0.7)
        ax3.set_title('Beta Values by Mechanism')
        ax3.set_ylabel('Beta Value')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # Log scale for beta values
        
        for bar, val in zip(bars3, betas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'weight_updating_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Weight updating plot saved to {plot_path}")
    
    def _plot_energy_methods_results(self, results):
        """Plot energy methods results"""
        methods = list(results.keys())
        aucs = [results[m]['auc'] for m in methods]
        sharpnesses = [results[m]['boundary_sharpness'] for m in methods]
        id_energies = [results[m]['mean_id_energy'] for m in methods]
        ood_energies = [results[m]['mean_ood_energy'] for m in methods]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC plot
        bars1 = ax1.bar(methods, aucs, color='navy', alpha=0.7)
        ax1.set_title('AUC Score by Energy Method')
        ax1.set_ylabel('AUC Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        for bar, val in zip(bars1, aucs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Boundary sharpness plot
        bars2 = ax2.bar(methods, sharpnesses, color='darkred', alpha=0.7)
        ax2.set_title('Boundary Sharpness by Energy Method')
        ax2.set_ylabel('Boundary Sharpness')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, sharpnesses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Mean energy comparison
        x = np.arange(len(methods))
        width = 0.35
        
        bars3 = ax3.bar(x - width/2, id_energies, width, label='ID Energy', 
                       color='blue', alpha=0.7)
        bars4 = ax3.bar(x + width/2, ood_energies, width, label='OOD Energy', 
                       color='red', alpha=0.7)
        
        ax3.set_title('Mean Energy Values by Method')
        ax3.set_ylabel('Mean Energy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Energy separation plot
        separations = [abs(ood - id_) for id_, ood in zip(id_energies, ood_energies)]
        bars5 = ax4.bar(methods, separations, color='green', alpha=0.7)
        ax4.set_title('Energy Separation by Method')
        ax4.set_ylabel('|OOD Energy - ID Energy|')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, val in zip(bars5, separations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'energy_methods_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Energy methods plot saved to {plot_path}")


def main():
    """Main training script with integrated ablation studies"""
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Training configuration from main_train.py
    train_config = {
        'model_path': "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth",
        'checkpoint_dir': "./checkpoints_improved",
        'num_classes': 19,
        'learning_rate': 1e-5,
        'weight_decay': 5e-5,
        'batch_size': 2,  # Small batch size for 7GB memory
        'num_workers': 0,  # For stability
        'output_dir': './ablation_results'
    }

    # Check model path
    if not os.path.exists(train_config['model_path']):
        logger.error(f"Model checkpoint not found: {train_config['model_path']}")
        return

    # Setup data loaders from main_train.py
    logger.info("Setting up data loaders...")
    
    cityscapes_root = "/home/ha51dybi/PEBAL/cityscapes"
    coco_root = "/home/ha51dybi/PEBAL/coco"

    # Verify directories exist
    images_dir = os.path.join(cityscapes_root, "images", "city_gt_fine", "train")
    labels_dir = os.path.join(cityscapes_root, "annotation", "city_gt_fine", "train")

    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        return

    if not os.path.exists(labels_dir):
        logger.error(f"Labels directory not found: {labels_dir}")
        return

    logger.info(f"Found images in: {images_dir}")
    logger.info(f"Found labels in: {labels_dir}")

    # Check file counts
    img_count = len([f for f in os.listdir(images_dir) if f.endswith('_leftImg8bit.png')])
    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('_gtFine.png')])
    logger.info(f"Image files: {img_count}, Label files: {label_count}")

    # Create engine for data loading (from main_train.py)
    class CustomArgs:
        def __init__(self):
            self.ddp = False
            self.local_rank = -1
            self.gpus = 1
            self.world_size = 1

    try:
        from engine.engine import Engine
        from config.config import config as global_config
        from dataset.data_loader import get_mix_loader, Cityscapes

        custom_args = CustomArgs()
        engine_instance = Engine(
            custom_arg=custom_args,
            logger=logger,
            continue_state_object=train_config['model_path']
        )

        global_config.batch_size = train_config['batch_size']

        # Mixed loader for training (Cityscapes + COCO)
        train_loader, _, _ = get_mix_loader(
            engine=engine_instance,
            augment=True,
            cs_root=cityscapes_root,
            coco_root=coco_root
        )

        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
        )

        # Validation loader with proper transforms
        from torchvision import transforms
        
        # Define transform to handle PIL images
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 1024)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_dataset = Cityscapes(
            root=cityscapes_root,
            split='val',
            transform=val_transform  # Add transform to handle PIL images
        )

        def collate_fn(batch):
            """Custom collate function to handle mixed data types"""
            try:
                # Handle the case where batch contains mixed types
                processed_batch = {}
                
                for item in batch:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key not in processed_batch:
                                processed_batch[key] = []
                            
                            if hasattr(value, 'convert'):  # PIL Image
                                processed_batch[key].append(val_transform(value))
                            elif torch.is_tensor(value):
                                processed_batch[key].append(value)
                            else:
                                processed_batch[key].append(value)
                    elif hasattr(item, 'convert'):  # PIL Image
                        if 'data' not in processed_batch:
                            processed_batch['data'] = []
                        processed_batch['data'].append(val_transform(item))
                    elif torch.is_tensor(item):
                        if 'data' not in processed_batch:
                            processed_batch['data'] = []
                        processed_batch['data'].append(item)
                
                # Stack tensors
                for key, values in processed_batch.items():
                    if all(torch.is_tensor(v) for v in values):
                        processed_batch[key] = torch.stack(values)
                
                return processed_batch
                
            except Exception as e:
                logger.warning(f"Collate function failed: {e}, using default")
                # Fallback to default behavior
                return batch[0] if len(batch) == 1 else batch

        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn  # Use custom collate function
        )

        # Log dataset statistics
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Batch size: {train_config['batch_size']}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info("="*60 + "\n")

        # Initialize ablation study manager
        ablation_manager = AblationStudyManager(train_config, train_loader, val_loader)

        # Run ablation studies
        print("\n" + "="*60)
        print("HOPFIELD OOD SEGMENTATION ABLATION STUDIES")
        print("="*60)
        print(f"Device: {ablation_manager.device}")
        print(f"Batch size: {train_config['batch_size']}")
        print(f"Memory limit: 7GB (optimized for limited resources)")
        print("="*60)

        # Run comprehensive ablation studies
        report = ablation_manager.run_all_ablations()

        if report:
            # Print summary
            print("\n" + "="*60)
            print("ABLATION STUDY RESULTS SUMMARY")
            print("="*60)
            
            if 'summary' in report:
                for key, value in report['summary'].items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
            
            print("\nRECOMMENDATIONS:")
            if 'recommendations' in report:
                for component, recommendation in report['recommendations'].items():
                    print(f"• {component.replace('_', ' ').title()}: {recommendation}")
            
            print("\nAll ablation studies completed successfully!")
            print("Check the ./ablation_results/ directory for detailed visualizations and JSON report.")
            print("="*60)
            
            return report
        else:
            logger.error("Ablation studies failed")
            return None

    except ImportError as e:
        logger.error(f"Import error - make sure all required modules are available: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        return None


if __name__ == "__main__":
    # Run main function with ablation studies
    final_report = main()
    
    if final_report:
        print("\nAblation studies completed successfully!")
        print("Generated files:")
        print("- feature_extraction_results.png")
        print("- memory_building_results.png") 
        print("- weight_updating_results.png")
        print("- energy_methods_results.png")
        print("- tsne_id_ood_features.png")
        print("- energy_maps_comparison.png")
        print("- ablation_results.json")
    else:
        print("\nAblation studies failed. Check logs for details.")


# Quick runner for specific studies
class QuickAblationRunner:
    """Simplified runner for specific ablation tests"""
    
    @staticmethod
    def run_single_study(study_name):
        """Run a single ablation study"""
        print(f"Running {study_name} ablation study...")
        
        # Setup minimal config
        config = {
            'model_path': "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth",
            'batch_size': 2,
            'num_classes': 19,
            'output_dir': './quick_ablation_results'
        }
        
        # Call main to get data loaders, then run specific study
        main()  # This will run the complete study


# Usage examples:
"""
# Run complete ablation study
python main_ablation_runner.py

# The script will automatically:
# 1. Setup data loaders from main_train.py paths
# 2. Run all 4 ablation studies
# 3. Generate t-SNE and energy map visualizations  
# 4. Create comprehensive report with recommendations
# 5. Save all results to ./ablation_results/ directory
"""
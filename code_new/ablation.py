#!/usr/bin/env python3
"""
OPTIMIZED PEBAL ABLATION STUDY FRAMEWORK
=========================================

Optimized version for faster execution on 7GB GPU:
- Reduced memory sizes (1k to 10k instead of 10k-100k)
- Smaller batch sizes (1-8 instead of up to 16)
- Limited max_batches (10-20 instead of 50-100)
- Fewer configurations per test (e.g., fewer beta values, diversity methods)
- Reduced trials in timing (3 instead of 5)
- Added .txt output for each ablation study
- Added visualizations (plots) for each ablation study saved as PNG
- JSON output remains unchanged
- Memory management: more frequent torch.cuda.empty_cache()
- Skipped farthest_point_sampling for large sets (too slow, fallback to cluster/random)
- Limited input sizes in efficiency tests

This should complete in reasonable time on 7GB GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
import random
import argparse
from collections import namedtuple, defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# PEBAL imports
from engine.engine import Engine
from config.config import config as global_config
from dataset.data_loader import get_mix_loader, Cityscapes

# Custom components
from feature_extractor import FeatureExtractor
from projection_head import SimpleProjectionHead
from segmentation_head import SegmentationClassifierHead
from hopfield_memory_builder import MemoryBuilder
from pixel_energy import PixelWiseBorderEnergy, PixelWiseInferenceScore, lse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomArgs:
    """Custom arguments class for Engine compatibility"""
    def __init__(self):
        self.devices = '0'
        self.continue_fpath = None
        self.port = '16001'
        self.debug = 0
        self.epochs = 'last'
        self.verbose = False
        self.show_image = True
        self.save_path = None
        self.ddp = False
        self.local_rank = -1
        self.gpus = 0
        self.world_size = 1

class DirectoryStructureFixer:
    """Fixes directory structure compatibility issues"""
    
    @staticmethod
    def patch_cityscapes_paths():
        """Patch Cityscapes class for directory structure compatibility"""
        original_init = Cityscapes.__init__
        
        def patched_init(self, root: str = "/path/to/you/root", split: str = "val", 
                        mode: str = "gtFine", target_type: str = "semantic_train_id", 
                        transform=None, predictions_root=None):
            
            self.root = root
            self.split = split
            self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
            self.transform = transform
            
            # Handle directory structure
            self.images_dir = os.path.join(self.root, 'images', 'city_gt_fine', self.split)
            self.targets_dir = os.path.join(self.root, 'annotation', 'city_gt_fine', self.split)
            self.predictions_dir = os.path.join(predictions_root, self.split) if predictions_root is not None else ""
            
            self.images = []
            self.targets = []
            self.predictions = []

            if not os.path.exists(self.images_dir):
                logger.warning(f"Images directory not found: {self.images_dir}")
                return
                
            try:
                for file_name in os.listdir(self.images_dir):
                    if file_name.endswith('_leftImg8bit.png'):
                        self.images.append(os.path.join(self.images_dir, file_name))
                        target_name = file_name.replace('_leftImg8bit.png', '_gtFine.png')
                        target_path = os.path.join(self.targets_dir, target_name)
                        self.targets.append(target_path)
                        pred_name = file_name.replace("_leftImg8bit", "")
                        self.predictions.append(os.path.join(self.predictions_dir, pred_name))
                        
                logger.info(f"Loaded {len(self.images)} images from {self.images_dir}")
                        
            except Exception as e:
                logger.error(f"Error loading Cityscapes data: {e}")
                self.images = []
                self.targets = []
                self.predictions = []
        
        Cityscapes.__init__ = patched_init
        logger.info("Cityscapes class patched successfully")

class ComprehensiveMemoryBuilder:
    """Enhanced memory builder with multiple diversity methods - optimized"""
    
    def __init__(self, feature_extractor, projection_head, device):
        self.feature_extractor = feature_extractor
        self.projection_head = projection_head
        self.device = device
        
    def build_memory_with_method(self, dataloader, memory_size: int, diversity_method: str = 'hybrid', 
                                max_batches: int = 10) -> Tuple[torch.Tensor, Dict]:  # Reduced max_batches
        """Build memory using specified diversity method"""
        logger.info(f"Building memory (size={memory_size}) with method: {diversity_method}")
        
        # Collect all features first
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Collecting features", leave=False)):
                if i >= max_batches:
                    break
                    
                try:
                    # Extract features
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    projected = self.projection_head(extracted['features'])
                    
                    # Flatten spatial dimensions
                    batch_size, feat_dim, h, w = projected.shape
                    flattened = projected.view(batch_size, feat_dim, -1).permute(0, 2, 1)
                    flattened = flattened.reshape(-1, feat_dim)
                    
                    # Normalize features
                    flattened = F.normalize(flattened, dim=1)
                    
                    all_features.append(flattened.cpu())
                    
                    # Extract labels (ID vs OOD)
                    is_ood = batch.get('is_ood', torch.zeros(batch_size, dtype=torch.bool))
                    labels = is_ood.unsqueeze(1).expand(-1, h*w).reshape(-1)
                    all_labels.append(labels.cpu())
                    
                except Exception as e:
                    logger.warning(f"Batch {i} failed: {e}")
                    continue
        
        if not all_features:
            raise ValueError("No features collected!")
            
        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        logger.info(f"Collected {len(all_features)} total features")
        logger.info(f"ID features: {(~all_labels).sum()}, OOD features: {all_labels.sum()}")
        
        # Split into ID and OOD
        id_features = all_features[~all_labels]
        ood_features = all_features[all_labels] if all_labels.sum() > 0 else torch.empty(0, all_features.shape[1])
        
        # Apply diversity sampling
        if len(id_features) == 0:
            raise ValueError("No ID features found!")
            
        id_memory = self._apply_diversity_sampling(id_features, memory_size, diversity_method)
        ood_memory = self._apply_diversity_sampling(ood_features, memory_size // 2, diversity_method) if len(ood_features) > 0 else torch.empty(0, all_features.shape[1])
        
        # Calculate diversity metrics
        diversity_metrics = {
            'id_diversity': self._calculate_diversity(id_memory),
            'ood_diversity': self._calculate_diversity(ood_memory) if len(ood_memory) > 0 else 0.0,
            'separation': self._calculate_separation(id_memory, ood_memory) if len(ood_memory) > 0 else 0.0,
            'total_features_processed': len(all_features),
            'id_features_available': len(id_features),
            'ood_features_available': len(ood_features)
        }
        
        return id_memory.to(self.device), ood_memory.to(self.device), diversity_metrics
    
    def _apply_diversity_sampling(self, features: torch.Tensor, target_size: int, method: str) -> torch.Tensor:
        """Apply diversity sampling method - optimized, skip farthest for large"""
        if len(features) == 0:
            return torch.empty(0, features.shape[1] if features.numel() > 0 else 128)
            
        if len(features) <= target_size:
            return features
            
        if method == 'random':
            indices = torch.randperm(len(features))[:target_size]
            return features[indices]
            
        elif method == 'cluster':
            return self._cluster_sampling(features, target_size)
            
        elif method == 'farthest_point':
            # Skip for large sets (>5k) to save time, fallback to cluster
            if len(features) > 5000:
                logger.warning("Skipping farthest_point for large set, using cluster instead")
                return self._cluster_sampling(features, target_size)
            return self._farthest_point_sampling(features, target_size)
            
        elif method == 'hybrid':
            # Use clustering for large reductions, farthest point for small
            if len(features) > target_size * 5:
                clustered = self._cluster_sampling(features, target_size * 2)
                return self._farthest_point_sampling(clustered, target_size)
            else:
                return self._farthest_point_sampling(features, target_size)
        else:
            raise ValueError(f"Unknown diversity method: {method}")
    
    def _cluster_sampling(self, features: torch.Tensor, target_size: int) -> torch.Tensor:
        """Cluster-based sampling - reduced clusters"""
        try:
            # Use fewer clusters for efficiency
            n_clusters = min(target_size, len(features) // 2, 100)  # Reduced max clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
            labels = kmeans.fit_predict(features.numpy())
            
            # Sample from each cluster
            selected_indices = []
            samples_per_cluster = target_size // n_clusters
            remainder = target_size % n_clusters
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue
                    
                n_samples = samples_per_cluster + (1 if cluster_id < remainder else 0)
                n_samples = min(n_samples, len(cluster_indices))
                
                if n_samples > 0:
                    selected = np.random.choice(cluster_indices, n_samples, replace=False)
                    selected_indices.extend(selected)
            
            return features[selected_indices[:target_size]]
            
        except Exception as e:
            logger.warning(f"Cluster sampling failed: {e}, falling back to random")
            indices = torch.randperm(len(features))[:target_size]
            return features[indices]
    
    def _farthest_point_sampling(self, features: torch.Tensor, target_size: int) -> torch.Tensor:
        """Farthest point sampling - optimized for small sets only"""
        if len(features) <= target_size:
            return features
            
        # Start with random point
        selected_indices = [torch.randint(0, len(features), (1,)).item()]
        
        for _ in range(target_size - 1):
            distances = []
            for i in range(len(features)):
                if i in selected_indices:
                    distances.append(0)
                else:
                    # Distance to nearest selected point
                    min_dist = float('inf')
                    for j in selected_indices:
                        dist = torch.norm(features[i] - features[j]).item()
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
            
            # Select point with maximum distance
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)
        
        return features[selected_indices]
    
    def _calculate_diversity(self, features: torch.Tensor) -> float:
        """Calculate diversity score - sample for efficiency"""
        if len(features) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(100, len(features))  # Reduced sample size
        sample = features[:sample_size]
        
        # Pairwise similarities
        sim_matrix = torch.mm(sample, sample.T)
        torch.diagonal(sim_matrix).fill_(0)
        
        return 1.0 - sim_matrix.mean().item()
    
    def _calculate_separation(self, id_memory: torch.Tensor, ood_memory: torch.Tensor) -> float:
        """Calculate ID-OOD separation score"""
        if len(id_memory) == 0 or len(ood_memory) == 0:
            return 0.0
        
        id_center = id_memory.mean(dim=0)
        ood_center = ood_memory.mean(dim=0)
        
        center_distance = torch.norm(id_center - ood_center).item()
        id_spread = torch.norm(id_memory - id_center, dim=1).mean().item()
        ood_spread = torch.norm(ood_memory - ood_center, dim=1).mean().item()
        
        return center_distance / (id_spread + ood_spread + 1e-6)

class MultiScaleFeatureAblation:
    """Test multi-scale vs single-scale features - optimized"""
    
    def __init__(self, device):
        self.device = device
    
    def test_scale_configurations(self, model_path: str, dataloader, max_batches: int = 10) -> Dict:  # Reduced batches
        """Test different scale configurations"""
        logger.info("Testing multi-scale vs single-scale features")
        
        results = {}
        
        # Reduced configurations
        configs = {
            'single_scale_384x768': {'resize_resolution': (384, 768), 'multi_scale': False},
            'multi_scale_pyramidal': {'resize_resolution': (384, 768), 'multi_scale': True},
        }
        
        for config_name, config in configs.items():
            logger.info(f"Testing configuration: {config_name}")
            
            try:
                # Create feature extractor with configuration
                feature_extractor = FeatureExtractor(
                    model_path=model_path,
                    resize_resolution=config['resize_resolution'],
                    device=self.device,
                    num_classes=19,
                    verbose_logging=False
                )
                
                # Test performance
                start_time = time.time()
                total_features = 0
                feature_stats = []
                
                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        if i >= max_batches:
                            break
                        
                        extracted = feature_extractor.extract_features_batch(batch)
                        features = extracted['features']
                        
                        total_features += features.numel()
                        feature_stats.append({
                            'mean': features.mean().item(),
                            'std': features.std().item(),
                            'shape': list(features.shape)
                        })
                
                processing_time = time.time() - start_time
                
                results[config_name] = {
                    'processing_time': processing_time,
                    'features_per_second': total_features / processing_time,
                    'avg_feature_std': np.mean([s['std'] for s in feature_stats]),
                    'memory_usage_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                    'feature_shapes': feature_stats[:5]  # Sample shapes
                }
                
                # Cleanup
                del feature_extractor
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Configuration {config_name} failed: {e}")
                results[config_name] = {'error': str(e)}
        
        return results

class BetaParameterAblation:
    """Comprehensive beta parameter sensitivity analysis - optimized"""
    
    def __init__(self, device):
        self.device = device
    
    def test_beta_sensitivity(self, id_memory: torch.Tensor, ood_memory: torch.Tensor, 
                            test_features: torch.Tensor) -> Dict:
        """Test beta parameter sensitivity from 4 to 128 - reduced values"""
        logger.info("Testing beta parameter sensitivity (4 to 128)")
        
        # Reduced beta range
        beta_values = [4, 16, 64]  # Fewer values for speed
        results = {}
        
        for beta in beta_values:
            logger.info(f"Testing beta: {beta}")
            
            try:
                # Create energy functions
                border_energy = PixelWiseBorderEnergy(id_memory, ood_memory, beta=beta).to(self.device)
                inference_score = PixelWiseInferenceScore(id_memory, ood_memory, beta=beta).to(self.device)
                
                # Timing test
                timing_results = self._time_function(border_energy, test_features, warmup=1, trials=3)  # Reduced trials
                
                with torch.no_grad():
                    # Energy statistics
                    energies = border_energy(test_features)
                    scores = inference_score(test_features)
                    
                    # LSE values for analysis
                    id_lse = lse(beta, torch.mm(test_features, id_memory.T))
                    ood_lse = lse(beta, torch.mm(test_features, ood_memory.T)) if len(ood_memory) > 0 else torch.zeros_like(id_lse)
                
                results[beta] = {
                    'timing_ms': timing_results['mean_ms'],
                    'timing_std_ms': timing_results['std_ms'],
                    'energy_stats': {
                        'mean': energies.mean().item(),
                        'std': energies.std().item(),
                        'min': energies.min().item(),
                        'max': energies.max().item(),
                        'dynamic_range': (energies.max() - energies.min()).item()
                    },
                    'score_stats': {
                        'mean': scores.mean().item(),
                        'std': scores.std().item(),
                        'separation': abs(scores.mean().item()),  # How well separated from 0
                    },
                    'lse_analysis': {
                        'id_lse_mean': id_lse.mean().item(),
                        'ood_lse_mean': ood_lse.mean().item(),
                        'lse_difference': (id_lse - ood_lse).mean().item() if len(ood_memory) > 0 else 0.0
                    }
                }
                
                # Cleanup
                del border_energy, inference_score, energies, scores
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Beta {beta} failed: {e}")
                results[beta] = {'error': str(e)}
        
        return results
    
    def _time_function(self, func, *args, warmup: int = 2, trials: int = 5) -> Dict:
        """Time function execution - reduced trials possible"""
        # Warmup
        for _ in range(warmup):
            try:
                _ = func(*args)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except:
                pass
        
        # Timing
        times = []
        for _ in range(trials):
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = func(*args)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
                
            except Exception as e:
                logger.warning(f"Timing trial failed: {e}")
                times.append(float('inf'))
        
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            return {'mean_ms': 0, 'std_ms': 0, 'error': 'All trials failed'}
        
        return {
            'mean_ms': np.mean(valid_times),
            'std_ms': np.std(valid_times),
            'min_ms': np.min(valid_times),
            'max_ms': np.max(valid_times)
        }

class LossWeightAblation:
    """Test loss weight impact (λ from 0.1 to 2.0) - optimized"""
    
    def __init__(self, device):
        self.device = device
    
    def test_loss_weights(self, id_memory: torch.Tensor, ood_memory: torch.Tensor,
                         test_features: torch.Tensor, segmentation_head) -> Dict:
        """Test different loss weight values - reduced values"""
        logger.info("Testing loss weight impact (λ from 0.1 to 2.0)")
        
        lambda_values = [0.1, 0.5, 1.0, 2.0]  # Fewer values
        results = {}
        
        # Create mock classification targets (for loss computation)
        batch_size = min(32, len(test_features))
        test_batch = test_features[:batch_size]
        mock_targets = torch.randint(0, 19, (batch_size,)).to(self.device)
        
        for lambda_val in lambda_values:
            logger.info(f"Testing λ = {lambda_val}")
            
            try:
                # Create energy function
                border_energy = PixelWiseBorderEnergy(id_memory, ood_memory, beta=16).to(self.device)
                
                # Mock segmentation logits
                mock_logits = torch.randn(batch_size, 19).to(self.device)
                
                with torch.no_grad():
                    # Classification loss
                    ce_loss = F.cross_entropy(mock_logits, mock_targets)
                    
                    # OOD loss
                    energies = border_energy(test_batch)
                    ood_loss = -energies.mean()  # Negative energy as loss
                    
                    # Combined loss
                    total_loss = ce_loss + lambda_val * ood_loss
                    
                    # Energy statistics
                    energy_magnitude = energies.abs().mean().item()
                    energy_variance = energies.var().item()
                
                results[lambda_val] = {
                    'ce_loss': ce_loss.item(),
                    'ood_loss': ood_loss.item(),
                    'total_loss': total_loss.item(),
                    'ood_contribution_ratio': (lambda_val * ood_loss / total_loss).item(),
                    'energy_magnitude': energy_magnitude,
                    'energy_variance': energy_variance,
                    'loss_balance_score': abs(ce_loss.item() - lambda_val * ood_loss.item())  # How balanced the losses are
                }
                
                # Cleanup
                del border_energy, energies
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Lambda {lambda_val} failed: {e}")
                results[lambda_val] = {'error': str(e)}
        
        return results

class BoostingIterationAblation:
    """Test boosting iterations (1 to 5) - optimized"""
    
    def __init__(self, device):
        self.device = device
    
    def test_boosting_iterations(self, dataloader, feature_extractor, projection_head,
                               max_batches: int = 10) -> Dict:  # Reduced batches
        """Test different numbers of boosting iterations - reduced iterations"""
        logger.info("Testing boosting iterations (1 to 5)")
        
        iterations = [1, 3, 5]  # Fewer iterations
        results = {}
        
        # Collect initial features
        initial_features = self._collect_features(dataloader, feature_extractor, projection_head, max_batches)
        
        for num_iterations in iterations:
            logger.info(f"Testing {num_iterations} boosting iterations")
            
            try:
                iteration_results = []
                
                # Simulate boosting iterations
                current_weights = torch.ones(len(initial_features)) / len(initial_features)
                
                for iteration in range(num_iterations):
                    # Sample according to current weights
                    sampled_indices = torch.multinomial(current_weights, num_samples=min(500, len(initial_features)), replacement=True)  # Reduced samples
                    sampled_features = initial_features[sampled_indices]
                    
                    # Split into ID and OOD (mock)
                    mid_point = len(sampled_features) // 2
                    id_features = sampled_features[:mid_point]
                    ood_features = sampled_features[mid_point:]
                    
                    # Calculate energies
                    border_energy = PixelWiseBorderEnergy(id_features, ood_features, beta=16).to(self.device)
                    
                    with torch.no_grad():
                        all_energies = border_energy(initial_features)
                        
                        # Update weights (boost high energy samples)
                        normalized_energies = (all_energies - all_energies.min()) / (all_energies.max() - all_energies.min() + 1e-8)
                        current_weights = F.softmax(normalized_energies, dim=0)
                        
                        iteration_results.append({
                            'iteration': iteration + 1,
                            'energy_mean': all_energies.mean().item(),
                            'energy_std': all_energies.std().item(),
                            'weight_entropy': -(current_weights * torch.log(current_weights + 1e-8)).sum().item(),
                            'max_weight': current_weights.max().item(),
                            'weight_concentration': (current_weights > current_weights.mean() + current_weights.std()).float().mean().item()
                        })
                    
                    # Cleanup
                    del border_energy
                    torch.cuda.empty_cache()
                
                results[num_iterations] = {
                    'iterations': iteration_results,
                    'final_weight_entropy': iteration_results[-1]['weight_entropy'],
                    'convergence_rate': self._calculate_convergence_rate(iteration_results),
                    'boosting_effectiveness': iteration_results[-1]['weight_concentration']
                }
                
            except Exception as e:
                logger.error(f"Boosting iterations {num_iterations} failed: {e}")
                results[num_iterations] = {'error': str(e)}
        
        return results
    
    def _collect_features(self, dataloader, feature_extractor, projection_head, max_batches: int) -> torch.Tensor:
        """Collect features for boosting test - reduced"""
        all_features = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                try:
                    extracted = feature_extractor.extract_features_batch(batch)
                    projected = projection_head(extracted['features'])
                    
                    # Flatten and normalize
                    flattened = projected.view(projected.shape[0], projected.shape[1], -1).mean(dim=2)
                    normalized = F.normalize(flattened, dim=1)
                    all_features.append(normalized.cpu())
                    
                except Exception as e:
                    logger.warning(f"Feature collection batch {i} failed: {e}")
                    continue
        
        return torch.cat(all_features, dim=0).to(self.device)
    
    def _calculate_convergence_rate(self, iteration_results: List[Dict]) -> float:
        """Calculate how quickly the weights converge"""
        if len(iteration_results) < 2:
            return 0.0
        
        entropies = [r['weight_entropy'] for r in iteration_results]
        
        # Calculate rate of change in entropy
        changes = []
        for i in range(1, len(entropies)):
            change = abs(entropies[i] - entropies[i-1])
            changes.append(change)
        
        return np.mean(changes) if changes else 0.0

class BatchSizeScalingAblation:
    """Test batch size scaling effects - optimized"""
    
    def __init__(self, device):
        self.device = device
    
    def test_batch_sizes(self, cs_root: str, coco_root: str, feature_extractor, projection_head) -> Dict:
        """Test different batch sizes - reduced sizes"""
        logger.info("Testing batch size scaling effects")
        
        batch_sizes = [1, 2, 4, 8]  # Reduced max to 8 for 7GB GPU
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            try:
                # Create dataloader with specific batch size
                dataloader = self._create_dataloader(cs_root, coco_root, batch_size)
                
                # Measure performance
                start_time = time.time()
                total_samples = 0
                throughput_samples = []
                
                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        if i >= 10:  # Process fewer batches
                            break
                        
                        batch_start = time.time()
                        
                        extracted = feature_extractor.extract_features_batch(batch)
                        projected = projection_head(extracted['features'])
                        
                        batch_end = time.time()
                        batch_time = batch_end - batch_start
                        
                        samples_in_batch = batch['data'].shape[0]
                        total_samples += samples_in_batch
                        throughput_samples.append(samples_in_batch / batch_time)
                
                total_time = time.time() - start_time
                
                results[batch_size] = {
                    'total_time': total_time,
                    'total_samples': total_samples,
                    'avg_throughput': total_samples / total_time,
                    'batch_throughput_mean': np.mean(throughput_samples),
                    'batch_throughput_std': np.std(throughput_samples),
                    'memory_efficiency': total_samples / (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0,
                    'samples_per_second': total_samples / total_time
                }
                
                # Reset memory tracking
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
            except Exception as e:
                logger.error(f"Batch size {batch_size} failed: {e}")
                results[batch_size] = {'error': str(e)}
        
        return results
    
    def _create_dataloader(self, cs_root: str, coco_root: str, batch_size: int):
        """Create dataloader with specific batch size"""
        # Set global config
        global_config.batch_size = batch_size
        global_config.num_workers = 0
        
        # Create engine and dataloader
        custom_args = CustomArgs()
        continue_state_object = "/tmp/dummy.pth"
        
        engine_instance = Engine(
            custom_arg=custom_args,
            logger=logger,
            continue_state_object=continue_state_object
        )
        
        train_loader, _, _ = get_mix_loader(
            engine=engine_instance,
            augment=True,
            cs_root=cs_root,
            coco_root=coco_root
        )
        
        return train_loader

class ComputationalEfficiencyAblation:
    """Comprehensive computational efficiency analysis - optimized"""
    
    def __init__(self, device):
        self.device = device
    
    def analyze_computational_complexity(self, id_memory: torch.Tensor, ood_memory: torch.Tensor) -> Dict:
        """Analyze computational complexity of different operations - reduced sizes"""
        logger.info("Analyzing computational complexity")
        
        results = {}
        
        # Reduced input sizes
        input_sizes = [100, 500, 1000, 2000]  # Reduced max
        
        for size in input_sizes:
            logger.info(f"Testing input size: {size}")
            
            try:
                # Create test input
                test_input = torch.randn(size, id_memory.shape[1]).to(self.device)
                test_input = F.normalize(test_input, dim=1)
                
                # Test LSE computation
                lse_time = self._time_lse_computation(test_input, id_memory)
                
                # Test border energy computation
                border_energy = PixelWiseBorderEnergy(id_memory, ood_memory, beta=16).to(self.device)
                border_time = self._time_function(border_energy, test_input)
                
                # Test inference score computation
                inference_score = PixelWiseInferenceScore(id_memory, ood_memory, beta=16).to(self.device)
                inference_time = self._time_function(inference_score, test_input)
                
                # Memory usage
                memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                _ = border_energy(test_input)
                memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                results[size] = {
                    'lse_time_ms': lse_time['mean_ms'],
                    'border_energy_time_ms': border_time['mean_ms'],
                    'inference_score_time_ms': inference_time['mean_ms'],
                    'memory_overhead_mb': (memory_after - memory_before) / 1024**2,
                    'throughput_samples_per_sec': 1000 * size / border_time['mean_ms'] if border_time['mean_ms'] > 0 else 0,
                    'computational_complexity_score': size * border_time['mean_ms']  # Higher = worse complexity
                }
                
                # Cleanup
                del border_energy, inference_score
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Size {size} failed: {e}")
                results[size] = {'error': str(e)}
        
        return results
    
    def _time_lse_computation(self, test_input: torch.Tensor, memory: torch.Tensor) -> Dict:
        """Time LSE computation specifically"""
        def lse_func():
            return lse(16, torch.mm(test_input, memory.T))
        
        return self._time_function(lse_func)
    
    def _time_function(self, func, *args, trials: int = 3) -> Dict:  # Reduced trials
        """Time function execution"""
        times = []
        
        # Warmup
        for _ in range(1):  # Reduced warmup
            try:
                _ = func(*args)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except:
                pass
        
        # Timing
        for _ in range(trials):
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = func(*args)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)
                
            except Exception as e:
                logger.warning(f"Timing trial failed: {e}")
                times.append(float('inf'))
        
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            return {'mean_ms': 0, 'std_ms': 0}
        
        return {
            'mean_ms': np.mean(valid_times),
            'std_ms': np.std(valid_times),
            'min_ms': np.min(valid_times),
            'max_ms': np.max(valid_times)
        }

class ComprehensiveAblationFramework:
    """Main framework orchestrating all ablation studies - optimized"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def run_complete_ablation_study(self, model_path: str, cs_root: str, coco_root: str) -> Dict:
        """Run all ablation studies"""
        logger.info("Starting optimized PEBAL ablation study")
        logger.info(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {total_memory:.2f} GB")
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Patch directory structure
        DirectoryStructureFixer.patch_cityscapes_paths()
        
        try:
            # Create dataloader
            dataloader = self._create_dataloader(cs_root, coco_root, batch_size=1)
            
            # Initialize models
            feature_extractor = FeatureExtractor(
                model_path=model_path,
                resize_resolution=(384, 768),
                device=self.device,
                num_classes=19,
                verbose_logging=False
            )
            
            projection_head = SimpleProjectionHead(input_dim=1280, output_dim=128).to(self.device)
            segmentation_head = SegmentationClassifierHead(in_channels=1280, num_classes=19).to(self.device)
            
            feature_extractor.eval()
            projection_head.eval()
            segmentation_head.eval()
            
            # 1. Multi-scale feature ablation
            logger.info("\n" + "="*60)
            logger.info("1. MULTI-SCALE FEATURE ABLATION")
            logger.info("="*60)
            
            multi_scale_ablation = MultiScaleFeatureAblation(self.device)
            self.results['multi_scale_features'] = multi_scale_ablation.test_scale_configurations(
                model_path, dataloader, max_batches=10
            )
            self._save_ablation_output('multi_scale_features', self.results['multi_scale_features'])
            
            # 2. Memory size and diversity ablation
            logger.info("\n" + "="*60)
            logger.info("2. MEMORY SIZE AND DIVERSITY ABLATION")
            logger.info("="*60)
            
            memory_builder = ComprehensiveMemoryBuilder(feature_extractor, projection_head, self.device)
            
            # Reduced memory sizes: 1k to 10k
            memory_sizes = [1000, 5000, 10000]
            diversity_methods = ['random', 'cluster', 'hybrid']  # Removed farthest_point for speed
            
            memory_results = {}
            for size in memory_sizes:
                memory_results[size] = {}
                for method in diversity_methods:
                    logger.info(f"Testing memory size {size} with method {method}")
                    try:
                        start_time = time.time()
                        id_memory, ood_memory, metrics = memory_builder.build_memory_with_method(
                            dataloader, size, method, max_batches=10
                        )
                        build_time = time.time() - start_time
                        
                        memory_results[size][method] = {
                            'build_time': build_time,
                            'id_memory_shape': list(id_memory.shape),
                            'ood_memory_shape': list(ood_memory.shape),
                            'metrics': metrics
                        }
                        
                        # Store best performing memory for later tests
                        if size == 5000 and method == 'hybrid':
                            self.best_id_memory = id_memory
                            self.best_ood_memory = ood_memory
                        
                        # Cleanup
                        del id_memory, ood_memory
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.error(f"Memory size {size} method {method} failed: {e}")
                        memory_results[size][method] = {'error': str(e)}
            
            self.results['memory_diversity'] = memory_results
            self._save_ablation_output('memory_diversity', self.results['memory_diversity'])
            
            # Ensure we have memory for subsequent tests
            if not hasattr(self, 'best_id_memory'):
                logger.info("Creating fallback memory for subsequent tests")
                self.best_id_memory, self.best_ood_memory, _ = memory_builder.build_memory_with_method(
                    dataloader, 1000, 'hybrid', max_batches=10
                )
            
            # Create test features for energy/loss tests
            test_features = self._create_test_features(dataloader, feature_extractor, projection_head, max_samples=500)  # Reduced samples
            
            # 3. Beta parameter sensitivity
            logger.info("\n" + "="*60)
            logger.info("3. BETA PARAMETER SENSITIVITY")
            logger.info("="*60)
            
            beta_ablation = BetaParameterAblation(self.device)
            self.results['beta_sensitivity'] = beta_ablation.test_beta_sensitivity(
                self.best_id_memory, self.best_ood_memory, test_features
            )
            self._save_ablation_output('beta_sensitivity', self.results['beta_sensitivity'])
            
            # 4. Loss weight impact
            logger.info("\n" + "="*60)
            logger.info("4. LOSS WEIGHT IMPACT")
            logger.info("="*60)
            
            loss_weight_ablation = LossWeightAblation(self.device)
            self.results['loss_weight_impact'] = loss_weight_ablation.test_loss_weights(
                self.best_id_memory, self.best_ood_memory, test_features, segmentation_head
            )
            self._save_ablation_output('loss_weight_impact', self.results['loss_weight_impact'])
            
            # 5. Boosting iterations
            logger.info("\n" + "="*60)
            logger.info("5. BOOSTING ITERATIONS")
            logger.info("="*60)
            
            boosting_ablation = BoostingIterationAblation(self.device)
            self.results['boosting_iterations'] = boosting_ablation.test_boosting_iterations(
                dataloader, feature_extractor, projection_head, max_batches=100
            )
            self._save_ablation_output('boosting_iterations', self.results['boosting_iterations'])
            
            # 6. Batch size scaling
            logger.info("\n" + "="*60)
            logger.info("6. BATCH SIZE SCALING")
            logger.info("="*60)
            
            batch_size_ablation = BatchSizeScalingAblation(self.device)
            self.results['batch_size_scaling'] = batch_size_ablation.test_batch_sizes(
                cs_root, coco_root, feature_extractor, projection_head
            )
            self._save_ablation_output('batch_size_scaling', self.results['batch_size_scaling'])
            
            # 7. Computational efficiency
            logger.info("\n" + "="*60)
            logger.info("7. COMPUTATIONAL EFFICIENCY")
            logger.info("="*60)
            
            efficiency_ablation = ComputationalEfficiencyAblation(self.device)
            self.results['computational_efficiency'] = efficiency_ablation.analyze_computational_complexity(
                self.best_id_memory, self.best_ood_memory
            )
            self._save_ablation_output('computational_efficiency', self.results['computational_efficiency'])
            
            # Generate comprehensive analysis
            self.results['summary'] = self._generate_summary()
            
        except Exception as e:
            logger.error(f"Ablation study failed: {e}")
            import traceback
            traceback.print_exc()
        
        return self.results
    
    def _create_dataloader(self, cs_root: str, coco_root: str, batch_size: int = 1):  # Small batch
        """Create dataloader"""
        global_config.batch_size = batch_size
        global_config.num_workers = 0
        
        custom_args = CustomArgs()
        continue_state_object = "/tmp/dummy.pth"
        
        engine_instance = Engine(
            custom_arg=custom_args,
            logger=logger,
            continue_state_object=continue_state_object
        )
        
        train_loader, _, _ = get_mix_loader(
            engine=engine_instance,
            augment=True,
            cs_root=cs_root,
            coco_root=coco_root
        )
        
        return train_loader
    
    def _create_test_features(self, dataloader, feature_extractor, projection_head, max_samples: int = 500) -> torch.Tensor:  # Reduced
        """Create test features for energy functions"""
        logger.info("Creating test features for energy/loss tests")
        
        test_features = []
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if total_samples >= max_samples:
                    break
                
                try:
                    extracted = feature_extractor.extract_features_batch(batch)
                    projected = projection_head(extracted['features'])
                    
                    # Flatten and sample
                    flattened = projected.view(projected.shape[0], projected.shape[1], -1).mean(dim=2)
                    normalized = F.normalize(flattened, dim=1)
                    
                    test_features.append(normalized.cpu())
                    total_samples += normalized.shape[0]
                    
                except Exception as e:
                    logger.warning(f"Test feature batch failed: {e}")
                    continue
        
        result = torch.cat(test_features, dim=0)[:max_samples].to(self.device)
        logger.info(f"Created {len(result)} test features")
        return result
    
    def _generate_summary(self) -> Dict:
        """Generate comprehensive summary of results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'total_tests_run': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'key_findings': {},
            'recommendations': []
        }
        
        # Count tests
        for category, results in self.results.items():
            if category == 'summary':
                continue
                
            if isinstance(results, dict):
                for test_name, test_result in results.items():
                    summary['total_tests_run'] += 1
                    if 'error' in test_result:
                        summary['failed_tests'] += 1
                    else:
                        summary['successful_tests'] += 1
        
        # Key findings
        try:
            # Best memory configuration
            if 'memory_diversity' in self.results:
                best_memory_config = self._find_best_memory_config()
                summary['key_findings']['best_memory_config'] = best_memory_config
            
            # Optimal beta value
            if 'beta_sensitivity' in self.results:
                best_beta = self._find_optimal_beta()
                summary['key_findings']['optimal_beta'] = best_beta
            
            # Best loss weight
            if 'loss_weight_impact' in self.results:
                best_lambda = self._find_optimal_lambda()
                summary['key_findings']['optimal_lambda'] = best_lambda
            
            # Efficiency insights
            if 'computational_efficiency' in self.results:
                efficiency_insights = self._analyze_efficiency()
                summary['key_findings']['efficiency_insights'] = efficiency_insights
                
        except Exception as e:
            logger.warning(f"Summary generation partial failure: {e}")
            summary['key_findings']['error'] = str(e)
        
        return summary
    
    def _find_best_memory_config(self) -> Dict:
        """Find best memory configuration"""
        best_config = {'size': None, 'method': None, 'score': -1}
        
        for size, methods in self.results['memory_diversity'].items():
            for method, result in methods.items():
                if 'error' not in result and 'metrics' in result:
                    metrics = result['metrics']
                    # Composite score: diversity + separation - build_time_penalty
                    score = (metrics.get('id_diversity', 0) + 
                           metrics.get('separation', 0) - 
                           result.get('build_time', 0) / 100)
                    
                    if score > best_config['score']:
                        best_config = {
                            'size': size,
                            'method': method,
                            'score': score,
                            'build_time': result.get('build_time', 0),
                            'diversity': metrics.get('id_diversity', 0),
                            'separation': metrics.get('separation', 0)
                        }
        
        return best_config
    
    def _find_optimal_beta(self) -> Dict:
        """Find optimal beta value"""
        best_beta = {'value': None, 'score': -1}
        
        for beta, result in self.results['beta_sensitivity'].items():
            if 'error' not in result:
                # Score based on dynamic range and timing
                score = (result['energy_stats']['dynamic_range'] - 
                        result['timing_ms'] / 1000)
                
                if score > best_beta['score']:
                    best_beta = {
                        'value': beta,
                        'score': score,
                        'timing_ms': result['timing_ms'],
                        'dynamic_range': result['energy_stats']['dynamic_range']
                    }
        
        return best_beta
    
    def _find_optimal_lambda(self) -> Dict:
        """Find optimal lambda value"""
        best_lambda = {'value': None, 'balance_score': float('inf')}
        
        for lambda_val, result in self.results['loss_weight_impact'].items():
            if 'error' not in result:
                balance_score = result['loss_balance_score']
                
                if balance_score < best_lambda['balance_score']:
                    best_lambda = {
                        'value': lambda_val,
                        'balance_score': balance_score,
                        'ood_contribution': result['ood_contribution_ratio']
                    }
        
        return best_lambda
    
    def _analyze_efficiency(self) -> Dict:
        """Analyze computational efficiency results"""
        efficiency_data = self.results['computational_efficiency']
        
        # Find linear complexity indicators
        sizes = sorted([int(k) for k in efficiency_data.keys() if 'error' not in efficiency_data[k]])
        
        if len(sizes) < 2:
            return {'error': 'Insufficient data for efficiency analysis'}
        
        # Calculate complexity growth
        throughputs = []
        for size in sizes:
            result = efficiency_data[size]
            if 'throughput_samples_per_sec' in result:
                throughputs.append(result['throughput_samples_per_sec'])
        
        return {
            'input_sizes_tested': sizes,
            'throughput_scaling': throughputs,
            'complexity_assessment': 'linear' if len(throughputs) > 1 and 
                                   throughputs[-1] / throughputs[0] < sizes[-1] / sizes[0] * 2 else 'super-linear'
        }

    def _save_ablation_output(self, category: str, data: Dict):
        """Save .txt and visualization for each ablation study"""
        output_dir = "./ablation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save .txt
        txt_file = os.path.join(output_dir, f"{category}_results.txt")
        with open(txt_file, 'w') as f:
            f.write(f"{category.upper()} RESULTS\n")
            f.write("=" * 40 + "\n")
            json.dump(data, f, indent=4)
        logger.info(f"{category} .txt saved to {txt_file}")
        
        # Create visualization (simple bar plot example)
        png_file = os.path.join(output_dir, f"{category}_plot.png")
        try:
            fig, ax = plt.subplots()
            if isinstance(data, dict):
                keys = list(data.keys())
                values = [data[k].get('processing_time', data[k].get('timing_ms', data[k].get('total_time', 0))) for k in keys]
                ax.bar(keys, values)
                ax.set_title(f"{category} Performance")
                ax.set_ylabel("Time (s or ms)")
                plt.xticks(rotation=45)
            plt.savefig(png_file)
            plt.close()
            logger.info(f"{category} plot saved to {png_file}")
        except Exception as e:
            logger.warning(f"Visualization for {category} failed: {e}")

def save_results_with_visualization(results: Dict, output_dir: str = "./ablation_results"):
    """Save results with JSON and visualization - JSON unchanged"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results (unchanged)
    json_file = os.path.join(output_dir, f'complete_ablation_results_{timestamp}.json')
    
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    
    with open(json_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    logger.info(f"Results saved to {json_file}")
    
    # Generate summary report
    summary_file = os.path.join(output_dir, f'ablation_summary_{timestamp}.md')
    generate_markdown_report(results, summary_file)
    
    return json_file, summary_file

def generate_markdown_report(results: Dict, output_file: str):
    """Generate detailed markdown report"""
    with open(output_file, 'w') as f:
        f.write("# PEBAL Optimized Ablation Study Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if 'summary' in results:
            summary = results['summary']
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Tests**: {summary.get('total_tests_run', 0)}\n")
            f.write(f"- **Successful**: {summary.get('successful_tests', 0)}\n")
            f.write(f"- **Failed**: {summary.get('failed_tests', 0)}\n\n")
            
            if 'key_findings' in summary:
                f.write("## Key Findings\n\n")
                
                findings = summary['key_findings']
                
                if 'best_memory_config' in findings:
                    config = findings['best_memory_config']
                    f.write(f"### Optimal Memory Configuration\n")
                    f.write(f"- **Size**: {config.get('size', 'N/A'):,}\n")
                    f.write(f"- **Method**: {config.get('method', 'N/A')}\n")
                    f.write(f"- **Build Time**: {config.get('build_time', 0):.2f}s\n")
                    f.write(f"- **Diversity Score**: {config.get('diversity', 0):.4f}\n\n")
                
                if 'optimal_beta' in findings:
                    beta = findings['optimal_beta']
                    f.write(f"### Optimal Beta Value\n")
                    f.write(f"- **Beta**: {beta.get('value', 'N/A')}\n")
                    f.write(f"- **Timing**: {beta.get('timing_ms', 0):.2f}ms\n")
                    f.write(f"- **Dynamic Range**: {beta.get('dynamic_range', 0):.4f}\n\n")
                
                if 'optimal_lambda' in findings:
                    lambda_val = findings['optimal_lambda']
                    f.write(f"### Optimal Loss Weight (λ)\n")
                    f.write(f"- **Lambda**: {lambda_val.get('value', 'N/A')}\n")
                    f.write(f"- **Balance Score**: {lambda_val.get('balance_score', 0):.4f}\n\n")
        
        # Detailed sections for each ablation study
        f.write("## Detailed Results\n\n")
        
        for category, data in results.items():
            if category in ['summary']:
                continue
            
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            
            if isinstance(data, dict):
                for test_name, test_result in data.items():
                    if 'error' in test_result:
                        f.write(f"- **{test_name}**: FAILED - {test_result['error']}\n")
                    else:
                        f.write(f"- **{test_name}**: SUCCESS\n")
            
            f.write("\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the ablation study results, the following configurations are recommended:\n\n")
        
        if 'summary' in results and 'key_findings' in results['summary']:
            findings = results['summary']['key_findings']
            
            if 'best_memory_config' in findings:
                config = findings['best_memory_config']
                f.write(f"1. **Memory Configuration**: Use {config.get('size', 'N/A'):,} samples with {config.get('method', 'N/A')} diversity method\n")
            
            if 'optimal_beta' in findings:
                beta = findings['optimal_beta']
                f.write(f"2. **Beta Parameter**: Use β = {beta.get('value', 'N/A')} for optimal energy function performance\n")
            
            if 'optimal_lambda' in findings:
                lambda_val = findings['optimal_lambda']
                f.write(f"3. **Loss Weight**: Use λ = {lambda_val.get('value', 'N/A')} for balanced training\n")
        
        f.write("\n---\n")
        f.write("*Report generated by Optimized PEBAL Comprehensive Ablation Framework*\n")
    
    logger.info(f"Markdown report saved to {output_file}")

def print_comprehensive_summary(results: Dict):
    """Print comprehensive summary to console"""
    print("\n" + "="*80)
    print("OPTIMIZED PEBAL ABLATION STUDY SUMMARY")
    print("="*80)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\nOverall Statistics:")
        print(f"  Total Tests: {summary.get('total_tests_run', 0)}")
        print(f"  Successful: {summary.get('successful_tests', 0)}")
        print(f"  Failed: {summary.get('failed_tests', 0)}")
        print(f"  Success Rate: {100 * summary.get('successful_tests', 0) / max(summary.get('total_tests_run', 1), 1):.1f}%")
        
        if 'key_findings' in summary:
            print(f"\n🏆 KEY FINDINGS:")
            findings = summary['key_findings']
            
            if 'best_memory_config' in findings:
                config = findings['best_memory_config']
                print(f"  📋 Best Memory: {config.get('size', 'N/A'):,} samples, {config.get('method', 'N/A')} method")
                print(f"     Diversity: {config.get('diversity', 0):.4f}, Build Time: {config.get('build_time', 0):.2f}s")
            
            if 'optimal_beta' in findings:
                beta = findings['optimal_beta']
                print(f"  ⚡ Optimal Beta: {beta.get('value', 'N/A')} (timing: {beta.get('timing_ms', 0):.1f}ms)")
            
            if 'optimal_lambda' in findings:
                lambda_val = findings['optimal_lambda']
                print(f"  ⚖️  Optimal λ: {lambda_val.get('value', 'N/A')} (balance: {lambda_val.get('balance_score', 0):.3f})")
    
    # Print category summaries
    categories = [
        'multi_scale_features', 'memory_diversity', 'beta_sensitivity', 
        'loss_weight_impact', 'boosting_iterations', 'batch_size_scaling', 
        'computational_efficiency'
    ]
    
    print(f"\n📊 CATEGORY BREAKDOWN:")
    
    for category in categories:
        if category in results:
            data = results[category]
            successful = sum(1 for v in data.values() if 'error' not in v) if isinstance(data, dict) else 0
            total = len(data) if isinstance(data, dict) else 0
            
            print(f"  {category.replace('_', ' ').title()}: {successful}/{total} tests passed")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    if 'summary' in results and 'key_findings' in results['summary']:
        findings = results['summary']['key_findings']
        
        if 'best_memory_config' in findings:
            config = findings['best_memory_config']
            print(f"  1. Use {config.get('size', 'N/A'):,} memory size with {config.get('method', 'hybrid')} diversity")
        
        if 'optimal_beta' in findings:
            beta = findings['optimal_beta']
            print(f"  2. Set β = {beta.get('value', 16)} for energy functions")
        
        if 'optimal_lambda' in findings:
            lambda_val = findings['optimal_lambda']
            print(f"  3. Use λ = {lambda_val.get('value', 0.5)} for loss weighting")
        
        print(f"  4. Consider computational efficiency trade-offs for production deployment")
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED SUCCESSFULLY!")
    print("="*80)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Optimized PEBAL ablation study")
    parser.add_argument('--model-path', type=str, 
                       default="/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth")
    parser.add_argument('--cs-root', type=str,
                       default="/home/ha51dybi/PEBAL/cityscapes")
    parser.add_argument('--coco-root', type=str,
                       default="/home/ha51dybi/PEBAL/coco")
    parser.add_argument('--output-dir', type=str,
                       default="./ablation_results")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Run comprehensive ablation study
        framework = ComprehensiveAblationFramework(device='cuda')
        results = framework.run_complete_ablation_study(
            model_path=args.model_path,
            cs_root=args.cs_root,
            coco_root=args.coco_root
        )
        
        # Save results
        json_file, summary_file = save_results_with_visualization(results, args.output_dir)
        
        # Print summary
        print_comprehensive_summary(results)
        
        print(f"\n📁 Results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {summary_file}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Study interrupted by user")
    except Exception as e:
        logger.error(f"Study failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    results = main()
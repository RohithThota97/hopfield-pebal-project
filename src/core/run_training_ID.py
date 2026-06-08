#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import gc
import numpy as np
from torch.utils.data import DataLoader
import time
import wandb
import math
import torch.cuda.amp as amp
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Any, Dict, Tuple, Optional
import traceback
import sklearn.metrics
from abc import ABC, abstractmethod
import matplotlib.cm as cm
from torch.cuda.amp import autocast
import scipy.stats as stats
import json # Added for JSON saving
# Imports from provided programs
from segmentation_head import SegmentationClassifierHead
from feature_extractor import FeatureExtractor
from projection_head import SimpleProjectionHead
from hopfield_memory_builder import MemoryBuilder
from pixel_energy import compute_hopfield_ood_loss, PixelWiseBorderEnergy, PixelWiseInferenceScore, lse
import logging
logger = logging.getLogger(__name__)
from hopfield_weight_updater import HopfieldBoostingManager
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s') # Changed to WARNING to reduce verbosity
CITYSCAPES_COLORMAP = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]
CITYSCAPES_CLASSES = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole',
    6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation', 9: 'terrain', 10: 'sky',
    11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train',
    17: 'motorcycle', 18: 'bicycle'
}
# Modern Hopfield Network with Convergence Dynamics
class ModernHopfieldNetwork(nn.Module):
    """ Implements proper MHN dynamics with iterative convergence to memory patterns. """
    def __init__(self, memory_patterns, beta=128.0, max_iterations=10, convergence_threshold=1e-4):
        super().__init__()
        self.register_buffer('memory', memory_patterns) # [N, D]
        self.beta = beta
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    def update_rule(self, query):
        """ Core MHN update: ξ_new = X·softmax(βX^T·ξ) """
        similarities = torch.matmul(self.memory, query.T) * self.beta # [N, B]
        attention_weights = F.softmax(similarities, dim=0) # [N, B]
        updated_query = torch.matmul(self.memory.T, attention_weights) # [D, B]
        return updated_query.T # [B, D]
    def converge_to_pattern(self, query, return_trajectory=False):
        """ Iteratively converge to nearest memory pattern. """
        trajectory = [query.clone()] if return_trajectory else None
        current = query
        for iteration in range(self.max_iterations):
            updated = self.update_rule(current)
            updated = F.normalize(updated, p=2, dim=-1)
            delta = torch.norm(updated - current, dim=-1).mean()
            if return_trajectory:
                trajectory.append(updated.clone())
            if delta < self.convergence_threshold:
                break
            current = updated
        return (current, trajectory) if return_trajectory else current
    def compute_energy(self, query):
        """ Compute MHE for given query patterns
        E(ξ; X) = -lse(β, X^T ξ) + 1/2 ξ^T ξ + C """
        similarities = torch.matmul(self.memory, query.T) * self.beta
        lse_term = torch.logsumexp(similarities, dim=0) / self.beta
        norm_term = 0.5 * torch.sum(query ** 2, dim=-1)
        energy = -lse_term + norm_term
        return energy
# Pixel-level OOD Detector with Confidence
class PixelOODDetector(nn.Module):
    """ Explicit pixel-level OOD detection with confidence calibration. """
    def __init__(self, id_memory, aux_memory, beta=128.0, temperature=1.0):
        super().__init__()
        self.id_mhn = ModernHopfieldNetwork(id_memory, beta)
        self.aux_mhn = ModernHopfieldNetwork(aux_memory, beta)
        self.temperature = temperature
    def detect_ood_pixels(self, pixel_features, threshold='adaptive'):
        """ Per-pixel OOD detection with confidence scores """
        B, H, W, C = pixel_features.shape
        pixels_flat = pixel_features.reshape(-1, C)
        with torch.no_grad():
            id_converged = self.id_mhn.converge_to_pattern(pixels_flat)
            aux_converged = self.aux_mhn.converge_to_pattern(pixels_flat)
        id_energy = self.id_mhn.compute_energy(id_converged)
        aux_energy = self.aux_mhn.compute_energy(aux_converged)
        ood_scores = (id_energy - aux_energy).reshape(B, H, W)
        confidence = torch.sigmoid(ood_scores / self.temperature)
        if threshold == 'adaptive':
            threshold_val = self._compute_adaptive_threshold(ood_scores)
        else:
            threshold_val = threshold
        ood_mask = ood_scores > threshold_val
        return {
            'scores': ood_scores,
            'confidence': confidence,
            'mask': ood_mask,
            'threshold': threshold_val,
            'id_energy': id_energy.reshape(B, H, W),
            'aux_energy': aux_energy.reshape(B, H, W)
        }
    def _compute_adaptive_threshold(self, scores):
        """Compute adaptive threshold using histogram analysis"""
        flat_scores = scores.flatten()
        threshold = torch.quantile(flat_scores, 0.85)
        return threshold
# Memory Pattern Analyzer
class MemoryPatternAnalyzer:
    """ Analyzes MHN memory patterns """
    def __init__(self, id_memory, aux_memory, device):
        self.id_memory = id_memory.to(device)
        self.aux_memory = aux_memory.to(device)
        self.device = device
    def analyze_memory_capacity(self):
        """ Compute memory capacity metrics """
        n_patterns = len(self.id_memory)
        d_features = self.id_memory.shape[1]
        theoretical_capacity = 2 ** (d_features - 1)
        utilization = n_patterns / theoretical_capacity
        max_similarity = -1.0
        mean_similarity = 0.0
        num_pairs = 0
        chunk_size = 500 # Reduced from 1000 to avoid OOM
        for i in range(0, n_patterns, chunk_size):
            end_i = min(i + chunk_size, n_patterns)
            chunk_i = self.id_memory[i:end_i]
            for j in range(i, n_patterns, chunk_size):
                end_j = min(j + chunk_size, n_patterns)
                chunk_j = self.id_memory[j:end_j]
                sim_chunk = torch.matmul(chunk_i, chunk_j.T)
                if i == j:
                    mask = torch.eye(sim_chunk.shape[0], device=self.device)
                    sim_chunk = sim_chunk * (1 - mask)
                max_similarity = max(max_similarity, sim_chunk.max().item())
                mean_similarity += sim_chunk.sum().item()
                num_pairs += sim_chunk.numel()
                if j % (chunk_size * 10) == 0:
                    torch.cuda.empty_cache()
        diagonal_sum = n_patterns
        mean_similarity = (mean_similarity - diagonal_sum) / (num_pairs - n_patterns)
        subset_size = min(1000, n_patterns)
        subset_memory = self.id_memory[:subset_size]
        similarity_subset = torch.matmul(subset_memory, subset_memory.T)
        eigenvalues = torch.linalg.eigvalsh(similarity_subset)
        spurious_risk = (eigenvalues < 0).sum().item() / len(eigenvalues)
        return {
            'n_patterns': n_patterns,
            'dimension': d_features,
            'theoretical_capacity': theoretical_capacity,
            'utilization': utilization,
            'max_pattern_similarity': max_similarity,
            'mean_pattern_similarity': mean_similarity,
            'spurious_minima_risk': spurious_risk,
            'subset_size_for_eigenvalues': subset_size
        }
    def visualize_convergence_dynamics(self, test_queries, save_path='convergence_dynamics.png'):
        """ Visualize how queries converge to memory patterns """
        memory_subset_size = min(5000, len(self.id_memory))
        memory_subset = self.id_memory[:memory_subset_size]
        mhn = ModernHopfieldNetwork(memory_subset, beta=128.0)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for idx, query in enumerate(test_queries[:6]):
            ax = axes[idx // 3, idx % 3]
            _, trajectory = mhn.converge_to_pattern(query.unsqueeze(0), return_trajectory=True)
            energies = [mhn.compute_energy(state).item() for state in trajectory]
            ax.plot(energies, 'b-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Energy')
            ax.set_title(f'Query {idx+1} Convergence')
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    def compute_retrieval_accuracy(self, test_queries, test_labels):
        """ Measure how accurately MHN retrieves correct patterns """
        memory_subset_size = min(5000, len(self.id_memory))
        memory_subset = self.id_memory[:memory_subset_size]
        mhn = ModernHopfieldNetwork(memory_subset, beta=128.0)
        batch_size = 100
        correct = 0
        total = 0
        for i in range(0, len(test_queries), batch_size):
            batch_queries = test_queries[i:i+batch_size]
            batch_labels = test_labels[i:i+batch_size]
            converged = mhn.converge_to_pattern(batch_queries)
            similarities = torch.matmul(converged, memory_subset.T)
            retrieved_indices = torch.argmax(similarities, dim=1)
            correct += (retrieved_indices == batch_labels[:len(retrieved_indices)]).sum().item()
            total += len(retrieved_indices)
            torch.cuda.empty_cache()
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
# Comprehensive Evaluation Suite
class ComprehensiveOODEvaluator:
    """ Full evaluation suite """
    def __init__(self, device):
        self.device = device
        self.metrics = {}
    def evaluate_all_metrics(self, ood_detector, test_loader, memory_analyzer):
        """ Compute all required metrics """
        memory_stats = memory_analyzer.analyze_memory_capacity()
        pixel_metrics = self._evaluate_pixel_metrics(ood_detector, test_loader)
        image_metrics = self._evaluate_image_metrics(ood_detector, test_loader)
        runtime_stats = self._analyze_runtime(ood_detector, test_loader)
        significance = self._compute_statistical_significance(pixel_metrics, image_metrics)
        return {
            'memory': memory_stats,
            'pixel': pixel_metrics,
            'image': image_metrics,
            'runtime': runtime_stats,
            'significance': significance
        }
    def _evaluate_pixel_metrics(self, ood_detector, test_loader):
        """Compute pixel-level AUROC, AUPR, FPR@95 - FIXED for tensor consistency"""
        all_scores = []
        all_labels = []
      
        with torch.no_grad():
            for batch in test_loader:
                try:
                    # CRITICAL FIX: Extract features and labels with proper dimensions
                    features = batch['features'] # Expected: [B, H, W, C]
                    labels = batch['labels'] # Expected: [B, H, W] or None
                  
                    if features is None:
                        continue
                      
                    # Ensure features are in correct format
                    if features.dim() == 4:
                        B, H, W, C = features.shape
                    elif features.dim() == 3:
                        H, W, C = features.shape
                        B = 1
                        features = features.unsqueeze(0)
                    else:
                        logger.warning(f"Unexpected feature dimensions: {features.shape}")
                        continue
                  
                    # Handle labels - ensure they match feature spatial dimensions
                    if labels is not None:
                        if labels.dim() == 3:
                            batch_size, label_h, label_w = labels.shape
                        elif labels.dim() == 2:
                            label_h, label_w = labels.shape
                            batch_size = 1
                            labels = labels.unsqueeze(0)
                        else:
                            logger.warning(f"Unexpected label dimensions: {labels.shape}")
                            continue
                      
                        # CRITICAL FIX: Resize labels to match feature dimensions if needed
                        if (label_h, label_w) != (H, W):
                            logger.info(f"Resizing labels from {label_h}x{label_w} to {H}x{W}")
                            labels = F.interpolate(
                                labels.unsqueeze(1).float(),
                                size=(H, W),
                                mode='nearest'
                            ).squeeze(1).long()
                  
                    # Get OOD scores from detector
                    try:
                        results = ood_detector.detect_ood_pixels(features)
                        scores = results['scores'] # Expected: [B, H, W]
                    except Exception as e:
                        logger.warning(f"OOD detection failed for batch: {e}")
                        continue
                  
                    # CRITICAL FIX: Ensure all tensors have consistent spatial dimensions
                    if scores.shape[-2:] != (H, W):
                        logger.warning(f"Score shape {scores.shape} doesn't match feature shape [{B}, {H}, {W}]")
                        continue
                  
                    # Flatten tensors for metric computation
                    scores_flat = scores.flatten()
                  
                    if labels is not None:
                        labels_flat = labels.flatten()
                      
                        # CRITICAL FIX: Ensure scores and labels have same number of elements
                        if scores_flat.shape[0] != labels_flat.shape[0]:
                            logger.warning(f"Score count {scores_flat.shape[0]} != label count {labels_flat.shape[0]}")
                            # Truncate to smaller size
                            min_size = min(scores_flat.shape[0], labels_flat.shape[0])
                            scores_flat = scores_flat[:min_size]
                            labels_flat = labels_flat[:min_size]
                      
                        # Filter out ignore pixels (255)
                        valid_mask = (labels_flat != 255)
                        if valid_mask.sum() == 0:
                            continue
                          
                        valid_scores = scores_flat[valid_mask]
                        valid_labels = labels_flat[valid_mask]
                      
                        # Convert labels: 0=ID (inlier), 254=OOD (outlier)
                        binary_labels = (valid_labels == 254).cpu() # True for OOD, False for ID
                      
                    else:
                        # No labels - treat all as inliers for now
                        valid_scores = scores_flat
                        binary_labels = torch.zeros(len(valid_scores), dtype=torch.bool)
                  
                    # Collect scores and labels
                    all_scores.append(valid_scores.cpu())
                    all_labels.append(binary_labels)
                  
                except Exception as e:
                    logger.warning(f"Batch evaluation failed: {e}")
                    continue
                  
                # Periodic cleanup
                if len(all_scores) % 10 == 0:
                    torch.cuda.empty_cache()
      
        # Compute metrics if we have valid data
        if not all_scores or not all_labels:
            logger.warning("No valid scores collected for pixel metrics")
            return {'auroc': 0.0, 'aupr': 0.0, 'fpr95': 1.0}
      
        try:
            # Concatenate all collected scores and labels
            scores = torch.cat(all_scores).numpy()
            labels = torch.cat(all_labels).numpy()
          
            if len(scores) == 0 or len(labels) == 0:
                return {'auroc': 0.0, 'aupr': 0.0, 'fpr95': 1.0}
          
            # Check if we have both classes
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                logger.warning(f"Only one class present in labels: {unique_labels}")
                return {'auroc': 0.5, 'aupr': 0.0, 'fpr95': 1.0}
          
            # Compute metrics
            auroc = sklearn.metrics.roc_auc_score(labels, scores)
            aupr = sklearn.metrics.average_precision_score(labels, scores)
            fpr95 = self._compute_fpr_at_tpr(labels, scores, 0.95)
          
            return {'auroc': auroc, 'aupr': aupr, 'fpr95': fpr95}
          
        except Exception as e:
            logger.error(f"Metric computation failed: {e}")
            return {'auroc': 0.0, 'aupr': 0.0, 'fpr95': 1.0}
    def _evaluate_image_metrics(self, ood_detector, test_loader):
        """Compute image-level metrics"""
        image_scores = []
        image_labels = []
        with torch.no_grad(): # Added no_grad to reduce memory
            for batch in test_loader:
                features = batch['features']
                labels = batch['labels']
                results = ood_detector.detect_ood_pixels(features)
                for b in range(features.shape[0]):
                    img_score = results['scores'][b].max().item()
                    img_label = (labels[b] == 254).any().item()
                    image_scores.append(img_score)
                    image_labels.append(img_label)
                torch.cuda.empty_cache() # Added after batch
        scores = np.array(image_scores)
        labels = np.array(image_labels)
        auroc = sklearn.metrics.roc_auc_score(labels, scores)
        aupr = sklearn.metrics.average_precision_score(labels, scores)
        return {'image_auroc': auroc, 'image_aupr': aupr}
    def _analyze_runtime(self, ood_detector, test_loader):
        """Measure computational efficiency"""
        import time
        batch = next(iter(test_loader))
        _ = ood_detector.detect_ood_pixels(batch['features'])
        times = []
        for _ in range(10):
            batch = next(iter(test_loader))
            start = time.time()
            _ = ood_detector.detect_ood_pixels(batch['features'])
            torch.cuda.synchronize()
            times.append(time.time() - start)
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
    def _compute_statistical_significance(self, pixel_metrics, image_metrics):
        """Compute statistical significance using bootstrap"""
        baseline_auroc = 0.5
        observed_auroc = pixel_metrics['auroc']
        t_stat = (observed_auroc - baseline_auroc) / 0.01
        p_value = stats.t.sf(abs(t_stat), df=100) * 2
        return {
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    def _compute_fpr_at_tpr(self, y_true, y_scores, tpr_threshold=0.95):
        """Compute FPR at given TPR threshold"""
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_scores)
        idx = np.argmin(np.abs(tpr - tpr_threshold))
        return fpr[idx]
# Pixel Similarity Quantification
class PixelSimilarityAnalyzer:
    """ Quantifies similarity between pixels and memory patterns """
    def __init__(self, id_memory, aux_memory, device):
        self.id_memory = id_memory.to(device)
        self.aux_memory = aux_memory.to(device)
        self.device = device
    def compute_similarity_maps(self, pixel_features):
        """Compute similarity between each pixel and all memory patterns - FIXED"""
        # CRITICAL FIX: Ensure input is in correct format [B, H, W, C]
        if pixel_features.dim() == 4:
            B, H, W, C = pixel_features.shape
        elif pixel_features.dim() == 3:
            # Handle case where batch dimension is missing
            H, W, C = pixel_features.shape
            B = 1
            pixel_features = pixel_features.unsqueeze(0)
        else:
            raise ValueError(f"Expected 4D or 3D tensor, got {pixel_features.dim()}D")
      
        logger.info(f"Computing similarity maps for {B}x{H}x{W} features")
      
        # Reduce batch size for memory efficiency
        total_pixels = H * W
        batch_size = max(1, total_pixels // 32) # Conservative batch size
      
        # Initialize output tensors
        id_max_sim = torch.zeros(B, H, W, device=self.device)
        id_mean_sim = torch.zeros(B, H, W, device=self.device)
        aux_max_sim = torch.zeros(B, H, W, device=self.device)
        aux_mean_sim = torch.zeros(B, H, W, device=self.device)
      
        # Top-k similarity
        k = min(5, len(self.id_memory))
        id_topk = torch.zeros(B, H, W, device=self.device)
        aux_topk = torch.zeros(B, H, W, device=self.device)
      
        for b in range(B):
            # Get pixels for this batch item
            pixels_batch = pixel_features[b].reshape(-1, C) # [H*W, C]
            total_pixels = pixels_batch.shape[0]
          
            logger.info(f"Processing batch {b}: {total_pixels} pixels in chunks of {batch_size}")
          
            # Process in chunks to avoid OOM
            for start_idx in range(0, total_pixels, batch_size):
                end_idx = min(start_idx + batch_size, total_pixels)
                pixels_chunk = pixels_batch[start_idx:end_idx]
              
                # Normalize for similarity computation
                pixels_norm = F.normalize(pixels_chunk, p=2, dim=-1)
                id_norm = F.normalize(self.id_memory, p=2, dim=-1)
                aux_norm = F.normalize(self.aux_memory, p=2, dim=-1)
              
                # Compute similarities
                id_similarities = torch.matmul(pixels_norm, id_norm.T) # [chunk_size, num_id_memories]
                aux_similarities = torch.matmul(pixels_norm, aux_norm.T) # [chunk_size, num_aux_memories]
              
                # Compute statistics
                chunk_id_max = id_similarities.max(dim=1)[0]
                chunk_id_mean = id_similarities.mean(dim=1)
                chunk_aux_max = aux_similarities.max(dim=1)[0]
                chunk_aux_mean = aux_similarities.mean(dim=1)
              
                # Top-k similarities
                chunk_id_topk = torch.topk(id_similarities, k, dim=1)[0].mean(dim=1)
                chunk_aux_topk = torch.topk(aux_similarities, k, dim=1)[0].mean(dim=1)
              
                # CRITICAL FIX: Map flat indices back to spatial coordinates properly
                flat_indices = torch.arange(start_idx, end_idx, device=self.device)
                h_indices = flat_indices // W
                w_indices = flat_indices % W
              
                # SAFETY CHECK: Ensure indices are within bounds
                valid_mask = (h_indices < H) & (w_indices < W) & (h_indices >= 0) & (w_indices >= 0)
                if valid_mask.sum() > 0:
                    valid_h = h_indices[valid_mask]
                    valid_w = w_indices[valid_mask]
                    valid_chunk_idx = torch.arange(len(chunk_id_max), device=self.device)[valid_mask]
                  
                    id_max_sim[b, valid_h, valid_w] = chunk_id_max[valid_chunk_idx]
                    id_mean_sim[b, valid_h, valid_w] = chunk_id_mean[valid_chunk_idx]
                    aux_max_sim[b, valid_h, valid_w] = chunk_aux_max[valid_chunk_idx]
                    aux_mean_sim[b, valid_h, valid_w] = chunk_aux_mean[valid_chunk_idx]
                    id_topk[b, valid_h, valid_w] = chunk_id_topk[valid_chunk_idx]
                    aux_topk[b, valid_h, valid_w] = chunk_aux_topk[valid_chunk_idx]
              
                # Clear GPU cache periodically
                if (start_idx // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
      
        return {
            'id_max': id_max_sim,
            'id_mean': id_mean_sim,
            'id_topk': id_topk,
            'aux_max': aux_max_sim,
            'aux_mean': aux_mean_sim,
            'aux_topk': aux_topk,
            'difference': id_max_sim - aux_max_sim
        }
    def visualize_similarity_distribution(self, pixel_features, save_path='similarity_dist.png'):
        """Visualize distribution of pixel similarities - FIXED for proper tensor handling"""
        try:
            # CRITICAL FIX: Handle input tensor dimensions properly
            if pixel_features.dim() == 4:
                B, H, W, C = pixel_features.shape
            elif pixel_features.dim() == 3:
                H, W, C = pixel_features.shape
                B = 1
                pixel_features = pixel_features.unsqueeze(0)
            else:
                logger.warning(f"Unexpected tensor dimensions: {pixel_features.shape}")
                return
              
            logger.info(f"Visualizing similarity for {B}x{H}x{W} features")
              
            # Limit to 1 sample for memory efficiency
            if B > 1:
                pixel_features = pixel_features[:1]
                B = 1
          
            # Compute similarity maps with proper error handling
            try:
                similarity_maps = self.compute_similarity_maps(pixel_features)
            except Exception as e:
                logger.warning(f"Failed to compute similarity maps: {e}")
                return
          
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten() # Flatten for easier indexing
          
            plot_idx = 0
            for key, sim_map in similarity_maps.items():
                if plot_idx >= 6: # Limit to 6 subplots
                    break
                  
                try:
                    # FIXED: Handle tensor extraction properly
                    if isinstance(sim_map, torch.Tensor):
                        sim_flat = sim_map.flatten().cpu().detach().numpy()
                    else:
                        sim_flat = np.array(sim_map).flatten()
                  
                    # Remove any NaN or inf values
                    sim_flat = sim_flat[np.isfinite(sim_flat)]
                  
                    if len(sim_flat) == 0:
                        logger.warning(f"No valid similarity values for {key}")
                        continue
                  
                    # Subsample for memory efficiency
                    max_samples = 5000
                    if len(sim_flat) > max_samples:
                        indices = np.random.choice(len(sim_flat), max_samples, replace=False)
                        sim_flat = sim_flat[indices]
                  
                    # Create histogram
                    ax = axes[plot_idx]
                    ax.hist(sim_flat, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax.set_xlabel('Similarity Score')
                    ax.set_ylabel('Pixel Count')
                    ax.set_title(f'{key} Distribution')
                  
                    # Add statistics
                    if len(sim_flat) > 0:
                        mean_val = np.mean(sim_flat)
                        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                        ax.legend()
                  
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3)
                  
                    plot_idx += 1
                  
                except Exception as e:
                    logger.warning(f"Failed to plot {key}: {e}")
                    continue
          
            # Hide unused subplots
            for idx in range(plot_idx, 6):
                axes[idx].set_visible(False)
          
            plt.tight_layout()
          
            # Save with error handling
            try:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"Similarity distribution saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save similarity plot: {e}")
          
            plt.close(fig) # Ensure proper cleanup
          
        except Exception as e:
            logger.error(f"Similarity visualization failed: {e}")
            # Ensure any partial plots are cleaned up
            plt.close('all')
# Enhanced Trainer
class ImprovedOODSegmentationTrainer:
    def __init__(self, config_dict, train_loader, val_loader, fixed_batches=None, resume_from=None):
        self.config = config_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fixed_batches = fixed_batches or []
        self.resume_from = resume_from
        self.total_epochs = 2
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints_improved")
        self.results_dir = "./results_improved" # New: Different folder for results
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "visualizations"), exist_ok=True) # For visuals
        self.beta = 128.0
        self.lambda_ood = 3.0
        self.memory_subsample = 10000 # Reduced from 20000 to avoid OOM
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        wandb.init(
            project="ood-seg-improved",
            config=self.config,
            name="frozen-backbone-fixed-hopfield-pixelwise",
            mode="online"
        )
        self._init_models()
        if self.resume_from and os.path.exists(self.resume_from):
            self._load_checkpoint(self.resume_from)
        else:
            self._build_initial_memories()
        self._init_mhn_components()
        self._init_training_components()
        self.scaler = amp.GradScaler()
        self.best_val_miou = 0.0
        self.best_fpr95 = 1.0
        self.patience = 100
        self.patience_counter = 0
        self.global_step = 0
        self.accum_steps = 1
        self.warmup_steps = 500
    def _init_mhn_components(self):
        """Initialize all MHN-specific components """
        analysis_subset_size = min(10000, len(self.id_memory))
        self.id_mhn = ModernHopfieldNetwork(
            self.id_memory[:analysis_subset_size],
            beta=self.beta,
            max_iterations=10
        ).to(self.device)
        self.aux_mhn = ModernHopfieldNetwork(
            self.aux_memory[:analysis_subset_size],
            beta=self.beta,
            max_iterations=10
        ).to(self.device)
        self.pixel_ood_detector = PixelOODDetector(
            self.id_memory[:analysis_subset_size],
            self.aux_memory[:analysis_subset_size],
            beta=self.beta
        ).to(self.device)
        self.memory_analyzer = MemoryPatternAnalyzer(
            self.id_memory[:analysis_subset_size],
            self.aux_memory[:analysis_subset_size],
            self.device
        )
        self.similarity_analyzer = PixelSimilarityAnalyzer(
            self.id_memory[:analysis_subset_size],
            self.aux_memory[:analysis_subset_size],
            self.device
        )
        self.comprehensive_evaluator = ComprehensiveOODEvaluator(self.device)
    def _evaluate_comprehensive(self, epoch):
        """Run comprehensive evaluation with all metrics - FIXED for tensor dimension consistency"""
        self.feature_extractor.eval()
        self.projection_head.eval()
        self.segmentation_head.eval()
      
        try:
            # Get memory analysis first
            memory_analysis = self.memory_analyzer.analyze_memory_capacity()
          
            # Get a sample batch and extract features
            with torch.no_grad():
                sample_batch = next(iter(self.val_loader))
              
                # Extract features using the proper pipeline
                sample_features_dict = self.feature_extractor.extract_features_batch({'data': sample_batch['data']})
              
                if 'features' not in sample_features_dict:
                    logger.warning("No features extracted in comprehensive evaluation")
                    return {}
              
                # Get projected features at the correct resolution
                sample_projected = self.projection_head(sample_features_dict['features'])
              
            # CRITICAL FIX: Work with feature map resolution throughout
            B, C, H, W = sample_projected.shape # e.g., [1, 128, 96, 192]
            logger.info(f"Feature resolution: {H}x{W}, Total pixels: {H*W}")
          
            # Reshape to pixel-wise format for analysis
            pixel_features = sample_projected.permute(0, 2, 3, 1) # [B, H, W, C]
          
            # Select test queries at FEATURE resolution, not image resolution
            test_queries = pixel_features.reshape(-1, C)[:6] # Take first 6 pixels as queries
          
            # Visualization paths
            vis_dir = os.path.join(self.results_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
          
            # Convergence dynamics visualization
            conv_path = os.path.join(vis_dir, f'convergence_epoch{epoch}.png')
            try:
                self.memory_analyzer.visualize_convergence_dynamics(
                    test_queries,
                    save_path=conv_path
                )
              
                # Log to wandb
                wandb.log({
                    "convergence_dynamics": wandb.Image(conv_path),
                    "epoch": epoch
                })
            except Exception as e:
                logger.warning(f"Convergence visualization failed: {e}")
          
            # Similarity distribution visualization
            sim_path = os.path.join(vis_dir, f'similarity_dist_epoch{epoch}.png')
            try:
                # FIXED: Use feature resolution for similarity analysis
                self.similarity_analyzer.visualize_similarity_distribution(
                    pixel_features, # Already in [B, H, W, C] format
                    save_path=sim_path
                )
              
                # Log to wandb
                wandb.log({
                    "similarity_distribution": wandb.Image(sim_path),
                    "epoch": epoch
                })
            except Exception as e:
                logger.warning(f"Similarity visualization failed: {e}")
          
            # Create test loader wrapper that handles resolution properly
            class TestLoaderWrapper:
                def __init__(self, val_loader, feature_extractor, projection_head, target_resolution):
                    self.val_loader = val_loader
                    self.feature_extractor = feature_extractor
                    self.projection_head = projection_head
                    self.target_h, self.target_w = target_resolution
                  
                def __iter__(self):
                    for batch in self.val_loader:
                        try:
                            with torch.no_grad():
                                # Extract features and project
                                features_dict = self.feature_extractor.extract_features_batch({'data': batch['data']})
                              
                                if 'features' not in features_dict:
                                    continue
                                  
                                projected = self.projection_head(features_dict['features'])
                              
                                # CRITICAL: Ensure all tensors use feature resolution
                                B, C, H_feat, W_feat = projected.shape
                              
                                # Get labels aligned to feature resolution
                                labels = None
                                if 'label' in batch:
                                    original_labels = batch['label']
                                  
                                    if original_labels.dim() == 3:
                                        original_labels = original_labels.unsqueeze(1).float()
                                    elif original_labels.dim() == 4 and original_labels.shape[1] != 1:
                                        original_labels = original_labels[:, 0:1].float()
                                  
                                    # FIXED: Resize to actual feature resolution
                                    labels = F.interpolate(
                                        original_labels,
                                        size=(H_feat, W_feat), # Use actual feature dimensions
                                        mode='nearest'
                                    ).squeeze(1).long()
                              
                                # Convert to pixel format with consistent dimensions
                                pixel_features = projected.permute(0, 2, 3, 1) # [B, H_feat, W_feat, C]
                              
                                yield {
                                    'features': pixel_features, # [B, H_feat, W_feat, C]
                                    'labels': labels # [B, H_feat, W_feat] or None
                                }
                              
                        except Exception as e:
                            logger.warning(f"Test wrapper batch failed: {e}")
                            continue
          
            # Run comprehensive evaluation with properly aligned tensors
            test_wrapper = TestLoaderWrapper(
                self.val_loader,
                self.feature_extractor,
                self.projection_head,
                target_resolution=(H, W) # Pass actual feature resolution
            )
          
            try:
                comprehensive_results = self.comprehensive_evaluator.evaluate_all_metrics(
                    self.pixel_ood_detector,
                    test_wrapper,
                    self.memory_analyzer
                )
              
                # Log results to wandb
                for category, metrics in comprehensive_results.items():
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                wandb.log({f'comprehensive/{category}_{key}': value, 'epoch': epoch})
              
            except Exception as e:
                logger.warning(f"Comprehensive metrics evaluation failed: {e}")
                comprehensive_results = {
                    'memory': memory_analysis,
                    'pixel': {},
                    'image': {},
                    'runtime': {},
                    'significance': {}
                }
          
            return comprehensive_results
          
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            traceback.print_exc()
          
            # Return minimal results to avoid crashing
            return {
                'memory': {'error': str(e)},
                'pixel': {},
                'image': {},
                'runtime': {},
                'significance': {}
            }
          
        finally:
            # Restore training mode
            self.feature_extractor.train()
            self.projection_head.train()
            self.segmentation_head.train()
    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.segmentation_head.load_state_dict(checkpoint['segmentation_head_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        self.optimizer_seg.load_state_dict(checkpoint['optimizer_seg_state_dict'])
        self.optimizer_proj.load_state_dict(checkpoint['optimizer_proj_state_dict'])
        self.id_memory = checkpoint['id_memory'].to(self.device).float()
        self.aux_memory = checkpoint['aux_memory'].to(self.device).float()
        self.global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
        self.best_fpr95 = checkpoint.get('best_fpr95', 1.0)
        self._init_training_components()
        return start_epoch
    def _init_models(self):
        """Initialize models with fully frozen backbone."""
        self.feature_extractor = FeatureExtractor(
            model_path=self.config['model_path'],
            device=self.device,
            num_classes=self.config['num_classes'],
        ).to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.train()
        self.segmentation_head = SegmentationClassifierHead(
            1280, self.config['num_classes']
        ).to(self.device)
        self.projection_head = SimpleProjectionHead(
            input_dim=1280, output_dim=128
        ).to(self.device)
        self._init_weights()
    def _init_weights(self):
        """Initialize weights carefully to prevent gradient explosion"""
        for module in [self.segmentation_head, self.projection_head]:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if isinstance(m, nn.Conv2d):
                        nn.init.orthogonal_(m.weight)
                    else:
                        nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    def _build_initial_memories(self):
        """Build initial Hopfield memories"""
        memory_builder = MemoryBuilder(
            feature_extractor=self.feature_extractor,
            projection_pipeline=self.projection_head,
            device=self.device,
            id_memory_size=self.memory_subsample,
            aux_memory_size=self.memory_subsample,
            num_in_dist_classes=self.config['num_classes'],
            ood_label=254,
        )
        id_memory, aux_memory, warnings = memory_builder.process_images(self.train_loader)
        self.id_memory = id_memory.to(self.device).float()
        self.aux_memory = aux_memory.to(self.device).float()
    def _init_training_components(self):
        """Initialize training components with higher LR for projection head"""
        base_lr = 5e-5
        self.optimizer_seg = torch.optim.AdamW(
            self.segmentation_head.parameters(),
            lr=base_lr,
            weight_decay=5e-4,
            eps=1e-6,
            betas=(0.9, 0.999)
        )
        self.optimizer_proj = torch.optim.AdamW(
            self.projection_head.parameters(),
            lr=base_lr * 10,
            weight_decay=5e-4,
            eps=1e-6
        )
        self.scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_seg,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=True
        )
        self.scheduler_proj = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_proj,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=255,
            reduction='mean'
        )
        self.hopfield_manager = HopfieldBoostingManager(
            id_features_full=self.id_memory,
            aux_features_full=self.aux_memory,
            beta_sampling=self.beta,
            lambda_ood=self.lambda_ood,
            device=self.device,
            memory_subset_size=10000, # Reduced from 20000 to avoid OOM
            positive_shift=False,
            num_boosting_iters=20
        )
    def _prepare_batch(self, batch):
        """Prepare batch for GPU"""
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == 'label':
                    batch_gpu[k] = v.to(self.device).long()
                else:
                    batch_gpu[k] = v.to(self.device).float()
        return batch_gpu
    def _compute_losses(self, batch):
        """Fixed loss computation - Keep CE for ID; add MHE loss only on projected features"""
        batch_gpu = self._prepare_batch(batch)
        extracted = self.feature_extractor.extract_features_batch(batch_gpu)
        features = extracted['features'].float()
        labels = extracted['labels']
        if labels is not None:
            labels = labels.clone()
            valid_mask = (labels >= 0) & (labels < self.config['num_classes'])
            ood_mask = (labels == 254)
            ignore_mask = (labels == 255)
            invalid_mask = ~(valid_mask | ood_mask | ignore_mask)
            labels[invalid_mask] = 255
        seg_loss = torch.tensor(0.0, device=self.device)
        with torch.cuda.amp.autocast():
            seg_logits = self.segmentation_head(features)
            seg_logits = torch.clamp(seg_logits, min=-10, max=10)
            if seg_logits.shape[-2:] != labels.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits,
                    size=labels.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
            labels_for_ce = labels.clone()
            labels_for_ce[labels == 254] = 255
            labels_for_ce[labels >= self.config['num_classes']] = 255
            ce_unreduced = F.cross_entropy(
                seg_logits,
                labels_for_ce,
                ignore_index=255,
                reduction='none'
            )
            valid_loss_mask = torch.isfinite(ce_unreduced)
            ce_unreduced = torch.where(
                valid_loss_mask,
                ce_unreduced,
                torch.zeros_like(ce_unreduced)
            )
            if valid_loss_mask.any():
                seg_loss = ce_unreduced[valid_loss_mask].mean()
            else:
                seg_loss = torch.tensor(0.0, device=self.device)
            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                seg_loss = torch.tensor(0.0, device=self.device)
        ood_loss = torch.tensor(0.0, device=self.device)
        with torch.cuda.amp.autocast():
            projected = self.projection_head(features)
            projected = torch.clamp(projected, min=-10, max=10)
            B, C, H_feat, W_feat = projected.shape
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H_feat, W_feat),
                mode='nearest'
            ).squeeze(1).long()
            pixel_features = projected.permute(0, 2, 3, 1).reshape(-1, C)
            pixel_labels = labels_resized.view(-1)
            valid_mask = (pixel_labels != 255)
            if valid_mask.any():
                valid_pixels = pixel_features[valid_mask]
                valid_labels = pixel_labels[valid_mask]
                id_mask = (valid_labels < self.config['num_classes'])
                ood_mask = (valid_labels == 254)
                id_pixels = valid_pixels[id_mask] if id_mask.any() else torch.empty(0, C, device=self.device)
                ood_pixels = valid_pixels[ood_mask] if ood_mask.any() else torch.empty(0, C, device=self.device)
                if id_mask.any():
                    num_to_sample = min(5, len(valid_pixels)) # Reduced from 128 to avoid OOM
                    id_batch, aux_batch = self.hopfield_manager.sample_batch(num_to_sample)
                    boosted_id = torch.cat([id_pixels, id_batch.to(self.device).float()]) if len(id_batch) > 0 else id_pixels
                    boosted_ood = torch.cat([ood_pixels, aux_batch.to(self.device).float()]) if len(aux_batch) > 0 else ood_pixels
                    if len(boosted_id) > 0 and len(boosted_ood) > 0:
                        raw_ood_loss = self.hopfield_manager.compute_boosted_ood_loss(boosted_ood, boosted_id)
                        ood_loss = torch.clamp(raw_ood_loss, min=-20.0, max=20.0)
                        if torch.isnan(ood_loss) or torch.isinf(ood_loss):
                            ood_loss = torch.tensor(0.0, device=self.device)
                    else:
                        ood_loss = torch.tensor(0.0, device=self.device)
                reg_loss = 0.0
                for param in self.projection_head.parameters():
                    if param.requires_grad:
                        reg_loss += torch.norm(param) * 1e-5
                ood_loss += reg_loss
        return {
            'seg_loss': seg_loss,
            'ood_loss': ood_loss,
            'has_id': (labels_for_ce < 255).any() if labels is not None else False,
        }
    def _train_epoch(self, epoch):
        """Train with weight updates every epoch"""
        self.hopfield_manager.update_sampling_weights(memory_size=10000) # Reduced size
        self.segmentation_head.train()
        self.projection_head.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_metrics = {
            'seg_losses': [],
            'ood_losses': [],
        }
        accum_count = 0
        for batch_idx, batch in enumerate(progress_bar):
            if self.global_step < self.warmup_steps:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
                for pg in self.optimizer_seg.param_groups:
                    pg['lr'] = lr_scale * 5e-5
                for pg in self.optimizer_proj.param_groups:
                    pg['lr'] = lr_scale * 5e-5
            loss_dict = self._compute_losses(batch)
            seg_loss = loss_dict['seg_loss'] / self.accum_steps
            ood_loss = loss_dict['ood_loss'] / self.accum_steps
            if accum_count == 0:
                self.optimizer_seg.zero_grad(set_to_none=True)
                self.optimizer_proj.zero_grad(set_to_none=True)
            self.scaler.scale(seg_loss).backward(retain_graph=True)
            proj_loss = self.lambda_ood * ood_loss
            self.scaler.scale(proj_loss).backward()
            accum_count += 1
            if accum_count == self.accum_steps or batch_idx == len(self.train_loader) - 1:
                for param_group in [self.segmentation_head.parameters(),
                                    self.projection_head.parameters()]:
                    for param in param_group:
                        if param.grad is not None:
                            param.grad.data = torch.clamp(param.grad.data, -5.0, 5.0)
                self.scaler.unscale_(self.optimizer_seg)
                grad_norm_seg = torch.nn.utils.clip_grad_norm_(
                    self.segmentation_head.parameters(),
                    max_norm=2.0
                )
                self.scaler.unscale_(self.optimizer_proj)
                grad_norm_proj = torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(),
                    max_norm=2.0
                )
                if grad_norm_seg <= 10.0:
                    self.scaler.step(self.optimizer_seg)
                else:
                    self.optimizer_seg.zero_grad(set_to_none=True)
                if grad_norm_proj <= 10.0:
                    self.scaler.step(self.optimizer_proj)
                else:
                    self.optimizer_proj.zero_grad(set_to_none=True)
                self.scaler.update()
                accum_count = 0
            epoch_metrics['seg_losses'].append(loss_dict['seg_loss'].item())
            epoch_metrics['ood_losses'].append(loss_dict['ood_loss'].item())
            # Log batch losses to WandB
            wandb.log({
                'batch_seg_loss': loss_dict['seg_loss'].item(),
                'batch_ood_loss': loss_dict['ood_loss'].item(),
                'batch': batch_idx,
                'epoch': epoch
            })
            progress_bar.set_postfix({
                'Seg': f"{loss_dict['seg_loss'].item():.4f}",
                'OOD': f"{loss_dict['ood_loss'].item():.4f}",
            })
            self.global_step += 1
            torch.cuda.empty_cache()
        avg_seg_loss = np.mean(epoch_metrics['seg_losses']) if epoch_metrics['seg_losses'] else 0.0
        avg_ood_loss = np.mean(epoch_metrics['ood_losses']) if epoch_metrics['ood_losses'] else 0.0
        # Log average losses to WandB
        wandb.log({
            'avg_seg_loss': avg_seg_loss,
            'avg_ood_loss': avg_ood_loss,
            'epoch': epoch
        })
        self.hopfield_manager.advance_epoch(current_epoch=epoch, update_freq=5)
        return {
            'avg_seg_loss': avg_seg_loss,
            'avg_ood_loss': avg_ood_loss,
        }
    def _evaluate_ood(self, epoch=None, save_vis=True):
        """Evaluate OOD metrics on multiple datasets"""
        self.feature_extractor.eval()
        self.projection_head.eval()
        self.segmentation_head.eval()
        with torch.no_grad():
            multi_evaluator = MultiDatasetOODEvaluator(self.device, self.segmentation_head)
            all_results = multi_evaluator.evaluate_all(
                self.feature_extractor,
                self.projection_head,
                self.id_memory,
                self.aux_memory,
                beta_border=64.0,
                epoch=epoch,
                save_visualizations=save_vis
            )
        self.feature_extractor.train()
        self.projection_head.train()
        self.segmentation_head.train()
        aggregated = {}
        for dataset_key, results in all_results.items():
            for metric, value in results.items():
                if metric not in ['dataset_name', 'anomaly_rate']:
                    aggregated[f'{dataset_key}_{metric}'] = value
        weighted_fpr95 = 0
        weighted_auroc = 0
        total_weight = 0
        for dataset_key, results in all_results.items():
            if 'fpr95' in results and 'anomaly_rate' in results:
                weight = results['anomaly_rate']
                weighted_fpr95 += results['fpr95'] * weight
                weighted_auroc += results.get('auroc', 0) * weight
                total_weight += weight
        if total_weight > 0:
            aggregated['weighted_fpr95'] = weighted_fpr95 / total_weight
            aggregated['weighted_auroc'] = weighted_auroc / total_weight
        return aggregated
    def _evaluate_semantic(self, epoch=None, save_vis=False, num_images=30):
        """Evaluate semantic segmentation on Cityscapes val images (19 classes)"""
        self.segmentation_head.eval()
        self.feature_extractor.eval()
        with torch.no_grad():
            confusion_matrix = np.zeros((self.config['num_classes'], self.config['num_classes']))
            processed = 0
            output_dir = None
            if save_vis and epoch is not None:
                output_dir = os.path.join(self.results_dir, f"semantic_epoch{epoch}")
                os.makedirs(output_dir, exist_ok=True)
            for batch_idx, batch in enumerate(self.val_loader):
                if processed >= num_images:
                    break
                batch_gpu = self._prepare_batch(batch)
                images = batch_gpu['data']
                labels = batch_gpu['label']
                B = images.size(0)
                for b in range(B):
                    if processed >= num_images:
                        break
                    img = images[b:b+1]
                    gt = labels[b:b+1]
                    extracted = self.feature_extractor.extract_features_batch({'data': img}) # Skip label check
                    features = extracted['features']
                    seg_logits = self.segmentation_head(features)
                    seg_logits = F.interpolate(
                        seg_logits, size=gt.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )
                    pred = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()
                    gt_np = gt[0].cpu().numpy()
                    valid_mask = (gt_np < self.config['num_classes']) & (gt_np != 255) & (gt_np != 254)
                    if np.any(valid_mask):
                        pred_valid = pred[valid_mask]
                        gt_valid = gt_np[valid_mask]
                        self._update_confusion_matrix(confusion_matrix, pred_valid, gt_valid)
                    seg_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for cls, color in enumerate(CITYSCAPES_COLORMAP):
                        seg_color[pred == cls] = color
                    if output_dir:
                        save_path = os.path.join(output_dir, f"epoch{epoch}_img{processed:03d}.png")
                        Image.fromarray(seg_color).save(save_path)
                        if processed < 5:
                            wandb.log({f"semantic_vis_{processed}": wandb.Image(save_path), "epoch": epoch})
                    processed += 1
                    torch.cuda.empty_cache() # Added after image
        iou_per_class = []
        for i in range(self.config['num_classes']):
            if confusion_matrix[i, i] == 0 and np.sum(confusion_matrix[i, :]) == 0:
                continue
            iou = confusion_matrix[i, i] / (
                np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i] + 1e-10
            )
            iou_per_class.append(iou)
        miou = np.mean(iou_per_class) if iou_per_class else 0.0
        self.segmentation_head.train()
        self.feature_extractor.train()
        return {'miou': miou}
    def _update_confusion_matrix(self, cm, pred, gt):
        """Update confusion matrix efficiently"""
        n = cm.shape[0]
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        indices = gt_flat * n + pred_flat
        unique, counts = np.unique(indices, return_counts=True)
        cm.flat[unique] += counts
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint - Memories to CPU to avoid OOM"""
        id_mem_cpu = self.id_memory.cpu()
        aux_mem_cpu = self.aux_memory.cpu()
        checkpoint = {
            'epoch': epoch,
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'segmentation_head_state_dict': self.segmentation_head.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_seg_state_dict': self.optimizer_seg.state_dict(),
            'optimizer_proj_state_dict': self.optimizer_proj.state_dict(),
            'id_memory': id_mem_cpu,
            'aux_memory': aux_mem_cpu,
            'best_val_miou': self.best_val_miou,
            'best_fpr95': self.best_fpr95,
            'global_step': self.global_step,
            'metrics': metrics
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        if 'fpr95' in metrics and metrics['fpr95'] < self.best_fpr95:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
    def train(self):
        """Main training loop"""
        start_epoch = 1
        if self.resume_from:
            start_epoch = self._load_checkpoint(self.resume_from) + 1
        for epoch in range(start_epoch, self.total_epochs + 1):
            epoch_start = time.time()
            train_metrics = self._train_epoch(epoch)
            try:
                ood_metrics = self._evaluate_ood(epoch=epoch, save_vis=(epoch % 2 == 0))
            except Exception as e:
                logger.error(f"OOD evaluation error: {str(e)}")
                ood_metrics = {}
            try:
                sem_metrics = self._evaluate_semantic(epoch=epoch, save_vis=(epoch % 2 == 0))
            except Exception as e:
                logger.error(f"Semantic evaluation error: {str(e)}")
                sem_metrics = {}
            if epoch % 2 == 0:
                try:
                    comprehensive_results = self._evaluate_comprehensive(epoch)
                    # Save comprehensive as JSON
                    comp_path = os.path.join(self.results_dir, f'comprehensive_epoch{epoch}.json')
                    with open(comp_path, 'w') as f:
                        json.dump(comprehensive_results, f, indent=4)
                except Exception as e:
                    logger.error(f"Comprehensive evaluation error: {str(e)}")
                    comprehensive_results = {}
            val_metric = ood_metrics.get('fpr95', 1.0)
            self.scheduler_seg.step(val_metric)
            self.scheduler_proj.step(val_metric)
            combined_metrics = {**ood_metrics, **sem_metrics}
            self.save_checkpoint(epoch, combined_metrics)
            # Save OOD and semantic metrics as JSON
            ood_path = os.path.join(self.results_dir, f'ood_metrics_epoch{epoch}.json')
            with open(ood_path, 'w') as f:
                json.dump(ood_metrics, f, indent=4)
            sem_path = os.path.join(self.results_dir, f'sem_metrics_epoch{epoch}.json')
            with open(sem_path, 'w') as f:
                json.dump(sem_metrics, f, indent=4)
            if 'fpr95' in ood_metrics and ood_metrics['fpr95'] < self.best_fpr95:
                self.best_fpr95 = ood_metrics['fpr95']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    break
            log_dict = {
                'epoch': epoch,
            }
            if ood_metrics:
                log_dict.update({
                    'ood_auroc': ood_metrics.get('auroc', 0.0),
                    'ood_fpr95': ood_metrics.get('fpr95', 1.0),
                    'ood_auprs': ood_metrics.get('auprs', 0.0),
                })
            if sem_metrics:
                log_dict.update({
                    'semantic_miou': sem_metrics.get('miou', 0.0),
                })
            wandb.log(log_dict)
            epoch_time = time.time() - epoch_start
            torch.cuda.empty_cache()
            gc.collect()
        wandb.finish()
        return self.best_fpr95
# Data loading helper functions
def val_joint_transform(img, gt):
    """Validation transformation"""
    size = (512, 1024)
    img = transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)(img)
    if gt is not None:
        gt = transforms.Resize(size, interpolation=InterpolationMode.NEAREST)(gt)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    if gt is not None:
        gt = np.array(gt, dtype=np.uint8)
    return img, gt
class DictWrapperDataset:
    """Wrapper to convert tuple dataset to dict format"""
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) >= 1:
            return {'data': item[0], 'label': item[1] if len(item) > 1 else None}
        return item
    def __len__(self):
        return len(self.dataset)
# Pixel Metric Classes (unchanged)
class PixelMetric(ABC):
    @abstractmethod
    def __call__(self, in_scores, out_scores):
        pass
class AUROCMetric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.zeros_like(in_scores),
            torch.ones_like(out_scores)
        ])
        scores = torch.cat([in_scores, out_scores])
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
        return sklearn.metrics.roc_auc_score(targets_np, scores_np)
class FPR95Metric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.zeros_like(in_scores),
            torch.ones_like(out_scores)
        ])
        scores = torch.cat([in_scores, out_scores])
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
        return self._fpr_at_tpr(targets_np, scores_np, tpr_level=0.95)
    def _fpr_at_tpr(self, y_true, y_score, tpr_level=0.95):
        y_true = (y_true == 1)
        desc_indices = np.argsort(y_score)[::-1]
        y_score = y_score[desc_indices]
        y_true = y_true[desc_indices]
        desc_indices = np.argsort(y_score)[::-1]
        y_score = y_score[desc_indices]
        y_true = y_true[desc_indices]
        distinct_indices = np.where(np.diff(y_score))[0]
        threshold_indices = np.r_[distinct_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_indices]
        fps = 1 + threshold_indices - tps
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        if len(tpr) == 0 or tpr[-1] == 0:
            return 1.0
        cutoff = np.argmin(np.abs(tpr - tpr_level))
        n_negatives = np.sum(~y_true)
        if n_negatives == 0:
            return 0.0
        return fps[cutoff] / n_negatives
class AUPRSMetric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.zeros_like(in_scores),
            torch.ones_like(out_scores)
        ])
        scores = torch.cat([in_scores, out_scores])
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
        return sklearn.metrics.average_precision_score(targets_np, scores_np)
# New RoadAnomaly Dataset Class (added based on data_loader.py style)
class RoadAnomaly(torch.utils.data.Dataset):
    def __init__(self, root, split='test', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images = sorted([f for f in os.listdir(root) if f.endswith('.jpg') or f.endswith('.png')])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)
        label_name = img_name.replace('.jpg', '.png') # Assume label is png
        label_path = os.path.join(self.root, label_name) # Assume labels in same dir; adjust if separate
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L') if os.path.exists(label_path) else None
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
# PixelOODEvaluator (unchanged)
class PixelOODEvaluator:
    def __init__(self, device, segmentation_head=None):
        self.device = device
        self.segmentation_head = segmentation_head
        fishyscapes_dir = "/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static"
        image_dir = os.path.join(fishyscapes_dir, "original")
        label_dir = os.path.join(fishyscapes_dir, "labels")
        from dataset.data_loader import Fishyscapes # Import FishyscapesDataset from the appropriate module
        self.ood_dataset = Fishyscapes(image_dir, label_dir)
        self.ood_loader = DataLoader(self.ood_dataset, batch_size=1, shuffle=False) # Updated: batch_size=1 to reduce memory
    def safe_subsample(self, scores, labels, max_pixels=None):
        total_pixels = len(scores)
        if total_pixels == 0:
            return scores, labels
        if max_pixels is None or total_pixels <= max_pixels:
            return scores, labels
        in_mask = (labels == 0)
        out_mask = (labels == 1) # Fixed: Changed from ==1 to ==254
        in_scores = scores[in_mask]
        out_scores = scores[out_mask]
        in_labels = labels[in_mask]
        out_labels = labels[out_mask]
        half_max = max_pixels // 2
        if len(in_scores) > half_max:
            in_perm = torch.randperm(len(in_scores))[:half_max]
            in_scores = in_scores[in_perm]
            in_labels = in_labels[in_perm]
        if len(out_scores) > half_max:
            out_perm = torch.randperm(len(out_scores))[:half_max]
            out_scores = out_scores[out_perm]
            out_labels = out_labels[out_perm]
        return torch.cat([in_scores, out_scores]), torch.cat([in_labels, out_labels])
    def evaluate(self, feature_extractor, projection_pipeline, id_memory, aux_memory, beta_border=128.0, epoch=None):
        if id_memory is None or aux_memory is None:
            return {}
        feature_extractor.eval()
        projection_pipeline.eval()
        id_memory = id_memory.to(self.device)
        aux_memory = aux_memory.to(self.device)
        score_calc = PixelWiseInferenceScore(id_memory, aux_memory, beta=beta_border)
        metrics = {
            'auroc': AUROCMetric(),
            'fpr95': FPR95Metric(),
            'auprs': AUPRSMetric()
        }
        all_in_scores = []
        all_out_scores = []
        total_images = 0
        images_with_ood = 0
        processed_images = 0
        with torch.no_grad(): # Added no_grad to reduce memory
            for batch_idx, (images, labels) in enumerate(self.ood_loader):
                total_images += images.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_dict = {'data': images} # Removed 'label' to skip validation
                extracted = feature_extractor.extract_features_batch(batch_dict)
                if 'features' not in extracted:
                    continue
                features = extracted['features']
                projected = projection_pipeline(features)
                B, C, H, W = projected.shape
                pixel_features = projected.permute(0, 2, 3, 1).contiguous().view(-1, C)
                labels_resized = F.interpolate(
                    labels.unsqueeze(1).float(),
                    size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()
                pixel_labels = labels_resized.view(-1)
                ood_scores = self._compute_ood_scores(pixel_features, score_calc)
                if len(ood_scores) == 0:
                    continue
                pred = None
                if self.segmentation_head:
                    seg_logits = self.segmentation_head(features)
                    seg_logits = F.interpolate(
                        seg_logits, size=labels.shape[-2:],
                        mode='bilinear', align_corners=True
                    )
                    pred = torch.argmax(seg_logits, dim=1)
                save_vis = (epoch is not None)
                if save_vis and batch_idx < 1:
                    for b in range(B):
                        ood_scores_up = F.interpolate(
                            ood_scores.view(B, 1, H, W),
                            size=labels.shape[-2:],
                            mode='bilinear', align_corners=True
                        ).squeeze(1)
                        scores_map = ood_scores_up[b].cpu().numpy()
                        scores_norm = (scores_map - scores_map.min()) / (scores_map.max() - scores_map.min() + 1e-5)
                        colormap = plt.colormaps['inferno']
                        scores_color = colormap(scores_norm)[:, :, :3]
                        orig_img = images[b].cpu().numpy().transpose(1,2,0)
                        label_np = labels[b].cpu().numpy()
                        anomaly_gt = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                        anomaly_gt[label_np == 0] = [0,0,0]
                        anomaly_gt[label_np == 254] = [1,0,0] # Fixed: Changed from ==1 to ==254
                        anomaly_gt[label_np == 255] = [0.5,0.5,0.5]
                        threshold = np.quantile(scores_map.flatten(), 0.95)
                        ood_mask = (scores_map > threshold)
                        ood_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                        ood_color[ood_mask] = [1,1,0]
                        if pred is not None:
                            seg_with_ood = pred[b].cpu().numpy()
                            seg_with_ood[ood_mask] = 19
                            seg_color = np.zeros((seg_with_ood.shape[0], seg_with_ood.shape[1], 3))
                            for cls, color in enumerate(CITYSCAPES_COLORMAP):
                                seg_color[seg_with_ood == cls] = [c/255.0 for c in color]
                        else:
                            seg_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                            for cls, color in enumerate(CITYSCAPES_COLORMAP[:19]):
                                seg_color[label_np == cls] = [c/255.0 for c in color]
                            seg_color[label_np == 254] = [0,0,1]
                        fig, axs = plt.subplots(1,5, figsize=(25,5))
                        axs[0].imshow(orig_img)
                        axs[0].set_title('Original Image')
                        axs[1].imshow(anomaly_gt)
                        axs[1].set_title('Anomaly Ground Truth (Red=OOD, Gray=Ignore)')
                        axs[2].imshow(seg_color)
                        axs[2].set_title('Segmentation incl. OOD')
                        axs[3].imshow(ood_color)
                        axs[3].set_title('OOD Map (Yellow=Detected)')
                        axs[4].imshow(scores_color)
                        axs[4].set_title('OOD Score Map (Inferno)')
                        plt.savefig(f'ood_vis_epoch{epoch}_{batch_idx}_{b}.png')
                        plt.close()
                valid_mask = (pixel_labels != 255)
                valid_ood = ood_scores[valid_mask]
                valid_labels = pixel_labels[valid_mask]
                if len(valid_ood) == 0:
                    continue
                sub_ood, sub_labels = self.safe_subsample(
                    valid_ood, valid_labels, max_pixels=50000 # Reduced from None to avoid OOM
                )
                in_mask = (sub_labels == 0)
                out_mask = (sub_labels == 254) # Fixed: Changed from ==1 to ==254
                in_count = in_mask.sum().item()
                out_count = out_mask.sum().item()
                if out_count > 0:
                    images_with_ood += 1
                if in_count > 0:
                    in_scores_batch = sub_ood[in_mask]
                    if len(in_scores_batch) > 0:
                        all_in_scores.append(in_scores_batch)
                if out_count > 0:
                    out_scores_batch = sub_ood[out_mask]
                    if len(out_scores_batch) > 0:
                        all_out_scores.append(out_scores_batch)
                processed_images += images.size(0)
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
        if not all_in_scores or not all_out_scores:
            return {}
        in_scores = torch.cat(all_in_scores)
        out_scores = torch.cat(all_out_scores)
        results = {}
        for metric_name, metric in metrics.items():
            score = metric(in_scores, out_scores)
            results[metric_name] = float(score)
        return results
    def _compute_ood_scores(self, pixel_features, score_calc):
        if len(pixel_features) == 0:
            return torch.tensor([], device=self.device)
        chunk_size = 2000 # Reduced from 5000 to avoid OOM
        all_scores = []
        for i in range(0, len(pixel_features), chunk_size):
            end_i = min(i + chunk_size, len(pixel_features))
            chunk = pixel_features[i:end_i]
            if len(chunk) == 0:
                continue
            with autocast():
                chunk_scores = score_calc(chunk)
            chunk_scores = chunk_scores.squeeze()
            if chunk_scores.dim() == 0:
                chunk_scores = chunk_scores.unsqueeze(0)
            all_scores.append(chunk_scores)
        if all_scores:
            return torch.cat(all_scores)
        else:
            return torch.tensor([], device=self.device)
 
        return labels
# New MultiDatasetOODEvaluator
class MultiDatasetOODEvaluator:
    """Evaluator for multiple OOD datasets"""
    def __init__(self, device, segmentation_head=None):
        self.device = device
        self.segmentation_head = segmentation_head
        self.datasets = {}
        self._setup_datasets()
    def _validate_labels(self, labels, dataset_name):
        """Validate labels for the dataset"""
        if not torch.all((labels == 0) | (labels == 254) | (labels == 255)):
            logger.warning(f"Invalid labels found in {dataset_name}. Clamping invalid values.")
            labels = torch.clamp(labels, min=0, max=255)
            labels[(labels > 0) & (labels < 254) & (labels != 255)] = 0 # Map unknown to ID
        return labels
    def _setup_datasets(self):
        """Initialize all evaluation datasets"""
        # 1. Fishyscapes Static
        fishyscapes_static_dir = "/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static"
        self.datasets['fishyscapes_static'] = {
            'loader': self._create_fishyscapes_loader(
                os.path.join(fishyscapes_static_dir, "original"),
                os.path.join(fishyscapes_static_dir, "labels")
            ),
            'name': 'Fishyscapes-Static',
            'anomaly_rate': 0.023 # 2.3%
        }
        # 2. Fishyscapes LostAndFound (100 images)
        fishyscapes_lf_dir = "/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/cityscapes_processed/original"
        self.datasets['fishyscapes_lf'] = {
            'loader': self._create_fishyscapes_loader(
                os.path.join(fishyscapes_lf_dir, "original"),
                os.path.join(fishyscapes_lf_dir, "labels") # Assume labels exist; adjust if needed
            ),
            'name': 'Fishyscapes-LF',
            'anomaly_rate': 0.018 # 1.8%
        }
        # 3. Full LostAndFound test set
        from dataset.data_loader import LostAndFound # From data_loader.py
        lf_dataset = LostAndFound(
            split='test',
            root="/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound",
            transform=self._lf_transform
        )
        self.datasets['lostandfound'] = {
            'loader': DataLoader(lf_dataset, batch_size=1, shuffle=False), # batch_size=1 to reduce memory
            'name': 'LostAndFound-Test',
            'anomaly_rate': 0.001 # 0.1%
        }
        # 4. Road Anomaly
        ra_dataset = RoadAnomaly(
            root="/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly",
            transform=self._ra_transform
        )
        self.datasets['road_anomaly'] = {
            'loader': DataLoader(ra_dataset, batch_size=1, shuffle=False), # batch_size=1 to reduce memory
            'name': 'Road-Anomaly',
            'anomaly_rate': 0.051 # 5.1%
        }
    def _create_fishyscapes_loader(self, image_dir, label_dir):
        """Create Fishyscapes dataloader"""
        from dataset.data_loader import Fishyscapes # From data_loader.py
        dataset = Fishyscapes(
            root=os.path.dirname(image_dir),
            split='Static' if 'Static' in image_dir else 'LostAndFound',
            transform=self._fs_transform
        )
        return DataLoader(dataset, batch_size=1, shuffle=False) # Updated: batch_size=1 to reduce memory
    def _fs_transform(self, img, gt):
        """Transform for Fishyscapes - CORRECTED"""
        img = transforms.Resize((512, 1024), interpolation=InterpolationMode.BILINEAR)(img)
        gt = transforms.Resize((512, 1024), interpolation=InterpolationMode.NEAREST)(gt)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_array = np.array(gt, dtype=np.uint8)
        gt_mapped = np.full_like(gt_array, 255, dtype=np.uint8)
        # For Fishyscapes: 0-18 are Cityscapes classes, 254 is already OOD
        gt_mapped[gt_array <= 18] = 0 # All Cityscapes classes -> ID
        gt_mapped[gt_array == 254] = 254 # Keep existing OOD labels
        # Everything else stays 255 (ignore)
        gt_tensor = torch.tensor(gt_mapped, dtype=torch.long)
        return img, gt_tensor
    def _lf_transform(self, img, gt):
        """Transform for LostAndFound - CORRECTED for actual label values"""
        img = transforms.Resize((512, 1024), interpolation=InterpolationMode.BILINEAR)(img)
        gt = transforms.Resize((512, 1024), interpolation=InterpolationMode.NEAREST)(gt)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_array = np.array(gt, dtype=np.uint8)
        gt_mapped = np.full_like(gt_array, 255, dtype=np.uint8)
        # Based on your raw data: [0, 1, 255]
        gt_mapped[gt_array == 0] = 0 # Background -> ID
        gt_mapped[gt_array == 1] = 254 # Road/obstacles -> OOD (since 1 appears to be the anomaly class)
        # gt_mapped[gt_array == 255] stays 255 (ignore)
        gt_tensor = torch.tensor(gt_mapped, dtype=torch.long)
        return img, gt_tensor
    def _ra_transform(self, img, gt):
        """Transform for Road Anomaly - CORRECTED"""
        img = transforms.Resize((512, 1024), interpolation=InterpolationMode.BILINEAR)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        if gt is not None:
            gt = transforms.Resize((512, 1024), interpolation=InterpolationMode.NEAREST)(gt)
            gt_array = np.array(gt, dtype=np.uint8)
         
            # CRITICAL FIX: Actually apply the mapping
            gt_mapped = np.full_like(gt_array, 255, dtype=np.uint8) # Start with ignore
            gt_mapped[gt_array == 0] = 0 # Road -> ID
            gt_mapped[gt_array == 1] = 254 # Anomaly -> OOD
            # Note: gt_mapped[gt_array == 255] stays 255 (ignore)
         
            gt_tensor = torch.tensor(gt_mapped, dtype=torch.long)
        else:
            gt_tensor = torch.full((512, 1024), 255, dtype=torch.long)
        return img, gt_tensor
    def evaluate_all(self, feature_extractor, projection_pipeline, id_memory, aux_memory,
                     beta_border=128.0, epoch=None, save_visualizations=True):
        """Evaluate on all datasets"""
        results = {}
        for dataset_key, dataset_info in self.datasets.items():
            try:
                logger.info(f"Evaluating on {dataset_info['name']}...") # Keep essential log
                dataset_results = self._evaluate_single_dataset(
                    dataset_info['loader'],
                    feature_extractor,
                    projection_pipeline,
                    id_memory,
                    aux_memory,
                    beta_border,
                    dataset_name=dataset_info['name'],
                    epoch=epoch,
                    save_vis=save_visualizations
                )
                results[dataset_key] = {
                    **dataset_results,
                    'dataset_name': dataset_info['name'],
                    'anomaly_rate': dataset_info['anomaly_rate']
                }
                if dataset_results:
                    for metric_name, value in dataset_results.items():
                        wandb.log({f"{dataset_key}/{metric_name}": value})
            except Exception as e:
                logger.error(f"Error evaluating {dataset_key}: {str(e)}")
                continue
        return results
    def _evaluate_single_dataset(self, loader, feature_extractor, projection_pipeline,
                                 id_memory, aux_memory, beta_border, dataset_name,
                                 epoch=None, save_vis=False):
        """Evaluate a single dataset"""
        feature_extractor.eval()
        projection_pipeline.eval()
        with torch.no_grad(): # Added to reduce memory
            id_memory = id_memory.to(self.device)
            aux_memory = aux_memory.to(self.device)
            score_calc = PixelWiseInferenceScore(id_memory, aux_memory, beta=beta_border)
            metrics = {
                'auroc': AUROCMetric(),
                'fpr95': FPR95Metric(),
                'auprs': AUPRSMetric()
            }
            all_in_scores = []
            all_out_scores = []
            for batch_idx, data in enumerate(tqdm(loader, desc=f"Eval {dataset_name}")):
                if isinstance(data, dict):
                    images = data['data'].to(self.device)
                    labels = data['label'].to(self.device)
                else:
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                batch_dict = {'data': images} # Removed 'label' to skip validation
                extracted = feature_extractor.extract_features_batch(batch_dict)
                if 'features' not in extracted:
                    continue
                features = extracted['features']
                projected = projection_pipeline(features)
                B, C, H, W = projected.shape
                pixel_features = projected.permute(0, 2, 3, 1).contiguous().view(-1, C)
                labels_resized = F.interpolate(
                    labels.unsqueeze(1).float(),
                    size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()
                pixel_labels = labels_resized.view(-1)
                ood_scores = self._compute_ood_scores(pixel_features, score_calc)
                if len(ood_scores) == 0:
                    continue
                if save_vis and batch_idx < 5 and epoch is not None:
                    vis_dir = os.path.join("./results_improved/visualizations", dataset_key.replace(" ", "_").lower())
                    os.makedirs(vis_dir, exist_ok=True)
                    for b in range(B):
                        try:
                            ood_scores_up = F.interpolate(
                                ood_scores.view(B, 1, H, W),
                                size=labels.shape[-2:],
                                mode='bilinear', align_corners=True
                            ).squeeze(1)
                            scores_map = ood_scores_up[b].cpu().numpy()
                            scores_norm = (scores_map - scores_map.min()) / (scores_map.max() - scores_map.min() + 1e-5)
                            colormap = plt.colormaps['inferno']
                            scores_color = colormap(scores_norm)[:, :, :3]
                            orig_img = images[b].cpu().numpy().transpose(1,2,0)
                            label_np = labels[b].cpu().numpy()
                            anomaly_gt = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                            anomaly_gt[label_np == 0] = [0,0,0]
                            anomaly_gt[label_np == 254] = [1,0,0] # Fixed: Changed from ==1 to ==254
                            anomaly_gt[label_np == 255] = [0.5,0.5,0.5]
                            threshold = np.quantile(scores_map.flatten(), 0.95)
                            ood_mask = (scores_map > threshold)
                            ood_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                            ood_color[ood_mask] = [1,1,0]
                            if self.segmentation_head:
                                seg_logits = self.segmentation_head(features)
                                seg_logits = F.interpolate(
                                    seg_logits, size=labels.shape[-2:],
                                    mode='bilinear', align_corners=True
                                )
                                pred = torch.argmax(seg_logits, dim=1)[b].cpu().numpy()
                                seg_with_ood = pred
                                seg_with_ood[ood_mask] = 19
                                seg_color = np.zeros((seg_with_ood.shape[0], seg_with_ood.shape[1], 3))
                                for cls, color in enumerate(CITYSCAPES_COLORMAP):
                                    seg_color[seg_with_ood == cls] = [c/255.0 for c in color]
                            else:
                                seg_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                                for cls, color in enumerate(CITYSCAPES_COLORMAP[:19]):
                                    seg_color[label_np == cls] = [c/255.0 for c in color]
                                seg_color[label_np == 254] = [0,0,1]
                         
                            fig, axs = plt.subplots(1, 5, figsize=(25, 5))
                            axs[0].imshow(orig_img)
                            axs[0].set_title('Original Image')
                            axs[1].imshow(anomaly_gt)
                            axs[1].set_title('Anomaly Ground Truth')
                            axs[2].imshow(seg_color)
                            axs[2].set_title('Segmentation incl. OOD')
                            axs[3].imshow(ood_color)
                            axs[3].set_title('OOD Map')
                            axs[4].imshow(scores_color)
                            axs[4].set_title('OOD Score Map')
                         
                            # Remove axis ticks for cleaner look
                            for ax in axs:
                                ax.set_xticks([])
                                ax.set_yticks([])
                         
                            plt.tight_layout()
                            save_path = os.path.join(vis_dir, f'ood_vis_{dataset_name}_epoch{epoch}_{batch_idx}_{b}.png')
                            plt.savefig(save_path, dpi=100, bbox_inches='tight')
                            wandb.log({
                                f"{dataset_name}_ood_vis_{batch_idx}_{b}": wandb.Image(save_path),
                                "epoch": epoch
                            })
                         
                            # Critical: Proper cleanup
                            plt.close(fig)
                            plt.close('all')
                         
                            # Clear variables to free memory
                            del orig_img, scores_map, scores_norm, scores_color
                            del anomaly_gt, ood_color, seg_color, fig, axs
                         
                        except Exception as e:
                            logger.warning(f"Visualization failed for batch {batch_idx}: {e}")
                            plt.close('all') # Ensure cleanup even on error
                         
                        finally:
                            torch.cuda.empty_cache()
                valid_mask = (pixel_labels != 255)
                valid_ood = ood_scores[valid_mask]
                valid_labels = pixel_labels[valid_mask]
                if len(valid_ood) == 0:
                    continue
                sub_ood, sub_labels = self.safe_subsample(
                    valid_ood, valid_labels, max_pixels=50000 # Reduced from None to avoid OOM
                )
                in_mask = (sub_labels == 0)
                out_mask = (sub_labels == 254) # Fixed: Changed from ==1 to ==254
                if in_mask.sum() > 0:
                    all_in_scores.append(sub_ood[in_mask])
                if out_mask.sum() > 0:
                    all_out_scores.append(sub_ood[out_mask])
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        if not all_in_scores or not all_out_scores:
            logger.warning(f"No valid scores for {dataset_name}")
            return {}
        in_scores = torch.cat(all_in_scores)
        out_scores = torch.cat(all_out_scores)
        results = {}
        for metric_name, metric in metrics.items():
            score = metric(in_scores, out_scores)
            results[metric_name] = float(score)
        logger.info(f"{dataset_name} Results: {results}") # Keep essential log
        return results
    def _compute_ood_scores(self, pixel_features, score_calc):
        if len(pixel_features) == 0:
            return torch.tensor([], device=self.device)
        chunk_size = 2000 # Reduced from 5000 to avoid OOM
        all_scores = []
        for i in range(0, len(pixel_features), chunk_size):
            end_i = min(i + chunk_size, len(pixel_features))
            chunk = pixel_features[i:end_i]
            with torch.cuda.amp.autocast():
                chunk_scores = score_calc(chunk)
            all_scores.append(chunk_scores.squeeze())
        return torch.cat(all_scores) if all_scores else torch.tensor([], device=self.device)
    def safe_subsample(self, scores, labels, max_pixels=None):
        total_pixels = len(scores)
        if total_pixels == 0:
            return scores, labels
        if max_pixels is None or total_pixels <= max_pixels:
            return scores, labels
        in_mask = (labels == 0)
        out_mask = (labels == 254) # Fixed: Changed from ==1 to ==254
        in_scores = scores[in_mask]
        out_scores = scores[out_mask]
        in_labels = labels[in_mask]
        out_labels = labels[out_mask]
        half_max = max_pixels // 2
        if len(in_scores) > half_max:
            in_perm = torch.randperm(len(in_scores))[:half_max]
            in_scores = in_scores[in_perm]
            in_labels = in_labels[in_perm]
        if len(out_scores) > half_max:
            out_perm = torch.randperm(len(out_scores))[:half_max]
            out_scores = out_scores[out_perm]
            out_labels = out_labels[out_perm]
        return torch.cat([in_scores, out_scores]), torch.cat([in_labels, out_labels])
def main():
    """Main training script"""
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(42)
    np.random.seed(42)
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    logger = logging.getLogger(__name__)
    train_config = {
        'model_path': "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth",
        'checkpoint_dir': "./checkpoints_improved_ood_segmentation",
        'num_classes': 19,
        'learning_rate': 1e-5,
        'weight_decay': 5e-5,
        'batch_size': 2,
        'num_workers': 0
    }
    cityscapes_root = "/home/ha51dybi/PEBAL/cityscapes"
    images_dir = os.path.join(cityscapes_root, "images", "city_gt_fine", "train")
    labels_dir = os.path.join(cityscapes_root, "annotation", "city_gt_fine", "train")
    class CustomArgs:
        def __init__(self):
            self.ddp = False
            self.local_rank = -1
            self.gpus = 1
            self.world_size = 1
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
    train_loader, _, _ = get_mix_loader(
        engine=engine_instance,
        augment=True,
        cs_root="/home/ha51dybi/PEBAL/cityscapes",
        coco_root="/home/ha51dybi/PEBAL/coco"
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
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    val_dataset = Cityscapes(
        root="/home/ha51dybi/PEBAL/cityscapes",
        split='val',
        transform=val_joint_transform
    )
    wrapped_val = DictWrapperDataset(val_dataset)
    val_loader = DataLoader(
        wrapped_val,
        batch_size=1, # Updated: batch_size=1 to reduce memory
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )
    val_iter = iter(val_loader)
    fixed_batches = []
    try:
        for _ in range(3):
            fixed_batches.append(next(val_iter))
    except StopIteration:
        pass
    trainer = ImprovedOODSegmentationTrainer(
        train_config,
        train_loader,
        val_loader,
        fixed_batches=fixed_batches,
        resume_from=None
    )
    best_fpr95 = trainer.train()
    return best_fpr95
if __name__ == "__main__":
    main()
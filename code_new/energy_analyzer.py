#!/usr/bin/env python3
# energy_beta_analyzer.py - Enhanced version with more testing, debugging, memory usage checks,
# additional metrics (FPR95, AUPR, KS test), multiple feature norm types, more beta values,
# and updated to use optimized classes and loss from pixel_energy.py for better stability and effectiveness.
# Added checks for real data performance and distribution separation suitable for OOD segmentation.
# Updated to incorporate reference Hopfield Boosting implementation for comparison.
# Now tests both 'optimized' and 'reference' implementations to evaluate performance of border energy and OOD loss.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import gc
import argparse
from tqdm import tqdm
import traceback
from scipy.stats import ks_2samp
from scipy.integrate import trapz

# Import your components
from feature_extractor import FeatureExtractor
from projection_head import SimpleProjectionHead
from hopfield_memory_builder import RealMemoryBuilder
from engine.engine import Engine
from dataset.data_loader import get_mix_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('energy_beta_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# ---- Pasted and integrated optimized components from pixel_energy.py ----

def lse(beta: float, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Computes the numerically stable Log-Sum-Exp operation."""
    if scores.numel() == 0:
        return torch.empty(0, device=scores.device, dtype=scores.dtype)
   
    # Improved numerical stability with max subtraction
    max_scores = torch.max(beta * scores, dim=dim, keepdim=True)[0]
    return (torch.logsumexp(beta * scores - max_scores, dim=dim) + max_scores.squeeze(dim)) / beta

def enhance_features(features, method='l2', **kwargs):
    """Enhanced feature normalization with multiple options."""
    if method == 'l2':
        return F.normalize(features, p=2, dim=1), None
    elif method == 'bn':
        bn = kwargs.get('bn', None)
        if bn is None:
            bn = nn.BatchNorm1d(features.shape[1], affine=False).to(features.device)
        features = bn(features)
        return F.normalize(features, p=2, dim=1), bn
    elif method == 'pca_whitening':
        try:
            u, s, v = torch.pca_lowrank(features, q=min(features.shape[0], features.shape[1]), center=True)
            whitened = torch.mm(features, v) / (torch.sqrt(s.unsqueeze(0)) + 1e-8)
            return F.normalize(whitened, p=2, dim=1), {'u': u, 's': s, 'v': v}
        except Exception as e:
            logger.warning(f"Warning: PCA failed ({e}), falling back to L2")
            return F.normalize(features, p=2, dim=1), None
    elif method == 'decorrelation':
        flat_features = features.detach().clone()
        mean = torch.mean(flat_features, dim=0, keepdim=True)
        centered = flat_features - mean
        cov = torch.mm(centered.t(), centered) / (centered.size(0) - 1 + 1e-8)
     
        # More robust regularization
        cov += torch.eye(cov.size(0), device=cov.device) * 1e-4
     
        try:
            u, s, v = torch.svd(cov)
            decorr_mat = torch.mm(torch.diag(1.0 / (torch.sqrt(s) + 1e-6)), u.t())
            decorrelated = torch.mm(centered, decorr_mat.t())
            return F.normalize(decorrelated, p=2, dim=1), {'mean': mean, 'decorr_mat': decorr_mat}
        except Exception as e:
            logger.warning(f"Warning: SVD failed ({e}), falling back to L2")
            return F.normalize(features, p=2, dim=1), None
 
    return F.normalize(features, p=2, dim=1), None

class PixelWiseBorderEnergy(nn.Module):
    """
    Optimized border energy calculation based on debug analysis.
    Fixed beta=8.0 based on optimal empirical performance.
    """
    def __init__(self, a: torch.Tensor, b: torch.Tensor, beta: float = 8.0,
                 feature_norm_type: str = 'l2'):
        super().__init__()
    
        self.beta = beta
    
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError(f"Expected 2D tensors in [memory_size, features] format, got shapes {a.shape} and {b.shape}")
    
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"Feature dimensions must match: {a.shape[1]} vs {b.shape[1]}")
    
        # Feature enhancement
        self.feature_norm_type = feature_norm_type
        self.norm_params = None
     
        if feature_norm_type != 'l2':
            a_norm, norm_params = enhance_features(a, method=feature_norm_type)
            b_norm, _ = enhance_features(b, method=feature_norm_type, **({} if norm_params is None else {'bn': norm_params} if feature_norm_type == 'bn' else {'pca': norm_params}))
            self.norm_params = norm_params
        else:
            a_norm = F.normalize(a, p=2, dim=1)
            b_norm = F.normalize(b, p=2, dim=1)
    
        self.register_buffer('id_memory', a_norm)
        self.register_buffer('ood_memory', b_norm)
     
    def forward(self, pixel_features: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        Optimized border energy calculation with better memory management.
        """
        if pixel_features.numel() == 0:
            return torch.empty(0, device=pixel_features.device)
    
        # Handle input shapes
        orig_shape = None
        if pixel_features.dim() == 4:
            orig_shape = (pixel_features.shape[0], pixel_features.shape[2], pixel_features.shape[3])
            pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous().view(-1, pixel_features.shape[1])
        elif pixel_features.dim() == 3:
            orig_shape = pixel_features.shape
            pixel_features = pixel_features.view(-1, pixel_features.shape[-1])
        elif pixel_features.dim() == 1:
            pixel_features = pixel_features.unsqueeze(0)
    
        # Apply feature enhancement if needed
        if self.feature_norm_type != 'l2' and self.norm_params is not None:
            pixel_features, _ = enhance_features(
                pixel_features, method=self.feature_norm_type,
                **({} if self.norm_params is None else
                   {'bn': self.norm_params} if self.feature_norm_type == 'bn' else
                   {'pca': self.norm_params})
            )
        else:
            pixel_features = F.normalize(pixel_features, p=2, dim=1)
         
        num_pixels = pixel_features.shape[0]
        energies = torch.zeros(num_pixels, device=pixel_features.device, dtype=pixel_features.dtype)
     
        # Process in smaller batches for memory efficiency
        for i in range(0, num_pixels, batch_size):
            chunk = pixel_features[i:i+batch_size]
        
            # Compute similarities: chunk @ memory^T gives [batch, memory_size]
            id_similarities = chunk @ self.id_memory.t() # [batch, N_id]
            ood_similarities = chunk @ self.ood_memory.t() # [batch, N_ood]
        
            # Compute lse values
            id_scores = lse(self.beta, id_similarities, dim=1)
            ood_scores = lse(self.beta, ood_similarities, dim=1)
         
            # Convert to probabilities
            max_score = torch.maximum(id_scores, ood_scores)
            id_prob = torch.exp(self.beta * (id_scores - max_score))
            ood_prob = torch.exp(self.beta * (ood_scores - max_score))
         
            # Normalize probabilities
            total_prob = id_prob + ood_prob + 1e-8
            id_prob = id_prob / total_prob
            ood_prob = ood_prob / total_prob
         
            # Border energy = uncertainty = variance of binary distribution
            # Maximum at p=0.5, minimum at p=0 or p=1
            border_energy_chunk = 4.0 * id_prob * ood_prob
         
            energies[i:i+batch_size] = border_energy_chunk
         
        # Ensure energies are always positive
        energies = torch.clamp(energies, min=1e-6)
       
        # Restore original shape if needed
        if orig_shape is not None:
            energies = energies.view(*orig_shape)
           
        return energies

class PixelWiseInferenceScore(nn.Module):
    """
    Inference score calculator with consistent beta=8.0.
    """
    def __init__(self, id_memory: torch.Tensor, ood_memory: torch.Tensor,
                 beta: float = 8.0, feature_norm_type: str = 'l2'):
        super().__init__()
    
        self.beta = beta
    
        if id_memory.dim() != 2 or ood_memory.dim() != 2:
            raise ValueError(f"Expected 2D memory tensors in [memory_size, features] format")
        
        if id_memory.shape[1] != ood_memory.shape[1]:
            raise ValueError(f"Feature dimensions must match: {id_memory.shape[1]} vs {ood_memory.shape[1]}")
    
        # Feature enhancement
        self.feature_norm_type = feature_norm_type
        self.norm_params = None
     
        if feature_norm_type != 'l2':
            id_memory_norm, norm_params = enhance_features(id_memory, method=feature_norm_type)
            ood_memory_norm, _ = enhance_features(ood_memory, method=feature_norm_type, **({} if norm_params is None else {'bn': norm_params} if feature_norm_type == 'bn' else {'pca': norm_params}))
            self.norm_params = norm_params
        else:
            id_memory_norm = F.normalize(id_memory, p=2, dim=1)
            ood_memory_norm = F.normalize(ood_memory, p=2, dim=1)
    
        self.register_buffer('id_memory', id_memory_norm)
        self.register_buffer('ood_memory', ood_memory_norm)
     
    def forward(self, pixel_features: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        Calculates the per-pixel inference score with improved memory efficiency.
        """
        if pixel_features.numel() == 0:
            return torch.empty(0, device=pixel_features.device)
     
        orig_shape = None
     
        if pixel_features.dim() == 4:
            orig_shape = (pixel_features.shape[0], pixel_features.shape[2], pixel_features.shape[3])
            pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous().view(-1, pixel_features.shape[1])
        elif pixel_features.dim() == 3:
            orig_shape = pixel_features.shape
            pixel_features = pixel_features.view(-1, pixel_features.shape[-1])
        elif pixel_features.dim() == 1:
            pixel_features = pixel_features.unsqueeze(0)
    
        if self.feature_norm_type != 'l2' and self.norm_params is not None:
            pixel_features, _ = enhance_features(
                pixel_features, method=self.feature_norm_type,
                **({} if self.norm_params is None else
                   {'bn': self.norm_params} if self.feature_norm_type == 'bn' else
                   {'pca': self.norm_params})
            )
        else:
            pixel_features = F.normalize(pixel_features, p=2, dim=1)
         
        num_pixels = pixel_features.shape[0]
        scores = torch.zeros(num_pixels, device=pixel_features.device, dtype=pixel_features.dtype)
     
        for i in range(0, num_pixels, batch_size):
            chunk = pixel_features[i:i+batch_size]
        
            id_similarities = chunk @ self.id_memory.t()
            ood_similarities = chunk @ self.ood_memory.t()
        
            lse_id = lse(self.beta, id_similarities, dim=1)
            lse_ood = lse(self.beta, ood_similarities, dim=1)
        
            # Inference score: s(ξ) = lse(β, X^T ξ) - lse(β, O^T ξ)
            scores[i:i+batch_size] = lse_id - lse_ood
     
        # Restore original shape if needed
        if orig_shape is not None:
            scores = scores.view(*orig_shape)
           
        return scores

# ---- Reference implementation from Hopfield Boosting ----

def logmeanexp(beta, scores, dim=-1):
    if scores.numel() == 0:
        return torch.empty(0, device=scores.device, dtype=scores.dtype)
    n = scores.size(dim)
    return torch.logsumexp(beta * scores, dim=dim) - torch.log(torch.tensor(n, dtype=scores.dtype, device=scores.device))

class ReferenceEnergy(nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, beta_a: float, beta_b: float,
                 feature_norm_type: str = 'l2'):
        super().__init__()
    
        self.beta_a = beta_a
        self.beta_b = beta_b
    
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError(f"Expected 2D tensors in [memory_size, features] format, got shapes {a.shape} and {b.shape}")
    
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"Feature dimensions must match: {a.shape[1]} vs {b.shape[1]}")
    
        # Feature enhancement
        self.feature_norm_type = feature_norm_type
        self.norm_params = None
     
        if feature_norm_type != 'l2':
            a_norm, norm_params = enhance_features(a, method=feature_norm_type)
            b_norm, _ = enhance_features(b, method=feature_norm_type, **({} if norm_params is None else {'bn': norm_params} if feature_norm_type == 'bn' else {'pca': norm_params}))
            self.norm_params = norm_params
        else:
            a_norm = F.normalize(a, p=2, dim=1)
            b_norm = F.normalize(b, p=2, dim=1)
    
        self.register_buffer('id_memory', a_norm)
        self.register_buffer('ood_memory', b_norm)
     
    def forward(self, pixel_features: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        if pixel_features.numel() == 0:
            return torch.empty(0, device=pixel_features.device), torch.empty(0, device=pixel_features.device)
    
        orig_shape = None
        if pixel_features.dim() == 4:
            orig_shape = (pixel_features.shape[0], pixel_features.shape[2], pixel_features.shape[3])
            pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous().view(-1, pixel_features.shape[1])
        elif pixel_features.dim() == 3:
            orig_shape = pixel_features.shape
            pixel_features = pixel_features.view(-1, pixel_features.shape[-1])
        elif pixel_features.dim() == 1:
            pixel_features = pixel_features.unsqueeze(0)
    
        if self.feature_norm_type != 'l2' and self.norm_params is not None:
            pixel_features, _ = enhance_features(
                pixel_features, method=self.feature_norm_type,
                **({} if self.norm_params is None else
                   {'bn': self.norm_params} if self.feature_norm_type == 'bn' else
                   {'pca': self.norm_params})
            )
        else:
            pixel_features = F.normalize(pixel_features, p=2, dim=1)
         
        num_pixels = pixel_features.shape[0]
        a_energies = torch.zeros(num_pixels, device=pixel_features.device, dtype=pixel_features.dtype)
        b_energies = torch.zeros(num_pixels, device=pixel_features.device, dtype=pixel_features.dtype)
     
        for i in range(0, num_pixels, batch_size):
            chunk = pixel_features[i:i+batch_size]
        
            attn_a = chunk @ self.id_memory.t()
            attn_b = chunk @ self.ood_memory.t()
        
            a_energy = -logmeanexp(self.beta_a, attn_a, dim=-1)
            b_energy = -logmeanexp(self.beta_b, attn_b, dim=-1)
         
            a_energies[i:i+batch_size] = a_energy
            b_energies[i:i+batch_size] = b_energy
         
        if orig_shape is not None:
            a_energies = a_energies.view(*orig_shape)
            b_energies = b_energies.view(*orig_shape)
           
        return a_energies, b_energies

class ReferenceBorderEnergy(ReferenceEnergy):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, beta: float = 8.0,
                 feature_norm_type: str = 'l2'):
        super().__init__(a, b, beta, beta, feature_norm_type)
        self.beta_border = beta
    
    def forward(self, pixel_features: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        a_energy, b_energy = super().forward(pixel_features, batch_size)
        union_energy = -1/self.beta_border * torch.logaddexp(-self.beta_border * a_energy, -self.beta_border * b_energy) + 1/self.beta_border * torch.log(torch.tensor(2.0, device=a_energy.device))
        border_energy = a_energy + b_energy - 2 * union_energy
        return border_energy

class ReferenceOneSidedEnergy(ReferenceEnergy):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, beta: float = 8.0,
                 feature_norm_type: str = 'l2'):
        super().__init__(a, b, beta, beta, feature_norm_type)
    
    def forward(self, pixel_features: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        a_energy, b_energy = super().forward(pixel_features, batch_size)
        one_sided_energy = a_energy - b_energy
        return one_sided_energy

# ---- Updated OOD loss to support both implementations ----

def compute_hopfield_ood_loss(
   id_pixels: torch.Tensor,
   ood_pixels: torch.Tensor,
   device: torch.device,
   imbalance_weight: float = 3.0,
   feature_norm_type: str = 'l2',
   lambda_ood: float = 1.0,
   separation_margin: float = 0.2,
   separation_weight: float = 1.5,
   beta: float = 8.0,
   impl_type: str = 'optimized',
   **kwargs
) -> torch.Tensor:
   """
   Optimized OOD loss computation with support for both implementations.
   """
   if id_pixels.numel() < 2 or ood_pixels.numel() < 2:
       placeholder_loss = torch.tensor(0.01, device=device, requires_grad=True)
       logger.warning("Insufficient pixels for OOD loss, using placeholder")
       return placeholder_loss * lambda_ood
  
   if id_pixels.dim() != 2 or ood_pixels.dim() != 2:
       raise ValueError(f"Expected 2D pixel tensors, got shapes {id_pixels.shape} and {ood_pixels.shape}")
   # Create energy calculator
   try:
       if impl_type == 'optimized':
           energy_calc = PixelWiseBorderEnergy(
               id_pixels, ood_pixels,
               beta=beta,
               feature_norm_type=feature_norm_type
           ).to(device)
       elif impl_type == 'reference':
           energy_calc = ReferenceBorderEnergy(
               id_pixels, ood_pixels,
               beta=beta,
               feature_norm_type=feature_norm_type
           ).to(device)
       else:
           raise ValueError(f"Unknown impl_type: {impl_type}")
   except Exception as e:
       logger.error(f"Error creating energy calculator: {e}")
       return torch.tensor(0.1, device=device, requires_grad=True) * lambda_ood
  
   # Calculate border energies for all pixels
   all_pixels_in_batch = torch.cat([id_pixels, ood_pixels], dim=0)
   try:
       border_energies = energy_calc(all_pixels_in_batch)
   except Exception as e:
       logger.error(f"Error computing border energies: {e}")
       return torch.tensor(0.1, device=device, requires_grad=True) * lambda_ood
  
   # Validate and fix energies if needed
   nan_count = torch.isnan(border_energies).sum().item()
   inf_count = torch.isinf(border_energies).sum().item()
  
   if nan_count > 0 or inf_count > 0:
       logger.warning(f"WARNING: {nan_count} NaN, {inf_count} Inf energies detected - fixing")
       border_energies = torch.nan_to_num(border_energies, nan=0.5, posinf=5.0, neginf=0.0)
      
   # Clamp to ensure positive energies (border energies should be positive)
   border_energies = torch.clamp(border_energies, min=1e-6)
  
   # Debug statistics
   if not hasattr(compute_hopfield_ood_loss, '_debug_counter'):
       compute_hopfield_ood_loss._debug_counter = 0
   compute_hopfield_ood_loss._debug_counter += 1
  
   # Separate ID and OOD energies
   num_id = id_pixels.shape[0]
   num_ood = ood_pixels.shape[0]
   id_energies = border_energies[:num_id]
   ood_energies = border_energies[num_id:]
   # Calculate statistics periodically
   if compute_hopfield_ood_loss._debug_counter % 50 == 0:
       energy_stats = {
           'min': border_energies.min().item(),
           'max': border_energies.max().item(),
           'mean': border_energies.mean().item(),
           'std': border_energies.std().item(),
           'id_mean': id_energies.mean().item(),
           'ood_mean': ood_energies.mean().item(),
           'beta': beta
       }
      
       logger.info(f"Energy stats ({impl_type}): β={beta}, min={energy_stats['min']:.4f}, "
                  f"max={energy_stats['max']:.4f}, mean={energy_stats['mean']:.4f}, "
                  f"id_mean={energy_stats['id_mean']:.4f}, ood_mean={energy_stats['ood_mean']:.4f}")
  
   # ID loss: penalize high energy (being near boundary)
   id_loss = id_energies.mean()
   # OOD loss: penalize low energy (being far from boundary)
   # We want OOD to have high energy, so minimize negative of energy
   ood_loss = -ood_energies.mean()
   # Basic combined loss
   total_loss = id_loss + imbalance_weight * ood_loss
   # Apply dynamic imbalance weighting based on pixel counts
   if num_ood > 0 and num_id > 0:
       scarcity_ratio = max(1.0, min(10.0, num_id / (num_ood + 1e-8)))
      
       if scarcity_ratio > 3.0:
           effective_weight = min(imbalance_weight * (scarcity_ratio / 3.0), 10.0)
           total_loss = id_loss + effective_weight * ood_loss
       
           if effective_weight > 5.0 and compute_hopfield_ood_loss._debug_counter % 100 == 0:
               logger.info(f"Strong imbalance weighting: ratio={scarcity_ratio:.2f}, weight={effective_weight:.2f}")
   # Add separation loss - enforce minimum margin between ID and OOD energies
   if num_ood > 0 and num_id > 0:
       separation = ood_energies.mean() - id_energies.mean()
      
       margin_tensor = torch.tensor(separation_margin, device=device)
      
       sep_loss = F.relu(margin_tensor - separation) * separation_weight
       total_loss = total_loss + sep_loss
      
       if compute_hopfield_ood_loss._debug_counter % 50 == 0:
           logger.info(f"Separation: {separation.item():.4f}, target={separation_margin:.3f}, loss={sep_loss.item():.4f}")
   # Final safety check
   if torch.isnan(total_loss) or torch.isinf(total_loss):
       logger.warning("WARNING: Invalid final loss, using emergency fallback")
       total_loss = torch.tensor(0.1, device=device, requires_grad=True)
  
   # Apply lambda weighting
   return total_loss * lambda_ood

# ---- End of components ----

# Enhanced utility functions for metrics
def compute_fpr95(labels_np, scores_np, is_ood_high_score=True):
    """Compute FPR at 95% TPR. High score indicates OOD if is_ood_high_score=True."""
    if not is_ood_high_score:
        scores_np = -scores_np  # Invert if high score is ID
    sorted_indices = np.argsort(-scores_np)  # Descending
    sorted_labels = labels_np[sorted_indices]
    n_pos = np.sum(sorted_labels)
    n_neg = len(sorted_labels) - n_pos
    cum_tp = np.cumsum(sorted_labels)
    cum_fp = np.cumsum(1 - sorted_labels)
    tpr = cum_tp / n_pos if n_pos > 0 else np.zeros_like(cum_tp)
    fpr = cum_fp / n_neg if n_neg > 0 else np.zeros_like(cum_fp)
    idx = np.searchsorted(tpr, 0.95)
    if idx < len(fpr):
        return fpr[idx]
    return 1.0

def compute_aupr(labels_np, scores_np, is_ood_high_score=True):
    """Compute AUPR. High score indicates OOD if is_ood_high_score=True."""
    if not is_ood_high_score:
        scores_np = -scores_np
    sorted_indices = np.argsort(-scores_np)
    sorted_labels = labels_np[sorted_indices]
    n_pos = np.sum(sorted_labels)
    cum_tp = np.cumsum(sorted_labels)
    cum_pred_pos = np.arange(1, len(labels_np) + 1)
    precision = cum_tp / cum_pred_pos
    recall = cum_tp / n_pos if n_pos > 0 else np.zeros_like(cum_tp)
    # Trapezoidal integration
    return trapz(precision, recall)

def get_gpu_memory(device):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return allocated, max_alloc
    return 0, 0

# Updated test function with support for multiple implementations
def test_energy_with_memories(id_memory, ood_memory, device, output_dir, beta_values=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], norm_types=['l2'], impl_types=['optimized', 'reference']):
    """Enhanced test: more betas, norm types, metrics, memory checks, and multiple implementations."""
    logger.info(f"Testing energy with real memories: ID={id_memory.shape}, OOD={ood_memory.shape}")
    logger.info(f"Norm types: {norm_types}, Impl types: {impl_types}")
    
    results = {}
    
    for impl_type in impl_types:
        logger.info(f"Processing impl_type: {impl_type}")
        results[impl_type] = {}
        for norm_type in norm_types:
            logger.info(f"Processing norm_type: {norm_type} for {impl_type}")
            results[impl_type][norm_type] = {}
            # Create figure for this impl and norm
            fig, axs = plt.subplots(len(beta_values), 2, figsize=(16, 8 * len(beta_values)))
            fig.suptitle(f'Energy Analysis ({impl_type}, Norm: {norm_type})', fontsize=16)
            
            for i, beta in enumerate(beta_values):
                logger.info(f"Testing beta={beta} for impl={impl_type}, norm={norm_type}...")
                
                try:
                    alloc, max_alloc = get_gpu_memory(device)
                    logger.info(f"Pre-sample GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")
                    
                    sample_size = min(10000, min(len(id_memory), len(ood_memory)))
                    id_indices = torch.randperm(len(id_memory))[:sample_size]
                    ood_indices = torch.randperm(len(ood_memory))[:sample_size]
                    
                    id_subset = id_memory[id_indices]
                    ood_subset = ood_memory[ood_indices]
                    
                    logger.info(f"Subsets shapes: ID {id_subset.shape}, OOD {ood_subset.shape}")
                    
                    alloc, max_alloc = get_gpu_memory(device)
                    logger.info(f"Post-sample GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")
                    
                    # Create border energy module
                    if impl_type == 'optimized':
                        border_energy = PixelWiseBorderEnergy(
                            a=id_subset,
                            b=ood_subset,
                            beta=beta,
                            feature_norm_type=norm_type
                        ).to(device)
                        inference_score = PixelWiseInferenceScore(
                            id_memory=id_subset,
                            ood_memory=ood_subset,
                            beta=beta,
                            feature_norm_type=norm_type
                        ).to(device)
                    elif impl_type == 'reference':
                        border_energy = ReferenceBorderEnergy(
                            a=id_subset,
                            b=ood_subset,
                            beta=beta,
                            feature_norm_type=norm_type
                        ).to(device)
                        inference_score = ReferenceOneSidedEnergy(
                            a=id_subset,
                            b=ood_subset,
                            beta=beta,
                            feature_norm_type=norm_type
                        ).to(device)
                    
                    alloc, max_alloc = get_gpu_memory(device)
                    logger.info(f"Post-modules GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")
                    
                    # Calculate border energies
                    with torch.no_grad():
                        batch_size = 2048
                        id_border_energies = border_energy(id_subset, batch_size=batch_size)
                        ood_border_energies = border_energy(ood_subset, batch_size=batch_size)
                    
                    alloc, max_alloc = get_gpu_memory(device)
                    logger.info(f"Post-border_compute GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")
                    
                    # Border stats
                    id_border_mean = id_border_energies.mean().item()
                    id_border_std = id_border_energies.std().item()
                    ood_border_mean = ood_border_energies.mean().item()
                    ood_border_std = ood_border_energies.std().item()
                    
                    border_labels_np = np.concatenate([np.zeros(len(id_border_energies)), np.ones(len(ood_border_energies))])
                    border_scores_np = np.concatenate([id_border_energies.cpu().numpy(), ood_border_energies.cpu().numpy()])
                    
                    sorted_indices = np.argsort(border_scores_np)
                    sorted_labels = border_labels_np[sorted_indices]
                    n_pos = np.sum(sorted_labels)
                    n_neg = len(sorted_labels) - n_pos
                    tpr = np.cumsum(sorted_labels) / n_pos if n_pos > 0 else np.zeros_like(sorted_labels)
                    fpr = np.cumsum(1 - sorted_labels) / n_neg if n_neg > 0 else np.zeros_like(sorted_labels)
                    border_auc = np.trapz(tpr, fpr)
                    
                    border_fpr95 = compute_fpr95(border_labels_np, border_scores_np, is_ood_high_score=True)
                    border_aupr = compute_aupr(border_labels_np, border_scores_np, is_ood_high_score=True)
                    border_ks_stat, border_ks_p = ks_2samp(id_border_energies.cpu().numpy(), ood_border_energies.cpu().numpy())
                    
                    # Inference scores
                    with torch.no_grad():
                        id_inference_scores = inference_score(id_subset, batch_size=batch_size)
                        ood_inference_scores = inference_score(ood_subset, batch_size=batch_size)
                    
                    alloc, max_alloc = get_gpu_memory(device)
                    logger.info(f"Post-inference_compute GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")
                    
                    # Inference stats
                    id_score_mean = id_inference_scores.mean().item()
                    id_score_std = id_inference_scores.std().item()
                    ood_score_mean = ood_inference_scores.mean().item()
                    ood_score_std = ood_inference_scores.std().item()
                    
                    # Determine if OOD has higher scores
                    is_inference_ood_high = ood_score_mean > id_score_mean
                    
                    score_labels_np = np.concatenate([np.zeros(len(id_inference_scores)), np.ones(len(ood_inference_scores))])
                    if is_inference_ood_high:
                        score_scores_np = np.concatenate([id_inference_scores.cpu().numpy(), ood_inference_scores.cpu().numpy()])
                    else:
                        score_scores_np = np.concatenate([-id_inference_scores.cpu().numpy(), -ood_inference_scores.cpu().numpy()])
                    
                    sorted_indices = np.argsort(score_scores_np)
                    sorted_labels = score_labels_np[sorted_indices]
                    n_pos = np.sum(sorted_labels)
                    n_neg = len(sorted_labels) - n_pos
                    tpr = np.cumsum(sorted_labels) / n_pos if n_pos > 0 else np.zeros_like(sorted_labels)
                    fpr = np.cumsum(1 - sorted_labels) / n_neg if n_neg > 0 else np.zeros_like(sorted_labels)
                    score_auc = np.trapz(tpr, fpr)
                    
                    score_fpr95 = compute_fpr95(score_labels_np, score_scores_np, is_ood_high_score=True)
                    score_aupr = compute_aupr(score_labels_np, score_scores_np, is_ood_high_score=True)
                    score_ks_stat, score_ks_p = ks_2samp(id_inference_scores.cpu().numpy(), ood_inference_scores.cpu().numpy())
                    
                    # Log stats
                    logger.info(f"Border Energy ({impl_type}, β={beta}, norm={norm_type}):")
                    logger.info(f"  ID: {id_border_mean:.4f}±{id_border_std:.4f}, OOD: {ood_border_mean:.4f}±{ood_border_std:.4f}")
                    logger.info(f"  Separation: {ood_border_mean - id_border_mean:.4f}, KS-stat: {border_ks_stat:.4f} (p={border_ks_p:.4f})")
                    logger.info(f"  AUROC: {border_auc:.4f}, AUPR: {border_aupr:.4f}, FPR95: {border_fpr95:.4f}")
                    
                    logger.info(f"Inference Score ({impl_type}, β={beta}, norm={norm_type}):")
                    logger.info(f"  ID: {id_score_mean:.4f}±{id_score_std:.4f}, OOD: {ood_score_mean:.4f}±{ood_score_std:.4f}")
                    sep = ood_score_mean - id_score_mean if is_inference_ood_high else id_score_mean - ood_score_mean
                    logger.info(f"  Separation: {sep:.4f}, KS-stat: {score_ks_stat:.4f} (p={score_ks_p:.4f})")
                    logger.info(f"  AUROC: {score_auc:.4f}, AUPR: {score_aupr:.4f}, FPR95: {score_fpr95:.4f}")
                    
                    # Plots
                    ax1 = axs[i, 0]
                    min_val = min(id_border_energies.min().item(), ood_border_energies.min().item())
                    max_val = max(id_border_energies.max().item(), ood_border_energies.max().item())
                    bins = np.linspace(min_val, max_val, 100)
                    ax1.hist(id_border_energies.cpu().numpy(), bins=bins, alpha=0.7, label='ID', density=True)
                    ax1.hist(ood_border_energies.cpu().numpy(), bins=bins, alpha=0.7, label='OOD', density=True)
                    ax1.set_title(f'Border ({impl_type}, β={beta})\nID: {id_border_mean:.4f}±{id_border_std:.4f}, OOD: {ood_border_mean:.4f}±{ood_border_std:.4f}\nSep: {ood_border_mean - id_border_mean:.4f}, KS: {border_ks_stat:.4f}\nAUROC: {border_auc:.4f}, AUPR: {border_aupr:.4f}, FPR95: {border_fpr95:.4f}')
                    ax1.set_xlabel('Border Energy')
                    ax1.set_ylabel('Density')
                    ax1.legend()
                    ax1.grid(alpha=0.3)
                    
                    ax2 = axs[i, 1]
                    min_val = min(id_inference_scores.min().item(), ood_inference_scores.min().item())
                    max_val = max(id_inference_scores.max().item(), ood_inference_scores.max().item())
                    bins = np.linspace(min_val, max_val, 100)
                    ax2.hist(id_inference_scores.cpu().numpy(), bins=bins, alpha=0.7, label='ID', density=True)
                    ax2.hist(ood_inference_scores.cpu().numpy(), bins=bins, alpha=0.7, label='OOD', density=True)
                    ax2.set_title(f'Inference ({impl_type}, β={beta})\nID: {id_score_mean:.4f}±{id_score_std:.4f}, OOD: {ood_score_mean:.4f}±{ood_score_std:.4f}\nSep: {sep:.4f}, KS: {score_ks_stat:.4f}\nAUROC: {score_auc:.4f}, AUPR: {score_aupr:.4f}, FPR95: {score_fpr95:.4f}')
                    ax2.set_xlabel('Inference Score')
                    ax2.set_ylabel('Density')
                    ax2.legend()
                    ax2.grid(alpha=0.3)
                    
                    # Store results
                    results[impl_type][norm_type][beta] = {
                        'border_energy': {
                            'id_mean': id_border_mean,
                            'id_std': id_border_std,
                            'ood_mean': ood_border_mean,
                            'ood_std': ood_border_std,
                            'separation': ood_border_mean - id_border_mean,
                            'ks_stat': border_ks_stat,
                            'ks_p': border_ks_p,
                            'auroc': border_auc,
                            'aupr': border_aupr,
                            'fpr95': border_fpr95
                        },
                        'inference_score': {
                            'id_mean': id_score_mean,
                            'id_std': id_score_std,
                            'ood_mean': ood_score_mean,
                            'ood_std': ood_score_std,
                            'separation': sep,
                            'ks_stat': score_ks_stat,
                            'ks_p': score_ks_p,
                            'auroc': score_auc,
                            'aupr': score_aupr,
                            'fpr95': score_fpr95,
                            'is_ood_high': is_inference_ood_high
                        }
                    }
                    
                    # Test OOD loss
                    try:
                        id_small = id_subset[:2000]
                        ood_small = ood_subset[:2000]
                        imbalance_weights = [1.0, 2.0, 3.0, 5.0, 10.0]
                        loss_values = []
                        for weight in imbalance_weights:
                            with torch.enable_grad():
                                loss = compute_hopfield_ood_loss(
                                    id_pixels=id_small,
                                    ood_pixels=ood_small,
                                    device=device,
                                    imbalance_weight=weight,
                                    feature_norm_type=norm_type,
                                    beta=beta,
                                    impl_type=impl_type
                                )
                                loss_values.append(loss.item())
                        
                        logger.info(f"OOD Loss ({impl_type}, β={beta}, norm={norm_type}):")
                        for w, lv in zip(imbalance_weights, loss_values):
                            logger.info(f"  Weight {w}: {lv:.4f}")
                        
                        results[impl_type][norm_type][beta]['ood_loss'] = {
                            'imbalance_weights': imbalance_weights,
                            'loss_values': loss_values
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating OOD loss for β={beta}, norm={norm_type}, impl={impl_type}: {e}")
                        results[impl_type][norm_type][beta]['ood_loss'] = {'error': str(e)}
                    
                    # Clean up
                    del border_energy, inference_score
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error for beta={beta}, norm={norm_type}, impl={impl_type}: {e}")
                    logger.error(traceback.format_exc())
                    results[impl_type][norm_type][beta] = {'error': str(e)}
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'real_energy_analysis_{impl_type}_{norm_type}.png'), dpi=150)
            plt.close()
    
    # Summary plots
    for impl_type in impl_types:
        for norm_type in norm_types:
            beta_with_res = [b for b in beta_values if b in results[impl_type][norm_type] and 'error' not in results[impl_type][norm_type][b]]
            if beta_with_res:
                fig, axs = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Summary Metrics vs Beta ({impl_type}, Norm: {norm_type})', fontsize=16)
                
                border_seps = [results[impl_type][norm_type][b]['border_energy']['separation'] for b in beta_with_res]
                axs[0,0].plot(beta_with_res, border_seps, marker='o', label='Border Energy')
                axs[0,0].set_title('Border Separation')
                axs[0,0].set_xlabel('Beta')
                axs[0,0].set_ylabel('Separation (OOD - ID)')
                axs[0,0].grid(alpha=0.3)
                
                inf_seps = [results[impl_type][norm_type][b]['inference_score']['separation'] for b in beta_with_res]
                axs[0,1].plot(beta_with_res, inf_seps, marker='o', label='Inference Score')
                axs[0,1].set_title('Inference Separation')
                axs[0,1].set_xlabel('Beta')
                axs[0,1].set_ylabel('Separation')
                axs[0,1].grid(alpha=0.3)
                
                border_aurocs = [results[impl_type][norm_type][b]['border_energy']['auroc'] for b in beta_with_res]
                inf_aurocs = [results[impl_type][norm_type][b]['inference_score']['auroc'] for b in beta_with_res]
                axs[1,0].plot(beta_with_res, border_aurocs, marker='o', label='Border')
                axs[1,0].plot(beta_with_res, inf_aurocs, marker='o', label='Inference')
                axs[1,0].set_title('AUROC vs Beta')
                axs[1,0].set_xlabel('Beta')
                axs[1,0].set_ylabel('AUROC')
                axs[1,0].legend()
                axs[1,0].grid(alpha=0.3)
                
                border_fpr = [results[impl_type][norm_type][b]['border_energy']['fpr95'] for b in beta_with_res]
                inf_fpr = [results[impl_type][norm_type][b]['inference_score']['fpr95'] for b in beta_with_res]
                axs[1,1].plot(beta_with_res, border_fpr, marker='o', label='Border')
                axs[1,1].plot(beta_with_res, inf_fpr, marker='o', label='Inference')
                axs[1,1].set_title('FPR95 vs Beta')
                axs[1,1].set_xlabel('Beta')
                axs[1,1].set_ylabel('FPR at 95% TPR')
                axs[1,1].legend()
                axs[1,1].grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'beta_summary_{impl_type}_{norm_type}.png'), dpi=150)
                plt.close()
    
    # Find optimal
    optimal = {'impl': None, 'norm': None, 'beta': None, 'auroc': 0}
    for impl in impl_types:
        for norm in norm_types:
            for b in beta_values:
                if b in results[impl][norm] and 'error' not in results[impl][norm][b]:
                    auroc = results[impl][norm][b]['inference_score']['auroc']
                    if auroc > optimal['auroc']:
                        optimal = {'impl': impl, 'norm': norm, 'beta': b, 'auroc': auroc, 'fpr95': results[impl][norm][b]['inference_score']['fpr95']}
    
    if optimal['beta']:
        logger.info(f"Optimal config: impl={optimal['impl']}, norm={optimal['norm']}, beta={optimal['beta']} (AUROC: {optimal['auroc']:.4f}, FPR95: {optimal['fpr95']:.4f})")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('-l', '--local_rank', default=-1, type=int)
    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--norm_types', default='l2', type=str, help='Comma-separated norm types, e.g., l2,bn,pca_whitening')
    parser.add_argument('--impl_types', default='optimized,reference', type=str, help='Comma-separated impl types, e.g., optimized,reference')
    args = parser.parse_args()
    
    norm_types = args.norm_types.split(',')
    impl_types = args.impl_types.split(',')
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"energy_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup paths (update as needed)
    cs_root = "/home/ha51dybi/PEBAL/cityscapes"
    coco_root = "/home/ha51dybi/PEBAL/coco"
    model_path = "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth"
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Starting enhanced energy analysis on {device}")
    logger.info(f"Output: {output_dir}")
    
    if device.type == 'cuda':
        alloc, max_alloc = get_gpu_memory(device)
        logger.info(f"Initial GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")
    
    try:
        engine_instance = Engine(
            custom_arg=args, 
            logger=logger, 
            continue_state_object=model_path
        )
        
        memory_mode = "build"  # or "load"
        
        if memory_mode == "build":
            train_loader, _, _ = get_mix_loader(
                engine=engine_instance, 
                augment=True,
                cs_root=cs_root,
                coco_root=coco_root,
            )
            
            # Modified to use ASPP-only feature extractor
            feature_extractor = FeatureExtractor(
                model_path=model_path,
                resize_resolution=(512, 512),
                device=device,
                num_classes=19,
                hybrid=False,  # Set to False for ASPP-only
                aspp_select=None,  # Not needed for ASPP-only
                project_dim=None
            )
            
            # Modified to use 1280 input dimension for ASPP
            projection_head = SimpleProjectionHead(
                input_dim=1280,  # Changed from 1536 to 1280 for ASPP
                output_dim=256
            ).to(device)
            
            # Debug feature dimensions before creating memory builder
            with torch.no_grad():
                dummy_input = {'data': torch.randn(1, 3, 512, 512).to(device)}
                feature_extractor.eval()
                extracted = feature_extractor.extract_features_batch(dummy_input)
                features = extracted['features']
                logger.info(f"ASPP feature shape: {features.shape}")
                
                projection_head.eval()
                projected = projection_head(features)
                logger.info(f"Projected feature shape: {projected.shape}")
            
            logger.info("Creating memory builder...")
            memory_builder = RealMemoryBuilder(
                feature_extractor=feature_extractor,
                projection_pipeline=projection_head,
                device=device,
                id_memory_size=20000,  # Increased for better real data representation
                aux_memory_size=20000,
                num_in_dist_classes=19,
                log_level=logging.INFO,
                min_pixels_per_class_image=20
            )
            
            # Process more batches for real data
            subset_size = 200  
            subset_loader = [batch for i, batch in enumerate(train_loader) if i < subset_size]
            
            logger.info(f"Building memory with {len(subset_loader)} batches...")
            id_memory, ood_memory, analysis = memory_builder.process_images(subset_loader)
            
            if id_memory is not None and ood_memory is not None:
                torch.save(id_memory, os.path.join(output_dir, 'id_memory.pt'))
                torch.save(ood_memory, os.path.join(output_dir, 'ood_memory.pt'))
                logger.info(f"Memories saved")
                
                # Memory after build
                alloc, max_alloc = get_gpu_memory(device)
                logger.info(f"Post-memory_build GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")
        else:
            # Load paths update as needed
            id_memory = torch.load('/path/to/id_memory.pt', map_location=device)
            ood_memory = torch.load('/path/to/ood_memory.pt', map_location=device)
        
        if id_memory is not None and ood_memory is not None:
            logger.info("Testing with enhanced setup...")
            beta_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
            energy_results = test_energy_with_memories(
                id_memory=id_memory,
                ood_memory=ood_memory,
                device=device,
                output_dir=output_dir,
                beta_values=beta_values,
                norm_types=norm_types,
                impl_types=impl_types
            )
            torch.save(energy_results, os.path.join(output_dir, 'energy_results.pt'))
            
            # Optimal already logged
        else:
            logger.error("No memories available")
        
        logger.info(f"Enhanced analysis completed. Check {output_dir} for results.")
        logger.info("Metrics like high AUROC/low FPR95 indicate good OOD pixel separation for segmentation.")
        logger.info("KS-stat >0.5 with low p-value suggests strong distribution separation.")
        logger.info("Compare 'optimized' vs 'reference' to see which performs better.")
        
    except Exception as e:
        logger.error(f"Main error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            alloc, max_alloc = get_gpu_memory(device)
            logger.info(f"Final GPU mem: {alloc:.2f} MB, max {max_alloc:.2f} MB")

if __name__ == '__main__':
    main()
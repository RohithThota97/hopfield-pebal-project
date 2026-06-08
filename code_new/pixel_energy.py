import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import logging
logger = logging.getLogger(__name__)

def lse(beta: float, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """LSE function as in Hopfield Boosting paper."""
    if scores.numel() == 0:
        return torch.empty(0, device=scores.device, dtype=torch.float32)
    clamped_scores = torch.clamp(beta * scores, min=-200, max=200)
    max_scores = torch.max(clamped_scores, dim=dim, keepdim=True)[0]
    return (torch.logsumexp(clamped_scores - max_scores, dim=dim) + max_scores.squeeze(dim)) / beta

class PixelWiseBorderEnergy(nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, beta: float = 128.0,
                 feature_norm_type: str = 'l2', positive_shift: bool = False):  # Enabled positive_shift
        super().__init__()
        self.beta = beta
        self.positive_shift = positive_shift
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {a.shape}, {b.shape}")
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"Feature dims mismatch: {a.shape[1]} vs {b.shape[1]}")
        # Removed normalization - assume pre-normed
        self.register_buffer('id_memory', a.float())  # float32
        self.register_buffer('ood_memory', b.float())  # float32

    def forward(self, pixel_features: torch.Tensor, batch_size: int = 2048, pre_normalized: bool = True) -> torch.Tensor:  # Increased batch_size
        if pixel_features.numel() == 0:
            return torch.empty(0, device=pixel_features.device, dtype=torch.float32)
        orig_shape = None
        is_spatial = pixel_features.dim() > 2
        if is_spatial:
            orig_shape = pixel_features.shape
            B, C, H, W = pixel_features.shape
            pixel_features = pixel_features.permute(0, 2, 3, 1).reshape(-1, C)
        # Assume pre_normalized=True, no norm here
        num_pixels = pixel_features.shape[0]
        energies = torch.zeros(num_pixels, device=pixel_features.device, dtype=torch.float32)
        for i in range(0, num_pixels, batch_size):
            end_i = min(i + batch_size, num_pixels)  # Fixed: Use end_i to avoid gaps
            chunk = pixel_features[i:end_i]
            if chunk.size(0) == 0:
                continue
            id_sim = chunk @ self.id_memory.t()
            ood_sim = chunk @ self.ood_memory.t()
            lse_id = lse(self.beta, id_sim, dim=1)
            lse_ood = lse(self.beta, ood_sim, dim=1)
            beta_lse_id = self.beta * lse_id
            beta_lse_ood = self.beta * lse_ood
            max_beta_lse = torch.maximum(beta_lse_id, beta_lse_ood)
            exp_id = torch.exp(beta_lse_id - max_beta_lse)
            exp_ood = torch.exp(beta_lse_ood - max_beta_lse)
            lse_union = (max_beta_lse + torch.log(exp_id + exp_ood)) / self.beta
            border_energy_chunk = -2 * lse_union + lse_id + lse_ood
            if self.positive_shift:
                border_energy_chunk = -border_energy_chunk
            energies[i:end_i] = border_energy_chunk  # Fixed: Use end_i
        if orig_shape and is_spatial:
            energies = energies.view(orig_shape[0], orig_shape[2], orig_shape[3])
        return energies

class PixelWiseInferenceScore(nn.Module):
    """Inference score: s(ξ) = lse(β, O^T ξ) - lse(β, X^T ξ) (high for OOD, low for ID)."""
    def __init__(self, id_memory: torch.Tensor, ood_memory: torch.Tensor,
                 beta: float = 128.0, feature_norm_type: str = 'l2'):
        super().__init__()
        self.beta = beta
        if id_memory.dim() != 2 or ood_memory.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {id_memory.shape}, {ood_memory.shape}")
        if id_memory.shape[1] != ood_memory.shape[1]:
            raise ValueError(f"Feature dims mismatch: {id_memory.shape[1]} vs {ood_memory.shape[1]}")
        
        # Normalize memories for better separation
        id_normalized = F.normalize(id_memory, p=2, dim=1)
        ood_normalized = F.normalize(ood_memory, p=2, dim=1)
        
        self.register_buffer('id_memory', id_normalized.float())
        self.register_buffer('ood_memory', ood_normalized.float())

    def forward(self, pixel_features: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
        if pixel_features.numel() == 0:
            return torch.empty(0, device=pixel_features.device, dtype=torch.float32)
        
        orig_shape = None
        is_spatial = pixel_features.dim() > 2
        if is_spatial:
            orig_shape = pixel_features.shape
            B, C, H, W = pixel_features.shape
            pixel_features = pixel_features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize pixel features to match normalized memories
        pixel_features = F.normalize(pixel_features, p=2, dim=1)
        
        num_pixels = pixel_features.shape[0]
        scores = torch.zeros(num_pixels, device=pixel_features.device, dtype=torch.float32)
        
        for i in range(0, num_pixels, batch_size):
            end_i = min(i + batch_size, num_pixels)
            chunk = pixel_features[i:end_i]
            if chunk.size(0) == 0:
                continue
            
            id_sim = chunk @ self.id_memory.t()
            ood_sim = chunk @ self.ood_memory.t()
            
            lse_id = lse(self.beta, id_sim, dim=1)
            lse_ood = lse(self.beta, ood_sim, dim=1)
            
            # FIXED: OOD - ID (high for OOD, low for ID)
            score_chunk = lse_ood - lse_id
            scores[i:end_i] = score_chunk
        
        if orig_shape and is_spatial:
            scores = scores.view(orig_shape[0], orig_shape[2], orig_shape[3])
        
        return scores

def compute_hopfield_ood_loss(
    id_pixels: torch.Tensor,
    ood_pixels: torch.Tensor,
    device: torch.device,
    feature_norm_type: str = 'l2',
    lambda_ood: float = 5.0,
    beta: float = 128.0,
    positive_shift: bool = False,  # Enabled
    batch_size: Optional[int] = None,
    pre_normalized: bool = True,
    spatial_shape: Optional[Tuple[int, int]] = None,
    force_positive_loss: bool = False,  # New: Force non-negative loss
    **kwargs
) -> torch.Tensor:
    """L_OOD = border_energies.mean() (Hopfield Boosting paper). Using total memory without subsampling. Removed TV."""
    # Handle empty inputs early
    if id_pixels.numel() == 0 or ood_pixels.numel() == 0:
        logger.warning("Empty ID/OOD pixels in batch - skipping loss")
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    # Handle spatial inputs
    id_orig_shape = None
    ood_orig_shape = None
    id_is_spatial = id_pixels.dim() > 2
    ood_is_spatial = ood_pixels.dim() > 2
    if id_is_spatial:
        id_orig_shape = id_pixels.shape
        B_id, C, H_id, W_id = id_pixels.shape
        id_pixels_flat = id_pixels.permute(0, 2, 3, 1).reshape(-1, C)
    else:
        id_pixels_flat = id_pixels
    if ood_is_spatial:
        ood_orig_shape = ood_pixels.shape
        B_ood, C, H_ood, W_ood = ood_pixels.shape
        ood_pixels_flat = ood_pixels.permute(0, 2, 3, 1).reshape(-1, C)
    else:
        ood_pixels_flat = ood_pixels
    # Check dims after flattening
    if id_pixels_flat.dim() != 2 or ood_pixels_flat.dim() != 2:
        raise ValueError(f"Expected 2D tensors after flattening, got {id_pixels_flat.shape}, {ood_pixels_flat.shape}")
    if id_pixels_flat.shape[1] != ood_pixels_flat.shape[1]:
        raise ValueError(f"Feature dims mismatch after flatten: {id_pixels_flat.shape[1]} vs {ood_pixels_flat.shape[1]}")
    num_id = id_pixels_flat.shape[0]
    num_ood = ood_pixels_flat.shape[0]
    if num_id < 1 or num_ood < 1:  
        logger.warning("Insufficient ID/OOD pixels - skipping loss")
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    all_pixels = torch.cat([id_pixels_flat, ood_pixels_flat], dim=0).to(device).float()  # float32
    try:
        energy_calc = PixelWiseBorderEnergy(id_pixels_flat, ood_pixels_flat, beta=beta,  # Removed cap
                                            feature_norm_type=feature_norm_type,
                                            positive_shift=positive_shift).to(device)
    except Exception as e:
        logger.error(f"Error creating energy calculator: {e}")
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    try:
        internal_batch_size = batch_size if batch_size is not None else 2048  # Increased for more pixels
        border_energies_flat = energy_calc(all_pixels, batch_size=internal_batch_size, pre_normalized=pre_normalized)
    except Exception as e:
        logger.error(f"Error computing border energies: {e}")
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    # Handle empty output
    if border_energies_flat.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    border_energies = torch.nan_to_num(border_energies_flat, nan=0.0, posinf=1.0, neginf=-1.0)
    border_energies = torch.clamp(border_energies, min=-100.0, max=100.0)
    num_id = id_pixels_flat.shape[0]
    id_energies = border_energies[:num_id]
    ood_energies = border_energies[num_id:]
    if id_energies.numel() == 0 or ood_energies.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
    id_mean = id_energies.mean().item()
    ood_mean = ood_energies.mean().item()
    logger.info(f"Energy stats: β={beta}, id_mean={id_mean:.4f}, ood_mean={ood_mean:.4f}")
    ood_loss = border_energies.mean()  # Core loss (pure Hopfield)
   
    total_loss = ood_loss * lambda_ood  # No TV/smoothness
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning("Invalid final loss, using fallback")
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    return total_loss
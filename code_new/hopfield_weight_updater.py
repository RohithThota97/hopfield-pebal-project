import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import time
import logging


from pixel_energy import PixelWiseBorderEnergy,PixelWiseInferenceScore,compute_hopfield_ood_loss

logger = logging.getLogger(__name__)

class HopfieldBoostingManager:
    def __init__(self, id_features_full: torch.Tensor, aux_features_full: torch.Tensor,
                 beta_sampling: float = 128.0, lambda_ood: float = 5.0, device: Optional[torch.device] = None,
                 memory_subset_size: int = 100000, positive_shift: bool = True, num_boosting_iters: int = 3) -> None:
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta_sampling
        self.lambda_ood = lambda_ood
        self.memory_subset_size = memory_subset_size
        self.positive_shift = positive_shift
        self.num_boosting_iters = num_boosting_iters
        self.feature_dim = id_features_full.shape[1] if id_features_full.numel() > 0 else 128

        if id_features_full.shape[1] != self.feature_dim or aux_features_full.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.feature_dim}")
        self.id_features_full = id_features_full.to(self.device)
        self.aux_features_full = aux_features_full.to(self.device)
        self.aux_sampling_weights = torch.ones(self.aux_features_full.shape[0], device=self.device) / self.aux_features_full.shape[0] if len(self.aux_features_full) > 0 else torch.empty(0, device=self.device)
        self.epoch_count = 0

        logger.info(f"HopfieldBoostingManager initialized for AUX boosting with beta={self.beta}, iters={self.num_boosting_iters}")

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.id_features_full) == 0:
            id_batch = torch.zeros((batch_size, self.feature_dim), device=self.device)
        else:
            id_indices = torch.randint(0, len(self.id_features_full), (batch_size,), device=self.device)
            id_batch = self.id_features_full[id_indices]
        if len(self.aux_features_full) == 0:
            aux_batch = torch.zeros((batch_size, self.feature_dim), device=self.device)
        else:
            aux_indices = torch.multinomial(self.aux_sampling_weights, batch_size, replacement=True)
            aux_batch = self.aux_features_full[aux_indices]
        id_batch = torch.nan_to_num(id_batch, nan=0.0)
        aux_batch = torch.nan_to_num(aux_batch, nan=0.0)
        id_batch = F.normalize(id_batch, dim=1)
        aux_batch = F.normalize(aux_batch, dim=1)
        return id_batch, aux_batch

    def update_sampling_weights(self, memory_size: Optional[int] = None) -> None:
        # Removed the epoch frequency check - now updates every epoch
        memory_size = memory_size if memory_size is not None else self.memory_subset_size
        id_mem_subset, aux_mem_subset = self._get_memory_subset(memory_size)
        if id_mem_subset.numel() == 0 or aux_mem_subset.numel() == 0:
            logger.warning("Invalid subsets - skipping AUX weight update")
            return
        energy_calc = PixelWiseBorderEnergy(id_mem_subset, aux_mem_subset, self.beta, positive_shift=self.positive_shift).to(self.device)
        batch_size = 1024
        energies = []
        with torch.no_grad():
            for i in range(0, len(self.aux_features_full), batch_size):
                batch = self.aux_features_full[i:i+batch_size]
                batch_energies = energy_calc(batch)
                energies.append(batch_energies)
        energies = torch.cat(energies)
        energies = torch.nan_to_num(energies, nan=0.0, posinf=0.0, neginf=0.0)
        self.aux_sampling_weights = F.softmax(self.beta * energies, dim=0)
        logger.info("AUX weights updated for boosting")

    def _get_memory_subset(self, subset_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        subset_size = subset_size if subset_size is not None else self.memory_subset_size
        id_size = min(subset_size, len(self.id_features_full))
        aux_size = min(subset_size, len(self.aux_features_full))
        if len(self.id_features_full) == 0:
            id_subset = torch.empty(0, self.feature_dim, device=self.device)
        else:
            id_indices = torch.randperm(len(self.id_features_full), device=self.device)[:id_size]
            id_subset = self.id_features_full[id_indices]
        if len(self.aux_features_full) == 0:
            aux_subset = torch.empty(0, self.feature_dim, device=self.device)
        else:
            replacement = aux_size > len(self.aux_features_full)
            aux_indices = torch.multinomial(self.aux_sampling_weights, aux_size, replacement=replacement)
            aux_subset = self.aux_features_full[aux_indices]
        return id_subset, aux_subset

    def compute_boosted_ood_loss(self, ood_pixels: torch.Tensor, id_pixels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if id_pixels is None:
            id_pixels = self.id_features_full[torch.randperm(len(self.id_features_full))[:min(1000, len(self.id_features_full))]] if len(self.id_features_full) > 0 else torch.empty(0, self.feature_dim, device=self.device)
        max_id_pixels = 1000
        max_ood_pixels = 1000
        if len(id_pixels) > max_id_pixels:
            id_pixels = id_pixels[torch.randperm(len(id_pixels))[:max_id_pixels]]
        if len(ood_pixels) > max_ood_pixels:
            ood_pixels = ood_pixels[torch.randperm(len(ood_pixels))[:max_ood_pixels]]
        if ood_pixels.dim() > 2:
            _, C, H, W = ood_pixels.shape
            ood_pixels = ood_pixels.permute(0, 2, 3, 1).reshape(-1, C)
            spatial_shape = (H, W)
        else:
            spatial_shape = None
        if ood_pixels.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dim mismatch for OOD pixels")
        num_pixels = ood_pixels.shape[0]
        if num_pixels == 0:
            return torch.tensor(0.0, device=self.device)
        total_loss = 0.0
        for iter in range(self.num_boosting_iters):
            id_batch, aux_batch = self.sample_batch(num_pixels)
            boosted_id = torch.cat([id_pixels.to(self.device), id_batch], dim=0)
            boosted_ood = torch.cat([ood_pixels.to(self.device), aux_batch], dim=0)
            context = torch.cuda.amp.autocast(enabled=True) if self.device.type == 'cuda' else torch.no_grad()
            with context:
                iter_loss = compute_hopfield_ood_loss(
                    boosted_id, boosted_ood, device=self.device,
                    lambda_ood=self.lambda_ood, beta=self.beta,
                    positive_shift=self.positive_shift,
                    batch_size=512,
                    pre_normalized=True,
                    spatial_shape=spatial_shape
                )
            total_loss += iter_loss
        avg_loss = total_loss / self.num_boosting_iters
        logger.info(f"AUX boosting loss (avg over {self.num_boosting_iters} iters): {avg_loss.item():.4f}")
        return avg_loss

    def advance_epoch(self, current_epoch: int, total_epochs: int = 100, update_freq: int = 1) -> None:
        self.epoch_count += 1
        # Always update weights every epoch when advance_epoch is called
        self.update_sampling_weights()
        logger.info(f"Epoch {self.epoch_count}: AUX boosting advanced")
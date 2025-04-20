# hopfield_pebal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging # Import logging

# Get logger instance (might be configured by main script)
logger = logging.getLogger(__name__)

if not logger.hasHandlers(): # Add handler if none exists
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class HopfieldPEBALLoss(nn.Module):
    """
    Combined loss for PEBAL-like OOD detection with Hopfield memory energy.
    Calculates segmentation loss, energy-based OOD loss, and potentially
    a Hopfield contrastive loss (currently placeholder).

    Expects targets to contain values in [0, num_classes-1] or ignore_index.
    """
    def __init__(self,
                 num_classes: int,
                 seg_weight: float = 1.0,
                 energy_weight: float = 0.1,
                 hopfield_weight: float = 0.0, # Placeholder weight for Hopfield contrastive loss
                 inlier_margin: float = 1.0,
                 outlier_margin: float = 10.0,
                 temperature: float = 1.0, # Temperature for PEBAL energy scaling (used in model, not directly here)
                 ignore_index: int = 255,
                 use_combined_energy: bool = True # Use combined energy from model if available
                 ):
        super().__init__()
        # Input validation
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if seg_weight < 0 or energy_weight < 0 or hopfield_weight < 0:
             logger.warning("Loss weights should ideally be non-negative.")
        if inlier_margin < 0 or outlier_margin < 0:
             logger.warning("Energy margins should ideally be non-negative.")

        self.num_classes = num_classes
        self.seg_weight = seg_weight
        self.energy_weight = energy_weight
        self.hopfield_weight = hopfield_weight
        self.inlier_margin = inlier_margin
        self.outlier_margin = outlier_margin
        # self.temperature = temperature # Temperature is applied in the model's energy calculation logic
        self.ignore_index = ignore_index
        self.use_combined_energy = use_combined_energy

        # Use CrossEntropyLoss with reduction='none' to calculate per-pixel loss
        # Ensure ignore_index is correctly passed
        # **CRITICAL:** Ensure num_classes matches the range of valid IDs in the target masks (e.g., 19 for Cityscapes trainIds 0-18)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        logger.info(f"Initialized CrossEntropyLoss with ignore_index={self.ignore_index} for {self.num_classes} classes.")

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Optional[torch.Tensor],
                ood_images: Optional[torch.Tensor] = None, # <-- ACCEPTED ARGUMENT
                model: Optional[nn.Module] = None         # <-- ACCEPTED ARGUMENT
                ) -> Dict[str, torch.Tensor]:
        """
        Calculate the combined loss.

        Args:
            outputs (Dict[str, torch.Tensor]): Dictionary from the model, expected keys:
                'seg_logits': Logits [B, C, H, W] (Note: C should be num_classes, not C+1).
                'combined_energy': Final energy score [B, 1, H, W] (preferred).
                'pebal_energy', 'memory_energy', 'feature_energy' (optional for logging/alternatives).
                'is_ood' (Optional): Boolean tensor [B] indicating OOD samples in the batch.
            targets (Optional[torch.Tensor]): Ground truth segmentation masks [B, H, W].
                                           Should contain values in [0, num_classes-1] or ignore_index.
                                           Should be None or contain ignore_index for OOD samples.
                                           Expected dtype: torch.long.
            ood_images (Optional[torch.Tensor]): Separate batch of OOD images [B_ood, 3, H, W].
                                                 Used if energy needs calculation within the loss.
            model (Optional[nn.Module]): The model instance, needed to calculate energy
                                         for `ood_images`.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss components:
                                     'total_loss', 'seg_loss', 'energy_loss', 'hopfield_loss'.
                                     Returns dict with zero losses if critical inputs are missing.
        """
        # --- Input Validation ---
        if 'seg_logits' not in outputs:
             logger.error("Loss calculation failed: 'seg_logits' missing from model outputs.")
             zero = torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
             return {'total_loss': zero, 'seg_loss': zero, 'energy_loss': zero, 'hopfield_loss': zero}
        if self.energy_weight > 0 and 'combined_energy' not in outputs and ood_images is None:
             logger.warning("'combined_energy' missing and no ood_images provided. Cannot calculate energy loss component from OOD images.")
             # Proceed without energy loss from external OOD images

        device = outputs['seg_logits'].device
        losses = {
            'total_loss': torch.tensor(0.0, device=device),
            'seg_loss': torch.tensor(0.0, device=device),
            'energy_loss': torch.tensor(0.0, device=device),
            'hopfield_loss': torch.tensor(0.0, device=device) # Placeholder
        }

        seg_logits = outputs['seg_logits']
        batch_size = seg_logits.shape[0]

        # Ensure logits have the correct number of channels (num_classes, not num_classes+1)
        if seg_logits.shape[1] != self.num_classes:
            logger.error(f"Loss Error: seg_logits channel dimension ({seg_logits.shape[1]}) does not match num_classes ({self.num_classes}).")
            zero = torch.tensor(0.0, device=device)
            return {'total_loss': zero, 'seg_loss': zero, 'energy_loss': zero, 'hopfield_loss': zero}


        # Determine which samples are In-Distribution (ID) and Out-of-Distribution (OOD) within the batch
        is_ood_mask_batch = outputs.get('is_ood', torch.zeros(batch_size, dtype=torch.bool, device=device))
        id_indices_batch = torch.where(~is_ood_mask_batch)[0]
        ood_indices_batch = torch.where(is_ood_mask_batch)[0]

        # --- 1. Segmentation Loss (only for ID samples in the batch) ---
        if self.seg_weight > 0 and len(id_indices_batch) > 0 and targets is not None:
            id_logits = seg_logits[id_indices_batch]
            id_targets = None
            if targets.shape[0] == batch_size: # Targets provided for the whole batch
                 id_targets = targets[id_indices_batch]
            elif targets.shape[0] == len(id_indices_batch): # Targets provided only for ID samples
                 id_targets = targets
            else:
                 logger.error(f"Target shape {targets.shape} incompatible with ID indices count {len(id_indices_batch)}. Skipping seg loss.")

            if id_targets is not None:
                # Ensure targets are Long type
                if id_targets.dtype != torch.long:
                     logger.warning(f"Targets dtype is {id_targets.dtype}, converting to torch.long for CrossEntropy.")
                     id_targets = id_targets.long()

                # Calculate per-pixel cross-entropy loss
                try:
                     # Validate dimensions
                     if id_logits.ndim != 4 or id_targets.ndim != 3:
                          raise ValueError(f"Incorrect dimensions for CE Loss. Logits: {id_logits.shape}, Targets: {id_targets.shape}")
                     if id_logits.shape[0] != id_targets.shape[0] or id_logits.shape[2:] != id_targets.shape[1:]:
                          raise ValueError(f"Mismatched shapes for CE Loss. Logits: {id_logits.shape}, Targets: {id_targets.shape}")

                     # **CRITICAL CHECK (Optional but recommended for debugging):**
                     # unique_targets = torch.unique(id_targets)
                     # invalid_targets = unique_targets[(unique_targets != self.ignore_index) & ((unique_targets < 0) | (unique_targets >= self.num_classes))]
                     # if len(invalid_targets) > 0:
                     #     logger.error(f"!!!!!!!! Invalid target values detected BEFORE CE loss: {invalid_targets.tolist()}. Num_classes={self.num_classes}, Ignore={self.ignore_index}")
                     #     # Consider raising an error here or handling it upstream
                     #     # raise ValueError(f"Invalid target values found: {invalid_targets}")

                     pixel_seg_loss = self.ce_loss(id_logits, id_targets) # Output shape: [B_id, H, W]

                     # Average loss over valid pixels (those not ignored)
                     valid_mask = (id_targets != self.ignore_index)
                     num_valid_pixels = valid_mask.sum()

                     if num_valid_pixels > 0:
                          # Apply mask and calculate mean only on valid pixels
                          seg_loss_val = torch.sum(pixel_seg_loss * valid_mask.float()) / num_valid_pixels

                          if not torch.isnan(seg_loss_val).item() and not torch.isinf(seg_loss_val).item():
                              losses['seg_loss'] = seg_loss_val * self.seg_weight
                              losses['total_loss'] += losses['seg_loss']
                          else:
                               logger.warning("NaN/Inf detected in segmentation loss value. Setting seg_loss to 0.")
                               losses['seg_loss'] = torch.tensor(0.0, device=device)
                     else:
                          losses['seg_loss'] = torch.tensor(0.0, device=device) # No valid pixels
                          logger.debug("Seg loss is 0 because all target pixels were ignored.")

                except RuntimeError as e:
                     # Catch device-side asserts specifically if possible, though they often raise later
                     if "CUDA error: device-side assert triggered" in str(e):
                         logger.error(f"CUDA device-side assert triggered during segmentation loss calculation. This usually means target labels are out of bounds [0, {self.num_classes-1}] and not ignore_index ({self.ignore_index}). Check dataset preprocessing/label mapping.", exc_info=False)
                         # Print target stats to help debug
                         unique_targets = torch.unique(id_targets)
                         logger.error(f"Unique target values in problematic batch: {unique_targets.tolist()}")
                         min_target, max_target = id_targets.min().item(), id_targets.max().item()
                         logger.error(f"Min/Max target values: {min_target}/{max_target}")
                     else:
                         logger.error(f"Error calculating segmentation loss: {e}", exc_info=True)
                     losses['seg_loss'] = torch.tensor(0.0, device=device) # Assign 0 on error
                except Exception as e:
                     logger.error(f"Unhandled error calculating segmentation loss: {e}", exc_info=True)
                     losses['seg_loss'] = torch.tensor(0.0, device=device) # Assign 0 on error


        # --- 2. Energy-based OOD Loss ---
        if self.energy_weight > 0:
            energy_loss_val = torch.tensor(0.0, device=device)
            combined_energy = outputs.get('combined_energy')

            if combined_energy is None:
                logger.warning("'combined_energy' not found in model outputs. Cannot calculate energy loss from batch outputs.")
            else:
                # Energy loss for ID samples in the batch
                if len(id_indices_batch) > 0:
                    id_energy = combined_energy[id_indices_batch]
                    # Margin loss: max(0, energy - margin) for in-distribution
                    id_energy_loss = torch.relu(id_energy - self.inlier_margin).mean()
                    if not torch.isnan(id_energy_loss).item(): energy_loss_val += id_energy_loss

                # Energy loss for OOD samples *in the batch*
                if len(ood_indices_batch) > 0:
                    ood_energy_batch = combined_energy[ood_indices_batch]
                    # Margin loss: max(0, margin - energy) for out-of-distribution
                    ood_energy_loss_batch = torch.relu(self.outlier_margin - ood_energy_batch).mean()
                    if not torch.isnan(ood_energy_loss_batch).item(): energy_loss_val += ood_energy_loss_batch

            # Energy loss for separately provided OOD images
            if ood_images is not None and model is not None:
                if ood_images.shape[0] > 0:
                    try:
                        original_mode = model.training
                        if original_mode: model.eval()

                        with torch.no_grad():
                            ood_images_dev = ood_images.to(device)
                            ood_outputs = model(ood_images_dev)
                            ood_energy_calc = ood_outputs.get('combined_energy')

                        if ood_energy_calc is not None:
                            ood_energy_loss_provided = torch.relu(self.outlier_margin - ood_energy_calc).mean()
                            if not torch.isnan(ood_energy_loss_provided).item(): energy_loss_val += ood_energy_loss_provided
                        else:
                            logger.warning("OOD images provided, but 'combined_energy' not found in their output during loss calculation.")

                        if original_mode: model.train()
                        del ood_images_dev, ood_outputs, ood_energy_calc
                    except Exception as e:
                         logger.error(f"Error calculating energy for provided OOD images: {e}", exc_info=True)
                         if 'original_mode' in locals() and original_mode: model.train()

            # Apply weight to the accumulated energy loss
            losses['energy_loss'] = energy_loss_val * self.energy_weight
            if not torch.isnan(losses['energy_loss']).item() and not torch.isinf(losses['energy_loss']).item():
                losses['total_loss'] += losses['energy_loss']
            else:
                 logger.warning("NaN/Inf detected in energy loss. Excluding from total_loss.")
                 losses['energy_loss'] = torch.tensor(0.0, device=device)


        # --- 3. Hopfield Contrastive Loss (Placeholder) ---
        losses['hopfield_loss'] = torch.tensor(0.0, device=device) # Explicitly set to 0


        # --- Final Check ---
        if torch.isnan(losses['total_loss']).item() or torch.isinf(losses['total_loss']).item():
            logger.error(f"NaN/Inf detected in final total_loss (Seg: {losses['seg_loss']:.4f}, Energy: {losses['energy_loss']:.4f}). Returning zero loss.")
            zero_loss = torch.tensor(0.0, device=device)
            return {'total_loss': zero_loss, 'seg_loss': zero_loss, 'energy_loss': zero_loss, 'hopfield_loss': zero_loss}

        return losses
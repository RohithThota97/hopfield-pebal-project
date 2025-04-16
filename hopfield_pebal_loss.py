import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class HopfieldPEBALLoss(nn.Module):
    """
    Loss function for Hopfield-PEBAL (Progressive Energy-Based Anomaly Learning) models,
    combining segmentation loss with energy-based OOD detection and Hopfield energy regularization.
    Includes progressive scaling for energy loss weights during training.

    The energy loss aims to push the energy of inlier samples below `inlier_margin`
    and the energy of outlier samples above `outlier_margin`.
    The Hopfield energy loss aims to minimize the Hopfield energy for inliers and
    push the Hopfield energy for outliers above `outlier_margin`.
    """
    def __init__(self,
                 num_classes: int,
                 seg_weight: float = 1.0,
                 energy_weight_init: float = 0.01, # Renamed for clarity
                 hopfield_weight_init: float = 0.01, # Renamed for clarity
                 energy_weight_max: float = 0.1, # Reduced max weight
                 hopfield_weight_max: float = 0.1, # Reduced max weight
                 inlier_margin: float = 1.0,
                 outlier_margin: float = 10.0,
                 temperature: float = 1.0, # Currently unused, but kept for potential future use
                 progressive_scaling: bool = True,
                 progressive_scaling_epochs: int = 15, # Epochs to reach max weights
                 ignore_index: int = 255,
                 energy_clip_value: float = 20.0, # Clipping value for energy stability
                 epsilon: float = 1e-6): # Small value for numerical stability
        """
        Initializes the HopfieldPEBALLoss module.

        Args:
            num_classes: Number of segmentation classes (excluding potential outlier/ignore class).
            seg_weight: Fixed weight for the segmentation loss component.
            energy_weight_init: Initial weight for the combined energy loss.
            hopfield_weight_init: Initial weight for the Hopfield energy loss.
            energy_weight_max: Maximum weight for the combined energy loss during progressive scaling.
            hopfield_weight_max: Maximum weight for the Hopfield energy loss during progressive scaling.
            inlier_margin: Target maximum energy for inlier samples.
            outlier_margin: Target minimum energy for outlier samples.
            temperature: Temperature scaling for energy (currently not used in calculation).
            progressive_scaling: If True, linearly increase energy weights from init to max
                                 over `progressive_scaling_epochs`.
            progressive_scaling_epochs: Number of epochs over which to scale the energy weights.
            ignore_index: Target value to ignore in segmentation loss (e.g., void pixels).
            energy_clip_value: Value to clip energies to [-clip, +clip] for stability.
            epsilon: Small constant added for numerical stability in ReLU calculations.
        """
        super().__init__()

        if not (0 <= energy_weight_init <= energy_weight_max):
            raise ValueError("energy_weight_init must be between 0 and energy_weight_max")
        if not (0 <= hopfield_weight_init <= hopfield_weight_max):
            raise ValueError("hopfield_weight_init must be between 0 and hopfield_weight_max")
        if progressive_scaling_epochs <= 0 and progressive_scaling:
             warnings.warn("progressive_scaling_epochs is <= 0 but progressive_scaling is True. "
                           "Weights will jump directly to max.")
             progressive_scaling_epochs = 1 # Avoid division by zero

        self.num_classes = num_classes
        self.seg_weight = seg_weight
        self.energy_weight_init = energy_weight_init
        self.hopfield_weight_init = hopfield_weight_init
        self.energy_weight_max = energy_weight_max
        self.hopfield_weight_max = hopfield_weight_max
        self.inlier_margin = inlier_margin
        self.outlier_margin = outlier_margin
        self.temperature = temperature # Retained but unused in current logic
        self.progressive_scaling = progressive_scaling
        self.progressive_scaling_epochs = progressive_scaling_epochs
        self.ignore_index = ignore_index
        self.energy_clip_value = energy_clip_value
        self.epsilon = epsilon

        # Use ignore_index for segmentation loss
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Internal state for progressive scaling
        self._current_epoch = 0

    def update_epoch(self, epoch: int):
        """
        Updates the current epoch number, used for progressive scaling calculation.

        Args:
            epoch: The current training epoch (0-based).
        """
        self._current_epoch = epoch

    def _calculate_current_weights(self) -> tuple[float, float]:
        """Calculates the energy and hopfield weights for the current epoch."""
        if not self.progressive_scaling:
            return self.energy_weight_init, self.hopfield_weight_init # Use initial if not scaling

        # Calculate progress, capped at 1.0
        progress = min(1.0, self._current_epoch / max(1, self.progressive_scaling_epochs))

        current_energy_weight = self.energy_weight_init + \
                                (self.energy_weight_max - self.energy_weight_init) * progress
        current_hopfield_weight = self.hopfield_weight_init + \
                                  (self.hopfield_weight_max - self.hopfield_weight_init) * progress

        return current_energy_weight, current_hopfield_weight

    def _compute_energy_loss(self,
                             energy: torch.Tensor,
                             inlier_mask: torch.Tensor,
                             outlier_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the margin-based energy loss for inliers and outliers.

        Args:
            energy: Energy map tensor [B, 1, H, W].
            inlier_mask: Boolean mask for inlier pixels [B, 1, H, W].
            outlier_mask: Boolean mask for outlier pixels [B, 1, H, W].

        Returns:
            Tuple containing:
            - inlier_loss: Loss for inlier samples.
            - outlier_loss: Loss for outlier samples.
            - total_energy_loss: Sum of inlier and outlier losses.
        """
        # Clip energy for stability before applying margin loss
        energy_clipped = torch.clamp(energy, min=-self.energy_clip_value, max=self.energy_clip_value)

        inlier_loss = torch.tensor(0.0, device=energy.device, dtype=energy.dtype)
        if torch.any(inlier_mask):
            inlier_energy = energy_clipped[inlier_mask] # Select inlier energies
            # Loss encourages energy < inlier_margin
            inlier_loss = torch.mean(F.relu(inlier_energy - self.inlier_margin + self.epsilon))

        outlier_loss = torch.tensor(0.0, device=energy.device, dtype=energy.dtype)
        if torch.any(outlier_mask):
            outlier_energy = energy_clipped[outlier_mask] # Select outlier energies
            # Loss encourages energy > outlier_margin
            outlier_loss = torch.mean(F.relu(self.outlier_margin - outlier_energy + self.epsilon))

        total_energy_loss = inlier_loss + outlier_loss
        return inlier_loss, outlier_loss, total_energy_loss


    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Computes the combined Hopfield-PEBAL loss.

        Args:
            outputs: Dictionary of model outputs, expected to contain:
                - 'logits': Raw segmentation logits [B, C, H, W].
                - 'combined_energy': Combined energy map [B, 1, H, W].
                - 'hopfield_energy': (Optional) Hopfield energy map [B, 1, H, W].
            targets: Ground truth segmentation mask [B, H, W], where `ignore_index`
                     indicates pixels to ignore (often void), and potentially another
                     value (e.g., num_classes) indicates OOD pixels if known.

        Returns:
            Dictionary containing loss components and the total loss:
            - 'total_loss': The final weighted combined loss.
            - 'seg_loss': Segmentation loss component (unweighted).
            - 'energy_loss': Combined energy loss component (unweighted).
            - 'inlier_energy_loss': Inlier part of energy loss (unweighted).
            - 'outlier_energy_loss': Outlier part of energy loss (unweighted).
            - 'hopfield_loss': Hopfield energy loss component (unweighted, if applicable).
            - 'hopfield_inlier_loss': Inlier part of Hopfield loss (unweighted, if applicable).
            - 'hopfield_outlier_loss': Outlier part of Hopfield loss (unweighted, if applicable).
            - 'energy_weight': Current weight used for combined energy loss.
            - 'hopfield_weight': Current weight used for Hopfield energy loss.
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=targets.device, dtype=torch.float32)

        # --- 1. Prepare Inputs and Masks ---
        logits = outputs.get('logits')
        if logits is None:
            raise KeyError("Outputs dictionary must contain 'logits' tensor.")

        # Check for NaN/Inf in logits early
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            warnings.warn("NaN or Inf detected in input logits. Replacing with zeros.")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                 torch.zeros_like(logits), logits)

        # Ensure target spatial dimensions match logits
        if logits.shape[-2:] != targets.shape[-2:]:
            targets_resampled = F.interpolate(targets.float().unsqueeze(1),
                                             size=logits.shape[-2:],
                                             mode='nearest').squeeze(1).long()
        else:
            targets_resampled = targets

        # Define outlier mask: Pixels marked with ignore_index are typically *not* outliers,
        # they are just ignored for segmentation loss calculation.
        # Outliers are often implicitly defined (pixels not belonging to known classes)
        # or explicitly marked with a specific index (e.g., num_classes).
        # Here, we assume pixels *not* marked with ignore_index are potentially inliers or outliers.
        # Let's refine the mask definition:
        # - `valid_mask`: Pixels not ignored for segmentation loss (target != ignore_index)
        # - `outlier_mask`: Pixels marked as outliers (e.g., target == num_classes or a specific OOD value)
        # - `inlier_mask`: Pixels that are valid and *not* outliers.

        # For this implementation, let's stick to the original logic where ignore_index
        # implies OOD/ignored for energy loss, unless a specific OOD class index is used.
        # If OOD is marked by index num_classes, use that. Otherwise, use ignore_index.
        # A common convention is to use ignore_index (e.g., 255) for void, not OOD.
        # Let's assume OOD pixels are *not* explicitly marked in `targets` and are inferred
        # from the energy. The loss uses *all* valid pixels (not ignore_index)
        # and pushes their energy based on whether they *should* be inliers or outliers.
        # The provided `outlier_mask` in the original code implies *known* OOD pixels.
        # Let's refine to handle this:
        valid_pixel_mask = (targets_resampled != self.ignore_index)

        # Default: assume all valid pixels are inliers unless specified otherwise
        # In many OOD segmentation setups, you don't have ground truth OOD masks during training.
        # The energy loss forces *all* valid pixels towards the inlier margin, relying on
        # model capacity and potentially separate OOD samples/augmentation for outlier signal.
        # If true OOD GT is available (e.g., `targets == num_classes`), use it.
        # Let's assume for now that OOD pixels are marked with `self.ignore_index`
        # for the purpose of energy loss, matching the original code's inferred behavior.
        # This might need adjustment based on the specific dataset/task.
        outlier_mask_bool = (targets_resampled == self.ignore_index) # Treat ignored pixels as outliers for energy
        inlier_mask_bool = valid_pixel_mask # Treat all non-ignored pixels as inliers for energy

        # --- 2. Segmentation Loss ---
        # Compute seg loss only on valid (non-ignored) pixels
        # Note: CrossEntropyLoss with ignore_index handles this internally.
        seg_loss = self.seg_loss_fn(logits, targets_resampled)
        if torch.isnan(seg_loss):
             warnings.warn("NaN detected in segmentation loss. Setting to 0.")
             seg_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

        losses['seg_loss'] = seg_loss.detach() # Store unweighted, detached loss
        total_loss += self.seg_weight * seg_loss

        # --- 3. Calculate Current Weights ---
        current_energy_weight, current_hopfield_weight = self._calculate_current_weights()
        losses['energy_weight'] = torch.tensor(current_energy_weight, device=total_loss.device)
        losses['hopfield_weight'] = torch.tensor(current_hopfield_weight, device=total_loss.device)

        # --- 4. Combined Energy Loss ---
        combined_energy = outputs.get('combined_energy')
        if combined_energy is not None and current_energy_weight > 0:
            # Ensure energy map dimensions match targets/masks
            if combined_energy.shape[-2:] != inlier_mask_bool.shape[-2:]:
                 combined_energy = F.interpolate(combined_energy,
                                                 size=inlier_mask_bool.shape[-2:],
                                                 mode='bilinear',
                                                 align_corners=False)

            # Add channel dimension to masks: [B, H, W] -> [B, 1, H, W]
            inlier_mask_energy = inlier_mask_bool.unsqueeze(1)
            outlier_mask_energy = outlier_mask_bool.unsqueeze(1)

            # Check energy stability
            if torch.isnan(combined_energy).any() or torch.isinf(combined_energy).any():
                warnings.warn("NaN or Inf detected in combined_energy. Replacing with zeros.")
                combined_energy = torch.where(torch.isnan(combined_energy) | torch.isinf(combined_energy),
                                              torch.zeros_like(combined_energy), combined_energy)

            # Compute energy loss components
            inlier_loss, outlier_loss, energy_loss = self._compute_energy_loss(
                combined_energy, inlier_mask_energy, outlier_mask_energy
            )

            if torch.isnan(energy_loss):
                 warnings.warn("NaN detected in combined energy loss. Skipping energy loss term.")
            else:
                losses['inlier_energy_loss'] = inlier_loss.detach()
                losses['outlier_energy_loss'] = outlier_loss.detach()
                losses['energy_loss'] = energy_loss.detach() # Store unweighted loss
                total_loss += current_energy_weight * energy_loss
        else:
             # Assign zero tensors if energy loss is skipped
             losses['inlier_energy_loss'] = torch.tensor(0.0, device=total_loss.device)
             losses['outlier_energy_loss'] = torch.tensor(0.0, device=total_loss.device)
             losses['energy_loss'] = torch.tensor(0.0, device=total_loss.device)


        # --- 5. Hopfield Energy Loss ---
        hopfield_energy = outputs.get('hopfield_energy')
        if hopfield_energy is not None and current_hopfield_weight > 0:
            # Ensure energy map dimensions match targets/masks
            if hopfield_energy.shape[-2:] != inlier_mask_bool.shape[-2:]:
                 hopfield_energy = F.interpolate(hopfield_energy,
                                                  size=inlier_mask_bool.shape[-2:],
                                                  mode='bilinear',
                                                  align_corners=False)

            # Add channel dimension to masks: [B, H, W] -> [B, 1, H, W]
            inlier_mask_hopfield = inlier_mask_bool.unsqueeze(1)
            outlier_mask_hopfield = outlier_mask_bool.unsqueeze(1)

            # Check energy stability
            if torch.isnan(hopfield_energy).any() or torch.isinf(hopfield_energy).any():
                warnings.warn("NaN or Inf detected in hopfield_energy. Replacing with zeros.")
                hopfield_energy = torch.where(torch.isnan(hopfield_energy) | torch.isinf(hopfield_energy),
                                              torch.zeros_like(hopfield_energy), hopfield_energy)

            # Clip Hopfield energy for stability
            hopfield_energy_clipped = torch.clamp(hopfield_energy, min=-self.energy_clip_value, max=self.energy_clip_value)

            # Compute Hopfield loss components
            hopfield_inlier_loss = torch.tensor(0.0, device=hopfield_energy.device, dtype=hopfield_energy.dtype)
            if torch.any(inlier_mask_hopfield):
                hopfield_inlier_energy = hopfield_energy_clipped[inlier_mask_hopfield]
                # For inliers, minimize Hopfield energy (pull towards zero or negative)
                hopfield_inlier_loss = torch.mean(hopfield_inlier_energy) # Simpler: mean penalty

            hopfield_outlier_loss = torch.tensor(0.0, device=hopfield_energy.device, dtype=hopfield_energy.dtype)
            if torch.any(outlier_mask_hopfield):
                hopfield_outlier_energy = hopfield_energy_clipped[outlier_mask_hopfield]
                # For outliers, maximize Hopfield energy (push above outlier_margin)
                hopfield_outlier_loss = torch.mean(F.relu(self.outlier_margin - hopfield_outlier_energy + self.epsilon))

            hopfield_loss = hopfield_inlier_loss + hopfield_outlier_loss

            if torch.isnan(hopfield_loss):
                 warnings.warn("NaN detected in Hopfield energy loss. Skipping Hopfield loss term.")
            else:
                losses['hopfield_inlier_loss'] = hopfield_inlier_loss.detach()
                losses['hopfield_outlier_loss'] = hopfield_outlier_loss.detach()
                losses['hopfield_loss'] = hopfield_loss.detach() # Store unweighted loss
                total_loss += current_hopfield_weight * hopfield_loss
        else:
            # Assign zero tensors if Hopfield loss is skipped
            losses['hopfield_inlier_loss'] = torch.tensor(0.0, device=total_loss.device)
            losses['hopfield_outlier_loss'] = torch.tensor(0.0, device=total_loss.device)
            losses['hopfield_loss'] = torch.tensor(0.0, device=total_loss.device)


        # --- 6. Final Check and Return ---
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            warnings.warn("NaN or Inf detected in final total_loss. Returning only weighted segmentation loss as fallback.")
            # Fallback to just seg_loss if total loss calculation failed
            total_loss = self.seg_weight * losses.get('seg_loss', torch.tensor(0.0, device=targets.device))
            # Ensure fallback is not NaN either
            if torch.isnan(total_loss):
                 total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True) # Ensure it requires grad


        losses['total_loss'] = total_loss
        return losses
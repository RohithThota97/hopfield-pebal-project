# hopfield_pebal_loss.py
# -*- coding: utf-8 -*-
"""
Loss function combining standard segmentation loss with energy-based OOD loss
and a Hopfield-based contrastive loss for the Hopfield-PEBAL model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class HopfieldPEBALLoss(nn.Module):
    """
    Combined loss function for Hopfield-PEBAL.

    Includes:
    1. Standard Semantic Segmentation Loss (Cross-Entropy).
    2. Energy-based Out-of-Distribution (OOD) Loss (Margin-based).
    3. Hopfield-based Contrastive Loss (Placeholder - requires specific implementation).
    """
    def __init__(self,
                 num_classes: int,
                 seg_weight: float = 1.0,
                 energy_weight: float = 0.5, # Added energy_weight parameter
                 hopfield_weight: float = 0.5,
                 inlier_margin: float = 1.0,
                 outlier_margin: float = 10.0,
                 temperature: float = 1.0,
                 ignore_index: int = 255):
        """
        Initializes the HopfieldPEBALLoss.

        Args:
            num_classes (int): Number of segmentation classes.
            seg_weight (float): Weight multiplier for the segmentation loss.
            energy_weight (float): Weight multiplier for the energy-based OOD loss.
            hopfield_weight (float): Weight multiplier for the Hopfield contrastive loss.
            inlier_margin (float): Target maximum energy for in-distribution samples.
                                    Loss is incurred if energy > inlier_margin.
            outlier_margin (float): Target minimum energy for out-of-distribution samples.
                                     Loss is incurred if energy < outlier_margin.
            temperature (float): Temperature scaling for energy calculation (often applied
                                 before this loss, e.g., logsumexp(logits/T)). This parameter
                                 might be used differently depending on the energy definition.
                                 Here it's stored but not directly used in the example loss.
            ignore_index (int): Class index to ignore in segmentation loss calculation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.seg_weight = seg_weight
        self.energy_weight = energy_weight # Store the energy weight
        self.hopfield_weight = hopfield_weight
        self.inlier_margin = inlier_margin
        self.outlier_margin = outlier_margin
        self.temperature = temperature # Store temperature, may be used by energy calculation method
        self.ignore_index = ignore_index

        # Standard segmentation loss component
        self.segmentation_loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')

        logger.info(f"HopfieldPEBALLoss initialized with weights: Seg={self.seg_weight}, "
                    f"Energy={self.energy_weight}, Hopfield={self.hopfield_weight}")
        logger.info(f"Energy margins: Inlier={self.inlier_margin}, Outlier={self.outlier_margin}")

    def _calculate_segmentation_loss(self, predictions, targets):
        """Calculates the weighted segmentation loss."""
        if self.seg_weight <= 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Ensure targets are long type
        targets = targets.long()

        # Calculate loss only on valid pixels
        valid_pixel_mask = targets != self.ignore_index
        if not valid_pixel_mask.any():
            # Avoid calculating loss if the target mask is entirely ignored pixels
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        loss = self.segmentation_loss_fn(predictions, targets)
        return loss * self.seg_weight

    def _calculate_energy_loss(self, energies, is_ood):
        """
        Calculates the weighted energy-based margin loss.
        Assumes lower energy is better for inliers, higher for outliers.
        """
        if self.energy_weight <= 0 or energies is None or is_ood is None:
            return torch.tensor(0.0, device=energies.device if energies is not None else is_ood.device if is_ood is not None else 'cpu', requires_grad=True)

        # Ensure energies are flattened per sample (B,)
        # Handle different potential input shapes (e.g., B,1,H,W or B,)
        if energies.ndim == 4: # Assume shape B, 1, H, W -> average over H, W
             energies_flat = energies.mean(dim=[1, 2, 3])
        elif energies.ndim == 2 and energies.shape[1] == 1: # Assume shape B, 1
             energies_flat = energies.squeeze(1)
        elif energies.ndim == 1: # Assume shape B,
             energies_flat = energies
        else:
             logger.warning(f"Unexpected energy shape: {energies.shape}. Cannot compute energy loss.")
             return torch.tensor(0.0, device=energies.device, requires_grad=True)

        inlier_energies = energies_flat[~is_ood]
        outlier_energies = energies_flat[is_ood]

        loss_in = torch.tensor(0.0, device=energies.device)
        if inlier_energies.numel() > 0:
            # Penalize inliers with energy > inlier_margin
            loss_in = torch.relu(inlier_energies - self.inlier_margin).mean()

        loss_out = torch.tensor(0.0, device=energies.device)
        if outlier_energies.numel() > 0:
            # Penalize outliers with energy < outlier_margin
            loss_out = torch.relu(self.outlier_margin - outlier_energies).mean()

        # Combine inlier and outlier losses (simple average)
        # Handle cases where one type might be missing in the batch
        num_terms = (inlier_energies.numel() > 0) + (outlier_energies.numel() > 0)
        if num_terms > 0:
            energy_loss = (loss_in + loss_out) / max(1, num_terms) # Avoid division by zero
        else:
            energy_loss = torch.tensor(0.0, device=energies.device) # No samples to compute loss on

        return energy_loss * self.energy_weight

    def _calculate_hopfield_loss(self, hopfield_associations, targets, is_ood):
        """
        Placeholder for the Hopfield-based contrastive loss.
        This needs to be implemented based on the specific output of the
        Hopfield layer and the desired contrastive learning strategy.
        """
        if self.hopfield_weight <= 0 or hopfield_associations is None:
            return torch.tensor(0.0, device=targets.device, requires_grad=True) # Use target device as fallback

        # --- Placeholder Logic ---
        # The actual implementation depends heavily on what `hopfield_associations` contains.
        # Example possibilities:
        # 1. Contrast features against memory bank items.
        # 2. Measure consistency between input features and retrieved patterns.
        # 3. Use Hopfield energy/attention scores directly in a loss.

        # Example: Assume hopfield_associations contains retrieved patterns (B, H*W, C)
        # and original features (B, H*W, C). We might want retrieved patterns for
        # inliers to be close to original features, and far for outliers.
        # This is highly speculative and needs adaptation.

        # retrieved_patterns, original_features = hopfield_associations # Example structure
        # loss = some_contrastive_function(retrieved_patterns, original_features, targets, is_ood)

        logger.warning("Hopfield loss calculation is currently a placeholder. "
                       "Implement the specific contrastive logic.")
        hopfield_loss_raw = torch.tensor(0.0, device=targets.device, requires_grad=True) # Return 0 for now

        # --- End Placeholder Logic ---

        return hopfield_loss_raw * self.hopfield_weight

    def forward(self, model_outputs, targets):
        """
        Calculate the combined loss based on model outputs and ground truth.

        Args:
            model_outputs (dict): A dictionary containing outputs from the HopfieldPEBALModel.
                                  Expected keys:
                                  - 'seg_logits' (Tensor): Raw segmentation logits (B, C, H, W).
                                  - 'energy' (Tensor, optional): Energy score per sample/pixel.
                                  - 'is_ood' (Tensor, optional): Boolean tensor indicating OOD status (B,).
                                  - 'hopfield_output' (Any, optional): Output from the Hopfield layer
                                     (e.g., retrieved patterns, attention scores, features).
            targets (Tensor): Ground truth segmentation mask (B, H, W).

        Returns:
            Tensor: The total combined loss, ready for backpropagation.
            dict: A dictionary containing the individual weighted loss components
                  (e.g., 'seg_loss', 'energy_loss', 'hopfield_loss', 'total_loss').
        """
        predictions = model_outputs.get('seg_logits')
        energies = model_outputs.get('energy')
        is_ood = model_outputs.get('is_ood')
        hopfield_associations = model_outputs.get('hopfield_output') # Or specific key

        if predictions is None:
            raise ValueError("Model output dictionary must contain 'seg_logits'.")

        # --- Initialize Loss Dictionary ---
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # --- 1. Segmentation Loss ---
        seg_loss = self._calculate_segmentation_loss(predictions, targets)
        losses['seg_loss'] = seg_loss
        total_loss = total_loss + seg_loss

        # --- 2. Energy-Based OOD Loss ---
        energy_loss = self._calculate_energy_loss(energies, is_ood)
        losses['energy_loss'] = energy_loss
        total_loss = total_loss + energy_loss

        # --- 3. Hopfield Contrastive Loss ---
        hopfield_loss = self._calculate_hopfield_loss(hopfield_associations, targets, is_ood)
        losses['hopfield_loss'] = hopfield_loss
        total_loss = total_loss + hopfield_loss

        # --- Store Total Loss ---
        losses['total_loss'] = total_loss

        return total_loss, losses

# Example usage (for testing the loss function standalone)
if __name__ == '__main__':
    # Setup dummy data
    B, C, H, W = 2, 19, 64, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy model outputs
    dummy_logits = torch.randn(B, C, H, W, device=device, requires_grad=True)
    dummy_targets = torch.randint(0, C, (B, H, W), device=device).long()
    # Add some ignored pixels
    dummy_targets[0, 10:20, 10:20] = 255

    # Simulate energy scores and OOD flags for a mixed batch
    dummy_energies = torch.randn(B, device=device) * 5.0 # Example energy values
    dummy_is_ood = torch.tensor([False, True], device=device) # First sample is ID, second is OOD

    # Dummy Hopfield output (replace with actual structure later)
    dummy_hopfield_output = {
        'retrieved': torch.randn(B, H*W, 64, device=device),
        'original': torch.randn(B, H*W, 64, device=device)
    }

    model_outputs = {
        'seg_logits': dummy_logits,
        'energy': dummy_energies,
        'is_ood': dummy_is_ood,
        'hopfield_output': dummy_hopfield_output
    }

    # Instantiate the loss
    criterion = HopfieldPEBALLoss(
        num_classes=C,
        seg_weight=1.0,
        energy_weight=0.5,
        hopfield_weight=0.2,
        inlier_margin=2.0,
        outlier_margin=8.0,
        temperature=1.0,
        ignore_index=255
    ).to(device)

    # Calculate loss
    total_loss, loss_components = criterion(model_outputs, dummy_targets)

    print(f"Total Loss: {total_loss.item()}")
    print("Loss Components:")
    for name, value in loss_components.items():
        print(f"  {name}: {value.item()}")

    # Test backward pass
    try:
        total_loss.backward()
        print("\nBackward pass successful.")
        # Check gradients (optional)
        # print(f"Gradient for logits (sample): {dummy_logits.grad.abs().mean().item()}")
    except Exception as e:
        print(f"\nError during backward pass: {e}")
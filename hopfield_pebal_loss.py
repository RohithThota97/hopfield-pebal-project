import torch
import torch.nn as nn
import torch.nn.functional as F

class HopfieldPEBALLoss(nn.Module):
    def __init__(self, num_classes, seg_weight=1.0, energy_weight=0.1, hopfield_weight=0.1, 
                 inlier_margin=1.0, outlier_margin=10.0, temperature=1.0,
                 progressive_scaling=True):
        """
        Loss function for Hopfield-PEBAL model with progressive scaling
        
        Args:
            num_classes: Number of segmentation classes
            seg_weight: Weight for segmentation loss
            energy_weight: Initial weight for energy loss (will increase if progressive_scaling=True)
            hopfield_weight: Initial weight for Hopfield energy loss (will increase if progressive_scaling=True)
            inlier_margin: Margin for inlier energy (known classes)
            outlier_margin: Margin for outlier energy (OOD)
            temperature: Temperature for energy scaling
            progressive_scaling: Whether to progressively scale energy weights during training
        """
        super(HopfieldPEBALLoss, self).__init__()
        
        self.num_classes = num_classes
        self.seg_weight = seg_weight
        self.energy_weight_max = 0.5  # Target weight for energy loss
        self.hopfield_weight_max = 0.5  # Target weight for hopfield loss
        self.energy_weight = energy_weight  # Starting weight (much lower)
        self.hopfield_weight = hopfield_weight  # Starting weight (much lower)
        self.inlier_margin = inlier_margin
        self.outlier_margin = outlier_margin
        self.temperature = temperature
        self.progressive_scaling = progressive_scaling
        self.epoch = 0  # Track current epoch
        self.total_epochs = 30  # Total expected epochs
        
        # Segmentation loss (ignore index 255 which is often used for "void" pixels)
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def update_epoch(self, epoch, total_epochs=None):
        """Update current epoch for progressive scaling"""
        self.epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
    
    def forward(self, outputs, targets, outlier_mask=None):
        """
        Compute combined loss for segmentation and energy-based OOD detection
        
        Args:
            outputs: Dictionary of model outputs
            targets: Target segmentation masks [B, H, W]
            outlier_mask: Optional binary mask for outlier pixels [B, H, W]
        
        Returns:
            Dictionary of loss components and total loss
        """
        # Calculate progressive scaling factors if enabled
        if self.progressive_scaling:
            # Start with smaller weights and gradually increase
            progress = min(1.0, self.epoch / (self.total_epochs * 0.5))  # Reach max at half of training
            current_energy_weight = self.energy_weight + (self.energy_weight_max - self.energy_weight) * progress
            current_hopfield_weight = self.hopfield_weight + (self.hopfield_weight_max - self.hopfield_weight) * progress
        else:
            current_energy_weight = self.energy_weight
            current_hopfield_weight = self.hopfield_weight
        
        # Initialize loss dictionary
        losses = {}
        
        # Get segmentation logits from outputs
        logits = outputs['logits']
        
        # Add numerical stability check
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Replace NaN/Inf values with zeros
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                 torch.zeros_like(logits), 
                                 logits)
        
        # Add epsilon to prevent division by zero or log(0)
        epsilon = 1e-6
        
        # Ensure targets have matching spatial dimensions with logits
        if logits.shape[-2:] != targets.shape[-2:]:
            targets = F.interpolate(targets.float().unsqueeze(1), 
                                    size=logits.shape[-2:], 
                                    mode='nearest').squeeze(1).long()
        
        # Compute segmentation loss
        seg_loss = self.seg_loss(logits, targets)
        losses['seg_loss'] = seg_loss
        
        # Initialize total loss with segmentation loss
        total_loss = self.seg_weight * seg_loss
        
        # If no outlier mask is provided, create one based on targets
        if outlier_mask is None:
            outlier_mask = (targets == 255)
        
        # Make sure outlier_mask has the right shape
        if outlier_mask.dim() != targets.dim():
            raise ValueError(f"outlier_mask dim ({outlier_mask.dim()}) doesn't match targets dim ({targets.dim()})")
        
        inlier_mask = ~outlier_mask
        
        # Get combined energy (this includes all energy terms)
        combined_energy = outputs['combined_energy']
        
        # Add numerical stability check
        if torch.isnan(combined_energy).any() or torch.isinf(combined_energy).any():
            # Replace NaN/Inf values with zeros
            combined_energy = torch.where(torch.isnan(combined_energy) | torch.isinf(combined_energy),
                                         torch.zeros_like(combined_energy),
                                         combined_energy)
        
        # Make sure energy has same spatial dimensions as the mask
        if combined_energy.shape[-2:] != outlier_mask.shape[-2:]:
            combined_energy = F.interpolate(combined_energy, 
                                           size=outlier_mask.shape[-2:], 
                                           mode='bilinear', 
                                           align_corners=False)
        
        # Extract inlier and outlier energies
        inlier_mask = inlier_mask.unsqueeze(1)  # Add channel dimension [B, 1, H, W]
        outlier_mask = outlier_mask.unsqueeze(1)  # Add channel dimension [B, 1, H, W]
        
        # Clip energy values to prevent extreme values
        combined_energy = torch.clamp(combined_energy, min=-20.0, max=20.0)  # Reduced from -/+100
        
        if torch.any(inlier_mask):
            inlier_energy = combined_energy[inlier_mask].view(-1)
            # More conservative ReLU margin with eps
            inlier_loss = torch.mean(F.relu(inlier_energy - self.inlier_margin + epsilon))
        else:
            inlier_loss = torch.tensor(0.0, device=combined_energy.device)
        
        if torch.any(outlier_mask):
            outlier_energy = combined_energy[outlier_mask].view(-1)
            # More conservative ReLU margin with eps
            outlier_loss = torch.mean(F.relu(self.outlier_margin - outlier_energy + epsilon))
        else:
            outlier_loss = torch.tensor(0.0, device=combined_energy.device)
        
        # Combine energy losses with scaling
        energy_loss = inlier_loss + outlier_loss
        
        # Apply scaling to energy loss
        scaled_energy_loss = energy_loss * current_energy_weight
        losses['energy_loss'] = energy_loss  # Store unscaled for logging
        
        # Add to total loss
        total_loss = total_loss + scaled_energy_loss
        
        # Process Hopfield-specific energy if available
        if 'hopfield_energy' in outputs and outputs['hopfield_energy'] is not None:
            hopfield_energy = outputs['hopfield_energy']
            
            # Add numerical stability check
            if torch.isnan(hopfield_energy).any() or torch.isinf(hopfield_energy).any():
                hopfield_energy = torch.where(torch.isnan(hopfield_energy) | torch.isinf(hopfield_energy),
                                             torch.zeros_like(hopfield_energy),
                                             hopfield_energy)
            
            # Make sure energy has same spatial dimensions as the mask
            if hopfield_energy.shape[-2:] != outlier_mask.shape[-2:]:
                hopfield_energy = F.interpolate(hopfield_energy, 
                                              size=outlier_mask.shape[-2:], 
                                              mode='bilinear', 
                                              align_corners=False)
            
            # Clip energy values to prevent extreme values
            hopfield_energy = torch.clamp(hopfield_energy, min=-20.0, max=20.0)  # Reduced from -/+100
            
            # Extract inlier and outlier Hopfield energies
            if torch.any(inlier_mask):
                hopfield_inlier_energy = hopfield_energy[inlier_mask].view(-1)
                # For inliers, we want to minimize Hopfield energy
                hopfield_inlier_loss = torch.mean(hopfield_inlier_energy)
            else:
                hopfield_inlier_loss = torch.tensor(0.0, device=hopfield_energy.device)
            
            if torch.any(outlier_mask):
                hopfield_outlier_energy = hopfield_energy[outlier_mask].view(-1)
                # For outliers, we want to maximize Hopfield energy (with margin)
                hopfield_outlier_loss = torch.mean(
                    F.relu(self.outlier_margin - hopfield_outlier_energy + epsilon)
                )
            else:
                hopfield_outlier_loss = torch.tensor(0.0, device=hopfield_energy.device)
            
            # Combine Hopfield energy losses
            hopfield_loss = hopfield_inlier_loss + hopfield_outlier_loss
            
            # Store unscaled for logging
            losses['hopfield_loss'] = hopfield_loss
            
            # Apply scaling to hopfield loss
            scaled_hopfield_loss = hopfield_loss * current_hopfield_weight
            
            # Add to total loss
            total_loss = total_loss + scaled_hopfield_loss
        
        # Check for NaN in total loss and replace with a default value if necessary
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Use just the segmentation loss as a fallback
            total_loss = self.seg_weight * seg_loss
            
        # Store total loss and current weights
        losses['total_loss'] = total_loss
        losses['energy_weight'] = torch.tensor(current_energy_weight)
        losses['hopfield_weight'] = torch.tensor(current_hopfield_weight)
        
        return losses
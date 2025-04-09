import torch
import torch.nn as nn
import torch.nn.functional as F

class HopfieldPEBALLoss(nn.Module):
    def __init__(self, num_classes, energy_weight=0.1, hopfield_weight=0.1, 
                 prototype_weight=0.05, anomaly_margin=10.0, known_margin=1.0,
                 temperature=0.1):
        super(HopfieldPEBALLoss, self).__init__()
        self.num_classes = num_classes
        self.energy_weight = energy_weight
        self.hopfield_weight = hopfield_weight
        self.prototype_weight = prototype_weight
        self.anomaly_margin = anomaly_margin
        self.known_margin = known_margin
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        
    def forward(self, outputs, targets, aux_images=None, is_anomaly=None):
        """
        Forward pass for Hopfield-PEBAL loss
        Args:
            outputs: Dict containing model outputs (logits, energy, features, memory_energies)
            targets: Segmentation targets
            aux_images: Optional auxiliary (OOD) images
            is_anomaly: Binary mask indicating anomaly pixels
        """
        loss_dict = {}
        
        # Standard cross-entropy loss for segmentation
        seg_loss = self.ce_loss(outputs['logits'], targets)
        loss_dict['seg_loss'] = seg_loss
        total_loss = seg_loss
        
        # Energy loss (PEBAL style)
        if 'energy' in outputs and self.energy_weight > 0:
            energy_loss = self._energy_loss(outputs['energy'], targets, is_anomaly)
            loss_dict['energy_loss'] = energy_loss
            total_loss = total_loss + self.energy_weight * energy_loss
        
        # Hopfield memory-based loss
        if 'memory_energies' in outputs and self.hopfield_weight > 0:
            hopfield_loss = self._hopfield_loss(outputs['memory_energies'], targets, is_anomaly)
            loss_dict['hopfield_loss'] = hopfield_loss
            total_loss = total_loss + self.hopfield_weight * hopfield_loss
        
        # Prototype contrastive loss when available
        if 'prototype_energies' in outputs and self.prototype_weight > 0:
            proto_loss = self._prototype_loss(outputs['prototype_energies'], targets, is_anomaly)
            loss_dict['proto_loss'] = proto_loss
            total_loss = total_loss + self.prototype_weight * proto_loss
            
        loss_dict['loss'] = total_loss
        return loss_dict
    
    def _energy_loss(self, energy, targets, is_anomaly=None):
        """PEBAL energy loss component"""
        # For known classes: reduce energy
        if is_anomaly is None:
            valid_mask = (targets != 255)
            ood_mask = (targets == 255)
        else:
            valid_mask = ~is_anomaly
            ood_mask = is_anomaly
        
        known_energy = energy[valid_mask]
        if known_energy.numel() == 0:
            known_loss = torch.tensor(0.0, device=energy.device)
        else:
            known_loss = torch.mean(torch.max(known_energy - (-self.known_margin), 
                                             torch.tensor(0.0).to(energy.device)))
        
        # For unknown or OOD: increase energy
        if ood_mask.any():
            unknown_energy = energy[ood_mask]
            unknown_loss = torch.mean(torch.max(self.anomaly_margin - unknown_energy, 
                                               torch.tensor(0.0).to(energy.device)))
        else:
            unknown_loss = torch.tensor(0.0, device=energy.device)
        
        return known_loss + unknown_loss
    
    def _hopfield_loss(self, memory_energies, targets, is_anomaly=None):
        """Hopfield memory retrieval loss component"""
        if memory_energies.numel() == 0:
            return torch.tensor(0.0, device=memory_energies.device)
        
        # Determine inlier and outlier masks
        if is_anomaly is None:
            inlier_mask = (targets != 255)
            outlier_mask = (targets == 255)
        else:
            inlier_mask = ~is_anomaly
            outlier_mask = is_anomaly
        
        inlier_energies = memory_energies[inlier_mask]
        outlier_energies = memory_energies[outlier_mask]
        
        # Inliers should have low memory energy (high similarity)
        inlier_loss = torch.mean(inlier_energies) if inlier_energies.numel() > 0 else torch.tensor(0.0).to(memory_energies.device)
        
        # Outliers should have high memory energy (low similarity)
        outlier_loss = torch.mean(torch.max(self.anomaly_margin - outlier_energies, 
                                           torch.tensor(0.0).to(memory_energies.device))) if outlier_energies.numel() > 0 else torch.tensor(0.0).to(memory_energies.device)
        
        return inlier_loss + outlier_loss
    
    def _prototype_loss(self, prototype_energies, targets, is_anomaly=None):
        """Prototype-based contrastive loss"""
        if prototype_energies.numel() == 0:
            return torch.tensor(0.0, device=prototype_energies.device)
            
        # Determine inlier and outlier masks
        if is_anomaly is None:
            inlier_mask = (targets != 255)
            outlier_mask = (targets == 255)
        else:
            inlier_mask = ~is_anomaly
            outlier_mask = is_anomaly
            
        if inlier_mask.sum() == 0 or outlier_mask.sum() == 0:
            return torch.tensor(0.0, device=prototype_energies.device)
            
        # Apply InfoNCE-like contrastive loss
        # Inliers should be close to prototypes (low energy)
        # Outliers should be far from prototypes (high energy)
        inlier_energies = prototype_energies[inlier_mask]
        outlier_energies = prototype_energies[outlier_mask]
        
        # Sample at most 1000 points for efficiency
        if inlier_energies.numel() > 1000:
            idx = torch.randperm(inlier_energies.numel())[:1000]
            inlier_energies = inlier_energies[idx]
        
        if outlier_energies.numel() > 1000:
            idx = torch.randperm(outlier_energies.numel())[:1000]
            outlier_energies = outlier_energies[idx]
            
        # Convert energies to similarities (1 - energy)
        inlier_sim = 1.0 - inlier_energies
        outlier_sim = 1.0 - outlier_energies
        
        # Normalize similarities
        inlier_sim = inlier_sim / self.temperature
        outlier_sim = outlier_sim / self.temperature
        
        # Compute contrastive loss
        pos_term = -torch.mean(inlier_sim)
        neg_term = torch.mean(F.softplus(outlier_sim))
        
        return pos_term + neg_term
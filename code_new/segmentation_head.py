import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SegmentationClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes
        
        # Add intermediate layers for stability
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.01)  # Lower momentum for stability
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        # Final classification layer with careful initialization
        self.final = nn.Conv2d(512, num_classes+1, kernel_size=1, bias=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Very careful initialization to prevent gradient explosion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Smaller initialization for final layer
                if m == self.final:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.xavier_normal_(m.weight, gain=0.5)  # Reduced gain
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        # Check for NaN/Inf in input
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.warning("NaN/Inf detected in input features")
            features = torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Apply intermediate layers with gradient-safe operations
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Get logits with clamping
        logits = self.final(x)
        
        # Aggressive clamping to prevent overflow
        logits = torch.clamp(logits, min=-5, max=5)
        
        # Optionally upsample
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits, 
                size=output_size, 
                mode='bilinear', 
                align_corners=True
            )
        
        return logits
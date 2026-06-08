import torch.nn as nn
import torch.nn.functional as F

class SimpleProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(input_dim, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Add dropout for variance
            nn.Conv2d(512, 256, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, output_dim, 1, bias=True)
        )
    
    def forward(self, x):
        x = self.projection(x)
        # Enforce unit normalization (critical for Hopfield)
        return F.normalize(x, p=2, dim=1)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

class FeatureDataset(Dataset):
 
    
    def __init__(self, feature_dir, feature_type='decoder', transform=None):
       
        self.feature_dir = feature_dir
        self.feature_type = feature_type
        self.transform = transform
        
       
        self.segmap_paths = sorted(glob.glob(os.path.join(feature_dir, "synthetic_ood", "segmap", "*_segmap.npy")))
        
      
        self.feature_paths = []
        for segmap_path in self.segmap_paths:
            base_name = os.path.basename(segmap_path).replace("_segmap.npy", "")
            feature_path = os.path.join(feature_dir, "synthetic_ood", feature_type, f"{base_name}_{feature_type}.npy")
            if os.path.exists(feature_path):
                self.feature_paths.append(feature_path)
            else:
        
                self.segmap_paths.remove(segmap_path)
        
        print(f"Found {len(self.feature_paths)} samples")
    
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
  
        feature = np.load(self.feature_paths[idx])
        feature_tensor = torch.from_numpy(feature).float()
        
  
        segmap = np.load(self.segmap_paths[idx])
      
        mask = (segmap == 254).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).float()
        

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        
        return feature_tensor, mask_tensor


class ProjectionHead(nn.Module):
   
    
    def __init__(self, in_channels, hidden_dim=128, out_dim=64):
       
        super(ProjectionHead, self).__init__()
        
   
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True)
        )
    
    def forward(self, x):
       
        return self.projection(x)
   
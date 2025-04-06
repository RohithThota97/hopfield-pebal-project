import torch
from torch.utils.data import Dataset
import random

class DummyDataset(Dataset):
    def __init__(self, length=10, image_size=(3, 256, 256)):
        self.length = length
        self.image_size = image_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Make random images
        image = torch.randn(self.image_size)
        # Make random segmentation targets
        target = torch.zeros(self.image_size[1], self.image_size[2], dtype=torch.long)
        
        # Let's say the first half are normal, second half are anomalies
        # We'll just do random in this dummy example
        is_anomaly = torch.zeros_like(target)
        if random.random() < 0.3:
            # 30% chance of an anomaly
            is_anomaly[target.shape[0]//2:, target.shape[1]//2:] = 1
        
        return {
            'image': image,
            'target': target,
            'is_anomaly': is_anomaly
        }
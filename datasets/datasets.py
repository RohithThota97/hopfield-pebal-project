from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("Hopfield-PEBAL")

def convert_label(mask):
    # Dummy conversion example. Customize if needed.
    return mask

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, num_classes=19):
        # Use pathlib.Path to handle paths
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        
        # We assume that mask files are images (e.g., PNG) and not JSON files.
        # Filter out any JSON files.
        self.image_paths = sorted([p for p in self.image_dir.glob("*.*")
                                    if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        self.mask_paths = sorted([p for p in self.mask_dir.glob("*.*")
                                   if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image files found in {self.image_dir}")
        if len(self.mask_paths) == 0:
            raise FileNotFoundError(f"No mask files found in {self.mask_dir}")
        
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and convert to RGB.
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            # Create a default image if loading fails.
            image = Image.new("RGB", (512, 1024), color=(0, 0, 0))
            
        # Load mask and ensure it is converted to grayscale ("L").
        try:
            # Convert mask to "L" to ensure a single channel.
            mask = Image.open(self.mask_paths[idx]).convert("L")
        except Exception as e:
            logger.error(f"Error loading mask {self.mask_paths[idx]}: {e}")
            # Create a blank mask if loading fails.
            mask = Image.new("L", (512, 1024), color=255)
        
        if self.transform:
            image = self.transform(image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # If no mask_transform is provided, convert mask to a numpy array and then to a tensor.
            mask = np.array(mask, dtype=np.int64)
            mask = torch.from_numpy(mask)
        
        return image, mask

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_files=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Get all image files (filtering by common extensions)
        self.image_paths = sorted({p for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'] 
                                    for p in self.root_dir.rglob(f"*{ext}")})
        if max_files is not None and len(self.image_paths) > max_files:
            self.image_paths = list(self.image_paths)[:max_files]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No valid image files found in {root_dir}")
        logger.info(f"Found {len(self.image_paths)} outlier images in {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, 0  # dummy label
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            image = Image.new("RGB", (512, 1024), color="black")
            if self.transform:
                image = self.transform(image)
            return image, 0
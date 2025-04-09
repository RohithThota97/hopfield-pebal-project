from PIL import Image
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
import logging
import os

logger = logging.getLogger("Hopfield-PEBAL")

def convert_label(mask):
    label_map = {
         -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
         7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
         16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
         25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
    }
    out_mask = np.full(mask.shape, 255, dtype=np.uint8)
    for k, v in label_map.items():
        out_mask[mask == k] = v
    return out_mask

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, num_classes=19):
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        
        # Verify directories exist
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
            
        self.image_paths = sorted(
            list(image_dir.glob("*.png")) +
            list(image_dir.glob("*.jpg")) +
            list(image_dir.glob("*.jpeg"))
        )
        
        # Specifically look for gtFine_labelIds files in the mask directory
        self.mask_paths = sorted(
            [p for p in mask_dir.glob("*gtFine_labelIds.png")] if any(mask_dir.glob("*gtFine_labelIds.png")) else
            ([p for p in mask_dir.glob("*.png")] + 
             list(mask_dir.glob("*.jpg")) + 
             list(mask_dir.glob("*.jpeg")))
        )
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image files found in {image_dir}")
        if len(self.mask_paths) == 0:
            raise FileNotFoundError(f"No mask files found in {mask_dir}")
        
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

        # If number of images and masks don't match, try to find matching pairs by filename
        if len(self.image_paths) != len(self.mask_paths):
            logger.warning(f"Number of images ({len(self.image_paths)}) does not match " 
                         f"number of masks ({len(self.mask_paths)}). Attempting to match.")
            
            # Try to match based on common naming pattern in Cityscapes
            img_basenames = {p.stem.split('_leftImg8bit')[0]: p for p in self.image_paths}
            mask_basenames = {p.stem.split('_gtFine_labelIds')[0]: p for p in self.mask_paths}
            
            common_keys = set(img_basenames.keys()) & set(mask_basenames.keys())
            
            if common_keys:
                logger.info(f"Found {len(common_keys)} matching pairs by filename")
                self.image_paths = [img_basenames[k] for k in common_keys]
                self.mask_paths = [mask_basenames[k] for k in common_keys]
            else:
                # If still no matches, try a different approach
                logger.warning("Could not match by filenames. Using minimum number of files.")
                min_len = min(len(self.image_paths), len(self.mask_paths))
                self.image_paths = self.image_paths[:min_len]
                self.mask_paths = self.mask_paths[:min_len]
        
        logger.info(f"SegmentationDataset: {len(self.image_paths)} samples from {image_dir} + {mask_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            image = Image.new("RGB", (512, 1024), color=(0, 0, 0))
            
        try:
            mask_pil = Image.open(self.mask_paths[idx])
        except Exception as e:
            logger.error(f"Error loading mask {self.mask_paths[idx]}: {e}")
            mask_pil = Image.new("L", (512, 1024), color=255)
        
        if self.transform:
            image = self.transform(image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask_pil)
        else:
            mask_np = np.array(mask_pil, dtype=np.int32)
            mask_np = convert_label(mask_np)
            mask = torch.from_numpy(mask_np).long()
        
        return image, mask

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_files=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Verify directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
            
        # Find all image files in directory and subdirectories
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']:
            self.image_paths.extend(list(self.root_dir.glob(f'*{ext}')))
            self.image_paths.extend(list(self.root_dir.glob(f'**/*{ext}')))
            
        self.image_paths = sorted(set(self.image_paths))
        
        # Sample if more than max_files
        if max_files is not None and len(self.image_paths) > max_files:
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_files)
            
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No valid image files found in {root_dir}")
            
        logger.info(f"Found {len(self.image_paths)} outlier images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0  # dummy label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (512, 1024), color='black')
            if self.transform:
                image = self.transform(image)
            return image, 0
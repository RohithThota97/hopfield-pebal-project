from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("Hopfield-PEBAL")

class FishyscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, num_classes=19, anomaly_id=19):
        """
        Dataset for Fishyscapes anomaly detection
        
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing label/mask images
            transform: Transforms to apply to input images
            mask_transform: Transforms to apply to mask images
            num_classes: Number of semantic segmentation classes
            anomaly_id: Class ID to use for anomalies (default: 19, which is after Cityscapes classes)
        """
        # Use pathlib.Path to handle paths
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes
        self.anomaly_id = anomaly_id
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        
        # Find all image files
        self.image_paths = sorted([p for p in self.image_dir.glob("*.*")
                                  if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        # Find matching mask files - we try to find masks with the same stem name
        self.mask_paths = []
        for img_path in self.image_paths:
            # Try different possible mask filenames
            possible_masks = [
                self.mask_dir / f"{img_path.stem}.png",
                self.mask_dir / f"{img_path.stem}.jpg",
                self.mask_dir / f"{img_path.stem}_labels.png",
                self.mask_dir / f"{img_path.name}"
            ]
            
            mask_found = False
            for mask_path in possible_masks:
                if mask_path.exists():
                    self.mask_paths.append(mask_path)
                    mask_found = True
                    break
            
            if not mask_found:
                logger.warning(f"No mask found for image {img_path.name}")
                # Use a placeholder mask path that will be handled in __getitem__
                self.mask_paths.append(None)
        
        logger.info(f"Found {len(self.image_paths)} images in {self.image_dir}")
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image files found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            # Create a default image if loading fails
            image = Image.new("RGB", (512, 1024), color=(0, 0, 0))
        
        # Load mask
        if self.mask_paths[idx] is not None:
            try:
                mask = Image.open(self.mask_paths[idx])
                
                # Handle different mask formats
                if mask.mode == 'RGB' or mask.mode == 'RGBA':
                    # For RGB masks, convert to numpy first
                    mask_np = np.array(mask)
                    
                    # Check if this is a colored label map (common in Fishyscapes)
                    if len(mask_np.shape) == 3 and mask_np.shape[2] >= 3:
                        # Check if this is a binary mask where red channel indicates anomaly
                        if np.any(mask_np[:, :, 0] > 200) and np.all(mask_np[:, :, 1] < 50) and np.all(mask_np[:, :, 2] < 50):
                            # Red channel has high values, treat as anomaly
                            final_mask = np.zeros_like(mask_np[:, :, 0])
                            final_mask[mask_np[:, :, 0] > 200] = self.anomaly_id
                            mask = Image.fromarray(final_mask.astype(np.uint8))
                        else:
                            # Convert RGB label map to single channel
                            # This is a simplification - you may need to customize based on your dataset
                            gray_mask = np.zeros_like(mask_np[:, :, 0])
                            
                            # Example: Mark pixels with high values in any channel as anomaly
                            anomaly_pixels = np.logical_or.reduce([
                                mask_np[:, :, 0] > 200,  # Red channel
                                mask_np[:, :, 1] > 200,  # Green channel
                                mask_np[:, :, 2] > 200   # Blue channel
                            ])
                            gray_mask[anomaly_pixels] = self.anomaly_id
                            mask = Image.fromarray(gray_mask.astype(np.uint8))
                    
                # Convert to grayscale as fallback
                mask = mask.convert("L")
                
            except Exception as e:
                logger.error(f"Error loading mask {self.mask_paths[idx]}: {e}")
                # Create a blank mask if loading fails
                mask = Image.new("L", image.size, color=255)  # Use image size
        else:
            # Create an empty mask if no mask path was found
            mask = Image.new("L", image.size, color=255)  # Use image size
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convert mask to tensor
            mask_np = np.array(mask, dtype=np.int64)
            
            # Check if this is a binary mask where 1 indicates anomaly and 0 is background
            if np.max(mask_np) == 1 and np.min(mask_np) == 0:
                # Map 1s to anomaly ID
                mask_np[mask_np == 1] = self.anomaly_id
            
            mask = torch.from_numpy(mask_np)
        
        return image, mask
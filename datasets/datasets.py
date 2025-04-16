import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError for specific handling
from torch.utils.data import Dataset

# --- Logging Setup ---
# Configure logger (ensure this is set up appropriately elsewhere in your project)
# Example basic config:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetRefinement") # Or use your project's logger

# --- Constants ---
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# Default image size for error handling (consider making this configurable)
DEFAULT_SIZE = (1024, 512) # Width, Height typical for Cityscapes

# --- Helper Function ---
def convert_label(mask: Image.Image) -> Image.Image:
    """
    Applies specific label conversions to the mask.

    Args:
        mask: PIL Image mask (single channel, e.g., "L" mode).

    Returns:
        PIL Image mask with converted labels.

    Note:
        This is a placeholder. Customize this function based on your
        segmentation task's label mapping requirements (e.g., mapping
        Cityscapes 'trainIds' to your model's required class indices,
        handling void labels like 255 or -1 appropriately).
        If no conversion is needed, this function can simply return the input mask.
    """
    # Example: If you need to map Cityscapes trainIds (0-18, 255) to a contiguous range
    # mask_np = np.array(mask, dtype=np.int64)
    # map_dict = {255: 19, -1: 19} # Example: Map void labels to class 19 (ignore index)
    # for k, v in map_dict.items():
    #     mask_np[mask_np == k] = v
    # return Image.fromarray(mask_np)

    # For now, just return the original mask if no specific conversion is implemented
    return mask

# --- Segmentation Dataset ---
class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation tasks.
    Loads images and their corresponding single-channel masks.
    Ensures image and mask pairs are correctly matched based on filename stems,
    with specific handling for common naming conventions like Cityscapes.
    """
    def __init__(self,
                 image_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None,
                 num_classes: int = 19,
                 image_suffix: str = '_leftImg8bit.png', # Default Cityscapes image suffix
                 mask_suffix: str = '_gtFine_labelIds.png'): # Default Cityscapes label mask suffix
        """
        Args:
            image_dir: Path to the root directory containing images (can have subdirs).
            mask_dir: Path to the root directory containing masks (structure should ideally mirror image_dir).
            transform: Optional transform to be applied to the image.
            mask_transform: Optional transform to be applied to the mask *after*
                            `convert_label` and *before* conversion to tensor.
                            Should handle spatial transformations consistently with `transform`.
            num_classes: The number of segmentation classes (metadata, not used internally here).
            image_suffix: The expected suffix of image files.
            mask_suffix: The expected suffix of the mask files (e.g., label IDs).
        """
        self.image_dir = Path(image_dir).resolve() # Use resolve for absolute paths
        self.mask_dir = Path(mask_dir).resolve()
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found or is not a directory: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found or is not a directory: {self.mask_dir}")

        self.files: List[Dict[str, Path]] = []
        # Use rglob to recursively find images matching the suffix
        logger.info(f"Searching for images ending with '{self.image_suffix}' in {self.image_dir}...")
        image_paths = sorted(self.image_dir.rglob(f"*{self.image_suffix}"))

        if not image_paths:
             logger.warning(f"No images found with suffix '{self.image_suffix}' in {self.image_dir} or its subdirectories.")
             # Don't raise error immediately, let the pair matching determine if data exists

        found_pairs = 0
        missing_masks = 0
        for img_path in image_paths:
            # Derive the base name required for the mask
            # e.g., 'path/to/city_000000_000019_leftImg8bit.png' -> 'city_000000_000019'
            base_name = img_path.name.replace(self.image_suffix, '')

            # Construct the expected mask filename
            mask_name = f"{base_name}{self.mask_suffix}"

            # Determine the relative path of the image within image_dir to find the corresponding mask subdir
            try:
                relative_img_dir = img_path.parent.relative_to(self.image_dir)
                mask_path = self.mask_dir / relative_img_dir / mask_name
            except ValueError:
                # Should not happen if img_path is within image_dir, but handle defensively
                logger.warning(f"Image path {img_path} seems not to be within {self.image_dir}. Searching mask directly in {self.mask_dir}.")
                mask_path = self.mask_dir / mask_name # Fallback: Check directly in mask_dir root


            # Check if the corresponding mask file exists in the expected location
            if mask_path.is_file():
                self.files.append({"image": img_path, "mask": mask_path})
                found_pairs += 1
            else:
                # Optional: Fallback check if masks are flat in the mask_dir root
                mask_path_flat = self.mask_dir / mask_name
                if mask_path_flat.is_file():
                     logger.debug(f"Found mask for {img_path.name} directly in {self.mask_dir}, not in expected subpath.")
                     self.files.append({"image": img_path, "mask": mask_path_flat})
                     found_pairs += 1
                else:
                    # Only log as warning if the mask is truly missing
                    logger.warning(f"No corresponding mask found for image: {img_path} (Expected mask: '{mask_name}' at {mask_path} or {mask_path_flat})")
                    missing_masks += 1


        if not self.files:
            raise FileNotFoundError(
                f"No matching image/mask pairs found using image suffix '{self.image_suffix}' "
                f"and mask suffix '{self.mask_suffix}' between {self.image_dir} and {self.mask_dir}. "
                f"Found {len(image_paths)} potential images but {missing_masks} missing masks. "
                f"Please check paths, suffixes, and directory structures."
            )

        logger.info(f"Found {found_pairs} matching image/mask pairs.")
        if missing_masks > 0:
             logger.warning(f"Could not find masks for {missing_masks} images.")


    def __len__(self) -> int:
        """Returns the total number of image/mask pairs."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns an image and its corresponding mask pair.

        Args:
            idx: Index of the pair to retrieve.

        Returns:
            A tuple containing (image_tensor, mask_tensor).
            If loading fails, returns default black image and default mask (value 255).
        """
        if idx >= len(self.files):
             raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.files)}")

        file_pair = self.files[idx]
        img_path = file_pair["image"]
        mask_path = file_pair["mask"]

        # --- Load Image ---
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image file not found at {img_path}. Returning default.")
            image = Image.new("RGB", DEFAULT_SIZE, color="black")
        except UnidentifiedImageError:
             logger.error(f"Could not identify image file format for {img_path}. Returning default.")
             image = Image.new("RGB", DEFAULT_SIZE, color="black")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}. Returning default.")
            image = Image.new("RGB", DEFAULT_SIZE, color="black")

        # --- Load Mask ---
        try:
            # Open mask and ensure it's single channel ('L' mode for labels)
            mask = Image.open(mask_path).convert("L")
            # Apply label conversion if needed (e.g., map trainIds)
            mask = convert_label(mask)
        except FileNotFoundError:
            logger.error(f"Mask file not found at {mask_path}. Returning default mask.")
            mask = Image.new("L", image.size, color=255) # Match potentially loaded image size
        except UnidentifiedImageError:
             logger.error(f"Could not identify image file format for {mask_path}. Returning default mask.")
             mask = Image.new("L", image.size, color=255)
        except Exception as e:
            logger.error(f"Error loading or converting mask {mask_path}: {e}. Returning default mask.")
            # Use size from image (even if default) or fallback to DEFAULT_SIZE
            img_size = image.size if hasattr(image, 'size') else DEFAULT_SIZE
            mask = Image.new("L", img_size, color=255) # 255 often used as ignore_index

        # --- Apply Transforms ---
        # Note: For spatial transforms (like resize, crop, flip), apply them
        # consistently to both image and mask. This often requires a custom transform
        # function or library that handles paired transformations.
        # Simple example: if transform includes ToTensor, do mask conversion manually.

        if self.transform:
            image = self.transform(image) # Assume transform outputs Tensor

        # Handle mask transformation and conversion to tensor
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # Ensure mask is tensor after transform
            if not isinstance(mask, torch.Tensor):
                 try:
                     # Assuming transform might return PIL Image or numpy array
                     mask_np = np.array(mask, dtype=np.int64)
                     # Remove singleton channel dimension if added by transform (e.g. ToTensor())
                     if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                         mask_np = mask_np.squeeze(0)
                     mask = torch.from_numpy(mask_np)
                 except Exception as e:
                     logger.error(f"Error converting mask to tensor after mask_transform for {mask_path}: {e}")
                     # Create a default tensor mask if conversion fails
                     # Try to get shape from image tensor if available
                     h, w = image.shape[-2:] if isinstance(image, torch.Tensor) and image.ndim >= 2 else DEFAULT_SIZE[::-1]
                     mask = torch.full((h, w), 255, dtype=torch.int64) # Use ignore_index

        else:
            # Default conversion if no mask_transform is provided
            try:
                mask_np = np.array(mask, dtype=np.int64)
                # Ensure mask is HW not HWC or CHW
                if mask_np.ndim == 3:
                     if mask_np.shape[0] == 1: mask_np = mask_np.squeeze(0) # CHW -> HW
                     elif mask_np.shape[-1] == 1: mask_np = mask_np.squeeze(-1) # HWC -> HW
                     else: logger.warning(f"Mask {mask_path} has unexpected shape {mask_np.shape} after conversion to numpy.")
                mask = torch.from_numpy(mask_np)
            except Exception as e:
                logger.error(f"Error converting mask to tensor (default path) for {mask_path}: {e}")
                h, w = image.shape[-2:] if isinstance(image, torch.Tensor) and image.ndim >= 2 else DEFAULT_SIZE[::-1]
                mask = torch.full((h, w), 255, dtype=torch.int64)

        # Ensure mask is LongTensor as expected for segmentation targets (e.g., CrossEntropyLoss)
        if mask.dtype != torch.int64:
             mask = mask.to(torch.int64)

        # Final check for shape consistency if possible
        if isinstance(image, torch.Tensor) and image.ndim >=2 and isinstance(mask, torch.Tensor) and mask.ndim >= 2:
            if image.shape[-2:] != mask.shape[-2:]:
                logger.warning(f"Image ({image.shape[-2:]}) and mask ({mask.shape[-2:]}) shapes mismatch for index {idx} ({img_path.name}). Check transforms.")

        return image, mask

# --- Simple Image Dataset ---
class SimpleImageDataset(Dataset):
    """
    A simple dataset to load images from a directory (recursively).
    Returns images and a dummy label (0). Useful for tasks like
    autoencoding, pre-training, or inference where only images are needed.
    """
    def __init__(self,
                 root_dir: Union[str, Path],
                 transform: Optional[Callable] = None,
                 max_files: Optional[int] = None):
        """
        Args:
            root_dir: Path to the root directory containing images.
            transform: Optional transform to be applied to the images.
            max_files: Optional maximum number of image files to load.
        """
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory not found or is not a directory: {self.root_dir}")

        # Recursively find all image files with specified extensions
        self.image_paths = []
        logger.info(f"Searching for images with extensions {IMAGE_EXTENSIONS} in {self.root_dir}...")
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(list(self.root_dir.rglob(f"*{ext}")))

        # Remove duplicates if any and sort
        self.image_paths = sorted(list(set(self.image_paths)))

        if not self.image_paths:
            raise FileNotFoundError(f"No image files with extensions {IMAGE_EXTENSIONS} found in {self.root_dir} or its subdirectories.")

        # Apply max_files limit
        if max_files is not None and max_files > 0:
            if len(self.image_paths) > max_files:
                logger.info(f"Found {len(self.image_paths)} images, limiting to first {max_files}.")
                self.image_paths = self.image_paths[:max_files]
            else:
                 logger.info(f"Found {len(self.image_paths)} images (requested max: {max_files}). Using all found images.")
        else:
             logger.info(f"Found {len(self.image_paths)} images in {self.root_dir}")


    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads and returns an image and a dummy label (0).

        Args:
            idx: Index of the image to retrieve.

        Returns:
            A tuple containing (image_tensor, 0).
            If loading fails, returns a default black image.
        """
        if idx >= len(self.image_paths):
             raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.image_paths)}")

        img_path = self.image_paths[idx]
        try:
            # Open image and ensure it's RGB
            image = Image.open(img_path).convert("RGB")

            # Apply transform if provided
            if self.transform:
                image = self.transform(image) # Assume transform outputs Tensor

            # Return dummy label 0
            return image, 0

        except FileNotFoundError:
            logger.error(f"Image file not found at {img_path}. Returning default.")
            image = Image.new("RGB", DEFAULT_SIZE, color="black")
        except UnidentifiedImageError:
            logger.error(f"Could not identify image file format for {img_path}. Returning default.")
            image = Image.new("RGB", DEFAULT_SIZE, color="black")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}. Returning default.")
            image = Image.new("RGB", DEFAULT_SIZE, color="black")

        # Apply transform even to default image (to maintain output type consistency)
        if self.transform:
            image = self.transform(image)

        return image, 0


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for example run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Example for SegmentationDataset (Cityscapes) ---
    # Define dummy transforms for demonstration
    from torchvision import transforms
    img_transform = transforms.Compose([
        transforms.Resize((256, 512)), # Example resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Mask transform: Only spatial transforms, no normalization!
    # ToTensor converts PIL to FloatTensor [0,1]. We want LongTensor [0, N-1].
    # So, usually apply spatial transforms first (as PIL), then convert to tensor manually.
    mask_transform = transforms.Compose([
         transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST), # Use NEAREST for masks
    ]) # Manual conversion to tensor happens inside __getitem__

    # --- !!! IMPORTANT: SET CORRECT PATHS FOR YOUR SYSTEM !!! ---
    # Adjust these paths to your actual Cityscapes dataset location
    cityscapes_img_dir = Path("./data/cityscapes/leftImg8bit/train") # Example path
    cityscapes_mask_dir = Path("./data/cityscapes/gtFine/train")    # Example path

    # Create dummy directories and files for testing if needed
    # (Create these paths if they don't exist before running)
    # Example:
    # cityscapes_img_dir.mkdir(parents=True, exist_ok=True)
    # (cityscapes_img_dir / "dummy_city").mkdir(exist_ok=True)
    # Image.new("RGB", (1024, 512)).save(cityscapes_img_dir / "dummy_city" / "dummy_000000_000000_leftImg8bit.png")
    #
    # cityscapes_mask_dir.mkdir(parents=True, exist_ok=True)
    # (cityscapes_mask_dir / "dummy_city").mkdir(exist_ok=True)
    # Image.new("L", (1024, 512), color=10).save(cityscapes_mask_dir / "dummy_city" / "dummy_000000_000000_gtFine_labelIds.png")


    try:
        print("\n--- Testing SegmentationDataset ---")
        seg_dataset = SegmentationDataset(
            image_dir=cityscapes_img_dir,
            mask_dir=cityscapes_mask_dir,
            transform=img_transform,
            mask_transform=mask_transform # Pass the spatial transform part
            # Suffixes default to Cityscapes, no need to specify unless different
        )
        print(f"SegmentationDataset created. Number of items: {len(seg_dataset)}")

        if len(seg_dataset) > 0:
            # Load one sample
            image_tensor, mask_tensor = seg_dataset[0]
            print(f"Loaded sample 0:")
            print(f"  Image tensor type: {type(image_tensor)}, shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            print(f"  Mask tensor type: {type(mask_tensor)}, shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
            # Check mask values (optional)
            unique_labels = torch.unique(mask_tensor)
            print(f"  Unique mask labels in sample 0: {unique_labels}")
        else:
            print("SegmentationDataset is empty, cannot load sample.")

    except FileNotFoundError as e:
        print(f"Error creating SegmentationDataset: {e}")
        print("Please ensure the dummy paths/files exist or update paths to your actual dataset.")
    except Exception as e:
        print(f"An unexpected error occurred with SegmentationDataset: {e}")


    # --- Example for SimpleImageDataset ---
    # Use the same image directory as above for testing
    try:
        print("\n--- Testing SimpleImageDataset ---")
        simple_dataset = SimpleImageDataset(
            root_dir=cityscapes_img_dir, # Use the same image source
            transform=img_transform,
            max_files=10 # Optional: limit number of files
        )
        print(f"SimpleImageDataset created. Number of items: {len(simple_dataset)}")

        if len(simple_dataset) > 0:
             # Load one sample
            image_tensor, label = simple_dataset[0]
            print(f"Loaded sample 0:")
            print(f"  Image tensor type: {type(image_tensor)}, shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            print(f"  Label: {label}")
        else:
             print("SimpleImageDataset is empty, cannot load sample.")

    except FileNotFoundError as e:
         print(f"Error creating SimpleImageDataset: {e}")
         print("Please ensure the image directory path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred with SimpleImageDataset: {e}")
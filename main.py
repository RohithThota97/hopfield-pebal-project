# -*- coding: utf-8 -*-
"""
Main training script for the Hopfield-PEBAL model for Out-of-Distribution Detection
in Semantic Segmentation.
"""

import os
# Set environment variable early to help avoid fragmentation, especially on multi-GPU systems.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import logging
import sys
import random
import importlib.util
from typing import Tuple, Optional, List # Added List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np

# Import custom modules
# Ensure the paths to custom modules are correct or add them to PYTHONPATH
try:
    from datasets import SegmentationDataset, SimpleImageDataset
    # Ensure the model class name matches the file (HopfieldPEBALModel in hopfield_pebal_model.py)
    from hopfield_pebal_model import HopfieldPEBALModel
    from hopfield_pebal_loss import HopfieldPEBALLoss
    from trainer import train_hopfield_pebal
    from pebal_integration import integrate_pebal_weights
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'datasets.py', 'hopfield_pebal_model.py', 'hopfield_pebal_loss.py', "
          "'trainer.py', and 'pebal_integration.py' are in the Python path.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Hopfield-PEBAL")

# --- Reproducibility ---
def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Ensure deterministic algorithms are used when possible
        # Note: some GPU operations are inherently non-deterministic
        # torch.use_deterministic_algorithms(True) # Can cause errors if ops not supported
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Can impact performance, but needed for determinism
    logger.info(f"Set random seed to {seed}")

# --- Argument Parsing ---
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Hopfield-PEBAL model for OOD detection in Semantic Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Dataset Paths ---
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--cityscapes_train_images', type=str,
                        default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/train',
                        help='Path to Cityscapes training images directory.')
    data_group.add_argument('--cityscapes_train_labels', type=str,
                        default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/train',
                        help='Path to Cityscapes training labels directory.')
    data_group.add_argument('--cityscapes_val_images', type=str,
                        default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/val',
                        help='Path to Cityscapes validation images directory.')
    data_group.add_argument('--cityscapes_val_labels', type=str,
                        default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/val',
                        help='Path to Cityscapes validation labels directory.')
    data_group.add_argument('--aux_images', type=str,
                        default='/home/ha51dybi/PEBAL/coco/train2017',
                        help='Path to auxiliary (OOD) images directory (e.g., COCO).')
    data_group.add_argument('--image_height', type=int, default=256,
                        help='Target image height for resizing.')
    data_group.add_argument('--image_width', type=int, default=512,
                        help='Target image width for resizing.')

    # --- Model Architecture ---
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--num_classes', type=int, default=19,
                        help='Number of semantic segmentation classes (e.g., Cityscapes has 19).')
    # Renamed feature_dim to hopfield_feature_dim to match model __init__
    model_group.add_argument('--hopfield_feature_dim', type=int, default=256,
                        help='Internal feature dimension for the Hopfield layer.')
    model_group.add_argument('--hopfield_beta', type=float, default=8.0,
                        help='Beta parameter (inverse temperature) for Hopfield association.')
    model_group.add_argument('--hopfield_memory_size', type=int, default=2000, # Renamed
                        help='Maximum size of the Hopfield memory bank.')
    model_group.add_argument('--hopfield_num_heads', type=int, default=4, # Renamed
                        help='Number of attention heads in the Hopfield layer.')
    model_group.add_argument('--insertion_point', type=str, default='after_backbone',
                        choices=['after_backbone', 'after_seghead'],
                        help='Location to insert the Hopfield layer in the segmentation model.')
    # Added target_feature_dim to match model init
    model_group.add_argument('--target_feature_dim', type=int, default=None,
                             help='Target dim for seg head input/Hopfield proj. Inferred if None.')
    model_group.add_argument('--use_efficient_memory', action='store_true',
                        help='Enable memory-efficient techniques (chunking, sampling).')
    model_group.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for processing large sequences if efficient memory is used.')
    model_group.add_argument('--sampling_stride', type=int, default=2,
                        help='Stride for spatial sampling before Hopfield if input is large.')
    model_group.add_argument('--use_simple_model', action='store_true',
                        help='Use a basic CNN model instead of DeepWV3Plus (for testing/debugging).')

    # --- Loss Function ---
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight multiplier for the standard segmentation loss.')
    loss_group.add_argument('--energy_weight', type=float, default=0.5,
                        help='Weight multiplier for the energy-based OOD loss.')
    loss_group.add_argument('--hopfield_weight', type=float, default=0.5,
                        help='Weight multiplier for the Hopfield contrastive loss.')
    loss_group.add_argument('--inlier_margin', type=float, default=1.0,
                        help='Margin for inlier energy (lower energy is better).')
    loss_group.add_argument('--outlier_margin', type=float, default=10.0,
                        help='Margin for outlier energy (higher energy is better).')
    loss_group.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature scaling for PEBAL energy calculation.')

    # --- Training Parameters ---
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU for training.')
    train_group.add_argument('--num_epochs', type=int, default=50,
                        help='Total number of training epochs.')
    train_group.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Base learning rate for the optimizer.')
    train_group.add_argument('--backbone_lr_factor', type=float, default=0.1,
                        help='Multiplier for the learning rate of the backbone weights.')
    train_group.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty) for the optimizer.')
    train_group.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading.')
    # Mixed precision is handled inside trainer now, but keep arg if needed elsewhere
    train_group.add_argument('--mixed_precision', action='store_true',
                        help='Enable Automatic Mixed Precision (currently forced False in trainer).')
    train_group.add_argument('--memory_update_freq', type=int, default=20,
                        help='Frequency (in batches) to update the Hopfield memory bank.')
    train_group.add_argument('--memory_update_batches', type=int, default=10,
                        help='Number of batches used to collect features for memory update.')

    # --- Miscellaneous ---
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    misc_group.add_argument('--save_path', type=str, default='./checkpoints/hopfield_pebal',
                        help='Directory to save model checkpoints and logs.')
    misc_group.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint file to resume training from.')
    misc_group.add_argument('--pebal_checkpoint', type=str, default=None,
                        help='Path to a pre-trained PEBAL checkpoint to initialize weights.')
    misc_group.add_argument('--debug', action='store_true',
                        help='Enable debug mode: uses smaller datasets and enables anomaly detection.')
    misc_group.add_argument('--debug_samples', type=int, default=50,
                        help='Number of samples per dataset to use in debug mode.')

    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.cityscapes_train_images):
        logger.warning(f"Cityscapes train images directory not found: {args.cityscapes_train_images}")
    # Check label paths too
    if not os.path.isdir(args.cityscapes_train_labels):
         logger.warning(f"Cityscapes train labels directory not found: {args.cityscapes_train_labels}")
    if not os.path.isdir(args.cityscapes_val_labels):
         logger.warning(f"Cityscapes val labels directory not found: {args.cityscapes_val_labels}")
    if not os.path.isdir(args.aux_images):
        logger.warning(f"Auxiliary images directory not found: {args.aux_images}")

    return args

# --- Model Loading Helpers ---
def create_simple_backbone_for_testing(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """Creates a simple CNN backbone and segmentation head for testing purposes."""
    class SimpleBackbone(nn.Module):
        def __init__(self, out_channels=128): # Make output dim configurable
            super().__init__()
            self.out_channels = out_channels
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # Adjust final conv to output desired channels
            self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
            x = self.pool1(x)
            x = self.conv2(x); x = self.bn2(x); x = self.relu(x)
            return x

    class SimpleSegHead(nn.Module):
        def __init__(self, in_channels: int, num_classes: int):
            super().__init__()
            # Simple head: 1x1 convolution to classify features
            self.conv = nn.Conv2d(in_channels, num_classes + 1, kernel_size=1) # Output N_classes + 1 for PEBAL energy

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Optional: Upsample features *before* classification if needed
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            return self.conv(x)

    logger.info("Created simple backbone and segmentation head for testing.")
    # Match the init signature of HopfieldPEBALModel's DummyBackbone example
    backbone = SimpleBackbone(out_channels=128)
    # Seg head expects input from backbone (or hopfield output proj)
    seg_head = SimpleSegHead(in_channels=backbone.out_channels, num_classes=num_classes)
    return backbone, seg_head

def import_and_get_deepwv3plus(num_classes: int) -> Optional[Tuple[nn.Module, nn.Module]]:
    """
    Attempts to import the DeepWV3Plus model and extract its backbone and head.
    Returns (backbone, segmentation_head) or None on failure.
    """
    # --- Attempt to find the model code ---

    # ***** CORRECTED PATH CALCULATION *****
    # Look for 'code' directory directly inside the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(script_dir, 'code')
    # *************************************

    if code_dir not in sys.path:
        sys.path.insert(0, code_dir) # Prepend to path
        logger.info(f"Added '{code_dir}' to Python path") # Will now log the correct path

    module_path = os.path.join(code_dir, 'model', 'wide_network.py')
    logger.info(f"Attempting to load DeepWV3Plus source from: {module_path}") # Log the path being checked

    if not os.path.exists(module_path):
        logger.error(f"DeepWV3Plus source file not found at: {module_path}")
        logger.error("Ensure the 'code' directory containing 'model/wide_network.py' exists in the same directory as main.py.")
        return None

    try:
        # --- Import the module ---
        spec = importlib.util.spec_from_file_location("wide_network", module_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not create module spec for {module_path}")
            return None
        wide_network_module = importlib.util.module_from_spec(spec)
        # Add module to sys.modules to allow potential relative imports within wide_network.py
        sys.modules["wide_network"] = wide_network_module
        spec.loader.exec_module(wide_network_module) # Potential error source if module has issues

        if not hasattr(wide_network_module, 'DeepWV3Plus'):
            logger.error("DeepWV3Plus class not found in the loaded module.")
            return None

        DeepWV3Plus = wide_network_module.DeepWV3Plus
        logger.info("Successfully imported DeepWV3Plus model.")

        # --- Instantiate and Split the Model ---
        # The model might take args/kwargs, check its __init__
        full_model = DeepWV3Plus(num_classes=num_classes)
        logger.info("Instantiated DeepWV3Plus.")

        # --- Define wrappers (Robust version) ---
        class BackboneWrapper(nn.Module):
             def __init__(self, model):
                 super().__init__()
                 # Try common WideResNet structure first
                 known_modules = ['mod1', 'pool2', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7']
                 features_list = []
                 try:
                      current_module = model
                      for name in known_modules:
                           module = getattr(current_module, name)
                           features_list.append(module)
                           # Special handling if some modules are nested differently might be needed
                      self.features = nn.Sequential(*features_list)
                      # Determine output channels (requires knowledge of model or probing)
                      self.output_channels = 2048 # Typical for WideResNet38
                      logger.info("Using known WideResNet38 structure for backbone.")
                 except AttributeError as e:
                      logger.warning(f"Could not find all expected DeepWV3Plus modules (modX/pool2): {e}")
                      logger.warning("Using fallback: Assuming first N-1 children are backbone.")
                      children = list(model.children())
                      if len(children) > 1:
                           self.features = nn.Sequential(*children[:-1])
                           # Attempt to infer output channels for fallback
                           self._infer_output_channels(model.device if hasattr(model, 'device') else 'cpu')
                      else:
                           logger.error("Cannot split model automatically in fallback.")
                           raise AttributeError("Cannot determine backbone modules.") from e

             def _infer_output_channels(self, device):
                 try:
                      dummy_input = torch.randn(1, 3, 64, 64, device=device) # Small dummy input
                      with torch.no_grad():
                           dummy_out = self.features(dummy_input)
                      self.output_channels = dummy_out.shape[1]
                      logger.info(f"Fallback inferred backbone output channels: {self.output_channels}")
                 except Exception as infer_e:
                      logger.error(f"Could not infer backbone output channels in fallback: {infer_e}")
                      self.output_channels = 2048 # Fallback guess

             def forward(self, x):
                 return self.features(x)

        class SegHeadWrapper(nn.Module):
             def __init__(self, model):
                 super().__init__()
                 # Capture the final classification layer(s)
                 try:
                     # Try common names like 'final', 'classifier', 'seg_head'
                     if hasattr(model, 'final'): self.classifier = model.final
                     elif hasattr(model, 'classifier'): self.classifier = model.classifier
                     elif hasattr(model, 'seg_head'): self.classifier = model.seg_head
                     else: raise AttributeError # Force fallback if known names fail
                 except AttributeError:
                     logger.warning("Could not find common head names ('final', 'classifier', 'seg_head').")
                     logger.warning("Using fallback: Assuming last child module is the head.")
                     children = list(model.children())
                     if len(children) > 1:
                          self.classifier = children[-1]
                     else:
                          logger.error("Cannot determine segmentation head module automatically.")
                          raise AttributeError("Cannot determine segmentation head module.")

             def forward(self, x):
                 return self.classifier(x)

        # Use the wrappers
        try:
             backbone = BackboneWrapper(full_model)
             segmentation_head = SegHeadWrapper(full_model)
             logger.info("Extracted backbone and segmentation head from DeepWV3Plus.")
             # Ensure backbone has output_channels attribute
             if not hasattr(backbone, 'output_channels'):
                 logger.warning("Backbone wrapper missing 'output_channels'. Attempting inference.")
                 backbone._infer_output_channels(full_model.device if hasattr(full_model, 'device') else 'cpu')

             return backbone, segmentation_head
        except AttributeError: # Catch errors from wrapper initialization
             logger.error("Failed to automatically split DeepWV3Plus using wrappers. Needs manual adaptation.")
             return None

    except ImportError as e:
        logger.error(f"Import error during DeepWV3Plus loading: {e}", exc_info=True)
        return None
    except AttributeError as e:
        logger.error(f"Attribute error, likely model structure mismatch or wrapper issue: {e}", exc_info=True)
        return None
    except Exception as e:
        # Log the full traceback for unexpected errors
        logger.exception(f"An unexpected error occurred while loading/splitting DeepWV3Plus: {e}")
        return None

# --- Main Function ---
def main():
    """Main function to set up and run the training process."""
    args = parse_args()
    set_seed(args.seed)

    # --- Setup ---
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.debug:
        logger.warning("--- DEBUG MODE ENABLED ---")
        logger.info("Anomaly detection enabled. Training will be slower.")
        torch.autograd.set_detect_anomaly(True)
    else:
         torch.autograd.set_detect_anomaly(False) # Ensure it's off for normal runs

    # --- Data Transforms ---
    target_size = (args.image_height, args.image_width)
    logger.info(f"Using image size: {target_size}")

    # Normalization values (usually ImageNet)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Consider adding more augmentations if needed
    train_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Slightly stronger jitter
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3), # Optional blur
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Specific transform for masks (no normalization, nearest interpolation)
    mask_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        # Squeeze channel dim & convert to Long. Handle potential float masks (0.0-1.0).
        transforms.Lambda(lambda x: (x.squeeze(0) * 255).long() if (x.ndim == 3 and x.max() <= 1.0 and x.min() >= 0) else x.squeeze(0).long())
    ])

    # --- Datasets ---
    logger.info("Loading datasets...")
    try:
        train_dataset = SegmentationDataset(
            args.cityscapes_train_images,
            args.cityscapes_train_labels,
            transform=train_transform,
            mask_transform=mask_transform,
            num_classes=args.num_classes # Pass num_classes if needed by dataset
        )
        val_dataset = SegmentationDataset(
            args.cityscapes_val_images,
            args.cityscapes_val_labels,
            transform=val_transform,
            mask_transform=mask_transform,
            num_classes=args.num_classes # Pass num_classes if needed by dataset
        )
        # Use val_transform for auxiliary data if no augmentation is desired
        aux_dataset = SimpleImageDataset(
            args.aux_images,
            transform=val_transform, # Use validation transform (no augmentation)
            max_files=None if not args.debug else args.debug_samples
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset directory error: {e}. Please check dataset paths.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating datasets: {e}", exc_info=True)
        sys.exit(1)

    # Reduce dataset size for debug mode
    if args.debug:
        logger.info(f"Reducing datasets to {args.debug_samples} samples for debugging.")
        train_len = len(train_dataset)
        val_len = len(val_dataset)
        if train_len > args.debug_samples:
            train_indices = torch.randperm(train_len)[:args.debug_samples]
            train_dataset = Subset(train_dataset, train_indices.tolist())
        if val_len > args.debug_samples:
            val_indices = torch.randperm(val_len)[:args.debug_samples]
            val_dataset = Subset(val_dataset, val_indices.tolist())
        # aux_dataset already limited by max_files if debug is True

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Auxiliary samples: {len(aux_dataset)}")

    # --- Data Loaders ---
    # Consider persistent_workers=True if num_workers > 0 for potential speedup
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    # Create aux_loader only if dataset has items
    if len(aux_dataset) > 0:
        aux_loader = DataLoader(
            aux_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(args.num_workers > 0)
        )
    else:
        logger.warning("Auxiliary dataset is empty or could not be loaded. Proceeding without auxiliary data.")
        aux_loader = None


    # --- Model Initialization ---
    logger.info("Creating model...")
    backbone: Optional[nn.Module] = None
    segmentation_head: Optional[nn.Module] = None

    if args.use_simple_model:
        logger.info("Using simple test model.")
        backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
    else:
        logger.info("Attempting to load and use DeepWV3Plus model.")
        # Pass num_classes needed by DeepWV3Plus constructor
        model_components = import_and_get_deepwv3plus(args.num_classes)
        if model_components:
            backbone, segmentation_head = model_components
            logger.info("Successfully loaded and prepared DeepWV3Plus.")
        else:
            logger.warning("Failed to load DeepWV3Plus. Falling back to simple test model.")
            backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)

    if backbone is None or segmentation_head is None:
        logger.error("Model creation failed (backbone or head is None). Exiting.")
        sys.exit(1)

    # Ensure backbone has output_channels attribute BEFORE passing to HopfieldPEBALModel
    # This might be set within the wrapper or needs inference here.
    if not hasattr(backbone, 'output_channels'):
        logger.warning("Backbone missing 'output_channels' attribute. Attempting inference.")
        try:
            # Move backbone to device before inference if not already there
            backbone_device = next(backbone.parameters()).device
            dummy_input = torch.randn(1, 3, args.image_height // 4, args.image_width // 4, device=backbone_device) # Smaller dummy input
            with torch.no_grad():
                backbone_output = backbone(dummy_input)
            backbone.output_channels = backbone_output.shape[1]
            logger.info(f"Inferred backbone output channels: {backbone.output_channels}")
        except Exception as e:
            logger.error(f"Could not infer backbone output channels: {e}. Feature dim might be incorrect.", exc_info=True)
            # Assign a default based on model type
            backbone_output_channels = 2048 if not args.use_simple_model else 128
            logger.warning(f"Assuming backbone output channels: {backbone_output_channels}")
            backbone.output_channels = backbone_output_channels # Set the attribute


    # Move base model components to the target device *before* initializing the main model
    backbone = backbone.to(device)
    segmentation_head = segmentation_head.to(device)
    logger.info(f"Backbone and SegHead moved to device: {device}")

    # Create the main Hopfield-PEBAL model
    try:
        model = HopfieldPEBALModel(
            backbone=backbone,
            segmentation_head=segmentation_head,
            num_classes=args.num_classes,
            hopfield_feature_dim=args.hopfield_feature_dim, # Use renamed arg
            hopfield_beta=args.hopfield_beta,
            hopfield_memory_size=args.hopfield_memory_size, # Use renamed arg
            hopfield_num_heads=args.hopfield_num_heads, # Use renamed arg
            insertion_point=args.insertion_point,
            target_feature_dim=args.target_feature_dim, # Pass the new arg
            use_efficient_memory=args.use_efficient_memory,
            chunk_size=args.chunk_size,
            sampling_stride=args.sampling_stride, # Pass the new arg
            # Pass memory logging args if needed
            # memory_log_interval=...,
            # memory_log_verbose=...
        ).to(device) # Ensure the final combined model is on the device
    except Exception as model_init_e:
         logger.error(f"Error initializing HopfieldPEBALModel: {model_init_e}", exc_info=True)
         sys.exit(1)


    logger.info(f"Hopfield-PEBAL model created. Insertion point: {args.insertion_point}")
    # Log parameter count after model is fully initialized and on device
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")


    # --- PEBAL Weight Integration (Optional) ---
    if args.pebal_checkpoint:
        logger.info(f"Attempting to integrate weights from PEBAL checkpoint: {args.pebal_checkpoint}")
        if os.path.isfile(args.pebal_checkpoint):
            try:
                # Assuming integrate_pebal_weights modifies model in-place or returns a new one
                model = integrate_pebal_weights(model, args.pebal_checkpoint, device)
                logger.info("Successfully integrated weights from PEBAL checkpoint.")
            except Exception as e:
                logger.error(f"Failed to integrate PEBAL weights: {e}. Check model compatibility and checkpoint file.", exc_info=True)
                # Decide whether to continue or exit
                # sys.exit(1)
        else:
             logger.error(f"PEBAL checkpoint file not found at {args.pebal_checkpoint}. Skipping integration.")


    # --- Loss Function ---
    try:
        criterion = HopfieldPEBALLoss(
            num_classes=args.num_classes,
            seg_weight=args.seg_weight,
            energy_weight=args.energy_weight,
            hopfield_weight=args.hopfield_weight,
            inlier_margin=args.inlier_margin,
            outlier_margin=args.outlier_margin,
            temperature=args.temperature,
            ignore_index=255 # Common ignore index for Cityscapes
        ).to(device)
        logger.info(f"Loss function created with weights: Seg={args.seg_weight}, Energy={args.energy_weight}, Hopfield={args.hopfield_weight}")
    except Exception as loss_init_e:
         logger.error(f"Error initializing HopfieldPEBALLoss: {loss_init_e}", exc_info=True)
         sys.exit(1)

    # --- Optimizer ---
    # Parameter groups for differential learning rates
    try:
         # Identify parameters based on submodule names (adjust if needed)
         backbone_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
         hopfield_params = [p for n, p in model.named_parameters() if n.startswith('hopfield.') or n.startswith('hopfield_input_proj.') or n.startswith('hopfield_output_proj.') and p.requires_grad]
         adapter_params = [p for n, p in model.named_parameters() if n.startswith('channel_adapter.') and p.requires_grad]
         seg_head_params = [p for n, p in model.named_parameters() if n.startswith('segmentation_head.') or n.startswith('final_classifier.') or n.startswith('energy_head.') and p.requires_grad]

         # Collect all used param ids to find remaining ones
         used_param_ids = set(id(p) for p_list in [backbone_params, hopfield_params, adapter_params, seg_head_params] for p in p_list)
         other_params = [p for p in model.parameters() if id(p) not in used_param_ids and p.requires_grad]

         param_groups = [
             {'params': backbone_params, 'lr': args.learning_rate * args.backbone_lr_factor, 'name': 'backbone'},
             {'params': hopfield_params, 'lr': args.learning_rate, 'name': 'hopfield'},
             {'params': adapter_params, 'lr': args.learning_rate, 'name': 'adapter'},
             {'params': seg_head_params, 'lr': args.learning_rate, 'name': 'head_related'},
             {'params': other_params, 'lr': args.learning_rate, 'name': 'other'} # Catch any remaining params
         ]

         # Filter out groups with no parameters
         param_groups = [pg for pg in param_groups if len(pg['params']) > 0]

         optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
         logger.info(f"Optimizer: AdamW (BaseLR={args.learning_rate}, BackboneLRFactor={args.backbone_lr_factor}, WD={args.weight_decay})")
         logger.info(f"Parameter groups created:")
         for pg in param_groups:
              logger.info(f"  Group '{pg['name']}': {len(pg['params'])} params, LR={pg['lr']:.2e}")

    except Exception as optim_e:
         logger.error(f"Error setting up optimizer: {optim_e}", exc_info=True)
         sys.exit(1)

    # --- Learning Rate Scheduler ---
    # Example: Cosine Annealing
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01)
    # logger.info("LR Scheduler: CosineAnnealingLR")

    # Example: ReduceLROnPlateau (as before)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False, # verbose=False to avoid duplicate logging with manual check
        threshold=0.01, threshold_mode='rel', cooldown=1, min_lr=1e-7 # Added cooldown, min_lr
    )
    logger.info("LR Scheduler: ReduceLROnPlateau (monitors validation loss)")


    # --- Resume Training (Optional) ---
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint for resuming training from: {args.resume}")
        try:
            # Load checkpoint to CPU first to avoid GPU mem spike if model is large
            checkpoint = torch.load(args.resume, map_location='cpu')

            # Load model state - handle potential missing/extra keys
            model.load_state_dict(checkpoint['model_state_dict'], strict=False) # Use strict=False initially

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                 try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 except Exception as e:
                     logger.warning(f"Could not load scheduler state: {e}. Resetting scheduler.")

            start_epoch = checkpoint.get('epoch', 0) + 1 # Start from next epoch
            best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Resume best loss

            # Load memory bank if saved
            if 'memory_bank' in checkpoint and checkpoint['memory_bank'] is not None:
                 if hasattr(model, 'hopfield') and hasattr(model.hopfield, 'set_memory'):
                      # Ensure memory bank is moved to the correct device
                      memory_bank_state = checkpoint['memory_bank'].to(device)
                      model.hopfield.set_memory(memory_bank_state)
                      logger.info(f"Loaded Hopfield memory bank ({memory_bank_state.shape}) from checkpoint.")
                 else:
                      logger.warning("Checkpoint contains 'memory_bank', but model.hopfield has no 'set_memory' method.")

            logger.info(f"Successfully resumed from epoch {start_epoch - 1}. Best validation loss so far: {best_val_loss:.4f}")

            # Move model back to target device after loading state dict
            model.to(device)

        except Exception as e:
            logger.error(f"Error loading checkpoint '{args.resume}': {e}. Starting training from scratch.", exc_info=True)
            start_epoch = 0
            # Re-initialize model on device if loading failed badly
            model.to(device)

    else:
        if args.resume:
            logger.warning(f"Resume checkpoint not found at '{args.resume}'. Starting training from scratch.")
        else:
            logger.info("Starting training from scratch.")


    # --- Log Final Configuration ---
    logger.info("--- Training Configuration ---")
    for arg, value in sorted(vars(args).items()):
         logger.info(f"  {arg}: {value}")
    logger.info("--- End Configuration ---")


    # --- Start Training ---
    logger.info("Starting training loop...")
    try:
        # Pass all necessary args to the trainer function
        trained_model = train_hopfield_pebal(
            train_loader=train_loader,
            val_loader=val_loader,
            aux_loader=aux_loader, # Pass aux_loader, could be None
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            start_epoch=start_epoch, # Pass start_epoch for resuming
            num_epochs=args.num_epochs,
            device=device,
            scheduler=scheduler, # Pass the scheduler instance
            save_path=args.save_path,
            memory_update_freq=args.memory_update_freq,
            memory_update_batches=args.memory_update_batches,
            mixed_precision=args.mixed_precision, # Trainer handles forcing False if needed
            use_efficient_memory=args.use_efficient_memory, # Pass efficiency flags
            chunk_size=args.chunk_size
        )
    except KeyboardInterrupt:
         logger.warning("Training interrupted by user (KeyboardInterrupt).")
         # Optionally save state before exiting
         # final_model_path = os.path.join(args.save_path, "interrupt_model.pth")
         # ... save logic ...
         sys.exit(0) # Clean exit
    except Exception as e:
        logger.exception(f"An critical error occurred during training: {e}") # Log full traceback
        sys.exit(1) # Exit after logging the error


    # --- Save Final Model ---
    # Consider saving the best model again here, in case the last epoch wasn't the best
    final_model_path = os.path.join(args.save_path, "final_model_state.pth")
    logger.info(f"Saving final model state to {final_model_path}")
    try:
        # Use the state dict of the returned model (which should have best weights loaded by trainer)
        final_state = {
            'epoch': args.num_epochs - 1, # Index of the last completed epoch
            'model_state_dict': trained_model.state_dict(),
            # Save optimizer/scheduler states if needed for fine-tuning later
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'args': vars(args) # Save args for reference
        }
        # Optionally save memory bank
        if hasattr(trained_model, 'hopfield') and hasattr(trained_model.hopfield, 'get_memory'):
             final_state['memory_bank'] = trained_model.hopfield.get_memory().cpu() # Get final memory

        torch.save(final_state, final_model_path)
        logger.info("Final model state saved successfully.")
    except Exception as e:
        logger.error(f"Error saving final model state: {e}", exc_info=True)

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
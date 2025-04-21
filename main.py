# main.py
# -*- coding: utf-8 -*-
"""
Main training script for the Hopfield-PEBAL model using Efficient Memory Manager
for Out-of-Distribution Detection in Semantic Segmentation.
"""

import os
# Set environment variable early - Helps prevent fragmentation but might increase overall reserved memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Alternative: 'max_split_size_mb:512'

import argparse
import logging
import sys
import random
import importlib.util
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np

# Import custom modules
try:
    # IMPORTANT: Ensure SegmentationDataset maps raw IDs -> train IDs (0-18) + ignore_index (255)!
    from datasets import SegmentationDataset, SimpleImageDataset
    from hopfield_pebal_model import HopfieldPEBALModel, EfficientSegmentationDecoder
    from hopfield_pebal_loss import HopfieldPEBALLoss
    from trainer import train_hopfield_pebal
    from pebal_integration import integrate_pebal_weights
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'datasets.py', 'hopfield_pebal_model.py', 'hopfield_pebal_loss.py', "
          "'trainer.py', and 'pebal_integration.py' are in the Python path or the same directory.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        print(f"Adding parent directory '{parent_dir}' to sys.path to search for modules.")
        sys.path.insert(0, parent_dir)
        try:
            from datasets import SegmentationDataset, SimpleImageDataset
            from hopfield_pebal_model import HopfieldPEBALModel, EfficientSegmentationDecoder
            from hopfield_pebal_loss import HopfieldPEBALLoss
            from trainer import train_hopfield_pebal
            from pebal_integration import integrate_pebal_weights
            print("Successfully imported custom modules after path addition.")
        except ImportError:
            print("Failed to import custom modules even after adding parent directory to path. Please check your project structure and PYTHONPATH.")
            sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Hopfield-PEBAL")
# Ensure dataset refinement logger (if used in datasets.py) also uses the main handler level
logging.getLogger('DatasetRefinement').setLevel(logging.INFO) # Match level

# --- Reproducibility ---
def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If you still get CUDA errors, uncommenting these might help pinpoint non-determinism, but slows things down.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # logger.info("Note: cudnn.deterministic=True, cudnn.benchmark=False set for reproducibility.")
    logger.info(f"Set random seed to {seed}")

# --- Argument Parsing ---
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Hopfield-PEBAL model with Efficient Memory Manager.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Dataset Paths ---
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--cityscapes_train_images', type=str, required=True, help='Path to Cityscapes training images directory.')
    # Clarified expected filename pattern for labels
    data_group.add_argument('--cityscapes_train_labels', type=str, required=True, help='Path to Cityscapes training labels directory (containing *_labelIds.png).')
    data_group.add_argument('--cityscapes_val_images', type=str, required=True, help='Path to Cityscapes validation images directory.')
    data_group.add_argument('--cityscapes_val_labels', type=str, required=True, help='Path to Cityscapes validation labels directory (containing *_labelIds.png).')
    data_group.add_argument('--aux_images', type=str, required=True, help='Path to auxiliary (OOD) images directory (e.g., COCO).')
    data_group.add_argument('--image_height', type=int, default=256, help='Target image height for resizing.')
    data_group.add_argument('--image_width', type=int, default=512, help='Target image width for resizing.')

    # --- Model Architecture ---
    model_group = parser.add_argument_group('Model Configuration')
    # ***** IMPORTANT: num_classes must match the number of *trainId* classes (0-18 for Cityscapes) *****
    model_group.add_argument('--num_classes', type=int, default=19, help='Number of semantic segmentation trainId classes.')
    model_group.add_argument('--memory_feature_dim', type=int, default=256, help='Internal feature dimension for the Memory Manager.')
    model_group.add_argument('--memory_beta', type=float, default=8.0, help='Beta parameter (inverse temperature) for memory energy calculation.')
    model_group.add_argument('--memory_size', type=int, default=2000, help='Maximum size of the memory bank.')
    model_group.add_argument('--insertion_point', type=str, default='after_backbone', choices=['after_backbone', 'after_seghead'], help='Location for memory interaction.')
    model_group.add_argument('--target_feature_dim', type=int, default=None, help='Target dim after backbone/head adapter. Default inferred.')
    model_group.add_argument('--use_efficient_memory', action='store_true', default=False, help='Enable memory-efficient techniques (sampling, cache clearing).')
    model_group.add_argument('--use_faiss', action='store_true', default=False, help='Use FAISS for accelerated nearest neighbor search.')
    # model_group.add_argument('--pq_bytes', type=int, default=8, help='Bytes for Product Quantization in FAISS (if used).') # <-- REMOVED based on error
    model_group.add_argument('--sampling_stride', type=int, default=2, help='Stride for spatial sampling before memory interaction.')
    model_group.add_argument('--chunk_size', type=int, default=1000, help='Chunk size for processing large sequences (Efficient Decoder).') # This might be specific to EffDec, check usage
    model_group.add_argument('--use_efficient_decoder', action='store_true', default=False, help='Use the EfficientSegmentationDecoder.')
    model_group.add_argument('--use_simple_model', action='store_true', default=False, help='Use a basic CNN model instead of DeepWV3Plus.')

    # --- Loss Function ---
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument('--seg_weight', type=float, default=1.0, help='Weight for segmentation loss.')
    loss_group.add_argument('--energy_weight', type=float, default=0.5, help='Weight for energy-based OOD loss.')
    loss_group.add_argument('--hopfield_weight', type=float, default=0.0, help='Weight for Hopfield contrastive loss (Likely 0.0).')
    loss_group.add_argument('--inlier_margin', type=float, default=1.0, help='Margin for inlier energy.')
    loss_group.add_argument('--outlier_margin', type=float, default=10.0, help='Margin for outlier energy.')
    loss_group.add_argument('--temperature', type=float, default=1.0, help='Temperature scaling for PEBAL energy.')
    # ***** IMPORTANT: ignore_index must match the void label used in the mapped masks (255 for Cityscapes trainId) - Used by the Loss Function *****
    loss_group.add_argument('--ignore_index', type=int, default=255, help='Index to ignore in segmentation loss (used by Loss Function).')


    # --- Training Parameters ---
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU for training.')
    train_group.add_argument('--num_epochs', type=int, default=50, help='Total number of training epochs.')
    train_group.add_argument('--learning_rate', type=float, default=1e-4, help='Base learning rate.')
    train_group.add_argument('--backbone_lr_factor', type=float, default=0.1, help='Multiplier for backbone LR.')
    train_group.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    train_group.add_argument('--grad_clip_norm', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable).')
    train_group.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    train_group.add_argument('--mixed_precision', action='store_true', default=False, help='Enable Automatic Mixed Precision training.')
    train_group.add_argument('--memory_update_freq', type=int, default=20, help='Frequency (batches) to update memory.')
    train_group.add_argument('--memory_update_batches', type=int, default=10, help='Number of batches used for memory update.')

    # --- Miscellaneous ---
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--seed', type=int, default=42, help='Random seed.')
    misc_group.add_argument('--save_path', type=str, default='./checkpoints/hopfield_pebal_efficient', help='Save directory.')
    misc_group.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume.')
    misc_group.add_argument('--pebal_checkpoint', type=str, default=None, help='Path to PEBAL checkpoint.')
    misc_group.add_argument('--debug', action='store_true', help='Enable debug mode (subset of data).')
    misc_group.add_argument('--debug_samples', type=int, default=10, help='Number of samples per dataset in debug mode.') # Reduced default
    # Option to enable stricter CUDA debugging
    misc_group.add_argument('--cuda_launch_blocking', action='store_true', help='Set CUDA_LAUNCH_BLOCKING=1 for debugging asserts (slow).')

    args = parser.parse_args()

    # Validation - Check paths
    for path_arg in ['cityscapes_train_images', 'cityscapes_train_labels',
                     'cityscapes_val_images', 'cityscapes_val_labels', 'aux_images']:
        path_val = getattr(args, path_arg)
        if not os.path.exists(path_val):
             logger.error(f"Required dataset path not found: --{path_arg} {path_val}")
             sys.exit(1)
        # Aux can be a file list or dir, skip dir check for it
        # Check if directory paths are actually directories
        elif not os.path.isdir(path_val) and path_arg != 'aux_images' and (path_arg.endswith('images') or path_arg.endswith('labels')):
             logger.error(f"Path specified for --{path_arg} is not a directory: {path_val}")
             sys.exit(1)

    # Default target_feature_dim based on insertion point (can be handled later in model init if needed)
    if args.target_feature_dim is None:
        if args.insertion_point == 'after_backbone':
             # Initial guess, might be refined later based on actual backbone output
             args.target_feature_dim = 128 if args.use_simple_model else 4096 # Rough guess
             logger.info(f"Setting initial target_feature_dim guess for 'after_backbone': {args.target_feature_dim}")
        elif args.insertion_point == 'after_seghead':
             logger.info("target_feature_dim not specified for 'after_seghead'. Model will attempt to infer.")

    return args

# --- Model Loading Helpers ---
def create_simple_backbone_for_testing(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """Creates a simple CNN backbone and segmentation head for testing purposes."""
    class SimpleBackbone(nn.Module):
        def __init__(self, out_channels=128):
            super().__init__()
            # Define a simple sequential backbone
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.output_channels = out_channels # Explicitly store output channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.features(x)

    class SimpleSegHead(nn.Module):
        def __init__(self, in_channels: int, num_classes: int):
            super().__init__()
            # Simple 1x1 convolution head
            # Ensure output has num_classes channels for CrossEntropyLoss
            self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            self.output_channels = num_classes # Store output channels for consistency

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    logger.info(f"Created simple backbone ({128}ch) and segmentation head ({num_classes} classes).")
    backbone = SimpleBackbone(out_channels=128)
    # The segmentation head MUST output num_classes channels
    seg_head = SimpleSegHead(in_channels=128, num_classes=num_classes)
    return backbone, seg_head

# Corrected import_and_get_deepwv3plus function
def import_and_get_deepwv3plus(num_classes: int) -> Optional[Tuple[nn.Module, nn.Module]]:
    """
    Attempts to import the DeepWV3Plus model and extract its backbone and head.
    Returns (backbone, segmentation_head) or None on failure.
    (Includes refined wrapper logic for module identification and output channel verification)
    """
    # Look for 'code' directory relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try common locations for the 'code' directory
    potential_code_dirs = [
        os.path.join(script_dir, 'code'),             # Sibling to main script
        os.path.join(script_dir, '..', 'code'),       # Parent directory contains 'code'
        os.path.join(script_dir, 'hop-pebal', 'code') # Inside a 'hop-pebal' subdirectory
    ]

    code_dir = None
    for potential_dir in potential_code_dirs:
        if os.path.isdir(potential_dir):
            code_dir = os.path.abspath(potential_dir)
            logger.info(f"Found 'code' directory at: {code_dir}")
            break

    if code_dir is None:
        logger.error("Could not find the 'code' directory containing model definitions in expected locations.")
        return None

    added_to_path = False
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
        added_to_path = True
        # logger.info(f"Added '{code_dir}' to Python path") # Less verbose log

    module_path = os.path.join(code_dir, 'model', 'wide_network.py')
    logger.info(f"Attempting to load DeepWV3Plus source from: {module_path}")

    if not os.path.exists(module_path):
        logger.error(f"DeepWV3Plus source file not found at: {module_path}")
        if added_to_path:
            try: sys.path.remove(code_dir)
            except ValueError: pass
        return None

    backbone = None
    segmentation_head = None
    try:
        spec = importlib.util.spec_from_file_location("wide_network", module_path)
        if spec is None or spec.loader is None: raise ImportError("Could not create module spec")
        wide_network_module = importlib.util.module_from_spec(spec)
        sys.modules["wide_network"] = wide_network_module # Register module
        spec.loader.exec_module(wide_network_module)

        if not hasattr(wide_network_module, 'DeepWV3Plus'): raise AttributeError("DeepWV3Plus class not found")
        DeepWV3Plus = wide_network_module.DeepWV3Plus
        logger.info("Successfully imported DeepWV3Plus module.")

        # --- Instantiate Model (Handle Aux Classifier) ---
        full_model: Optional[nn.Module] = None
        has_aux = True # Assume aux exists unless instantiation fails
        try:
            # Try with aux=True first (common in segmentation)
            # Instantiate with num_classes for the *main* head
            full_model = DeepWV3Plus(num_classes=num_classes, aux=True)
            logger.info("Instantiated DeepWV3Plus with aux=True.")
        except TypeError:
            logger.warning("DeepWV3Plus instantiation with aux=True failed. Trying without 'aux' argument.")
            try:
                full_model = DeepWV3Plus(num_classes=num_classes)
                has_aux = False # Infer aux is not supported or default is false
                logger.info("Instantiated DeepWV3Plus (no 'aux' argument detected or needed).")
            except Exception as model_init_e:
                logger.error(f"Failed to instantiate DeepWV3Plus even without 'aux': {model_init_e}", exc_info=True)
                return None
        except Exception as model_init_e:
             logger.error(f"Failed to instantiate DeepWV3Plus with aux=True: {model_init_e}", exc_info=True)
             return None

        if full_model is None: # Should not happen if error handling above is correct, but check anyway
            logger.error("Model instantiation resulted in None.")
            return None

        # --- Define Wrappers (Refactored for Correct Syntax and Readability) ---
        class BackboneWrapper(nn.Module):
            # Stores output_channels attribute
            def __init__(self, model: nn.Module, has_aux_classifier: bool):
                super().__init__()
                self.features: Optional[nn.Module] = None
                self.output_channels: Optional[int] = None

                known_modules = ['mod1', 'pool2', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7']
                features_list = []
                current_module = model
                try:
                    for name in known_modules:
                        module = getattr(current_module, name)
                        features_list.append(module)
                    self.features = nn.Sequential(*features_list)
                    self.output_channels = 4096 # Known for WideResNet38 backbone output after mod7
                    logger.info(f"Using known WideResNet38 structure for backbone. Output channels set to {self.output_channels}.")
                except AttributeError as e:
                    logger.warning(f"Modules not found via known names: {e}. Attempting fallback: children[:-2 or -1].")
                    all_children = list(model.children())
                    # Determine how many modules to exclude (1 for main head, 2 if aux head exists)
                    num_to_exclude = 2 if has_aux_classifier and len(all_children) > 2 else 1
                    if len(all_children) > num_to_exclude:
                        self.features = nn.Sequential(*all_children[:-num_to_exclude])
                        logger.info(f"Using fallback: children[:-{num_to_exclude}] as backbone (assuming final {num_to_exclude} module(s) are head(s)).")
                    else:
                        logger.error("Cannot split model automatically (fallback failed - not enough children).")
                        self.features = None

                    # If fallback succeeded, try to infer output channels
                    if self.features:
                        # Determine device from model parameters
                        model_device = next(iter(model.parameters()), torch.device('cpu')).device
                        self._infer_output_channels(model_device)
                    else:
                        # If features is None, set a fallback guess
                        self.output_channels = 4096 # Fallback guess

                # Final check if output_channels is still None
                if self.output_channels is None:
                    logger.warning("Output channels still None after init attempts. Setting default 4096.")
                    self.output_channels = 4096

            def _infer_output_channels(self, device: torch.device):
                """Tries to infer output channels using a dummy input."""
                if not self.features: return
                try:
                    # Use a smaller dummy input to save memory
                    dummy_input = torch.randn(1, 3, 64, 128, device=device)
                    self.features.eval() # Set to eval mode for inference
                    with torch.no_grad():
                        dummy_out = self.features(dummy_input)
                    self.output_channels = dummy_out.shape[1]
                    logger.info(f"Inferred backbone output channels (fallback): {self.output_channels}")
                    self.features.train() # Set back to train mode
                except Exception as infer_e:
                    logger.error(f"Could not infer backbone output channels in fallback: {infer_e}")
                    # Keep the fallback guess if inference fails
                    if self.output_channels is None: self.output_channels = 4096

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.features is None: raise RuntimeError("BackboneWrapper features not initialized.")
                return self.features(x)

        class SegHeadWrapper(nn.Module):
             # Stores output_channels attribute and verifies it
            def __init__(self, model: nn.Module, has_aux_classifier: bool, expected_out_classes: int):
                super().__init__()
                self.classifier: Optional[nn.Module] = None
                self.output_channels: Optional[int] = None # Added

                common_head_names = ['final', 'seg_head', 'classifier', 'segmentation_head']
                potential_head = None

                # Try finding the main head by common names
                for name in common_head_names:
                     if hasattr(model, name):
                         potential_head = getattr(model, name)
                         # Ensure it's actually a module, not just an attribute
                         if isinstance(potential_head, nn.Module):
                            self.classifier = potential_head
                            logger.info(f"Using attribute '{name}' as segmentation head.")
                            break
                         else:
                             potential_head = None # Reset if it wasn't a module

                # Fallback using child indices if name search failed
                if self.classifier is None:
                    logger.warning("Could not find common head attribute names. Using fallback: Assuming second-to-last or last child.")
                    all_children = list(model.children())
                    if has_aux_classifier and len(all_children) >= 2:
                        # If aux head exists, assume main head is the second-to-last child
                        self.classifier = all_children[-2]
                        logger.info("Using second-to-last child module as segmentation head (assuming aux exists).")
                    elif not has_aux_classifier and len(all_children) >= 1:
                        # If no aux head, assume main head is the last child
                        self.classifier = all_children[-1]
                        logger.info("Using last child module as segmentation head (assuming no aux).")
                    else:
                        # Not enough children to determine the head
                        logger.error("Fallback failed: Not enough children modules to determine segmentation head.")

                # If no head was found by name or fallback
                if self.classifier is None:
                    raise AttributeError("Cannot determine segmentation head module automatically.")

                # Verify or set the output channels attribute based on the found classifier
                self._verify_or_set_output_channels(model, expected_out_classes)


            def _verify_or_set_output_channels(self, model: nn.Module, expected_out_classes: int):
                 """Infers output channels from the head and verifies against expected."""
                 if self.classifier is None: return
                 inferred_channels = None
                 try:
                     # Find the last conv/linear layer in the classifier module
                     last_layer = None
                     for m in reversed(list(self.classifier.modules())):
                         if isinstance(m, (nn.Conv2d, nn.Linear)):
                             last_layer = m
                             break

                     if last_layer is not None:
                         if isinstance(last_layer, nn.Conv2d):
                             inferred_channels = last_layer.out_channels
                         elif isinstance(last_layer, nn.Linear):
                             # Linear layers usually flatten spatial dims, so out_features is channels
                             inferred_channels = last_layer.out_features
                         logger.info(f"Inferred segmentation head output channels from last layer ({type(last_layer).__name__}): {inferred_channels}")
                     else:
                         logger.warning("Could not find Conv2d/Linear layer in segmentation head to infer output channels.")

                 except Exception as e:
                     logger.warning(f"Error inferring segmentation head output channels: {e}.")

                 # Compare inferred channels with expected num_classes
                 if inferred_channels is not None:
                     if inferred_channels != expected_out_classes:
                         logger.warning(f"Head's inferred output channels ({inferred_channels}) mismatch expected num_classes ({expected_out_classes}). Check model definition/wrappers. Using expected value {expected_out_classes}.")
                         # We MUST trust the expected_out_classes for loss compatibility
                         self.output_channels = expected_out_classes
                     else:
                         # Inferred matches expected, store it
                         self.output_channels = inferred_channels
                 else:
                     # Fallback: Assume the head is correctly defined for num_classes
                     logger.warning(f"Could not definitively infer head output channels. Assuming {expected_out_classes} based on num_classes arg.")
                     self.output_channels = expected_out_classes

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.classifier is None: raise RuntimeError("SegHeadWrapper classifier not initialized.")
                return self.classifier(x)

        # --- Instantiate Wrappers ---
        try:
            # Pass has_aux flag to BackboneWrapper for correct splitting logic
            backbone = BackboneWrapper(full_model, has_aux_classifier=has_aux)
            if not hasattr(backbone, 'output_channels') or backbone.output_channels is None:
                 logger.error("Backbone output channels not set after wrapper initialization.")
                 return None # Cannot proceed without knowing backbone output dim

            # Pass has_aux and num_classes to SegHeadWrapper for splitting and verification
            segmentation_head = SegHeadWrapper(full_model, has_aux_classifier=has_aux, expected_out_classes=num_classes)
            # Verify the final output channels attribute matches num_classes
            if not hasattr(segmentation_head, 'output_channels') or segmentation_head.output_channels is None:
                 logger.warning("Segmentation head output channels not determined or attribute missing. Assuming compatible with num_classes.")
            elif segmentation_head.output_channels != num_classes:
                 # This is a critical mismatch for the loss function
                 logger.error(f"CRITICAL: Segmentation head wrapper finalized with {segmentation_head.output_channels} channels, but expected {num_classes} for loss. Check model structure or wrapper logic.")
                 # Return None or raise error? Let's return None to prevent proceeding.
                 return None

            logger.info("Successfully extracted backbone and segmentation head from DeepWV3Plus.")
            return backbone, segmentation_head

        except AttributeError as wrap_e:
            logger.error(f"Failed to automatically split DeepWV3Plus using wrappers: {wrap_e}", exc_info=True)
            return None
        except Exception as wrap_e_gen:
            logger.error(f"Unexpected error during model wrapping: {wrap_e_gen}", exc_info=True)
            return None

    except ImportError as e:
        logger.error(f"Import error related to DeepWV3Plus: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading/splitting DeepWV3Plus: {e}")
        return None
    finally:
        # Clean up sys.path modification if it was added
        if added_to_path and code_dir and code_dir in sys.path:
            try:
                sys.path.remove(code_dir)
                # logger.debug(f"Removed '{code_dir}' from Python path.") # Verbose log
            except ValueError:
                pass # Ignore error if path was already removed
# --- End Model Loading Helpers ---

# --- Main Function ---
def main():
    """Main function to set up and run the training process."""
    args = parse_args()
    set_seed(args.seed)

    # --- Setup ---
    if args.cuda_launch_blocking:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logger.warning("CUDA_LAUNCH_BLOCKING set to 1. This will significantly slow down execution but improve debugging for CUDA errors.")

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Debug Mode Setup ---
    if args.debug:
        logger.warning("--- DEBUG MODE ENABLED ---")
        logger.info(f"--- Using subset of {args.debug_samples} samples per dataset ---")
        # Optionally enable anomaly detection only if debugging complex gradient issues
        # torch.autograd.set_detect_anomaly(True)
        # logger.warning("--- PyTorch Anomaly Detection ENABLED (Performance impact!) ---")
    else:
        # Disable anomaly detection for performance in normal runs
        torch.autograd.set_detect_anomaly(False)


    # --- Data Transforms ---
    target_size = (args.image_height, args.image_width)
    logger.info(f"Using image size: {target_size}")
    norm_mean=[0.485, 0.456, 0.406]; norm_std=[0.229, 0.224, 0.225]

    # Train transforms: Resize, Augmentations, ToTensor, Normalize
    train_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        # Reduce aggressive augmentation in debug mode? Optional.
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1) if not args.debug else transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    # Validation/Aux transforms: Resize, ToTensor, Normalize
    val_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # *** SIMPLIFIED Mask Transform ***
    # ASSUMPTION: SegmentationDataset.__getitem__ loads the mask (e.g., PIL uint8),
    # maps raw IDs -> train IDs (0-18) + ignore_index (255), THEN this transform is applied.
    mask_transform = transforms.Compose([
        # Resize using NEAREST interpolation for labels to avoid creating new values
        transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST),
        # Convert the (H, W) [PIL/np uint8] mask to (1, H, W) [torch float 0-1] tensor
        transforms.ToTensor(),
        # Squeeze the channel dimension (1, H, W) -> (H, W)
        # Convert the resulting float tensor (values should be label_id / 255.0) to Long type.
        # Rounding might be needed if ToTensor doesn't produce exact label_id/255.0, but usually okay.
        # Direct conversion to long truncates, which is correct here assuming integer inputs to ToTensor.
        transforms.Lambda(lambda x: x.squeeze(0).long())
    ])
    logger.info("Defined data transforms. CRITICAL: Assumes SegmentationDataset performs rawId->trainId mapping BEFORE mask_transform.")


    # --- Datasets ---
    logger.info("Loading datasets...")
    try:
        # --- IMPORTANT: Instantiate SegmentationDataset ---
        # Ensure your SegmentationDataset class handles the mapping from Cityscapes
        # raw label IDs (in *_labelIds.png) to train IDs (0-18) and maps other classes
        # to the value specified by args.ignore_index (default 255).
        # This mapping should happen *inside* the dataset's __getitem__, before transforms.
        # The ignore_index argument is NOT passed here; it's used by the Loss function.
        full_train_dataset = SegmentationDataset(
            image_dir=args.cityscapes_train_images,
            mask_dir=args.cityscapes_train_labels,
            transform=train_transform,
            mask_transform=mask_transform,
            num_classes=args.num_classes # Pass num_classes (metadata) if needed by dataset
            # <<< ignore_index REMOVED from here >>>
        )
        full_val_dataset = SegmentationDataset(
            image_dir=args.cityscapes_val_images,
            mask_dir=args.cityscapes_val_labels,
            transform=val_transform,
            mask_transform=mask_transform,
            num_classes=args.num_classes # Pass num_classes (metadata) if needed by dataset
            # <<< ignore_index REMOVED from here >>>
        )

        # Handle Aux dataset
        max_aux_files = args.debug_samples if args.debug else None
        _aux_limited_by_max_files = False # Flag to track if subsetting is needed later
        full_aux_dataset = None # Initialize explicitly
        if os.path.exists(args.aux_images):
            if os.path.isdir(args.aux_images):
                 # Pass max_files argument if SimpleImageDataset supports it
                 try:
                     full_aux_dataset = SimpleImageDataset(args.aux_images, transform=val_transform, max_files=max_aux_files)
                     if args.debug and max_aux_files is not None and len(full_aux_dataset) > 0:
                         logger.info(f"Loaded auxiliary dataset limited to {len(full_aux_dataset)} files (max_files={max_aux_files}).")
                         _aux_limited_by_max_files = True # Mark that it was limited by the dataset class
                     elif args.debug and max_aux_files is not None:
                          logger.warning(f"Auxiliary dataset loaded with max_files={max_aux_files}, but resulted in 0 images.")
                 except TypeError: # Handle if SimpleImageDataset doesn't have max_files
                     logger.warning("SimpleImageDataset does not seem to support 'max_files'. Loading all aux images first (might be slow in debug).")
                     full_aux_dataset = SimpleImageDataset(args.aux_images, transform=val_transform)
                     _aux_limited_by_max_files = False # Not limited by max_files, might need subsetting
            else:
                 logger.warning(f"Aux path {args.aux_images} exists but is not a directory. Assuming it's a file list (not implemented here). Skipping Aux.")
        else:
            logger.warning(f"Auxiliary image directory/path not found: {args.aux_images}. Skipping auxiliary dataset.")

    except Exception as e:
        logger.error(f"Error creating initial datasets: {e}", exc_info=True)
        # Add hint about label mapping
        logger.error("Ensure your SegmentationDataset correctly maps Cityscapes label IDs to train IDs (0-18) and the ignore index value (e.g., 255).")
        sys.exit(1)

    # --- Optional: Add detailed check for mask values in debug mode ---
    if args.debug and isinstance(full_train_dataset, SegmentationDataset) and len(full_train_dataset) > 0:
        logger.info("--- Running Debug Check on Sample Mask ---")
        try:
            # Check a few random samples
            num_check = min(5, len(full_train_dataset))
            indices_to_check = random.sample(range(len(full_train_dataset)), num_check)
            all_checks_passed = True
            for i, idx_to_check in enumerate(indices_to_check):
                 _, sample_mask = full_train_dataset[idx_to_check] # Get mask after transforms

                 if not isinstance(sample_mask, torch.Tensor):
                      logger.error(f"DEBUG CHECK FAILED: Sample mask {idx_to_check} is not a tensor (type: {type(sample_mask)}). Check dataset output and transforms.")
                      all_checks_passed = False
                      continue
                 if sample_mask.dtype != torch.long:
                      logger.error(f"DEBUG CHECK FAILED: Sample mask {idx_to_check} is not Long type (dtype: {sample_mask.dtype}). Check mask_transform.")
                      all_checks_passed = False
                      # continue # Keep checking other aspects

                 unique_vals = torch.unique(sample_mask)
                 logger.debug(f"Debug check sample {i+1}/{num_check} (index {idx_to_check}): Unique mask values: {unique_vals.tolist()}")

                 # Check for values outside the valid range [0, num_classes-1] AND not ignore_index
                 invalid_mask_vals = unique_vals[(unique_vals != args.ignore_index) & ((unique_vals < 0) | (unique_vals >= args.num_classes))]

                 if len(invalid_mask_vals) > 0:
                      logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                      logger.error(f"DEBUG CHECK FAILED: Invalid mask values found in sample {idx_to_check}: {invalid_mask_vals.tolist()}")
                      logger.error(f"Expected range [0, {args.num_classes-1}] OR ignore_index {args.ignore_index}.")
                      logger.error(f"CHECK YOUR SegmentationDataset ID MAPPING LOGIC!")
                      logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                      all_checks_passed = False
                      # Optional: break after first failure
                      # break

            if all_checks_passed:
                 logger.info("--- Debug Check on Sample Masks Passed ---")
            else:
                 logger.error("--- Debug Check on Sample Masks FAILED ---")
                 # sys.exit(1) # Optionally exit if debug check fails

        except Exception as e:
            logger.error(f"Error during debug mask check: {e}", exc_info=True)
    elif args.debug and len(full_train_dataset) == 0:
         logger.warning("Debug mode: Train dataset is empty, skipping mask check.")


    # --- Apply Debug Subsetting ---
    if args.debug:
        logger.info(f"Applying debug subsetting: using up to {args.debug_samples} samples for Train/Val.")

        # Subset Training Data
        num_train_to_sample = min(len(full_train_dataset), args.debug_samples)
        if len(full_train_dataset) > num_train_to_sample:
            # Use random indices for subsetting
            train_indices = torch.randperm(len(full_train_dataset))[:num_train_to_sample].tolist()
            train_dataset = Subset(full_train_dataset, train_indices)
            logger.info(f"Train dataset subsetted to {len(train_dataset)} random samples.")
        else:
            # No subsetting needed if dataset is smaller than requested samples
            train_dataset = full_train_dataset
            logger.info(f"Using full training dataset ({len(train_dataset)} samples) as it's <= debug_samples.")

        # Subset Validation Data
        num_val_to_sample = min(len(full_val_dataset), args.debug_samples)
        if len(full_val_dataset) > num_val_to_sample:
            val_indices = torch.randperm(len(full_val_dataset))[:num_val_to_sample].tolist()
            val_dataset = Subset(full_val_dataset, val_indices)
            logger.info(f"Validation dataset subsetted to {len(val_dataset)} random samples.")
        else:
            val_dataset = full_val_dataset
            logger.info(f"Using full validation dataset ({len(val_dataset)} samples) as it's <= debug_samples.")

        # Aux dataset: Subset only if it exists AND was not already limited by max_files in the Dataset class
        aux_dataset = full_aux_dataset # Start with the potentially already limited dataset
        if aux_dataset and not _aux_limited_by_max_files:
             num_aux_to_sample = min(len(aux_dataset), args.debug_samples)
             if len(aux_dataset) > num_aux_to_sample:
                 aux_indices = torch.randperm(len(aux_dataset))[:num_aux_to_sample].tolist()
                 aux_dataset = Subset(aux_dataset, aux_indices)
                 logger.info(f"Auxiliary dataset subsetted to {len(aux_dataset)} random samples (post-load).")
             # No subsetting needed if aux is smaller than debug_samples
             elif len(aux_dataset) > 0:
                 logger.info(f"Using full auxiliary dataset ({len(aux_dataset)} samples) as it's <= debug_samples.")

    else:
        # Not in debug mode, use the full datasets
        train_dataset = full_train_dataset
        val_dataset = full_val_dataset
        aux_dataset = full_aux_dataset

    # Log final effective dataset sizes
    logger.info(f"Final dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    logger.info(f"Final dataset sizes - Auxiliary: {len(aux_dataset) if aux_dataset else 'None'}")


    # --- Data Loaders ---
    # Use persistent_workers=True if num_workers > 0 for potentially faster startup after first epoch
    # Disable persistent_workers if num_workers is 0 or in debug mode for simplicity/potential issues
    persistent_workers_flag = (args.num_workers > 0) and (not args.debug)
    if persistent_workers_flag:
        logger.info("Using persistent workers for DataLoaders.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent_workers_flag)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers_flag)
    # Only create aux_loader if aux_dataset exists and has items
    aux_loader = None # Initialize
    if aux_dataset and len(aux_dataset) > 0:
        # Ensure batch size isn't larger than aux dataset size if drop_last=True
        aux_drop_last = True # Usually true for training-like aux loops
        aux_batch_size = args.batch_size
        if len(aux_dataset) < args.batch_size:
             logger.warning(f"Auxiliary dataset size ({len(aux_dataset)}) is smaller than batch size ({args.batch_size}). Setting drop_last=False for aux_loader.")
             aux_drop_last = False
             # Keep original batch size, loader will just have one smaller batch
             # aux_batch_size = len(aux_dataset) # Alternative: reduce batch size

        aux_loader = DataLoader(aux_dataset, batch_size=aux_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=aux_drop_last, persistent_workers=persistent_workers_flag)
    else:
        logger.warning("Auxiliary dataset is empty or was not loaded. Aux loader will be None.")


    # --- Model Initialization ---
    logger.info("Creating model...")
    backbone: Optional[nn.Module] = None
    segmentation_head: Optional[nn.Module] = None
    if args.use_simple_model:
        # Ensure the simple head outputs args.num_classes channels
        backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
        logger.info("Using simple test model.")
    else:
        logger.info("Attempting to load DeepWV3Plus...")
        # Ensure DeepWV3Plus head outputs args.num_classes channels
        model_components = import_and_get_deepwv3plus(args.num_classes)
        if model_components:
            backbone, segmentation_head = model_components
            logger.info("Successfully loaded and split DeepWV3Plus.")
        else:
            logger.warning("Failed to load DeepWV3Plus, falling back to simple model.")
            backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)

    if backbone is None or segmentation_head is None:
        logger.error("Model creation failed (backbone or head is None). Exiting.")
        sys.exit(1)

    # Verify backbone output channels (should be set by helpers)
    if not hasattr(backbone, 'output_channels') or backbone.output_channels is None:
        logger.error("Backbone is missing 'output_channels' attribute after initialization. Cannot proceed.")
        sys.exit(1)
    else:
        logger.info(f"Backbone initialized with output_channels: {backbone.output_channels}")

    # Verify segmentation head output channels matches num_classes (should be set/verified by helpers)
    if hasattr(segmentation_head, 'output_channels') and segmentation_head.output_channels is not None:
         logger.info(f"Segmentation head initialized with output_channels: {segmentation_head.output_channels}")
         if segmentation_head.output_channels != args.num_classes:
              # This check should ideally be redundant if the helper functions worked correctly
              logger.error(f"CRITICAL ERROR: Segmentation head final output channels ({segmentation_head.output_channels}) do NOT match num_classes ({args.num_classes}) required for loss calculation. Check model definition or wrappers.")
              sys.exit(1)
    else:
         # This case indicates a problem in the helper function's verification
         logger.error(f"CRITICAL ERROR: Segmentation head output channels attribute missing or None after initialization. Cannot verify compatibility with num_classes ({args.num_classes}).")
         sys.exit(1)


    # Create the main Hopfield-PEBAL model
    try:
        # Define efficient_decoder_kwargs conditionally
        # Pass relevant args if needed by EfficientSegmentationDecoder init
        eff_decoder_kwargs = {
             # Add any relevant args from parser if needed by EfficientSegmentationDecoder
             # e.g., 'attention_heads': args.eff_decoder_heads (if added to parser)
             # 'feature_dim': args.eff_decoder_dim (if added to parser)
             'attn_max_tokens': args.chunk_size # Use chunk_size as attn_max_tokens? Check HopfieldPEBALModel usage
        } if args.use_efficient_decoder else None

        if eff_decoder_kwargs:
             logger.info(f"Passing efficient_decoder_kwargs: {eff_decoder_kwargs}")

        model = HopfieldPEBALModel(
            backbone=backbone,
            segmentation_head=segmentation_head,
            num_classes=args.num_classes, # This MUST match the head's output channels
            memory_feature_dim=args.memory_feature_dim,
            memory_size=args.memory_size,
            memory_beta=args.memory_beta,
            insertion_point=args.insertion_point,
            target_feature_dim=args.target_feature_dim, # Can be None if insertion='after_seghead'
            use_efficient_memory=args.use_efficient_memory,
            use_faiss=args.use_faiss,
            # pq_bytes=args.pq_bytes, # <-- REMOVED based on error
            sampling_stride=args.sampling_stride,
            use_efficient_decoder=args.use_efficient_decoder,
            efficient_decoder_kwargs=eff_decoder_kwargs # Pass the dictionary here
        ).to(device) # Move the final composed model to the device
        logger.info(f"HopfieldPEBALModel created and moved to {device}.")

    except Exception as e:
        logger.error(f"Error initializing HopfieldPEBALModel: {e}", exc_info=True)
        sys.exit(1)

    is_faiss_used = args.use_faiss and hasattr(model, 'memory_manager') and hasattr(model.memory_manager, 'use_faiss') and model.memory_manager.use_faiss
    logger.info(f"Model configuration: Insertion Point: {args.insertion_point}, Use Efficient Decoder: {args.use_efficient_decoder}, Use FAISS: {is_faiss_used}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")


    # --- PEBAL Weight Integration ---
    if args.pebal_checkpoint:
        if os.path.isfile(args.pebal_checkpoint):
            logger.info(f"Attempting to integrate PEBAL checkpoint: {args.pebal_checkpoint}")
            try:
                # The function should handle loading and potential mismatches internally
                model = integrate_pebal_weights(model, args.pebal_checkpoint, device)
                logger.info("Successfully integrated PEBAL weights into the model.")
            except Exception as e:
                logger.error(f"Failed to integrate PEBAL weights from {args.pebal_checkpoint}: {e}.", exc_info=True)
                # Decide if you want to exit or continue without PEBAL weights
                # sys.exit(1) # Optionally exit if PEBAL weights are critical
        else:
            logger.error(f"PEBAL checkpoint file not found: {args.pebal_checkpoint}. Skipping integration.")


    # --- Loss Function ---
    try:
        # ***** CRITICAL: Ensure num_classes and ignore_index match dataset preprocessing and model head *****
        # The ignore_index argument passed here tells the CrossEntropyLoss which label value to ignore.
        criterion = HopfieldPEBALLoss(
            num_classes=args.num_classes, # Must match head output and trainIds (0-18)
            seg_weight=args.seg_weight,
            energy_weight=args.energy_weight,
            hopfield_weight=args.hopfield_weight,
            inlier_margin=args.inlier_margin,
            outlier_margin=args.outlier_margin,
            temperature=args.temperature,
            ignore_index=args.ignore_index # Must match void label (e.g., 255) in masks provided by dataset
        ).to(device)
        logger.info(f"Loss function created on {device}. NumClasses={args.num_classes}, IgnoreIndex={args.ignore_index}. Weights: Seg={args.seg_weight}, Energy={args.energy_weight}")
    except Exception as e:
        logger.error(f"Error initializing loss function: {e}", exc_info=True)
        sys.exit(1)


    # --- Optimizer ---
    try:
        # Carefully define parameter groups using module instances where possible
        backbone_params = []
        mem_manager_params = []
        adapter_params = []
        head_related_params = [] # Group all head/projection layers here
        other_params = []
        processed_param_ids = set() # Use parameter IDs for tracking

        # Map module names (or prefixes) to their instances and the target list
        param_map = {}
        # Check if attributes exist before adding to map
        if hasattr(model, 'backbone') and isinstance(model.backbone, nn.Module):
            param_map['backbone'] = (model.backbone, backbone_params)
        if hasattr(model, 'memory_manager') and isinstance(model.memory_manager, nn.Module):
             param_map['memory_manager'] = (model.memory_manager, mem_manager_params)
        if hasattr(model, 'channel_adapter') and isinstance(model.channel_adapter, nn.Module):
             param_map['channel_adapter'] = (model.channel_adapter, adapter_params)
        # Group all segmentation/energy/projection heads together
        if hasattr(model, 'segmentation_head') and isinstance(model.segmentation_head, nn.Module):
             param_map['segmentation_head'] = (model.segmentation_head, head_related_params)
        # Check for original head only if it's different from the current one
        if hasattr(model, '_original_segmentation_head') and isinstance(model._original_segmentation_head, nn.Module) and model._original_segmentation_head is not model.segmentation_head:
             param_map['_original_segmentation_head'] = (model._original_segmentation_head, head_related_params)
        if hasattr(model, 'energy_head') and isinstance(model.energy_head, nn.Module):
             param_map['energy_head'] = (model.energy_head, head_related_params)
        if hasattr(model, 'memory_input_proj') and isinstance(model.memory_input_proj, nn.Module):
             param_map['memory_input_proj'] = (model.memory_input_proj, head_related_params)
        # Check for efficient decoder instance only if it's different
        if hasattr(model, '_efficient_decoder_instance') and isinstance(model._efficient_decoder_instance, nn.Module) and model._efficient_decoder_instance is not model.segmentation_head:
            param_map['_efficient_decoder_instance'] = (model._efficient_decoder_instance, head_related_params)
        # Also include the final_seghead_proj if it exists and is not Identity
        if hasattr(model, 'final_seghead_proj') and isinstance(model.final_seghead_proj, nn.Module) and not isinstance(model.final_seghead_proj, nn.Identity):
             param_map['final_seghead_proj'] = (model.final_seghead_proj, head_related_params)
        # Also include the final_classifier if it exists and is not Identity
        if hasattr(model, 'final_classifier') and isinstance(model.final_classifier, nn.Module) and not isinstance(model.final_classifier, nn.Identity):
             param_map['final_classifier'] = (model.final_classifier, head_related_params)


        # Iterate through all trainable parameters
        all_named_params = list(model.named_parameters()) # Get list once

        for name, param in all_named_params:
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in processed_param_ids: continue # Skip if already processed

            assigned = False
            # Check if the parameter belongs to any of the mapped module instances
            for mod_name, (module_instance, param_list) in param_map.items():
                 # Efficiently check if param is within the module's parameters
                 if any(param is p for p in module_instance.parameters()):
                     param_list.append(param)
                     processed_param_ids.add(param_id)
                     assigned = True
                     # logger.debug(f"Assigning '{name}' (id: {param_id}) to group '{mod_name}' via instance check.")
                     break # Assign to the first matching module instance
            if assigned: continue

            # If not assigned by instance, assign to 'other' group
            other_params.append(param)
            processed_param_ids.add(param_id)
            logger.debug(f"Assigning parameter '{name}' (id: {param_id}) to 'other' group (not found in mapped modules).")

        # Verify all trainable params were assigned
        num_assigned = sum(len(pg) for pg in [backbone_params, mem_manager_params, adapter_params, head_related_params, other_params])
        if num_assigned != trainable_params:
             logger.warning(f"Parameter assignment mismatch! Assigned: {num_assigned}, Trainable: {trainable_params}. Some params might have default LR or be missed.")
             # Find unassigned params for debugging
             unassigned = [n for n, p in all_named_params if p.requires_grad and id(p) not in processed_param_ids]
             if unassigned:
                 logger.warning(f"Unassigned trainable params found: {unassigned}")
                 # Add unassigned to 'other' as a safety measure
                 other_params.extend([p for n, p in all_named_params if id(p) in [id(param) for name, param in unassigned]]) # Get param objects
                 logger.warning(f"Added {len(unassigned)} unassigned params to 'other' group.")

        # Create the final list of parameter groups for the optimizer
        param_groups = []
        if backbone_params: param_groups.append({'params': backbone_params, 'lr': args.learning_rate * args.backbone_lr_factor, 'name': 'backbone'})
        if mem_manager_params: param_groups.append({'params': mem_manager_params, 'lr': args.learning_rate, 'name': 'memory_manager'})
        if adapter_params: param_groups.append({'params': adapter_params, 'lr': args.learning_rate, 'name': 'adapter'})
        # Consolidate all head/projection layers into one group
        if head_related_params: param_groups.append({'params': head_related_params, 'lr': args.learning_rate, 'name': 'head_related'})
        if other_params: param_groups.append({'params': other_params, 'lr': args.learning_rate, 'name': 'other'})

        if not param_groups:
             logger.error("No trainable parameters found or assigned to optimizer groups!")
             sys.exit(1)
        # Check if total params in groups match trainable params
        total_params_in_groups = sum(len(pg['params']) for pg in param_groups)
        if total_params_in_groups != trainable_params:
             logger.warning(f"Mismatch after group creation! Params in groups: {total_params_in_groups}, Total trainable: {trainable_params}")


        optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
        logger.info(f"Optimizer: AdamW (BaseLR={args.learning_rate}, BackboneLRFactor={args.backbone_lr_factor}, WeightDecay={args.weight_decay})")
        # Log the actual parameter counts per group from the optimizer
        for i, pg in enumerate(optimizer.param_groups):
            group_name = pg.get('name', f'group_{i}')
            num_params_in_group = sum(p.numel() for p in pg['params'])
            logger.info(f"  Optimizer Group '{group_name}': {len(pg['params'])} tensors, {num_params_in_group:,} parameters, LR={pg['lr']:.2e}")

    except Exception as e:
        logger.error(f"Error setting up optimizer: {e}", exc_info=True)
        sys.exit(1)


    # --- LR Scheduler ---
    # ReduceLROnPlateau monitors a metric (validation loss) and reduces LR if it stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Reduce LR when the metric has stopped decreasing
        factor=0.5,          # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
        verbose=True,        # Print a message when the learning rate is reduced
        threshold=0.01,      # Threshold for measuring the new optimum, to only focus on significant changes
        threshold_mode='rel',# Relative threshold mode
        cooldown=1,          # Number of epochs to wait before resuming normal operation after lr has been reduced
        min_lr=1e-7          # Lower bound on the learning rate
    )
    logger.info("LR Scheduler: ReduceLROnPlateau (monitors validation loss, patience=5, factor=0.5)")


    # --- Resume Training ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint: {args.resume}")
        try:
            # Load checkpoint onto CPU first to avoid GPU memory issues if loading from a different setup
            checkpoint = torch.load(args.resume, map_location='cpu')
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Load model state dict (be lenient with missing/unexpected keys)
            if 'model_state_dict' in checkpoint:
                # --- Load model state dict with careful filtering ---
                loaded_state_dict = checkpoint['model_state_dict']
                current_model_dict = model.state_dict()

                # Filter state dict:
                # 1. Keep keys that exist in the current model.
                # 2. Keep keys where the shape matches exactly.
                filtered_state_dict = {}
                mismatched_keys = []
                unexpected_keys_in_ckpt = []
                for k, v in loaded_state_dict.items():
                    if k in current_model_dict:
                        if current_model_dict[k].shape == v.shape:
                            filtered_state_dict[k] = v
                        else:
                            mismatched_keys.append(f"{k} (ckpt: {v.shape}, model: {current_model_dict[k].shape})")
                    else:
                        unexpected_keys_in_ckpt.append(k)

                missing_keys_in_model = set(current_model_dict.keys()) - set(filtered_state_dict.keys())

                if mismatched_keys:
                    logger.warning(f"Mismatched shapes loading model state (keys ignored): {mismatched_keys}")
                if missing_keys_in_model:
                     logger.warning(f"Keys missing in checkpoint or incompatible, present in model (will be randomly initialized): {missing_keys_in_model}")
                if unexpected_keys_in_ckpt:
                    logger.warning(f"Keys present in checkpoint but not in model (ignored): {unexpected_keys_in_ckpt}")

                # Load the compatible state using strict=False
                model.load_state_dict(filtered_state_dict, strict=False)
                logger.info("Model state loaded successfully (potentially partially).")
            else:
                logger.warning("Checkpoint does not contain 'model_state_dict'. Model weights not loaded.")

            # Load optimizer state dict
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded successfully.")
                    # Manually move optimizer states to the correct device if needed
                    if device != torch.device('cpu'):
                        for state in optimizer.state.values():
                             for k, v in state.items():
                                 if isinstance(v, torch.Tensor):
                                     try:
                                         state[k] = v.to(device)
                                     except Exception as e_state:
                                         logger.warning(f"Could not move optimizer state tensor {k} to {device}: {e_state}")
                        # logger.info(f"Attempted to move optimizer states to {device}.") # Less verbose
                except ValueError as e:
                    logger.warning(f"Could not load optimizer state due to likely parameter group mismatch: {e}. Optimizer will start fresh.")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}. Optimizer will start fresh.")
            else:
                logger.warning("Optimizer state not found in checkpoint. Optimizer will start fresh.")

            # Load scheduler state dict (if applicable and present)
            if scheduler and hasattr(scheduler, 'load_state_dict'):
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        logger.info("Scheduler state loaded successfully.")
                    except Exception as e:
                        logger.warning(f"Could not load scheduler state: {e}. Scheduler might reset.")
                elif scheduler: # Check scheduler exists even if state is missing
                    logger.warning("Scheduler state not found in checkpoint or was None. Scheduler might reset.")

            # Load epoch and best validation loss
            start_epoch = checkpoint.get('epoch', -1) + 1 # Start from the next epoch
            # Ensure best_val_loss is float, handle potential None or other types gracefully
            resumed_best_loss = checkpoint.get('best_val_loss', float('inf'))
            if not isinstance(resumed_best_loss, (float, int)):
                 logger.warning(f"Invalid type for 'best_val_loss' in checkpoint ({type(resumed_best_loss)}). Resetting to infinity.")
                 best_val_loss = float('inf')
            else:
                 best_val_loss = float(resumed_best_loss)

            logger.info(f"Resuming training from epoch {start_epoch}. Previous best val loss: {best_val_loss if best_val_loss != float('inf') else 'inf'}")

            # Ensure model is on the correct device AFTER loading state dict
            model.to(device)

        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {args.resume}. Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            model.to(device) # Ensure model is on device
        except Exception as e:
            logger.error(f"Error loading checkpoint '{args.resume}': {e}. Starting training from scratch.", exc_info=True)
            start_epoch = 0
            best_val_loss = float('inf')
            # Re-ensure model is on device after potential error during loading
            model.to(device)

    else:
        if args.resume: # Checkpoint path was given but file not found
             logger.warning(f"Resume checkpoint specified ('{args.resume}') but not found. Starting training from scratch.")
        else: # No checkpoint path was given
             logger.info("No resume checkpoint specified. Starting training from scratch.")
        # Ensure model is on the correct device if not resuming
        model.to(device) # Model should already be on device from init, but re-affirm


    # --- Log Final Configuration ---
    logger.info("--- Training Configuration ---")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Start Epoch: {start_epoch}")
    logger.info(f"  Initial Best Val Loss: {best_val_loss if best_val_loss != float('inf') else 'inf'}")
    logger.info("--- End Configuration ---")


    # --- Start Training ---
    logger.info(f"Starting training loop from epoch {start_epoch} for {args.num_epochs} epochs...")
    trained_model = None # Initialize to handle potential early exit or errors
    final_best_val_loss = best_val_loss # Keep track of the best loss achieved in this run
    completed_epochs = start_epoch - 1 # Track last fully completed epoch index

    try:
        # The train function should return the model, the best val loss achieved, and last completed epoch index
        trained_model, final_best_val_loss, completed_epochs = train_hopfield_pebal(
            train_loader=train_loader,
            val_loader=val_loader,
            aux_loader=aux_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            start_epoch=start_epoch,
            num_epochs=args.num_epochs, # Target number of epochs
            device=device,
            scheduler=scheduler,
            save_path=args.save_path,
            memory_update_freq=args.memory_update_freq,
            memory_update_batches=args.memory_update_batches,
            mixed_precision=args.mixed_precision,
            use_efficient_memory=args.use_efficient_memory,
            best_val_loss_initial=best_val_loss, # Pass the potentially resumed best loss
            grad_clip_norm=args.grad_clip_norm # Pass clipping value
        )
        # If training completes normally, completed_epochs should reflect the index of the last epoch finished
        # (e.g., if num_epochs=50, start_epoch=0, last completed epoch index is 49)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt). Saving current state.")
        # Use the model state as it was when interrupted
        trained_model = model # Assume model holds the latest state
        # completed_epochs holds the index of the last fully completed epoch (returned by trainer)
        logger.info(f"Interruption occurred after epoch {completed_epochs} completed.")

    except Exception as e:
        # Check specifically for the CUDA assert error that caused the original issue
        if "CUDA error: device-side assert triggered" in str(e):
             logger.error(f"FATAL TRAINING ERROR: CUDA device-side assert triggered during training loop. This likely means invalid label IDs reached the loss function.")
             logger.error(f"Check your 'SegmentationDataset' implementation for correct rawId -> trainId mapping (range [0, {args.num_classes-1}]) and handling of ignore_index ({args.ignore_index}).", exc_info=False)
        else:
             # Log other exceptions with full traceback
             logger.exception(f"A critical error occurred during training: {e}")

        # Attempt to save the model state before exiting
        trained_model = model # Assume model holds the latest state
        # completed_epochs holds the last successfully completed epoch index before the error
        logger.error(f"Attempting to save model state before exiting due to error after epoch {completed_epochs}.")
        # Optionally re-raise the exception or exit differently
        # raise e # Re-raise to see full stack trace if needed
        # sys.exit(1) # Exit with error code


    # --- Save Final Model State (if training ran at least partially) ---
    # Check if at least one epoch was completed or if resuming from a later epoch
    if trained_model is not None and completed_epochs >= start_epoch:
        # Save based on the last epoch that was *completed*.
        save_epoch_num = completed_epochs
        final_model_path = os.path.join(args.save_path, f"final_model_epoch_{save_epoch_num}.pth")
        logger.info(f"Saving final model state (from completed epoch {save_epoch_num}) to {final_model_path}")
        try:
            # Ensure model is on CPU before saving state dict to avoid GPU memory info in file
            trained_model.cpu()
            final_state = {
                'epoch': save_epoch_num, # Save the last completed epoch index
                'model_state_dict': trained_model.state_dict(),
                # Get current state dicts from optimizer/scheduler
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
                'best_val_loss': final_best_val_loss, # Save the best validation loss achieved during the entire training run
                'args': vars(args) # Save args used for this training run for reference
            }
            torch.save(final_state, final_model_path)
            logger.info("Final model state saved successfully.")
        except Exception as e:
            logger.error(f"Error saving final model state: {e}", exc_info=True)
    elif trained_model is not None and start_epoch > 0 and completed_epochs < start_epoch:
         logger.warning(f"Training started from epoch {start_epoch} but ended before completing it (last completed: {completed_epochs}). Saving state from before training loop started might be incorrect. Saving current model state as 'interrupted_epoch_{start_epoch}.pth'.")
         interrupted_path = os.path.join(args.save_path, f"interrupted_epoch_{start_epoch}.pth")
         try:
            trained_model.cpu()
            # Save minimal state
            torch.save({'model_state_dict': trained_model.state_dict()}, interrupted_path)
            logger.info(f"Interrupted model state saved to {interrupted_path}")
         except Exception as e:
              logger.error(f"Error saving interrupted model state: {e}", exc_info=True)
    else:
        logger.warning("Training did not run, failed very early, or `trained_model` is None. Final model state not saved.")


    logger.info("Training script finished.")

if __name__ == "__main__":
    # The dynamic path addition for imports is handled within the try-except block at the top
    main()
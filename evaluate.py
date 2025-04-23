#!/usr/bin/env python
# evaluate.py (Version 5.1 - Fixed PEBAL Dim Mismatch & Improved Key Mapping)

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import sys
import gc
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json # For saving parameters

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("Hopfield-PEBAL-Eval")

# --- Determine paths and modify sys.path ---
try:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    # Determine if script is in the project root or in a subdirectory
    if (script_dir / 'code').exists() and (script_dir / 'code').is_dir():
        # Script is in project root, code dir is a subdirectory
        project_root = script_dir
        code_dir = project_root / 'code'
    elif script_dir.name == 'scripts' and (script_dir.parent / 'code').exists():
        # Script is in a 'scripts' directory, code dir is sibling
        project_root = script_dir.parent
        code_dir = project_root / 'code'
    else:
        # Fallback - assume current dir is code dir
        project_root = script_dir.parent # Assume script is in 'code' or similar
        code_dir = script_dir

    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Code directory: {code_dir}")

    # Add code directory to sys.path
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        logger.info(f"Added code directory to Python path: {code_dir}")

    # Verify code directory exists
    if not code_dir.exists():
        logger.warning(f"Code directory '{code_dir}' does not exist. Imports will likely fail.")
    elif not code_dir.is_dir():
        logger.warning(f"'{code_dir}' exists but is not a directory. Imports will likely fail.")

except Exception as path_err:
    logger.error(f"Error determining paths: {path_err}. Using fallbacks.", exc_info=True)
    # Use current directory as fallback
    script_dir = Path.cwd()
    project_root = script_dir
    code_dir = script_dir / 'code' if (script_dir / 'code').exists() else script_dir

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        logger.info(f"Added fallback code directory to Python path: {code_dir}")

# --- Import psutil (optional) ---
try:
    import psutil
    PSUTIL_AVAILABLE = True
    logger.info("psutil imported successfully for memory tracking")
except ImportError:
    logger.warning("psutil not available, CPU memory tracking will be limited")
    class DummyProcess:
        def memory_info(self):
            class MemInfo: rss = 0
            return MemInfo()
    class DummyPsutil:
        def Process(self, *args, **kwargs): return DummyProcess()
    psutil = DummyPsutil()
    PSUTIL_AVAILABLE = False

# --- Import placeholder functions if needed ---
try:
    from model.mynn import initialize_weights, Upsample
    logger.info("Imported initialize_weights and Upsample from model.mynn")
    MYNN_IMPORTED = True
except ImportError:
    logger.warning("Could not import from model.mynn. Using placeholder functions.")
    MYNN_IMPORTED = False

    def initialize_weights(*args, **kwargs):
        logger.warning("Using placeholder initialize_weights function")
        pass

    def Upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        logger.warning("Using placeholder Upsample function (torch.nn.functional.interpolate)")
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# --- Import core model components ---
# Hopfield PEBAL Model
try:
    # Try top-level import first (assuming code dir is in sys.path)
    from hopfield_pebal_model import HopfieldPEBALModel, MemoryTracker
    HOPFIELD_MODEL_IMPORTED = True
    logger.info("Successfully imported HopfieldPEBALModel (top-level)")
except ImportError:
    try:
        # Try as a submodule if top-level fails
        from model.hopfield_pebal_model import HopfieldPEBALModel, MemoryTracker
        HOPFIELD_MODEL_IMPORTED = True
        logger.info("Successfully imported HopfieldPEBALModel (from model submodule)")
    except ImportError as e:
        logger.critical(f"Failed to import HopfieldPEBALModel: {e}", exc_info=True)
        logger.critical("This is required for evaluation. Cannot continue without this model definition.")
        HOPFIELD_MODEL_IMPORTED = False

# Dataset classes
try:
    from datasets.datasets import SegmentationDataset
    from datasets.fishyscapes_dataset import FishyscapesDataset
    DATASETS_IMPORTED = True
    logger.info("Successfully imported dataset classes")
except ImportError as e:
    logger.critical(f"Dataset import error: {e}", exc_info=True)
    logger.critical("Cannot proceed without dataset classes")
    DATASETS_IMPORTED = False

# WiderResNet and structures
try:
    from model.wide_resnet_base import WiderResNetA2, _NETS as WIDER_RESNET_STRUCTURES
    logger.info("Imported WiderResNetA2 and structures")
    WIDERESNET_IMPORTED = True
except ImportError as e:
    logger.critical(f"Failed to import WiderResNetA2: {e}", exc_info=True)
    logger.critical("This component is required for DeepWV3Plus")
    WIDERESNET_IMPORTED = False
    WIDER_RESNET_STRUCTURES = {} # Empty placeholder

# DeepWV3Plus network
try:
    from model.wide_network import DeepWV3Plus
    DEEPWV3PLUS_IMPORTED = True
    logger.info("Imported DeepWV3Plus class")
except ImportError as e:
    logger.critical(f"Failed to import DeepWV3Plus: {e}", exc_info=True)
    logger.critical("This model is required if using deepwv3plus as base_model")
    DEEPWV3PLUS_IMPORTED = False

# --- Check critical imports and exit early if missing ---
def check_critical_imports(args):
    """Check if all critical imports for the selected model are available"""
    critical_import_errors = []

    # Core requirements for all model types
    if not HOPFIELD_MODEL_IMPORTED:
        critical_import_errors.append("HopfieldPEBALModel is required but failed to import")

    if not DATASETS_IMPORTED:
        critical_import_errors.append("Dataset classes are required but failed to import")

    # Model-specific requirements
    if args.base_model == 'deepwv3plus':
        if not WIDERESNET_IMPORTED:
            critical_import_errors.append("WiderResNetA2 is required for deepwv3plus but failed to import")
        if not DEEPWV3PLUS_IMPORTED:
            critical_import_errors.append("DeepWV3Plus is required but failed to import")

    if critical_import_errors:
        for err in critical_import_errors:
            logger.critical(f"CRITICAL ERROR: {err}")
        return False
    return True

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Hopfield-PEBAL model for OOD detection')
    # Dataset paths
    parser.add_argument('--test_images', type=str, default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/val', help='Path to INLIER test image directory')
    parser.add_argument('--test_labels', type=str, default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/val', help='Path to INLIER test label directory (expecting labelIds)')
    parser.add_argument('--lostandfound_images', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound/original', help='Path to LostAndFound image directory')
    parser.add_argument('--lostandfound_labels', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound/labels', help='Path to LostAndFound label directory')
    parser.add_argument('--static_images', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static/original', help='Path to Static image directory')
    parser.add_argument('--static_labels', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static/labels', help='Path to Static label directory')
    parser.add_argument('--road_anomaly_images', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly/original', help='Path to Road Anomaly image directory')
    parser.add_argument('--road_anomaly_labels', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly/labels', help='Path to Road Anomaly label directory')
    parser.add_argument('--dataset', type=str, default='all', choices=['inlier', 'lostandfound', 'static', 'road_anomaly', 'all'], help='Which dataset(s) to evaluate on')
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (REQUIRED)')
    parser.add_argument('--base_model', type=str, default='deepwv3plus', choices=['deepwv3plus', 'simple'], help='Base segmentation model')
    parser.add_argument('--wider_resnet_variant', type=str, default='38', choices=['16', '20', '38'], help='WiderResNet variant used for the backbone (e.g., 38 for [3,3,6,3,1,1]) - MUST MATCH CHECKPOINT')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of INLIER classes')
    parser.add_argument('--memory_feature_dim', type=int, default=256, help='Dimension of memory features')
    parser.add_argument('--memory_beta', type=float, default=8.0, help='Beta for memory energy')
    parser.add_argument('--memory_size', type=int, default=2000, help='Memory bank size')
    parser.add_argument('--attention_heads', type=int, default=4, help='Attention heads (for efficient decoder)')
    parser.add_argument('--insertion_point', type=str, default='after_backbone', choices=['after_backbone', 'after_seghead'], help='PEBAL insertion point')
    parser.add_argument('--target_feature_dim', type=int, default=304, help='Target dimension expected by segmentation head *after* potential adapter') # Clarified help text
    parser.add_argument('--use_efficient_decoder', action='store_true', help='Use EfficientSegmentationDecoder')
    parser.add_argument('--disable_faiss', action='store_true', help='Disable FAISS')
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--output_dir', type=str, default='./results/eval', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize first few samples')
    parser.add_argument('--save_outputs', action='store_true', help='Save detailed outputs (can be large!)')
    parser.add_argument('--anomaly_id', type=int, default=19, help='Anomaly class ID in OOD datasets')
    parser.add_argument('--void_id', type=int, default=255, help='Void/ignore class ID in labels')
    # Debugging/Utility
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--check_files_exist', action='store_true', default=True, help='Check dataset files exist before loading')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU execution')
    parser.add_argument('--img_height', type=int, default=256, help='Evaluation image height')
    parser.add_argument('--img_width', type=int, default=512, help='Evaluation image width')
    return parser.parse_args()


# --- Simple Model Creation (for testing, unchanged) ---
def create_simple_backbone_for_testing(num_classes=19, img_h=256, img_w=512):
    class SimpleBackbone(nn.Module):
        def __init__(self): super().__init__(); self.conv1=nn.Conv2d(3,64,7,2,3,bias=False); self.bn1=nn.BatchNorm2d(64); self.relu=nn.ReLU(True); self.pool1=nn.MaxPool2d(3,2,1); self.conv2=nn.Conv2d(64,128,3,1,1,bias=False); self.bn2=nn.BatchNorm2d(128); self.conv3=nn.Conv2d(128,256,3,2,1,bias=False); self.bn3=nn.BatchNorm2d(256)
        def forward(self, x): x=self.relu(self.bn1(self.conv1(x))); x=self.pool1(x); x=self.relu(self.bn2(self.conv2(x))); x=self.relu(self.bn3(self.conv3(x))); return x
    class SimpleSegHead(nn.Module):
        def __init__(self,in_channels,num_classes): super().__init__(); self.head=nn.Sequential(nn.Conv2d(in_channels,128,3,1,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, num_classes, 1)); self._in_channels=in_channels
        def forward(self, x): return self.head(x)
    logger.info("Creating simple backbone and head for testing."); b=SimpleBackbone(); out_dim=256
    try:
        b.eval(); dummy_input=torch.zeros(1, 3, img_h, img_w)
        with torch.no_grad(): out_dim=b(dummy_input).shape[1]
    except Exception as e: logger.warning(f"Could not determine simple backbone output dimension: {e}. Assuming 256."); out_dim = 256
    return b, SimpleSegHead(out_dim, num_classes)

# --- DeepWV3Plus Import Function (unchanged) ---
def import_deepwv3plus(num_classes: int, structure: List[int], variant_name: str) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """
    Imports and instantiates DeepWV3Plus, then extracts backbone and segmentation head parts.
    Uses the instantiation method that seemed to work based on prior logs.
    The 'structure' and 'variant_name' are primarily for logging and potential future use,
    but NOT passed directly to the DeepWV3Plus constructor in this version.
    """
    if not DEEPWV3PLUS_IMPORTED or not WIDERESNET_IMPORTED:
        logger.critical("DeepWV3Plus or WiderResNetA2 failed to import during setup. Cannot create model.")
        return None, None

    logger.info(f"Attempting to initialize DeepWV3Plus. Requested variant: {variant_name}, structure: {structure}")
    logger.info(f"Note: Variant/structure info is used for mapping/logging, not passed directly to DeepWV3Plus constructor.")

    fm: Optional[DeepWV3Plus] = None # Initialize fm
    try:
        # *** CORRECTED INSTANTIATION: Only pass arguments DeepWV3Plus expects ***
        # Based on previous logs and typical design, it likely only needs num_classes.
        # If it needs other args, they must be added here based on its definition.
        fm = DeepWV3Plus(num_classes=num_classes)
        logger.info(f"Initialized DeepWV3Plus successfully (passed num_classes={num_classes}).")

    except TypeError as te:
        logger.error(f"TypeError initializing DeepWV3Plus: {te}", exc_info=True)
        logger.error("This likely means the arguments passed to DeepWV3Plus() are incorrect.")
        logger.error(f"Check the __init__ signature in model/wide_network.py. Tried passing num_classes={num_classes}.")
        return None, None
    except Exception as e:
         logger.error(f"Unexpected error initializing DeepWV3Plus: {e}", exc_info=True)
         return None, None

    if fm is None:
        logger.error("DeepWV3Plus instance is None after initialization block.")
        return None, None

    # --- Extract backbone and head parts ---
    # This extraction logic depends HEAVILY on the internal naming convention of DeepWV3Plus.
    # If DeepWV3Plus changes internally, this section WILL need updates.
    logger.info("Extracting backbone and head modules from DeepWV3Plus instance...")
    bb_sequential_part_names = ['mod1', 'pool2', 'mod2', 'pool3', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7']
    bb_modules_dict = OrderedDict()
    final_head_module = None

    # Extract backbone sequential path first
    for name in bb_sequential_part_names:
        if hasattr(fm, name):
            bb_modules_dict[name] = getattr(fm, name)
            logger.debug(f"Adding backbone part: {name}")
        else:
            # This might be okay if the variant doesn't use all modules, but log it.
            logger.debug(f"DeepWV3Plus instance does not have attribute: {name}. Skipping.")

    # Extract the final classification layer sequence
    # Check common names. Add more if your DeepWV3Plus uses different names.
    potential_head_names = ['final', 'classifier', 'seg_head', 'aspp_head'] # Check DeepWV3Plus definition
    extracted_head_name = None
    for head_name in potential_head_names:
        if hasattr(fm, head_name):
            final_head_module = getattr(fm, head_name)
            extracted_head_name = head_name
            logger.debug(f"Found final head module: '{extracted_head_name}'")
            break
    if final_head_module is None:
        logger.error(f"Could not find a final head module (tried {potential_head_names}) in the DeepWV3Plus instance. Check its structure.")
        # Depending on HopfieldPEBALModel, maybe we can proceed without a head? Unlikely.

    # --- Logging skipped parts ---
    all_extracted_names = list(bb_modules_dict.keys()) + ([extracted_head_name] if extracted_head_name else [])
    for name, module in fm.named_children():
        if name not in all_extracted_names:
            logger.debug(f"Note: Module part '{name}' from DeepWV3Plus was not included in the extracted backbone path or final head.")

    if not bb_modules_dict:
        logger.error("Failed to extract any backbone modules (likely due to initialization issues or missing attributes in DeepWV3Plus instance).")
        return None, None

    backbone = nn.Sequential(bb_modules_dict)
    logger.info(f"Extracted backbone sequence ({len(list(backbone.children()))} modules).")

    if final_head_module is None:
        logger.error("Failed to extract a module to use as segmentation head.")
        # Return backbone only if HopfieldPEBALModel can handle a None head, otherwise return None, None
        return backbone, None

    # --- Create Segmentation Head Wrapper (unchanged logic) ---
    class SegHeadWrapper(nn.Module):
        def __init__(self, head_nn: nn.Module):
            super().__init__()
            self.head = head_nn
            self._in_channels = None
            first_conv = None
            # Try to find the first conv layer to infer input channels
            if isinstance(head_nn, nn.Conv2d):
                first_conv = head_nn
            elif isinstance(head_nn, nn.Sequential):
                for layer in head_nn.modules(): # Check recursively
                    if isinstance(layer, nn.Conv2d):
                        first_conv = layer
                        break
            elif hasattr(head_nn, 'modules'): # Check generic modules attribute
                 for layer in head_nn.modules():
                    if isinstance(layer, nn.Conv2d):
                        first_conv = layer
                        break

            if first_conv and hasattr(first_conv, 'in_channels'):
                 self._in_channels = first_conv.in_channels
                 logger.info(f"SegHeadWrapper inferred input channels from head's first Conv2d: {self._in_channels}")
            else: logger.warning("Could not infer input channels for SegHeadWrapper from head module structure.")
        def forward(self, x: torch.Tensor) -> torch.Tensor: return self.head(x)

    segmentation_head = SegHeadWrapper(final_head_module)
    logger.info(f"Extracted '{extracted_head_name}' module as segmentation head wrapper.")

    return backbone, segmentation_head


# --- Model Loading Function (Refined Key Mapping & PEBAL Dim Fix) ---
def load_model(args, device):
    """Load base model structure, load checkpoint with refined key mapping, wrap in HopfieldPEBALModel."""
    logger.info(f"Loading base model '{args.base_model}'...")

    backbone, segmentation_head = None, None
    actual_backbone_output_dim = None # Will be determined later

    if args.base_model == 'simple':
        backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes, args.img_height, args.img_width)
    elif args.base_model == 'deepwv3plus':
        # Ensure necessary classes were imported successfully
        if not WIDERESNET_IMPORTED:
             raise ImportError("WiderResNetA2 failed to import during setup. Cannot load 'deepwv3plus' model.")
        if not DEEPWV3PLUS_IMPORTED:
             raise ImportError("DeepWV3Plus failed to import during setup. Cannot load 'deepwv3plus' model.")

        variant = args.wider_resnet_variant
        # Retrieve the expected structure based on the variant name (used for logging/verification)
        if variant in WIDER_RESNET_STRUCTURES:
            structure = WIDER_RESNET_STRUCTURES[variant]['structure']
            logger.info(f"Using WiderResNet variant '{variant}' definition with structure: {structure}")
        else:
            # Use default if variant not found, but warn heavily as it might cause load errors
            logger.error(f"Unknown WiderResNet variant specified: {variant}. Available: {list(WIDER_RESNET_STRUCTURES.keys())}")
            default_variant = '38' # Or choose another appropriate default
            structure = WIDER_RESNET_STRUCTURES.get(default_variant, {}).get('structure')
            if structure is None:
                 raise ValueError(f"Default WiderResNet variant '{default_variant}' structure not found in WIDER_RESNET_STRUCTURES.")
            logger.warning(f"Falling back to default variant '{default_variant}' structure: {structure}. THIS MAY CAUSE CHECKPOINT LOADING ERRORS if it doesn't match the actual checkpoint structure.")
            variant = default_variant # Update variant name for consistency

        # --- Instantiate base model components using the corrected import function ---
        backbone, segmentation_head = import_deepwv3plus(args.num_classes, structure, variant)

    else:
        raise ValueError(f"Unsupported base_model type: {args.base_model}")

    if backbone is None:
        raise RuntimeError("Base model backbone loading/extraction failed (backbone is None). Check logs from import_deepwv3plus. Possible reasons: DeepWV3Plus init error, missing attributes in DeepWV3Plus instance.")
    if segmentation_head is None:
        logger.warning("Base model segmentation head is None after extraction. HopfieldPEBALModel instantiation might fail or behave unexpectedly if it requires a head.")

    # --- Load Checkpoint ---
    full_state_dict = None
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint required but not found: {args.checkpoint}")
    logger.info(f"Loading checkpoint file: {args.checkpoint}")
    try:
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            logger.info("Checkpoint loaded with weights_only=True.")
        except (RuntimeError, AttributeError) as e: # AttributeError might occur if trying to unpickle non-tensor data
             logger.warning(f"Could not load checkpoint with weights_only=True ({e}). Retrying with weights_only=False (might be less safe).")
             checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

        key_options = ['state_dict', 'model_state_dict', 'model', 'net']
        state_dict_key = next((k for k in key_options if isinstance(checkpoint, dict) and k in checkpoint), None)

        if state_dict_key:
            full_state_dict = checkpoint[state_dict_key]
            logger.info(f"Using state dict from key: '{state_dict_key}'")
        elif isinstance(checkpoint, dict):
             # Check if it looks like a state_dict (contains tensors)
             if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                 full_state_dict = checkpoint
                 logger.info("Using root dictionary as state dict (assuming it contains model weights).")
             else:
                # Check for nested structure if no top-level tensors found
                nested_key = next((k for k, v in checkpoint.items() if isinstance(v, dict) and any(isinstance(sv, torch.Tensor) for sv in v.values())), None)
                if nested_key:
                    logger.warning(f"Root dictionary doesn't look like a state dict. Trying nested key: '{nested_key}'")
                    full_state_dict = checkpoint[nested_key]
                    state_dict_key = nested_key # Update key for logging
                else:
                    raise TypeError(f"Checkpoint is a dictionary but neither root nor common keys ('{key_options}') seem to contain a valid state_dict.")
        else:
            raise TypeError(f"Checkpoint is not a dictionary or recognized structure. Type: {type(checkpoint)}")

        if not isinstance(full_state_dict, dict):
            raise TypeError(f"Loaded state_dict (from key: '{state_dict_key}') is not a dictionary. Type: {type(full_state_dict)}")

    except Exception as e:
        logger.error(f"Failed to load or parse checkpoint file: {args.checkpoint}", exc_info=True)
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    # --- Get Backbone Module Mapping Directly from Instantiated Backbone ---
    try:
        backbone_module_names = [name for name, _ in backbone.named_children()]
        if not backbone_module_names: raise ValueError("Instantiated backbone has no named children modules.")
        map_orig_name_to_sequential_idx = {name: i for i, name in enumerate(backbone_module_names)}
        logger.debug(f"Created backbone module mapping (OriginalName -> SeqIdx): {map_orig_name_to_sequential_idx}")
    except Exception as e:
        logger.error(f"Could not create backbone module mapping: {e}", exc_info=True)
        raise RuntimeError("Failed to determine backbone structure for checkpoint key mapping.") from e

    # --- Filter state dicts and map keys ---
    backbone_state_dict = OrderedDict(); head_state_dict = OrderedDict(); pebal_state_dict = OrderedDict()
    processed_keys = set()
    logger.info("Mapping checkpoint keys to current model structure...")
    # Stage 1: Iterate through all keys in the loaded checkpoint state_dict
    for k_ckpt, v in full_state_dict.items():
        mapped = False
        key_parts = k_ckpt.split('.')
        current_key_part = k_ckpt # For logging

        # Stage 2: Handle potential prefixes (DataParallel, module wrappers)
        prefix_to_strip = ""
        # Add more prefixes if needed based on how checkpoints were saved
        # Order matters: Check for longer prefixes first
        if current_key_part.startswith('module.base_model.'): prefix_to_strip = 'module.base_model.'
        elif current_key_part.startswith('module.backbone.'): prefix_to_strip = 'module.backbone.' # Added
        elif current_key_part.startswith('module.'): prefix_to_strip = 'module.'
        elif current_key_part.startswith('base_model.'): prefix_to_strip = 'base_model.'
        elif current_key_part.startswith('backbone.'): prefix_to_strip = 'backbone.' # Added

        if prefix_to_strip:
            key_without_prefix = current_key_part[len(prefix_to_strip):]
            key_parts = key_without_prefix.split('.')
            current_key_part = key_without_prefix
            logger.debug(f"Stripped prefix '{prefix_to_strip}' from '{k_ckpt}', now '{current_key_part}'")

        # Stage 3: Try Mapping to Backbone using the dynamic map
        is_backbone_key = False
        potential_orig_mod_name = None
        key_suffix = ""
        # Check if key starts with 'MOD_NAME...' (after stripping prefixes)
        if key_parts[0] in map_orig_name_to_sequential_idx:
             potential_orig_mod_name = key_parts[0]
             key_suffix = '.'.join(key_parts[1:])
             is_backbone_key = True
        # Map to sequential index: 'MOD_NAME.layer.weight' -> 'SEQ_IDX.layer.weight'
        if is_backbone_key and potential_orig_mod_name is not None:
            try:
                target_seq_idx = map_orig_name_to_sequential_idx[potential_orig_mod_name]
                new_key = f"{target_seq_idx}.{key_suffix}" if key_suffix else str(target_seq_idx) # Handle cases where mod is final layer
                backbone_state_dict[new_key] = v
                processed_keys.add(k_ckpt); mapped = True
                logger.debug(f"Mapped BB key: '{k_ckpt}' -> '{new_key}' (via mod '{potential_orig_mod_name}')")
            except KeyError: logger.warning(f"Logic error: Mod name '{potential_orig_mod_name}' not in map '{map_orig_name_to_sequential_idx}' for key '{k_ckpt}'.")
            except Exception as map_err: logger.warning(f"Error mapping BB key '{k_ckpt}': {map_err}")

        # Stage 4: Try mapping to Head (if it exists)
        if not mapped and segmentation_head is not None:
             # Prefixes expected in checkpoint keys for the head part
             head_prefixes_ckpt = ['segmentation_head.', 'final.', 'classifier.', '_original_segmentation_head.', 'aspp_head.', 'module.final.', 'module.segmentation_head.'] # Add more as needed
             target_head_prefix = "head." # Prefix inside SegHeadWrapper
             for prefix in head_prefixes_ckpt:
                  if k_ckpt.startswith(prefix): # Check original key with potential prefixes
                      rest_of_key = k_ckpt[len(prefix):]
                      mapped_head_key = f"{target_head_prefix}{rest_of_key}"
                      head_state_dict[mapped_head_key] = v
                      processed_keys.add(k_ckpt); mapped = True
                      logger.debug(f"Mapped Head key: '{k_ckpt}' -> '{mapped_head_key}' (via prefix '{prefix}')")
                      break
                  # Also check the key *after* stripping common prefixes
                  elif current_key_part.startswith(prefix):
                      rest_of_key = current_key_part[len(prefix):]
                      mapped_head_key = f"{target_head_prefix}{rest_of_key}"
                      head_state_dict[mapped_head_key] = v
                      processed_keys.add(k_ckpt); mapped = True
                      logger.debug(f"Mapped Head key: '{k_ckpt}' -> '{mapped_head_key}' (via prefix '{prefix}' on stripped key)")
                      break


        # Stage 5: Collect PEBAL / Hopfield keys (keep original relative name after stripping)
        if not mapped:
             pebal_prefixes = ['energy_head.', 'memory_input_proj.', 'memory_manager.', 'final_seghead_proj.',
                               'feature_adapter.', 'pebal_head.', '_memory_module.', '_pebal_module.',
                               'adapter.', 'hopfield_memory.', 'memory_readout.', 'memory_scorer.',
                               'efficient_memory.', 'memory_bank.', 'pebal_module.', 'seg_adapter.', 'ood_head.' ] # Add known prefixes
             if any(current_key_part.startswith(p) for p in pebal_prefixes):
                 pebal_state_dict[current_key_part] = v
                 processed_keys.add(k_ckpt)
                 mapped = True # Mark as processed
                 logger.debug(f"Collected PEBAL/Hopfield key: '{k_ckpt}' -> '{current_key_part}'")

    # --- Load filtered state dicts into components ---
    if backbone_state_dict:
        logger.info(f"Loading {len(backbone_state_dict)} mapped keys into backbone structure...")
        try:
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            if missing: logger.warning(f" Backbone MISSING keys: {missing}")
            if unexpected:
                 # This is often a critical error indicating architecture mismatch
                 logger.error(f" Backbone UNEXPECTED keys: {unexpected}")
                 logger.critical("UNEXPECTED keys indicate ARCHITECTURE MISMATCH between checkpoint and current model.")
                 logger.critical(f"Ensure DeepWV3Plus internal structure (defined by variant '{args.wider_resnet_variant}') matches the checkpoint.")
                 # Allow proceeding with warning for now, but error might occur later
                 # raise RuntimeError(f"Architecture mismatch loading backbone weights. Unexpected keys: {unexpected}.")
            logger.info("Backbone weights loaded successfully (onto CPU).")
        except RuntimeError as e:
             logger.critical(f"CRITICAL: Runtime error loading backbone state dict. Likely architecture mismatch.", exc_info=True)
             raise e
    else:
        # If the mapping logic failed to find BB keys, this warning is important
        logger.warning("!!! WARNING: No checkpoint keys were mapped to the backbone structure. Backbone weights might be missing or random. !!!")


    if segmentation_head is not None and head_state_dict:
        logger.info(f"Loading {len(head_state_dict)} mapped keys into segmentation head structure...")
        # Print structure for debugging mismatch
        logger.debug(f"Segmentation head structure: {segmentation_head}")
        missing, unexpected = segmentation_head.load_state_dict(head_state_dict, strict=False)
        if missing: logger.warning(f" SegHead MISSING keys: {missing}") # Check if these match the 'UNEXPECTED' keys from the log
        if unexpected: logger.warning(f" SegHead UNEXPECTED keys: {unexpected}") # Check if these match the 'MISSING' keys from the log
        logger.info("Segmentation head weights loaded successfully (onto CPU).")
    elif segmentation_head is not None:
        logger.warning("No checkpoint keys mapped to the segmentation head structure.")

    # --- Move components to target device ---
    backbone = backbone.to(device)
    if segmentation_head: segmentation_head = segmentation_head.to(device)
    logger.info(f"Base model components moved to {device}")

    # --- Determine Actual Backbone Output Dimension ---
    # Perform a dummy forward pass to find the feature dimension at the insertion point
    actual_backbone_output_dim = None
    if args.insertion_point == 'after_backbone':
        try:
            logger.debug("Determining actual backbone output dimension...")
            backbone.eval() # Ensure backbone is in eval mode
            # Create a dummy input matching expected size but on the correct device
            # Use slightly smaller size if memory is tight, but keep channels/batch size
            dummy_h = min(args.img_height, 128)
            dummy_w = min(args.img_width, 256)
            dummy_input = torch.zeros(1, 3, dummy_h, dummy_w, device=device)
            with torch.no_grad():
                dummy_output = backbone(dummy_input)
            actual_backbone_output_dim = dummy_output.shape[1] # Get channel dimension
            logger.info(f"Determined actual backbone output dimension (at insertion point): {actual_backbone_output_dim}")
            del dummy_input, dummy_output # Free memory
            if device.type == 'cuda': torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to determine backbone output dimension via dummy pass: {e}", exc_info=True)
            logger.error("Cannot proceed reliably without knowing the backbone output dimension for PEBAL.")
            raise RuntimeError("Failed to determine backbone output dimension.") from e

    # --- Determine Effective Target Dimension for PEBAL Input ---
    # This dimension dictates whether an adapter is needed *before* memory_input_proj
    effective_target_dim_for_pebal_input = args.target_feature_dim # Default

    if args.insertion_point == 'after_backbone':
        if actual_backbone_output_dim is None:
            # Should have been caught above, but double-check
            raise RuntimeError("Backbone output dimension calculation failed, cannot proceed.")
        # ** THE FIX **: If inserting after backbone, the PEBAL components in the CHECKPOINT
        # likely operated on the RAW backbone features. We need to tell the HopfieldPEBALModel
        # constructor that the dimension *at this point* is the actual backbone output dimension
        # to prevent it from inserting an unnecessary adapter based on args.target_feature_dim
        # which might be intended only for the final seg head input.
        effective_target_dim_for_pebal_input = actual_backbone_output_dim
        logger.info(f"For 'after_backbone' insertion, setting effective target dim for PEBAL input to {effective_target_dim_for_pebal_input} (matching backbone output) to align with checkpoint expectations.")
    # Else (e.g., 'after_seghead'), the user's target_feature_dim likely applies directly.

    # --- Instantiate HopfieldPEBALModel ---
    if not HOPFIELD_MODEL_IMPORTED:
        raise ImportError("HopfieldPEBALModel class is required but failed to import or is unavailable.")

    logger.info("Instantiating HopfieldPEBALModel...")
    try:
        hopfield_kwargs = {
            'backbone': backbone,
            'segmentation_head': segmentation_head,
            'num_classes': args.num_classes,
            'memory_feature_dim': args.memory_feature_dim, # Dim *inside* memory
            'memory_size': args.memory_size,
            'insertion_point': args.insertion_point,
            # ** USE THE CALCULATED DIMENSION HERE **
            'target_feature_dim': effective_target_dim_for_pebal_input,
            'use_efficient_memory': True,
            'use_faiss': (not args.disable_faiss),
            'memory_log_interval': 1000,
            'memory_log_verbose': args.debug,
            'use_efficient_decoder': args.use_efficient_decoder,
            'efficient_decoder_kwargs': {'attention_heads': args.attention_heads} if args.use_efficient_decoder else None,
            'memory_beta': args.memory_beta,
            # Pass the original target dim intended for the segmentation head separately if the model needs it
            # This depends on HopfieldPEBALModel's __init__ signature. Assuming it only uses target_feature_dim for the adapter logic.
            # 'final_seghead_target_dim': args.target_feature_dim # Example if needed
        }
        logger.debug(f"HopfieldPEBALModel Kwargs: { {k: v for k, v in hopfield_kwargs.items() if k not in ['backbone', 'segmentation_head']} }") # Log kwargs

        model = HopfieldPEBALModel(**hopfield_kwargs).to(device)
        logger.info("HopfieldPEBALModel instantiated successfully.")

    except TypeError as te:
         logger.error(f"TypeError creating HopfieldPEBALModel: {te}", exc_info=True)
         logger.error("Check if HopfieldPEBALModel.__init__ signature matches the provided arguments.")
         raise
    except Exception as e:
        logger.error(f"Error creating HopfieldPEBALModel: {e}", exc_info=True)
        raise

    # --- Load PEBAL-specific weights into the combined model ---
    if pebal_state_dict:
        logger.info(f"Loading {len(pebal_state_dict)} PEBAL/Hopfield-specific keys into HopfieldPEBALModel...")
        # Now, the model's layers (like memory_input_proj) should have the correct dimensions matching the checkpoint
        missing, unexpected = model.load_state_dict(pebal_state_dict, strict=False)
        if missing: logger.warning(f" HopfieldPEBALModel MISSING PEBAL keys: {missing}")
        if unexpected: logger.warning(f" HopfieldPEBALModel UNEXPECTED PEBAL keys: {unexpected}") # Should be fewer/none now
        logger.info("PEBAL/Hopfield weights loaded successfully.")
    else:
        logger.warning("No PEBAL/Hopfield-specific keys found/collected from the checkpoint.")

    # --- Final Check for Unused Checkpoint Keys ---
    if full_state_dict:
        unused_keys = set(full_state_dict.keys()) - processed_keys
        if unused_keys:
            logger.warning(f"Checkpoint keys COMPLETELY UNUSED after loading: {len(unused_keys)} keys.")
            # Log unused keys especially if backbone/head mapping failed
            log_limit = 20; unused_list = sorted(list(unused_keys))
            logger.warning(f"First {min(log_limit, len(unused_list))} unused keys: {unused_list[:log_limit]}")
            if len(backbone_state_dict) == 0:
                 logger.error("Crucially, no backbone keys were used. Check checkpoint key names and mapping logic (prefixes like 'backbone.', 'module.backbone.').")
            if len(head_state_dict) == 0 and segmentation_head is not None:
                 logger.warning("No segmentation head keys were used. Check checkpoint key names and mapping logic (prefixes like 'segmentation_head.', 'final.').")
            try:
                unused_path = os.path.join(args.output_dir, "_UNUSED_CHECKPOINT_KEYS.txt")
                with open(unused_path, 'w') as f: f.write(f"{len(unused_keys)} unused keys:\n" + "\n".join(unused_list))
            except Exception as write_err: logger.warning(f"Could not write unused keys file: {write_err}")
        else:
            logger.info("All keys from the checkpoint were processed.")

    return model


# --- Evaluation Metrics Functions (unchanged) ---
def evaluate_segmentation(predictions: np.ndarray, targets: np.ndarray, num_classes: int, void_id: int = 255) -> float:
    """Calculate mean IoU for segmentation predictions"""
    try:
        predictions = predictions.flatten()
        targets = targets.flatten()
        valid_mask = (targets != void_id)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if predictions.size == 0 or targets.size == 0:
            logger.debug("evaluate_segmentation: No valid pixels.")
            return 0.0

        # Ensure predictions are within the valid class range
        predictions = np.clip(predictions, 0, num_classes-1)
        # Ensure targets are within the valid class range (excluding void handled above)
        targets = np.clip(targets, 0, num_classes-1)

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        # Use np.add.at for safe accumulation
        np.add.at(conf_matrix, (targets.astype(np.int32), predictions.astype(np.int32)), 1)

        intersection = np.diag(conf_matrix)
        ground_truth_set = conf_matrix.sum(axis=1)
        predicted_set = conf_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        # Handle division by zero for classes not present or predicted
        iou = np.zeros_like(intersection, dtype=np.float32)
        valid_union = union > 0
        iou[valid_union] = intersection[valid_union] / union[valid_union].astype(np.float32)

        # Calculate mean IoU only over classes present in the ground truth
        valid_iou_mask = ground_truth_set > 0
        mean_iou = np.mean(iou[valid_iou_mask]) if np.any(valid_iou_mask) else 0.0

        return mean_iou if not np.isnan(mean_iou) else 0.0
    except Exception as e:
        logger.error(f"Error in evaluate_segmentation: {e}", exc_info=True)
        return 0.0

def evaluate_ood_detection(energy_maps: np.ndarray, targets: np.ndarray, anomaly_id: int,
                           void_id: int = 255, return_scores: bool = False) -> Any:
    """Calculate OOD detection metrics"""
    flat_energy, binary_targets = np.array([]), np.array([])
    num_ood, num_inlier, total_valid = 0, 0, 0

    try:
        flat_energy = energy_maps.flatten()
        flat_targets = targets.flatten()
        valid_mask = (flat_targets != void_id)

        if not np.any(valid_mask):
            logger.debug("evaluate_ood_detection: No valid pixels.")
            return (0.5, 0.0, 1.0, flat_energy, binary_targets) if return_scores else (0.5, 0.0, 1.0)

        flat_energy = flat_energy[valid_mask]
        binary_targets = (flat_targets[valid_mask] == anomaly_id).astype(int)

        num_ood = np.sum(binary_targets == 1)
        num_inlier = np.sum(binary_targets == 0)
        total_valid = len(binary_targets)

        if num_ood == 0 and num_inlier == 0: # Double check if somehow total_valid became 0
             logger.debug("evaluate_ood_detection: No valid non-void pixels after masking.")
             return (0.5, 0.0, 1.0, flat_energy, binary_targets) if return_scores else (0.5, 0.0, 1.0)
        elif num_ood == 0:
            logger.debug(f"No OOD pixels (ID {anomaly_id}) found among {total_valid} valid pixels.")
            # AUPRC is ill-defined (or trivial=1.0) if only negatives exist. Sklearn AP returns NaN.
            # Let's return proportion of positives (0.0) as AUPRC and neutral AUROC/FPR95.
            return (0.5, 0.0, 1.0, flat_energy, binary_targets) if return_scores else (0.5, 0.0, 1.0)
        elif num_inlier == 0:
            logger.debug(f"No Inlier pixels found among {total_valid} valid pixels (all OOD).")
            # AUPRC is ill-defined (or trivial=1.0) if only positives exist. Sklearn AP returns NaN.
            # Let's return proportion of positives (1.0) as AUPRC and neutral AUROC/FPR95.
            return (0.5, 1.0, 0.0, flat_energy, binary_targets) if return_scores else (0.5, 1.0, 0.0)

        # Handle non-finite values robustly
        finite_mask = np.isfinite(flat_energy)
        if not np.all(finite_mask):
            num_non_finite = np.sum(~finite_mask)
            logger.warning(f"{num_non_finite} non-finite energy values found out of {flat_energy.size}. Replacing.")
            # Replace with values that won't crash metrics but might skew them
            # Use median/mean of finite values if available, else 0
            if np.any(finite_mask):
                median_finite = np.median(flat_energy[finite_mask])
                flat_energy[~finite_mask] = median_finite
            else:
                flat_energy[~finite_mask] = 0.0 # Fallback if all are non-finite

        # Calculate metrics
        auroc = roc_auc_score(binary_targets, flat_energy)
        auprc = average_precision_score(binary_targets, flat_energy)

        # Calculate FPR@95TPR
        fpr_roc, tpr_roc, _ = roc_curve(binary_targets, flat_energy)
        target_tpr = 0.95
        fpr95 = 1.0 # Default to worst case

        # Find the first threshold where TPR >= target_tpr
        if np.any(tpr_roc >= target_tpr):
             # Find indices where TPR is >= target
             valid_indices = np.where(tpr_roc >= target_tpr)[0]
             if len(valid_indices) > 0:
                 # roc_curve sorts thresholds such that TPR is non-decreasing.
                 # The first index corresponds to the highest threshold (lowest FPR) achieving the target TPR.
                 fpr95 = fpr_roc[valid_indices[0]]
        else:
            # If target TPR is never reached, FPR@95 is undefined (or arguably 1.0)
            logger.debug(f"Target TPR ({target_tpr}) never reached. Max TPR: {np.max(tpr_roc):.4f}. Setting FPR@95TPR to 1.0.")
            fpr95 = 1.0

        return (auroc, auprc, fpr95, flat_energy, binary_targets) if return_scores else (auroc, auprc, fpr95)

    except ValueError as e:
        logger.error(f"ValueError calculating OOD metrics: {e}. Check score/target shapes and content.", exc_info=True)
        ood_prop = float(num_ood)/total_valid if total_valid > 0 else 0.0 # Best guess for AUPRC baseline
        return (0.5, ood_prop, 1.0, flat_energy, binary_targets) if return_scores else (0.5, ood_prop, 1.0)

    except Exception as e:
        logger.error(f"Unexpected error calculating OOD metrics: {e}", exc_info=True)
        ood_prop = float(num_ood)/total_valid if total_valid > 0 else 0.0
        return (0.5, ood_prop, 1.0, flat_energy, binary_targets) if return_scores else (0.5, ood_prop, 1.0)


# --- Visualization (unchanged) ---
def visualize_results(image: np.ndarray, target: np.ndarray, prediction: np.ndarray,
                     energy: np.ndarray, output_path: str, num_classes: int,
                     anomaly_id: int, void_id: int = 255):
    """Visualize and save results as a figure with 4 subplots"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        base_filename = os.path.basename(output_path).replace('.png', '')
        fig.suptitle(f"Sample: {base_filename}", fontsize=16)

        # Display original image
        img_display = image.transpose(1, 2, 0) if image.shape[0]==3 and image.ndim==3 else image
        # Ensure it's float for manipulation
        img_display = img_display.astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        # Apply broadcasting for mean/std
        img_display = np.clip(std * img_display + mean, 0, 1)
        axes[0,0].imshow(img_display)
        axes[0,0].set_title('Original Image (Re-normalized)')
        axes[0,0].axis('off')

        # Display ground truth
        cmap_gt = plt.get_cmap('tab20', num_classes + 2) # Use a colormap with enough distinct colors
        colors_gt = cmap_gt(np.arange(num_classes + 2))
        anomaly_color = np.array([1.0, 0.0, 0.0, 1.0]) # Red for anomaly
        void_color = np.array([0.0, 0.0, 0.0, 1.0])    # Black for void

        # Handle potential non-integer target types
        target_int = target.astype(int)
        tgt_colored = np.zeros((*target.shape, 4), dtype=np.float32)
        for i in range(num_classes):
            tgt_colored[target_int == i] = colors_gt[i]
        tgt_colored[target_int == anomaly_id] = anomaly_color
        tgt_colored[target_int == void_id] = void_color

        axes[0,1].imshow(tgt_colored)
        axes[0,1].set_title(f'Ground Truth (Anomaly={anomaly_id}, Void={void_id})')
        axes[0,1].axis('off')

        # Display prediction
        cmap_pred = plt.get_cmap('tab20', num_classes)
        pred_colors = cmap_pred(np.arange(num_classes))
        pred_colored = np.zeros((*prediction.shape, 4), dtype=np.float32)
        # Handle potential non-integer prediction types and clip
        pred_clipped = np.clip(prediction.astype(int), 0, num_classes - 1)

        for i in range(num_classes):
            pred_colored[pred_clipped == i] = pred_colors[i]

        axes[1,0].imshow(pred_colored)
        axes[1,0].set_title('Prediction (Inlier Classes)')
        axes[1,0].axis('off')

        # Display energy map
        # Handle potential non-finite values before finding min/max
        energy_finite_mask = np.isfinite(energy)
        if np.any(energy_finite_mask):
            energy_finite = energy[energy_finite_mask]
            energy_min = np.min(energy_finite)
            energy_max = np.max(energy_finite)
             # Use percentile for potentially better contrast if range is huge
            # energy_min = np.percentile(energy_finite, 1)
            # energy_max = np.percentile(energy_finite, 99)
            energy_display = np.nan_to_num(energy, nan=energy_min, posinf=energy_max, neginf=energy_min)
        else:
            # All non-finite? Display as zeros.
            energy_min, energy_max = 0, 0
            energy_display = np.zeros_like(energy)
            logger.warning(f"All energy values non-finite for sample {base_filename}. Displaying zeros.")


        im = axes[1,1].imshow(energy_display, cmap='viridis', vmin=energy_min, vmax=energy_max)
        axes[1,1].set_title(f'OOD Energy (Min: {energy_min:.2f}, Max: {energy_max:.2f})')
        axes[1,1].axis('off')

        plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory

        logger.debug(f"Visualization saved: {output_path}")

    except Exception as e:
        logger.error(f"Visualization error for {output_path}: {e}", exc_info=False)


# --- Dataset Checking (unchanged) ---
def check_dataset_files(path1: Optional[str], path2: Optional[str], dataset_name: str) -> bool:
    """Verify dataset file paths exist and are accessible"""
    logger.info(f"Checking '{dataset_name}' dataset paths:")
    path1_obj = Path(path1) if path1 else None
    path2_obj = Path(path2) if path2 else None

    logger.info(f"  Images dir: {path1_obj}")
    logger.info(f"  Labels dir: {path2_obj}")

    paths_ok = True
    for path_obj, name in [(path1_obj, 'Images'), (path2_obj, 'Labels')]:
        if not path_obj:
            logger.error(f"  {name} path not provided.")
            paths_ok = False
            continue

        if not path_obj.exists():
            logger.error(f"  {name} path does not exist: {path_obj}")
            paths_ok = False
            continue

        if not path_obj.is_dir():
            logger.error(f"  {name} path is not a directory: {path_obj}")
            paths_ok = False
            continue

        # Try listing items, handle potential permission errors
        try:
            # Check if *any* file exists, not just directories
            has_items = any(p.is_file() for p in path_obj.rglob('*'))
            if not has_items:
                logger.warning(f"  {name} directory exists but appears empty or contains only subdirs: {path_obj}")
                 # Allow proceeding but warn
            else:
                logger.info(f"  Found files in {name} dir.")
        except OSError as e:
            logger.error(f"  Cannot access {name} directory {path_obj}: {e}")
            paths_ok = False

    if not paths_ok:
        logger.error(f"Dataset file check FAILED for '{dataset_name}'.")
    else:
        logger.info(f"Dataset file check preliminarily PASSED for '{dataset_name}'.")

    return paths_ok


# --- Dataset Evaluation Function (unchanged) ---
def evaluate_on_dataset(args, model, dataset_name, device):
    """Evaluate model on a specific dataset."""
    logger.info(f"===== Evaluating on {dataset_name} dataset =====")
    image_path, label_path, output_dir_ds = None, None, None
    dataset_class_to_use : Optional[type] = None # Type hint
    dataset_kwargs = {}
    is_ood_dataset = False

    # Configure paths and dataset class based on dataset_name
    output_dir_ds = Path(args.output_dir) / f"{dataset_name}_results"
    try:
        output_dir_ds.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results directory: {output_dir_ds}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir_ds}: {e}")
        return None

    # Ensure the actual Dataset classes were imported
    if not DATASETS_IMPORTED:
         logger.error(f"Cannot evaluate {dataset_name} because Dataset classes failed to import earlier.")
         (output_dir_ds / "_FAIL_DATASET_CLASS_IMPORT.txt").touch(exist_ok=True)
         return None

    if dataset_name == 'inlier':
        image_path = args.test_images
        label_path = args.test_labels
        dataset_class_to_use = SegmentationDataset
        dataset_kwargs = {
            'image_dir': image_path,
            'mask_dir': label_path,
            'image_suffix': '_leftImg8bit.png',
            'mask_suffix': '_gtFine_labelIds.png'
        }
    elif dataset_name == 'lostandfound':
        image_path = args.lostandfound_images
        label_path = args.lostandfound_labels
        is_ood_dataset = True
        dataset_class_to_use = FishyscapesDataset
        dataset_kwargs = {
            'image_dir': image_path,
            'mask_dir': label_path,
            'dataset_type': 'LostAndFound',
            'image_suffix': '.png',
            'mask_suffix': '.png'
        }
    elif dataset_name == 'static':
        image_path = args.static_images
        label_path = args.static_labels
        is_ood_dataset = True
        dataset_class_to_use = FishyscapesDataset
        dataset_kwargs = {
            'image_dir': image_path,
            'mask_dir': label_path,
            'dataset_type': 'Static',
            'image_suffix': '.png',
            'mask_suffix': '.png'
        }
    elif dataset_name == 'road_anomaly':
        image_path = args.road_anomaly_images
        label_path = args.road_anomaly_labels
        is_ood_dataset = True
        dataset_class_to_use = FishyscapesDataset
        dataset_kwargs = {
            'image_dir': image_path,
            'mask_dir': label_path,
            'dataset_type': 'RoadAnomaly',
            'image_suffix': '.png',
            'mask_suffix': '.png'
        }
    else:
        logger.error(f"Unknown dataset name provided: {dataset_name}")
        (output_dir_ds / "_FAIL_UNKNOWN_DATASET.txt").touch(exist_ok=True)
        return None

    # Check dataset paths
    if args.check_files_exist and not check_dataset_files(image_path, label_path, dataset_name):
        logger.error(f"Dataset check failed for {dataset_name}. Skipping this dataset.")
        try:
            (output_dir_ds / "_SKIPPED_DATASET_CHECK_FAILED.txt").write_text(
                f"Skipped {dataset_name} due to failed dataset path/content check."
            )
        except OSError as write_err:
            logger.warning(f"Could not write skip file: {write_err}")
        return None

    # Define transforms
    eval_img_size = (args.img_height, args.img_width)
    logger.info(f"Evaluation image size (H, W): {eval_img_size}")

    transform = transforms.Compose([
        transforms.Resize(eval_img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def pil_to_long_tensor(img: Image.Image) -> torch.Tensor:
        """Converts a PIL mask image to a Long Tensor, handling potential errors."""
        try:
            # Ensure image mode is suitable (e.g., 'L', 'I', 'P')
            if img.mode not in ['L', 'I', 'P', 'I;16']:
                 logger.warning(f"Mask image mode is '{img.mode}', attempting conversion to 'L'. Data loss might occur.")
                 img = img.convert('L')
            # Convert to numpy array first
            np_img = np.array(img, dtype=np.int64) # Use int64 for safety
            return torch.from_numpy(np_img)
        except Exception as e:
            logger.error(f"Error converting mask PIL to Long Tensor: {e}. Mask might be invalid.", exc_info=False)
            # Return a tensor full of void IDs as fallback
            return torch.full(eval_img_size, args.void_id, dtype=torch.long)

    mask_transform = transforms.Compose([
        transforms.Resize(eval_img_size, interpolation=InterpolationMode.NEAREST), # MUST be NEAREST for masks
        transforms.Lambda(pil_to_long_tensor)
    ])

    # Create Dataset and DataLoader
    if dataset_class_to_use is None:
         logger.error(f"Dataset class was not assigned for {dataset_name}. Cannot proceed.")
         return None

    try:
        # Update base kwargs with common args
        dataset_kwargs.update({
            'transform': transform,
            'mask_transform': mask_transform,
            'img_height': args.img_height,
            'img_width': args.img_width,
            'void_id': args.void_id
        })

        # Add specific args if dataset class accepts them using inspect
        import inspect
        sig = inspect.signature(dataset_class_to_use.__init__)
        accepted_params = sig.parameters.keys()

        final_dataset_kwargs = {}
        for k, v in dataset_kwargs.items():
            if k in accepted_params:
                final_dataset_kwargs[k] = v

        # Add class/anomaly IDs if needed and accepted
        if 'num_classes' in accepted_params:
            final_dataset_kwargs['num_classes'] = args.num_classes
        if is_ood_dataset and 'anomaly_id' in accepted_params:
            final_dataset_kwargs['anomaly_id'] = args.anomaly_id

        logger.debug(f"Instantiating {dataset_name} ({dataset_class_to_use.__name__}) with args: {list(final_dataset_kwargs.keys())}")
        dataset = dataset_class_to_use(**final_dataset_kwargs)

        if len(dataset) == 0:
            # Provide more specific guidance
            raise ValueError(f"Dataset '{dataset_name}' initialized but found 0 samples. "
                             f"Check paths ('{image_path}', '{label_path}'), "
                             f"suffixes (e.g., image: '{dataset_kwargs.get('image_suffix', 'N/A')}', mask: '{dataset_kwargs.get('mask_suffix', 'N/A')}'), "
                             f"and ensure files exist directly within the specified directories or subdirectories.")

        logger.info(f"Created {dataset_name} dataset ({len(dataset)} samples).")

        # Create DataLoader
        is_cuda = device.type == 'cuda' and not args.force_cpu
        persistent_workers = (args.num_workers > 0 and sys.platform != "win32") # Persistent workers often problematic on Windows
        if persistent_workers:
            logger.debug(f"Using persistent workers ({args.num_workers}) for DataLoader.")
        else:
             if args.num_workers > 0: logger.debug("Not using persistent workers for DataLoader.")


        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False, # Evaluation should not shuffle
            num_workers=args.num_workers,
            pin_memory=is_cuda, # Pin memory if using GPU
            drop_last=False, # Process all samples
            persistent_workers=persistent_workers,
            prefetch_factor=2 if args.num_workers > 0 else None,
            # Consider timeout if workers hang
            # timeout=120 if args.num_workers > 0 else 0
        )

    except FileNotFoundError as fnf_err:
         logger.error(f"FileNotFoundError creating dataset/loader for '{dataset_name}': {fnf_err}", exc_info=True)
         logger.error("Check if dataset paths are correct and accessible by the user running the script.")
         try:
             (output_dir_ds / f"_FAIL_DATALOADER_FILENOTFOUND.txt").write_text(f"{fnf_err}")
         except OSError as write_err:
             logger.warning(f"Could not write dataloader failure file: {write_err}")
         return None

    except ValueError as val_err: # Catch the empty dataset error specifically
        logger.error(f"ValueError creating dataset/loader for '{dataset_name}': {val_err}", exc_info=False) # Don't need full traceback for this
        try:
            (output_dir_ds / f"_FAIL_DATALOADER_VALUEERROR_EMPTY.txt").write_text(f"{val_err}")
        except OSError as write_err:
             logger.warning(f"Could not write dataloader failure file: {write_err}")
        return None

    except Exception as e:
        logger.error(f"Failed to create dataset/loader for '{dataset_name}': {e}", exc_info=True)
        try:
            (output_dir_ds / f"_FAIL_DATALOADER_{type(e).__name__}.txt").write_text(f"{e}")
        except OSError as write_err:
            logger.warning(f"Could not write dataloader failure file: {write_err}")
        return None

    # Evaluation Loop
    model.eval() # Ensure model is in evaluation mode
    metrics = {}
    all_ood_scores_targets = [] # Store tuples of (scores_np, targets_np)
    all_mious = []
    processed_samples = 0
    outputs_to_save = []
    mem_tracker = getattr(model, 'memory_tracker', None) # Get tracker if exists
    process = psutil.Process()

    with torch.no_grad(): # Disable gradient calculations globally
        autocast_enabled = (device.type == 'cuda') and (not args.force_cpu)
        # Prefer bfloat16 if available and device supports it, otherwise float16
        amp_dtype = torch.bfloat16 if (autocast_enabled and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16
        logger.info(f"Using AMP autocast: {autocast_enabled} with dtype: {amp_dtype if autocast_enabled else 'N/A'}")

        # Use autocast context manager
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled, dtype=amp_dtype if autocast_enabled else None):
            pbar = tqdm(data_loader, desc=f"Evaluating {dataset_name}", leave=False, dynamic_ncols=True)

            for i, batch_data in enumerate(pbar):
                batch_stems = [] # Initialize for each batch
                try:
                    start_cpu_mem_mb = process.memory_info().rss / (1024*1024) if PSUTIL_AVAILABLE else 0
                    start_gpu_mem_mb = torch.cuda.memory_allocated(device)/(1024*1024) if device.type == 'cuda' else 0

                    # --- Unpack Batch Data Robustly ---
                    if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 2:
                        logger.error(f"Batch {i}: Invalid data format received from DataLoader (expected list/tuple of >= 2 items). Type: {type(batch_data)}, Len: {len(batch_data) if isinstance(batch_data, (list, tuple)) else 'N/A'}. Skipping.")
                        continue

                    images, masks = batch_data[0], batch_data[1]

                    # Check tensor types and shapes
                    if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
                        logger.error(f"Batch {i}: Expected Tensors, got Image Type: {type(images)}, Mask Type: {type(masks)}. Skipping.")
                        continue
                    if images.ndim != 4 or masks.ndim != 3:
                         logger.error(f"Batch {i}: Unexpected tensor dimensions. Images: {images.ndim} (expected 4), Masks: {masks.ndim} (expected 3). Skipping.")
                         continue

                    # Get file stems if provided by dataset
                    if len(batch_data) >= 3 and isinstance(batch_data[2], (list, tuple)):
                         batch_stems = list(batch_data[2])
                         # Verify length matches batch size
                         if len(batch_stems) != images.shape[0]:
                             logger.warning(f"Batch {i}: Number of stems ({len(batch_stems)}) doesn't match batch size ({images.shape[0]}). Generating fallback stems.")
                             batch_stems = [f"{dataset_name}_batch{i}_idx{b}" for b in range(images.shape[0])]
                    else:
                         # Generate fallback stems if not provided
                         batch_stems = [f"{dataset_name}_batch{i}_idx{b}" for b in range(images.shape[0])]
                    # --- End Unpack Batch Data ---

                    images = images.to(device, non_blocking=True)
                    # Keep masks on CPU for now, move only if needed on device, or process directly on CPU later

                    # --- Model Inference ---
                    outputs = model(images)

                    # Check output type and get required tensors
                    if not isinstance(outputs, dict):
                        logger.error(f"Batch {i}: Model output is not a dictionary (Type: {type(outputs)}). Cannot extract logits/energy. Skipping.")
                        continue

                    logits = outputs.get('seg_logits')
                    energy = outputs.get('combined_energy') # Or 'ood_energy', 'energy_score' etc.

                    if logits is None:
                        logger.error(f"Batch {i}: Model output dictionary missing 'seg_logits'. Keys: {outputs.keys()}. Skipping.")
                        continue
                    if not isinstance(logits, torch.Tensor):
                         logger.error(f"Batch {i}: 'seg_logits' is not a Tensor (Type: {type(logits)}). Skipping.")
                         continue

                    if energy is None:
                        logger.warning(f"Batch {i}: Model output dictionary missing energy score ('combined_energy'). Keys: {outputs.keys()}. OOD metrics will be based on zeros.")
                        # Create a zero energy map with expected shape [B, 1, H, W]
                        energy = torch.zeros((logits.shape[0], 1, *logits.shape[2:]), device=device, dtype=logits.dtype)
                    elif not isinstance(energy, torch.Tensor):
                         logger.warning(f"Batch {i}: Energy score ('combined_energy') is not a Tensor (Type: {type(energy)}). OOD metrics will be based on zeros.")
                         energy = torch.zeros((logits.shape[0], 1, *logits.shape[2:]), device=device, dtype=logits.dtype)
                    # --- End Model Inference ---


                    # --- Post-processing ---
                    target_size = (args.img_height, args.img_width) # Use args definition H, W

                    # Resize logits if necessary (model might output at different stride)
                    if logits.shape[2:] != target_size:
                        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

                    # Resize energy if necessary (should usually match logits or be Bx1xHxW)
                    if energy.shape[2:] != target_size:
                        # Check if energy is per-pixel or maybe per-image
                        if energy.ndim == 4 and energy.shape[1] == 1: # Expect Bx1xHxW
                             energy = F.interpolate(energy, size=target_size, mode='bilinear', align_corners=False)
                        # Add handling here if energy has a different format (e.g., Bx1) -> replicate spatially?
                        else:
                             logger.warning(f"Batch {i}: Energy map has unexpected shape {energy.shape}. Cannot reliably resize. Using original size or zeros.")
                             # Fallback: create zeros if shape mismatch is severe
                             if energy.ndim != 4 or energy.shape[1] != 1:
                                  energy = torch.zeros((logits.shape[0], 1, *target_size), device=device, dtype=logits.dtype)


                    # Convert results to NumPy arrays on CPU for evaluation
                    # Use float32 for energy initially for metrics, can convert later for saving
                    predictions_batch = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                    energy_batch = energy.squeeze(1).cpu().float().numpy() # Squeeze channel dim -> BxHxW
                    masks_batch = masks.cpu().numpy() # Masks were already on CPU

                    # Get images as numpy only if needed for viz/saving
                    images_batch_np = images.cpu().numpy() if (args.visualize or args.save_outputs) else None
                    # --- End Post-processing ---


                    # --- Process each sample in the batch ---
                    current_batch_size = images.shape[0]
                    for b in range(current_batch_size):
                        pred_map = predictions_batch[b]
                        mask_map = masks_batch[b]
                        energy_map = energy_batch[b]
                        image_np = images_batch_np[b] if images_batch_np is not None else None
                        current_stem = batch_stems[b] # Use stem corresponding to batch index

                        # --- Calculate Segmentation Metrics ---
                        sample_miou = evaluate_segmentation(pred_map, mask_map, args.num_classes, args.void_id)
                        all_mious.append(sample_miou)

                        # --- Calculate OOD Detection Metrics ---
                        # Check if there are any valid pixels (non-void) in the ground truth mask
                        if np.any(mask_map != args.void_id):
                            ood_result_tuple = evaluate_ood_detection(
                                energy_map, mask_map, args.anomaly_id, args.void_id, return_scores=True
                            )
                            # evaluate_ood_detection returns tuple: (auroc, auprc, fpr95, scores, targets) or (defaults, scores, targets)
                            if ood_result_tuple and len(ood_result_tuple) == 5:
                                scores_np, targets_np = ood_result_tuple[3], ood_result_tuple[4]
                                # Only store if there were valid pixels processed by evaluate_ood_detection
                                if scores_np is not None and targets_np is not None and scores_np.size > 0 and targets_np.size > 0:
                                    all_ood_scores_targets.append((scores_np, targets_np))
                                elif args.debug:
                                     logger.debug(f"Sample {current_stem}: No valid scores/targets returned from evaluate_ood_detection (likely only void pixels).")
                            else:
                                logger.warning(f"Sample {current_stem}: Unexpected return value from evaluate_ood_detection.")
                        elif args.debug:
                             logger.debug(f"Sample {current_stem}: Mask contains only void pixels. Skipping OOD metric calculation.")

                        processed_samples += 1
                        sample_idx_global = (i * args.batch_size) + b # Assuming constant batch size except last

                        # --- Optionally save detailed outputs ---
                        if args.save_outputs:
                            # Save potentially large arrays efficiently
                            outputs_to_save.append({
                                'index': sample_idx_global,
                                'stem': current_stem,
                                # Save masks/preds as uint8 if possible
                                'target': mask_map.astype(np.uint8),
                                'prediction': pred_map.astype(np.uint8),
                                # Save energy as float16 to save space
                                'energy': energy_map.astype(np.float16)
                            })

                        # --- Optionally visualize the first few samples ---
                        if args.visualize and sample_idx_global < 10:
                            if image_np is None:
                                # Reload image if needed for visualization only
                                logger.warning(f"Visualization requested but image tensor wasn't kept. Skipping viz for sample {sample_idx_global}.")
                            else:
                                vis_filename = output_dir_ds / f"vis_{Path(current_stem).stem}_{sample_idx_global:04d}.png"
                                visualize_results(
                                    image_np, mask_map, pred_map, energy_map,
                                    str(vis_filename), args.num_classes,
                                    args.anomaly_id, args.void_id
                                )

                except Exception as batch_err:
                    # Log error with batch info
                    first_stem = batch_stems[0] if batch_stems else 'N/A'
                    logger.error(f"Error processing batch {i} (first stem: {first_stem}): {batch_err}", exc_info=True)
                    # Optionally try to clear memory tracker if it exists
                    if mem_tracker and hasattr(mem_tracker, 'clear_memory'):
                        try: mem_tracker.clear_memory(f"Batch {i} Error")
                        except Exception: pass # Avoid crashing logger
                    continue # Skip to the next batch

                finally: # Ensure cleanup happens even if errors occur
                    # Batch logging (memory, etc.)
                    if args.debug and (i % 20 == 0 or i == len(data_loader) - 1):
                        end_cpu_mem_mb = process.memory_info().rss / (1024*1024) if PSUTIL_AVAILABLE else 0
                        end_gpu_mem_mb = torch.cuda.memory_allocated(device)/(1024*1024) if device.type == 'cuda' else 0
                        logger.debug(f"Batch {i}/{len(data_loader)-1}: CPU Mem {start_cpu_mem_mb:.1f}->{end_cpu_mem_mb:.1f} MB | "
                                     f"GPU Mem {start_gpu_mem_mb:.1f}->{end_gpu_mem_mb:.1f} MB")

                    # Periodic memory cleanup
                    if i > 0 and i % 50 == 0: # Adjust frequency as needed
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                            logger.debug(f"Performed periodic GC and CUDA cache clear at batch {i}")

                    # Clear potentially large tensors from loop scope explicitly? Usually not needed with GC.
                    # del images, masks, logits, energy, predictions_batch, energy_batch, masks_batch, images_batch_np
                    # del outputs, ood_result_tuple

    # --- Calculate Final Metrics ---
    if processed_samples == 0:
        logger.error(f"No samples processed successfully for {dataset_name}. Check DataLoader and batch processing loop.")
        (output_dir_ds / "_FAIL_NO_SAMPLES_PROCESSED.txt").touch(exist_ok=True)
        return None

    # Calculate final segmentation mIoU (handle NaNs from samples with only void)
    valid_mious = [m for m in all_mious if not np.isnan(m)]
    final_miou = np.mean(valid_mious) if valid_mious else 0.0
    metrics['mIoU'] = final_miou
    logger.info(f"{dataset_name} Final Mean Sample mIoU (over {len(valid_mious)} samples with valid pixels): {final_miou:.6f}")

    # Calculate global OOD detection metrics if applicable
    if all_ood_scores_targets:
        logger.info(f"Calculating global OOD metrics from {len(all_ood_scores_targets)} samples' scores...")
        try:
            # Concatenate scores and targets from all samples
            # Ensure they are numpy arrays before concatenating
            global_energies = np.concatenate([s[0] for s in all_ood_scores_targets if isinstance(s[0], np.ndarray)])
            global_targets = np.concatenate([s[1] for s in all_ood_scores_targets if isinstance(s[1], np.ndarray)])

            if global_energies.size == 0 or global_targets.size == 0:
                raise ValueError("Concatenated OOD scores/targets are empty after filtering.")

            num_global_ood = np.sum(global_targets == 1) # OOD is 1
            num_global_inlier = np.sum(global_targets == 0) # Inlier is 0
            logger.info(f"Total pixels for global OOD metrics: {len(global_energies)} (OOD: {num_global_ood}, Inlier: {num_global_inlier})")

            # Use the same evaluation function, ensuring anomaly_id=1 (as targets are binary 0/1) and void_id=-1 (or any unused value)
            global_auroc, global_auprc, global_fpr95 = evaluate_ood_detection(
                global_energies, global_targets, anomaly_id=1, void_id=-1, return_scores=False
            )

            metrics['AUROC'] = global_auroc
            metrics['AUPRC'] = global_auprc
            metrics['FPR@95TPR'] = global_fpr95

            logger.info(f"{dataset_name} Global OOD - AUROC: {global_auroc:.6f}, AUPRC: {global_auprc:.6f}, FPR@95TPR: {global_fpr95:.6f}")

        except ValueError as e: # Catch specific concatenation or empty array errors
             logger.error(f"ValueError calculating global OOD metrics for {dataset_name}: {e}", exc_info=True)
             metrics['AUROC'] = 0.5
             metrics['AUPRC'] = 0.0 # Indicate failure
             metrics['FPR@95TPR'] = 1.0
             (output_dir_ds / "_FAIL_GLOBAL_OOD_CALC_VALUEERROR.txt").write_text(f"{e}")
        except Exception as e:
            logger.error(f"Failed to calculate global OOD metrics for {dataset_name}: {e}", exc_info=True)
            metrics['AUROC'] = 0.5
            metrics['AUPRC'] = 0.0
            metrics['FPR@95TPR'] = 1.0
            (output_dir_ds / "_FAIL_GLOBAL_OOD_CALC.txt").write_text(f"{e}")

    else:
        logger.warning(f"No valid OOD scores/targets collected for {dataset_name}. Cannot calculate global OOD metrics. Setting defaults.")
        metrics['AUROC'] = 0.5
        metrics['AUPRC'] = 0.0
        metrics['FPR@95TPR'] = 1.0

    # --- Save Results ---
    try:
        # Save metrics to NPY and TXT
        metrics_npy_path = output_dir_ds / "metrics.npy"
        np.save(metrics_npy_path, metrics)

        metrics_txt_path = output_dir_ds / "metrics.txt"
        with open(metrics_txt_path, 'w') as f:
            f.write(f"Metrics for {dataset_name} ({processed_samples} samples processed):\n{'='*20}\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.6f}\n")

        logger.info(f"Metrics saved to {metrics_npy_path} and {metrics_txt_path}")

    except Exception as e:
        logger.error(f"Failed to save metrics files: {e}")

    # Save detailed outputs if requested and available
    if args.save_outputs and outputs_to_save:
        outputs_path = output_dir_ds / "detailed_outputs.npz"
        logger.info(f"Saving detailed outputs for {len(outputs_to_save)} samples to {outputs_path}...")

        save_dict = {}
        try:
            # Efficiently stack arrays for specific keys
            keys_to_stack = ['target', 'prediction', 'energy']
            keys_regular = ['index', 'stem']

            for k in keys_regular:
                 if outputs_to_save and k in outputs_to_save[0]:
                     save_dict[k] = np.array([item[k] for item in outputs_to_save])

            for k in keys_to_stack:
                 if outputs_to_save and k in outputs_to_save[0]:
                     # Check types before stacking
                     arrays_to_stack = [item[k] for item in outputs_to_save]
                     if all(isinstance(arr, np.ndarray) for arr in arrays_to_stack):
                         save_dict[k] = np.stack(arrays_to_stack, axis=0)
                     else:
                          logger.warning(f"Cannot stack key '{k}' as not all items are numpy arrays. Saving as object array.")
                          save_dict[k] = np.array(arrays_to_stack, dtype=object)

            if save_dict:
                np.savez_compressed(outputs_path, **save_dict)
                logger.info(f"Detailed outputs saved: {outputs_path} (Size: {outputs_path.stat().st_size/1e6:.2f} MB)")
            else:
                logger.warning("No data prepared for saving in detailed_outputs.npz")

        except Exception as e:
            logger.error(f"Failed to save detailed outputs NPZ: {e}", exc_info=True)
            # Try saving a failure marker
            (outputs_path.parent / "_FAIL_SAVE_DETAILED_OUTPUTS.txt").write_text(f"{e}")


    logger.info(f"===== Evaluation finished for {dataset_name} =====")
    return metrics


# --- Main Evaluation Function (unchanged) ---
def evaluate(args):
    # Setup logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Remove existing handlers to avoid duplicate logs if re-run in same session
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', force=True) # Use force=True for reconfiguration
    logger.setLevel(log_level) # Ensure our specific logger uses the level too

    logger.info("Starting Hopfield-PEBAL Evaluation Script...")
    logger.info("Script Arguments:")
    args_dict_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    for k, v in sorted(args_dict_log.items()): # Log sorted args
        logger.info(f"  {k}: {v}")

    # Check if critical imports succeeded
    if not check_critical_imports(args):
        logger.critical("Critical imports failed. Cannot continue evaluation.")
        sys.exit(1)

    # --- Device Setup ---
    if args.force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Optional: Set specific GPU if multiple available
        # torch.cuda.set_device(0) # Or get from args
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU.")

    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"CUDNN Version: {torch.backends.cudnn.version()}")
        # Enable cuDNN benchmark mode if input sizes are constant, can speed up training/eval
        # Might use more memory initially. Disable if input sizes vary a lot.
        torch.backends.cudnn.benchmark = (args.batch_size > 1)
        logger.info(f"cuDNN Benchmark enabled: {torch.backends.cudnn.benchmark}")

    # --- Output Directory and Parameter Saving ---
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output Directory: {output_dir.resolve()}") # Log absolute path

        params_path = output_dir / "parameters.json"
        # Use a simple encoder for Path objects
        class PathEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Path):
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
        with open(params_path, 'w') as f:
            json.dump(vars(args), f, indent=4, cls=PathEncoder)

        logger.info(f"Run parameters saved to: {params_path}")

    except Exception as e:
        # Use absolute path in error message
        logger.critical(f"CRITICAL FAILURE: Could not create output directory or save parameters: {output_dir.resolve()}. Error: {e}. Exiting.", exc_info=True)
        sys.exit(1)

    # --- Load Model ---
    model = None
    try:
        model = load_model(args, device)
        model.eval() # Ensure model is in eval mode after loading

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model Loaded: {type(model).__name__} | Total Params: {total_params:,} | Trainable Params: {trainable_params:,}")

        # Optional: Log model structure at debug level
        if args.debug:
             logger.debug(f"Model Structure:\n{model}")

    except FileNotFoundError as e:
        logger.critical(f"FAIL: Checkpoint not found: {e}")
        (output_dir / "_FAIL_CHECKPOINT_NOT_FOUND.txt").write_text(str(e))
        sys.exit(1)

    except (RuntimeError, ValueError, ImportError, TypeError) as e:
        logger.critical(f"FAIL: Error during model loading: {type(e).__name__} - {e}", exc_info=True)

        fail_filename = "_FAIL_MODEL_LOAD_ERROR.txt"
        err_str = str(e).lower()

        if "mismatch" in err_str or "unexpected key" in err_str or "missing key" in err_str:
            fail_filename = "_FAIL_MODEL_LOAD_ARCH_MISMATCH.txt"
            logger.critical("Likely ARCHITECTURE MISMATCH between checkpoint and evaluation setup/arguments.")
            logger.critical("Check base_model, variant, insertion_point, target_feature_dim against training config.")

        elif isinstance(e, ImportError):
            fail_filename = "_FAIL_MODEL_LOAD_IMPORT_ERROR.txt"

        elif isinstance(e, TypeError) and 'unexpected keyword argument' in err_str:
            fail_filename = "_FAIL_MODEL_INIT_ARG_ERROR.txt"
            logger.critical("Likely argument mismatch calling model constructor (__init__). Check HopfieldPEBALModel arguments.")

        (output_dir / fail_filename).write_text(f"{type(e).__name__}: {e}\n\nCheck logs for details.")
        sys.exit(1)

    except Exception as e:
        logger.critical(f"FAIL: Unexpected error loading model: {e}", exc_info=True)
        (output_dir / "_FAIL_MODEL_LOAD_UNEXPECTED.txt").write_text(f"Unexpected Error:\n{type(e).__name__}: {e}")
        sys.exit(1)

    # --- Determine Datasets to Evaluate ---
    datasets_to_evaluate = []
    # Use Path objects for easier checking
    ds_map = {
        'inlier': (Path(args.test_images) if args.test_images else None, Path(args.test_labels) if args.test_labels else None),
        'lostandfound': (Path(args.lostandfound_images) if args.lostandfound_images else None, Path(args.lostandfound_labels) if args.lostandfound_labels else None),
        'static': (Path(args.static_images) if args.static_images else None, Path(args.static_labels) if args.static_labels else None),
        'road_anomaly': (Path(args.road_anomaly_images) if args.road_anomaly_images else None, Path(args.road_anomaly_labels) if args.road_anomaly_labels else None)
    }

    if args.dataset == 'all':
        # Evaluate datasets where both image and label paths are provided and exist (basic check)
        datasets_to_evaluate = [name for name, paths in ds_map.items() if paths[0] and paths[1] and paths[0].exists() and paths[1].exists()]
        # Log skipped ones due to missing paths
        for name, paths in ds_map.items():
             if name not in datasets_to_evaluate:
                  if not paths[0] or not paths[1]: logger.warning(f"Skipping '{name}' for 'all' evaluation: Image or Label path not provided.")
                  elif not paths[0].exists() or not paths[1].exists(): logger.warning(f"Skipping '{name}' for 'all' evaluation: Image or Label path does not exist.")
    elif args.dataset in ds_map:
        paths = ds_map[args.dataset]
        if paths[0] and paths[1] and paths[0].exists() and paths[1].exists():
            datasets_to_evaluate.append(args.dataset)
        else:
            logger.warning(f"Requested dataset '{args.dataset}' but paths are missing or do not exist. Image: {paths[0]}, Label: {paths[1]}")

    logger.info(f"Datasets selected for evaluation: {datasets_to_evaluate}")

    if not datasets_to_evaluate:
        logger.error("No valid datasets selected or paths provided/exist. Exiting.")
        (output_dir / "_FAIL_NO_VALID_DATASETS.txt").touch(exist_ok=True)
        sys.exit(1)

    # --- Run Evaluation Loop ---
    all_metrics = {}
    evaluation_successful_at_least_once = False

    for dataset_name in datasets_to_evaluate:
        metrics = None
        dataset_output_dir = output_dir / f"{dataset_name}_results" # Ensure dataset dir is defined here too

        try:
            # Call the evaluation function for the specific dataset
            metrics = evaluate_on_dataset(args, model, dataset_name, device)

            if metrics is not None and isinstance(metrics, dict) and metrics:
                all_metrics[dataset_name] = metrics
                evaluation_successful_at_least_once = True
                logger.info(f"Successfully completed evaluation for {dataset_name}.")
            else:
                # This case implies evaluate_on_dataset returned None or empty dict (e.g., dataloader failed)
                logger.warning(f"Evaluation for {dataset_name} did not return valid metrics (likely skipped or failed early).")

        except Exception as e:
            # Catch any unexpected errors escaping evaluate_on_dataset
            logger.error(f"FAIL: Unhandled exception during '{dataset_name}' evaluation main loop: {e}", exc_info=True)
            try:
                # Try to create dataset-specific failure file
                dataset_output_dir.mkdir(parents=True, exist_ok=True)
                (dataset_output_dir / "_FAIL_UNHANDLED_EXCEPTION.txt").write_text(f"Unhandled Exception in main loop:\n{type(e).__name__}: {e}\n\nCheck logs.")
            except Exception as write_err:
                 logger.error(f"Could not write unhandled exception file for {dataset_name}: {write_err}")

        finally:
            # Clean up memory after each dataset evaluation (important!)
            logger.info(f"Cleaning up memory after evaluating {dataset_name}...")
            del metrics # Delete the dictionary holding results
            gc.collect() # Run garbage collection
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
                logger.info(f"CUDA Memory Summary after {dataset_name} cleanup: "
                            f"Allocated={torch.cuda.memory_allocated()/1e6:.1f}MB, "
                            f"Reserved={torch.cuda.memory_reserved()/1e6:.1f}MB")
            else:
                logger.info("CPU mode, GC performed.")

            logger.info(f"--- Finished cleanup for {dataset_name} ---")


    # --- Save Combined Results ---
    if evaluation_successful_at_least_once and all_metrics:
        logger.info("Saving combined metrics...")
        combined_npy_path = output_dir / "all_metrics.npy"
        combined_txt_path = output_dir / "all_metrics.txt"

        try:
            np.save(combined_npy_path, all_metrics) # Save dict of dicts
            logger.info(f"Combined metrics saved (NPY): {combined_npy_path}")

            with open(combined_txt_path, 'w') as f:
                f.write(f"Combined Evaluation Metrics ({args.checkpoint})\n" + "=" * 25 + "\n")

                for ds, metrics_dict in all_metrics.items():
                    f.write(f"\n--- {ds} ---\n")

                    if metrics_dict:
                        for k, v in metrics_dict.items():
                            f.write(f"  {k}: {v:.6f}\n")
                    else:
                        f.write("  (No valid metrics recorded)\n")

            logger.info(f"Combined metrics text saved: {combined_txt_path}")

        except Exception as e:
            logger.error(f"Failed to save combined metrics: {e}")

    elif not evaluation_successful_at_least_once:
        logger.error("Evaluation finished, but NO datasets were processed successfully.")
        (output_dir / "_FAIL_NO_DATASETS_SUCCESSFUL.txt").touch(exist_ok=True)

    else: # Should not happen if evaluation_successful_at_least_once is True, but good practice
        logger.warning("Evaluation potentially successful, but combined metrics dict is unexpectedly empty.")
        (output_dir / "_WARN_EMPTY_COMBINED_METRICS.txt").touch(exist_ok=True)

    logger.info("===== Evaluation Script Finished =====")

if __name__ == "__main__":
    args = parse_args()
    # Main script execution starts here
    evaluate(args)
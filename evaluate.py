#!/usr/bin/env python
# evaluate.py (Version 5.2.1 - Fixed Key Mapping & Inspect Import)

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
import inspect # Needed for dataset loading

# --- Setup Minimal Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("Hopfield-PEBAL-Eval")

# --- Determine paths and modify sys.path ---
try:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    # Determine project root based on expected structure
    if (script_dir / 'code').exists() and (script_dir / 'code').is_dir():
        project_root = script_dir
        code_dir = project_root / 'code'
    elif script_dir.name == 'scripts' and (script_dir.parent / 'code').exists():
        project_root = script_dir.parent
        code_dir = project_root / 'code'
    else:
        project_root = script_dir.parent # Assume script is in 'code' or similar
        code_dir = script_dir

    logger.info(f"Project root suspected: {project_root}")
    logger.info(f"Code directory suspected: {code_dir}")

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        logger.info(f"Added code directory to Python path: {code_dir}")

    if not code_dir.is_dir():
        logger.warning(f"Code directory '{code_dir}' not found or not a directory. Imports might fail.")

except Exception as path_err:
    logger.error(f"Error determining paths: {path_err}. Using current directory as fallback.", exc_info=True)
    code_dir = Path.cwd()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        logger.info(f"Added fallback code directory to Python path: {code_dir}")


# --- Import core model components and helpers ---
MYNN_IMPORTED = False
HOPFIELD_MODEL_IMPORTED = False
DATASETS_IMPORTED = False
WIDERESNET_IMPORTED = False
DEEPWV3PLUS_IMPORTED = False

# Try importing helpers, define placeholders ONLY if needed
try:
    from model.mynn import initialize_weights, Upsample
    logger.info("Imported initialize_weights and Upsample from model.mynn")
    MYNN_IMPORTED = True
except ImportError:
    logger.warning("Could not import from model.mynn. Using placeholder functions if needed.")
    def initialize_weights(*args, **kwargs): pass # Placeholder
    def Upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False) # Placeholder


# Hopfield PEBAL Model (CRITICAL)
try:
    # Assuming model definitions are in project_root/code/model/
    from hopfield_pebal_model import HopfieldPEBALModel
    HOPFIELD_MODEL_IMPORTED = True
    logger.info("Successfully imported HopfieldPEBALModel")
except ImportError as e:
    logger.critical(f"Failed to import HopfieldPEBALModel: {e}", exc_info=True)
    logger.critical("CRITICAL: Cannot continue without HopfieldPEBALModel definition.")

# Dataset classes (CRITICAL)
try:
    # Assuming dataset definitions are in project_root/code/datasets/
    from datasets.datasets import SegmentationDataset
    from datasets.fishyscapes_dataset import FishyscapesDataset
    DATASETS_IMPORTED = True
    logger.info("Successfully imported dataset classes")
except ImportError as e:
    logger.critical(f"Dataset import error: {e}", exc_info=True)
    logger.critical("CRITICAL: Cannot continue without dataset classes.")

# WiderResNet and structures (Required for deepwv3plus)
try:
    # Assuming model definitions are in project_root/code/model/
    from model.wide_resnet_base import WiderResNetA2, _NETS as WIDER_RESNET_STRUCTURES
    logger.info("Imported WiderResNetA2 and structures")
    WIDERESNET_IMPORTED = True
except ImportError as e:
    logger.error(f"Failed to import WiderResNetA2: {e}", exc_info=True)
    logger.error("This component is required if using base_model='deepwv3plus'.")
    WIDER_RESNET_STRUCTURES = {} # Empty placeholder

# DeepWV3Plus network (Required for deepwv3plus)
try:
    # Assuming model definitions are in project_root/code/model/
    from model.wide_network import DeepWV3Plus
    DEEPWV3PLUS_IMPORTED = True
    logger.info("Imported DeepWV3Plus class")
except ImportError as e:
    logger.error(f"Failed to import DeepWV3Plus: {e}", exc_info=True)
    logger.error("This model is required if using base_model='deepwv3plus'.")


# --- Check critical imports and exit early if missing ---
def check_critical_imports(args):
    """Check if all critical imports for the selected model are available"""
    critical_import_errors = []
    if not HOPFIELD_MODEL_IMPORTED: critical_import_errors.append("HopfieldPEBALModel")
    if not DATASETS_IMPORTED: critical_import_errors.append("Dataset classes")
    if args.base_model == 'deepwv3plus':
        if not WIDERESNET_IMPORTED: critical_import_errors.append("WiderResNetA2 (for deepwv3plus)")
        if not DEEPWV3PLUS_IMPORTED: critical_import_errors.append("DeepWV3Plus (for deepwv3plus)")

    if critical_import_errors:
        logger.critical(f"CRITICAL: Missing required imports: {', '.join(critical_import_errors)}")
        return False
    return True

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Hopfield-PEBAL model for OOD detection')
    # Dataset paths
    parser.add_argument('--test_images', type=str, help='Path to INLIER test image directory')
    parser.add_argument('--test_labels', type=str, help='Path to INLIER test label directory (labelIds)')
    parser.add_argument('--lostandfound_images', type=str, help='Path to LostAndFound image directory')
    parser.add_argument('--lostandfound_labels', type=str, help='Path to LostAndFound label directory')
    parser.add_argument('--static_images', type=str, help='Path to Static image directory')
    parser.add_argument('--static_labels', type=str, help='Path to Static label directory')
    parser.add_argument('--road_anomaly_images', type=str, help='Path to Road Anomaly image directory')
    parser.add_argument('--road_anomaly_labels', type=str, help='Path to Road Anomaly label directory')
    parser.add_argument('--dataset', type=str, default='all', choices=['inlier', 'lostandfound', 'static', 'road_anomaly', 'all'], help='Which dataset(s) to evaluate on')
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (REQUIRED)')
    parser.add_argument('--base_model', type=str, default='deepwv3plus', choices=['deepwv3plus', 'simple'], help='Base segmentation model')
    parser.add_argument('--wider_resnet_variant', type=str, default='38', choices=['16', '20', '38'], help='WiderResNet variant for backbone (MUST MATCH CHECKPOINT)')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of INLIER classes')
    parser.add_argument('--memory_feature_dim', type=int, default=256, help='Dimension of memory features')
    parser.add_argument('--memory_beta', type=float, default=8.0, help='Beta for memory energy')
    parser.add_argument('--memory_size', type=int, default=2000, help='Memory bank size')
    parser.add_argument('--attention_heads', type=int, default=4, help='Attention heads (for efficient decoder)')
    parser.add_argument('--insertion_point', type=str, default='after_backbone', choices=['after_backbone', 'after_seghead'], help='PEBAL insertion point')
    parser.add_argument('--target_feature_dim', type=int, default=304, help='Target dimension expected by segmentation head *after* potential adapter')
    parser.add_argument('--use_efficient_decoder', action='store_true', help='Use EfficientSegmentationDecoder')
    parser.add_argument('--disable_faiss', action='store_true', help='Disable FAISS for memory')
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--output_dir', type=str, default='./results/eval', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize first few samples')
    parser.add_argument('--save_outputs', action='store_true', help='Save detailed outputs (can be large!)')
    parser.add_argument('--anomaly_id', type=int, default=19, help='Anomaly class ID in OOD datasets')
    parser.add_argument('--void_id', type=int, default=255, help='Void/ignore class ID in labels')
    # Utility
    parser.add_argument('--check_files_exist', action='store_true', default=True, help='Check dataset files exist before loading')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU execution')
    parser.add_argument('--img_height', type=int, default=256, help='Evaluation image height')
    parser.add_argument('--img_width', type=int, default=512, help='Evaluation image width')
    return parser.parse_args()

# --- Simple Model Creation (for testing) ---
def create_simple_backbone_for_testing(num_classes=19, img_h=256, img_w=512):
    class SimpleBackbone(nn.Module):
        def __init__(self): super().__init__(); self.conv1=nn.Conv2d(3,64,7,2,3,bias=False); self.bn1=nn.BatchNorm2d(64); self.relu=nn.ReLU(True); self.pool1=nn.MaxPool2d(3,2,1); self.conv2=nn.Conv2d(64,128,3,1,1,bias=False); self.bn2=nn.BatchNorm2d(128); self.conv3=nn.Conv2d(128,256,3,2,1,bias=False); self.bn3=nn.BatchNorm2d(256)
        def forward(self, x): x=self.relu(self.bn1(self.conv1(x))); x=self.pool1(x); x=self.relu(self.bn2(self.conv2(x))); x=self.relu(self.bn3(self.conv3(x))); return x
    class SimpleSegHead(nn.Module):
        def __init__(self,in_channels,num_classes): super().__init__(); self.head=nn.Sequential(nn.Conv2d(in_channels,128,3,1,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, num_classes, 1)); self._in_channels=in_channels
        def forward(self, x): return self.head(x)
    logger.info("Creating simple backbone and head for testing.")
    b = SimpleBackbone()
    # Determine output dim dynamically
    out_dim = 256
    try:
        b.eval(); dummy_input=torch.zeros(1, 3, img_h, img_w)
        with torch.no_grad(): out_dim=b(dummy_input).shape[1]
    except Exception as e: logger.warning(f"Could not determine simple backbone output dimension: {e}. Assuming 256.")
    return b, SimpleSegHead(out_dim, num_classes)

# --- DeepWV3Plus Import Function ---
def import_deepwv3plus(num_classes: int) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """Imports and instantiates DeepWV3Plus, then extracts backbone and segmentation head parts."""
    if not DEEPWV3PLUS_IMPORTED or not WIDERESNET_IMPORTED:
        logger.critical("DeepWV3Plus or WiderResNetA2 failed to import. Cannot create model.")
        return None, None

    logger.info(f"Attempting to initialize DeepWV3Plus with num_classes={num_classes}")
    fm: Optional[DeepWV3Plus] = None
    try:
        # Instantiate DeepWV3Plus (assuming it only needs num_classes)
        fm = DeepWV3Plus(num_classes=num_classes)
        logger.info(f"Initialized DeepWV3Plus successfully.")
    except Exception as e:
         logger.error(f"Error initializing DeepWV3Plus: {e}", exc_info=True)
         return None, None

    if fm is None: return None, None

    # --- Extract backbone and head parts based on common naming conventions ---
    bb_sequential_part_names = ['mod1', 'pool2', 'mod2', 'pool3', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7']
    bb_modules_dict = OrderedDict()
    final_head_module = None
    extracted_head_name = None

    # Extract backbone parts
    for name in bb_sequential_part_names:
        if hasattr(fm, name):
            bb_modules_dict[name] = getattr(fm, name)

    # Extract the final classification layer sequence
    potential_head_names = ['final', 'classifier', 'seg_head', 'aspp_head']
    for head_name in potential_head_names:
        if hasattr(fm, head_name):
            final_head_module = getattr(fm, head_name)
            extracted_head_name = head_name
            break

    if not bb_modules_dict:
        logger.error("Failed to extract any backbone modules from DeepWV3Plus instance.")
        return None, None
    backbone = nn.Sequential(bb_modules_dict)

    if final_head_module is None:
        logger.error(f"Could not find a final head module (tried {potential_head_names}) in DeepWV3Plus.")
        return backbone, None # Return backbone only if Hopfield model can handle it

    # --- Create Segmentation Head Wrapper ---
    class SegHeadWrapper(nn.Module):
        def __init__(self, head_nn: nn.Module):
            super().__init__()
            self.head = head_nn
            self._in_channels = None
            # Try to infer input channels (best effort)
            first_conv = None
            if isinstance(head_nn, (nn.Conv2d, nn.ConvTranspose2d)): first_conv = head_nn
            elif hasattr(head_nn, 'modules'):
                for layer in head_nn.modules():
                    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                        first_conv = layer; break
            if first_conv and hasattr(first_conv, 'in_channels'): self._in_channels = first_conv.in_channels
            else: logger.warning("Could not infer input channels for SegHeadWrapper.")
        def forward(self, x: torch.Tensor) -> torch.Tensor: return self.head(x)

    segmentation_head = SegHeadWrapper(final_head_module)
    logger.info(f"Extracted backbone ({len(list(backbone.children()))} modules) and head ('{extracted_head_name}').")
    return backbone, segmentation_head

# --- Model Loading Function (REVISED Key Mapping) ---
def load_model(args, device):
    """Load base model structure, load checkpoint with refined key mapping, wrap in HopfieldPEBALModel."""
    logger.info(f"Loading base model '{args.base_model}'...")

    backbone, segmentation_head = None, None
    actual_backbone_output_dim = None

    if args.base_model == 'simple':
        backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes, args.img_height, args.img_width)
    elif args.base_model == 'deepwv3plus':
        if not WIDERESNET_IMPORTED or not DEEPWV3PLUS_IMPORTED:
             raise ImportError("Required classes for 'deepwv3plus' (WiderResNetA2, DeepWV3Plus) failed to import.")
        # Note: import_deepwv3plus assumes structure based on num_classes, variant is mostly for logging/potential future use
        backbone, segmentation_head = import_deepwv3plus(args.num_classes)
    else:
        raise ValueError(f"Unsupported base_model type: {args.base_model}")

    if backbone is None:
        raise RuntimeError("Base model backbone loading/extraction failed.")
    if segmentation_head is None:
        logger.warning("Base model segmentation head is None after extraction.")

    # --- Load Checkpoint ---
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    try:
        try: checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception: checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False); logger.warning("Loaded checkpoint with weights_only=False.")

        key_options = ['state_dict', 'model_state_dict', 'model', 'net']
        state_dict_key = next((k for k in key_options if isinstance(checkpoint, dict) and k in checkpoint), None)
        full_state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint

        if not isinstance(full_state_dict, dict): raise TypeError("Loaded state_dict is not a dictionary.")
        if not any(isinstance(v, torch.Tensor) for v in full_state_dict.values()): raise TypeError("Loaded dictionary contains no tensors.")
        logger.info(f"Successfully loaded state_dict with {len(full_state_dict)} keys.")

    except Exception as e:
        raise RuntimeError(f"Checkpoint loading/parsing failed: {e}") from e

    # --- Get Backbone Module Mapping ---
    try:
        backbone_module_names = [name for name, _ in backbone.named_children()]
        if not backbone_module_names: raise ValueError("Instantiated backbone has no named children.")
        map_orig_name_to_sequential_idx = {name: i for i, name in enumerate(backbone_module_names)}
        logger.info(f"Backbone module mapping created (OrigName -> SeqIdx): {map_orig_name_to_sequential_idx}")
    except Exception as e:
        raise RuntimeError(f"Could not create backbone module mapping: {e}") from e

    # --- Filter state dicts and map keys ---
    backbone_state_dict = OrderedDict(); head_state_dict = OrderedDict(); pebal_state_dict = OrderedDict()
    processed_keys = set()
    logger.info("Mapping checkpoint keys...")

    for k_ckpt, v in full_state_dict.items():
        mapped = False
        current_key_part = k_ckpt

        # Strip common prefixes like 'module.'
        prefixes_to_strip = ['module.'] # Keep it simple, specific prefixes handled below
        for prefix in prefixes_to_strip:
            if current_key_part.startswith(prefix):
                current_key_part = current_key_part[len(prefix):]
                # logger.debug(f"Stripped prefix '{prefix}' -> '{current_key_part}'") # Optional debug
                break

        key_parts = current_key_part.split('.')

        # --- Try Mapping to Backbone ---
        # Handles keys like 'backbone.mod1.block...' or 'base_model.backbone.mod1...'
        # Assumes map_orig_name_to_sequential_idx contains ['mod1', 'pool2', 'mod2', ...]
        is_backbone_key = False
        potential_orig_mod_name = None
        key_suffix = ""

        # Check for 'backbone.MOD_NAME...' structure
        if len(key_parts) > 1 and key_parts[0] == 'backbone' and key_parts[1] in map_orig_name_to_sequential_idx:
             potential_orig_mod_name = key_parts[1]
             key_suffix = '.'.join(key_parts[2:])
             is_backbone_key = True
        # Check for 'base_model.backbone.MOD_NAME...' (less common but possible)
        elif len(key_parts) > 2 and key_parts[0] == 'base_model' and key_parts[1] == 'backbone' and key_parts[2] in map_orig_name_to_sequential_idx:
             potential_orig_mod_name = key_parts[2]
             key_suffix = '.'.join(key_parts[3:])
             is_backbone_key = True
             # logger.debug(f"Matched backbone key via 'base_model.backbone.' prefix: {k_ckpt}") # Optional debug
        # Check if it directly starts with MOD_NAME (e.g., if checkpoint only saved backbone)
        elif key_parts[0] in map_orig_name_to_sequential_idx:
             potential_orig_mod_name = key_parts[0]
             key_suffix = '.'.join(key_parts[1:])
             is_backbone_key = True
             # logger.debug(f"Matched backbone key directly by module name: {k_ckpt}") # Optional debug


        # Map to sequential index if matched
        if is_backbone_key and potential_orig_mod_name is not None:
            try:
                target_seq_idx = map_orig_name_to_sequential_idx[potential_orig_mod_name]
                # Construct the key for the nn.Sequential backbone (e.g., '0.block1...')
                new_key = f"{target_seq_idx}.{key_suffix}" if key_suffix else str(target_seq_idx)
                backbone_state_dict[new_key] = v
                processed_keys.add(k_ckpt); mapped = True
                # logger.debug(f"Mapped BB key: '{k_ckpt}' -> '{new_key}'") # Optional debug
            except KeyError: logger.warning(f"Logic error mapping BB key: Mod name '{potential_orig_mod_name}' not in map for key '{k_ckpt}'.")
            except Exception as map_err: logger.warning(f"Error mapping BB key '{k_ckpt}': {map_err}")

        # --- Try mapping to Head (if it exists) ---
        if not mapped and segmentation_head is not None:
             # Prefixes expected in checkpoint keys for the head part (relative to stripped key)
             # Check the unused keys log: '_original_segmentation_head.head...', 'segmentation_head.head...'
             head_prefixes_ckpt = [
                 'segmentation_head.head.',
                 '_original_segmentation_head.head.',
                 'final.', # Add other possibilities from DeepWV3Plus structure
                 'classifier.',
                 'aspp_head.',
                 # Handle cases where head wasn't wrapped? Unlikely with SegHeadWrapper
                 'head.' # If checkpoint saved head directly
             ]
             target_head_prefix = "head." # Prefix inside SegHeadWrapper corresponding to self.head

             for prefix in head_prefixes_ckpt:
                  if current_key_part.startswith(prefix):
                      rest_of_key = current_key_part[len(prefix):]
                      # Map to the structure inside SegHeadWrapper ( self.head.<rest_of_key> )
                      mapped_head_key = f"{target_head_prefix}{rest_of_key}"
                      head_state_dict[mapped_head_key] = v
                      processed_keys.add(k_ckpt); mapped = True
                      # logger.debug(f"Mapped Head key: '{k_ckpt}' -> '{mapped_head_key}' (via prefix '{prefix}')") # Optional debug
                      break

        # --- Collect PEBAL / Hopfield keys ---
        if not mapped:
             # Use the key *after* initial stripping ('module.') for PEBAL matching
             pebal_prefixes = ['energy_head.', 'memory_input_proj.', 'memory_manager.', 'final_seghead_proj.',
                               'feature_adapter.', 'pebal_head.', '_memory_module.', '_pebal_module.',
                               'adapter.', 'hopfield_memory.', 'memory_readout.', 'memory_scorer.',
                               'efficient_memory.', 'memory_bank.', 'pebal_module.', 'seg_adapter.', 'ood_head.' ]
             if any(current_key_part.startswith(p) for p in pebal_prefixes):
                 # Keep the relative PEBAL key name
                 pebal_state_dict[current_key_part] = v
                 processed_keys.add(k_ckpt)
                 mapped = True
                 # logger.debug(f"Collected PEBAL/Hopfield key: '{k_ckpt}' -> '{current_key_part}'") # Optional debug

    # --- Load filtered state dicts into components ---
    # Load into backbone FIRST
    if backbone_state_dict:
        logger.info(f"Loading {len(backbone_state_dict)} keys into backbone...")
        try:
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            if missing: logger.warning(f" Backbone MISSING keys after mapping: {missing}") # Should be fewer now
            if unexpected: logger.error(f" Backbone UNEXPECTED keys after mapping: {unexpected}. Indicates mapping error or structure mismatch.")
            logger.info("Backbone weights loaded successfully (onto CPU).")
        except RuntimeError as e:
             logger.critical(f"CRITICAL: Runtime error loading backbone state dict. Likely architecture mismatch.", exc_info=True)
             raise e
    else:
        # This is now the expected critical warning if mapping fails
        logger.warning("!!! WARNING: No checkpoint keys were mapped to the backbone structure. Backbone weights might be missing or random. !!!")

    # Load into head SECOND
    if segmentation_head is not None and head_state_dict:
        logger.info(f"Loading {len(head_state_dict)} keys into segmentation head...")
        try:
            missing, unexpected = segmentation_head.load_state_dict(head_state_dict, strict=False)
            if missing: logger.warning(f" SegHead MISSING keys after mapping: {missing}")
            if unexpected: logger.warning(f" SegHead UNEXPECTED keys after mapping: {unexpected}")
            logger.info("Segmentation head weights loaded successfully (onto CPU).")
        except RuntimeError as e:
            logger.error(f"Runtime error loading segmentation head state dict.", exc_info=True)
            # Allow continuing, but head weights might be wrong
    elif segmentation_head is not None:
        logger.warning("No checkpoint keys mapped to the segmentation head structure.")

    # --- Move components to target device ---
    backbone = backbone.to(device)
    if segmentation_head: segmentation_head = segmentation_head.to(device)
    logger.info(f"Base model components moved to {device}")

    # --- Determine Actual Backbone Output Dimension for PEBAL Input ---
    effective_target_dim_for_pebal_input = args.target_feature_dim # Default
    if args.insertion_point == 'after_backbone':
        try:
            backbone.eval()
            # Use smaller dummy input if memory is tight
            dummy_h = min(args.img_height, 128)
            dummy_w = min(args.img_width, 256)
            dummy_input = torch.zeros(1, 3, dummy_h, dummy_w, device=device)
            with torch.no_grad(): dummy_output = backbone(dummy_input)
            actual_backbone_output_dim = dummy_output.shape[1]
            effective_target_dim_for_pebal_input = actual_backbone_output_dim # Use actual backbone output dim
            logger.info(f"Determined backbone output dim: {actual_backbone_output_dim}. Using this for PEBAL input.")
            del dummy_input, dummy_output
            if device.type == 'cuda': torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"Failed to determine backbone output dimension: {e}") from e

    # --- Instantiate HopfieldPEBALModel ---
    if not HOPFIELD_MODEL_IMPORTED:
        raise ImportError("HopfieldPEBALModel class failed to import.")

    logger.info("Instantiating HopfieldPEBALModel...")
    # Pass the *already loaded* backbone and head
    try:
        model = HopfieldPEBALModel(
            backbone=backbone,                  # Pass the loaded backbone
            segmentation_head=segmentation_head, # Pass the loaded head
            num_classes=args.num_classes,
            memory_feature_dim=args.memory_feature_dim,
            memory_size=args.memory_size,
            insertion_point=args.insertion_point,
            target_feature_dim=effective_target_dim_for_pebal_input, # Use the determined dim for adapter logic
            use_efficient_memory=True,
            use_faiss=(not args.disable_faiss),
            memory_log_interval=100000, # Reduce logging
            memory_log_verbose=False,    # Disable verbose memory logging
            use_efficient_decoder=args.use_efficient_decoder,
            efficient_decoder_kwargs={'attention_heads': args.attention_heads} if args.use_efficient_decoder else None,
            memory_beta=args.memory_beta,
        ).to(device)
        logger.info("HopfieldPEBALModel instantiated successfully.")

    except Exception as e:
        logger.error(f"Error creating HopfieldPEBALModel: {e}", exc_info=True)
        raise

    # --- Load PEBAL-specific weights into the *combined* model ---
    # This should now only contain actual PEBAL keys
    if pebal_state_dict:
        logger.info(f"Loading {len(pebal_state_dict)} PEBAL/Hopfield keys into combined model...")
        # strict=False is important here because the combined model also contains the backbone/head,
        # but pebal_state_dict should *only* have keys for the PEBAL parts.
        missing, unexpected = model.load_state_dict(pebal_state_dict, strict=False)

        # Report missing/unexpected keys *relative to the PEBAL dict*
        # Missing keys here means the model has PEBAL components whose keys weren't in the dict
        if missing: logger.warning(f" HopfieldPEBALModel MISSING PEBAL keys: {missing}")
        # Unexpected keys here means the pebal_state_dict contained keys not found in the model's PEBAL parts
        if unexpected: logger.warning(f" HopfieldPEBALModel UNEXPECTED PEBAL keys: {unexpected}")
        logger.info("PEBAL/Hopfield weights loaded.")
    else:
        logger.warning("No PEBAL/Hopfield-specific keys found/collected from the checkpoint.")

    # --- Final Check for Unused Checkpoint Keys ---
    unused_keys = set(full_state_dict.keys()) - processed_keys
    if unused_keys:
        logger.warning(f"{len(unused_keys)} checkpoint keys were COMPLETELY UNUSED.")
        log_limit = 20; unused_list = sorted(list(unused_keys))
        logger.warning(f" First {min(log_limit, len(unused_list))} unused keys: {unused_list[:log_limit]}")
        # Check if loading failed critically
        if len(backbone_state_dict) == 0:
             logger.error("CRITICAL CHECK: No backbone keys were loaded despite mapping logic. Checkpoint keys may not match expected structure ('backbone.modX...' or 'modX...').")
        if len(head_state_dict) == 0 and segmentation_head is not None:
             logger.warning("No segmentation head keys were loaded. Checkpoint keys may not match expected structure ('segmentation_head.head...' etc.).")
        # Save unused keys list for reference
        try:
            output_dir_path = Path(args.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            unused_path = output_dir_path / "_UNUSED_CHECKPOINT_KEYS.txt"
            with open(unused_path, 'w') as f: f.write(f"{len(unused_keys)} unused keys:\n" + "\n".join(unused_list))
        except Exception as write_err: logger.warning(f"Could not write unused keys file: {write_err}")
    else:
        logger.info("All keys from the checkpoint were processed and loaded into corresponding modules.")

    return model

# --- Evaluation Metrics Functions ---
def evaluate_segmentation(predictions: np.ndarray, targets: np.ndarray, num_classes: int, void_id: int = 255) -> float:
    """Calculate mean IoU for segmentation predictions"""
    try:
        predictions = predictions.flatten()
        targets = targets.flatten()
        valid_mask = (targets != void_id)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if predictions.size == 0: return 0.0

        predictions = np.clip(predictions, 0, num_classes-1)
        targets = np.clip(targets, 0, num_classes-1) # Targets should already be valid

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(conf_matrix, (targets.astype(np.int32), predictions.astype(np.int32)), 1)

        intersection = np.diag(conf_matrix)
        ground_truth_set = conf_matrix.sum(axis=1)
        predicted_set = conf_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        iou = np.zeros(num_classes, dtype=np.float32)
        valid_union = union > 0
        iou[valid_union] = intersection[valid_union] / union[valid_union].astype(np.float32)

        valid_iou_mask = ground_truth_set > 0
        mean_iou = np.mean(iou[valid_iou_mask]) if np.any(valid_iou_mask) else 0.0
        return mean_iou if not np.isnan(mean_iou) else 0.0
    except Exception as e:
        logger.error(f"Error in evaluate_segmentation: {e}", exc_info=True)
        return 0.0

def evaluate_ood_detection(energy_maps: np.ndarray, targets: np.ndarray, anomaly_id: int,
                           void_id: int = 255, return_scores: bool = False) -> Any:
    """Calculate OOD detection metrics (AUROC, AUPRC, FPR@95TPR)"""
    flat_energy, binary_targets = np.array([]), np.array([])
    num_ood, num_inlier = 0, 0

    try:
        flat_energy = energy_maps.flatten()
        flat_targets = targets.flatten()
        valid_mask = (flat_targets != void_id)

        if not np.any(valid_mask):
            # Return neutral defaults if no valid pixels
            return (0.5, 0.0, 1.0, flat_energy, binary_targets) if return_scores else (0.5, 0.0, 1.0)

        flat_energy = flat_energy[valid_mask]
        binary_targets = (flat_targets[valid_mask] == anomaly_id).astype(int)

        num_ood = np.sum(binary_targets == 1)
        num_inlier = np.sum(binary_targets == 0)

        # Handle cases with only one class present
        if num_ood == 0:
            logger.debug(f"No OOD pixels (ID {anomaly_id}) found in this sample/batch.") # Changed to debug
            return (0.5, 0.0, 1.0, flat_energy, binary_targets) if return_scores else (0.5, 0.0, 1.0)
        if num_inlier == 0:
            logger.debug(f"No Inlier pixels found in this sample/batch (all OOD).") # Changed to debug
            return (0.5, 1.0, 0.0, flat_energy, binary_targets) if return_scores else (0.5, 1.0, 0.0)

        # Handle non-finite energy values robustly
        finite_mask = np.isfinite(flat_energy)
        if not np.all(finite_mask):
            num_non_finite = np.sum(~finite_mask)
            logger.warning(f"{num_non_finite} non-finite energy values found. Replacing with median.")
            median_finite = np.median(flat_energy[finite_mask]) if np.any(finite_mask) else 0.0
            # Use np.nan_to_num for robust replacement
            flat_energy = np.nan_to_num(flat_energy, nan=median_finite, posinf=median_finite, neginf=median_finite)


        # Calculate metrics
        auroc = roc_auc_score(binary_targets, flat_energy)
        auprc = average_precision_score(binary_targets, flat_energy)

        # Calculate FPR@95TPR
        fpr_roc, tpr_roc, _ = roc_curve(binary_targets, flat_energy)
        target_tpr = 0.95
        fpr95 = 1.0 # Default to worst case
        if np.any(tpr_roc >= target_tpr):
             valid_indices = np.where(tpr_roc >= target_tpr)[0]
             if len(valid_indices) > 0: fpr95 = fpr_roc[valid_indices[0]]
        else: logger.debug(f"Target TPR ({target_tpr}) never reached. Max TPR: {np.max(tpr_roc):.4f}. FPR@95TPR set to 1.0.") # Changed to debug

        return (auroc, auprc, fpr95, flat_energy, binary_targets) if return_scores else (auroc, auprc, fpr95)

    except ValueError as e:
        logger.error(f"ValueError calculating OOD metrics: {e}.", exc_info=True)
        ood_prop = float(num_ood)/(num_ood + num_inlier) if (num_ood + num_inlier) > 0 else 0.0
        return (0.5, ood_prop, 1.0, flat_energy, binary_targets) if return_scores else (0.5, ood_prop, 1.0)
    except Exception as e:
        logger.error(f"Unexpected error calculating OOD metrics: {e}", exc_info=True)
        ood_prop = float(num_ood)/(num_ood + num_inlier) if (num_ood + num_inlier) > 0 else 0.0
        return (0.5, ood_prop, 1.0, flat_energy, binary_targets) if return_scores else (0.5, ood_prop, 1.0)

# --- Visualization ---
def visualize_results(image: np.ndarray, target: np.ndarray, prediction: np.ndarray,
                     energy: np.ndarray, output_path: str, num_classes: int,
                     anomaly_id: int, void_id: int = 255):
    """Visualize and save results as a figure with 4 subplots"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        base_filename = Path(output_path).stem
        fig.suptitle(f"Sample: {base_filename}", fontsize=16)

        # Image (denormalize)
        img_display = image.transpose(1, 2, 0) if image.shape[0]==3 and image.ndim==3 else image
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img_display = np.clip(std * img_display + mean, 0, 1)
        axes[0,0].imshow(img_display); axes[0,0].set_title('Original Image'); axes[0,0].axis('off')

        # Ground Truth Colormap
        cmap_gt = plt.get_cmap('tab20', num_classes + 2)
        colors_gt = cmap_gt(np.arange(num_classes + 2))
        anomaly_color = np.array([1.0, 0.0, 0.0, 1.0]); void_color = np.array([0.0, 0.0, 0.0, 1.0])
        target_int = target.astype(int)
        tgt_colored = np.zeros((*target.shape, 4), dtype=np.float32)
        for i in range(num_classes): tgt_colored[target_int == i] = colors_gt[i]
        tgt_colored[target_int == anomaly_id] = anomaly_color
        tgt_colored[target_int == void_id] = void_color
        axes[0,1].imshow(tgt_colored); axes[0,1].set_title(f'Ground Truth (Anomaly={anomaly_id})'); axes[0,1].axis('off')

        # Prediction Colormap
        cmap_pred = plt.get_cmap('tab20', num_classes)
        pred_colors = cmap_pred(np.arange(num_classes))
        pred_colored = np.zeros((*prediction.shape, 4), dtype=np.float32)
        pred_clipped = np.clip(prediction.astype(int), 0, num_classes - 1)
        for i in range(num_classes): pred_colored[pred_clipped == i] = pred_colors[i]
        axes[1,0].imshow(pred_colored); axes[1,0].set_title('Prediction (Inlier Classes)'); axes[1,0].axis('off')

        # Energy Map
        energy_finite = energy[np.isfinite(energy)]
        energy_min = np.min(energy_finite) if energy_finite.size > 0 else 0
        energy_max = np.max(energy_finite) if energy_finite.size > 0 else 1
        # Use percentiles for better contrast if range is huge
        # energy_min = np.percentile(energy_finite, 1) if energy_finite.size > 0 else 0
        # energy_max = np.percentile(energy_finite, 99) if energy_finite.size > 0 else 1
        energy_display = np.nan_to_num(energy, nan=energy_min, posinf=energy_max, neginf=energy_min)
        im = axes[1,1].imshow(energy_display, cmap='viridis', vmin=energy_min, vmax=energy_max)
        axes[1,1].set_title(f'OOD Energy (Min: {energy_min:.2f}, Max: {energy_max:.2f})')
        axes[1,1].axis('off')
        plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory
    except Exception as e:
        logger.error(f"Visualization error for {output_path}: {e}", exc_info=False)

# --- Dataset Checking ---
def check_dataset_files(path1: Optional[str], path2: Optional[str], dataset_name: str) -> bool:
    """Verify dataset directories exist and contain some files."""
    logger.info(f"Checking '{dataset_name}' dataset paths: Img='{path1}', Lbl='{path2}'")
    path1_obj = Path(path1) if path1 else None
    path2_obj = Path(path2) if path2 else None
    paths_ok = True
    for path_obj, name in [(path1_obj, 'Images'), (path2_obj, 'Labels')]:
        if not path_obj: logger.error(f"  {name} path not provided."); paths_ok = False; continue
        if not path_obj.is_dir(): logger.error(f"  {name} path is not a directory: {path_obj}"); paths_ok = False; continue
        try:
            if not any(p.is_file() for p in path_obj.rglob('*')):
                logger.warning(f"  {name} directory exists but appears empty: {path_obj}") # Warn but allow proceeding
        except OSError as e: logger.error(f"  Cannot access {name} directory {path_obj}: {e}"); paths_ok = False
    if not paths_ok: logger.error(f"Dataset file check FAILED for '{dataset_name}'.")
    return paths_ok

# --- Dataset Evaluation Function ---
def evaluate_on_dataset(args, model, dataset_name, device):
    """Evaluate model on a specific dataset."""
    logger.info(f"===== Evaluating on {dataset_name} dataset =====")
    image_path, label_path = None, None
    dataset_class_to_use : Optional[type] = None
    dataset_kwargs = {}
    is_ood_dataset = False

    output_dir_ds = Path(args.output_dir) / f"{dataset_name}_results"
    try: output_dir_ds.mkdir(parents=True, exist_ok=True)
    except OSError as e: logger.error(f"Failed to create output directory {output_dir_ds}: {e}"); return None

    if not DATASETS_IMPORTED:
         logger.error(f"Cannot evaluate {dataset_name}: Dataset classes failed to import."); return None

    # Configure paths and dataset class
    ds_configs = {
        'inlier': (args.test_images, args.test_labels, SegmentationDataset, False,
                   {'image_suffix': '_leftImg8bit.png', 'mask_suffix': '_gtFine_labelIds.png'}),
        'lostandfound': (args.lostandfound_images, args.lostandfound_labels, FishyscapesDataset, True,
                         {'dataset_type': 'LostAndFound', 'image_suffix': '.png', 'mask_suffix': '.png'}),
        'static': (args.static_images, args.static_labels, FishyscapesDataset, True,
                   {'dataset_type': 'Static', 'image_suffix': '.png', 'mask_suffix': '.png'}),
        'road_anomaly': (args.road_anomaly_images, args.road_anomaly_labels, FishyscapesDataset, True,
                         {'dataset_type': 'RoadAnomaly', 'image_suffix': '.png', 'mask_suffix': '.png'})
    }

    if dataset_name in ds_configs:
        image_path, label_path, dataset_class_to_use, is_ood_dataset, specific_kwargs = ds_configs[dataset_name]
        dataset_kwargs = {'image_dir': image_path, 'mask_dir': label_path, **specific_kwargs}
    else:
        logger.error(f"Unknown dataset name provided: {dataset_name}"); return None

    # Check dataset paths if requested
    if args.check_files_exist and not check_dataset_files(image_path, label_path, dataset_name):
        logger.error(f"Dataset check failed for {dataset_name}. Skipping."); return None

    # Define transforms
    eval_img_size = (args.img_height, args.img_width)
    transform = transforms.Compose([
        transforms.Resize(eval_img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def pil_to_long_tensor(img: Image.Image) -> torch.Tensor:
        try:
             # Ensure image mode is suitable before converting
             if img.mode not in ['L', 'I', 'P', 'I;16']:
                 logger.warning(f"Mask image mode is '{img.mode}', attempting conversion to 'L'.")
                 img = img.convert('L')
             return torch.from_numpy(np.array(img, dtype=np.int64))
        except Exception as e:
            logger.error(f"Error converting mask PIL to Long Tensor: {e}. Mask might be invalid.", exc_info=False)
            return torch.full(eval_img_size, args.void_id, dtype=torch.long)

    mask_transform = transforms.Compose([
        transforms.Resize(eval_img_size, interpolation=InterpolationMode.NEAREST), # MUST be NEAREST
        transforms.Lambda(pil_to_long_tensor)
    ])

    # Create Dataset and DataLoader
    if dataset_class_to_use is None: logger.error(f"Dataset class not assigned for {dataset_name}."); return None
    try:
        # Update base kwargs with common args and filter based on class signature
        dataset_kwargs.update({'transform': transform, 'mask_transform': mask_transform, 'img_height': args.img_height, 'img_width': args.img_width, 'void_id': args.void_id})

        # Use inspect (imported globally) to filter kwargs
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

        if len(dataset) == 0: raise ValueError(f"Dataset '{dataset_name}' initialized but found 0 samples. Check paths and suffixes.")
        logger.info(f"Created {dataset_name} dataset ({len(dataset)} samples).")

        is_cuda = device.type == 'cuda' and not args.force_cpu
        persistent_workers = (args.num_workers > 0 and sys.platform != "win32")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=is_cuda, drop_last=False, persistent_workers=persistent_workers,
                                 prefetch_factor=2 if args.num_workers > 0 else None)
    except Exception as e:
        logger.error(f"Failed to create dataset/loader for '{dataset_name}': {e}", exc_info=True)
        (output_dir_ds / f"_FAIL_DATALOADER_{type(e).__name__}.txt").touch(exist_ok=True)
        return None

    # --- Evaluation Loop ---
    model.eval()
    metrics = {}
    all_ood_scores_targets = []
    all_mious = []
    processed_samples = 0
    outputs_to_save = []

    with torch.no_grad():
        autocast_enabled = (device.type == 'cuda') and (not args.force_cpu)
        amp_dtype = torch.bfloat16 if (autocast_enabled and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16
        if autocast_enabled: logger.info(f"Using AMP autocast with dtype: {amp_dtype}")

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled, dtype=amp_dtype if autocast_enabled else None):
            pbar = tqdm(data_loader, desc=f"Evaluating {dataset_name}", leave=False, dynamic_ncols=True)
            for i, batch_data in enumerate(pbar):
                batch_stems = []
                try:
                    # Unpack Batch Data
                    if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 2:
                        logger.error(f"Batch {i}: Invalid data format. Skipping."); continue
                    images, masks = batch_data[0], batch_data[1]
                    if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
                        logger.error(f"Batch {i}: Expected Tensors. Skipping."); continue
                    # Get stems or generate fallbacks
                    batch_stems = list(batch_data[2]) if (len(batch_data) >= 3 and isinstance(batch_data[2], (list, tuple)) and len(batch_data[2]) == images.shape[0]) \
                                   else [f"{dataset_name}_batch{i}_idx{b}" for b in range(images.shape[0])]

                    images = images.to(device, non_blocking=True)

                    # Model Inference
                    outputs = model(images)
                    if not isinstance(outputs, dict): logger.error(f"Batch {i}: Model output not a dict. Skipping."); continue
                    logits = outputs.get('seg_logits')
                    energy = outputs.get('combined_energy') # Assuming this key holds the OOD score

                    if logits is None: logger.error(f"Batch {i}: Missing 'seg_logits' in output. Skipping."); continue
                    if energy is None:
                        logger.warning(f"Batch {i}: Missing 'combined_energy' in output. Using zeros for OOD.")
                        energy = torch.zeros((logits.shape[0], 1, *logits.shape[2:]), device=device, dtype=logits.dtype)

                    # Post-processing
                    target_size = (args.img_height, args.img_width)
                    if logits.shape[2:] != target_size: logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
                    if energy.ndim == 4 and energy.shape[1] == 1 and energy.shape[2:] != target_size:
                        energy = F.interpolate(energy, size=target_size, mode='bilinear', align_corners=False)
                    elif energy.ndim != 4 or energy.shape[1] != 1 or energy.shape[2:] != target_size:
                         # Handle cases where energy is not Bx1xHxW or wrong H/W
                         logger.warning(f"Batch {i}: Energy map shape {energy.shape} unexpected/unresizable. Using zeros.")
                         energy = torch.zeros((logits.shape[0], 1, *target_size), device=device, dtype=logits.dtype)


                    predictions_batch = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                    energy_batch = energy.squeeze(1).cpu().float().numpy() # BxHxW
                    masks_batch = masks.cpu().numpy()
                    images_batch_np = images.cpu().numpy() if (args.visualize or args.save_outputs) else None

                    # Process each sample
                    for b in range(images.shape[0]):
                        pred_map, mask_map, energy_map = predictions_batch[b], masks_batch[b], energy_batch[b]
                        image_np, current_stem = images_batch_np[b] if images_batch_np is not None else None, batch_stems[b]

                        sample_miou = evaluate_segmentation(pred_map, mask_map, args.num_classes, args.void_id)
                        all_mious.append(sample_miou)

                        if np.any(mask_map != args.void_id): # Only calculate OOD if non-void pixels exist
                            ood_result = evaluate_ood_detection(energy_map, mask_map, args.anomaly_id, args.void_id, return_scores=True)
                            if ood_result and len(ood_result) == 5 and ood_result[3] is not None and ood_result[3].size > 0:
                                all_ood_scores_targets.append((ood_result[3], ood_result[4])) # (scores_np, targets_np)

                        processed_samples += 1
                        sample_idx_global = (i * args.batch_size) + b

                        if args.save_outputs:
                            outputs_to_save.append({
                                'index': sample_idx_global, 'stem': current_stem,
                                'target': mask_map.astype(np.uint8), 'prediction': pred_map.astype(np.uint8),
                                'energy': energy_map.astype(np.float16) # Save space
                            })

                        if args.visualize and sample_idx_global < 10:
                            if image_np is None: logger.warning(f"Cannot visualize sample {sample_idx_global}, image data not kept.")
                            else:
                                vis_filename = output_dir_ds / f"vis_{Path(current_stem).stem}_{sample_idx_global:04d}.png"
                                visualize_results(image_np, mask_map, pred_map, energy_map, str(vis_filename),
                                                  args.num_classes, args.anomaly_id, args.void_id)

                except Exception as batch_err:
                    first_stem = batch_stems[0] if batch_stems else 'N/A'
                    logger.error(f"Error processing batch {i} (first stem: {first_stem}): {batch_err}", exc_info=True)
                    continue # Skip to next batch

                finally:
                    # Periodic memory cleanup
                    if i > 0 and i % 100 == 0:
                        gc.collect()
                        if device.type == 'cuda': torch.cuda.empty_cache()

    # --- Calculate Final Metrics ---
    if processed_samples == 0:
        logger.error(f"No samples processed successfully for {dataset_name}."); return None

    valid_mious = [m for m in all_mious if not np.isnan(m)]
    final_miou = np.mean(valid_mious) if valid_mious else 0.0
    metrics['mIoU'] = final_miou
    logger.info(f"{dataset_name} Final Mean Sample mIoU: {final_miou:.6f}")

    if all_ood_scores_targets:
        logger.info(f"Calculating global OOD metrics from {len(all_ood_scores_targets)} samples...")
        try:
            global_energies = np.concatenate([s[0] for s in all_ood_scores_targets if isinstance(s[0], np.ndarray)])
            global_targets = np.concatenate([s[1] for s in all_ood_scores_targets if isinstance(s[1], np.ndarray)])
            if global_energies.size == 0: raise ValueError("Concatenated OOD scores/targets empty.")

            num_global_ood = np.sum(global_targets == 1); num_global_inlier = np.sum(global_targets == 0)
            logger.info(f"Total pixels for global OOD: {len(global_energies)} (OOD: {num_global_ood}, Inlier: {num_global_inlier})")

            # Only proceed if both classes are present globally
            if num_global_ood > 0 and num_global_inlier > 0:
                global_auroc, global_auprc, global_fpr95 = evaluate_ood_detection(
                    global_energies, global_targets, anomaly_id=1, void_id=-1, return_scores=False # Use binary targets
                )
                metrics['AUROC'] = global_auroc; metrics['AUPRC'] = global_auprc; metrics['FPR@95TPR'] = global_fpr95
                logger.info(f"{dataset_name} Global OOD - AUROC: {global_auroc:.6f}, AUPRC: {global_auprc:.6f}, FPR@95TPR: {global_fpr95:.6f}")
            else:
                logger.warning(f"Cannot calculate global OOD metrics for {dataset_name}: only one class present globally (OOD={num_global_ood}, Inlier={num_global_inlier}).")
                metrics['AUROC'], metrics['AUPRC'], metrics['FPR@95TPR'] = 0.5, float(num_global_ood > 0), float(num_global_inlier == 0) # Assign trivial scores

        except Exception as e:
            logger.error(f"Failed to calculate global OOD metrics for {dataset_name}: {e}", exc_info=True)
            metrics['AUROC'], metrics['AUPRC'], metrics['FPR@95TPR'] = 0.5, 0.0, 1.0
    else:
        logger.warning(f"No valid OOD scores collected for {dataset_name}. Cannot calculate global OOD metrics.")
        metrics['AUROC'], metrics['AUPRC'], metrics['FPR@95TPR'] = 0.5, 0.0, 1.0

    # --- Save Results ---
    try:
        metrics_npy_path = output_dir_ds / "metrics.npy"; np.save(metrics_npy_path, metrics)
        metrics_txt_path = output_dir_ds / "metrics.txt"
        with open(metrics_txt_path, 'w') as f:
            f.write(f"Metrics for {dataset_name}:\n" + "="*20 + "\n")
            for k, v in metrics.items(): f.write(f"  {k}: {v:.6f}\n")
        logger.info(f"Metrics saved to {output_dir_ds}")
    except Exception as e: logger.error(f"Failed to save metrics files: {e}")

    if args.save_outputs and outputs_to_save:
        outputs_path = output_dir_ds / "detailed_outputs.npz"
        logger.info(f"Saving detailed outputs for {len(outputs_to_save)} samples to {outputs_path}...")
        save_dict = {}
        try:
            keys_to_stack = ['target', 'prediction', 'energy']; keys_regular = ['index', 'stem']
            for k in keys_regular:
                 if outputs_to_save and k in outputs_to_save[0]:
                     save_dict[k] = np.array([item[k] for item in outputs_to_save])
            for k in keys_to_stack:
                 if outputs_to_save and k in outputs_to_save[0]:
                     arrays_to_stack = [item[k] for item in outputs_to_save]
                     if all(isinstance(arr, np.ndarray) for arr in arrays_to_stack):
                         save_dict[k] = np.stack(arrays_to_stack, axis=0)
                     else:
                          logger.warning(f"Cannot stack key '{k}', saving as object array.")
                          save_dict[k] = np.array(arrays_to_stack, dtype=object)

            if save_dict:
                np.savez_compressed(outputs_path, **save_dict)
                logger.info(f"Detailed outputs saved ({outputs_path.stat().st_size/1e6:.2f} MB)")
            else:
                logger.warning("No data prepared for saving in detailed_outputs.npz")

        except Exception as e: logger.error(f"Failed to save detailed outputs NPZ: {e}", exc_info=True)

    logger.info(f"===== Evaluation finished for {dataset_name} =====")
    return metrics

# --- Main Evaluation Function ---
def evaluate(args):
    # Set logging level based on whether user wants debug info (implicitly via existence of --debug flag, though it's removed now)
    # Retaining this logic to allow DEBUG level logging if needed, without an explicit flag
    log_level = logging.DEBUG if '--debug' in sys.argv else logging.INFO # Check original argv
    logging.getLogger().setLevel(log_level) # Set root logger level
    logger.setLevel(log_level) # Set specific logger level

    logger.info("Starting Hopfield-PEBAL Evaluation Script (Version 5.2.1)...")
    logger.info(f"Output directory: {args.output_dir}")

    # Check critical imports early
    if not check_critical_imports(args):
        logger.critical("Critical imports failed. Cannot continue.")
        sys.exit(1)

    # --- Device Setup ---
    if args.force_cpu or not torch.cuda.is_available(): device = torch.device("cpu")
    else: device = torch.device("cuda"); torch.backends.cudnn.benchmark = (args.batch_size > 1)
    logger.info(f"Using device: {device}")
    if device.type == 'cuda': logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Output Directory and Parameter Saving ---
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / "parameters.json"
        class PathEncoder(json.JSONEncoder):
            def default(self, obj): return str(obj) if isinstance(obj, Path) else super().default(obj)
        # Exclude the removed --debug flag if present in original args
        args_to_save = {k: v for k, v in vars(args).items() if k != 'debug'}
        with open(params_path, 'w') as f: json.dump(args_to_save, f, indent=4, cls=PathEncoder)
        logger.info(f"Run parameters saved to: {params_path}")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to create output dir or save params: {output_dir}. Error: {e}. Exiting.", exc_info=True)
        sys.exit(1)

    # --- Load Model ---
    model = None
    try:
        model = load_model(args, device)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model Loaded: {type(model).__name__} | Total Params: {total_params:,}")
    except FileNotFoundError as e: logger.critical(f"FAIL: Checkpoint not found: {e}"); sys.exit(1)
    except (RuntimeError, ValueError, ImportError, TypeError) as e:
        logger.critical(f"FAIL: Error during model loading: {type(e).__name__} - {e}", exc_info=True)
        fail_msg = "Likely ARCHITECTURE MISMATCH or key mapping error." if "mismatch" in str(e).lower() or "key" in str(e).lower() or "size" in str(e).lower() else ""
        logger.critical(fail_msg + " Check logs for details on missing/unexpected keys.")
        (output_dir / "_FAIL_MODEL_LOAD_ERROR.txt").touch(exist_ok=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FAIL: Unexpected error loading model: {e}", exc_info=True)
        (output_dir / "_FAIL_MODEL_LOAD_UNEXPECTED.txt").touch(exist_ok=True)
        sys.exit(1)


    # --- Determine Datasets to Evaluate ---
    datasets_to_evaluate = []
    ds_map = {
        'inlier': (args.test_images, args.test_labels), 'lostandfound': (args.lostandfound_images, args.lostandfound_labels),
        'static': (args.static_images, args.static_labels), 'road_anomaly': (args.road_anomaly_images, args.road_anomaly_labels)
    }
    selected_datasets = ds_map.keys() if args.dataset == 'all' else [args.dataset]

    for name in selected_datasets:
        if name in ds_map:
            paths = ds_map[name]
            # Check if paths are provided *and* exist
            if paths[0] and paths[1]:
                img_path = Path(paths[0])
                lbl_path = Path(paths[1])
                if img_path.exists() and lbl_path.exists():
                     datasets_to_evaluate.append(name)
                else:
                     logger.warning(f"Skipping '{name}': Paths do not exist (Img: {img_path.exists()}, Lbl: {lbl_path.exists()})")
            else:
                logger.warning(f"Skipping '{name}': Image or Label path not provided in arguments.")
        elif name != 'all': logger.warning(f"Requested dataset '{name}' not recognized.")

    logger.info(f"Datasets selected for evaluation: {datasets_to_evaluate}")
    if not datasets_to_evaluate:
        logger.error("No valid datasets selected or available based on provided paths. Exiting."); sys.exit(1)

    # --- Run Evaluation Loop ---
    all_metrics = {}
    evaluation_successful = False
    for dataset_name in datasets_to_evaluate:
        metrics = None
        dataset_output_dir = output_dir / f"{dataset_name}_results"
        try:
            metrics = evaluate_on_dataset(args, model, dataset_name, device)
            if metrics: all_metrics[dataset_name] = metrics; evaluation_successful = True
            else: logger.warning(f"Evaluation for {dataset_name} returned no metrics (skipped or failed).")
        except Exception as e:
            logger.error(f"FAIL: Unhandled exception during '{dataset_name}' evaluation: {e}", exc_info=True)
            try:
                dataset_output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists before writing fail file
                (dataset_output_dir / "_FAIL_UNHANDLED_EXCEPTION.txt").touch(exist_ok=True)
            except Exception: pass # Avoid crashing the logger itself
        finally:
            # Minimal cleanup log
            logger.info(f"Cleaning up after evaluating {dataset_name}...")
            del metrics; gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()

    # --- Save Combined Results ---
    if evaluation_successful and all_metrics:
        logger.info("Saving combined metrics...")
        combined_npy_path = output_dir / "all_metrics.npy"
        combined_txt_path = output_dir / "all_metrics.txt"
        try:
            np.save(combined_npy_path, all_metrics)
            with open(combined_txt_path, 'w') as f:
                f.write(f"Combined Evaluation Metrics ({Path(args.checkpoint).name})\n" + "=" * 25 + "\n")
                for ds, metrics_dict in all_metrics.items():
                    f.write(f"\n--- {ds} ---\n")
                    if metrics_dict:
                        for k, v in metrics_dict.items(): f.write(f"  {k}: {v:.6f}\n")
                    else: f.write("  (No valid metrics recorded)\n")
            logger.info(f"Combined metrics saved: {combined_txt_path}")
        except Exception as e: logger.error(f"Failed to save combined metrics: {e}")
    elif not evaluation_successful:
        logger.error("Evaluation finished, but NO datasets were processed successfully.")
        (output_dir / "_FAIL_NO_DATASETS_SUCCESSFUL.txt").touch(exist_ok=True)

    logger.info("===== Evaluation Script Finished =====")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
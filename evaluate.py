#!/usr/bin/env python
# evaluate.py (Version 3 - Dataset Arg Fix + Checkpoint Mapping Fix - V3 Load Model)

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, UnidentifiedImageError # Added from datasets.py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import sys
import importlib.util
import gc
from collections import OrderedDict
from pathlib import Path # Added from datasets.py
from typing import Callable, Dict, List, Optional, Tuple, Union # Added from datasets.py


# Import psutil (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logging.warning("psutil not available, CPU memory tracking will be limited")
    class DummyProcess:
        def memory_info(self):
            class MemInfo:
                rss = 0;
            return MemInfo() # type: ignore
    class DummyPsutil:
        def Process(self, *args, **kwargs):
            return DummyProcess()
    psutil = DummyPsutil()
    PSUTIL_AVAILABLE = False

# --- Import custom modules ---
try:
    # Ensure the path to hopfield_pebal_model.py is correct or in PYTHONPATH
    from hopfield_pebal_model import HopfieldPEBALModel, MemoryTracker
    HOPFIELD_MODEL_IMPORTED = True
except ImportError as e:
    logging.error(f"Failed to import HopfieldPEBALModel: {e}")
    HOPFIELD_MODEL_IMPORTED = False
    # Define placeholder class
    class HopfieldPEBALModel(nn.Module):
        def __init__(self, backbone, segmentation_head, **kwargs):
            super(HopfieldPEBALModel, self).__init__()
            self.backbone = backbone; self.segmentation_head = segmentation_head
            self.num_classes = kwargs.get('num_classes', 19); logging.warning("Using placeholder HopfieldPEBALModel")
            self.memory_tracker = MemoryTracker(verbose=False) if 'MemoryTracker' in globals() else None
        def forward(self, x):
            f = self.backbone(x); l = self.segmentation_head(f); e = torch.rand_like(x[:, :1])
            return {'seg_logits': l, 'combined_energy': e, 'memory_energy': e.clone(), 'feature_energy': e.clone(), 'pebal_energy': e.clone(), 'is_ood': torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)}
        def update_memory(self, *args, **kwargs): pass
        def eval(self): self.backbone.eval(); self.segmentation_head.eval()
        def train(self): self.backbone.train(); self.segmentation_head.train()
    class MemoryTracker:
        def __init__(self, *args, **kwargs): pass
        def log_memory_usage(self, *args, **kwargs): pass
        def clear_memory(self, *args, **kwargs): pass

# Import dataset classes
try:
    # Ensure the path to datasets module is correct or in PYTHONPATH
    from datasets.datasets import SegmentationDataset
    from datasets.fishyscapes_dataset import FishyscapesDataset
    DATASETS_IMPORTED = True
    logging.info("Successfully imported custom dataset classes.")
except ImportError as e:
    logging.error(f"Dataset import error: {e}")
    logging.warning("Could not import custom dataset classes. Using mock implementations.")
    DATASETS_IMPORTED = False
    # Define mock classes (if imports fail)
    class MockSegmentationDataset(torch.utils.data.Dataset):
         def __init__(self, image_dir, mask_dir, **kwargs): # Match expected args
             self.image_dir = Path(image_dir); self.mask_dir = Path(mask_dir); logging.debug("MockSegmentationDataset init")
             self.files = [{"image": self.image_dir / "img.png", "mask": self.mask_dir / "mask.png"}] # Dummy file
             self.transform = kwargs.get('transform')
             self.mask_transform = kwargs.get('mask_transform')
             # Handle suffixes for mock case
             self.image_suffix = kwargs.get('image_suffix', '.png')
             self.mask_suffix = kwargs.get('mask_suffix', '.png')
             if self.image_dir.is_dir() and self.mask_dir.is_dir():
                 # Basic mock file finding
                 image_files = {p.stem.replace(self.image_suffix.replace('.',''),''): p for p in self.image_dir.glob(f'*{self.image_suffix}')}
                 mask_files = {p.stem.replace(self.mask_suffix.replace('.',''),''): p for p in self.mask_dir.glob(f'*{self.mask_suffix}')}
                 common_stems = list(image_files.keys() & mask_files.keys())
                 self.files = [{"image": image_files[stem], "mask": mask_files[stem]} for stem in sorted(common_stems)]
                 if not self.files: self.files = [{"image": self.image_dir / "img.png", "mask": self.mask_dir / "mask.png"}] # Fallback dummy
             else:
                 logging.warning("Mock dataset dirs not found, using single dummy file.")

         def __len__(self): return len(self.files)
         def __getitem__(self, idx):
             # Create dummy data on the fly instead of reading files
             img_pil = Image.new('RGB', (512, 256)) # W, H
             mask_pil = Image.new('L', (512, 256), color=255) # W, H
             img_tensor = torch.zeros((3, 256, 512)) # C, H, W
             mask_tensor = torch.full((256, 512), 255, dtype=torch.long) # H, W

             if self.transform: img_tensor = self.transform(img_pil)
             if self.mask_transform: mask_tensor = self.mask_transform(mask_pil)

             return img_tensor, mask_tensor

    class MockFishyscapesDataset(MockSegmentationDataset): # Inherit and adjust if needed
        def __init__(self, image_dir, mask_dir, anomaly_id=19, **kwargs):
            # Assume Fishyscapes doesn't use standard Cityscapes suffixes
            kwargs.pop('image_suffix', None)
            kwargs.pop('mask_suffix', None)
            super().__init__(image_dir, mask_dir, **kwargs)
            self.anomaly_id = anomaly_id; logging.debug("MockFishyscapesDataset init")

    if not DATASETS_IMPORTED:
        SegmentationDataset = MockSegmentationDataset
        FishyscapesDataset = MockFishyscapesDataset
# --- End Imports ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hopfield-PEBAL-Evaluation")

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Hopfield-PEBAL model for OOD detection')
    # Dataset paths
    parser.add_argument('--test_images', type=str, default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/val', help='Path to INLIER test image directory')
    parser.add_argument('--test_labels', type=str, default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/val', help='Path to INLIER test label directory (expecting labelIds)')
    # Fishyscapes paths
    parser.add_argument('--lostandfound_images', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound/original', help='Path to LostAndFound image directory')
    parser.add_argument('--lostandfound_labels', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound/labels', help='Path to LostAndFound label directory')
    parser.add_argument('--static_images', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static/original', help='Path to Static image directory')
    parser.add_argument('--static_labels', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static/labels', help='Path to Static label directory')
    parser.add_argument('--road_anomaly_images', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly/original', help='Path to Road Anomaly image directory')
    parser.add_argument('--road_anomaly_labels', type=str, default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly/labels', help='Path to Road Anomaly label directory')
    # Evaluation dataset selection
    parser.add_argument('--dataset', type=str, default='all', choices=['inlier', 'lostandfound', 'static', 'road_anomaly', 'all'], help='Which dataset(s) to evaluate on')
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/latest_model.pth', help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of INLIER classes')
    parser.add_argument('--memory_feature_dim', type=int, default=256, help='Dimension of memory features')
    parser.add_argument('--memory_beta', type=float, default=8.0, help='Beta for memory energy')
    parser.add_argument('--memory_size', type=int, default=2000, help='Memory bank size')
    parser.add_argument('--attention_heads', type=int, default=4, help='Attention heads (for efficient decoder)')
    parser.add_argument('--insertion_point', type=str, default='after_backbone', choices=['after_backbone', 'after_seghead'], help='PEBAL insertion point')
    parser.add_argument('--target_feature_dim', type=int, default=304, help='Target dimension after adapter')
    parser.add_argument('--use_efficient_decoder', action='store_true', help='Use EfficientSegmentationDecoder')
    parser.add_argument('--disable_faiss', action='store_true', help='Disable FAISS')
    parser.add_argument('--base_model', type=str, default='deepwv3plus', choices=['deepwv3plus', 'simple'], help='Base segmentation model')
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--output_dir', type=str, default='./results/all', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize first few samples')
    parser.add_argument('--save_outputs', action='store_true', help='Save detailed outputs')
    parser.add_argument('--anomaly_id', type=int, default=19, help='Anomaly class ID in OOD datasets')
    parser.add_argument('--void_id', type=int, default=255, help='Void/ignore class ID in labels')
    # Debugging/Utility
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--check_files_exist', action='store_true', default=True, help='Check dataset files exist')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU')
    parser.add_argument('--img_height', type=int, default=256, help='Evaluation image height')
    parser.add_argument('--img_width', type=int, default=512, help='Evaluation image width')
    return parser.parse_args()

# --- Simple Model Creation ---
def create_simple_backbone_for_testing(num_classes=19):
    class SimpleBackbone(nn.Module):
        def __init__(self): super().__init__(); self.conv1=nn.Conv2d(3,64,7,2,3,bias=False); self.bn1=nn.BatchNorm2d(64); self.relu=nn.ReLU(True); self.pool1=nn.MaxPool2d(3,2,1); self.conv2=nn.Conv2d(64,128,3,1,1,bias=False); self.bn2=nn.BatchNorm2d(128); self.conv3=nn.Conv2d(128,256,3,2,1,bias=False); self.bn3=nn.BatchNorm2d(256)
        def forward(self, x): x=self.relu(self.bn1(self.conv1(x))); x=self.pool1(x); x=self.relu(self.bn2(self.conv2(x))); x=self.relu(self.bn3(self.conv3(x))); return x
    class SimpleSegHead(nn.Module):
        def __init__(self,i,n): super().__init__(); self.head=nn.Sequential(nn.Conv2d(i,128,3,1,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, n + 1, 1)); self._in_channels=i # Output n+1 for background/void? Check model usage. Usually just n. Let's assume n.
        def forward(self, x): return self.head(x)
    logger.info("Creating simple backbone and head for testing."); b=SimpleBackbone(); out_dim=256
    try:
        # Use eval mode for deterministic output shape calculation
        b.eval()
        dummy_input=torch.zeros(1,3,args.img_height, args.img_width) # Use eval size
        out_dim=b(dummy_input).shape[1]
        logger.info(f"Simple backbone determined output dimension: {out_dim}")
    except Exception as e:
        logger.warning(f"Could not determine simple backbone output dimension automatically: {e}. Assuming 256.")
        out_dim = 256 # Fallback
    # Ensure head outputs correct number of classes (num_classes, not num_classes+1 usually)
    return b, SimpleSegHead(out_dim, num_classes) # Output num_classes logits


# --- DeepWV3Plus Import Function ---
def import_deepwv3plus(num_classes=19):
    """
    Imports and instantiates the DeepWV3Plus model from the specified path.
    It extracts the backbone and head portions for use in the PEBAL model.
    """
    code_dir = '/home/ha51dybi/hop-pebal/code' # Path to the directory containing the 'model' package
    if code_dir not in sys.path:
        sys.path.append(code_dir)
        logger.info(f"Added {code_dir} to Python path")

    try:
        # Ensure the path to wide_network.py is correct relative to code_dir
        from model.wide_network import DeepWV3Plus # Import the class definition
        logger.info("Imported DeepWV3Plus class")

        # Instantiate based *only* on the definition in wide_network.py
        fm = DeepWV3Plus(num_classes=num_classes)
        logger.info("Initialized DeepWV3Plus instance (using code definition)")

    except ImportError as e:
        logger.error(f"ImportError: Could not import DeepWV3Plus from {code_dir}. Check path and file existence. Error: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error initializing DeepWV3Plus instance: {e}", exc_info=True)
        return None, None

    # Extract backbone and head parts based on expected names in the *code definition*
    bb_parts = ['mod1', 'pool2', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7'] # Expected backbone parts
    head_part = 'final' # Expected head part name in the code definition
    bb_modules = []
    head_module = None
    skipped_modules = []

    for name, module in fm.named_children():
        if name in bb_parts:
            bb_modules.append(module)
        elif name == head_part:
            head_module = module
        else:
            skipped_modules.append(name)
            logger.debug(f"Skipping module part '{name}' during extraction (not in bb_parts or head_part).")

    # Optional: Log skipped modules if they were unexpected based on code definition
    expected_skipped = ['gaussian_smoothing', 'pool3', 'aspp', 'bot_fine', 'bot_aspp'] # From code provided
    unexpected_skipped = [m for m in skipped_modules if m not in expected_skipped]
    if unexpected_skipped:
         logger.warning(f"Ignored unexpected DeepWV3Plus modules during extraction: {unexpected_skipped}")

    if not bb_modules:
        raise ValueError("Failed to extract any backbone modules (mod1-7, pool2) from DeepWV3Plus instance.")

    if head_module is None:
        # Fallback: Try finding a 'classifier' attribute if 'final' wasn't found (less likely based on provided code)
        head_module = getattr(fm, 'classifier', None)
        if head_module: logger.warning("Using 'classifier' attribute as segmentation head (fallback).")
        else: raise ValueError("Failed to extract segmentation head module (expected 'final') from DeepWV3Plus instance.")

    # Create Sequential backbone and wrap the head
    backbone = nn.Sequential(*bb_modules)
    logger.info(f"Extracted backbone (contains {len(list(backbone.children()))} modules based on code definition)")

    class SegHeadWrapper(nn.Module):
        def __init__(self, head_nn):
            super().__init__()
            self.head = head_nn
            self._in_channels = None
            # Try to infer input channels from the first conv layer in the head sequence
            first_conv = None
            if isinstance(head_nn, nn.Sequential):
                 for layer in head_nn:
                     if isinstance(layer, nn.Conv2d):
                         first_conv = layer
                         break
            elif isinstance(head_nn, nn.Conv2d): # If head is just a single conv layer
                 first_conv = head_nn

            if first_conv:
                self._in_channels = first_conv.in_channels
                logger.info(f"Segmentation head wrapper created. Inferred input channels: {self._in_channels}")
            else:
                 logger.warning("Could not infer input channels for segmentation head wrapper.")


        def forward(self, x):
            return self.head(x)

    segmentation_head = SegHeadWrapper(head_module)
    logger.info("Extracted segmentation head wrapper.")

    return backbone, segmentation_head


# --- **REVISED V3** Model Loading Function (Handles different key naming conventions) ---
def load_model(args, device):
    """Load base model, load checkpoint into components with refined key mapping,
       then wrap in HopfieldPEBALModel and load its specific weights."""
    logger.info(f"Loading base model '{args.base_model}'...")
    # Step 1: Create base model structure from code definition
    backbone, segmentation_head = (create_simple_backbone_for_testing(args.num_classes) if args.base_model == 'simple'
                                   else import_deepwv3plus(args.num_classes))
    if backbone is None or segmentation_head is None:
        raise RuntimeError("Base model loading failed. Check import_deepwv3plus logs.")

    # Load checkpoint to CPU first
    full_state_dict = None
    pebal_state_dict = OrderedDict() # To store PEBAL-specific keys
    if os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint file: {args.checkpoint}")
        try:
            # Set weights_only=True if you are sure the checkpoint doesn't contain malicious code via pickle
            # Set weights_only=False if the checkpoint might contain custom classes/functions needed for loading
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                key = next((k for k in ['state_dict', 'model_state_dict', 'model', 'net'] if k in checkpoint), None)
                full_state_dict = checkpoint[key] if key else checkpoint
                logger.info(f"Using state dict from checkpoint key: '{key if key else 'root'}'")
            else:
                full_state_dict = checkpoint
                logger.info("Using loaded object directly as state dict.")
        except Exception as e:
            logger.error(f"Error loading checkpoint file: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}") from e
    else:
        logger.error(f"Checkpoint file not found: {args.checkpoint}.") # Changed to error as it's likely required
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Filter state dict for backbone and original head, separate PEBAL keys
    backbone_state_dict = OrderedDict()
    head_state_dict = OrderedDict()
    processed_keys = set() # Keep track of keys assigned to BB, Head, or PEBAL

    if full_state_dict:
        # Define mapping from original DeepWV3Plus module names (in code) to Sequential indices
        # From import_deepwv3plus: ['mod1', 'pool2', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7']
        # Indices:                  0       1        2       3       4       5       6       7
        original_bb_names = ['mod1', 'pool2', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7']
        num_extracted_bb = len(list(backbone.children())) if isinstance(backbone, nn.Sequential) else 0
        logger.info(f"Extracted {num_extracted_bb} backbone modules, expected names: {original_bb_names[:num_extracted_bb]}")

        # --- Key Mapping Logic ---
        logger.info("Mapping checkpoint keys...")
        for k_ckpt, v in full_state_dict.items():
            mapped = False

            # --- Try mapping to Backbone ---
            # Handle 'backbone.features.X...' pattern (observed in logs)
            if k_ckpt.startswith('backbone.features.'):
                parts = k_ckpt.split('.')
                # Ensure format is backbone.features.<idx>.<rest>
                if len(parts) > 3 and parts[2].isdigit():
                    feature_idx = int(parts[2])
                    # Map feature index to module name/index (needs verification based on WiderResNetA2 structure)
                    # This mapping ASSUMES 'features.X' corresponds directly to the module index
                    # This might be INCORRECT if WiderResNetA2 nests things differently or includes pooling inside features.
                    # Example: features.0 -> mod1 (idx 0), features.1->pool2 (idx 1), features.2->mod2 (idx 2) ???
                    # --> Let's map purely based on index for now and refine if needed
                    # map_feature_to_original_idx = {
                    #     0: 0, # features.0 -> mod1 (index 0)
                    #     # 1: 1, # IS features.1 pool2? Or is pool2 separate? ASSUME pool2 is separate.
                    #     2: 2, # features.2 -> mod2 (index 2)
                    #     3: 3, # features.3 -> mod3 (index 3)
                    #     4: 4, # features.4 -> mod4 (index 4)
                    #     5: 5, # features.5 -> mod5 (index 5)
                    #     6: 6, # features.6 -> mod6 (index 6)
                    #     7: 7  # features.7 -> mod7 (index 7)
                    # }
                    # SAFER Assumption: 'backbone.features.X' corresponds to module 'X' in the ORIGINAL WiderResNet structure
                    # We need to map this 'X' to the index in our *extracted* sequential backbone.
                    # Let's try matching the module *name* if possible.
                    # If the checkpoint uses 'mod1', 'mod2' etc directly, that's easier.

                    # --> Revised approach: Try matching expected prefixes first.
                    matched_prefix = False
                    for idx, name in enumerate(original_bb_names):
                         prefix_in_ckpt = f"backbone.{name}." # e.g., backbone.mod1.
                         if k_ckpt.startswith(prefix_in_ckpt) and idx < num_extracted_bb:
                             new_key = f"{idx}{k_ckpt[len(prefix_in_ckpt)-1:]}" # e.g., 0.conv...
                             backbone_state_dict[new_key] = v
                             processed_keys.add(k_ckpt)
                             mapped = True
                             matched_prefix = True
                             logger.debug(f"Mapped BB key (prefix): {k_ckpt} -> {new_key}")
                             break
                    if matched_prefix: continue # Already mapped

                    # --> Fallback: Try the 'backbone.features.X' pattern
                    # THIS MAPPING IS HIGHLY SUSPECT AND DEPENDS ON WiderResNetA2 internals matching the extracted sequence.
                    # If `backbone` is Sequential(mod1, pool2, mod2, ...), then:
                    # feature_idx 0 might map to original_idx 0 (mod1)
                    # feature_idx 2 might map to original_idx 2 (mod2)
                    # feature_idx 3 might map to original_idx 3 (mod3) etc.
                    map_feature_to_original_idx = { 0: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7 } # Tentative!

                    if feature_idx in map_feature_to_original_idx:
                        original_idx = map_feature_to_original_idx[feature_idx]
                        if original_idx < num_extracted_bb:
                            # Construct new key: 'original_idx.rest.of.key'
                            rest_of_key = '.'.join(parts[3:])
                            new_key = f"{original_idx}.{rest_of_key}"
                            if new_key not in backbone_state_dict: # Avoid overwrite
                                backbone_state_dict[new_key] = v
                                processed_keys.add(k_ckpt)
                                mapped = True
                                logger.debug(f"Mapped BB key (features fallback): {k_ckpt} -> {new_key}")
                        else:
                             logger.debug(f"Skipping BB key {k_ckpt}: Mapped index {original_idx} >= num extracted {num_extracted_bb}")
                    else:
                        logger.debug(f"Skipping BB key {k_ckpt}: Feature index {feature_idx} not in explicit features map.")

            # --- Try mapping to original Segmentation Head ---
            # Handle '_original_segmentation_head.classifier...' pattern (observed in logs)
            head_prefix = '_original_segmentation_head.classifier.'
            if not mapped and k_ckpt.startswith(head_prefix):
                 # Map to 'head.rest.of.key'
                 rest_of_key = k_ckpt[len(head_prefix):]
                 new_key = f"head.{rest_of_key}"
                 head_state_dict[new_key] = v
                 processed_keys.add(k_ckpt)
                 mapped = True
                 logger.debug(f"Mapped Head key: {k_ckpt} -> {new_key}")

            # --- Fallback: Try direct 'final.' or 'classifier.' prefixes for head ---
            if not mapped:
                for prefix in ['final.', 'classifier.']:
                    if k_ckpt.startswith(prefix):
                         new_key = f"head.{k_ckpt[len(prefix):]}"
                         if new_key not in head_state_dict: # Avoid overwrite
                             head_state_dict[new_key] = v
                             processed_keys.add(k_ckpt)
                             mapped = True
                             logger.debug(f"Mapped Head key (fallback): {k_ckpt} -> {new_key}")
                             break # Stop after first match

            # --- Collect PEBAL-specific keys ---
            # If a key wasn't mapped to backbone or head, check if it belongs to PEBAL components
            if not mapped:
                 pebal_prefixes = ['energy_head.', 'memory_input_proj.', 'memory_manager.',
                                   'final_seghead_proj.', 'feature_adapter.', 'pebal_head.',
                                   'memory_projector.', 'query_projector.', 'key_projector.', # Add others if needed
                                   '_memory_module.', '_pebal_module.'] # Check HopfieldPEBALModel attributes
                 if any(k_ckpt.startswith(p) for p in pebal_prefixes):
                     pebal_state_dict[k_ckpt] = v
                     processed_keys.add(k_ckpt)
                     logger.debug(f"Collected PEBAL key: {k_ckpt}")
                 # else: # Log keys that weren't mapped and aren't recognized PEBAL keys?
                 #     logger.debug(f"Key {k_ckpt} not mapped to BB/Head and not recognized PEBAL prefix.")
                 #     pass

        # --- Load filtered state dicts into base components ---
        if backbone_state_dict:
            logger.info(f"Loading {len(backbone_state_dict)} keys into backbone structure...")
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            if missing: logger.warning(f" Backbone MISSING keys (expected by code but not found in mapped ckpt keys): {missing}")
            if unexpected: logger.warning(f" Backbone UNEXPECTED keys (in mapped ckpt keys but not in code structure): {unexpected}")
        else:
            logger.warning("No checkpoint keys were successfully mapped to the backbone structure.") # This was hit in previous log

        if head_state_dict:
            logger.info(f"Loading {len(head_state_dict)} keys into segmentation head structure...")
            missing, unexpected = segmentation_head.load_state_dict(head_state_dict, strict=False)
            if missing: logger.warning(f" SegHead MISSING keys: {missing}")
            if unexpected: logger.warning(f" SegHead UNEXPECTED keys: {unexpected}")
        else:
            logger.warning("No checkpoint keys were successfully mapped to the segmentation head structure.")

    else:
        # This case should not happen now due to FileNotFoundError raise above
        logger.critical("No checkpoint state_dict loaded. Cannot proceed.")
        raise ValueError("full_state_dict is None, checkpoint loading must have failed.")


    # --- Move components to device ---
    backbone = backbone.to(device)
    segmentation_head = segmentation_head.to(device)
    logger.info(f"Base model components moved to {device}")

    # --- Instantiate HopfieldPEBALModel ---
    if not HOPFIELD_MODEL_IMPORTED:
        logger.critical("HopfieldPEBALModel not imported. Cannot create final model.")
        raise ImportError("HopfieldPEBALModel class is required but failed to import.")

    logger.info("Instantiating HopfieldPEBALModel...")
    try:
        model = HopfieldPEBALModel(
            backbone=backbone, # May or may not have loaded weights
            segmentation_head=segmentation_head, # May or may not have loaded weights
            num_classes=args.num_classes, memory_feature_dim=args.memory_feature_dim,
            memory_size=args.memory_size, insertion_point=args.insertion_point,
            target_feature_dim=args.target_feature_dim, use_efficient_memory=True, # Assuming default
            use_faiss=(not args.disable_faiss), memory_log_interval=30, # Assuming default
            memory_log_verbose=args.debug,
            use_efficient_decoder=args.use_efficient_decoder,
            efficient_decoder_kwargs={'attention_heads': args.attention_heads} if args.use_efficient_decoder else None,
            memory_beta=args.memory_beta
        ).to(device)
        logger.info("HopfieldPEBALModel instantiated successfully.")
    except Exception as e:
        logger.error(f"Error creating HopfieldPEBALModel: {e}", exc_info=True); raise

    # --- Load PEBAL-specific weights into the final model ---
    if pebal_state_dict:
        logger.info(f"Loading {len(pebal_state_dict)} PEBAL-specific keys into HopfieldPEBALModel...")
        # Load into the *overall* model object
        missing, unexpected = model.load_state_dict(pebal_state_dict, strict=False)
        # Report missing/unexpected keys relative to the PEBAL keys we extracted
        # Missing means PEBAL components expected a key that wasn't in the checkpoint's PEBAL keys
        if missing: logger.warning(f" HopfieldPEBALModel MISSING PEBAL keys (expected but not in ckpt): {missing}")
        # Unexpected means the checkpoint had PEBAL keys that the current HopfieldPEBALModel doesn't have attributes for
        if unexpected: logger.warning(f" HopfieldPEBALModel UNEXPECTED PEBAL keys (in ckpt but not in model): {unexpected}")
    else:
        logger.warning("No PEBAL-specific keys found/collected from the checkpoint to load.")

    # --- Final Check for Unused Keys ---
    if full_state_dict:
        unused_keys = set(full_state_dict.keys()) - processed_keys
        if unused_keys:
            logger.warning(f"Checkpoint keys COMPLETELY UNUSED (not mapped to BB, Head, or PEBAL): {sorted(list(unused_keys))}")
        else:
            logger.info("All keys from the checkpoint were processed (mapped to BB, Head, or PEBAL).")

    return model


# --- Evaluation Metrics Functions ---
def evaluate_segmentation(predictions: np.ndarray, targets: np.ndarray, num_classes: int, void_id: int = 255):
    """Calculates mean Intersection over Union (mIoU) for semantic segmentation."""
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Filter out void labels
    valid_mask = (targets != void_id)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    # Handle case where no valid pixels exist
    if predictions.size == 0:
        logger.debug("evaluate_segmentation: No valid pixels after filtering void.")
        return 0.0

    # Ensure predictions are within the valid class range [0, num_classes-1]
    # This is important if the model predicts values outside this range.
    predictions = np.clip(predictions, 0, num_classes - 1)

    # Create confusion matrix only for valid target classes [0, num_classes-1]
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid_target_mask = (targets >= 0) & (targets < num_classes)
    valid_predictions_for_conf = predictions[valid_target_mask]
    valid_targets_for_conf = targets[valid_target_mask]

    # Use np.add.at for efficient accumulation
    np.add.at(conf_matrix, (valid_targets_for_conf, valid_predictions_for_conf), 1)

    # Calculate IoU per class
    intersection = np.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(axis=1)
    predicted_set = conf_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    # Avoid division by zero
    iou = intersection / (union.astype(np.float32) + 1e-8)

    # Calculate mIoU only over classes present in the ground truth
    valid_iou_mask = ground_truth_set > 0
    if np.any(valid_iou_mask):
        mean_iou = np.mean(iou[valid_iou_mask])
    else:
        logger.debug("evaluate_segmentation: No ground truth pixels found for any valid class.")
        mean_iou = 0.0

    return mean_iou if not np.isnan(mean_iou) else 0.0

def evaluate_ood_detection(energy_maps: np.ndarray, targets: np.ndarray, anomaly_id: int, void_id: int = 255, return_scores: bool = False):
    """Calculates OOD detection metrics (AUROC, AUPRC, FPR@95TPR)."""
    flat_energy = energy_maps.flatten()
    flat_targets = targets.flatten()

    # Filter out void labels
    valid_mask = (flat_targets != void_id)
    if not np.any(valid_mask):
         logger.debug("evaluate_ood_detection: No valid pixels found after filtering void.")
         # Return dummy values and empty arrays if requested
         return (0.5, 0.0, 1.0, np.array([]), np.array([])) if return_scores else (0.5, 0.0, 1.0)

    flat_energy = flat_energy[valid_mask]
    # Create binary labels: 1 for anomaly, 0 for inlier
    binary_targets = (flat_targets[valid_mask] == anomaly_id).astype(int)

    num_ood = np.sum(binary_targets == 1)
    num_inlier = np.sum(binary_targets == 0)

    # Handle edge cases where only one class is present after filtering
    if num_ood == 0:
        logger.debug(f"evaluate_ood_detection: No OOD pixels (ID {anomaly_id}) found after filtering void.")
        # AUROC is undefined (or 0.5 by convention), AUPRC is baseline (0), FPR95 is 1.0
        return (0.5, 0.0, 1.0, flat_energy, binary_targets) if return_scores else (0.5, 0.0, 1.0)
    if num_inlier == 0:
        logger.debug("evaluate_ood_detection: No Inlier pixels found after filtering void.")
        # AUROC is undefined (or 0.5 by convention), AUPRC is 1.0, FPR95 is 0.0
        return (0.5, 1.0, 0.0, flat_energy, binary_targets) if return_scores else (0.5, 1.0, 0.0)

    try:
        # Calculate metrics
        auroc = roc_auc_score(binary_targets, flat_energy)
        auprc = average_precision_score(binary_targets, flat_energy)

        # Calculate FPR at 95% TPR
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(binary_targets, flat_energy)
        target_tpr = 0.95
        fpr95 = 1.0 # Default value if 95% TPR is not reached

        if np.max(tpr_roc) >= target_tpr:
            # Find the first index where TPR >= target_tpr
            # roc_curve sorts thresholds decreasingly, so TPR/FPR increase with index
            valid_indices = np.where(tpr_roc >= target_tpr)[0]
            fpr95 = fpr_roc[valid_indices[0]] # Get FPR at the first threshold meeting the TPR condition
        else:
             logger.debug(f"Max TPR ({np.max(tpr_roc):.4f}) is less than target {target_tpr}. Setting FPR@95TPR to 1.0")

        return (auroc, auprc, fpr95, flat_energy, binary_targets) if return_scores else (auroc, auprc, fpr95)

    except ValueError as e:
         # Can happen if only one class present, though handled above, but good to catch.
         logger.error(f"ValueError during OOD metrics calculation (Likely only one class present?): {e}", exc_info=False)
         # Return default/fallback values consistent with edge cases
         ood_proportion = float(num_ood) / (num_ood + num_inlier) if (num_ood + num_inlier) > 0 else 0.0
         return (0.5, ood_proportion, 1.0, flat_energy, binary_targets) if return_scores else (0.5, ood_proportion, 1.0)
    except Exception as e:
        logger.error(f"Unexpected error calculating OOD metrics: {e}", exc_info=True)
        # Return safe defaults
        ood_proportion = float(num_ood) / (num_ood + num_inlier) if (num_ood + num_inlier) > 0 else 0.0
        return (0.5, ood_proportion, 1.0, flat_energy, binary_targets) if return_scores else (0.5, ood_proportion, 1.0)


# --- Visualization ---
def visualize_results(image: np.ndarray, target: np.ndarray, prediction: np.ndarray, energy: np.ndarray, output_path: str, num_classes: int, anomaly_id: int):
    """Saves a visualization comparing image, GT, prediction, and energy."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Sample: {os.path.basename(output_path).replace('visualization_','').replace('.png','')}", fontsize=16)

        # De-normalize image for display (assuming CHW input)
        img_display = image.transpose(1, 2, 0) if image.shape[0]==3 and image.ndim==3 else image # CHW -> HWC if needed
        mean=np.array([0.485, 0.456, 0.406]); std=np.array([0.229, 0.224, 0.225])
        img_display = np.clip(std * img_display + mean, 0, 1) # De-normalize

        axes[0,0].imshow(img_display); axes[0,0].set_title('Original Image'); axes[0,0].axis('off')

        # Create color maps
        # Use a known colormap like 'tab20' which handles up to 20 distinct classes well
        # Add extra colors for anomaly and void
        cmap_gt = plt.get_cmap('tab20', num_classes + 2)
        colors = cmap_gt(np.arange(num_classes + 2))
        # Define specific colors (optional customization)
        anomaly_color = colors[num_classes]  # Use the (N+1)-th color for anomaly
        void_color = np.array([0, 0, 0, 1])      # Black for void/ignore

        # Color Ground Truth
        tgt_colored = np.zeros((*target.shape, 4), dtype=np.float32) # Use RGBA
        for i in range(num_classes):
            tgt_colored[target == i] = colors[i]
        tgt_colored[target == anomaly_id] = anomaly_color
        tgt_colored[target == 255] = void_color # Assuming void_id is 255
        axes[0,1].imshow(tgt_colored); axes[0,1].set_title(f'Ground Truth (Anomaly ID: {anomaly_id})'); axes[0,1].axis('off')

        # Color Prediction (only for inlier classes 0..N-1)
        cmap_pred = plt.get_cmap('tab20', num_classes)
        pred_colors = cmap_pred(np.arange(num_classes))
        pred_colored = np.zeros((*prediction.shape, 4), dtype=np.float32) # Use RGBA
        # Clip predictions to be within 0..N-1 before coloring
        pred_clipped = np.clip(prediction, 0, num_classes - 1)
        for i in range(num_classes):
            pred_colored[pred_clipped == i] = pred_colors[i]
        axes[1,0].imshow(pred_colored); axes[1,0].set_title('Predicted Segmentation (Inlier Classes)'); axes[1,0].axis('off')

        # Energy map
        # Choose a sequential colormap like 'viridis' or 'plasma'
        energy_min, energy_max = np.min(energy), np.max(energy)
        im=axes[1,1].imshow(energy, cmap='viridis', vmin=energy_min, vmax=energy_max)
        axes[1,1].set_title(f'OOD Energy Score (Min: {energy_min:.2f}, Max: {energy_max:.2f})')
        axes[1,1].axis('off')
        plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04) # Add colorbar

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory
        logger.debug(f"Visualization saved: {output_path}")
    except Exception as e:
        logger.error(f"Visualization error for {output_path}: {e}", exc_info=False)


# --- Dataset Checking ---
def check_dataset_files(path1, path2, dataset_name):
    """Checks if dataset image and label directories exist and are not empty."""
    logger.info(f"Checking '{dataset_name}' dataset paths:")
    logger.info(f"  Images dir: {path1}")
    logger.info(f"  Labels dir: {path2}")
    paths_ok = True
    for p, name in [(path1, 'Images'), (path2, 'Labels')]:
        if not p: # Check if path is None or empty string
            logger.error(f"{name} path not provided for {dataset_name}.")
            paths_ok = False; continue
        path_obj = Path(p)
        if not path_obj.is_dir():
            logger.error(f"{name} path is not a valid directory: {p}")
            paths_ok = False; continue
        try:
            # Check if directory is readable and list first few items
            items = list(path_obj.iterdir())
            if not items:
                logger.warning(f"{name} directory exists but appears empty: {p}")
            else:
                 limit = 5
                 files = [f.name for f in items if f.is_file()][:limit]
                 dirs = [f.name for f in items if f.is_dir()][:limit]
                 logger.info(f"  Found ~{len(items)} items in {name} dir.")
                 if files: logger.info(f"    Files (up to {limit}): {files}")
                 if dirs: logger.info(f"    Subdirs (up to {limit}): {dirs}")

        except OSError as e:
            logger.error(f"Cannot access or list {name} directory {p}: {e}")
            paths_ok = False

    if not paths_ok:
        logger.error(f"Dataset file check failed for '{dataset_name}'. Evaluation might fail.")
    else:
        logger.info(f"Dataset file check preliminary passed for '{dataset_name}'.")
    return paths_ok


# --- Dataset Evaluation Function ---
def evaluate_on_dataset(args, model, dataset_name, device):
    """Evaluate model on a specific dataset."""
    logger.info(f"===== Evaluating on {dataset_name} dataset =====")

    # --- Determine Dataset Paths and Class ---
    is_ood_dataset = False
    dataset_class_to_use = None
    dataset_kwargs = {} # Arguments for the dataset constructor
    check_path1 = None
    check_path2 = None

    # Select dataset parameters based on name
    if dataset_name == 'inlier':
        check_path1 = args.test_images
        check_path2 = args.test_labels
        output_dir = os.path.join(args.output_dir, 'inlier_results')
        dataset_class_to_use = SegmentationDataset
        dataset_kwargs = {'image_dir': args.test_images, 'mask_dir': args.test_labels}
        # Add Cityscapes suffixes if SegmentationDataset uses them
        dataset_kwargs['image_suffix'] = '_leftImg8bit.png'
        dataset_kwargs['mask_suffix'] = '_gtFine_labelIds.png'

    elif dataset_name == 'lostandfound':
        check_path1 = args.lostandfound_images
        check_path2 = args.lostandfound_labels
        output_dir = os.path.join(args.output_dir, 'lostandfound_results')
        is_ood_dataset = True
        dataset_class_to_use = FishyscapesDataset # Assuming this class exists and handles paths/suffixes
        dataset_kwargs = {'image_dir': args.lostandfound_images, 'mask_dir': args.lostandfound_labels, 'dataset_type': 'LostAndFound'} # Pass type if needed by class

    elif dataset_name == 'static':
        check_path1 = args.static_images
        check_path2 = args.static_labels
        output_dir = os.path.join(args.output_dir, 'static_results')
        is_ood_dataset = True
        dataset_class_to_use = FishyscapesDataset
        dataset_kwargs = {'image_dir': args.static_images, 'mask_dir': args.static_labels, 'dataset_type': 'Static'} # Pass type if needed

    elif dataset_name == 'road_anomaly':
        check_path1 = args.road_anomaly_images
        check_path2 = args.road_anomaly_labels
        output_dir = os.path.join(args.output_dir, 'road_anomaly_results')
        is_ood_dataset = True
        dataset_class_to_use = FishyscapesDataset
        dataset_kwargs = {'image_dir': args.road_anomaly_images, 'mask_dir': args.road_anomaly_labels, 'dataset_type': 'RoadAnomaly'} # Pass type if needed
    else:
        logger.error(f"Unknown dataset name provided: {dataset_name}"); return None

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results directory: {output_dir}")

    # --- Check Dataset Files ---
    if args.check_files_exist:
        if not check_dataset_files(check_path1, check_path2, dataset_name):
            logger.error(f"Dataset check failed for {dataset_name}. Skipping this dataset.")
            with open(os.path.join(output_dir, "_SKIPPED_DATASET_CHECK_FAILED.txt"), 'w') as f:
                f.write(f"Skipped {dataset_name} due to failed dataset check.")
            return None

    # --- Set up Transforms ---
    eval_img_size = (args.img_height, args.img_width) # H, W
    logger.info(f"Evaluation image size (H, W): {eval_img_size}")
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize(eval_img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
        ])
    # Mask transforms - resize only, then convert to LongTensor
    def pil_to_long_tensor(img):
        return torch.from_numpy(np.array(img, dtype=np.int64)) # Ensure Long type

    mask_transform = transforms.Compose([
        transforms.Resize(eval_img_size, interpolation=InterpolationMode.NEAREST), # Use NEAREST for labels
        transforms.Lambda(pil_to_long_tensor)
        ])

    # --- Create Dataset and DataLoader ---
    try:
        # Add common args needed by both dataset types
        dataset_kwargs['transform'] = transform
        dataset_kwargs['mask_transform'] = mask_transform
        # Pass num_classes only if the dataset class expects it
        import inspect
        sig = inspect.signature(dataset_class_to_use.__init__)
        if 'num_classes' in sig.parameters:
            dataset_kwargs['num_classes'] = args.num_classes
        if is_ood_dataset:
            # Add anomaly_id only if expected by FishyscapesDataset
            if 'anomaly_id' in sig.parameters:
                 dataset_kwargs['anomaly_id'] = args.anomaly_id

        logger.debug(f"Instantiating {dataset_name} ({dataset_class_to_use.__name__}) with args: {list(dataset_kwargs.keys())}")
        dataset = dataset_class_to_use(**dataset_kwargs)

        if len(dataset) == 0:
            logger.error(f"Dataset '{dataset_name}' initialized but resulted in 0 samples. Check paths, suffixes, and filters.")
            with open(os.path.join(output_dir, "_FAILED_DATASET_EMPTY.txt"), 'w') as f:
                f.write(f"Failed {dataset_name}: Dataset empty after initialization.")
            return None
        logger.info(f"Created {dataset_name} dataset ({len(dataset)} samples).")

    except TypeError as e:
        logger.error(f"TypeError creating dataset '{dataset_name}': {e}")
        logger.error(f"Please check the constructor arguments required by {dataset_class_to_use.__name__}.")
        logger.error(f"Arguments provided were: {list(dataset_kwargs.keys())}")
        with open(os.path.join(output_dir, "_FAILED_DATASET_INIT_TYPE_ERROR.txt"), 'w') as f:
            f.write(f"Failed {dataset_name} due to TypeError: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating dataset '{dataset_name}': {e}", exc_info=True)
        with open(os.path.join(output_dir, "_FAILED_DATASET_INIT_ERROR.txt"), 'w') as f:
            f.write(f"Failed {dataset_name} due to Error: {e}")
        return None

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), # pin_memory only beneficial for CUDA
        drop_last=False)

    # --- Evaluation Loop ---
    model.eval() # Set model to evaluation mode
    metrics = {}
    all_ood_scores = [] # List to store (flat_energy, binary_target) tuples or arrays
    total_miou = 0.0
    processed_samples = 0
    outputs_to_save = []
    mem_tracker = getattr(model, 'memory_tracker', MemoryTracker(verbose=False)) # Use model's tracker if available

    # Get process info for memory tracking
    process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else psutil.Process()

    with torch.no_grad(): # Disable gradient calculations for inference
        for i, batch_data in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
            try:
                start_cpu_mem = process.memory_info().rss / (1024 * 1024) # MB
                start_gpu_mem = torch.cuda.memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0 # MB

                if not (isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2):
                    logger.error(f"Batch {i}: Invalid data format from DataLoader. Expected tuple/list of length >= 2. Got: {type(batch_data)}")
                    continue
                # Assume first element is image, second is mask
                images, masks = batch_data[0], batch_data[1]

                # Move data to the evaluation device
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True) # Keep masks on device for potential slicing/masking

                # --- Model Forward Pass ---
                # Use autocast for potential speedup with mixed precision, if supported and desired
                # with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                outputs = model(images)

                # --- Extract Outputs ---
                # Ensure keys exist and have expected shape
                logits = outputs.get('seg_logits')
                energy = outputs.get('combined_energy') # Use 'combined_energy' as the OOD score

                if logits is None: logger.error(f"Batch {i}: 'seg_logits' missing from model output."); continue
                if energy is None: logger.warning(f"Batch {i}: 'combined_energy' missing from model output. OOD metrics will be skipped."); # Allow continuing for mIoU
                if logits.ndim != 4 or logits.shape[0] != images.shape[0] or logits.shape[2:] != images.shape[2:]:
                    logger.error(f"Batch {i}: Invalid logits shape. Got {logits.shape}, expected [B, C, H, W] matching input H, W."); continue
                if energy is not None and (energy.ndim != 4 or energy.shape[0] != images.shape[0] or energy.shape[1] != 1 or energy.shape[2:] != images.shape[2:]):
                    logger.error(f"Batch {i}: Invalid energy shape. Got {energy.shape}, expected [B, 1, H, W] matching input H, W."); energy = None # Invalidate bad energy

                # --- Process Batch Results ---
                # Move results needed for numpy/CPU processing
                predictions_batch = torch.argmax(logits, dim=1).cpu().numpy()
                energy_batch = energy.cpu().numpy() if energy is not None else np.zeros((images.shape[0], 1, *images.shape[2:])) # Dummy energy if missing
                masks_batch = masks.cpu().numpy()
                images_batch_np = images.cpu().numpy() # For saving/visualization

                for b in range(images.shape[0]): # Iterate through samples in the batch
                    pred_map = predictions_batch[b] # H, W
                    mask_map = masks_batch[b]       # H, W
                    energy_map = energy_batch[b, 0] # H, W (squeeze channel dim)
                    image_np = images_batch_np[b]   # C, H, W

                    # Calculate Segmentation mIoU for this sample
                    # Note: mIoU calculated per sample and averaged later might differ slightly from global confusion matrix mIoU
                    batch_miou = evaluate_segmentation(pred_map, mask_map, args.num_classes, args.void_id)
                    total_miou += batch_miou

                    # Collect OOD scores and targets if energy is valid
                    if energy is not None:
                        ood_result = evaluate_ood_detection(energy_map, mask_map, args.anomaly_id, args.void_id, return_scores=True)
                        if ood_result is not None and ood_result[3].size > 0: # Check if valid scores were returned
                           all_ood_scores.append((ood_result[3], ood_result[4])) # Store (flat_energies, flat_binary_targets)

                    processed_samples += 1
                    sample_index_global = i * args.batch_size + b

                    # Save detailed outputs if requested
                    if args.save_outputs:
                         # Save smaller data types if possible
                         outputs_to_save.append({
                            'index': sample_index_global,
                            'target': mask_map.astype(np.uint8),      # H, W uint8
                            'prediction': pred_map.astype(np.uint8),  # H, W uint8
                            'energy': energy_map.astype(np.float16) # H, W float16 (check range/precision)
                         })

                    # Visualize if requested (only first few samples)
                    if args.visualize and sample_index_global < 10:
                        vis_path = os.path.join(output_dir, f"visualization_{sample_index_global:04d}.png")
                        visualize_results(image_np, mask_map, pred_map, energy_map, vis_path, args.num_classes, args.anomaly_id)

                # --- Memory Logging and Cleanup ---
                end_cpu_mem = process.memory_info().rss / (1024 * 1024)
                end_gpu_mem = torch.cuda.memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0
                logger.debug(f"Batch {i}: CPU Mem {start_cpu_mem:.1f}->{end_cpu_mem:.1f} MB | GPU Mem {start_gpu_mem:.1f}->{end_gpu_mem:.1f} MB")
                mem_tracker.log_memory_usage(f"Batch {i} End") # Use internal tracker if available

                # Optional: Force garbage collection periodically
                if i > 0 and i % 50 == 0:
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()
                    logger.debug(f"Batch {i}: Forced garbage collection and CUDA cache clear.")

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}", exc_info=True)
                mem_tracker.clear_memory(f"Batch {i} Error Cleanup") # Use internal tracker if available
                if device.type == 'cuda': torch.cuda.empty_cache() # Clear cache on error too
                continue # Skip to the next batch

    # --- Aggregate and Report Metrics ---
    if processed_samples == 0:
        logger.error(f"No samples were successfully processed for {dataset_name}. Check logs for batch errors.")
        with open(os.path.join(output_dir, "_FAILED_NO_SAMPLES_PROCESSED.txt"), 'w') as f:
            f.write(f"Failed {dataset_name}: No samples processed successfully.")
        return None

    # Final mIoU
    final_miou = total_miou / processed_samples
    metrics['mIoU'] = final_miou
    logger.info(f"{dataset_name} Final Mean Sample mIoU: {final_miou:.6f}")

    # Final OOD Metrics (calculated globally)
    if all_ood_scores:
        logger.info("Calculating global OOD metrics from collected scores...")
        # Concatenate all energies and targets
        try:
            global_energies = np.concatenate([s[0] for s in all_ood_scores])
            global_targets = np.concatenate([s[1] for s in all_ood_scores])

            if global_energies.size == 0:
                logger.warning(f"Concatenated OOD scores are empty for {dataset_name}.")
                metrics['AUROC'] = 0.5; metrics['AUPRC'] = 0.0; metrics['FPR@95TPR'] = 1.0
            else:
                logger.info(f"Total pixels for OOD evaluation: {len(global_energies)}")
                # Use anomaly_id=1 because targets are already binary (0 or 1)
                global_auroc, global_auprc, global_fpr95 = evaluate_ood_detection(
                    global_energies, global_targets, anomaly_id=1, void_id=-1 # Use dummy void_id
                )
                metrics['AUROC'] = global_auroc
                metrics['AUPRC'] = global_auprc
                metrics['FPR@95TPR'] = global_fpr95
                logger.info(f"{dataset_name} Global OOD - AUROC: {global_auroc:.6f}, AUPRC: {global_auprc:.6f}, FPR@95TPR: {global_fpr95:.6f}")
        except Exception as e:
            logger.error(f"Error calculating global OOD metrics for {dataset_name}: {e}", exc_info=True)
            metrics['AUROC'] = 0.5; metrics['AUPRC'] = 0.0; metrics['FPR@95TPR'] = 1.0 # Fallback metrics
    else:
        logger.warning(f"No valid OOD scores were collected for {dataset_name} (or energy was missing). Skipping OOD metrics.")
        metrics['AUROC'] = 0.5; metrics['AUPRC'] = 0.0; metrics['FPR@95TPR'] = 1.0 # Indicate metrics weren't calculated


    # --- Save Results ---
    metrics_npy_path = os.path.join(output_dir, "metrics.npy")
    metrics_txt_path = os.path.join(output_dir, "metrics.txt")
    try:
        np.save(metrics_npy_path, metrics)
        logger.info(f"Metrics dictionary saved: {metrics_npy_path}")
        with open(metrics_txt_path, 'w') as f:
             f.write(f"Metrics for {dataset_name}:\n")
             f.write("=========================\n")
             for k, v in metrics.items():
                 f.write(f"  {k}: {v:.6f}\n")
        logger.info(f"Metrics text saved: {metrics_txt_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics files: {e}")

    if args.save_outputs and outputs_to_save:
        outputs_path = os.path.join(output_dir, "detailed_outputs.npz") # Use npz for potential compression
        try:
            # Convert list of dicts to a dict of lists/arrays for saving
            save_dict = {}
            if outputs_to_save:
                keys = outputs_to_save[0].keys()
                for key in keys:
                     # Stack arrays for 'target', 'prediction', 'energy'
                     if key in ['target', 'prediction', 'energy']:
                         save_dict[key] = np.stack([item[key] for item in outputs_to_save], axis=0)
                     else: # Keep 'index' as a list or simple array
                         save_dict[key] = np.array([item[key] for item in outputs_to_save])

            np.savez_compressed(outputs_path, **save_dict) # Save as compressed npz
            logger.info(f"Detailed outputs saved: {outputs_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed outputs: {e}", exc_info=True)

    logger.info(f"===== Evaluation finished for {dataset_name} =====")
    return metrics


# --- Main Evaluation Function ---
def evaluate(args):
    # --- Setup ---
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")
        # Set other loggers to debug?
        # logging.getLogger('hopfield_pebal_model').setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    logger.info("Starting Hopfield-PEBAL Evaluation Script...")
    logger.info(f"Script Arguments: {vars(args)}")

    # Setup device
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True # Can improve performance if input sizes don't vary much
            torch.backends.cudnn.deterministic = False
            logger.info(f"CUDA available. Using device: {device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU device.")

    # Create output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output Directory: {args.output_dir}")
        # Save parameters used for this run
        params_path=os.path.join(args.output_dir, "parameters.txt");
        with open(params_path,'w') as f:
            import json
            json.dump(vars(args), f, indent=4) # Save as JSON for easier parsing
        logger.info(f"Run parameters saved to: {params_path}")
    except Exception as e:
        logger.critical(f"Failed to create output directory or save parameters: {e}. Exiting.")
        return # Cannot proceed without output directory

    # --- Load Model ---
    try:
        model = load_model(args, device)
        model.eval() # Ensure model is in eval mode
        # Log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model Loaded: {type(model).__name__}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")

    except FileNotFoundError as e:
         logger.critical(f"FAIL: {e}") # Already logged in load_model
         with open(os.path.join(args.output_dir, "_FAILED_CHECKPOINT_NOT_FOUND.txt"), 'w') as f: f.write(str(e))
         return
    except (RuntimeError, ValueError, ImportError) as e: # Catch errors during loading/instantiation
        logger.critical(f"FAIL: Model loading/instantiation failed: {e}", exc_info=True)
        with open(os.path.join(args.output_dir, "_FAILED_MODEL_LOAD_ERROR.txt"), 'w') as f: f.write(str(e))
        return
    except Exception as e:
        logger.critical(f"FAIL: An unexpected error occurred during model loading: {e}", exc_info=True)
        with open(os.path.join(args.output_dir, "_FAILED_MODEL_LOAD_UNEXPECTED_ERROR.txt"), 'w') as f: f.write(str(e))
        return

    # --- Determine Datasets to Evaluate ---
    if args.dataset == 'all':
        datasets_to_evaluate = ['inlier', 'lostandfound', 'static', 'road_anomaly']
        # Filter based on provided paths
        datasets_to_evaluate = [d for d in datasets_to_evaluate if
                                (d == 'inlier' and args.test_images and args.test_labels) or
                                (d != 'inlier' and getattr(args, f"{d}_images", None) and getattr(args, f"{d}_labels", None))]
    else:
        datasets_to_evaluate = [args.dataset]
    logger.info(f"Datasets selected for evaluation: {datasets_to_evaluate}")

    # --- Run Evaluation Loop ---
    all_metrics = {}
    evaluation_successful = False # Track if at least one dataset finished

    for dataset_name in datasets_to_evaluate:
        metrics = None
        try:
            # Run evaluation for the current dataset
            metrics = evaluate_on_dataset(args, model, dataset_name, device)

            if metrics is not None: # Check if evaluation returned metrics
                all_metrics[dataset_name] = metrics
                evaluation_successful = True # Mark success if metrics were returned
            else:
                 logger.warning(f"Evaluation for {dataset_name} did not return metrics. Check logs for errors.")
                 # Optionally create a failure marker file
                 fail_path = os.path.join(args.output_dir, dataset_name + "_results", "_FAILED_EVAL_RETURNED_NONE.txt")
                 os.makedirs(os.path.dirname(fail_path), exist_ok=True)
                 with open(fail_path, 'w') as f: f.write("Evaluation function returned None.")


        except Exception as e:
            # Catch any unhandled exceptions during a dataset's evaluation
            logger.error(f"FAIL: Unhandled error during evaluation of '{dataset_name}': {e}", exc_info=True)
            # Create a failure marker file in that dataset's results directory
            fail_path = os.path.join(args.output_dir, dataset_name + "_results", "_FAILED_UNHANDLED_EXCEPTION.txt")
            os.makedirs(os.path.dirname(fail_path), exist_ok=True) # Ensure dir exists
            with open(fail_path, 'w') as f: f.write(f"Unhandled Exception: {e}\nSee main log for traceback.")

        finally:
            # Cleanup after evaluating each dataset
            del metrics # Explicitly delete metrics object
            gc.collect() # Run garbage collection
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
            logger.info(f"Cleaned up memory after evaluating {dataset_name}.")

    # --- Save Combined Results ---
    if evaluation_successful and all_metrics:
        logger.info("Saving combined metrics for all successful datasets...")
        combined_npy_path = os.path.join(args.output_dir, "all_metrics.npy")
        combined_txt_path = os.path.join(args.output_dir, "all_metrics.txt")
        try:
            np.save(combined_npy_path, all_metrics)
            logger.info(f"Combined metrics dictionary saved: {combined_npy_path}")

            with open(combined_txt_path, 'w') as f:
                f.write("Combined Evaluation Metrics\n")
                f.write("===========================\n")
                for dataset_name, metrics_dict in all_metrics.items():
                    f.write(f"\n--- {dataset_name} ---\n")
                    if metrics_dict:
                        for metric_name, metric_value in metrics_dict.items():
                            f.write(f"  {metric_name}: {metric_value:.6f}\n")
                    else:
                        f.write("  No metrics available (evaluation might have failed).\n")
            logger.info(f"Combined metrics text summary saved: {combined_txt_path}")

        except Exception as e:
            logger.error(f"Failed to save combined metrics files: {e}")

    elif not evaluation_successful:
        logger.error("Evaluation finished, but no datasets were processed successfully.")
        # Create a failure marker file in the main output directory
        with open(os.path.join(args.output_dir, "_FAILED_NO_DATASETS_SUCCESSFUL.txt"), 'w') as f:
            f.write("The evaluation script ran, but failed to produce results for any dataset.")
    else:
        # Case where evaluation_successful is True but all_metrics is empty (shouldn't happen with current logic)
        logger.warning("Evaluation marked successful, but combined metrics dictionary is empty.")


    logger.info("===== Evaluation Script Finished =====")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
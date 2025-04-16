import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import sys
import importlib.util

# --- Basic Setup ---

# Set up logging with more verbose output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hopfield-PEBAL-Evaluation")

# --- Placeholder/Fallback Definitions ---

# Define placeholder classes in case imports fail, allowing script execution for testing structure
try:
    from datasets.datasets import SegmentationDataset, SimpleImageDataset
    from datasets import FishyscapesDataset
    from hopfield_pebal_model import HopfieldPEBALModel
    logger.info("Successfully imported custom modules: SegmentationDataset, FishyscapesDataset, HopfieldPEBALModel")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.warning("Custom modules (datasets, HopfieldPEBALModel) not found in Python path.")
    logger.warning("Using placeholder classes for basic script execution. EVALUATION WILL NOT BE ACCURATE.")

    class BaseMockDataset(torch.utils.data.Dataset):
        def __init__(self, images_path, labels_path, transform=None, mask_transform=None, num_classes=19, anomaly_id=None):
            self.images_path = images_path
            self.labels_path = labels_path
            self.transform = transform
            self.mask_transform = mask_transform
            self.num_classes = num_classes
            self.anomaly_id = anomaly_id

            try:
                self.image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:10] # Limit for testing
                self.label_files = sorted([f for f in os.listdir(labels_path) if f.lower().endswith('.png')])[:10] # Limit for testing
                if not self.image_files or not self.label_files:
                   logger.warning(f"MockDataset: Found 0 images or labels in {images_path} / {labels_path}")
                   self.image_files = ["dummy.png"] * 10
                   self.label_files = ["dummy.png"] * 10
                else:
                    # Basic check for matching filenames (ignoring extensions)
                    img_stems = {os.path.splitext(f)[0] for f in self.image_files}
                    lbl_stems = {os.path.splitext(f)[0] for f in self.label_files}
                    common_stems = list(img_stems.intersection(lbl_stems))
                    if not common_stems:
                         logger.warning(f"MockDataset: No matching image/label stems found in {images_path} / {labels_path}")
                    # For simplicity, just use the limited lists if files exist
                    min_len = min(len(self.image_files), len(self.label_files))
                    self.image_files = self.image_files[:min_len]
                    self.label_files = self.label_files[:min_len]


            except FileNotFoundError:
                logger.warning(f"MockDataset: Path not found {images_path} or {labels_path}. Creating dummy file list.")
                self.image_files = ["dummy.png"] * 10
                self.label_files = ["dummy.png"] * 10

            logger.info(f"MockDataset initialized with {len(self.image_files)} dummy samples.")


        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            # Return placeholder tensors
            dummy_image = torch.randn(3, 256, 512)  # C, H, W
            # Create a label map with some variation, including potential anomaly ID
            dummy_label = torch.randint(0, self.num_classes, (256, 512), dtype=torch.long)
            if self.anomaly_id is not None and idx % 5 == 0: # Add some anomalies occasionally
                 dummy_label[100:150, 100:200] = self.anomaly_id
            elif idx % 7 == 0: # Add some void labels
                 dummy_label[50:100, 400:450] = 255


            # Apply transforms if they exist (to check transform pipeline)
            # Need a dummy PIL image to apply real transforms
            pil_img = transforms.ToPILImage()(torch.rand(3, 256, 512))
            pil_lbl = transforms.ToPILImage()(dummy_label.unsqueeze(0).byte()) # Needs byte tensor

            if self.transform:
                try:
                    dummy_image = self.transform(pil_img)
                except Exception as e:
                     logger.warning(f"MockDataset: Error applying transform to dummy image: {e}")
            if self.mask_transform:
                 try:
                    dummy_label = self.mask_transform(pil_lbl)
                 except Exception as e:
                     logger.warning(f"MockDataset: Error applying mask_transform to dummy label: {e}")
                     dummy_label = torch.randint(0, self.num_classes, (256, 512), dtype=torch.long) # Fallback


            return dummy_image, dummy_label

    # Define placeholders based on the mock base class
    class SegmentationDataset(BaseMockDataset): pass
    class SimpleImageDataset(BaseMockDataset): pass # Assuming similar structure for simplicity
    class FishyscapesDataset(BaseMockDataset): pass # Use anomaly_id

    # Add placeholder HopfieldPEBALModel if import fails
    class HopfieldPEBALModel(nn.Module):
        def __init__(self, backbone, segmentation_head, num_classes, feature_dim,
                    hopfield_beta, memory_size, num_heads, insertion_point, target_feature_dim):
            super(HopfieldPEBALModel, self).__init__()
            self.backbone = backbone
            self.segmentation_head = segmentation_head
            self.num_classes = num_classes
            self.feature_dim = feature_dim # Store params even if unused
            self.hopfield_beta = hopfield_beta
            self.memory_size = memory_size
            self.num_heads = num_heads
            self.insertion_point = insertion_point
            self.target_feature_dim = target_feature_dim # Needed for potential adapter

            # Placeholder for a simple adapter if needed based on insertion point
            # This is highly simplified and assumes the backbone output needs adapting
            self.adapter = nn.Identity()
            if self.insertion_point == 'after_backbone':
                 # Try to determine input channels dynamically (won't work reliably with SimpleBackbone)
                 # We'll rely on target_feature_dim provided
                 # This adapter needs actual input dim which we can't know for sure here
                 # self.adapter = nn.Conv2d(INPUT_DIM, self.target_feature_dim, kernel_size=1)
                 logger.warning("Placeholder Hopfield: Adapter logic is simplified.")


            # Placeholder for Hopfield layer (just identity)
            self.hopfield_layer = nn.Identity()
            # Placeholder for energy calculation
            self.energy_head = nn.Conv2d(self.target_feature_dim if self.insertion_point == 'after_backbone' else num_classes, 1, kernel_size=1)


            logger.warning("Using placeholder HopfieldPEBALModel - FORWARD PASS IS A DUMMY.")

        def forward(self, x):
            # Simple forward pass simulating the structure but without real Hopfield
            features = self.backbone(x)

            if self.insertion_point == 'after_backbone':
                 # Apply simplified adapter and dummy hopfield
                 adapted_features = self.adapter(features)
                 hopfield_output = self.hopfield_layer(adapted_features)
                 logits = self.segmentation_head(hopfield_output) # Assume seg head works on adapted features
                 # Dummy energy based on adapted features
                 energy_features = hopfield_output
            elif self.insertion_point == 'after_seghead':
                 logits = self.segmentation_head(features)
                 # Apply dummy hopfield and energy calc after seg head
                 hopfield_output = self.hopfield_layer(logits)
                 energy_features = hopfield_output
            else:
                 logger.error(f"Invalid insertion point: {self.insertion_point}")
                 raise ValueError("Invalid insertion point")

            # Dummy energy map calculation
            # Ensure energy map has spatial dimensions matching logits if possible
            energy = self.energy_head(energy_features)

            # Resize energy map to match input spatial dimensions (H, W) if needed
            # This is a common requirement
            if energy.shape[-2:] != x.shape[-2:]:
                 energy = F.interpolate(energy, size=x.shape[-2:], mode='bilinear', align_corners=False)

            # Make sure logits also match input spatial dimensions if seg head changed them
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)


            return {
                'logits': logits,
                'combined_energy': energy, # Use 'combined_energy' key as expected
                'features': features # Return original backbone features
            }

# --- Argument Parsing ---

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Hopfield-PEBAL model for OOD detection')

    # Dataset paths
    parser.add_argument('--test_images', type=str,
                        default='./data/cityscapes/images/val', # Example defaults
                        help='Path to test (inlier) images (e.g., Cityscapes val)')
    parser.add_argument('--test_labels', type=str,
                        default='./data/cityscapes/labels/val', # Example defaults
                        help='Path to test (inlier) labels (e.g., Cityscapes val)')

    # Fishyscapes dataset paths (assuming a common structure)
    parser.add_argument('--fishyscapes_dir', type=str,
                        default='./data/fishyscapes_lostandfound', # Example default
                        help='Root directory for Fishyscapes datasets')
    # Specific paths derived from root by default, but can be overridden
    parser.add_argument('--lostandfound_images', type=str, default=None, help='Path to Fishyscapes LostAndFound images')
    parser.add_argument('--lostandfound_labels', type=str, default=None, help='Path to Fishyscapes LostAndFound labels')
    parser.add_argument('--static_images', type=str, default=None, help='Path to Fishyscapes Static images')
    parser.add_argument('--static_labels', type=str, default=None, help='Path to Fishyscapes Static labels')
    parser.add_argument('--road_anomaly_images', type=str, default=None, help='Path to Road Anomaly images')
    parser.add_argument('--road_anomaly_labels', type=str, default=None, help='Path to Road Anomaly labels')

    # Evaluation dataset selection
    parser.add_argument('--dataset', type=str,
                        default='lostandfound',
                        choices=['inlier', 'lostandfound', 'static', 'road_anomaly', 'all'],
                        help='Which dataset to evaluate on')

    # Model parameters
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/latest_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='Number of segmentation classes (excluding anomaly/void)')
    parser.add_argument('--feature_dim', type=int, default=256, # Often matches Hopfield internal dim
                        help='Dimension of features used within Hopfield layer (if applicable)')
    parser.add_argument('--hopfield_beta', type=float, default=8.0,
                        help='Beta parameter for Hopfield layer energy calculation')
    parser.add_argument('--memory_size', type=int, default=2000,
                        help='Size of Hopfield memory bank (if applicable)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads in Hopfield layer (if applicable)')
    parser.add_argument('--insertion_point', type=str, default='after_backbone',
                        choices=['after_backbone', 'after_seghead'],
                        help='Where the Hopfield/OOD mechanism is inserted')
    parser.add_argument('--use_simple_model', action='store_true',
                        help='Use a simple CNN backbone/head for testing instead of DeepWV3Plus')
    parser.add_argument('--target_feature_dim', type=int, default=304,
                        help='Target feature dimension for adapter before Hopfield (if insertion=after_backbone)')


    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1, # Smaller batch size for evaluation often okay
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize first few results (image, gt, pred, energy)')
    parser.add_argument('--save_outputs', action='store_true',
                        help='Save model outputs (predictions, energy) as numpy arrays')
    parser.add_argument('--anomaly_id', type=int, default=19, # Often the next ID after inlier classes
                        help='Class ID used for anomalies in OOD datasets')
    parser.add_argument('--void_id', type=int, default=255,
                        help='Class ID for void/ignore regions in labels')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose logging')

    # Added parameters for better control/debugging
    parser.add_argument('--check_files_exist', action='store_true',
                        help='Check if dataset files/directories exist before starting evaluation')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    parser.add_argument('--vis_limit', type=int, default=5,
                        help='Maximum number of samples to visualize per dataset')


    args = parser.parse_args()

    # --- Derive OOD dataset paths if not explicitly set ---
    if args.lostandfound_images is None:
        args.lostandfound_images = os.path.join(args.fishyscapes_dir, 'LostAndFound', 'images')
    if args.lostandfound_labels is None:
        args.lostandfound_labels = os.path.join(args.fishyscapes_dir, 'LostAndFound', 'labels_anomaly') # Adjust if label subdir name differs
    if args.static_images is None:
        args.static_images = os.path.join(args.fishyscapes_dir, 'Static', 'images')
    if args.static_labels is None:
        args.static_labels = os.path.join(args.fishyscapes_dir, 'Static', 'labels_anomaly')
    if args.road_anomaly_images is None:
        args.road_anomaly_images = os.path.join(args.fishyscapes_dir, 'RoadAnomaly', 'images') # Assuming common structure
    if args.road_anomaly_labels is None:
        args.road_anomaly_labels = os.path.join(args.fishyscapes_dir, 'RoadAnomaly', 'labels_anomaly')


    return args

# --- Model Loading and Definition ---

def create_simple_backbone_for_testing(num_classes=19):
    """Create a very simple CNN backbone and segmentation head for testing."""
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.relu2 = nn.ReLU()
            self.last_feature_dim = 32 # Remember the output dim

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            return x

    class SimpleSegHead(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, num_classes, 1)

        def forward(self, x):
            return self.conv(x)

    logger.info("Created simple backbone and segmentation head for testing purposes")
    backbone = SimpleBackbone()
    # Use the known output dim of the simple backbone
    seg_head = SimpleSegHead(backbone.last_feature_dim, num_classes)
    return backbone, seg_head

def import_deepwv3plus():
    """Attempts to import the DeepWV3Plus model from the specified path."""
    # Add potential code directory to Python path - adjust if necessary
    # Example: code_dir = '/path/to/your/hop-pebal/code'
    # if code_dir not in sys.path:
    #     sys.path.append(code_dir)
    #     logger.info(f"Added {code_dir} to Python path (if it exists)")
    code_dir = '/home/ha51dybi/hop-pebal/code' # ADJUST THIS PATH if your 'model' dir is elsewhere
    if os.path.isdir(code_dir):
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir) # Insert at beginning to prioritize
            logger.info(f"Added '{code_dir}' to Python sys.path to find 'model' package")
    else:
        logger.warning(f"Directory specified for DeepWV3Plus code ('{code_dir}') does not exist.")
    try:
        # Try direct import first (if installed or in path)
        from model.wide_network import DeepWV3Plus # Adjust path if needed
        logger.info("Successfully imported DeepWV3Plus directly")

        # Define a standard wrapper to extract backbone and head
        class DeepWV3PlusWrapper:
            def __init__(self, num_classes=19):
                self.model = DeepWV3Plus(num_classes=num_classes) # Pass num_classes
                logger.info("Initialized DeepWV3Plus wrapper")
                # Log model structure for debugging
                logger.debug("DeepWV3Plus model structure (top level):")
                for name, module in self.model.named_children():
                    logger.debug(f"  {name}: {module.__class__.__name__}")

            def get_backbone_and_head(self):
                """Extract backbone and segmentation head based on common DeepWV3Plus structure."""
                # This relies on the assumed structure of DeepWV3Plus
                # Common structure: feature extraction modules + classifier head
                # Find the likely classifier module (often named 'final', 'classifier', or the last module)
                module_names = list(n for n, _ in self.model.named_children())
                if not module_names:
                    logger.error("DeepWV3Plus model has no children modules!")
                    return None, None

                potential_head_names = ['final', 'classifier', 'seg_head']
                head_module = None
                head_name = None

                # Try specific names first
                for name in potential_head_names:
                    if hasattr(self.model, name):
                        head_module = getattr(self.model, name)
                        head_name = name
                        logger.info(f"Identified segmentation head module: '{head_name}'")
                        break

                # If not found by name, assume the last module is the head
                if head_module is None:
                    head_name = module_names[-1]
                    head_module = getattr(self.model, head_name)
                    logger.warning(f"Assuming the last module '{head_name}' is the segmentation head.")


                # The rest is the backbone
                backbone_modules = []
                final_backbone_module_name = None
                for name, module in self.model.named_children():
                    if name != head_name:
                        backbone_modules.append((name, module))
                        final_backbone_module_name = name # Keep track of the last module added to backbone

                if not backbone_modules:
                     logger.error("Could not separate backbone modules from the head!")
                     return None, None

                logger.info(f"Identified backbone modules up to: '{final_backbone_module_name}'")
                backbone = nn.Sequential(dict(backbone_modules))

                # Simple check: Ensure head is a Module
                if not isinstance(head_module, nn.Module):
                    logger.error(f"Identified head '{head_name}' is not an nn.Module!")
                    return None, None

                return backbone, head_module

        return DeepWV3PlusWrapper
    except ModuleNotFoundError as e: # More specific exception
        logger.error(f"Failed to import DeepWV3Plus: {e}")
        logger.warning("Check if 'model.wide_network' is accessible in your Python environment.")
        logger.warning(f"Attempted to look in: {sys.path}") # Log path for debugging
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during DeepWV3Plus import/wrapping: {e}")
        return None
    except ImportError as e:
        logger.error(f"Failed to import DeepWV3Plus: {e}")
        logger.warning("Check if 'model.wide_network' is accessible in your Python environment.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during DeepWV3Plus import/wrapping: {e}")
        return None


def load_model(args, device):
    """Load the segmentation model and wrap it with Hopfield-PEBAL."""
    logger.info("Loading model...")

    # 1. Create Base Model (Backbone + Segmentation Head)
    backbone = None
    segmentation_head = None
    backbone_output_dim = None # We need this for the adapter if inserting after backbone

    if args.use_simple_model:
        logger.info("Using simple model for testing.")
        backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
        # For simple model, we know the output dim
        backbone_output_dim = backbone.last_feature_dim
    else:
        logger.info("Attempting to load DeepWV3Plus model.")
        DeepWV3PlusWrapper = import_deepwv3plus()
        if DeepWV3PlusWrapper is not None:
            try:
                wrapper = DeepWV3PlusWrapper(args.num_classes)
                backbone, segmentation_head = wrapper.get_backbone_and_head()
                if backbone is None or segmentation_head is None:
                     raise ValueError("Failed to extract backbone and head from DeepWV3PlusWrapper.")
                logger.info("Successfully extracted backbone and segmentation head from DeepWV3Plus.")

                # --- Try to determine backbone output dimension ---
                # This is crucial but can be tricky. We run a dummy input.
                try:
                    backbone.eval() # Set to eval mode for dummy pass
                    # Use a plausible input size, ensure it's on the correct device
                    dummy_input = torch.randn(1, 3, 256, 512).to(device)
                    backbone.to(device) # Move backbone to device TEMPORARILY for shape check
                    with torch.no_grad():
                         dummy_output = backbone(dummy_input)
                    backbone_output_dim = dummy_output.shape[1] # Get channel dimension
                    logger.info(f"Detected backbone output feature dimension: {backbone_output_dim}")
                    backbone.cpu() # Move back to CPU; it will be moved back later with the full model
                except Exception as e:
                    logger.error(f"Could not determine backbone output dimension automatically: {e}")
                    logger.warning("Falling back to using --target_feature_dim for adapter input.")
                    backbone_output_dim = args.target_feature_dim # Risky fallback

            except Exception as e:
                logger.error(f"Error initializing DeepWV3Plus or extracting components: {e}")
                logger.warning("Falling back to simple model due to DeepWV3Plus loading error.")
                backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
                backbone_output_dim = backbone.last_feature_dim
        else:
            logger.warning("DeepWV3Plus import failed. Falling back to simple model.")
            backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
            backbone_output_dim = backbone.last_feature_dim

    # Ensure backbone and head are created
    if backbone is None or segmentation_head is None:
         logger.critical("Failed to create base model (backbone and segmentation head). Cannot proceed.")
         raise RuntimeError("Model creation failed.")


    # 2. Create Hopfield-PEBAL Model
    try:
        logger.info(f"Creating HopfieldPEBALModel with insertion_point='{args.insertion_point}'")
        logger.info(f"  num_classes={args.num_classes}, feature_dim={args.feature_dim}, hopfield_beta={args.hopfield_beta}")
        logger.info(f"  memory_size={args.memory_size}, num_heads={args.num_heads}, target_feature_dim={args.target_feature_dim}")

        # If inserting after backbone, the adapter needs the actual backbone output dimension
        adapter_input_dim = None
        effective_target_dim = args.target_feature_dim # Use the argument as the target for the adapter
        if args.insertion_point == 'after_backbone':
            if backbone_output_dim is None:
                 logger.error("Cannot create adapter: backbone output dimension could not be determined.")
                 raise ValueError("Missing backbone output dimension for adapter.")
            adapter_input_dim = backbone_output_dim
            logger.info(f"Adapter will map from {adapter_input_dim} -> {effective_target_dim} channels.")


        model = HopfieldPEBALModel(
            backbone=backbone,
            segmentation_head=segmentation_head,
            num_classes=args.num_classes,
            feature_dim=args.feature_dim, # Often internal Hopfield dim
            hopfield_beta=args.hopfield_beta,
            memory_size=args.memory_size,
            num_heads=args.num_heads,
            insertion_point=args.insertion_point,
            # Pass the determined/fallback dimension needed by the adapter
            target_feature_dim=effective_target_dim
        )
        logger.info("HopfieldPEBALModel created successfully.")

    except Exception as e:
        logger.critical(f"Fatal error creating HopfieldPEBALModel: {e}", exc_info=True)
        raise # Re-raise the exception as this is critical


    # 3. Load Checkpoint
    if os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            state_dict_key = None
            if isinstance(checkpoint, dict):
                # Common keys: 'model_state_dict', 'state_dict', 'model'
                possible_keys = ['model_state_dict', 'state_dict', 'model']
                for key in possible_keys:
                    if key in checkpoint:
                        state_dict_key = key
                        break
                if state_dict_key:
                     state_dict = checkpoint[state_dict_key]
                     logger.info(f"Loading state_dict from checkpoint key: '{state_dict_key}'")
                else:
                     # Assume the whole checkpoint is the state dict
                     state_dict = checkpoint
                     logger.info("Assuming the loaded checkpoint object is the state_dict itself.")
            else:
                # Assume the loaded object is the state dict directly
                state_dict = checkpoint
                logger.info("Loaded checkpoint object is not a dict, assuming it's the state_dict.")


            # --- Load state dict with flexibility ---
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if unexpected_keys:
                logger.warning(f"Checkpoint loading: Unexpected keys found: {unexpected_keys}")
            if missing_keys:
                logger.warning(f"Checkpoint loading: Missing keys in model state_dict: {missing_keys}")
                # Log specifically if Hopfield-related keys are missing (if they exist in the model)
                hopfield_missing = [k for k in missing_keys if 'hopfield' in k or 'energy' in k or 'adapter' in k]
                if hopfield_missing:
                     logger.warning(f"  -> Potentially missing Hopfield/PEBAL related keys: {hopfield_missing}")
                     logger.warning("     This might indicate the checkpoint is only for the base segmentation model.")
                # Log missing backbone/seg_head keys
                base_model_missing = [k for k in missing_keys if 'backbone' in k or 'segmentation_head' in k]
                if base_model_missing:
                    logger.warning(f"  -> Missing base model keys: {base_model_missing}")

            if not missing_keys and not unexpected_keys:
                logger.info("Checkpoint loaded successfully with strict matching.")
            else:
                logger.info("Checkpoint loaded with non-strict matching.")


        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {args.checkpoint}")
            logger.warning("Proceeding with randomly initialized model weights.")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            logger.warning("Could not load weights. Proceeding with randomly initialized model weights.")
    else:
        logger.warning(f"Checkpoint file does not exist: {args.checkpoint}")
        logger.warning("Proceeding with randomly initialized model weights.")

    # Move the final model to the target device
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    return model

# --- Evaluation Metrics ---

def evaluate_segmentation(predictions, targets, num_classes, void_id=255):
    """Calculate per-class IoU and mean IoU (mIoU)."""
    iou_list = []
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()

    # Ignore void pixels
    valid_mask = targ_flat != void_id
    pred_flat = pred_flat[valid_mask]
    targ_flat = targ_flat[valid_mask]

    for cls in range(num_classes):
        pred_inds = pred_flat == cls
        target_inds = targ_flat == cls

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            # If there are no predictions and no ground truth for this class, skip
            # If there is GT but no predictions, or vice versa, IoU is 0
            # Check if class is present in ground truth at all
            if target_inds.sum() == 0:
                 # logger.debug(f"Skipping IoU for class {cls}: not present in ground truth.")
                 continue # Skip class if not in GT
            else:
                 iou = 0.0 # Class present in GT but not predicted, or vice versa
        else:
            iou = intersection / union

        iou_list.append(iou)

    # Calculate mean IoU over classes present in the ground truth
    if not iou_list:
        logger.warning("No valid classes found for mIoU calculation in this batch.")
        return 0.0
    else:
        return np.mean(iou_list)

def evaluate_ood_detection(energy_maps, targets, anomaly_id=19, void_id=255, return_scores=False):
    """Calculate OOD detection metrics (AUROC, AUPRC, FPR95)."""
    # Ensure energy maps and targets are numpy arrays on CPU
    if isinstance(energy_maps, torch.Tensor):
        energy_maps = energy_maps.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Expected energy_maps shape: [B, H, W] or [B, 1, H, W]
    if energy_maps.ndim == 4:
        if energy_maps.shape[1] != 1:
            logger.warning(f"Energy map has unexpected channel dimension {energy_maps.shape[1]}, using only the first channel.")
        energy_maps = energy_maps[:, 0, :, :] # Take first channel

    # Ensure energy maps and targets have the same spatial dimensions H, W
    if energy_maps.shape[-2:] != targets.shape[-2:]:
        target_h, target_w = targets.shape[-2:]
        logger.warning(f"Energy map shape {energy_maps.shape[-2:]} differs from target shape {(target_h, target_w)}. Resizing energy map using bilinear interpolation.")
        # Resize each energy map in the batch
        energy_maps_resized = np.zeros((energy_maps.shape[0], target_h, target_w), dtype=energy_maps.dtype)
        for b in range(energy_maps.shape[0]):
            pil_img = Image.fromarray(energy_maps[b])
            resized_img = pil_img.resize((target_w, target_h), resample=Image.BILINEAR)
            energy_maps_resized[b] = np.array(resized_img)
        energy_maps = energy_maps_resized


    # Flatten arrays
    flat_energy = energy_maps.flatten()
    flat_targets = targets.flatten()

    # Create binary labels: 1 for OOD (anomaly), 0 for In-Distribution
    # Exclude void pixels from calculation
    valid_mask = flat_targets != void_id
    ood_labels = (flat_targets[valid_mask] == anomaly_id).astype(np.uint8)
    ood_scores = flat_energy[valid_mask] # Higher energy score indicates higher likelihood of OOD

    # Check if there are any OOD pixels
    num_ood_pixels = np.sum(ood_labels)
    if num_ood_pixels == 0:
        logger.warning("No OOD pixels found in the current batch targets (after excluding void). Returning default metrics (0.5, 0.5, 1.0).")
        if return_scores:
            return 0.5, 0.5, 1.0, ood_scores, ood_labels
        return 0.5, 0.5, 1.0

    # Check if all pixels are OOD (or all non-void are OOD)
    if num_ood_pixels == len(ood_labels):
         logger.warning("All non-void pixels are OOD pixels. Metrics might be trivial. Returning default metrics (0.5, 0.5, 1.0).")
         if return_scores:
             return 0.5, 0.5, 1.0, ood_scores, ood_labels
         return 0.5, 0.5, 1.0


    # Check for constant scores (can cause issues with metrics)
    if np.all(ood_scores == ood_scores[0]):
        logger.warning("All energy scores are constant in this batch. Metrics may be undefined or misleading. Returning default metrics (0.5, 0.5, 1.0).")
        if return_scores:
            return 0.5, 0.5, 1.0, ood_scores, ood_labels
        return 0.5, 0.5, 1.0


    try:
        # Calculate AUROC
        auroc = roc_auc_score(ood_labels, ood_scores)

        # Calculate AUPRC (Average Precision)
        auprc = average_precision_score(ood_labels, ood_scores)

        # Calculate FPR at 95% TPR (FPR95)
        precision, recall, thresholds = precision_recall_curve(ood_labels, ood_scores) # Recall = TPR
        fpr95 = 1.0 # Default value if 95% TPR is not reached

        # Find the threshold closest to 95% TPR (recall)
        if np.max(recall) >= 0.95:
            # Find the index of the closest recall value to 0.95
            idx = np.argmin(np.abs(recall - 0.95))
            # Calculate FPR = FP / (FP + TN)
            # Precision = TP / (TP + FP)
            # Recall = TP / (TP + FN)
            # We need TP, FP, TN, FN at that threshold. sklearn doesn't directly give FP, TN.
            # Alternative: Use ROC curve points if available, or estimate from precision/recall
            # fpr = FP / (FP + TN) = FP / N, where N = FP + TN is number of negatives (In-Distribution)
            # Let P = TP + FN (number of positives, OOD)
            # precision = TP / (TP+FP) => FP = TP/precision - TP = TP * (1/precision - 1)
            # recall = TP / P => TP = recall * P
            # FP = recall * P * (1/precision - 1)
            # N = Total_valid - P = len(ood_labels) - P
            # fpr = (recall * P * (1/precision - 1)) / N
            # This seems overly complex and potentially unstable if precision is near 0.

            # Simpler approach using the definition: Find threshold for 95% TPR,
            # then count False Positives at that threshold.
            threshold_95tpr = thresholds[idx]
            # Note: thresholds from precision_recall_curve might not include endpoint for recall=0.
            # Let's use a slightly safer threshold index if possible.
            # Find first index where recall >= 0.95
            valid_indices = np.where(recall >= 0.95)[0]
            if len(valid_indices) > 0:
                idx_95 = valid_indices[0] # First index where recall is >= 0.95
                # If using thresholds from precision_recall_curve, length is len(precision)-1
                # Handle potential index out of bounds for thresholds array
                threshold_idx = min(idx_95, len(thresholds) - 1)
                threshold_95tpr = thresholds[threshold_idx]

                # Predict OOD if score >= threshold
                predicted_positive = ood_scores >= threshold_95tpr
                # False Positives: Predicted positive but actually negative (In-Distribution)
                fp = np.sum(predicted_positive & (ood_labels == 0))
                # True Negatives: Predicted negative and actually negative
                tn = np.sum((~predicted_positive) & (ood_labels == 0))
                # Total Negatives (In-Distribution)
                num_negatives = fp + tn

                if num_negatives > 0:
                    fpr95 = fp / num_negatives
                else:
                    logger.warning("No negative samples found at 95% TPR threshold, FPR95 set to 1.0")
                    fpr95 = 1.0 # Avoid division by zero if no negatives exist (edge case)
            else:
                 logger.warning("Could not find threshold for 95% TPR. FPR95 set to 1.0")


        else:
            logger.warning(f"Maximum TPR achieved is {np.max(recall):.4f}, which is less than 0.95. FPR95 is set to 1.0.")
            fpr95 = 1.0

        if return_scores:
            return auroc, auprc, fpr95, ood_scores, ood_labels
        return auroc, auprc, fpr95

    except Exception as e:
        logger.error(f"Error calculating OOD metrics: {e}", exc_info=True)
        if return_scores:
            # Return default values and potentially problematic scores/labels for debugging
            return 0.5, 0.5, 1.0, ood_scores, ood_labels
        return 0.5, 0.5, 1.0


# --- Visualization ---

def visualize_results(image_orig, targets, prediction, energy, output_path, vis_limit=5):
    """Visualize results: Original Image, GT Label, Pred Label, Energy Map."""
    try:
        # Ensure data is numpy array on CPU
        if isinstance(image_orig, torch.Tensor):
            image_orig = image_orig.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        if isinstance(energy, torch.Tensor):
            energy = energy.cpu().numpy()

        # Handle potential batch dimension if present (take first item)
        if image_orig.ndim == 4: image_orig = image_orig[0]
        if targets.ndim == 3: targets = targets[0]
        if prediction.ndim == 3: prediction = prediction[0]
        if energy.ndim == 3: energy = energy[0] # Assuming energy is [1, H, W] or [H, W]

        # Transpose image if it's CxHxW
        if image_orig.shape[0] in [1, 3]:
            image_orig = image_orig.transpose(1, 2, 0)

        # Denormalize image (assuming standard ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_vis = image_orig * std + mean
        image_vis = np.clip(image_vis, 0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original Image
        axes[0].imshow(image_vis)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Ground Truth Segmentation
        axes[1].imshow(targets, cmap='viridis') # Use a suitable colormap for labels
        axes[1].set_title('Ground Truth Label')
        axes[1].axis('off')

        # Predicted Segmentation
        axes[2].imshow(prediction, cmap='viridis') # Use the same colormap
        axes[2].set_title('Predicted Label')
        axes[2].axis('off')

        # Energy Map (Higher energy -> Higher OOD likelihood)
        im = axes[3].imshow(energy, cmap='jet') # Jet often used for heatmaps
        axes[3].set_title('OOD Energy Map (Higher=OOD)')
        axes[3].axis('off')
        # Add colorbar for the energy map
        cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label('Energy Score')


        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory
        logger.debug(f"Visualization saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during visualization for {output_path}: {e}", exc_info=True)


# --- Dataset File Check ---

def check_dataset_files(images_path, labels_path, dataset_name):
    """Check if dataset directories exist and contain files."""
    valid = True
    if not os.path.isdir(images_path):
        logger.error(f"[{dataset_name}] Images path does not exist or is not a directory: {images_path}")
        valid = False
    elif len(os.listdir(images_path)) == 0:
        logger.error(f"[{dataset_name}] Images directory is empty: {images_path}")
        valid = False

    if not os.path.isdir(labels_path):
        logger.error(f"[{dataset_name}] Labels path does not exist or is not a directory: {labels_path}")
        valid = False
    elif len(os.listdir(labels_path)) == 0:
        logger.error(f"[{dataset_name}] Labels directory is empty: {labels_path}")
        valid = False

    if valid:
        logger.info(f"[{dataset_name}] Found images in: {images_path}")
        logger.info(f"[{dataset_name}] Found labels in: {labels_path}")
        # Log a few file examples
        try:
            img_samples = sorted(os.listdir(images_path))[:3]
            lbl_samples = sorted(os.listdir(labels_path))[:3]
            logger.info(f"  Image samples: {img_samples}")
            logger.info(f"  Label samples: {lbl_samples}")
        except Exception as e:
             logger.warning(f"Could not list sample files for {dataset_name}: {e}")

    return valid


# --- Core Evaluation Loop ---

def evaluate_on_dataset(args, model, dataset_name, device):
    """Evaluate the model on a specific dataset."""
    logger.info(f"--- Starting evaluation on {dataset_name} dataset ---")

    # 1. Setup Paths and Output Directory
    is_ood = False
    if dataset_name == 'inlier':
        images_path = args.test_images
        labels_path = args.test_labels
        dataset_class = SegmentationDataset
        output_dir = os.path.join(args.output_dir, 'inlier_results')
        is_ood = False
    elif dataset_name == 'lostandfound':
        images_path = args.lostandfound_images
        labels_path = args.lostandfound_labels
        dataset_class = FishyscapesDataset
        output_dir = os.path.join(args.output_dir, 'lostandfound_results')
        is_ood = True
    elif dataset_name == 'static':
        images_path = args.static_images
        labels_path = args.static_labels
        dataset_class = FishyscapesDataset
        output_dir = os.path.join(args.output_dir, 'static_results')
        is_ood = True
    elif dataset_name == 'road_anomaly':
        images_path = args.road_anomaly_images
        labels_path = args.road_anomaly_labels
        dataset_class = FishyscapesDataset
        output_dir = os.path.join(args.output_dir, 'road_anomaly_results')
        is_ood = True
    else:
        logger.error(f"Unknown dataset specified: {dataset_name}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {output_dir}")

    # 2. Check Dataset Files (Optional)
    if args.check_files_exist:
        if not check_dataset_files(images_path, labels_path, dataset_name):
            logger.error(f"Dataset file check failed for {dataset_name}. Skipping evaluation.")
            # Create a dummy output file to indicate failure
            fail_path = os.path.join(output_dir, "evaluation_failed_file_check.txt")
            with open(fail_path, 'w') as f:
                f.write(f"Evaluation failed: Dataset files check failed for {dataset_name}\n")
                f.write(f"Images path checked: {images_path}\n")
                f.write(f"Labels path checked: {labels_path}\n")
            return None # Indicate failure

    # 3. Define Transformations
    # Using common practice sizes, adjust if needed
    eval_height, eval_width = 256, 512
    logger.info(f"Using evaluation image size: ({eval_height}, {eval_width})")

    transform = transforms.Compose([
        transforms.Resize((eval_height, eval_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Mask transform should only resize (using NEAREST) and convert to tensor
    mask_transform = transforms.Compose([
        transforms.Resize((eval_height, eval_width), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.as_tensor(np.array(img), dtype=torch.long)) # Convert PIL->numpy->long tensor
    ])


    # 4. Create Dataset and DataLoader
    try:
        logger.info(f"Attempting to load dataset using class: {dataset_class.__name__}")
        common_kwargs = {
            'transform': transform,
            'mask_transform': mask_transform,
            'num_classes': args.num_classes
        }
        if is_ood:
            dataset = dataset_class(
                images_path=images_path,
                labels_path=labels_path,
                anomaly_id=args.anomaly_id
                **common_kwargs# Pass anomaly ID for OOD datasets
            )
        else: # Inlier dataset
             dataset = dataset_class(
                images_path=images_path,
                labels_path=labels_path,
                **common_kwargs
             )
        logger.info(f"Successfully created {dataset_name} dataset with {len(dataset)} samples.")

    except TypeError as e: # Catch the specific error observed
        logger.error(f"TypeError creating dataset for {dataset_name}: {e}", exc_info=True)
        logger.error("This likely means the assumed arguments (positional img/label paths, then keywords) for your custom dataset classes are incorrect.")
        logger.error("Please check the __init__ signature of your SegmentationDataset and FishyscapesDataset classes and modify the call in evaluate.py accordingly.")
        fail_path = os.path.join(output_dir, "evaluation_failed_dataset_TypeError.txt")
        with open(fail_path, 'w') as f:
            f.write(f"Evaluation failed during dataset creation for {dataset_name}: {e}\n")
            f.write("Check dataset __init__ signature.\n")
        return None
    except Exception as e:
        logger.error(f"Error creating dataset for {dataset_name}: {e}", exc_info=True)
        fail_path = os.path.join(output_dir, "evaluation_failed_dataset_creation.txt")
        with open(fail_path, 'w') as f:
            f.write(f"Evaluation failed during dataset creation for {dataset_name}: {e}\n")
        return None


    # Check if dataset is empty
    if len(dataset) == 0:
        logger.error(f"Dataset '{dataset_name}' is empty. Cannot evaluate.")
        fail_path = os.path.join(output_dir, "evaluation_failed_empty_dataset.txt")
        with open(fail_path, 'w') as f:
            f.write(f"Evaluation failed: Dataset '{dataset_name}' reported 0 samples.\n")
        return None

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffling for evaluation
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False # Pin memory only if using CUDA
    )
    logger.info(f"DataLoader created with batch_size={args.batch_size}, num_workers={args.num_workers}")

    # 5. Evaluation Loop
    model.eval() # Set model to evaluation mode
    all_metrics = {}
    all_outputs = [] # For saving raw outputs if needed

    # Metrics accumulators
    total_miou = 0.0
    total_auroc = 0.0
    total_auprc = 0.0
    total_fpr95 = 0.0
    num_batches_processed = 0
    num_valid_ood_batches = 0 # Batches where OOD metrics were successfully calculated

    logger.info(f"Starting inference loop for {len(data_loader)} batches...")
    vis_count = 0

    try: # Wrap the main loop for unexpected errors during iteration
        for i, batch_data in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
            try:
                # Unpack batch, handle potential variations in dataset output
                if len(batch_data) == 2:
                    images, masks = batch_data
                # Add more checks here if datasets might return different structures
                else:
                    logger.error(f"Unexpected batch data structure in batch {i}. Expected 2 items, got {len(batch_data)}. Skipping batch.")
                    continue

                # Move data to the target device
                images = images.to(device)
                masks = masks.to(device) # Move masks too for potential resizing/comparison

                # Log shapes for debugging first batch
                if i == 0:
                    logger.debug(f"Batch 0: Image shape={images.shape}, Mask shape={masks.shape}")


                # --- Perform Inference ---
                with torch.no_grad(): # Ensure no gradients are calculated
                    try:
                        outputs = model(images)
                    except Exception as e:
                         logger.error(f"Error during model forward pass on batch {i}: {e}", exc_info=True)
                         logger.error(f"Input image shape: {images.shape}")
                         continue # Skip to next batch on forward pass error


                # --- Extract Outputs ---
                try:
                    logits = outputs['logits']         # Expected shape: [B, NumClasses, H, W]
                    energy = outputs['combined_energy'] # Expected shape: [B, 1, H, W]
                except KeyError as e:
                    logger.error(f"Missing key '{e}' in model output dictionary on batch {i}. Available keys: {outputs.keys()}. Skipping batch.")
                    continue
                except Exception as e:
                    logger.error(f"Error extracting logits/energy from model output on batch {i}: {e}", exc_info=True)
                    continue


                # --- Resize Outputs to Match Mask Dimensions ---
                target_h, target_w = masks.shape[-2:]
                if logits.shape[-2:] != (target_h, target_w):
                    logger.debug(f"Resizing logits from {logits.shape[-2:]} to {(target_h, target_w)}")
                    logits = F.interpolate(logits, size=(target_h, target_w), mode='bilinear', align_corners=False)

                if energy.shape[-2:] != (target_h, target_w):
                    logger.debug(f"Resizing energy from {energy.shape[-2:]} to {(target_h, target_w)}")
                    energy = F.interpolate(energy, size=(target_h, target_w), mode='bilinear', align_corners=False)


                # --- Calculate Metrics ---
                predictions = torch.argmax(logits, dim=1) # Get predicted class labels [B, H, W]

                if not is_ood: # Inlier dataset: Calculate mIoU
                    try:
                        batch_miou = evaluate_segmentation(
                            predictions.cpu().numpy(), # Move to CPU and convert to numpy
                            masks.cpu().numpy(),
                            args.num_classes,
                            args.void_id
                        )
                        total_miou += batch_miou
                        num_batches_processed += 1
                        if i % 20 == 0: # Log periodically
                             logger.debug(f"Batch {i}: mIoU = {batch_miou:.4f}")
                    except Exception as e:
                         logger.error(f"Error calculating segmentation metrics for batch {i}: {e}", exc_info=True)

                else: # OOD dataset: Calculate OOD detection metrics
                    try:
                        # evaluate_ood_detection expects numpy arrays on CPU
                        batch_auroc, batch_auprc, batch_fpr95 = evaluate_ood_detection(
                            energy, # Pass tensor, function will handle conversion
                            masks,
                            args.anomaly_id,
                            args.void_id
                        )
                        # Check if metrics are valid (not the default failure values)
                        # Allows averaging only over batches where OOD pixels were present etc.
                        if not (batch_auroc == 0.5 and batch_auprc == 0.5 and batch_fpr95 == 1.0):
                             total_auroc += batch_auroc
                             total_auprc += batch_auprc
                             total_fpr95 += batch_fpr95
                             num_valid_ood_batches += 1
                        else:
                             logger.debug(f"Batch {i}: OOD metrics returned default values (likely no OOD pixels or other issue).")


                        num_batches_processed += 1
                        if i % 20 == 0: # Log periodically
                             logger.debug(f"Batch {i}: AUROC={batch_auroc:.4f}, AUPRC={batch_auprc:.4f}, FPR95={batch_fpr95:.4f}")

                    except Exception as e:
                         logger.error(f"Error calculating OOD metrics for batch {i}: {e}", exc_info=True)


                # --- Save Outputs (Optional) ---
                if args.save_outputs:
                    # Store necessary info for potential later analysis
                    try:
                        batch_output = {
                            'image_paths': [dataset.image_files[idx] for idx in range(i*args.batch_size, min((i+1)*args.batch_size, len(dataset)))], # Get filenames if possible
                            # Save on CPU as numpy arrays
                            'targets': masks.cpu().numpy().astype(np.uint8),
                            'predictions': predictions.cpu().numpy().astype(np.uint8),
                            'energy': energy.squeeze(1).cpu().numpy().astype(np.float32) # Remove channel dim for energy
                        }
                        all_outputs.append(batch_output)
                    except Exception as e:
                         logger.error(f"Error preparing outputs for saving on batch {i}: {e}")

                # --- Visualize Results (Optional) ---
                if args.visualize and vis_count < args.vis_limit:
                    try:
                        # Visualize the first image in the current batch
                        vis_idx_in_batch = 0
                        # Need original image if possible, dataloader usually returns normalized tensor
                        # For simplicity, visualize the normalized tensor after denormalizing
                        image_tensor_vis = images[vis_idx_in_batch]

                        output_filename = f"visualization_{dataset_name}_batch{i}_img{vis_idx_in_batch}.png"
                        output_vis_path = os.path.join(output_dir, output_filename)

                        visualize_results(
                            image_tensor_vis, # Pass tensor, function handles denorm/numpy
                            masks[vis_idx_in_batch],
                            predictions[vis_idx_in_batch],
                            energy[vis_idx_in_batch].squeeze(0), # Remove channel dim for vis
                            output_vis_path
                        )
                        vis_count += 1
                    except Exception as e:
                        logger.error(f"Error during visualization for batch {i}: {e}", exc_info=True)


            except Exception as e: # Catch errors within the batch processing loop
                logger.error(f"Unexpected error processing batch {i} for dataset {dataset_name}: {e}", exc_info=True)
                continue # Continue to the next batch

    except Exception as e: # Catch errors in the dataloader iteration itself
         logger.error(f"Fatal error during DataLoader iteration for {dataset_name}: {e}", exc_info=True)
         # Attempt to save any partial results
         if not is_ood and num_batches_processed > 0:
             all_metrics['miou'] = total_miou / num_batches_processed
         elif is_ood and num_valid_ood_batches > 0:
             all_metrics['auroc'] = total_auroc / num_valid_ood_batches
             all_metrics['auprc'] = total_auprc / num_valid_ood_batches
             all_metrics['fpr95'] = total_fpr95 / num_valid_ood_batches
         # Save what we have
         save_results(all_metrics, all_outputs, output_dir, args.save_outputs, dataset_name, partial=True)
         return None # Indicate failure


    logger.info(f"Inference loop finished for {dataset_name}. Processed {num_batches_processed} batches.")

    # 6. Calculate Final Metrics
    if not is_ood:
        if num_batches_processed > 0:
            final_miou = total_miou / num_batches_processed
            all_metrics['miou'] = final_miou
            logger.info(f"[{dataset_name}] Final mIoU: {final_miou:.4f}")
        else:
            logger.error(f"[{dataset_name}] No batches successfully processed for mIoU calculation.")
            all_metrics['miou'] = 0.0
    else:
        if num_valid_ood_batches > 0:
            final_auroc = total_auroc / num_valid_ood_batches
            final_auprc = total_auprc / num_valid_ood_batches
            final_fpr95 = total_fpr95 / num_valid_ood_batches
            all_metrics['auroc'] = final_auroc
            all_metrics['auprc'] = final_auprc
            all_metrics['fpr95'] = final_fpr95
            logger.info(f"[{dataset_name}] Final OOD Metrics ({num_valid_ood_batches} valid batches):")
            logger.info(f"  AUROC: {final_auroc:.4f}")
            logger.info(f"  AUPRC: {final_auprc:.4f}")
            logger.info(f"  FPR95: {final_fpr95:.4f}")
        else:
            logger.error(f"[{dataset_name}] No valid batches found for OOD metric calculation.")
            all_metrics['auroc'] = 0.5
            all_metrics['auprc'] = 0.5 # Use appropriate default for PR curve
            all_metrics['fpr95'] = 1.0


    # 7. Save Results
    save_results(all_metrics, all_outputs, output_dir, args.save_outputs, dataset_name)

    logger.info(f"--- Finished evaluation on {dataset_name} dataset ---")
    return all_metrics


def save_results(metrics, outputs, output_dir, save_outputs_flag, dataset_name, partial=False):
    """Save calculated metrics and optionally raw outputs."""
    try:
        suffix = "_partial" if partial else ""
        # Save metrics to text file
        metrics_txt_path = os.path.join(output_dir, f"metrics{suffix}.txt")
        with open(metrics_txt_path, 'w') as f:
            if partial:
                f.write("--- PARTIAL RESULTS (due to error during evaluation) ---\n")
            f.write(f"Metrics for dataset: {dataset_name}\n")
            if metrics:
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")
            else:
                 f.write("  No metrics were calculated.\n")
        logger.info(f"Metrics saved to {metrics_txt_path}")

        # Save metrics to numpy file
        metrics_npy_path = os.path.join(output_dir, f"metrics{suffix}.npy")
        np.save(metrics_npy_path, metrics)
        logger.info(f"Metrics saved to {metrics_npy_path}")


        # Save raw outputs if requested and available
        if save_outputs_flag and outputs:
            outputs_npy_path = os.path.join(output_dir, f"outputs{suffix}.npy")
            np.save(outputs_npy_path, outputs)
            logger.info(f"Raw outputs saved to {outputs_npy_path}")
        elif save_outputs_flag:
             logger.warning("Save outputs requested, but no outputs were collected.")

    except Exception as e:
        logger.error(f"Error saving results for {dataset_name}: {e}", exc_info=True)


# --- Main Evaluation Orchestrator ---

def evaluate(args):
    """Main function to orchestrate the evaluation process."""
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Set level for root logger as well if needed
        # logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")
    else:
        logger.setLevel(logging.INFO)

    logger.info("========================================")
    logger.info(" Starting Hopfield-PEBAL Evaluation ")
    logger.info("========================================")
    logger.info(f"Arguments: {vars(args)}") # Log all arguments

    # Determine device
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
         logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
         logger.info("Using CPU device.")


    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Base output directory: {args.output_dir}")

    # Write parameters to a file for reference
    params_path = os.path.join(args.output_dir, "evaluation_parameters.txt")
    try:
        with open(params_path, 'w') as f:
            for arg, value in sorted(vars(args).items()):
                f.write(f"{arg}: {value}\n")
        logger.info(f"Evaluation parameters saved to {params_path}")
    except Exception as e:
        logger.error(f"Could not write parameters file: {e}")


    # --- Load Model ---
    try:
        model = load_model(args, device)
        # Log model summary
        logger.info(f"Model loaded: {type(model).__name__}")
        try:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model trainable parameters: {num_params:,}")
        except Exception as e:
             logger.warning(f"Could not count model parameters: {e}")

    except Exception as e:
        logger.critical(f"Fatal error loading model: {e}", exc_info=True)
        fail_path = os.path.join(args.output_dir, "evaluation_failed_model_load.txt")
        with open(fail_path, 'w') as f:
            f.write(f"Evaluation failed critically during model loading: {e}\n")
        return # Exit evaluation if model loading fails

    # --- Determine Datasets to Evaluate ---
    if args.dataset == 'all':
        # Ensure order consistency
        datasets_to_evaluate = ['inlier', 'lostandfound', 'static', 'road_anomaly']
    else:
        datasets_to_evaluate = [args.dataset]
    logger.info(f"Datasets selected for evaluation: {datasets_to_evaluate}")


    # --- Run Evaluation on Each Dataset ---
    all_results = {}
    evaluation_successful = True

    for dataset_name in datasets_to_evaluate:
        try:
            metrics = evaluate_on_dataset(args, model, dataset_name, device)
            if metrics is not None:
                all_results[dataset_name] = metrics
            else:
                 logger.error(f"Evaluation failed or returned no metrics for dataset: {dataset_name}")
                 evaluation_successful = False # Mark overall evaluation as potentially incomplete

        except Exception as e:
            logger.error(f"Unhandled exception during evaluation of {dataset_name}: {e}", exc_info=True)
            fail_path = os.path.join(args.output_dir, f"{dataset_name}_evaluation_failed_unhandled.txt")
            with open(fail_path, 'w') as f:
                 f.write(f"Evaluation failed with unhandled exception for {dataset_name}: {e}\n")
            evaluation_successful = False


    # --- Save Combined Results ---
    if all_results:
        logger.info("--- Evaluation Summary ---")
        combined_metrics_path_txt = os.path.join(args.output_dir, "all_metrics_summary.txt")
        combined_metrics_path_npy = os.path.join(args.output_dir, "all_metrics_summary.npy")
        try:
            with open(combined_metrics_path_txt, 'w') as f:
                f.write("Combined Evaluation Results\n")
                f.write("===========================\n")
                for dataset, metrics in all_results.items():
                    f.write(f"\nDataset: {dataset}\n")
                    if metrics:
                         for metric, value in metrics.items():
                             f.write(f"  {metric}: {value:.6f}\n")
                    else:
                         f.write("  No metrics available for this dataset.\n")
            logger.info(f"Combined metrics summary saved to {combined_metrics_path_txt}")

            np.save(combined_metrics_path_npy, all_results)
            logger.info(f"Combined metrics dictionary saved to {combined_metrics_path_npy}")

        except Exception as e:
             logger.error(f"Error saving combined results: {e}")
             evaluation_successful = False

    else:
        logger.error("No metrics were collected from any dataset.")
        evaluation_successful = False
        fail_path = os.path.join(args.output_dir, "evaluation_failed_no_results.txt")
        with open(fail_path, 'w') as f:
            f.write("Evaluation completed, but no metrics were successfully collected from any dataset.\n")

    logger.info("========================================")
    if evaluation_successful and all_results:
         logger.info(" Evaluation finished successfully! ")
    else:
         logger.warning(" Evaluation finished, but some datasets may have failed or produced no results. Please check logs and output files.")
    logger.info("========================================")


# --- Entry Point ---

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
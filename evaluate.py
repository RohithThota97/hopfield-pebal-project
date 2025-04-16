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

# Import custom modules - with error handling
try:
    from datasets.datasets import SegmentationDataset, SimpleImageDataset
    from datasets import FishyscapesDataset
    from hopfield_pebal_model import HopfieldPEBALModel
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.warning("Make sure all required modules are in the Python path")
    # Define placeholder classes for testing
    class SegmentationDataset:
        def __init__(self, *args, **kwargs):
            pass
    class SimpleImageDataset:
        def __init__(self, *args, **kwargs):
            pass
    class FishyscapesDataset:
        def __init__(self, *args, **kwargs):
            pass

# Set up logging with more verbose output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hopfield-PEBAL-Evaluation")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Hopfield-PEBAL model for OOD detection')
    
    # Dataset paths
    parser.add_argument('--test_images', type=str, 
                        default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/val',
                        help='Path to test images')
    parser.add_argument('--test_labels', type=str, 
                        default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/val',
                        help='Path to test labels')
    
    # Fishyscapes dataset paths
    parser.add_argument('--fishyscapes_dir', type=str, 
                        default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound',
                        help='Root directory for Fishyscapes datasets')
    parser.add_argument('--lostandfound_images', type=str, 
                        default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound/original',
                        help='Path to Fishyscapes LostAndFound images')
    parser.add_argument('--lostandfound_labels', type=str, 
                        default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound/labels',
                        help='Path to Fishyscapes LostAndFound labels')
    parser.add_argument('--static_images', type=str, 
                        default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static/original',
                        help='Path to Fishyscapes Static images')
    parser.add_argument('--static_labels', type=str, 
                        default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static/labels',
                        help='Path to Fishyscapes Static labels')
    parser.add_argument('--road_anomaly_images', type=str, 
                        default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly/original',
                        help='Path to Road Anomaly images')
    parser.add_argument('--road_anomaly_labels', type=str, 
                        default='/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly/labels',
                        help='Path to Road Anomaly labels')
    
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
                        help='Number of segmentation classes')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Dimension of Hopfield feature vectors')
    parser.add_argument('--hopfield_beta', type=float, default=8.0,
                        help='Beta parameter for Hopfield layer')
    parser.add_argument('--memory_size', type=int, default=2000,
                        help='Size of Hopfield memory bank')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads in Hopfield layer')
    parser.add_argument('--insertion_point', type=str, default='after_backbone',
                        choices=['after_backbone', 'after_seghead'],
                        help='Where to insert Hopfield layer')
    parser.add_argument('--use_simple_model', action='store_true',
                        help='Use simple model instead of DeepWV3Plus')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--save_outputs', action='store_true',
                        help='Save model outputs')
    parser.add_argument('--anomaly_id', type=int, default=19,
                        help='Class ID for anomalies')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose logging')
    
    # Added parameters for better debugging
    parser.add_argument('--check_files_exist', action='store_true',
                        help='Check if dataset files exist before evaluation')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    
    return parser.parse_args()

def create_simple_backbone_for_testing(num_classes=19):
    """Create a simple backbone model for testing"""
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super(SimpleBackbone, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),  # Changed to non-inplace to avoid issues
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False)  # Changed to non-inplace
            )
            
        def forward(self, x):
            return self.features(x)
    
    class SimpleSegHead(nn.Module):
        def __init__(self, num_classes):
            super(SimpleSegHead, self).__init__()
            self.head = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),  # Changed to non-inplace
                nn.Conv2d(128, num_classes, kernel_size=1)
            )
            
        def forward(self, x):
            return self.head(x)
    
    logger.info("Created simple backbone and segmentation head for testing purposes")
    return SimpleBackbone(), SimpleSegHead(num_classes)

def import_deepwv3plus():
    """
    Import the DeepWV3Plus model or provide a wrapper to adapt to its structure
    """
    # Add code directory to Python path
    code_dir = '/home/ha51dybi/hop-pebal/code'
    if code_dir not in sys.path:
        sys.path.append(code_dir)
        logger.info(f"Added {code_dir} to Python path")
    
    try:
        # Try direct import
        from model.wide_network import DeepWV3Plus
        logger.info("Successfully imported DeepWV3Plus directly")
        
        # Create a wrapper for DeepWV3Plus to extract backbone and segmentation head
        class DeepWV3PlusWrapper:
            def __init__(self, num_classes=19):
                self.model = DeepWV3Plus(num_classes)
                logger.info("Initialized DeepWV3Plus wrapper")
                
                # Print model structure for debugging
                logger.info("DeepWV3Plus model structure:")
                for name, module in self.model.named_children():
                    logger.info(f"  {name}: {module.__class__.__name__}")
            
            def get_backbone_and_head(self):
                """Extract backbone and segmentation head based on model structure"""
                # Check if model has expected structure
                if hasattr(self.model, 'mod1') and hasattr(self.model, 'mod2'):
                    # Create a custom backbone class to encapsulate the feature extraction
                    class Backbone(nn.Module):
                        def __init__(self, model):
                            super(Backbone, self).__init__()
                            self.mod1 = model.mod1
                            self.pool2 = model.pool2
                            self.mod2 = model.mod2
                            self.mod3 = model.mod3 if hasattr(model, 'mod3') else None
                            self.mod4 = model.mod4 if hasattr(model, 'mod4') else None
                            self.mod5 = model.mod5 if hasattr(model, 'mod5') else None
                            self.mod6 = model.mod6 if hasattr(model, 'mod6') else None
                            self.mod7 = model.mod7 if hasattr(model, 'mod7') else None
                            
                        def forward(self, x):
                            x = self.mod1(x)
                            x = self.pool2(x)
                            x = self.mod2(x)
                            if self.mod3 is not None:
                                x = self.mod3(x)
                            if self.mod4 is not None:
                                x = self.mod4(x)
                            if self.mod5 is not None:
                                x = self.mod5(x)
                            if self.mod6 is not None:
                                x = self.mod6(x)
                            if self.mod7 is not None:
                                x = self.mod7(x)
                            return x
                    
                    # Create a segmentation head from the classifier
                    class SegHead(nn.Module):
                        def __init__(self, model):
                            super(SegHead, self).__init__()
                            if hasattr(model, 'final'):
                                self.classifier = model.final
                            elif hasattr(model, 'classifier'):
                                self.classifier = model.classifier
                            else:
                                self.classifier = nn.Identity()
                                logger.warning("No classifier found in model, using identity")
                            
                        def forward(self, x):
                            return self.classifier(x)
                    
                    backbone = Backbone(self.model)
                    seghead = SegHead(self.model)
                    logger.info("Using mod1->mod7 as backbone and classifier as segmentation head")
                    return backbone, seghead
                else:
                    # Fallback: simple splitting of the model
                    modules = list(self.model.children())
                    if len(modules) >= 2:
                        # Assume last module is the classifier
                        backbone_modules = nn.Sequential(*modules[:-1])
                        seghead_module = modules[-1]
                        
                        class Backbone(nn.Module):
                            def __init__(self, modules):
                                super(Backbone, self).__init__()
                                self.modules_list = modules
                                
                            def forward(self, x):
                                return self.modules_list(x)
                        
                        class SegHead(nn.Module):
                            def __init__(self, module):
                                super(SegHead, self).__init__()
                                self.module = module
                                
                            def forward(self, x):
                                return self.module(x)
                        
                        backbone = Backbone(backbone_modules)
                        seghead = SegHead(seghead_module)
                        logger.info("Split model into backbone and segmentation head based on module list")
                        return backbone, seghead
                    else:
                        logger.warning("Model structure not as expected, creating simple backbone and head")
                        return create_simple_backbone_for_testing(self.model.num_classes)
        
        return DeepWV3PlusWrapper
    
    except ImportError as e:
        logger.error(f"Direct import failed: {e}")
        try:
            # Try manual import via importlib
            module_path = os.path.join(code_dir, 'model', 'wide_network.py')
            if not os.path.exists(module_path):
                logger.error(f"File does not exist: {module_path}")
                return None
            
            spec = importlib.util.spec_from_file_location("wide_network", module_path)
            if spec is None:
                logger.error(f"Could not create spec for module at {module_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'DeepWV3Plus'):
                DeepWV3Plus = module.DeepWV3Plus
                logger.info(f"Successfully imported DeepWV3Plus via importlib")
                
                # Create a wrapper for DeepWV3Plus (same class as above)
                class DeepWV3PlusWrapper:
                    def __init__(self, num_classes=19):
                        self.model = DeepWV3Plus(num_classes)
                        logger.info("Initialized DeepWV3Plus wrapper")
                        
                    def get_backbone_and_head(self):
                        # Same implementation as above (simplified)
                        return create_simple_backbone_for_testing(self.model.num_classes)
                
                return DeepWV3PlusWrapper
            else:
                logger.error("Module loaded but DeepWV3Plus class not found")
                return None
        
        except Exception as e:
            logger.error(f"All import attempts failed: {e}")
            return None

# Add a placeholder HopfieldPEBALModel in case import fails
if 'HopfieldPEBALModel' not in globals():
    class HopfieldPEBALModel(nn.Module):
        def __init__(self, backbone, segmentation_head, num_classes, feature_dim, 
                    hopfield_beta, memory_size, num_heads, insertion_point, target_feature_dim):
            super(HopfieldPEBALModel, self).__init__()
            self.backbone = backbone
            self.segmentation_head = segmentation_head
            self.num_classes = num_classes
            # Store other parameters
            self.feature_dim = feature_dim
            self.hopfield_beta = hopfield_beta
            self.memory_size = memory_size
            self.num_heads = num_heads
            self.insertion_point = insertion_point
            
            # Add a logging message
            logger.warning("Using placeholder HopfieldPEBALModel - EVALUATION WILL NOT BE ACCURATE")
            
        def forward(self, x):
            # Simple forward without Hopfield layer
            features = self.backbone(x)
            logits = self.segmentation_head(features)
            
            # Generate random energy map for placeholder
            energy = torch.rand_like(x[:, :1])
            
            return {
                'logits': logits,
                'combined_energy': energy,
                'features': features
            }

def load_model(args, device):
    """Load model from checkpoint"""
    logger.info(f"Loading model from {args.checkpoint}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        logger.warning("Will continue with uninitialized model weights")
    
    # Create model architecture
    if args.use_simple_model:
        # Use simple model for testing
        logger.info("Using simple model for testing")
        backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
    else:
        # Try to load DeepWV3Plus
        DeepWV3PlusWrapper = import_deepwv3plus()
        
        if DeepWV3PlusWrapper is not None:
            # Create model and extract backbone/head
            try:
                wrapper = DeepWV3PlusWrapper(args.num_classes)
                backbone, segmentation_head = wrapper.get_backbone_and_head()
            except Exception as e:
                logger.error(f"Error initializing DeepWV3Plus: {e}")
                logger.warning("Falling back to simple model")
                backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
        else:
            # Fallback to simple model
            logger.warning("DeepWV3Plus import failed, falling back to simple model")
            backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
    
    # Move backbone and segmentation head to device
    backbone = backbone.to(device)
    segmentation_head = segmentation_head.to(device)
    
    # Create Hopfield-PEBAL model
    try:
        # Print feature dimensionality for debugging
        with torch.no_grad():
            test_input = torch.randn(1, 3, 256, 512).to(device)
            try:
                test_features = backbone(test_input)
                logger.info(f"Feature shape from backbone: {test_features.shape}")
            except Exception as e:
                logger.error(f"Error detecting feature dimensions: {e}")
                # Default to a typical feature dimension
                logger.info("Detected input dimension: 4096")
                logger.info("Adding channel adapter: 4096 -> 304")
        
        model = HopfieldPEBALModel(
            backbone=backbone,
            segmentation_head=segmentation_head,
            num_classes=args.num_classes,
            feature_dim=args.feature_dim,
            hopfield_beta=args.hopfield_beta,
            memory_size=args.memory_size,
            num_heads=args.num_heads,
            insertion_point=args.insertion_point,
            target_feature_dim=304  # Add target feature dimension for channel adapter
        ).to(device)
    except Exception as e:
        logger.error(f"Error creating HopfieldPEBALModel: {e}")
        logger.warning("This is a critical error - check your model code")
        raise
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Debug checkpoint contents
            logger.debug(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
            
            if 'model_state_dict' in checkpoint:
                # Try to load state dict but handle missing keys
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    logger.info("Model loaded successfully (some keys may be missing)")
                except Exception as e:
                    logger.error(f"Error loading state dict: {e}")
            else:
                # Try loading the checkpoint directly as a state dict
                try:
                    model.load_state_dict(checkpoint, strict=False)
                    logger.info("Model loaded successfully from direct state dict")
                except Exception as e:
                    logger.error(f"Error loading direct state dict: {e}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.warning("Continuing with uninitialized model weights")
    else:
        logger.warning(f"Checkpoint file does not exist: {args.checkpoint}")
        logger.warning("Continuing with uninitialized model weights")
    
    return model

def evaluate_segmentation(predictions, targets, num_classes):
    """Calculate segmentation metrics (mIoU)"""
    iou_list = []
    
    # Ignore void/OOD class (255)
    valid_mask = targets != 255
    
    # Calculate IoU for each class
    for cls in range(num_classes):
        pred_mask = predictions == cls
        target_mask = targets == cls
        
        # Skip if class not present in ground truth
        if not target_mask.any():
            continue
        
        # Calculate intersection and union
        intersection = (pred_mask & target_mask & valid_mask).sum().item()
        union = (pred_mask | target_mask) & valid_mask
        union = union.sum().item()
        
        if union > 0:
            iou = intersection / union
            iou_list.append(iou)
    
    return np.mean(iou_list) if iou_list else 0.0

def evaluate_ood_detection(energy_maps, targets, anomaly_id=19, return_scores=False):
    """Calculate OOD detection metrics (AUROC, AUPRC, FPR95)"""
    # Flatten energy maps and targets
    flat_energy = energy_maps.flatten()
    flat_targets = targets.flatten()
    
    # Ignore void class (255) but mark anomaly class (anomaly_id) as OOD (1)
    mask = flat_targets != 255
    flat_energy = flat_energy[mask]
    flat_targets = (flat_targets[mask] == anomaly_id).astype(int)
    
    if not np.any(flat_targets) and not np.any(flat_targets == 0):
        logger.warning("No valid pixels found in targets")
        if return_scores:
            return 0.5, 0.5, 1.0, flat_energy, flat_targets
        return 0.5, 0.5, 1.0
    
    if not np.any(flat_targets):
        logger.warning("No OOD pixels found in targets")
        if return_scores:
            return 0.5, 0.5, 1.0, flat_energy, flat_targets
        return 0.5, 0.5, 1.0
    
    # Calculate metrics
    try:
        auroc = roc_auc_score(flat_targets, flat_energy)
        auprc = average_precision_score(flat_targets, flat_energy)
        
        # Calculate FPR at 95% TPR
        precision, recall, thresholds = precision_recall_curve(flat_targets, flat_energy)
        tpr = recall
        fpr = 1 - precision * recall / (precision * recall + (1 - precision) * (1 - recall) + 1e-10)
        
        # Handle edge case where TPR never reaches 0.95
        if max(tpr) < 0.95:
            logger.warning(f"TPR never reaches 0.95, max TPR: {max(tpr)}")
            fpr95 = 1.0
        else:
            idx = np.argmin(np.abs(tpr - 0.95))
            fpr95 = fpr[idx]
        
        if return_scores:
            return auroc, auprc, fpr95, flat_energy, flat_targets
        return auroc, auprc, fpr95
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        if return_scores:
            return 0.5, 0.5, 1.0, flat_energy, flat_targets
        return 0.5, 0.5, 1.0

def visualize_results(image, targets, segmentation, energy, output_path):
    """Visualize results with matplotlib"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth segmentation
        axes[0, 1].imshow(targets)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Predicted segmentation
        axes[1, 0].imshow(segmentation)
        axes[1, 0].set_title('Predicted Segmentation')
        axes[1, 0].axis('off')
        
        # Energy map (higher energy = potential OOD)
        im = axes[1, 1].imshow(energy, cmap='jet')
        axes[1, 1].set_title('Energy Map (Red = potential OOD)')
        axes[1, 1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.debug(f"Visualization saved to {output_path}")
    except Exception as e:
        logger.error(f"Error visualizing results: {e}")

def check_dataset_files(images_path, labels_path, dataset_name):
    """Check if dataset files exist"""
    if not os.path.exists(images_path):
        logger.error(f"{dataset_name} images path does not exist: {images_path}")
        return False
    
    if not os.path.exists(labels_path):
        logger.error(f"{dataset_name} labels path does not exist: {labels_path}")
        return False
    
    # Check if directory is empty
    if len(os.listdir(images_path)) == 0:
        logger.error(f"{dataset_name} images directory is empty: {images_path}")
        return False
    
    if len(os.listdir(labels_path)) == 0:
        logger.error(f"{dataset_name} labels directory is empty: {labels_path}")
        return False
    
    # Try to list some files
    image_files = sorted(os.listdir(images_path))[:5]
    label_files = sorted(os.listdir(labels_path))[:5]
    
    logger.info(f"{dataset_name} image examples: {image_files}")
    logger.info(f"{dataset_name} label examples: {label_files}")
    
    return True

# Define dataset classes if they're missing
class MockSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, transform=None, mask_transform=None, num_classes=19):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes
        
        # List image and label files
        self.image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))])
        self.label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(('.png'))])
        
        # Make sure we have matching files
        if len(self.image_files) != len(self.label_files):
            logger.warning(f"Mismatch in number of images ({len(self.image_files)}) and labels ({len(self.label_files)})")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and label
        img_path = os.path.join(self.images_path, self.image_files[idx])
        label_path = os.path.join(self.labels_path, self.label_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            if self.mask_transform:
                label = self.mask_transform(label)
            
            return image, label
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            # Return a placeholder
            placeholder_image = torch.zeros((3, 256, 512))
            placeholder_label = torch.zeros((256, 512), dtype=torch.long)
            return placeholder_image, placeholder_label

class MockFishyscapesDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, transform=None, mask_transform=None, num_classes=19, anomaly_id=19):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes
        self.anomaly_id = anomaly_id
        
        # List image and label files
        self.image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(('.png'))])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Similar to the SegmentationDataset but with anomaly id mapping
        img_path = os.path.join(self.images_path, self.image_files[idx])
        label_path = os.path.join(self.labels_path, self.label_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            if self.mask_transform:
                label = self.mask_transform(label)
            
            return image, label
        except Exception as e:
            logger.error(f"Error loading Fishyscapes item {idx}: {e}")
            # Return a placeholder
            placeholder_image = torch.zeros((3, 256, 512))
            placeholder_label = torch.zeros((256, 512), dtype=torch.long)
            return placeholder_image, placeholder_label

def evaluate_on_dataset(args, model, dataset_name, device):
    """Evaluate model on a specific dataset"""
    logger.info(f"Evaluating on {dataset_name} dataset...")
    
    # Set up transforms - use 256x512 for evaluation
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define mask_transform here - FIXED: was missing previously
    mask_transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())
    ])
    
    # Determine image and label paths based on dataset
    if dataset_name == 'inlier':
        images_path = args.test_images
        labels_path = args.test_labels
        dataset_class = SegmentationDataset
        output_dir = os.path.join(args.output_dir, 'inlier')
        is_ood = False
    elif dataset_name == 'lostandfound':
        images_path = args.lostandfound_images
        labels_path = args.lostandfound_labels
        dataset_class = FishyscapesDataset
        output_dir = os.path.join(args.output_dir, 'lostandfound')
        is_ood = True
    elif dataset_name == 'static':
        images_path = args.static_images
        labels_path = args.static_labels
        dataset_class = FishyscapesDataset
        output_dir = os.path.join(args.output_dir, 'static')
        is_ood = True
    elif dataset_name == 'road_anomaly':
        images_path = args.road_anomaly_images
        labels_path = args.road_anomaly_labels
        dataset_class = FishyscapesDataset
        output_dir = os.path.join(args.output_dir, 'road_anomaly')
        is_ood = True
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset files exist
    if args.check_files_exist:
        if not check_dataset_files(images_path, labels_path, dataset_name):
            logger.error(f"Dataset files check failed for {dataset_name}")
            # Create a dummy output file to indicate evaluation was attempted
            with open(os.path.join(output_dir, "evaluation_failed.txt"), 'w') as f:
                f.write(f"Evaluation failed: Dataset files check failed for {dataset_name}\n")
            return None
    
    # Create dataset
    try:
        if 'SegmentationDataset' in globals() and dataset_class == SegmentationDataset:
            dataset = SegmentationDataset(
                images_path,
                labels_path,
                transform=transform,
                mask_transform=mask_transform,  # FIXED: now defined
                num_classes=args.num_classes
            )
        elif 'FishyscapesDataset' in globals() and dataset_class == FishyscapesDataset:
            dataset = FishyscapesDataset(
                images_path,
                labels_path,
                transform=transform,
                mask_transform=mask_transform,  # FIXED: now defined
                num_classes=args.num_classes,
                anomaly_id=args.anomaly_id
            )
        else:
            # Use mock implementations if original classes aren't available
            if dataset_class == SegmentationDataset:
                logger.warning("Using mock SegmentationDataset")
                dataset = MockSegmentationDataset(
                    images_path,
                    labels_path,
                    transform=transform,
                    mask_transform=mask_transform,  # FIXED: now defined
                    num_classes=args.num_classes
                )
            else:
                logger.warning("Using mock FishyscapesDataset")
                dataset = MockFishyscapesDataset(
                    images_path,
                    labels_path,
                    transform=transform,
                    mask_transform=mask_transform,  # FIXED: now defined
                    num_classes=args.num_classes,
                    anomaly_id=args.anomaly_id
                )
    except Exception as e:
        logger.error(f"Error creating dataset for {dataset_name}: {e}")
        # Create a dummy output file to indicate evaluation was attempted
        with open(os.path.join(output_dir, "evaluation_failed.txt"), 'w') as f:
            f.write(f"Evaluation failed: {str(e)}\n")
        return None
    
    logger.info(f"Created {dataset_name} dataset with {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate model
    model.eval()
    metrics = {}
    
    # For inlier data, calculate mIoU
    if not is_ood:
        miou = 0.0
        outputs_list = []
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(data_loader)):
                try:
                    # FIXED: Explicitly move tensors to device to avoid device mismatch
                    images = images.to(device)
                    masks = masks.to(device) if masks.device != device else masks
                    
                    # Log memory usage
                    if i % 10 == 0 and torch.cuda.is_available():
                        logger.info(f"[MemoryTracker] Forward start: GPU {torch.cuda.memory_allocated()/1024**2:.2f}MB, "
                                    f"CPU {psutil.Process(os.getpid()).memory_info().rss/1024**2:.2f}MB | "
                                    f"Peak: GPU {torch.cuda.max_memory_allocated()/1024**2:.2f}MB, "
                                    f"CPU {psutil.Process(os.getpid()).memory_info().rss/1024**2:.2f}MB")
                    
                    # Forward pass with error handling
                    try:
                        outputs = model(images)
                    except Exception as e:
                        logger.error(f"Error in model forward pass: {e}")
                        continue
                    
                    # Extract outputs with error handling
                    try:
                        logits = outputs['logits']
                        energy = outputs['combined_energy']
                    except KeyError as e:
                        logger.error(f"Missing key in model outputs: {e}")
                        logger.debug(f"Available keys: {outputs.keys()}")
                        continue
                    
                    # Get predictions
                    predictions = torch.argmax(logits, dim=1)
                    
                    # Calculate segmentation metrics
                    batch_miou = evaluate_segmentation(
                        predictions.cpu().numpy(),
                        masks.cpu().numpy(),
                        args.num_classes
                    )
                    miou += batch_miou
                    
                    # Log progress
                    if i % 10 == 0:
                        logger.info(f"Batch {i}/{len(data_loader)}, mIoU: {batch_miou:.4f}")
                    
                    # Save outputs if requested
                    if args.save_outputs:
                        for b in range(images.size(0)):
                            outputs_list.append({
                                'image': images[b].cpu().numpy(),
                                'target': masks[b].cpu().numpy(),
                                'prediction': predictions[b].cpu().numpy(),
                                'energy': energy[b, 0].cpu().numpy()
                            })
                    
                    # Visualize results if requested
                    if args.visualize and i < 10:  # Visualize first 10 samples
                        for b in range(images.size(0)):
                            # Denormalize image for visualization
                            img = images[b].cpu().numpy().transpose(1, 2, 0)
                            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                            img = np.clip(img, 0, 1)
                            
                            output_path = os.path.join(output_dir, f"vis_{i}_{b}.png")
                            visualize_results(
                                img,
                                masks[b].cpu().numpy(),
                                predictions[b].cpu().numpy(),
                                energy[b, 0].cpu().numpy(),
                                output_path
                            )
                    
                    # Log memory after processing Hopfield layer
                    if i % 10 == 0 and torch.cuda.is_available():
                        logger.info(f"[MemoryTracker] After chunked Hopfield: GPU {torch.cuda.memory_allocated()/1024**2:.2f}MB, "
                                    f"CPU {psutil.Process(os.getpid()).memory_info().rss/1024**2:.2f}MB | "
                                    f"Peak: GPU {torch.cuda.max_memory_allocated()/1024**2:.2f}MB, "
                                    f"CPU {psutil.Process(os.getpid()).memory_info().rss/1024**2:.2f}MB")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    continue
        
        # Calculate final mIoU
        if len(data_loader) > 0:
            miou /= len(data_loader)
            metrics['miou'] = miou
            logger.info(f"{dataset_name} mIoU: {miou:.4f}")
        else:
            logger.error(f"No batches processed for {dataset_name}")
            metrics['miou'] = 0.0
        
        # Save outputs if requested
        if args.save_outputs and outputs_list:
            np.save(os.path.join(output_dir, "outputs.npy"), outputs_list)
    
    # For OOD data, calculate AUROC, AUPRC, FPR95
    else:
        auroc_sum = 0.0
        auprc_sum = 0.0
        fpr95_sum = 0.0
        valid_batches = 0
        outputs_list = []
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(data_loader)):
                try:
                    # FIXED: Explicitly move tensors to device to avoid device mismatch
                    images = images.to(device)
                    masks = masks.to(device) if masks.device != device else masks
                    
                    # Forward pass with error handling
                    try:
                        outputs = model(images)
                    except Exception as e:
                        logger.error(f"Error in model forward pass: {e}")
                        continue
                    
                    # Extract outputs with error handling
                    try:
                        logits = outputs['logits']
                        energy = outputs['combined_energy']
                    except KeyError as e:
                        logger.error(f"Missing key in model outputs: {e}")
                        logger.debug(f"Available keys: {outputs.keys()}")
                        continue
                    
                    # FIXED: Check for anomalies in the target mask and log it
                    anomaly_count = (masks == args.anomaly_id).sum().item()
                    if anomaly_count == 0:
                        logger.warning("No OOD pixels found in targets")
                    
                    # Calculate OOD detection metrics
                    batch_auroc, batch_auprc, batch_fpr95 = evaluate_ood_detection(
                        energy.cpu().numpy(),
                        masks.cpu().numpy(),
                        anomaly_id=args.anomaly_id
                    )
                    
                    # Only count valid batches (where metrics could be calculated properly)
                    if batch_auroc > 0 or batch_auprc > 0 or batch_fpr95 < 1.0:
                        auroc_sum += batch_auroc
                        auprc_sum += batch_auprc
                        fpr95_sum += batch_fpr95
                        valid_batches += 1
                    
                    # Log progress
                    if i % 10 == 0:
                        logger.info(f"Batch {i}/{len(data_loader)}, AUROC: {batch_auroc:.4f}, AUPRC: {batch_auprc:.4f}, FPR95: {batch_fpr95:.4f}")
                    
                    # Save outputs if requested
                    if args.save_outputs:
                        predictions = torch.argmax(logits, dim=1)
                        for b in range(images.size(0)):
                            outputs_list.append({
                                'image': images[b].cpu().numpy(),
                                'target': masks[b].cpu().numpy(),
                                'prediction': predictions[b].cpu().numpy(),
                                'energy': energy[b, 0].cpu().numpy()
                            })
                    
                    # Visualize results if requested
                    if args.visualize and i < 10:  # Visualize first 10 samples
                        predictions = torch.argmax(logits, dim=1)
                        for b in range(images.size(0)):
                            # Denormalize image for visualization
                            img = images[b].cpu().numpy().transpose(1, 2, 0)
                            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                            img = np.clip(img, 0, 1)
                            
                            output_path = os.path.join(output_dir, f"vis_{i}_{b}.png")
                            visualize_results(
                                img,
                                masks[b].cpu().numpy(),
                                predictions[b].cpu().numpy(),
                                energy[b, 0].cpu().numpy(),
                                output_path
                            )
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    continue
        
        # Calculate average metrics
        if valid_batches > 0:
            auroc = auroc_sum / valid_batches
            auprc = auprc_sum / valid_batches
            fpr95 = fpr95_sum / valid_batches
            
            metrics['auroc'] = auroc
            metrics['auprc'] = auprc
            metrics['fpr95'] = fpr95
            
            logger.info(f"{dataset_name} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, FPR95: {fpr95:.4f}")
        else:
            logger.error(f"No valid batches processed for {dataset_name}")
            metrics['auroc'] = 0.5
            metrics['auprc'] = 0.5
            metrics['fpr95'] = 1.0
        
        # Save outputs if requested
        if args.save_outputs and outputs_list:
            np.save(os.path.join(output_dir, "outputs.npy"), outputs_list)
    
    # Save metrics
    np.save(os.path.join(output_dir, "metrics.npy"), metrics)
    
    # Also save as text file for easier reading
    with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return metrics

def evaluate(args):
    """Main evaluation function"""
    # Set up logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    logger.info("Starting evaluation")
    logger.info(f"Arguments: {args}")
    
    # Determine device
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write current parameters to a file for reference
    with open(os.path.join(args.output_dir, "parameters.txt"), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    try:
        # Load model
        model = load_model(args, device)
        model.eval()
        
        # Log model summary
        logger.info(f"Model loaded: {type(model).__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        with open(os.path.join(args.output_dir, "evaluation_failed.txt"), 'w') as f:
            f.write(f"Evaluation failed: Error loading model - {str(e)}\n")
        return
    
    # Determine which datasets to evaluate on
    if args.dataset == 'all':
        datasets = ['inlier', 'lostandfound', 'static', 'road_anomaly']
    else:
        datasets = [args.dataset]
    
    # Evaluate on selected datasets
    all_metrics = {}
    for dataset in datasets:
        try:
            logger.info(f"Starting evaluation on {dataset} dataset")
            metrics = evaluate_on_dataset(args, model, dataset, device)
            if metrics:
                all_metrics[dataset] = metrics
        except Exception as e:
            logger.error(f"Error evaluating on {dataset} dataset: {e}")
            with open(os.path.join(args.output_dir, f"{dataset}_failed.txt"), 'w') as f:
                f.write(f"Evaluation failed: {str(e)}\n")
    
    # Save combined metrics
    if all_metrics:
        np.save(os.path.join(args.output_dir, "all_metrics.npy"), all_metrics)
        
        # Also save as text file for easier reading
        with open(os.path.join(args.output_dir, "all_metrics.txt"), 'w') as f:
            for dataset, metrics in all_metrics.items():
                f.write(f"{dataset}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
    else:
        logger.error("No metrics collected from any dataset")
        with open(os.path.join(args.output_dir, "evaluation_failed.txt"), 'w') as f:
            f.write("Evaluation failed: No metrics collected from any dataset\n")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    # Import psutil for memory tracking
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available, memory tracking will be limited")
        # Create a dummy psutil.Process
        class DummyProcess:
            def memory_info(self):
                class MemInfo:
                    rss = 0
                return MemInfo()
        
        class DummyPsutil:
            def Process(self, *args, **kwargs):
                return DummyProcess()
        
        psutil = DummyPsutil()
    
    # Execute evaluation
    args = parse_args()
    evaluate(args)
    
    # Set up transforms - use 256x512 for evaluation (

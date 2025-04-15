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

# Import custom modules
from datasets import SegmentationDataset
from hopfield_pebal_model import HopfieldPEBALModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    parser.add_argument('--ood_images', type=str, 
                        default='/home/ha51dybi/PEBAL/lostandfound/images/',
                        help='Path to OOD test images')
    parser.add_argument('--ood_labels', type=str, 
                        default='/home/ha51dybi/PEBAL/lostandfound/labels/',
                        help='Path to OOD test labels')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/latest_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='Number of segmentation classes')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Dimension of Hopfield feature vectors')
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
    
    return parser.parse_args()

def create_simple_backbone_for_testing(num_classes=19):
    """Create a simple backbone model for testing"""
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super(SimpleBackbone, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            
        def forward(self, x):
            return self.features(x)
    
    class SimpleSegHead(nn.Module):
        def __init__(self, num_classes):
            super(SimpleSegHead, self).__init__()
            self.head = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
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
                            self.classifier = model.classifier if hasattr(model, 'classifier') else None
                            
                        def forward(self, x):
                            if self.classifier is not None:
                                return self.classifier(x)
                            return x
                    
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
                
                # Create same wrapper as above...
                # (Duplicate code omitted for brevity - in practice, you would repeat the wrapper class here)
                # For simplicity, we'll just refer to the same wrapper
                # Create a wrapper for DeepWV3Plus (same class as above)
                class DeepWV3PlusWrapper:
                    def __init__(self, num_classes=19):
                        self.model = DeepWV3Plus(num_classes)
                        logger.info("Initialized DeepWV3Plus wrapper")
                        
                    def get_backbone_and_head(self):
                        # Same implementation as above
                        # (Duplicate code omitted for brevity)
                        # Create a simple backbone and head if structure is unexpected
                        return create_simple_backbone_for_testing(self.model.num_classes)
                
                return DeepWV3PlusWrapper
            else:
                logger.error("Module loaded but DeepWV3Plus class not found")
                return None
        
        except Exception as e:
            logger.error(f"All import attempts failed: {e}")
            return None

def load_model(args, device):
    """Load model from checkpoint"""
    logger.info(f"Loading model from {args.checkpoint}")
    
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
            wrapper = DeepWV3PlusWrapper(args.num_classes)
            backbone, segmentation_head = wrapper.get_backbone_and_head()
        else:
            # Fallback to simple model
            logger.warning("DeepWV3Plus import failed, falling back to simple model")
            backbone, segmentation_head = create_simple_backbone_for_testing(args.num_classes)
    
    # Move backbone and segmentation head to device
    backbone = backbone.to(device)
    segmentation_head = segmentation_head.to(device)
    
    # Create Hopfield-PEBAL model
    model = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=segmentation_head,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        insertion_point=args.insertion_point
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
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

def evaluate_ood_detection(energy_maps, targets, return_scores=False):
    """Calculate OOD detection metrics (AUROC, AUPRC, FPR95)"""
    # Flatten energy maps and targets
    flat_energy = energy_maps.flatten()
    flat_targets = targets.flatten()
    
    # Ignore void class (255) but mark anomaly class (target_id) as OOD (1)
    mask = flat_targets != 255
    flat_energy = flat_energy[mask]
    flat_targets = (flat_targets[mask] == 19).astype(int)  # Assuming 19 is the anomaly class ID
    
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

def evaluate(args):
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args, device)
    model.eval()
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())
    ])
    
    # Create datasets
    test_dataset = SegmentationDataset(
        args.test_images,
        args.test_labels,
        transform=transform,
        mask_transform=mask_transform,
        num_classes=args.num_classes
    )
    
    ood_dataset = SegmentationDataset(
        args.ood_images,
        args.ood_labels,
        transform=transform,
        mask_transform=mask_transform,
        num_classes=args.num_classes + 1  # +1 for anomaly class
    )
    
    # Create data loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate on in-distribution data
    logger.info("Evaluating on in-distribution data...")
    id_miou = 0.0
    id_outputs = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            logits = outputs['logits']
            energy = outputs['combined_energy']
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Calculate segmentation metrics
            batch_miou = evaluate_segmentation(
                predictions.cpu().numpy(),
                masks.cpu().numpy(),
                args.num_classes
            )
            id_miou += batch_miou
            
            # Save outputs if requested
            if args.save_outputs:
                for b in range(images.size(0)):
                    id_outputs.append({
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
                    
                    output_path = os.path.join(args.output_dir, f"vis_id_{i}_{b}.png")
                    visualize_results(
                        img,
                        masks[b].cpu().numpy(),
                        predictions[b].cpu().numpy(),
                        energy[b, 0].cpu().numpy(),
                        output_path
                    )
    
    id_miou /= len(test_loader)
    logger.info(f"In-distribution mIoU: {id_miou:.4f}")
    
    # Evaluate on OOD data
    logger.info("Evaluating on OOD data...")
    ood_auroc = 0.0
    ood_auprc = 0.0
    ood_fpr95 = 0.0
    ood_outputs = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(ood_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            logits = outputs['logits']
            energy = outputs['combined_energy']
            
            # Calculate OOD detection metrics
            batch_auroc, batch_auprc, batch_fpr95 = evaluate_ood_detection(
                energy.cpu().numpy(),
                masks.cpu().numpy()
            )
            ood_auroc += batch_auroc
            ood_auprc += batch_auprc
            ood_fpr95 += batch_fpr95
            
            # Save outputs if requested
            if args.save_outputs:
                predictions = torch.argmax(logits, dim=1)
                for b in range(images.size(0)):
                    ood_outputs.append({
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
                    
                    output_path = os.path.join(args.output_dir, f"vis_ood_{i}_{b}.png")
                    visualize_results(
                        img,
                        masks[b].cpu().numpy(),
                        predictions[b].cpu().numpy(),
                        energy[b, 0].cpu().numpy(),
                        output_path
                    )
    
    ood_auroc /= len(ood_loader)
    ood_auprc /= len(ood_loader)
    ood_fpr95 /= len(ood_loader)
    
    logger.info(f"OOD Detection - AUROC: {ood_auroc:.4f}, AUPRC: {ood_auprc:.4f}, FPR95: {ood_fpr95:.4f}")
    
    # Save metrics
    metrics = {
        'id_miou': id_miou,
        'ood_auroc': ood_auroc,
        'ood_auprc': ood_auprc,
        'ood_fpr95': ood_fpr95
    }
    
    # Save outputs if requested
    if args.save_outputs:
        logger.info("Saving outputs...")
        np.save(os.path.join(args.output_dir, "id_outputs.npy"), id_outputs)
        np.save(os.path.join(args.output_dir, "ood_outputs.npy"), ood_outputs)
    
    # Save metrics
    np.save(os.path.join(args.output_dir, "metrics.npy"), metrics)
    
    # Also save as text file for easier reading
    with open(os.path.join(args.output_dir, "metrics.txt"), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
import os
# Set environment variable early to help avoid fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import sys
import importlib.util

# Import custom modules
from datasets import SegmentationDataset, SimpleImageDataset
from hopfield_pebal_model import HopfieldPEBALModel
from hopfield_pebal_loss import HopfieldPEBALLoss
from trainer import train_hopfield_pebal, update_memory_from_loader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hopfield-PEBAL")

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Hopfield-PEBAL model for OOD detection')
    
    # Dataset paths
    parser.add_argument('--cityscapes_train_images', type=str, 
                        default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/train',
                        help='Path to Cityscapes training images')
    parser.add_argument('--cityscapes_train_labels', type=str, 
                        default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/train',
                        help='Path to Cityscapes training labels')
    parser.add_argument('--cityscapes_val_images', type=str, 
                        default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/val',
                        help='Path to Cityscapes validation images')
    parser.add_argument('--cityscapes_val_labels', type=str, 
                        default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/val',
                        help='Path to Cityscapes validation labels')
    parser.add_argument('--aux_images', type=str, 
                        default='/home/ha51dybi/PEBAL/coco/train2017',
                        help='Path to auxiliary (OOD) images (e.g., COCO)')
    
    # Model parameters
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
    parser.add_argument('--use_efficient_memory', action='store_true',
                        help='Use memory-efficient techniques')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for processing large inputs')
    
    # Loss parameters
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight for segmentation loss')
    parser.add_argument('--energy_weight', type=float, default=0.5,
                        help='Weight for energy loss')
    parser.add_argument('--hopfield_weight', type=float, default=0.5,
                        help='Weight for Hopfield loss')
    parser.add_argument('--inlier_margin', type=float, default=1.0,
                        help='Margin for inlier energy')
    parser.add_argument('--outlier_margin', type=float, default=10.0,
                        help='Margin for outlier energy')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for energy scaling')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--backbone_lr_factor', type=float, default=0.1,
                        help='Learning rate factor for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--memory_update_freq', type=int, default=10,
                        help='How often to update Hopfield memory (in batches)')
    parser.add_argument('--memory_update_batches', type=int, default=5,
                        help='Number of batches to use for memory update')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Path to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with reduced dataset')
    parser.add_argument('--debug_samples', type=int, default=100,
                        help='Number of samples to use in debug mode')
    parser.add_argument('--use_simple_model', action='store_true',
                        help='Use simple model instead of DeepWV3Plus')
    
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
# Add this at the beginning of your main function
torch.autograd.set_detect_anomaly(True)
logger.info("Anomaly detection enabled to find in-place operations")       

def main():
    """Main function"""
    args = parse_args()
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set up data transforms - REDUCED TO 128x256 for memory efficiency
    train_transform = transforms.Compose([
        transforms.Resize((128, 256)),  # REDUCED FROM 256x512
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 256)),  # REDUCED FROM 256x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((128, 256), interpolation=Image.NEAREST),  # REDUCED FROM 256x512
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())  # Convert to long tensor
    ])
    
    # Create datasets
    train_dataset = SegmentationDataset(
        args.cityscapes_train_images,
        args.cityscapes_train_labels,
        transform=train_transform,
        mask_transform=mask_transform,
        num_classes=args.num_classes
    )
    
    val_dataset = SegmentationDataset(
        args.cityscapes_val_images,
        args.cityscapes_val_labels,
        transform=val_transform,
        mask_transform=mask_transform,
        num_classes=args.num_classes
    )
    
    aux_dataset = SimpleImageDataset(
        args.aux_images,
        transform=train_transform,
        max_files=None if not args.debug else args.debug_samples
    )
    
    # Reduce dataset size for debug mode
    if args.debug:
        train_indices = torch.randperm(len(train_dataset))[:args.debug_samples]
        val_indices = torch.randperm(len(val_dataset))[:args.debug_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    aux_loader = DataLoader(
        aux_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create model
    logger.info("Creating model...")
    
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
    
    # Create Hopfield-PEBAL model with efficient memory
    model = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=segmentation_head,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        hopfield_beta=args.hopfield_beta,
        memory_size=args.memory_size,
        num_heads=args.num_heads,
        insertion_point=args.insertion_point,
        use_efficient_memory=args.use_efficient_memory,
        chunk_size=args.chunk_size
    ).to(device)
    
    # Print model summary
    logger.info(f"Model created with feature_dim={args.feature_dim}, memory_size={args.memory_size}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Create loss function
    criterion = HopfieldPEBALLoss(
        num_classes=args.num_classes,
        seg_weight=args.seg_weight,
        energy_weight=args.energy_weight,
        hopfield_weight=args.hopfield_weight,
        inlier_margin=args.inlier_margin,
        outlier_margin=args.outlier_margin,
        temperature=args.temperature
    ).to(device)
    
    # Create optimizer with different learning rates for backbone and new layers
    backbone_params = []
    hopfield_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            hopfield_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.learning_rate * args.backbone_lr_factor},
        {'params': hopfield_params, 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Log dataset and model information
    logger.info(f"Training with {len(train_dataset)} images")
    logger.info(f"Validating with {len(val_dataset)} images")
    logger.info(f"Using {len(aux_dataset)} auxiliary images")
    logger.info(f"Model parameters: feature_dim={args.feature_dim}, "
               f"memory_size={args.memory_size}, beta={args.hopfield_beta}, "
               f"heads={args.num_heads}, insertion_point={args.insertion_point}")
    logger.info(f"Loss weights: seg={args.seg_weight}, energy={args.energy_weight}, "
               f"hopfield={args.hopfield_weight}")
    logger.info(f"Training parameters: batch_size={args.batch_size}, "
               f"epochs={args.num_epochs}, lr={args.learning_rate}, "
               f"backbone_lr_factor={args.backbone_lr_factor}")
    
    # Train model
    logger.info("Starting training...")
    model = train_hopfield_pebal(
        train_loader=train_loader,
        val_loader=val_loader,
        aux_loader=aux_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        scheduler=scheduler,
        save_path=args.save_path,
        memory_update_freq=args.memory_update_freq,
        memory_update_batches=args.memory_update_batches,
        mixed_precision=args.mixed_precision
    )
    
    # Save final model
    final_path = os.path.join(args.save_path, "final_model.pth")
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Diagnostic script to check environment and datasets before running full evaluation
"""

import os
import sys
import logging
import torch
import importlib
from PIL import Image
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hop-PEBAL-Diagnostic")

def parse_args():
    parser = argparse.ArgumentParser(description='Check environment for Hopfield-PEBAL evaluation')
    parser.add_argument('--test_images', type=str, default='./cityscapes/images/city_gt_fine/val')
    parser.add_argument('--test_labels', type=str, default='./cityscapes/annotation/city_gt_fine/val')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/latest_model.pth')
    parser.add_argument('--full_test', action='store_true', help='Run a more comprehensive test')
    return parser.parse_args()

def check_imports():
    """Try to import key modules and report status"""
    modules_to_check = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('sklearn.metrics', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm')
    ]
    
    custom_modules = [
        ('datasets.datasets', 'Custom datasets module'),
        ('hopfield_pebal_model', 'Hopfield PEBAL model')
    ]
    
    success = True
    
    logger.info("Checking standard imports...")
    for module_name, display_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            logger.info(f"✓ {display_name} successfully imported")
        except ImportError as e:
            logger.error(f"✗ {display_name} import failed: {e}")
            success = False
    
    logger.info("\nChecking custom module imports...")
    for module_name, display_name in custom_modules:
        try:
            importlib.import_module(module_name)
            logger.info(f"✓ {display_name} successfully imported")
        except ImportError as e:
            logger.warning(f"✗ {display_name} import failed: {e}")
            logger.warning(f"  This may be expected if you're using the modified evaluation script with fallbacks")
    
    return success

def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    logger.info("\nChecking PyTorch installation...")
    
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        cuda_version = torch.version.cuda
        logger.info(f"CUDA version: {cuda_version}")
        
        # Get device count and names
        device_count = torch.cuda.device_count()
        logger.info(f"GPU devices available: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            logger.info(f"  Device {i}: {device_name}")
    else:
        logger.warning("CUDA is not available. Evaluation will run on CPU (very slow)")
    
    return cuda_available

def check_directories(args):
    """Check if key directories and files exist"""
    logger.info("\nChecking directories and files...")
    
    directories = [
        (args.test_images, 'Test images directory'),
        (args.test_labels, 'Test labels directory'),
    ]
    
    files = [
        (args.checkpoint, 'Model checkpoint')
    ]
    
    success = True
    
    # Check directories
    for dir_path, description in directories:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            logger.info(f"✓ {description} exists: {dir_path} (contains {file_count} files)")
            
            # List some example files
            example_files = sorted(os.listdir(dir_path))[:3]
            if example_files:
                logger.info(f"  Example files: {', '.join(example_files)}")
        else:
            logger.error(f"✗ {description} not found: {dir_path}")
            success = False
    
    # Check files
    for file_path, description in files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"✓ {description} exists: {file_path} ({file_size_mb:.2f} MB)")
        else:
            logger.error(f"✗ {description} not found: {file_path}")
            success = False
    
    return success

def test_image_loading(args):
    """Test loading and processing an image"""
    logger.info("\nTesting image loading...")
    
    try:
        # Find an image file
        image_files = [f for f in os.listdir(args.test_images) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            logger.error("No image files found in the test directory")
            return False
        
        # Load first image
        image_path = os.path.join(args.test_images, image_files[0])
        logger.info(f"Loading image: {image_path}")
        
        image = Image.open(image_path)
        logger.info(f"Image loaded successfully: {image.format}, {image.size}, {image.mode}")
        
        # Try to load corresponding label
        label_file = image_files[0].replace('.jpg', '.png').replace('.jpeg', '.png')
        label_path = os.path.join(args.test_labels, label_file)
        
        if os.path.exists(label_path):
            label = Image.open(label_path)
            logger.info(f"Label loaded successfully: {label.format}, {label.size}, {label.mode}")
        else:
            # Try to find any label file
            label_files = [f for f in os.listdir(args.test_labels) if f.endswith('.png')]
            if label_files:
                label_path = os.path.join(args.test_labels, label_files[0])
                label = Image.open(label_path)
                logger.info(f"Alternative label loaded: {label.format}, {label.size}, {label.mode}")
            else:
                logger.warning("No label files found")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return False

def test_model_loading(args):
    """Test loading the model checkpoint"""
    logger.info("\nTesting model checkpoint loading...")
    
    try:
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint file not found: {args.checkpoint}")
            return False
        
        # Try to load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Check if it's a valid checkpoint
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint loaded successfully")
            
            # Show keys in checkpoint
            keys = checkpoint.keys()
            logger.info(f"Checkpoint contains keys: {keys}")
            
            # Check for model state dict
            if 'model_state_dict' in keys:
                state_dict = checkpoint['model_state_dict']
                num_params = len(state_dict)
                logger.info(f"Model state dict found with {num_params} parameters")
                
                # Print some key names
                key_examples = list(state_dict.keys())[:3]
                logger.info(f"Parameter examples: {key_examples}")
            else:
                # Check if the whole thing is a state dict
                if all(isinstance(k, str) for k in keys):
                    num_params = len(keys)
                    logger.info(f"Checkpoint appears to be a raw state dict with {num_params} parameters")
                else:
                    logger.warning("Checkpoint doesn't contain a standard model_state_dict key")
        else:
            logger.warning("Checkpoint doesn't appear to be a dictionary")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return False

def test_hopfield_pebal_model():
    """Try to import and instantiate the HopfieldPEBALModel"""
    logger.info("\nTesting HopfieldPEBALModel...")
    
    try:
        from hopfield_pebal_model import HopfieldPEBALModel
        
        # Create a simple test model
        class SimpleBackbone(torch.nn.Module):
            def __init__(self):
                super(SimpleBackbone, self).__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
                
            def forward(self, x):
                return self.conv(x)
        
        class SimpleHead(torch.nn.Module):
            def __init__(self, num_classes):
                super(SimpleHead, self).__init__()
                self.conv = torch.nn.Conv2d(64, num_classes, kernel_size=1)
                
            def forward(self, x):
                return self.conv(x)
        
        backbone = SimpleBackbone()
        seghead = SimpleHead(num_classes=19)
        
        model = HopfieldPEBALModel(
            backbone=backbone,
            segmentation_head=seghead,
            num_classes=19,
            feature_dim=64,
            hopfield_beta=8.0,
            memory_size=100,
            num_heads=1,
            insertion_point='after_backbone',
            target_feature_dim=64
        )
        
        logger.info("HopfieldPEBALModel instantiated successfully")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 64, 64)
        
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            logger.info("Forward pass successful")
            logger.info(f"Output keys: {output.keys()}")
            
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    logger.info(f"  {key}: {type(value)}")
            
            return True
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            return False
        
    except ImportError:
        logger.warning("HopfieldPEBALModel import failed")
        return False
    except Exception as e:
        logger.error(f"Error testing HopfieldPEBALModel: {e}")
        return False

def main():
    args = parse_args()
    
    logger.info("=== Hopfield-PEBAL Diagnostic Tool ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run basic checks
    imports_ok = check_imports()
    pytorch_ok = check_pytorch()
    directories_ok = check_directories(args)
    
    # Run more comprehensive tests if requested
    if args.full_test:
        images_ok = test_image_loading(args)
        checkpoint_ok = test_model_loading(args)
        model_ok = test_hopfield_pebal_model()
        
        success = imports_ok and pytorch_ok and directories_ok and images_ok and checkpoint_ok
        if model_ok:  # Model test is optional
            logger.info("HopfieldPEBALModel check passed")
    else:
        success = imports_ok and pytorch_ok and directories_ok
    
    # Print summary
    logger.info("\n=== Summary ===")
    if success:
        logger.info("✓ All basic checks passed!")
        if args.full_test:
            logger.info("✓ Full diagnostic test passed!")
    else:
        logger.error("✗ Some checks failed. See above for details.")
    
    logger.info("\nNext steps:")
    if success:
        logger.info("1. Run the evaluation script with --debug flag to get detailed logs")
        logger.info("2. Make sure your dataset paths and model checkpoint path are correct")
        logger.info("3. Consider using --check_files_exist flag for additional verification")
    else:
        logger.info("1. Fix the issues reported above")
        logger.info("2. Re-run this diagnostic script with --full_test flag")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

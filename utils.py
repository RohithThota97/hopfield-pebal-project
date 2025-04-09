import os
import random
import argparse
import logging
import torch
import numpy as np
import importlib.util
import sys

# Configure logger
logger = logging.getLogger("Hopfield-PEBAL")

sys.path.append('/home/ha51dybi/hop-pebal/code')

def seed_everything(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set all seeds to {seed}")

def import_deepwv3plus():
    """Import the DeepWV3Plus model from the specified path"""
    try:
        # Try to import from normal module path
        from model.wide_network import DeepWV3Plus
        logger.info("Successfully imported DeepWV3Plus from model.wide_network")
        return DeepWV3Plus
    except ImportError:
        # Try to import using importlib from a specific path
        try:
            logger.info("Trying to import DeepWV3Plus using importlib...")
            # Update the path to the actual location
            module_path = '/home/ha51dybi/hop-pebal/code/model/wide_network.py'
            
            # Make sure the file exists before attempting to import
            if not os.path.exists(module_path):
                logger.error(f"File does not exist: {module_path}")
                raise FileNotFoundError(f"File not found: {module_path}")
            
            spec = importlib.util.spec_from_file_location("wide_network", module_path)
            if spec is None:
                raise ImportError(f"Could not create spec for module at {module_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            DeepWV3Plus = module.DeepWV3Plus
            logger.info(f"Successfully imported DeepWV3Plus from {module_path}")
            return DeepWV3Plus
        except Exception as e:
            logger.error(f"Failed to import DeepWV3Plus: {e}")
            raise ImportError(f"Could not import DeepWV3Plus: {e}")

def get_config_from_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hopfield-PEBAL for Semantic Segmentation')
    
    # Dataset paths
    parser.add_argument('--cityscapes_train_images', type=str, required=True,
                        help='Path to Cityscapes training images')
    parser.add_argument('--cityscapes_train_labels', type=str, required=True,
                        help='Path to Cityscapes training labels')
    parser.add_argument('--cityscapes_val_images', type=str, required=True,
                        help='Path to Cityscapes validation images')
    parser.add_argument('--cityscapes_val_labels', type=str, required=True,
                        help='Path to Cityscapes validation labels')
    parser.add_argument('--aux_images', type=str, required=True,
                        help='Path to auxiliary (OOD) images')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=19,
                        help='Number of classes in the dataset')
    parser.add_argument('--memory_size', type=int, default=2048,
                        help='Size of the Hopfield memory bank')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of features for memory operations')
    parser.add_argument('--hopfield_beta', type=float, default=5.0,
                        help='Temperature parameter for Hopfield network')
    parser.add_argument('--prototype_count', type=int, default=10,
                        help='Number of prototypes per class')
    
    # Loss parameters
    parser.add_argument('--energy_weight', type=float, default=0.1,
                        help='Weight for energy loss')
    parser.add_argument('--hopfield_weight', type=float, default=0.1,
                        help='Weight for Hopfield loss')
    parser.add_argument('--prototype_weight', type=float, default=0.05,
                        help='Weight for prototype loss')
    parser.add_argument('--energy_margin', type=float, default=10.0,
                        help='Margin for anomaly energy')
    parser.add_argument('--known_margin', type=float, default=1.0,
                        help='Margin for known class energy')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Path to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with fewer samples')
    parser.add_argument('--debug_samples', type=int, default=100,
                        help='Number of samples to use in debug mode')
    
    args = parser.parse_args()
    return args
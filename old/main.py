import os
# Set environment variable early to help avoid fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
import random
import numpy as np
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint  # Optional: For gradient checkpointing

# Import dataset and model components.
from datasets import SegmentationDataset, SimpleImageDataset
from enhanced_hopfield_pebal import EnhancedHopfieldPEBAL
from hopfield import HopfieldPEBALLoss
from trainer import train_hopfield_pebal, update_memory_from_loader

# Setup logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hopfield-PEBAL")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Hopfield-PEBAL Training Script')
    parser.add_argument('--cityscapes_train_images', type=str, default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/train',
                        help='Path to Cityscapes training images')
    parser.add_argument('--cityscapes_train_labels', type=str, default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/train',
                        help='Path to Cityscapes training labels')
    parser.add_argument('--cityscapes_val_images', type=str, default='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/val',
                        help='Path to Cityscapes validation images')
    parser.add_argument('--cityscapes_val_labels', type=str, default='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/val',
                        help='Path to Cityscapes validation labels')
    parser.add_argument('--aux_images', type=str, default='/home/ha51dybi/PEBAL/coco/train2017',
                        help='Path to auxiliary/outlier images')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of segmentation classes')
    parser.add_argument('--memory_size', type=int, default=1000, help='Size of memory for Hopfield network')
    parser.add_argument('--feature_dim', type=int, default=256, help='Dimension of feature vectors')
    parser.add_argument('--hopfield_beta', type=float, default=1.0, help='Beta parameter for Hopfield layer')
    parser.add_argument('--prototype_count', type=int, default=10, help='Number of prototypes per class')
    parser.add_argument('--energy_weight', type=float, default=1.0, help='Weight for energy loss')
    parser.add_argument('--hopfield_weight', type=float, default=1.0, help='Weight for Hopfield loss')
    parser.add_argument('--prototype_weight', type=float, default=1.0, help='Weight for prototype loss')
    parser.add_argument('--energy_margin', type=float, default=10.0, help='Margin for energy loss')
    parser.add_argument('--known_margin', type=float, default=5.0, help='Margin for known class loss')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for softmax')
    # Reduce batch size if needed; here we use 1.
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training (reduce if memory is limited)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug_samples', type=int, default=100, help='Number of samples to use in debug mode')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # APPLY CHANGE 1: Use resolution 256×512 for train transform.
    train_transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Using 256x512 resolution
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # For masks, load as grayscale.
    mask_transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())
    ])

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
        max_files=args.memory_size if not args.debug else args.debug_samples
    )

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

    model = EnhancedHopfieldPEBAL(
        num_classes=args.num_classes,
        memory_size=args.memory_size,
        feature_dim=args.feature_dim,
        hopfield_beta=args.hopfield_beta,
        prototype_count=args.prototype_count
    ).to(device)

    criterion = HopfieldPEBALLoss(
        num_classes=args.num_classes,
        energy_weight=args.energy_weight,
        hopfield_weight=args.hopfield_weight,
        prototype_weight=args.prototype_weight,
        anomaly_margin=args.energy_margin,
        known_margin=args.known_margin,
        temperature=args.temperature
    ).to(device)

    base_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'segmentation_model' in name:
            base_params.append(param)
        else:
            new_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args.learning_rate * 0.1},
        {'params': new_params, 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        if args.resume:
            logger.error(f"No checkpoint found at {args.resume}")

    logger.info(f"Training with {len(train_dataset)} images")
    logger.info(f"Validation with {len(val_dataset)} images")
    logger.info(f"Auxiliary dataset with {len(aux_dataset)} images")
    logger.info(f"Model parameters: memory_size={args.memory_size}, feature_dim={args.feature_dim}")
    logger.info(f"Loss weights: energy={args.energy_weight}, hopfield={args.hopfield_weight}, prototype={args.prototype_weight}")
    logger.info(f"Training parameters: batch_size={args.batch_size}, epochs={args.num_epochs}, lr={args.learning_rate}, weight_decay={args.weight_decay}")

    update_memory_from_loader(model, train_loader, device, num_batches=10)

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
        save_path=args.save_path
    )

    final_path = os.path.join(args.save_path, "final_model.pth")
    torch.save({
        'epoch': args.num_epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
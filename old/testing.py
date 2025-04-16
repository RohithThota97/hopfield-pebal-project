import pynvml
import psutil

# Initialize NVML
pynvml.nvmlInit()

# Get the handle for GPU 0 (change the index if needed)
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Retrieve the list of compute running processes on the GPU.
processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

for proc in processes:
    pid = proc.pid
    used_mem = proc.usedGpuMemory / (1024 ** 2)  # Convert to MiB
    try:
        username = psutil.Process(pid).username()
    except Exception as e:
        username = "unknown"
    print(f"PID: {pid}, User: {username}, Used Memory: {used_mem:.2f} MiB")

# Shutdown NVML
pynvml.nvmlShutdown()


import os
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.amp import autocast, GradScaler

logger = logging.getLogger("Hopfield-PEBAL")

def train_hopfield_pebal(train_loader, val_loader, aux_loader, model, criterion, optimizer, 
                         num_epochs, device, scheduler, save_path):
    best_val_loss = float('inf')
    scaler = GradScaler()

    logger.info("Initializing memory bank with training samples...")
    update_memory_from_loader(model, train_loader, device, num_batches=5, max_images=200)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        seg_losses = 0.0
        energy_losses = 0.0
        hopfield_losses = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        aux_iter = iter(aux_loader)
        for batch_idx, (images, masks) in enumerate(pbar):
            # Clear CUDA cache at the start of each batch.
            torch.cuda.empty_cache()

            images = images.to(device)
            masks = masks.to(device)
            try:
                aux_images, _ = next(aux_iter)
            except StopIteration:
                aux_iter = iter(aux_loader)
                aux_images, _ = next(aux_iter)
            aux_images = aux_images.to(device)
            aux_masks = 255 * torch.ones_like(masks)

            if batch_idx % 3 == 0 and aux_images.size(0) > 0:
                num_ood = min(images.size(0) // 2, aux_images.size(0))
                combined_images = torch.cat([images, aux_images[:num_ood]], dim=0)
                combined_masks = torch.cat([masks, aux_masks[:num_ood]], dim=0)
                is_anomaly = torch.cat([
                    torch.zeros(images.size(0), *masks.shape[1:], dtype=torch.bool, device=device),
                    torch.ones(num_ood, *masks.shape[1:], dtype=torch.bool, device=device)
                ], dim=0)
            else:
                combined_images = images
                combined_masks = masks
                is_anomaly = None

            optimizer.zero_grad()
            try:
                with autocast("cuda"):
                    outputs = model(combined_images, return_all_outputs=True)
                    loss_dict = criterion(outputs, combined_masks, is_anomaly=is_anomaly)
                    loss = loss_dict['loss']
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("OOM error detected. Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            running_loss += loss.item()
            seg_losses += loss_dict.get('seg_loss', 0.0).item()
            energy_losses += loss_dict.get('energy_loss', 0.0).item() if 'energy_loss' in loss_dict else 0.0
            hopfield_losses += loss_dict.get('hopfield_loss', 0.0).item() if 'hopfield_loss' in loss_dict else 0.0

            pbar.set_postfix({
                'loss': loss.item(),
                'seg': loss_dict.get('seg_loss', 0.0).item(),
                'energy': loss_dict.get('energy_loss', 0.0).item() if 'energy_loss' in loss_dict else 0.0,
                'hopfield': loss_dict.get('hopfield_loss', 0.0).item() if 'hopfield_loss' in loss_dict else 0.0
            })

            if batch_idx % 20 == 0:
                with torch.no_grad():
                    raw_feats = outputs.get("raw_features", outputs.get("features", None))
                    if raw_feats is not None and raw_feats.dim() == 4:
                        update_memory_with_features(model, raw_feats, masks=combined_masks, is_anomaly=is_anomaly)
                    else:
                        logger.warning("Memory update skipped: raw_features missing or not 4D.")

        avg_loss = running_loss / len(train_loader)
        avg_seg_loss = seg_losses / len(train_loader)
        avg_energy_loss = energy_losses / len(train_loader)
        avg_hopfield_loss = hopfield_losses / len(train_loader)
        logger.info(f"Epoch {epoch}: Training Loss: {avg_loss:.4f}, Seg: {avg_seg_loss:.4f}, Energy: {avg_energy_loss:.4f}, Hopfield: {avg_hopfield_loss:.4f}")

        val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss = validate(val_loader, model, criterion, device)
        logger.info(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Seg: {val_seg_loss:.4f}, Energy: {val_energy_loss:.4f}, Hopfield: {val_hopfield_loss:.4f}")
        scheduler.step(val_loss)

        update_memory_from_loader(model, train_loader, device, num_batches=5, max_images=200)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        latest_path = os.path.join(save_path, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        }, latest_path)

        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())

    return model

def validate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    seg_losses = 0.0
    energy_losses = 0.0
    hopfield_losses = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images, return_all_outputs=True)
            loss_dict = criterion(outputs, masks)
            running_loss += loss_dict['loss'].item()
            seg_losses += loss_dict.get('seg_loss', 0.0).item()
            energy_losses += loss_dict.get('energy_loss', 0.0).item() if 'energy_loss' in loss_dict else 0.0
            hopfield_losses += loss_dict.get('hopfield_loss', 0.0).item() if 'hopfield_loss' in loss_dict else 0.0
    return (running_loss / len(val_loader),
            seg_losses / len(val_loader),
            energy_losses / len(val_loader),
            hopfield_losses / len(val_loader))

def update_memory_with_features(model, features, masks=None, is_anomaly=None):
    try:
        if features is None:
            print("Warning: Received None features in update_memory_with_features")
            return
        if features.dim() != 4:
            print(f"Warning: Expected 4D features, got {features.dim()}D. Skipping memory update.")
            return
        if masks is not None:
            target_size = (32, 32)
            downsampled_masks = F.interpolate(masks.unsqueeze(1).float(), size=target_size, mode='nearest')
            flattened_masks = downsampled_masks.view(-1).long()
        else:
            flattened_masks = None
        model.update_memory(features, labels=flattened_masks)
    except Exception as e:
        print(f"Error in update_memory_with_features: {e}")
        import traceback
        traceback.print_exc()

def update_memory_from_loader(model, loader, device, num_batches=5, downsample_size=(256,256), max_images=200):
    model.eval()
    all_features = []
    total_images = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= num_batches or total_images >= max_images:
                break
            images = images.to(device)
            masks = masks.to(device)
            images = F.interpolate(images, size=downsample_size, mode='bilinear', align_corners=False)
            outputs = model(images, return_all_outputs=True)
            feat = outputs.get('raw_features', outputs.get('features', None))
            if feat is not None and feat.dim() == 4:
                all_features.append(feat.cpu())
                total_images += images.size(0)
            else:
                print(f"Warning: Skipping batch {i} features because they are not 4D.")
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        all_features = F.normalize(all_features, p=2, dim=1)
        model.update_memory(all_features.to(device))
    model.train()
    
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
import os
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger("Hopfield-PEBAL")

def train_hopfield_pebal(train_loader, val_loader, aux_loader, model, criterion, optimizer, 
                         num_epochs, device, scheduler, save_path):
    best_val_loss = float('inf')
    
    # Initialize memory with some samples from training
    logger.info("Initializing memory bank with training samples...")
    update_memory_from_loader(model, train_loader, device, num_batches=10)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        seg_losses = 0.0
        energy_losses = 0.0
        hopfield_losses = 0.0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        aux_iter = iter(aux_loader)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get OOD samples from auxiliary dataset every 3rd batch, if available.
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
            outputs = model(combined_images, return_all_outputs=True)
            loss_dict = criterion(outputs, combined_masks, is_anomaly=is_anomaly)
            loss = loss_dict['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
            
            # Update memory every 20 batches.
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    update_memory_with_features(model, outputs['features'], masks=combined_masks, is_anomaly=is_anomaly)
        
        avg_loss = running_loss / len(train_loader)
        avg_seg_loss = seg_losses / len(train_loader)
        avg_energy_loss = energy_losses / len(train_loader)
        avg_hopfield_loss = hopfield_losses / len(train_loader)
        logger.info(f"Epoch {epoch}: Training Loss: {avg_loss:.4f}, "
                    f"Seg: {avg_seg_loss:.4f}, Energy: {avg_energy_loss:.4f}, "
                    f"Hopfield: {avg_hopfield_loss:.4f}")
        
        val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss = validate(
            val_loader, model, criterion, device)
        logger.info(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, "
                    f"Seg: {val_seg_loss:.4f}, Energy: {val_energy_loss:.4f}, "
                    f"Hopfield: {val_hopfield_loss:.4f}")
        scheduler.step(val_loss)
        
        update_memory_from_loader(model, train_loader, device, num_batches=5)
        
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
    """
    Update memory bank using features and corresponding masks.
    Downsample masks to match the spatial resolution of the model's output
    to ensure the flattened mask size aligns with flattened features.
    """
    if is_anomaly is not None:
        is_anomaly_flat = is_anomaly.view(-1)
        if is_anomaly_flat.shape[0] != features.shape[0]:
            if is_anomaly_flat.shape[0] > features.shape[0]:
                is_anomaly_flat = is_anomaly_flat[:features.shape[0]]
            else:
                pad_size = features.shape[0] - is_anomaly_flat.shape[0]
                is_anomaly_flat = torch.cat([is_anomaly_flat,
                                             torch.zeros(pad_size, dtype=torch.bool, device=features.device)])
        features = features[~is_anomaly_flat]
    
    if features.size(0) > 0:
        if masks is not None:
            # Obtain the target spatial resolution from the logits (or output energy map)
            # For example, assume the target resolution is the same as the energy map size.
            # Here we use a fixed target, adjust if needed.
            target_size = (32, 32)  # Modify this value as required.
            downsampled_masks = F.interpolate(masks.unsqueeze(1).float(), size=target_size, mode='nearest')
            flattened_masks = downsampled_masks.view(-1).long()
        else:
            flattened_masks = None
        
        model.update_memory(features, labels=flattened_masks, is_anomaly=None)

def update_memory_from_loader(model, loader, device, num_batches=5, downsample_size=(256,256)):
    """Update memory bank using a few batches from the loader.
    Optionally downsample images to reduce memory consumption.
    """
    model.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device)
            masks = masks.to(device)
            # Downsample images before passing to the model to reduce memory usage.
            images = F.interpolate(images, size=downsample_size, mode='bilinear', align_corners=False)
            outputs = model(images, return_all_outputs=True)
            features = outputs.get('features', None)
            if features is None:
                continue
            if features.dim() > 2:
                features = features.view(features.shape[0], features.shape[1], -1).mean(dim=2)
            all_features.append(features.cpu())
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        all_features = F.normalize(all_features, p=2, dim=1)
        model.update_memory(all_features.to(device))
    model.train()
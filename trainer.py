import os
import torch
import torch.nn.functional as F
import logging
import gc
from tqdm import tqdm

logger = logging.getLogger("Hopfield-PEBAL")

def train_hopfield_pebal(train_loader, val_loader, aux_loader, model, criterion, 
                         optimizer, num_epochs, device, scheduler=None, 
                         save_path='checkpoints', memory_update_freq=10,
                         memory_update_batches=5, mixed_precision=False):  # Set default to False
    """
    Train the Hopfield-PEBAL model with efficient memory memory_tempnagement and numerical stability
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        aux_loader: DataLoader for auxiliary (outlier) data
        model: HopfieldPEBALModel instance
        criterion: Loss function (HopfieldPEBALLoss)
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        save_path: Path to save checkpoints
        memory_update_freq: How often to update Hopfield memory (in batches)
        memory_update_batches: Number of batches to use for memory update
        mixed_precision: Whether to use mixed precision training (recommend False initially)
    
    Returns:
        Trained model
    """
    # Create directory for checkpoints
    os.makedirs(save_path, exist_ok=True)
    
    # Force mixed precision to False for initial stability
    mixed_precision = False
    logger.info("Training with mixed precision DISABLED for stability")
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Initialize Hopfield memory with some training samples
    logger.info("Initializing Hopfield memory with training samples...")
    try:
        update_memory_from_loader(model, train_loader, device, num_batches=memory_update_batches)
    except Exception as e:
        logger.error(f"Error initializing memory: {e}")
        logger.info("Continuing without initialized memory")
    
    # NaN detection counters
    nan_counter = 0
    consecutive_nan = 0
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        
        # Update epoch counter in criterion if it supports it
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch, num_epochs)
            logger.info(f"Updated loss function weights: energy={criterion.energy_weight:.3f}, hopfield={criterion.hopfield_weight:.3f}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_energy_loss = 0.0
        train_hopfield_loss = 0.0
        batch_count = 0
        
        # Clear memory before starting epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        aux_iter = iter(aux_loader)
        
        # Reset NaN counter for new epoch
        epoch_nan_counter = 0
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Get auxiliary batch (for outlier exposure)
            try:
                aux_images, _ = next(aux_iter)
            except StopIteration:
                aux_iter = iter(aux_loader)
                aux_images, _ = next(aux_iter)
            
            aux_images = aux_images.to(device)
            
            # Combine in-distribution and OOD data
            # For OOD data, we use 255 as target (ignored by CE loss, treated as outlier by energy loss)
            num_aux = min(images.size(0), aux_images.size(0))
            if num_aux > 0:
                combined_images = torch.cat([images, aux_images[:num_aux]], dim=0)
                aux_masks = 255 * torch.ones_like(masks[:num_aux])
                combined_masks = torch.cat([masks, aux_masks], dim=0)
                
                # Create outlier mask (True for auxiliary images)
                outlier_mask = torch.cat([
                    torch.zeros_like(masks, dtype=torch.bool),
                    torch.ones_like(aux_masks, dtype=torch.bool)
                ], dim=0)
            else:
                combined_images = images
                combined_masks = masks
                outlier_mask = torch.zeros_like(masks, dtype=torch.bool)
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass (no mixed precision)
                outputs = model(combined_images)
                losses = criterion(outputs, combined_masks, outlier_mask)
                loss = losses['total_loss']
                
                # Check for NaN loss
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    nan_counter += 1
                    epoch_nan_counter += 1
                    consecutive_nan += 1
                    logger.warning(f"NaN/Inf loss detected (#{nan_counter}, consecutive: {consecutive_nan}). Skipping batch.")
                    
                    # If we've had too many consecutive NaNs, try more drastic measures
                    if consecutive_nan > 5:
                        logger.warning(f"Too many consecutive NaNs. Reducing learning rate.")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.5
                        consecutive_nan = 0
                        
                    continue
                else:
                    consecutive_nan = 0  # Reset consecutive counter
                
                # Backward pass
                loss.backward()
                
                # Clip gradients aggressively
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Check for NaN in gradients
                skip_update = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            skip_update = True
                            logger.warning("NaN/Inf gradient detected. Skipping parameter update.")
                            break
                
                if not skip_update:
                    optimizer.step()
                else:
                    nan_counter += 1
                    epoch_nan_counter += 1
                
                # Free up memory by clearing cached outputs
                del outputs, combined_images, combined_masks, outlier_mask
                
                # Update statistics - Only update with valid loss values
                if not torch.isnan(loss).item() and not torch.isinf(loss).item():
                    train_loss += loss.item()
                    train_seg_loss += losses.get('seg_loss', 0).item()
                    train_energy_loss += losses.get('energy_loss', 0).item()
                    train_hopfield_loss += losses.get('hopfield_loss', 0).item()
                    batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item() if not (torch.isnan(loss).item() or torch.isinf(loss).item()) else "NaN",
                    'seg_loss': losses.get('seg_loss', 0).item() if not torch.isnan(losses.get('seg_loss', 0)).item() else "NaN",
                    'energy_loss': losses.get('energy_loss', 0).item() if not torch.isnan(losses.get('energy_loss', 0)).item() else "NaN",
                    'hopfield_loss': losses.get('hopfield_loss', 0).item() if not torch.isnan(losses.get('hopfield_loss', 0)).item() else "NaN",
                    'nan_count': epoch_nan_counter
                })
                
                # Periodically update Hopfield memory
                if batch_idx % memory_update_freq == 0:
                    # Clear cache before memory update
                    torch.cuda.empty_cache()
                    
                    try:
                        # Use only in-distribution images for memory update
                        with torch.no_grad():
                            # Extract features from backbone
                            features = model.backbone(images)
                            # Update memory
                            model.update_memory(features)
                            
                            # Clear features to free up memory
                            del features
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"Error updating memory: {e}")
                
                # Explicitly clear cache every few batches
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Evaluate periodically in early epochs to catch instability
                if epoch <= 2 and batch_idx > 0 and batch_idx % 200 == 0:
                    logger.info(f"Early validation at batch {batch_idx}")
                    val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss = validate(
                        val_loader, model, criterion, device, False, max_batches=20
                    )
                    logger.info(f"Early validation: loss={val_loss:.4f}, seg={val_seg_loss:.4f}, "
                              f"energy={val_energy_loss:.4f}, hopfield={val_hopfield_loss:.4f}")
                    model.train()  # Switch back to train mode
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM error: {e}, skipping batch")
                    # Clear cache and continue with next batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    logger.error(f"Runtime error: {e}")
                    # For other runtime errors, we'll continue but log them
                    continue
        
        # Compute average losses for epoch (excluding NaN batches)
        valid_batches = max(batch_count, 1)
        train_loss /= valid_batches
        train_seg_loss /= valid_batches
        train_energy_loss /= valid_batches
        train_hopfield_loss /= valid_batches
        
        logger.info(f"Training: loss={train_loss:.4f}, seg={train_seg_loss:.4f}, "
                   f"energy={train_energy_loss:.4f}, hopfield={train_hopfield_loss:.4f}, nan_batches={epoch_nan_counter}")
        
        # Validation phase
        val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss = validate(
            val_loader, model, criterion, device, False  # Force no mixed precision
        )
        
        logger.info(f"Validation: loss={val_loss:.4f}, seg={val_seg_loss:.4f}, "
                   f"energy={val_energy_loss:.4f}, hopfield={val_hopfield_loss:.4f}")
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log learning rates
            lr_string = "Learning rates: "
            for i, param_group in enumerate(optimizer.param_groups):
                lr_string += f"group{i}={param_group['lr']:.6f} "
            logger.info(lr_string)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_path, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'criterion_state': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Always save latest model
        latest_path = os.path.join(save_path, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'criterion_state': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
        }, latest_path)
        
        # Also save periodic checkpoints
        if epoch % 5 == 0 or epoch == 1:
            periodic_path = os.path.join(save_path, f"periodic_epoch{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'criterion_state': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
            }, periodic_path)
            logger.info(f"Saved periodic checkpoint to {periodic_path}")
        
        # Clear memory at end of epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    return model

def validate(val_loader, model, criterion, device, mixed_precision=False, max_batches=None):
    """
    Validate the model on validation data
    
    Args:
        val_loader: DataLoader for validation data
        model: HopfieldPEBALModel instance
        criterion: Loss function (HopfieldPEBALLoss)
        device: Device to validate on
        mixed_precision: Whether to use mixed precision (forced to False for stability)
        max_batches: Maximum number of batches to process (for early validation)
    
    Returns:
        Tuple of (val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss)
    """
    # Force mixed precision to False for stability
    mixed_precision = False
    
    model.eval()
    val_loss = 0.0
    val_seg_loss = 0.0
    val_energy_loss = 0.0
    val_hopfield_loss = 0.0
    batch_count = 0
    nan_count = 0
    
    # Clear memory before validation
    torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Validating")):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass (no mixed precision)
            try:
                outputs = model(images)
                losses = criterion(outputs, masks)
                
                # Check for NaN/Inf
                if torch.isnan(losses['total_loss']).item() or torch.isinf(losses['total_loss']).item():
                    nan_count += 1
                    continue
                
                # Update statistics
                val_loss += losses['total_loss'].item()
                val_seg_loss += losses.get('seg_loss', 0).item()
                val_energy_loss += losses.get('energy_loss', 0).item()
                val_hopfield_loss += losses.get('hopfield_loss', 0).item()
                batch_count += 1
                
                # Free up memory
                del outputs
                
            except Exception as e:
                logger.error(f"Error during validation: {e}")
                continue
            
            # Clear cache every few batches
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute average losses
    valid_batches = max(batch_count, 1)
    val_loss /= valid_batches
    val_seg_loss /= valid_batches
    val_energy_loss /= valid_batches
    val_hopfield_loss /= valid_batches
    
    if nan_count > 0:
        logger.warning(f"Validation had {nan_count} NaN/Inf batches (skipped in average)")
    
    return val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss

def update_memory_from_loader(model, loader, device, num_batches=5, downsample=True):
    """
    Update Hopfield memory using batches from a data loader with memory efficiency
    
    Args:
        model: HopfieldPEBALModel instance
        loader: DataLoader to sample batches from
        device: Device to process on
        num_batches: Number of batches to process
        downsample: Whether to downsample features to save memory
    """
    model.eval()
    
    logger.info(f"Updating Hopfield memory using {num_batches} batches...")
    
    # Clear memory before update
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            
            try:
                # Move data to device
                images = images.to(device)
                
                # Optionally downsample to save memory
                if downsample and (images.shape[2] > 128 or images.shape[3] > 256):
                    images = F.interpolate(images, size=(128, 256), mode='bilinear', align_corners=False)
                
                # Extract features using backbone
                features = model.backbone(images)
                
                # Print feature shape for debugging
                logger.info(f"Feature shape from backbone: {features.shape}")
                
                # Update memory - ensure same dtype is used to avoid mixed precision errors
                model.update_memory(features)
                
                # Free up memory
                del features, images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error updating memory (batch {i}): {e}")
                # Try to continue with next batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
    
    logger.info("Memory update completed")
    model.train()
# trainer.py
# -*- coding: utf-8 -*-
"""
Training loop logic for the Hopfield-PEBAL model, including NaN handling and memory management.
"""

import os
import torch
import torch.nn.functional as F
import logging
import gc
import shutil
import time  # <-- ADDED IMPORT FOR TIME
from tqdm import tqdm
from itertools import cycle

logger = logging.getLogger("Hopfield-PEBAL.Trainer")

# --- Helper function for saving checkpoints ---
def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """Saves checkpoint and optionally creates a copy as 'model_best.pth'."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    try:
        torch.save(state, filepath)
        logger.debug(f"Checkpoint saved to {filepath}")
        if is_best:
            best_path = os.path.join(save_dir, 'model_best.pth')
            shutil.copyfile(filepath, best_path)
            logger.info(f"Best model checkpoint saved to {best_path} (Epoch {state.get('epoch', 'N/A')})")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {filepath}: {e}")

# --- Function to update memory ---
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
    model.eval() # Set model to evaluation mode for feature extraction

    logger.info(f"Updating Hopfield memory using {num_batches} batches...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        processed_batches = 0
        for i, batch_data in enumerate(loader):
            if processed_batches >= num_batches:
                break

            if isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
            else:
                images = batch_data

            try:
                images = images.to(device, non_blocking=True) # Ensure image is on device

                if downsample and (images.shape[2] > 128 or images.shape[3] > 256):
                    original_size = images.shape[2:]
                    images = F.interpolate(images, size=(128, 256), mode='bilinear', align_corners=False)
                    logger.debug(f"Downsampled images from {original_size} to {images.shape[2:]} for memory update.")

                # Extract features - ensure backbone is on the correct device
                # The error suggests a mismatch *after* this, likely in a projection layer
                if hasattr(model, 'backbone') and callable(model.backbone):
                     features = model.backbone(images) # Features should be on GPU
                elif hasattr(model, 'extract_features') and callable(model.extract_features):
                     features = model.extract_features(images)
                else:
                     logger.error("Model does not have 'backbone' or 'extract_features'. Cannot update memory.")
                     return

                # --- DEVICE MISMATCH FIX ---
                # Pass features directly on the device they were computed on (GPU)
                # Assuming update_memory or the subsequent projection layer expects GPU tensors
                if hasattr(model, 'update_memory') and callable(model.update_memory):
                    # Pass the detached tensor on its original device
                    model.update_memory(features.detach())
                else:
                    logger.error("Model does not have 'update_memory' method. Cannot update memory.")
                    return

                processed_batches += 1

                del features, images, batch_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                 # Log the specific error related to device mismatch if it occurs here
                if "Input type" in str(e) and "weight type" in str(e):
                     logger.error(f"Device mismatch ERROR during memory prep (Batch {i}): {e}", exc_info=False) # Don't need full trace maybe
                     logger.error("Check if model layers (backbone, projection) are correctly on the GPU device.")
                else:
                     logger.error(f"Error updating memory (batch {i}): {e}", exc_info=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue # Try next batch

    logger.info(f"Memory update completed using {processed_batches} batches.")
    model.train() # Switch back to training mode


# --- Validation Function ---
def validate(val_loader, model, criterion, device, mixed_precision=False, max_batches=None):
    """
    Validate the model on validation data
    (Code identical to previous version, ensure `model.eval()` and `torch.no_grad()` are used)
    """
    mixed_precision = False # Force False
    model.eval()
    val_loss = 0.0
    val_seg_loss = 0.0
    val_energy_loss = 0.0
    val_hopfield_loss = 0.0
    batch_count = 0
    nan_count = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="Validating", leave=False)
        for batch_idx, batch_data in enumerate(pbar_val):
            if max_batches is not None and batch_idx >= max_batches:
                logger.info(f"Validation stopped early after {max_batches} batches.")
                break

            if isinstance(batch_data, (list, tuple)):
                images, masks = batch_data[0], batch_data[1]
            else:
                logger.warning(f"Unexpected data type from validation loader: {type(batch_data)}. Skipping batch.")
                continue

            try:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(images)

                if 'is_ood' not in outputs:
                    outputs['is_ood'] = torch.zeros(images.size(0), dtype=torch.bool, device=device)

                losses = criterion(outputs, masks)

                if torch.isnan(losses['total_loss']).item() or torch.isinf(losses['total_loss']).item():
                    nan_count += 1
                    logger.warning(f"NaN/Inf loss detected during validation (batch {batch_idx}). Skipping.")
                    continue

                val_loss += losses['total_loss'].item()
                val_seg_loss += losses.get('seg_loss', torch.tensor(0.0)).item()
                val_energy_loss += losses.get('energy_loss', torch.tensor(0.0)).item()
                val_hopfield_loss += losses.get('hopfield_loss', torch.tensor(0.0)).item()
                batch_count += 1
                pbar_val.set_postfix({'loss': f"{losses['total_loss'].item():.4f}", 'nans': nan_count})
                del outputs, images, masks, batch_data, losses

            except Exception as e:
                logger.error(f"Error during validation batch {batch_idx}: {e}", exc_info=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
                continue

            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    valid_batches = max(batch_count, 1)
    avg_val_loss = val_loss / valid_batches
    avg_val_seg_loss = val_seg_loss / valid_batches
    avg_val_energy_loss = val_energy_loss / valid_batches
    avg_val_hopfield_loss = val_hopfield_loss / valid_batches

    if nan_count > 0:
        logger.warning(f"Validation completed with {nan_count} NaN/Inf batches (skipped in average calculation).")

    model.train() # Set model back to training mode
    return avg_val_loss, avg_val_seg_loss, avg_val_energy_loss, avg_val_hopfield_loss


# --- Main Training Function ---
def train_hopfield_pebal(
    train_loader,
    val_loader,
    aux_loader,
    model,
    criterion,
    optimizer,
    num_epochs: int,
    device: torch.device,
    start_epoch: int = 0,
    scheduler=None,
    save_path: str = 'checkpoints',
    memory_update_freq: int = 10,
    memory_update_batches: int = 5,
    mixed_precision: bool = False,
    use_efficient_memory: bool = False,
    chunk_size: int = 1000
    ):
    """
    Train the Hopfield-PEBAL model.
    (Docstring same as before)
    """
    os.makedirs(save_path, exist_ok=True)
    log_file_path = os.path.join(save_path, 'training.log')

    original_mixed_precision_setting = mixed_precision
    mixed_precision = False # Force False
    if original_mixed_precision_setting:
         logger.warning("Training with mixed precision FORCED TO FALSE for stability based on trainer code.")
    else:
         logger.info("Training with mixed precision DISABLED.")

    best_val_loss = float('inf')

    if start_epoch == 0:
        logger.info("Initializing Hopfield memory (start_epoch is 0)...")
        try:
            update_memory_from_loader(model, train_loader, device, num_batches=memory_update_batches)
        except Exception as e:
            logger.error(f"Error initializing memory: {e}", exc_info=True)
    else:
        logger.info(f"Resuming training from epoch {start_epoch}. Skipping initial memory population.")

    total_nan_batches = 0

    logger.info(f"Starting training loop from epoch {start_epoch} to {num_epochs - 1}")
    for epoch in range(start_epoch, num_epochs):
        # --- NAME ERROR FIX: time.time() needs 'import time' ---
        epoch_start_time = time.time() # For timing epochs
        logger.info(f"===== Epoch {epoch}/{num_epochs - 1} =====")

        model.train()
        train_loss_accum = 0.0
        train_seg_loss_accum = 0.0
        train_energy_loss_accum = 0.0
        train_hopfield_loss_accum = 0.0
        batch_count_epoch = 0
        nan_batches_epoch = 0
        consecutive_nan_epoch = 0

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        if aux_loader:
            aux_iter = iter(cycle(aux_loader))
        else:
            aux_iter = None

        for batch_idx, train_batch_data in enumerate(pbar_train):
            # (Batch preparation logic - identical to previous version)
            if isinstance(train_batch_data, (list, tuple)):
                images, masks = train_batch_data[0], train_batch_data[1]
            else: continue # Skip unexpected data

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if aux_iter:
                try:
                    aux_batch_data = next(aux_iter)
                    if isinstance(aux_batch_data, (list, tuple)): aux_images = aux_batch_data[0]
                    else: aux_images = aux_batch_data
                    aux_images = aux_images.to(device, non_blocking=True)
                    num_aux = images.size(0)
                    if aux_images.size(0) < num_aux: num_aux = aux_images.size(0)

                    if num_aux > 0:
                        combined_images = torch.cat([images, aux_images[:num_aux]], dim=0)
                        ignore_idx = getattr(criterion, 'ignore_index', 255)
                        aux_masks = torch.full_like(masks[:num_aux], fill_value=ignore_idx)
                        combined_masks = torch.cat([masks, aux_masks], dim=0)
                        outlier_mask = torch.zeros(combined_images.size(0), dtype=torch.bool, device=device)
                        outlier_mask[images.size(0):images.size(0)+num_aux] = True
                    else:
                        combined_images, combined_masks = images, masks
                        outlier_mask = torch.zeros(images.size(0), dtype=torch.bool, device=device)
                except StopIteration:
                    combined_images, combined_masks = images, masks
                    outlier_mask = torch.zeros(images.size(0), dtype=torch.bool, device=device)
                    aux_iter = None
                except Exception as e:
                    logger.error(f"Error processing auxiliary data batch: {e}", exc_info=True)
                    combined_images, combined_masks = images, masks
                    outlier_mask = torch.zeros(images.size(0), dtype=torch.bool, device=device)
            else:
                combined_images = images
                combined_masks = masks
                outlier_mask = torch.zeros(images.size(0), dtype=torch.bool, device=device)

            # --- Forward, Loss, Backward, Optimize ---
            optimizer.zero_grad(set_to_none=True)
            try:
                outputs = model(combined_images)
                if 'is_ood' not in outputs: outputs['is_ood'] = outlier_mask
                losses = criterion(outputs, combined_masks)
                loss = losses['total_loss']

                # (NaN Handling - identical to previous version)
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    total_nan_batches += 1; nan_batches_epoch += 1; consecutive_nan_epoch += 1
                    logger.warning(f"NaN/Inf loss (Epoch {epoch}, Batch {batch_idx}, Total: {total_nan_batches}, Consec: {consecutive_nan_epoch}). Skipping batch.")
                    if consecutive_nan_epoch > 5:
                        logger.warning(f"Reducing LR due to {consecutive_nan_epoch} consecutive NaNs.")
                        for i, pg in enumerate(optimizer.param_groups): old_lr = pg['lr']; pg['lr'] = max(old_lr * 0.5, 1e-7); logger.info(f"  LR Group {i}: {old_lr:.6e} -> {pg['lr']:.6e}")
                        consecutive_nan_epoch = 0
                    del outputs, loss, losses, combined_images, combined_masks, outlier_mask, images, masks; gc.collect();
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue
                else:
                    consecutive_nan_epoch = 0

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                skip_update = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        skip_update = True; total_nan_batches +=1; nan_batches_epoch += 1
                        logger.warning(f"NaN/Inf gradient after clipping (Epoch {epoch}, Batch {batch_idx}). Skipping step.")
                        break

                if not skip_update: optimizer.step()
                else: optimizer.zero_grad(set_to_none=True)

                # (Accumulate Stats & Update Progress Bar - identical)
                if not (torch.isnan(loss).item() or torch.isinf(loss).item()):
                    train_loss_accum += loss.item()
                    train_seg_loss_accum += losses.get('seg_loss', torch.tensor(0.0)).item()
                    train_energy_loss_accum += losses.get('energy_loss', torch.tensor(0.0)).item()
                    train_hopfield_loss_accum += losses.get('hopfield_loss', torch.tensor(0.0)).item()
                    batch_count_epoch += 1
                    pbar_train.set_postfix({'loss': f"{loss.item():.4f}", 'grad_norm': f"{grad_norm:.2f}", 'nans': nan_batches_epoch})

                # (Memory Management - identical)
                del outputs, loss, losses, combined_images, combined_masks, outlier_mask, images, masks
                if batch_idx % 10 == 0: gc.collect();
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                # (Periodic Hopfield Memory Update - identical)
                if batch_idx > 0 and batch_idx % memory_update_freq == 0:
                    logger.info(f"Triggering periodic memory update at epoch {epoch}, batch {batch_idx}...")
                    try:
                        update_memory_from_loader(model, train_loader, device, num_batches=memory_update_batches)
                    except Exception as e: logger.error(f"Error during periodic memory update: {e}", exc_info=True)
                    finally: model.train()

            except RuntimeError as e:
                 if "out of memory" in str(e).lower(): logger.error(f"OOM error (Epoch {epoch}, Batch {batch_idx}): {e}. Skipping batch."); gc.collect();
                 if torch.cuda.is_available(): torch.cuda.empty_cache(); continue
                 else: logger.error(f"Runtime error (Epoch {epoch}, Batch {batch_idx}): {e}", exc_info=True); continue
            except Exception as e: logger.error(f"Generic error in training loop (Epoch {epoch}, Batch {batch_idx}): {e}", exc_info=True); continue

        # --- End of Training Epoch ---
        pbar_train.close()
        # (Logging Avg Losses - identical)
        valid_batches_epoch = max(batch_count_epoch, 1)
        avg_train_loss = train_loss_accum / valid_batches_epoch; avg_train_seg_loss = train_seg_loss_accum / valid_batches_epoch
        avg_train_energy_loss = train_energy_loss_accum / valid_batches_epoch; avg_train_hopfield_loss = train_hopfield_loss_accum / valid_batches_epoch
        logger.info(f"Epoch {epoch} Train Summary: Avg Loss={avg_train_loss:.4f}, Seg={avg_train_seg_loss:.4f}, Energy={avg_train_energy_loss:.4f}, Hopfield={avg_train_hopfield_loss:.4f}")
        if nan_batches_epoch > 0: logger.warning(f"Epoch {epoch} encountered {nan_batches_epoch} NaN/Inf batches during training.")

        # --- Validation Phase ---
        logger.info(f"Starting validation for epoch {epoch}...")
        val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss = validate(val_loader, model, criterion, device, False)
        logger.info(f"Epoch {epoch} Validation Summary: Avg Loss={val_loss:.4f}, Seg={val_seg_loss:.4f}, Energy={val_energy_loss:.4f}, Hopfield={val_hopfield_loss:.4f}")

        # --- Learning Rate Scheduling ---
        # (LR Scheduling Logic - identical)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_loss)
            else: scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr: logger.info(f"Learning rate updated by scheduler: {current_lr:.6e} -> {new_lr:.6e}")
        lr_string = "Current Learning Rates: ";
        for i, pg in enumerate(optimizer.param_groups): lr_string += f"Group{i}={pg['lr']:.6e} "
        logger.info(lr_string)

        # --- Checkpoint Saving ---
        # (Checkpoint Saving Logic - identical)
        is_best = val_loss < best_val_loss
        if is_best: best_val_loss = val_loss; logger.info(f"*** New best validation loss: {best_val_loss:.4f} at epoch {epoch} ***")
        checkpoint_state = {
            'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'best_val_loss': best_val_loss,
            'current_val_loss': val_loss, 'criterion_state': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
            'memory_bank': model.hopfield_layer.memory_bank.cpu() if hasattr(model, 'hopfield_layer') and hasattr(model.hopfield_layer, 'memory_bank') else None}
        save_checkpoint(checkpoint_state, False, save_path, filename='latest_checkpoint.pth')
        if is_best: save_checkpoint(checkpoint_state, True, save_path) # Saves as model_best.pth
        if epoch % 5 == 0: periodic_filename = f"checkpoint_epoch_{epoch:03d}.pth"; save_checkpoint(checkpoint_state, False, save_path, filename=periodic_filename); logger.info(f"Saved periodic checkpoint: {periodic_filename}")

        # --- Epoch Timing & Cleanup ---
        # --- NAME ERROR FIX: time.time() needs 'import time' ---
        epoch_end_time = time.time()
        logger.info(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- End of Training ---
    # (End of Training Logging and Best Model Loading - identical)
    logger.info("="*20 + " Training Loop Finished " + "="*20)
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
    best_model_path = os.path.join(save_path, 'model_best.pth')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model weights from {best_model_path}...")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Successfully loaded best model weights.")
        except Exception as e: logger.error(f"Failed to load best model weights from {best_model_path}: {e}. Returning model with last epoch weights.")
    else: logger.warning(f"Best model checkpoint ('model_best.pth') not found in {save_path}. Returning model with last epoch weights.")

    return model
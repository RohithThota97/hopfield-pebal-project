# trainer.py
# -*- coding: utf-8 -*-
"""
Training loop logic for the Hopfield-PEBAL model, including NaN handling and memory management.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F # Make sure F is imported
import logging
import gc
import shutil
import time
from tqdm import tqdm
from itertools import cycle
from typing import Dict, Optional, Union, List

# Assuming hopfield_pebal_model is in the same directory or python path
try:
    # Assuming HopfieldPEBALModel and EfficientMemoryManager are defined in hopfield_pebal_model
    from hopfield_pebal_model import HopfieldPEBALModel, EfficientMemoryManager, MemoryTracker
except ImportError:
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.warning("Could not import HopfieldPEBALModel/EfficientMemoryManager/MemoryTracker from hopfield_pebal_model.py. Ensure it's accessible.")
    # Define dummy classes if import fails
    class HopfieldPEBALModel(nn.Module): pass
    class EfficientMemoryManager: pass
    class MemoryTracker: pass

# Get logger instance
logger = logging.getLogger("Hopfield-PEBAL.Trainer")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Helper function for saving checkpoints ---
def save_checkpoint(state: Dict, is_best: bool, save_dir: str, filename: str ='checkpoint.pth'):
    """Saves checkpoint and optionally creates a copy as 'model_best.pth'."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    try:
        torch.save(state, filepath)
        logger.debug(f"Checkpoint saved to {filepath}")
        if is_best:
            best_path = os.path.join(save_dir, 'model_best.pth')
            shutil.copyfile(filepath, best_path)
            epoch_num = state.get('epoch', 'N/A')
            best_loss_val = state.get('best_val_loss', float('inf'))
            best_loss_str = f"{best_loss_val:.4f}" if isinstance(best_loss_val, (float, int)) else str(best_loss_val)
            logger.info(f"Best model checkpoint saved to {best_path} (Epoch {epoch_num}, Loss: {best_loss_str})")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {filepath}: {e}", exc_info=True)

# --- Function to update memory ---
def update_memory_from_loader(model: HopfieldPEBALModel, loader: torch.utils.data.DataLoader, device: torch.device, num_batches: int = 5, downsample_input: bool = True):
    """
    Extracts features, prepares them (adapt, project), resizes labels,
    and updates the Hopfield memory using batches from a data loader.
    Expects model.update_memory to handle tensors on the provided 'device'.
    """
    if not hasattr(model, 'update_memory') or not callable(model.update_memory):
         logger.error("Model does not have a callable 'update_memory' method. Skipping memory update.")
         return
    # Check needed components exist
    required_attrs = ['backbone', 'channel_adapter', 'memory_input_proj', '_check_and_handle_nan_inf']
    if hasattr(model, 'insertion_point') and model.insertion_point == 'after_seghead':
        required_attrs.append('segmentation_head')
    missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr) or not getattr(model, attr)]
    if missing_attrs:
        logger.error(f"Model is missing required attributes for memory update: {missing_attrs}. Skipping.")
        return

    model.eval()
    logger.info(f"Starting memory update process using up to {num_batches} batches...")

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    features_for_update_list = []
    labels_for_update_list = [] # Will store potentially resized labels or None
    processed_batches = 0
    total_images_processed = 0
    mem_loader_iter = iter(loader)

    with torch.no_grad():
        for i in range(num_batches):
            try:
                batch_data = next(mem_loader_iter)
            except StopIteration:
                logger.warning(f"Memory update loader exhausted after {i} batches (requested {num_batches}). Stopping update.")
                break

            images, labels = None, None
            if isinstance(batch_data, dict):
                images = batch_data.get('image')
                labels = batch_data.get('mask') # Original labels from loader
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                images = batch_data[0]
                if len(batch_data) >= 2: labels = batch_data[1]
            else: images = batch_data

            if images is None:
                logger.warning(f"No 'image' found in batch {i} for memory update. Skipping.")
                continue

            try:
                images = images.to(device, non_blocking=True)
                if labels is not None:
                    # **Ensure labels are long and on device initially**
                    labels = labels.long().to(device, non_blocking=True)

                # Optional input image downsampling (also tries to downsample labels)
                if downsample_input and (images.shape[2] > 256 or images.shape[3] > 512):
                    original_size = images.shape[2:]
                    target_size = (min(original_size[0], 256), min(original_size[1], 512))
                    if target_size != original_size:
                        images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)
                        if labels is not None and labels.ndim >= 2 and labels.shape[-2:] == original_size:
                             # Downsample labels here if image was downsampled
                             label_interp_input = labels
                             # Ensure B, C, H, W for interpolate
                             added_dims = 0
                             if label_interp_input.ndim == 2: # H, W -> 1, 1, H, W
                                 label_interp_input = label_interp_input.unsqueeze(0).unsqueeze(1)
                                 added_dims = 2
                             elif label_interp_input.ndim == 3: # B, H, W -> B, 1, H, W
                                 label_interp_input = label_interp_input.unsqueeze(1)
                                 added_dims = 1

                             resized_labels = F.interpolate(label_interp_input.float(), size=target_size, mode='nearest').long()

                             # Restore original dims
                             if added_dims >= 1: resized_labels = resized_labels.squeeze(1) # Remove C
                             if added_dims >= 2: resized_labels = resized_labels.squeeze(0) # Remove B

                             labels = resized_labels # Update labels variable

                        logger.debug(f"Downsampled input from {original_size} to {target_size} for memory update.")

                # --- Feature Extraction ---
                features_raw = model.backbone(images)
                if isinstance(features_raw, (list, tuple)): features_raw = features_raw[-1]
                features_raw = model._check_and_handle_nan_inf(features_raw, "MemUpdate Backbone Features")

                # --- Feature Preparation ---
                features_to_project = None
                if model.insertion_point == 'after_backbone':
                    features_adapted = model.channel_adapter(features_raw)
                    features_adapted = model._check_and_handle_nan_inf(features_adapted, "MemUpdate Features Adapted")
                    features_to_project = features_adapted
                else: # after_seghead
                    if not hasattr(model, 'segmentation_head') or model.segmentation_head is None:
                         logger.error("Model segmentation_head missing for 'after_seghead' memory update. Skipping batch.")
                         continue
                    # Use the appropriate head based on model structure
                    seg_features = model.segmentation_head(features_raw)
                    if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                    seg_features = model._check_and_handle_nan_inf(seg_features, "MemUpdate SegHead Features")
                    features_adapted = model.channel_adapter(seg_features)
                    features_adapted = model._check_and_handle_nan_inf(features_adapted, "MemUpdate Features Adapted")
                    features_to_project = features_adapted
                    del seg_features

                features_mem_proj = model.memory_input_proj(features_to_project)
                features_mem_proj = model._check_and_handle_nan_inf(features_mem_proj, "MemUpdate Memory Proj")

                # --- Append Features and Process/Resize Labels ---
                features_for_update_list.append(features_mem_proj.detach())

                # <<< START: LABEL RESIZING LOGIC >>>
                processed_label = None # Initialize processed label for this batch
                if labels is not None:
                    target_spatial_shape = features_mem_proj.shape[-2:] # Get target H, W (e.g., (64, 128))
                    original_label_shape = labels.shape # For logging

                    # Check if label already matches the feature map's spatial dimensions
                    if labels.ndim >= 2 and labels.shape[-2:] == target_spatial_shape:
                        processed_label = labels.detach()
                        # Ensure it's at least 3D (B, H, W) if it came in as H, W
                        if processed_label.ndim == 2:
                            processed_label = processed_label.unsqueeze(0) # B=1, H, W
                    else:
                        # Spatial dimensions mismatch, resize needed
                        logger.debug(f"Label shape {original_label_shape} mismatch feature shape {features_mem_proj.shape}, resizing label to {target_spatial_shape}...")
                        try:
                            label_to_resize = labels # Work with a reference
                            current_ndim = label_to_resize.ndim
                            added_batch = False
                            added_channel = False

                            # Ensure label is suitable for interpolation (needs B, C, H, W)
                            if current_ndim == 2: # H, W
                                label_to_resize = label_to_resize.unsqueeze(0) # B=1, H, W
                                added_batch = True
                                current_ndim = 3

                            if current_ndim == 3: # B, H, W
                                label_to_resize = label_to_resize.unsqueeze(1) # B, 1, H, W
                                added_channel = True
                            elif current_ndim == 4: # B, C, H, W
                                if label_to_resize.shape[1] != 1:
                                    logger.warning(f"Label has shape {label_to_resize.shape}, expected C=1. Using first channel for resizing.")
                                    label_to_resize = label_to_resize[:, 0:1, :, :] # Select first channel, keep C dim
                            else:
                                logger.warning(f"Label has unexpected dimensions {current_ndim} (original: {original_label_shape}). Cannot resize.")
                                label_to_resize = None # Mark as unusable

                            # Perform interpolation if possible
                            if label_to_resize is not None:
                                resized_label = F.interpolate(label_to_resize.float(), size=target_spatial_shape, mode='nearest')
                                # Convert back to Long type and remove added channel dim if needed
                                if added_channel:
                                    processed_label = resized_label.squeeze(1).long() # B, H, W
                                else:
                                    processed_label = resized_label.long() # B, C, H, W (if C was originally > 1 or 1)
                                    # Force squeeze C=1 as update_memory likely expects B,H,W
                                    if processed_label.ndim == 4 and processed_label.shape[1] == 1:
                                        processed_label = processed_label.squeeze(1) # Ensure B, H, W

                                # If we added a batch dim for a single H,W image, keep it (B=1, H, W)
                                if processed_label.ndim == 2 and added_batch:
                                     processed_label = processed_label.unsqueeze(0) # Ensure Batch dim exists

                                # Final check after resize
                                if processed_label.ndim < 2 or processed_label.shape[-2:] != target_spatial_shape:
                                     logger.warning(f"Label resize inconsistency. Original: {original_label_shape}, Target: {target_spatial_shape}, After Resize: {processed_label.shape}. Discarding label.")
                                     processed_label = None
                                else:
                                     processed_label = processed_label.detach() # Success!

                            else: # label_to_resize was marked as None earlier
                                processed_label = None

                        except Exception as resize_e:
                             logger.error(f"Error resizing label (Original: {original_label_shape}, Target: {target_spatial_shape}): {resize_e}", exc_info=True)
                             processed_label = None

                # Append the potentially resized label (or None if resize failed or no label existed)
                labels_for_update_list.append(processed_label)
                # <<< END: LABEL RESIZING LOGIC >>>

                processed_batches += 1
                total_images_processed += images.shape[0]

                # --- Memory Management ---
                del images, labels, features_raw, features_adapted, features_to_project, features_mem_proj, batch_data
                if 'seg_features' in locals(): del seg_features
                if 'processed_label' in locals(): del processed_label # Clean up temp var
                if (i + 1) % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"Error processing batch {i} for memory update prep: {e}", exc_info=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
                continue # Skip to next batch

    # --- Actual Memory Update Call ---
    if features_for_update_list:
        try:
            logger.info(f"Calling model.update_memory with {len(features_for_update_list)} batches ({total_images_processed} total images)...")
            all_features_mem_proj = torch.cat(features_for_update_list, dim=0)

            # Concatenate labels, handling None entries carefully
            all_labels = None
            valid_labels = [lbl for lbl in labels_for_update_list if lbl is not None and isinstance(lbl, torch.Tensor)]
            if valid_labels:
                # Check if all valid labels have the same spatial dimensions as the first one
                first_shape = valid_labels[0].shape[-2:]
                if all(lbl.shape[-2:] == first_shape for lbl in valid_labels):
                     # Check if the number of features matches the number of labels (batch dim)
                     num_feat_samples = all_features_mem_proj.shape[0]
                     num_lbl_samples = sum(lbl.shape[0] for lbl in valid_labels)

                     if num_feat_samples == num_lbl_samples:
                         try:
                              # Ensure all labels have same ndim before cat (should be B, H, W)
                              if all(lbl.ndim == valid_labels[0].ndim for lbl in valid_labels):
                                  all_labels = torch.cat(valid_labels, dim=0)
                                  logger.info(f"Prepared labels for memory update with shape: {all_labels.shape} on device {all_labels.device}")
                              else:
                                  logger.error(f"Inconsistent label dimensions before concatenation. Skipping labels.")
                                  all_labels = None
                         except Exception as cat_e:
                              logger.error(f"Error concatenating labels: {cat_e}. Labels will be None.", exc_info=True)
                              all_labels = None
                     else:
                         logger.warning(f"Mismatch between feature sample count ({num_feat_samples}) and label sample count ({num_lbl_samples}) after processing. Labels set to None.")
                         all_labels = None
                else:
                     logger.warning("Inconsistent spatial dimensions found in processed labels across batches. Labels set to None.")
                     all_labels = None
            else:
                logger.info("No valid labels found or provided for memory update.")

            # <<< REVERTED CHANGE: Call update_memory directly with tensors on the main device >>>
            # The model's update_memory method should now handle device placement correctly.
            model.update_memory(all_features_mem_proj, labels=all_labels)

            logger.info("Memory update call completed.")

        except Exception as e:
            logger.error(f"Error during model.update_memory call: {e}", exc_info=True)
        finally:
            # Clean up tensors
            del features_for_update_list, labels_for_update_list, all_features_mem_proj, valid_labels
            if 'all_labels' in locals(): del all_labels
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
    else:
        logger.warning("No features were successfully prepared to update memory.")

    logger.info(f"Memory update process finished after processing {processed_batches} batches.")
    model.train() # Ensure model is set back to training mode


# --- Validation Function ---
def validate(val_loader: torch.utils.data.DataLoader, model: HopfieldPEBALModel, criterion: torch.nn.Module, device: torch.device, mixed_precision: bool = False, max_batches: Optional[int] = None):
    """
    Validate the model on validation data.
    """
    model.eval()
    val_loss = 0.0
    val_seg_loss = 0.0
    val_energy_loss = 0.0
    val_hopfield_loss = 0.0
    batch_count = 0
    nan_count = 0

    if not callable(criterion):
        logger.error("Criterion is not callable. Cannot perform validation.")
        return float('inf'), float('inf'), float('inf'), float('inf')

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    amp_device_type = device.type
    logger.info(f"Starting validation{f' (max {max_batches} batches)' if max_batches else ''}...")
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="Validating", leave=False, total=max_batches if max_batches else len(val_loader))
        for batch_idx, batch_data in enumerate(pbar_val):
            if max_batches is not None and batch_idx >= max_batches:
                logger.debug(f"Validation stopped early after {max_batches} batches.")
                break

            images, masks = None, None
            if isinstance(batch_data, dict):
                images = batch_data.get('image')
                masks = batch_data.get('mask')
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                images, masks = batch_data[0], batch_data[1]
            else:
                logger.warning(f"Unexpected data type or structure from validation loader: {type(batch_data)}. Skipping batch.")
                continue

            if images is None or masks is None:
                 logger.warning(f"Missing 'image' or 'mask' in validation batch {batch_idx}. Skipping.")
                 continue

            try:
                images = images.to(device, non_blocking=True)
                masks = masks.long().to(device, non_blocking=True) # Ensure masks are Long type

                # Forward pass
                with torch.amp.autocast(device_type=amp_device_type, enabled=mixed_precision):
                    outputs = model(images)
                    if 'is_ood' not in outputs:
                        outputs['is_ood'] = torch.zeros(images.size(0), dtype=torch.bool, device=device)

                    # Calculate loss (Pass None for ood_images and model during validation)
                    losses = criterion(outputs, masks, ood_images=None, model=None)

                if losses is None or not isinstance(losses, dict) or 'total_loss' not in losses or not isinstance(losses['total_loss'], torch.Tensor):
                     logger.warning(f"Invalid loss dictionary or total_loss from criterion in validation batch {batch_idx}. Skipping.")
                     continue

                current_total_loss = losses['total_loss']

                if torch.isnan(current_total_loss).item() or torch.isinf(current_total_loss).item():
                    nan_count += 1
                    logger.warning(f"NaN/Inf loss detected during validation (batch {batch_idx}). Skipping loss accumulation.")
                    continue

                # Accumulate losses
                val_loss += current_total_loss.item()
                val_seg_loss += losses.get('seg_loss', torch.tensor(0.0)).item()
                val_energy_loss += losses.get('energy_loss', torch.tensor(0.0)).item()
                val_hopfield_loss += losses.get('hopfield_loss', torch.tensor(0.0)).item()
                batch_count += 1

                pbar_val.set_postfix({'loss': f"{current_total_loss.item():.4f}", 'nans': nan_count})

                # Memory management
                del outputs, images, masks, batch_data, losses, current_total_loss
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as e:
                 # Check specifically for the device-side assert during validation as well
                 if "CUDA error: device-side assert triggered" in str(e):
                     logger.error(f"Validation Error: CUDA device-side assert triggered (batch {batch_idx}). Check validation target labels.", exc_info=False)
                     unique_targets = torch.unique(masks)
                     logger.error(f"Unique target values in problematic validation batch: {unique_targets.tolist()}")
                     nan_count += 1 # Count as an issue
                 elif "out of memory" in str(e).lower():
                      logger.error(f"OOM error during validation batch {batch_idx}. Skipping.", exc_info=False)
                 else:
                      logger.error(f"Runtime error during validation batch {batch_idx}: {e}", exc_info=True)
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 gc.collect()
                 continue # Skip to next batch on error
            except Exception as e:
                logger.error(f"Unexpected error during validation batch {batch_idx}: {e}", exc_info=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
                continue

    pbar_val.close()

    if batch_count == 0:
        logger.warning("Validation completed, but no valid batches were processed (batch_count is 0). Returning inf loss.")
        return float('inf'), float('inf'), float('inf'), float('inf')

    avg_val_loss = val_loss / batch_count
    avg_val_seg_loss = val_seg_loss / batch_count
    avg_val_energy_loss = val_energy_loss / batch_count
    avg_val_hopfield_loss = val_hopfield_loss / batch_count

    if nan_count > 0:
        logger.warning(f"Validation completed with {nan_count} problematic batches (skipped in average calculation).")

    logger.info(f"Validation finished. Avg Loss: {avg_val_loss:.4f}")
    model.train() # Ensure model is set back to training mode
    return avg_val_loss, avg_val_seg_loss, avg_val_energy_loss, avg_val_hopfield_loss


# --- Main Training Function ---
def train_hopfield_pebal(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    aux_loader: Optional[torch.utils.data.DataLoader],
    model: HopfieldPEBALModel,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    start_epoch: int = 0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_path: str = 'checkpoints',
    memory_update_freq: int = 10,
    memory_update_batches: int = 5,
    memory_update_downsample_input: bool = True, # Flag for mem update downsampling
    mixed_precision: bool = False,
    use_efficient_memory: bool = False, # This might influence which memory manager is used
    best_val_loss_initial: float = float('inf'),
    grad_clip_norm: float = 1.0
    ):
    """
    Train the Hopfield-PEBAL model.
    """
    os.makedirs(save_path, exist_ok=True)

    # --- Mixed Precision Setup ---
    scaler = None
    amp_device_type = device.type
    if mixed_precision and amp_device_type == 'cuda' and hasattr(torch.cuda.amp, 'GradScaler'):
        scaler = torch.cuda.amp.GradScaler()
        logger.info(f"Automatic Mixed Precision (AMP) enabled with GradScaler for device '{amp_device_type}'.")
    elif mixed_precision:
        logger.warning(f"Mixed precision requested but not supported or enabled for device '{amp_device_type}'. Disabled.")
        mixed_precision = False
    else:
        logger.info("Mixed precision disabled.")

    best_val_loss = best_val_loss_initial
    logger.info(f"Initial best validation loss: {best_val_loss if best_val_loss != float('inf') else 'inf'}")
    last_completed_epoch = start_epoch - 1 # Track the last epoch that fully completed

    # --- Initial Memory Population ---
    if start_epoch == 0:
        logger.info("Populating Hopfield memory initially (start_epoch is 0)...")
        try:
            initial_mem_batches = max(memory_update_batches, 10) # Use more batches for initial population
            update_memory_from_loader(
                model,
                train_loader,
                device,
                num_batches=initial_mem_batches,
                downsample_input=memory_update_downsample_input # Use specific flag
            )
        except Exception as e:
            logger.error(f"Error during initial memory population: {e}", exc_info=True)
    else:
        logger.info(f"Resuming training from epoch {start_epoch}. Skipping initial memory population.")

    total_nan_batches_overall = 0

    logger.info(f"Starting training loop from epoch {start_epoch} to {num_epochs - 1}")
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logger.info(f"===== Epoch {epoch}/{num_epochs - 1} =====")
        current_epoch = epoch # Keep track for saving purposes if interrupted

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
        aux_iter = iter(cycle(aux_loader)) if aux_loader else None

        for batch_idx, train_batch_data in enumerate(pbar_train):
            images, masks = None, None
            if isinstance(train_batch_data, dict):
                images = train_batch_data.get('image')
                masks = train_batch_data.get('mask')
            elif isinstance(train_batch_data, (list, tuple)) and len(train_batch_data) >= 2:
                images, masks = train_batch_data[0], train_batch_data[1]
            else:
                logger.warning(f"Unexpected train batch data type or structure: {type(train_batch_data)}. Skipping batch {batch_idx}.")
                continue

            if images is None or masks is None:
                 logger.warning(f"Missing 'image' or 'mask' in train batch {batch_idx}. Skipping.")
                 continue

            try:
                images = images.to(device, non_blocking=True)
                masks = masks.long().to(device, non_blocking=True) # Ensure masks are Long

                # Combine with auxiliary data
                combined_images = images
                combined_masks = masks
                num_inliers = images.size(0) # Track original image count before potential combination
                outlier_mask = torch.zeros(num_inliers, dtype=torch.bool, device=device)
                aux_images_for_loss = None
                num_aux = 0 # Track how many aux images were actually added

                if aux_iter:
                    try:
                        aux_batch_data = next(aux_iter)
                        aux_images = None
                        if isinstance(aux_batch_data, dict): aux_images = aux_batch_data.get('image')
                        elif isinstance(aux_batch_data, (list, tuple)): aux_images = aux_batch_data[0]
                        elif isinstance(aux_batch_data, torch.Tensor): aux_images = aux_batch_data
                        else: logger.warning(f"Unexpected aux batch data type: {type(aux_batch_data)}. Skipping aux data.")

                        if aux_images is not None:
                            aux_images = aux_images.to(device, non_blocking=True)
                            aux_images_for_loss = aux_images # For loss function
                            # Match batch sizes if different (take minimum)
                            num_aux_available = aux_images.size(0)
                            num_to_combine = min(num_inliers, num_aux_available)
                            num_aux = num_to_combine # Update actual number added

                            if num_aux > 0:
                                aux_images_matched = aux_images[:num_aux]
                                # Only combine if num_aux > 0
                                combined_images = torch.cat([images[:num_aux], aux_images_matched], dim=0) # Use matched num_aux for images too
                                ignore_idx = getattr(criterion, 'ignore_index', 255)
                                aux_masks = torch.full_like(masks[:num_aux], fill_value=ignore_idx) # Match aux_images size
                                combined_masks = torch.cat([masks[:num_aux], aux_masks], dim=0) # Use matched num_aux for masks too
                                # Outlier mask should match the final combined batch size
                                outlier_mask = torch.zeros(combined_images.size(0), dtype=torch.bool, device=device)
                                outlier_mask[num_aux:] = True # Mark the second half (aux) as outliers
                            else:
                                logger.debug("Aux images batch size was 0 or mismatch. Not combining for model input.")
                                # Reset combined tensors to original if no aux added
                                combined_images = images
                                combined_masks = masks
                        else: logger.warning("Could not extract aux images from aux_batch_data.")

                    except StopIteration: aux_iter = None; logger.warning("Aux loader exhausted.")
                    except Exception as e: logger.error(f"Error processing aux data: {e}", exc_info=True)


                # --- Forward, Loss, Backward, Optimize ---
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=amp_device_type, enabled=mixed_precision):
                    outputs = model(combined_images)
                    # Ensure is_ood is present based on whether aux data was actually used
                    if 'is_ood' not in outputs:
                         outputs['is_ood'] = outlier_mask # Use the prepared outlier mask


                    losses = criterion(outputs, combined_masks, ood_images=aux_images_for_loss, model=model) # model needed for OOD energy calc if ood_images provided

                    if losses is None or not isinstance(losses, dict) or 'total_loss' not in losses or not isinstance(losses['total_loss'], torch.Tensor):
                         logger.error(f"Invalid loss dictionary or total_loss from criterion at batch {batch_idx}. Skipping.")
                         # Cleanup
                         del outputs, combined_images, combined_masks, outlier_mask, images, masks
                         if 'aux_images' in locals(): del aux_images
                         if 'aux_images_matched' in locals(): del aux_images_matched
                         if 'aux_masks' in locals(): del aux_masks
                         if 'aux_images_for_loss' in locals(): del aux_images_for_loss
                         if torch.cuda.is_available(): torch.cuda.empty_cache()
                         gc.collect()
                         continue
                    loss = losses['total_loss']

                # Check for NaN/Inf Loss *before* backward
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    total_nan_batches_overall += 1; nan_batches_epoch += 1; consecutive_nan_epoch += 1
                    logger.warning(f"NaN/Inf loss detected (Epoch {epoch}, Batch {batch_idx}, Total: {total_nan_batches_overall}, Consec: {consecutive_nan_epoch}). Skipping batch.")
                    if consecutive_nan_epoch >= 5:
                        logger.warning(f"Reducing LR due to {consecutive_nan_epoch} consecutive NaNs.")
                        for i_lr, pg in enumerate(optimizer.param_groups):
                            old_lr = pg['lr']; pg['lr'] = max(old_lr * 0.5, 1e-7); logger.info(f"  LR Group {i_lr}: {old_lr:.6e} -> {pg['lr']:.6e}")
                        consecutive_nan_epoch = 0
                    # Cleanup
                    del outputs, loss, losses, combined_images, combined_masks, outlier_mask, images, masks
                    if 'aux_images' in locals(): del aux_images
                    if 'aux_images_matched' in locals(): del aux_images_matched
                    if 'aux_masks' in locals(): del aux_masks
                    if 'aux_images_for_loss' in locals(): del aux_images_for_loss
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    consecutive_nan_epoch = 0

                # --- Backward Pass and Optimizer Step ---
                grad_norm = torch.tensor(float('nan')) # Initialize
                if scaler: # Mixed precision
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer) # Unscale before clipping
                    if grad_clip_norm > 0: # Use the parameter here
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else: # Standard precision
                    loss.backward()
                    if grad_clip_norm > 0: # Use the parameter here
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

                # Check grad norm
                if grad_clip_norm > 0 and (torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item()):
                    logger.warning(f"NaN/Inf gradient norm detected (Epoch {epoch}, Batch {batch_idx}). Grad Norm: {grad_norm.item()}")

                # --- Accumulate Stats & Update Progress Bar ---
                train_loss_accum += loss.item()
                train_seg_loss_accum += losses.get('seg_loss', torch.tensor(0.0)).item()
                train_energy_loss_accum += losses.get('energy_loss', torch.tensor(0.0)).item()
                train_hopfield_loss_accum += losses.get('hopfield_loss', torch.tensor(0.0)).item()
                batch_count_epoch += 1
                pbar_train.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'grad': f"{grad_norm.item():.2f}" if grad_clip_norm > 0 and not torch.isnan(grad_norm) else "N/A",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'nans': nan_batches_epoch
                })

                # --- Periodic Memory Update ---
                # Ensure memory_update_freq > 0 and use batch_idx for frequency
                if memory_update_freq > 0 and batch_idx % memory_update_freq == 0 and batch_idx > 0: # Avoid update on batch 0? (Allowing for now)
                     # Use batch_idx for frequency check to be consistent across epochs if loader restarts
                     logger.info(f"Triggering periodic memory update at epoch {epoch}, batch index {batch_idx}...")
                     try:
                         update_memory_from_loader(
                             model,
                             train_loader,
                             device,
                             num_batches=memory_update_batches,
                             downsample_input=memory_update_downsample_input # Use specific flag
                         )
                     except Exception as e:
                         logger.error(f"Error during periodic memory update: {e}", exc_info=True)
                     model.train() # Ensure model is back in train mode

                # --- Batch Cleanup ---
                del outputs, loss, losses, combined_images, combined_masks, outlier_mask, images, masks
                if 'aux_images' in locals(): del aux_images
                if 'aux_images_matched' in locals(): del aux_images_matched
                if 'aux_masks' in locals(): del aux_masks
                if 'aux_images_for_loss' in locals(): del aux_images_for_loss
                if batch_idx % 50 == 0: # Less frequent cleanup
                     gc.collect()
                     if torch.cuda.is_available(): torch.cuda.empty_cache()

            # --- Error Handling (OOM, CUDA Asserts, etc.) ---
            except RuntimeError as e:
                 if "CUDA error: device-side assert triggered" in str(e):
                     logger.error(f"RuntimeError: CUDA device-side assert triggered (Ep {epoch}, B {batch_idx}). Check target labels! Skipping batch.", exc_info=False)
                     unique_targets = torch.unique(masks) # Check original masks
                     logger.error(f"Unique target values in problematic batch: {unique_targets.tolist()}")
                 elif "out of memory" in str(e).lower():
                     logger.error(f"OOM error detected (Ep {epoch}, B {batch_idx}). Skipping batch.", exc_info=False)
                 else:
                     logger.error(f"Runtime error occurred (Ep {epoch}, B {batch_idx}): {e}", exc_info=True)
                 # Cleanup and continue
                 optimizer.zero_grad(set_to_none=True) # Clear potentially corrupted grads
                 gc.collect()
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 continue
            except Exception as e:
                 logger.error(f"An unexpected error occurred in the training loop (Ep {epoch}, B {batch_idx}): {e}", exc_info=True)
                 optimizer.zero_grad(set_to_none=True)
                 gc.collect()
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 continue


        # --- End of Training Epoch ---
        pbar_train.close()
        if batch_count_epoch == 0:
             logger.error(f"Epoch {epoch} finished without processing any valid batches. Check data or NaN/Assert issues.")
             # Optionally add logic to stop training if this happens consecutively
             continue # Skip validation and checkpointing

        # Log Average Training Losses
        avg_train_loss = train_loss_accum / batch_count_epoch
        avg_train_seg_loss = train_seg_loss_accum / batch_count_epoch
        avg_train_energy_loss = train_energy_loss_accum / batch_count_epoch
        avg_train_hopfield_loss = train_hopfield_loss_accum / batch_count_epoch
        logger.info(f"Epoch {epoch} Train Summary: Avg Loss={avg_train_loss:.4f}, Seg={avg_train_seg_loss:.4f}, Energy={avg_train_energy_loss:.4f}, Hopfield={avg_train_hopfield_loss:.4f}")
        if nan_batches_epoch > 0:
             logger.warning(f"Epoch {epoch} encountered {nan_batches_epoch} NaN/Inf loss batches during training.")

        # --- Validation Phase ---
        val_loss, val_seg_loss, val_energy_loss, val_hopfield_loss = validate(
            val_loader, model, criterion, device, mixed_precision=mixed_precision
        )
        logger.info(f"Epoch {epoch} Validation Summary: Avg Loss={val_loss:.4f}, Seg={val_seg_loss:.4f}, Energy={val_energy_loss:.4f}, Hopfield={val_hopfield_loss:.4f}")

        # --- Learning Rate Scheduling ---
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                logger.info(f"Learning rate reduced by scheduler: {current_lr:.6e} -> {new_lr:.6e}")
        lr_log_str = ", ".join([f"Group{i_lr}={pg['lr']:.2e}" for i_lr, pg in enumerate(optimizer.param_groups)])
        logger.info(f"Current LRs: [{lr_log_str}]")


        # --- Checkpoint Saving ---
        is_best = val_loss < best_val_loss and val_loss != float('inf') # Ensure valid loss is best
        if is_best:
            best_val_loss = val_loss
            logger.info(f"*** New best validation loss: {best_val_loss:.4f} at epoch {epoch} ***")

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
            'best_val_loss': best_val_loss,
            'current_val_loss': val_loss,
            'amp_scaler_state_dict': scaler.state_dict() if scaler else None, # Save scaler state
        }
        # Always save the latest checkpoint
        save_checkpoint(checkpoint_state, False, save_path, filename='latest_checkpoint.pth')

        if is_best:
            save_checkpoint(checkpoint_state, True, save_path) # Saves as model_best.pth

        # Optional: Save periodic checkpoints
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            periodic_filename = f"checkpoint_epoch_{epoch:03d}.pth"
            save_checkpoint(checkpoint_state, False, save_path, filename=periodic_filename)

        last_completed_epoch = epoch # Mark this epoch as completed successfully

        # --- Epoch Timing & Cleanup ---
        epoch_end_time = time.time()
        logger.info(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    # --- End of Training ---
    logger.info("="*20 + " Training Loop Finished " + "="*20)
    logger.info(f"Training loop intended for {num_epochs} epochs finished or was interrupted.")
    logger.info(f"Last completed epoch: {last_completed_epoch}")
    logger.info(f"Best validation loss achieved during training: {best_val_loss:.4f}")
    if total_nan_batches_overall > 0:
        logger.warning(f"Total NaN/Inf loss batches encountered during entire training: {total_nan_batches_overall}")

    # --- Load Best Model ---
    best_model_path = os.path.join(save_path, 'model_best.pth')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model weights found at {best_model_path}...")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            # Handle potential issues loading state dict (e.g., different model structure)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys: logger.warning(f"Missing keys when loading best model state_dict: {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys when loading best model state_dict: {unexpected_keys}")
            logger.info(f"Successfully loaded best model weights from epoch {checkpoint.get('epoch', 'N/A')}.")
        except Exception as e:
            logger.error(f"Failed to load best model weights: {e}. Returning model with last epoch weights.", exc_info=True)
    else:
        logger.warning(f"Best model checkpoint ('model_best.pth') not found in {save_path}. Returning model with last epoch weights.")

    return model, best_val_loss, last_completed_epoch # Return model, best loss, and last completed epoch
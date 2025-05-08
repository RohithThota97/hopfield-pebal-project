import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import time

logger = logging.getLogger("memory_utils")

def update_memory_banks(extractor, projection_head, dataloader, device, max_patterns=1000, 
                        max_batches=20, samples_per_batch=2):
    """
    Update memory banks with feature patterns from both inlier and outlier regions.
    
    Args:
        extractor: Feature extractor model
        projection_head: Projection head model
        dataloader: Data loader for mixed data
        device: Computation device
        max_patterns: Maximum number of patterns to store
        max_batches: Maximum number of batches to process
        samples_per_batch: Number of images to process per batch
        
    Returns:
        Tuple of inlier and outlier patterns tensors
    """
    logger.info("Starting memory bank update...")
    projection_head.eval()
    
    # Use lists for collecting patterns to avoid memory issues
    inlier_patterns_list = []
    outlier_patterns_list = []
    
    # Track the current count to avoid unnecessary processing
    inlier_count = 0
    outlier_count = 0
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Updating memory banks")):
            # Early termination if we have enough patterns
            if inlier_count >= max_patterns and outlier_count >= max_patterns:
                logger.info(f"Collected enough patterns: {inlier_count} inliers, {outlier_count} outliers")
                break
                
            if batch_idx >= max_batches:
                logger.info(f"Reached max batches: {max_batches}")
                break
            
            try:
                # Get data (process only a subset if needed)
                images = batch['data'][:samples_per_batch].to(device)
                labels = batch['label'][:samples_per_batch].to(device)
                
                # Check for OOD labels (both 254 and 255 can be OOD indicators)
                ood_mask_254 = (labels == 254).float()
                ood_mask_255 = (labels == 255).float()
                
                # Choose mask with more OOD pixels, or combine them
                if ood_mask_254.sum() > 0 and ood_mask_255.sum() > 0:
                    # Both masks have OOD pixels, combine them
                    ood_mask = torch.clamp(ood_mask_254 + ood_mask_255, 0, 1)
                else:
                    # Use the mask with more OOD pixels
                    ood_mask = ood_mask_254 if ood_mask_254.sum() > ood_mask_255.sum() else ood_mask_255
                
                # Check if 'is_ood' field exists and use it if no OOD pixels found
                if ood_mask.sum() == 0 and 'is_ood' in batch:
                    is_ood = batch['is_ood'][:samples_per_batch].to(device)
                    if is_ood.sum() > 0:
                        ood_mask = is_ood.float()
                
                # Memory management - clear cache before extraction
                torch.cuda.empty_cache()
                features = extractor.extract_features(images)
                
                # Project features
                projected = projection_head(features)
                
                # Free memory
                del features
                torch.cuda.empty_cache()
                
                # Get spatial dimensions of projected features
                batch_size, channels, height, width = projected.shape
                
                # Downsample OOD labels to match feature map size
                if ood_mask.shape[-2:] != (height, width):
                    # Add channel dimension if missing
                    if len(ood_mask.shape) == 3:
                        ood_mask = ood_mask.unsqueeze(1)
                    elif len(ood_mask.shape) == 2:
                        ood_mask = ood_mask.unsqueeze(1).unsqueeze(1)
                    
                    ood_mask = F.interpolate(
                        ood_mask,
                        size=(height, width),
                        mode='nearest'
                    )
                    
                    # Remove channel dimension if it was added
                    if len(ood_mask.shape) == 4 and ood_mask.shape[1] == 1:
                        ood_mask = ood_mask.squeeze(1)
                
                # Process each image in the batch
                for b in range(batch_size):
                    # Get inlier mask
                    inlier_mask = (ood_mask[b] < 0.5)
                    
                    if torch.any(inlier_mask) and inlier_count < max_patterns:
                        # Reshape feature map to be [C, H*W]
                        proj_features = projected[b].reshape(channels, -1)
                        flat_mask = inlier_mask.reshape(-1)
                        
                        # Select features corresponding to inlier pixels
                        inlier_feats = proj_features[:, flat_mask]
                        
                        # Transpose to have features as rows [N, C]
                        inlier_feats = inlier_feats.t()
                        
                        # Sample if too many (to avoid memory issues)
                        if inlier_feats.shape[0] > 50:
                            indices = torch.randperm(inlier_feats.shape[0])[:50]
                            inlier_feats = inlier_feats[indices]
                        
                        # Add to collection
                        inlier_patterns_list.append(inlier_feats.cpu())
                        inlier_count += inlier_feats.shape[0]
                    
                    # Get outlier mask
                    outlier_mask = (ood_mask[b] >= 0.5)
                    
                    if torch.any(outlier_mask) and outlier_count < max_patterns:
                        # Reshape feature map to be [C, H*W]
                        proj_features = projected[b].reshape(channels, -1)
                        flat_mask = outlier_mask.reshape(-1)
                        
                        # Select features corresponding to outlier pixels
                        outlier_feats = proj_features[:, flat_mask]
                        
                        # Transpose to have features as rows [N, C]
                        outlier_feats = outlier_feats.t()
                        
                        # Sample if too many
                        if outlier_feats.shape[0] > 50:
                            indices = torch.randperm(outlier_feats.shape[0])[:50]
                            outlier_feats = outlier_feats[indices]
                        
                        # Add to collection
                        outlier_patterns_list.append(outlier_feats.cpu())
                        outlier_count += outlier_feats.shape[0]
                
                # Free memory
                del projected, ood_mask
                torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    # Process collected patterns
    logger.info("Processing collected patterns...")
    
    # Handle inlier patterns
    if inlier_patterns_list:
        try:
            inlier_patterns = torch.cat(inlier_patterns_list, dim=0).to(device)
            logger.info(f"Combined inlier patterns: {inlier_patterns.shape}")
            
            # Sample if too many
            if inlier_patterns.shape[0] > max_patterns:
                indices = torch.randperm(inlier_patterns.shape[0])[:max_patterns]
                inlier_patterns = inlier_patterns[indices]
                
            logger.info(f"Final inlier patterns: {inlier_patterns.shape}")
        except Exception as e:
            logger.error(f"Error processing inlier patterns: {str(e)}")
            # Create random patterns as fallback
            inlier_patterns = torch.randn(100, projection_head.projection[-1].out_channels, device=device)
            logger.warning("Using random inlier patterns due to error")
    else:
        # No patterns collected, create random ones
        inlier_patterns = torch.randn(100, projection_head.projection[-1].out_channels, device=device)
        logger.warning("No inlier patterns found, using random patterns")
    
    # Handle outlier patterns
    if outlier_patterns_list:
        try:
            outlier_patterns = torch.cat(outlier_patterns_list, dim=0).to(device)
            logger.info(f"Combined outlier patterns: {outlier_patterns.shape}")
            
            # Sample if too many
            if outlier_patterns.shape[0] > max_patterns:
                indices = torch.randperm(outlier_patterns.shape[0])[:max_patterns]
                outlier_patterns = outlier_patterns[indices]
                
            logger.info(f"Final outlier patterns: {outlier_patterns.shape}")
        except Exception as e:
            logger.error(f"Error processing outlier patterns: {str(e)}")
            # Create random patterns as fallback
            outlier_patterns = torch.randn(100, projection_head.projection[-1].out_channels, device=device)
            logger.warning("Using random outlier patterns due to error")
    else:
        # No patterns collected, create random ones
        outlier_patterns = torch.randn(100, projection_head.projection[-1].out_channels, device=device)
        logger.warning("No outlier patterns found, using random patterns")
    
    logger.info(f"Memory bank update completed in {time.time() - start_time:.2f} seconds")
    
    return inlier_patterns, outlier_patterns
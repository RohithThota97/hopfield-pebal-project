import argparse
import logging
import os
import random
import sys
import torch
import numpy as np
from datetime import datetime
import torch.multiprocessing as mp
import traceback


from feature_extractor import FeatureExtractor
from projection_head import ProjectionHead
from border_energy import BorderEnergy
from memory import update_memory_banks
import torch.nn.functional as F
from tqdm import tqdm



logger = logging.getLogger("hopfield_trainer")

#loss calculation

def compute_loss(projected, ood_mask, inlier_patterns, outlier_patterns, energy_fn):
   
    energy = energy_fn(projected, inlier_patterns, outlier_patterns)
    
   
    logger.info(f"Energy shape: {energy.shape}, min: {energy.min().item():.4f}, max: {energy.max().item():.4f}")
    

    energy_flat = energy.reshape(-1)
    ood_mask_flat = ood_mask.reshape(-1)
    
  
    inlier_mask = (ood_mask_flat < 0.5)
    outlier_mask = (ood_mask_flat >= 0.5)
    
 
    inlier_threshold = -2.0
    outlier_threshold = 2.0
    

    loss_components = {}
    

    if inlier_mask.sum() > 0:
        inlier_energy = energy_flat[inlier_mask]
        loss_inlier = torch.mean(F.relu(inlier_energy - inlier_threshold)**2)
        loss_components['inlier'] = loss_inlier.item()
    else:
        loss_inlier = torch.tensor(0.0, device=projected.device)
        loss_components['inlier'] = 0.0
    

    if outlier_mask.sum() > 0:
        outlier_energy = energy_flat[outlier_mask]
        loss_outlier = torch.mean(F.relu(outlier_threshold - outlier_energy)**2)
        loss_components['outlier'] = loss_outlier.item()
    else:
        loss_outlier = torch.tensor(0.0, device=projected.device)
        loss_components['outlier'] = 0.0
    

    loss_smooth = 0.0
    

    if energy.shape[1] > 1:  
        for i in range(energy.shape[1] - 1):
            loss_smooth += torch.mean(torch.abs(energy[:, i+1:i+2, :] - energy[:, i:i+1, :]))
    
   
    if energy.shape[2] > 1: 
        for i in range(energy.shape[2] - 1):
            loss_smooth += torch.mean(torch.abs(energy[:, :, i+1:i+2] - energy[:, :, i:i+1]))
    

    num_smooth_dims = (energy.shape[1] > 1) + (energy.shape[2] > 1)
    if num_smooth_dims > 0:
        loss_smooth /= num_smooth_dims
    
    loss_components['smooth'] = loss_smooth.item()
    

    loss_sparse = torch.mean(torch.abs(projected))
    loss_components['sparse'] = loss_sparse.item()
    

    smoothness_weight = 5e-4
    sparsity_weight = 3e-6
    

    num_inliers = inlier_mask.sum().item()
    num_outliers = outlier_mask.sum().item()
    logger.info(f"Inliers: {num_inliers}, Outliers: {num_outliers}")
    logger.info(f"Loss components: {loss_components}")
    
    # Total loss
    total_loss = loss_inlier + loss_outlier + smoothness_weight * loss_smooth + sparsity_weight * loss_sparse
    
    return total_loss

def save_checkpoint(projection_head, optimizer, epoch, batch_idx, loss, 
                    inlier_patterns, outlier_patterns, output_dir='.'):
    """Save checkpoint and memory banks"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
    memory_path = os.path.join(output_dir, f'memory_banks_epoch_{epoch}_batch_{batch_idx}.pth')
    
    # Save model checkpoint
    try:
        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': projection_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving model checkpoint: {str(e)}")
    
    # Save memory banks 
    try:
        torch.save({
            'inlier_patterns': inlier_patterns.cpu() if inlier_patterns is not None else None,
            'outlier_patterns': outlier_patterns.cpu() if outlier_patterns is not None else None,
            'inlier_count': len(inlier_patterns) if inlier_patterns is not None else 0,
            'outlier_count': len(outlier_patterns) if outlier_patterns is not None else 0,
        }, memory_path)
        logger.info(f"Saved memory banks to {memory_path}")
    except Exception as e:
        logger.error(f"Error saving memory banks: {str(e)}")
        
#   Training         

def train_hopfield(num_epochs=10, batch_size=1, learning_rate=0.001, 
                  max_batches=5, max_patterns=200, output_dir='.'):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
  
    logger.info("Initializing feature extractor...")
    extractor = FeatureExtractor()
    

    logger.info("Setting up dataloader...")
    dataloader = extractor.get_dataloader()
    logger.info(f"Dataloader created with {len(dataloader)} batches")
    

    logger.info("Extracting features to determine dimensions...")
    sample_batch = next(iter(dataloader))
    sample_images = sample_batch['data'][:1].to(device)
    sample_features = extractor.extract_features(sample_images)
    input_dim = sample_features.shape[1]
    output_dim = 128 
    
    logger.info(f"Feature extraction complete. Shape: {sample_features.shape}")
    logger.info(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    

    del sample_features
    torch.cuda.empty_cache()
    

    logger.info("Initializing projection head...")
    projection_head = ProjectionHead(input_dim, output_dim=output_dim).to(device)
    

    logger.info(f"Setting up optimizer with lr={learning_rate}...")
    optimizer = torch.optim.Adam(projection_head.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    

    logger.info("Initializing energy function...")
    energy_fn = BorderEnergy(beta=4.0)
    

    inlier_patterns = None
    outlier_patterns = None
    

    os.makedirs(output_dir, exist_ok=True)
    

    logger.info(f"Starting training for {num_epochs} epochs...")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")
        

        logger.info(f"Updating memory banks...")
        inlier_patterns, outlier_patterns = update_memory_banks(
            extractor, projection_head, dataloader, device, 
            max_patterns=max_patterns, max_batches=max_batches)
        

        projection_head.train()
        running_loss = 0.0
        batch_count = 0
        

        for batch_idx, batch in enumerate(dataloader):
            try:
          
                images = batch['data'][:batch_size].to(device)
                labels = batch['label'][:batch_size].to(device)
                
  
                logger.debug(f"Label shape: {labels.shape}, unique values: {torch.unique(labels).tolist()}")
                

                features = extractor.extract_features(images)
                projected = projection_head(features)
                
           
                logger.debug(f"Projected shape: {projected.shape}")
                
 
                del features
                torch.cuda.empty_cache()
    
                ood_mask = torch.zeros_like(labels, dtype=torch.float32)
                ood_mask[labels == 254] = 1.0
                ood_mask[labels == 255] = 1.0
                

                logger.debug(f"OOD mask shape: {ood_mask.shape}, sum: {ood_mask.sum().item()}")
     
                if len(ood_mask.shape) < 4:
                    if len(ood_mask.shape) == 3:  # [B, H, W]
                        ood_mask = ood_mask.unsqueeze(1)  # [B, 1, H, W]
                    elif len(ood_mask.shape) == 2:  # [H, W]
                        ood_mask = ood_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
             
                ood_mask = torch.nn.functional.interpolate(
                    ood_mask,
                    size=(projected.shape[2], projected.shape[3]),
                    mode='nearest'
                )
                

                if ood_mask.shape[1] == 1:
                    ood_mask = ood_mask.squeeze(1)
                

                logger.debug(f"Resized OOD mask shape: {ood_mask.shape}, sum: {ood_mask.sum().item()}")
                
     
                if inlier_patterns is None or outlier_patterns is None or inlier_patterns.shape[0] == 0 or outlier_patterns.shape[0] == 0:
                    logger.warning("No patterns available, skipping batch...")
                    continue
                

                loss = compute_loss(projected, ood_mask, inlier_patterns, outlier_patterns, energy_fn)
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                

                optimizer.zero_grad()
                loss.backward()
                
        
                torch.nn.utils.clip_grad_norm_(projection_head.parameters(), max_norm=10.0)
                
           
                optimizer.step()
                

                running_loss += loss.item()
                batch_count += 1
            
                if batch_idx % 10 == 0:
                    save_checkpoint(
                        projection_head, optimizer, epoch, batch_idx, loss.item(),
                        inlier_patterns, outlier_patterns, output_dir
                    )
                
             
                del projected, ood_mask, loss
                torch.cuda.empty_cache()
                

                if max_batches is not None and batch_idx >= max_batches:
                    logger.info(f"Reached max batches: {max_batches}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
  
        epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds, Average Loss: {avg_loss:.4f}")
            
 
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate updated to: {current_lr:.6f}")
            
   
            save_checkpoint(
                projection_head, optimizer, epoch, -1, avg_loss,
                inlier_patterns, outlier_patterns, output_dir
            )
            

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': projection_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, best_model_path)
                logger.info(f"New best model saved with loss: {best_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds with no valid batches")
    
    logger.info(f"Training completed! Best loss: {best_loss:.4f}")
    return projection_head

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hopfield Boosting OOD detector')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--max_batches', type=int, default=10, help='Maximum number of batches to process')
    parser.add_argument('--max_patterns', type=int, default=200, help='Maximum number of patterns to store')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
    parser.add_argument('--pebal_path', type=str, default=None, help='Path to PEBAL code directory')
    
    args = parser.parse_args()
    

    if args.pebal_path:
        sys.path.append(args.pebal_path)
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    log_file = os.path.join(args.output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(logging_level)
    

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    

    train_hopfield(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_batches=args.max_batches,
        max_patterns=args.max_patterns,
        output_dir=args.output_dir
    )

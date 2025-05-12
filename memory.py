import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import time

logger = logging.getLogger("memory_utils")



def update_memory_banks(extractor, projection_head, dataloader, device, max_patterns=1000, 
                        max_batches=20, samples_per_batch=2):
    
    logger.info("Starting Hopfield memory bank update...")
    projection_head.eval()
    

    all_patterns_list = []
    pattern_count = 0
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Updating Hopfield memory")):
            if pattern_count >= max_patterns:
                logger.info(f"Collected enough patterns: {pattern_count}")
                break
                
            if batch_idx >= max_batches:
                logger.info(f"Reached max batches: {max_batches}")
                break
            
            try:

                images = batch['data'][:samples_per_batch].to(device)
                

                torch.cuda.empty_cache()
                features = extractor.extract_features(images)
           
                projected = projection_head(features)
                

                del features
                torch.cuda.empty_cache()
                
  
                batch_size, channels, height, width = projected.shape
                

                for b in range(batch_size):

                    proj_features = projected[b].reshape(channels, -1)
                    
              
                    patterns = proj_features.t()
                    
                
                    if patterns.shape[0] > 50:
                        indices = torch.randperm(patterns.shape[0])[:50]
                        patterns = patterns[indices]
                    
      
                    all_patterns_list.append(patterns.cpu())
                    pattern_count += patterns.shape[0]
                    
           
                    if pattern_count >= max_patterns:
                        break
                
                del projected
                torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    logger.info("Processing collected Hopfield patterns...")
    

    if all_patterns_list:
        try:
            all_patterns = torch.cat(all_patterns_list, dim=0).to(device)
            logger.info(f"Combined patterns: {all_patterns.shape}")
            

            if all_patterns.shape[0] > max_patterns:
                indices = torch.randperm(all_patterns.shape[0])[:max_patterns]
                all_patterns = all_patterns[indices]
                
            logger.info(f"Final patterns: {all_patterns.shape}")
            
            # Normalize patterns for Hopfield network
            all_patterns = F.normalize(all_patterns, p=2, dim=1)
            
        
            inlier_patterns = all_patterns
            outlier_patterns = all_patterns
            
        except Exception as e:
            logger.error(f"Error processing patterns: {str(e)}")

            random_patterns = torch.randn(100, projection_head.projection[-1].out_channels, device=device)
            random_patterns = F.normalize(random_patterns, p=2, dim=1)
            inlier_patterns = random_patterns
            outlier_patterns = random_patterns
            logger.warning("Using random patterns due to error")
    else:

        random_patterns = torch.randn(100, projection_head.projection[-1].out_channels, device=device)
        random_patterns = F.normalize(random_patterns, p=2, dim=1)
        inlier_patterns = random_patterns
        outlier_patterns = random_patterns
        logger.warning("No patterns found, using random patterns")
    
    logger.info(f"Hopfield memory bank update completed in {time.time() - start_time:.2f} seconds")
    
    return inlier_patterns, outlier_patterns

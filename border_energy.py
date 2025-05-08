import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("border_energy")

class BorderEnergy(nn.Module):
    def __init__(self, beta=4.0):
        super(BorderEnergy, self).__init__()
        self.beta = beta
        logger.info(f"Initialized BorderEnergy with beta={beta}")
    
    def forward(self, query, inlier_patterns, outlier_patterns):
        

        batch_size, channels, height, width = query.shape
        
        # Reshape query to [B*H*W, C]
        query_reshaped = query.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Normalize all vectors
        query_norm = F.normalize(query_reshaped, p=2, dim=1)
        inlier_norm = F.normalize(inlier_patterns, p=2, dim=1)
        outlier_norm = F.normalize(outlier_patterns, p=2, dim=1)
        
        # Process in chunks to avoid OOM
        chunk_size = min(10000, query_norm.shape[0])  
        num_chunks = (query_norm.shape[0] + chunk_size - 1) // chunk_size
        
        energy = torch.zeros(query_norm.shape[0], device=query_norm.device)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, query_norm.shape[0])
            
            
            query_chunk = query_norm[start_idx:end_idx]
            
      
            inlier_sim = query_chunk @ inlier_norm.t()
            outlier_sim = query_chunk @ outlier_norm.t()
            
            # Compute energy terms using log-sum-exp
            inlier_energy = -torch.logsumexp(self.beta * inlier_sim, dim=1) / self.beta
            outlier_energy = -torch.logsumexp(self.beta * outlier_sim, dim=1) / self.beta
            
            # Compute joint energy efficiently
            joint_sim = torch.cat([inlier_sim, outlier_sim], dim=1)
            joint_energy = -torch.logsumexp(self.beta * joint_sim, dim=1) / self.beta
            
            # Border energy formula: E_b = -2*log(p(x|in) * p(x|out)) + log(p(x|in)) + log(p(x|out))
            # Equivalent to: 2*joint_energy - inlier_energy - outlier_energy
            energy[start_idx:end_idx] = 2 * joint_energy - inlier_energy - outlier_energy
            

            del inlier_sim, outlier_sim, joint_sim
            torch.cuda.empty_cache()
        
        # Reshape back to [B, H, W]
        energy = energy.reshape(batch_size, height, width)
        
        return energy
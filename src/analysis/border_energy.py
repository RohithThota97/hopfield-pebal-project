import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger("border_energy")

class BorderEnergy(nn.Module):
    def __init__(self, beta=4.0):
        super(BorderEnergy, self).__init__()
        self.beta = beta
        logger.info(f"Initialized Hopfield BorderEnergy with beta={beta}")
    
    def forward(self, query, inlier_patterns, outlier_patterns):
  
        patterns = inlier_patterns
        
        batch_size, channels, height, width = query.shape
        

        query_reshaped = query.permute(0, 2, 3, 1).reshape(-1, channels)
        

        query_norm = F.normalize(query_reshaped, p=2, dim=1)
        patterns_norm = F.normalize(patterns, p=2, dim=1)
        

        chunk_size = min(10000, query_norm.shape[0])  
        num_chunks = (query_norm.shape[0] + chunk_size - 1) // chunk_size
        
        energy = torch.zeros(query_norm.shape[0], device=query_norm.device)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, query_norm.shape[0])
            
            query_chunk = query_norm[start_idx:end_idx]
   
            similarity = self.beta * (query_chunk @ patterns_norm.t())
            

            energy[start_idx:end_idx] = -torch.logsumexp(similarity, dim=1) / self.beta
            
            del similarity
            torch.cuda.empty_cache()
        

        energy = energy.reshape(batch_size, height, width)
        
        return energy

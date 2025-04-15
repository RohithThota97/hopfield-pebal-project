import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threading

class ModernHopfieldLayer(nn.Module):
    def __init__(self, input_dim, output_dim=None, num_heads=4, beta=8.0, memory_size=1000, update_memory=True):
        """
        Modern Hopfield Layer with fixed memory isolation to prevent autograd errors
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (defaults to input_dim)
            num_heads: Number of attention heads
            beta: Temperature parameter for attention
            memory_size: Size of memory bank
            update_memory: Whether to update memory during training
        """
        super(ModernHopfieldLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_heads = num_heads
        self.beta = beta
        self.memory_size = memory_size
        self.update_memory = update_memory
        
        # Memory lock for thread safety
        self.memory_lock = threading.Lock()
        
        # Initialize memory as a persistent attribute (not a registered buffer)
        # This prevents it from being part of the computation graph
        self._memory = torch.randn(memory_size, input_dim)
        self.memory_initialized = False
        
        # Projection layers for query, key, value (multi-head)
        head_dim = self.output_dim // num_heads
        self.head_dim = head_dim
        
        self.query_proj = nn.Linear(input_dim, head_dim * num_heads)
        self.key_proj = nn.Linear(input_dim, head_dim * num_heads)
        self.value_proj = nn.Linear(input_dim, head_dim * num_heads)
        
        # Output projection
        self.output_proj = nn.Linear(head_dim * num_heads, self.output_dim)
        
    def to(self, device):
        """Override to method to ensure memory is moved to device"""
        result = super().to(device)
        with self.memory_lock:
            self._memory = self._memory.to(device)
        return result
    
    def forward(self, x):
        """
        Forward pass with memory isolation to prevent autograd errors
        
        Args:
            x: Input tensor of shape [B, N, C]
                B = batch size
                N = number of tokens/pixels
                C = input dimension
        
        Returns:
            retrieved: Output tensor of shape [B, N, C]
            energy: Energy values of shape [B, N]
        """
        # x shape: [B, N, C] where B=batch_size, N=num_pixels, C=channels
        batch_size, num_pixels, _ = x.shape
        device = x.device
        
        # Ensure memory is on the same device as input
        with self.memory_lock:
            if self._memory.device != device:
                self._memory = self._memory.to(device)
            
            # CRITICAL: Create a completely detached copy of memory for this computation
            memory_copy = self._memory.clone().detach().requires_grad_(False)
        
        # Project input to get query
        queries = self.query_proj(x)  # [B, N, H*D]
        queries = queries.view(batch_size, num_pixels, self.num_heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)  # [B, H, N, D]
        
        # Initialize memory if needed
        if not self.memory_initialized and self.update_memory:
            with torch.no_grad():
                if num_pixels >= self.memory_size:
                    # Take random subset if more pixels than memory size
                    idx = torch.randperm(num_pixels)[:self.memory_size]
                    sample_features = x[0, idx, :].detach().cpu()  # Take from first batch
                else:
                    # Take all available pixels
                    sample_features = x[0, :min(num_pixels, self.memory_size), :].detach().cpu()
                
                # Update memory with lock
                with self.memory_lock:
                    new_memory = self._memory.cpu().clone()
                    new_memory[:len(sample_features)] = sample_features
                    self._memory = new_memory.to(device)
                    self.memory_initialized = True
                
                # Update memory_copy for this forward pass
                memory_copy = self._memory.clone().detach().requires_grad_(False)
        
        # Get keys and values from memory copy
        keys = self.key_proj(memory_copy)  # [M, H*D]
        values = self.value_proj(memory_copy)  # [M, H*D]
        
        # Reshape for multi-head attention
        memory_size = memory_copy.size(0)
        keys = keys.view(memory_size, self.num_heads, self.head_dim)
        keys = keys.permute(1, 0, 2)  # [H, M, D]
        values = values.view(memory_size, self.num_heads, self.head_dim)
        values = values.permute(1, 0, 2)  # [H, M, D]
        
        # Compute attention scores
        # [B, H, N, D] x [H, M, D]T -> [B, H, N, M]
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply beta parameter (temperature scaling) - non-inplace
        attention_scores = attention_scores * self.beta
        
        # Store maximum attention score for energy calculation
        max_scores, _ = torch.max(attention_scores, dim=-1)  # [B, H, N]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, H, N, M]
        
        # Apply attention to values
        # [B, H, N, M] x [H, M, D] -> [B, H, N, D]
        retrieved = torch.matmul(attention_weights, values)
        
        # Reshape and project output
        retrieved = retrieved.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
        retrieved = retrieved.view(batch_size, num_pixels, self.num_heads * self.head_dim)  # [B, N, H*D]
        retrieved = self.output_proj(retrieved)  # [B, N, C]
        
        # Calculate energy based on maximum attention score
        # Higher energy means more dissimilar to memory patterns (potential OOD)
        # Average across heads for final energy value
        energy = -torch.mean(max_scores, dim=1)  # [B, N]
        
        # Update memory with current batch if required (during training)
        # This is completely detached from computation graph
        if self.training and self.update_memory:
            self._update_memory_detached(x)
        
        return retrieved, energy
    
    def _update_memory_detached(self, x):
        """
        Update memory with current batch data (completely detached from graph)
        
        Args:
            x: Input tensor for memory update
        """
        with torch.no_grad():
            # Flatten batch for sampling
            flat_features = x.reshape(-1, self.input_dim).detach().cpu()
            num_total = flat_features.size(0)
            
            # Sample random subset for memory update
            update_size = min(num_total, self.memory_size // 10)  # Update 10% of memory
            if update_size > 0:
                idx = torch.randperm(num_total)[:update_size]
                update_features = flat_features[idx]
                
                # Update memory locations randomly with lock
                with self.memory_lock:
                    memory_idx = torch.randperm(self.memory_size)[:update_size]
                    # Create new memory tensor
                    new_memory = self._memory.cpu().clone()
                    new_memory[memory_idx] = update_features
                    self._memory = new_memory.to(x.device)
    
    def get_memory(self):
        """Return a copy of the current memory"""
        with self.memory_lock:
            return self._memory.clone().detach()
    
    def set_memory(self, new_memory):
        """Set memory with a new tensor (completely detached)"""
        with self.memory_lock:
            self._memory = new_memory.detach().clone()
            self.memory_initialized = True
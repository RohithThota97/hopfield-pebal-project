import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threading
import logging
from typing import Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)
# Example basic configuration if logger is not configured elsewhere
# logging.basicConfig(level=logging.INFO) 

class ModernHopfieldLayer(nn.Module):
    """
    Modern Hopfield Network Layer with efficient memory isolation
    and improved numerical stability.

    This implementation provides:
    - Thread-safe memory management using a lock.
    - Detached memory bank to prevent accidental inclusion in the computation graph.
    - Chunked processing for large inputs to manage memory usage.
    - Numerical stability improvements (scaling, optional Laplace attention).
    - Optional memory bank normalization (L2).
    - Different strategies for updating the memory bank during training.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (Optional[int]): Output feature dimension. Defaults to input_dim.
        num_heads (int): Number of attention heads. Must be > 0.
        beta (float): Temperature parameter for attention scaling (higher beta -> sharper attention).
                      Defaults to 8.0.
        memory_size (int): Size of the memory bank (number of stored patterns). Must be > 0.
        update_memory (bool): Whether to update the memory bank with input samples
                              during training. Defaults to True.
        normalize_memory (bool): Whether to L2-normalize memory vectors upon initialization
                                 and update. Defaults to True.
        use_laplace_attention (bool): If True, use Laplace kernel (exp(-abs(score)))
                                      instead of standard Softmax (exp(score)/sum(exp(score))).
                                      Can be more robust to outliers. Defaults to False.
        dropout (float): Dropout probability applied to the attention weights. Defaults to 0.0.
        chunk_threshold (int): Number of input tokens/pixels above which chunked
                               computation is used to save memory. Defaults to 10000.
        memory_update_strategy (str): Strategy for updating memory during training.
                                      Options: 'fifo', 'random', 'diversity'.
                                      Defaults to 'diversity'.
        memory_update_fraction (float): Fraction of the memory bank to potentially
                                        update in each training step. Defaults to 0.1.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: Optional[int] = None,
                 num_heads: int = 4,
                 beta: float = 8.0,
                 memory_size: int = 1000,
                 update_memory: bool = True,
                 normalize_memory: bool = True,
                 use_laplace_attention: bool = False,
                 dropout: float = 0.0,
                 chunk_threshold: int = 10000,
                 memory_update_strategy: str = 'diversity',
                 memory_update_fraction: float = 0.1):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if chunk_threshold <= 0:
             raise ValueError(f"chunk_threshold must be positive, got {chunk_threshold}")
        if memory_update_strategy not in ['fifo', 'random', 'diversity']:
            raise ValueError(f"Unknown memory_update_strategy: {memory_update_strategy}")
        if not 0.0 < memory_update_fraction <= 1.0:
             raise ValueError(f"memory_update_fraction must be in (0, 1], got {memory_update_fraction}")

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_heads = num_heads
        self.beta = beta
        self.memory_size = memory_size
        self.update_memory = update_memory
        self.normalize_memory = normalize_memory
        self.use_laplace_attention = use_laplace_attention
        self.chunk_threshold = chunk_threshold
        self.memory_update_strategy = memory_update_strategy
        self.memory_update_fraction = memory_update_fraction

        if self.output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim ({self.output_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.head_dim = self.output_dim // num_heads
        self.scale = math.sqrt(self.head_dim) # Scaling factor for attention

        # Memory lock for thread safety when accessing/modifying _memory
        self.memory_lock = threading.Lock()

        # Initialize memory as a tensor attribute (not a registered buffer)
        # This ensures it's moved with .to(device) but NOT part of state_dict
        # or the computation graph unless explicitly handled.
        # Initialized on CPU by default, moved by the overridden `to` method.
        initial_memory = torch.randn(memory_size, input_dim)
        if self.normalize_memory:
            initial_memory = F.normalize(initial_memory, p=2, dim=1)
        self._memory: torch.Tensor = initial_memory
        self.memory_initialized: bool = False
        self._memory_pointer: int = 0 # For FIFO updates

        # Projection layers
        self.query_proj = nn.Linear(input_dim, self.head_dim * num_heads)
        self.key_proj = nn.Linear(input_dim, self.head_dim * num_heads)
        self.value_proj = nn.Linear(input_dim, self.head_dim * num_heads)
        self.output_proj = nn.Linear(self.head_dim * num_heads, self.output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize projection layer weights with Xavier uniform and biases to zero."""
        for proj in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    # Override `to` method to ensure `_memory` is moved correctly
    def to(self, *args, **kwargs):
        """Moves the module and its memory bank to the specified device."""
        new_module = super().to(*args, **kwargs)
        # Ensure memory is also moved, using the lock for thread safety
        with self.memory_lock:
            # Extract device from args/kwargs
            device = None
            if args:
                if isinstance(args[0], torch.device) or isinstance(args[0], str):
                    device = args[0]
            if 'device' in kwargs:
                device = kwargs['device']
                
            if device is not None and self._memory.device != torch.device(device):
                 logger.debug(f"Moving memory bank to device: {device}")
                 self._memory = self._memory.to(device)
            elif device is None and self._memory.device != next(self.parameters()).device:
                 # Handle case like module.cuda() without explicit device
                 target_device = next(self.parameters()).device
                 logger.debug(f"Moving memory bank to inferred device: {target_device}")
                 self._memory = self._memory.to(target_device)
                 
        return new_module

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Modern Hopfield Layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C]
                B = batch size
                N = number of tokens/pixels (sequence length)
                C = input feature dimension (must match self.input_dim)
            return_attention (bool): If True, return the attention weights.
                                     Note: Attention weights are not returned
                                     when using chunked computation for memory efficiency.
                                     Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - retrieved (torch.Tensor): Output tensor of shape [B, N, Cout],
              where Cout is self.output_dim.
            - energy (torch.Tensor): Energy values (negative mean max attention score
              before softmax) of shape [B, N]. Lower energy indicates better
              match with memory patterns.
            - attention_weights (Optional[torch.Tensor]): Attention weights of shape
              [B, H, N, M] (H=num_heads, M=memory_size) if return_attention is True
              and chunking is not used. Otherwise, None.
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions (B, N, C), got {x.dim()}")
        if x.shape[2] != self.input_dim:
            raise ValueError(f"Input feature dimension ({x.shape[2]}) does not match "
                             f"layer's input_dim ({self.input_dim})")

        batch_size, num_tokens, _ = x.shape
        device = x.device

        # --- Memory Handling ---
        # Ensure memory is on the same device as input and initialize if needed
        with self.memory_lock:
            if self._memory.device != device:
                logger.warning(f"Memory bank device ({self._memory.device}) differs from input device ({device}). Moving memory.")
                self._memory = self._memory.to(device)

            if not self.memory_initialized and self.update_memory:
                logger.info("Initializing memory bank with first batch.")
                self._initialize_memory_with_samples(x) # Pass full x

            # CRITICAL: Create a detached copy of memory for this forward pass
            # This prevents gradients flowing into the memory bank itself.
            memory_copy = self._memory.clone().detach()

        # --- Projections ---
        # Project input to get queries
        # [B, N, C] -> [B, N, H*D]
        queries = self.query_proj(x)

        # Project memory copy to get keys and values
        # [M, C] -> [M, H*D]
        keys = self.key_proj(memory_copy)
        values = self.value_proj(memory_copy)

        # --- Reshape for Multi-Head Attention ---
        # Queries: [B, N, H*D] -> [B, N, H, D] -> [B, H, N, D]
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Keys: [M, H*D] -> [M, H, D] -> [H, M, D]
        memory_size = memory_copy.size(0)
        keys = keys.view(memory_size, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Values: [M, H*D] -> [M, H, D] -> [H, M, D]
        values = values.view(memory_size, self.num_heads, self.head_dim).permute(1, 0, 2)

        # --- Attention Computation ---
        # Decide whether to use chunking based on the number of tokens
        use_chunking = num_tokens > self.chunk_threshold
        
        if use_chunking:
            logger.debug(f"Using chunked attention for {num_tokens} tokens (threshold: {self.chunk_threshold}).")
            # Note: Chunked attention does not return full attention weights for efficiency
            max_scores, attended_values = self._chunked_attention(queries, keys, values, num_tokens)
            attention_weights = None # Cannot return full weights in chunked mode
            if return_attention:
                 logger.warning("Cannot return attention weights when using chunked computation.")

        else:
            max_scores, attended_values, attention_weights_raw = self._compute_attention(queries, keys, values)
            # Only return weights if requested
            attention_weights = attention_weights_raw if return_attention else None
            
        # --- Output Processing ---
        # Concatenate heads and project
        # [B, H, N, D] -> [B, N, H, D] -> [B, N, H*D]
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        output = attended_values.view(batch_size, num_tokens, self.num_heads * self.head_dim)
        
        # [B, N, H*D] -> [B, N, Cout]
        retrieved = self.output_proj(output)

        # --- Energy Calculation ---
        # Energy is based on the maximum attention score *before* softmax/normalization.
        # Lower energy (more negative score) indicates a better match.
        # Average energy across heads.
        energy = -torch.mean(max_scores, dim=1)  # [B, H, N] -> [B, N]

        # --- Memory Update (Detached) ---
        # If training and update_memory is enabled, update the memory bank
        # using the current input batch. This happens outside the computation graph.
        if self.training and self.update_memory:
            self._update_memory_detached(x) # Pass original input x

        return retrieved, energy, attention_weights

    def _compute_attention(self, 
                           queries: torch.Tensor, 
                           keys: torch.Tensor, 
                           values: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes attention scores, max scores, weights, and attended values."""
        # queries: [B, H, N, D], keys: [H, M, D], values: [H, M, D]
        
        # Calculate scaled dot-product attention scores
        # [B, H, N, D] x [H, D, M] -> [B, H, N, M]
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.scale

        # Apply beta scaling (temperature)
        # Using inplace ops can sometimes cause issues with autograd, prefer non-inplace
        attention_scores = attention_scores * self.beta

        # Get maximum score per query before softmax/normalization for energy calculation
        # Keepdims=False results in [B, H, N]
        max_scores, _ = torch.max(attention_scores, dim=-1)

        # Calculate attention weights
        if self.use_laplace_attention:
            # Laplace Attention: e^(-abs(beta * score / scale)) - More robust to outliers
            # Note: beta scaling is already applied. We use abs distance.
            # Using logsumexp trick for numerical stability might be better if needed,
            # but direct computation is often fine.
            attention_scores_norm = attention_scores - attention_scores.max(dim=-1, keepdim=True)[0] # Stabilize exp
            attention_weights = torch.exp(-torch.abs(attention_scores_norm)) # Use normalized scores here
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8) # Normalize row-wise
        else:
            # Standard Softmax Attention
            # Subtract max score for numerical stability before softmax
            attention_scores_stable = attention_scores - max_scores.unsqueeze(-1)
            attention_weights = F.softmax(attention_scores_stable, dim=-1) # [B, H, N, M]

        # Apply dropout to attention weights
        attention_weights_dropped = self.dropout(attention_weights)

        # Apply attention weights to values
        # [B, H, N, M] x [H, M, D] -> [B, H, N, D]
        attended_values = torch.matmul(attention_weights_dropped, values)

        return max_scores, attended_values, attention_weights # Return raw weights before dropout

    def _chunked_attention(self, 
                           queries: torch.Tensor, 
                           keys: torch.Tensor, 
                           values: torch.Tensor, 
                           num_tokens: int
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes attention in chunks along the token dimension to save memory."""
        # queries: [B, H, N, D], keys: [H, M, D], values: [H, M, D]
        chunk_size = self.chunk_threshold // 2 # Heuristic chunk size
        
        max_scores_list = []
        attended_values_list = []

        for i in range(0, num_tokens, chunk_size):
            end_idx = min(i + chunk_size, num_tokens)
            queries_chunk = queries[:, :, i:end_idx, :] # [B, H, chunk, D]

            # Compute attention scores for this chunk
            attention_scores_chunk = torch.matmul(queries_chunk, keys.transpose(-1, -2)) / self.scale
            attention_scores_chunk = attention_scores_chunk * self.beta # Apply beta

            # Get max scores for energy
            max_scores_chunk, _ = torch.max(attention_scores_chunk, dim=-1) # [B, H, chunk]
            max_scores_list.append(max_scores_chunk)

            # Compute attention weights for the chunk
            if self.use_laplace_attention:
                attention_scores_norm = attention_scores_chunk - attention_scores_chunk.max(dim=-1, keepdim=True)[0]
                attention_weights_chunk = torch.exp(-torch.abs(attention_scores_norm))
                attention_weights_chunk = attention_weights_chunk / (attention_weights_chunk.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                attention_scores_stable = attention_scores_chunk - max_scores_chunk.unsqueeze(-1)
                attention_weights_chunk = F.softmax(attention_scores_stable, dim=-1) # [B, H, chunk, M]
            
            # Apply dropout
            attention_weights_dropped = self.dropout(attention_weights_chunk)

            # Apply attention to values
            attended_values_chunk = torch.matmul(attention_weights_dropped, values) # [B, H, chunk, D]
            attended_values_list.append(attended_values_chunk)

            # Try to free memory explicitly (use with caution, might slow down)
            # del attention_scores_chunk, attention_weights_chunk, attention_weights_dropped, attention_scores_stable
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache() # Can impact performance

        # Concatenate chunks along the token dimension (dim=2)
        max_scores = torch.cat(max_scores_list, dim=2) # [B, H, N]
        attended_values = torch.cat(attended_values_list, dim=2) # [B, H, N, D]

        return max_scores, attended_values

    @torch.no_grad()
    def _initialize_memory_with_samples(self, x: torch.Tensor):
        """Initializes the memory bank using samples from the first input batch."""
        batch_size, num_tokens, _ = x.shape
        device = x.device

        # Flatten batch and token dimensions: [B, N, C] -> [B*N, C]
        flat_x = x.reshape(-1, self.input_dim).detach()

        # Sample 'memory_size' vectors
        num_available = flat_x.shape[0]
        if num_available == 0:
            logger.warning("Cannot initialize memory, input batch is empty.")
            # Keep the random initialization, mark as initialized to avoid re-attempt
            with self.memory_lock:
                 self.memory_initialized = True
            return

        if num_available >= self.memory_size:
            # Sufficient samples available, sample randomly without replacement
            indices = torch.randperm(num_available, device=device)[:self.memory_size]
            sampled_features = flat_x[indices]
        else:
            # Not enough unique samples, use all and repeat/sample with replacement
            logger.warning(f"Batch size ({num_available}) is smaller than memory size "
                           f"({self.memory_size}). Using repeated samples for initialization.")
            indices = torch.randint(0, num_available, (self.memory_size,), device=device)
            sampled_features = flat_x[indices]

        # Normalize if required
        if self.normalize_memory:
            sampled_features = F.normalize(sampled_features, p=2, dim=1)

        # Update memory bank safely
        with self.memory_lock:
            # Ensure the sampled features are on the same device as the memory buffer expects
            self._memory = sampled_features.to(self._memory.device) 
            self.memory_initialized = True
            logger.info(f"Memory bank initialized with {self._memory.shape[0]} samples.")


    @torch.no_grad()
    def _update_memory_detached(self, x: torch.Tensor):
        """
        Updates the memory bank with features from the current input batch.
        This operation is detached from the computation graph.

        Args:
            x (torch.Tensor): Input tensor [B, N, C] used to extract features.
        """
        batch_size, num_tokens, _ = x.shape
        device = x.device # Device of the incoming data

        # Flatten input: [B, N, C] -> [B*N, C]
        flat_features = x.reshape(-1, self.input_dim).detach()

        if flat_features.shape[0] == 0:
            # Nothing to update with
            return
            
        # Normalize features if the memory bank is normalized
        if self.normalize_memory:
            flat_features = F.normalize(flat_features, p=2, dim=1)

        # Determine number of memory slots to update
        num_updates = min(flat_features.size(0), int(self.memory_size * self.memory_update_fraction))
        
        if num_updates == 0:
            return # No updates needed or possible

        # --- Select features from the current batch to add to memory ---
        if flat_features.size(0) <= num_updates:
            # Use all available features if fewer than num_updates
            update_features = flat_features
        else:
            # Sample `num_updates` features randomly from the batch
            indices = torch.randperm(flat_features.size(0), device=device)[:num_updates]
            update_features = flat_features[indices]

        # Ensure update features are on the correct device before modifying memory
        update_features = update_features.to(self._memory.device)

        # --- Select memory slots to replace and update ---
        with self.memory_lock:
            if self.memory_update_strategy == 'fifo':
                # Replace oldest entries in a circular manner
                ptr = self._memory_pointer
                # Calculate indices to update
                indices = torch.arange(ptr, ptr + num_updates, device=self._memory.device) % self.memory_size
                self._memory[indices] = update_features
                # Update pointer
                self._memory_pointer = (ptr + num_updates) % self.memory_size

            elif self.memory_update_strategy == 'random':
                # Replace random entries
                indices = torch.randperm(self.memory_size, device=self._memory.device)[:num_updates]
                self._memory[indices] = update_features

            elif self.memory_update_strategy == 'diversity':
                # Replace memory slots that are 'least diverse' (most similar to others)
                # with the incoming features (which were randomly sampled).
                # This is a simplified diversity strategy. A more complex one might
                # try to add the *most diverse* incoming features.
                if num_updates < self.memory_size: # Avoid if replacing everything
                    # Compute pairwise similarity within the memory bank
                    # Ensure memory copy for calculation is on the compute device
                    memory_compute_device = update_features.device
                    memory_on_compute = self._memory.to(memory_compute_device)
                    
                    # Cosine similarity for normalized vectors, dot product otherwise
                    if self.normalize_memory:
                        sim_matrix = torch.mm(memory_on_compute, memory_on_compute.t())
                    else:
                         # Use normalized versions just for diversity check
                         mem_norm = F.normalize(memory_on_compute, p=2, dim=1)
                         sim_matrix = torch.mm(mem_norm, mem_norm.t())

                    # Diversity score: lower score means more similar to others (less diverse)
                    # Sum similarities, excluding self-similarity (diagonal)
                    diversity_scores = sim_matrix.sum(dim=1) - torch.diag(sim_matrix)

                    # Find indices of the *least* diverse (highest similarity sum) memory vectors
                    _, replace_indices = torch.topk(diversity_scores, k=num_updates, largest=True)
                    
                    # Ensure indices are on the memory's device
                    replace_indices = replace_indices.to(self._memory.device)

                    # Update memory
                    self._memory[replace_indices] = update_features
                else: # If num_updates >= memory_size, just replace randomly/all
                     indices = torch.randperm(self.memory_size, device=self._memory.device)[:num_updates]
                     self._memory[indices] = update_features

            else:
                # Should not happen due to init check, but fallback to random
                logger.warning(f"Unknown memory update strategy '{self.memory_update_strategy}', falling back to 'random'.")
                indices = torch.randperm(self.memory_size, device=self._memory.device)[:num_updates]
                self._memory[indices] = update_features
                
    def get_memory(self) -> torch.Tensor:
        """Returns a detached copy of the current memory bank."""
        with self.memory_lock:
            return self._memory.clone().detach()

    def set_memory(self, new_memory: torch.Tensor):
        """
        Replaces the current memory bank with a new one.
        The provided tensor will be detached and cloned.
        Normalization will be applied if self.normalize_memory is True.

        Args:
            new_memory (torch.Tensor): Tensor of shape [M, C] where M is the
                                       new memory size and C is input_dim.
        """
        if new_memory.dim() != 2 or new_memory.shape[1] != self.input_dim:
            raise ValueError(f"New memory must have shape [M, {self.input_dim}], "
                             f"got {new_memory.shape}")

        with self.memory_lock:
            new_mem_detached = new_memory.detach().clone()
            if self.normalize_memory:
                new_mem_detached = F.normalize(new_mem_detached, p=2, dim=1)

            # Update memory size attribute if it changes
            self.memory_size = new_mem_detached.shape[0]
            self._memory = new_mem_detached.to(self._memory.device) # Ensure it's on the module's device
            self.memory_initialized = True
            self._memory_pointer = 0 # Reset pointer
            logger.info(f"Memory bank explicitly set. New size: {self.memory_size}")

    def reset_memory(self):
        """Resets the memory bank to random values and marks it as uninitialized."""
        with self.memory_lock:
            logger.info("Resetting memory bank to random initialization.")
            # Determine device from existing parameters or memory
            try:
                target_device = next(self.parameters()).device
            except StopIteration: # No parameters yet
                target_device = self._memory.device
            
            new_memory = torch.randn(self.memory_size, self.input_dim, device=target_device)
            if self.normalize_memory:
                new_memory = F.normalize(new_memory, p=2, dim=1)
            self._memory = new_memory
            self.memory_initialized = False
            self._memory_pointer = 0
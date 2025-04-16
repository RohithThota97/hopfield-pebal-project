import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import psutil
import time
from typing import Dict, Tuple, Optional, Union, List

# Assuming hopfield_layer.py contains the ModernHopfieldLayer class
# from hopfield_layer import ModernHopfieldLayer 
# Placeholder if the actual import is missing/different
class ModernHopfieldLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, beta, memory_size, update_memory):
        super().__init__()
        # Dummy implementation for placeholder
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.beta = beta
        self.memory_size = memory_size
        self.update_memory = update_memory
        self.memory = nn.Parameter(torch.randn(memory_size, input_dim) * 0.01) # Small init
        # Dummy projection layers - replace with actual logic
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dummy forward pass - returns input and dummy energy
        # x shape: [B, SequenceLength, InputDim]
        # In a real implementation, this would perform attention with self.memory
        b, seq_len, _ = x.shape
        
        # Placeholder for attention mechanism
        # Project query, key, value
        q = self.query_proj(x)
        k = self.key_proj(self.memory) # Key from memory
        v = self.value_proj(self.memory) # Value from memory
        
        # Simplified attention score calculation (dot product)
        # attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.input_dim) # [B, SeqLen, MemorySize]
        # For energy calculation, often the dot product before softmax is used
        # Using query-memory dot product as a proxy for energy
        # This is a simplification, real energy might be calculated differently
        energy_proxy = torch.matmul(q, self.memory.transpose(-2, -1)) # [B, SeqLen, MemorySize]
        # Take max energy over memory slots as a representative energy per token
        # Note: The original paper might define energy differently (e.g., negative dot product)
        hopfield_energy = torch.max(energy_proxy, dim=-1)[0] # [B, SeqLen] 
        
        # Dummy output (just projecting input) - replace with actual attention output
        retrieved = self.out_proj(x)
        
        return retrieved, hopfield_energy # shape: [B, SeqLen, OutputDim], [B, SeqLen]

    def get_memory(self) -> torch.Tensor:
        return self.memory.data
    
    def set_memory(self, new_memory: torch.Tensor):
        assert new_memory.shape == self.memory.shape, "New memory shape mismatch"
        self.memory.data.copy_(new_memory)

# --- Memory Tracker ---
class MemoryTracker:
    """
    Utility class to track and report GPU and CPU memory usage.
    Provides peak memory usage tracking.
    """
    def __init__(self, log_interval: int = 10, verbose: bool = True):
        """
        Args:
            log_interval (int): Minimum time interval (in seconds) between logs.
            verbose (bool): Whether to print logs.
        """
        self.log_interval = log_interval
        self.verbose = verbose
        self.last_log_time = time.time()
        self.peak_gpu_mem = 0
        self.peak_cpu_mem = 0
        self._process = psutil.Process() # Cache process handle

    def _bytes_to_mb(self, b: int) -> float:
        return b / (1024 * 1024)

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Returns current and peak GPU memory usage in MB."""
        current_gpu_mem = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Ensure accurate measurement
            current_gpu_mem = self._bytes_to_mb(torch.cuda.memory_allocated())
            self.peak_gpu_mem = max(self.peak_gpu_mem, current_gpu_mem)
        return current_gpu_mem, self.peak_gpu_mem

    def get_cpu_memory_usage(self) -> Tuple[float, float]:
        """Returns current and peak CPU memory usage (RSS) in MB."""
        current_cpu_mem = self._bytes_to_mb(self._process.memory_info().rss)
        self.peak_cpu_mem = max(self.peak_cpu_mem, current_cpu_mem)
        return current_cpu_mem, self.peak_cpu_mem

    def log_memory_usage(self, operation_name: str = ""):
        """Logs memory usage if verbose and enough time has passed."""
        if not self.verbose:
            return
            
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            gpu_mem, peak_gpu = self.get_gpu_memory_usage()
            cpu_mem, peak_cpu = self.get_cpu_memory_usage()
            op_str = f" [{operation_name}]" if operation_name else ""
            print(f"[MemoryTracker]{op_str}: GPU {gpu_mem:.2f}MB (Peak: {peak_gpu:.2f}MB) | "
                  f"CPU {cpu_mem:.2f}MB (Peak: {peak_cpu:.2f}MB)")
            self.last_log_time = current_time

    def clear_memory(self, operation_name: str = ""):
        """Performs garbage collection and clears CUDA cache."""
        op_str = f" (Before clear {operation_name})" if operation_name else ""
        # self.log_memory_usage(f"Pre-Clear{op_str}") # Optional: log before clearing
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # self.log_memory_usage(f"Post-Clear{op_str}") # Optional: log after clearing

# --- Hopfield PEBAL Model ---
class HopfieldPEBALModel(nn.Module):
    """
    Integrates a Modern Hopfield Layer into a PEBAL-like segmentation model.

    Features:
    - Dynamic input dimension detection.
    - Optional channel adapter to match feature dimensions.
    - Configurable Hopfield layer insertion point.
    - Memory efficiency options (chunking, strided sampling, explicit clearing).
    - Computes multiple energy terms (Hopfield, feature-based, PEBAL).
    - Robust NaN/Inf handling.
    - Weight initialization.
    - Memory tracking utility.
    """
    def __init__(self, 
                 backbone: nn.Module, 
                 segmentation_head: nn.Module, 
                 num_classes: int = 19, 
                 hopfield_feature_dim: int = 256, 
                 hopfield_beta: float = 8.0, 
                 hopfield_memory_size: int = 1000, 
                 hopfield_num_heads: int = 4, 
                 insertion_point: str = 'after_backbone',
                 target_feature_dim: Optional[int] = None, # Target dim for seg head / hopfield input proj
                 use_efficient_memory: bool = True, 
                 chunk_size: int = 1000,
                 sampling_stride: int = 2,
                 memory_log_interval: int = 10,
                 memory_log_verbose: bool = True):
        """
        Args:
            backbone: CNN backbone for feature extraction (e.g., ResNet).
            segmentation_head: Segmentation head module.
            num_classes: Number of segmentation classes (excluding potential OOD class).
            hopfield_feature_dim: Internal feature dimension for the Hopfield layer.
            hopfield_beta: Beta parameter for Hopfield energy scaling.
            hopfield_memory_size: Size of the Hopfield memory bank.
            hopfield_num_heads: Number of attention heads in Hopfield layer.
            insertion_point: Where to insert Hopfield ('after_backbone' or 'after_seghead').
            target_feature_dim: Desired feature dimension before the segmentation head
                                (if insertion='after_backbone') or before the final classifier
                                (if insertion='after_seghead'). If None, it's inferred or
                                uses hopfield_feature_dim.
            use_efficient_memory: Enable memory-saving techniques like chunking and sampling.
            chunk_size: Size of chunks for processing large sequences in Hopfield.
            sampling_stride: Stride for spatial sampling before Hopfield if input is large.
            memory_log_interval: Interval in seconds for logging memory usage.
            memory_log_verbose: Enable memory logging.
        """
        super().__init__()
        
        assert insertion_point in ['after_backbone', 'after_seghead'], \
            "insertion_point must be 'after_backbone' or 'after_seghead'"
            
        self.backbone = backbone
        self.segmentation_head = segmentation_head
        self.num_classes = num_classes
        self.insertion_point = insertion_point
        self.use_efficient_memory = use_efficient_memory
        self.chunk_size = chunk_size
        self.sampling_stride = sampling_stride
        
        # Memory tracker for monitoring memory usage
        self.memory_tracker = MemoryTracker(log_interval=memory_log_interval, verbose=memory_log_verbose)
        
        # 1. Detect input feature dimension dynamically
        self._input_dim_after_feature_extractor = self._detect_feature_dimensions()
        if self._input_dim_after_feature_extractor is None:
            raise RuntimeError("Failed to detect feature dimensions automatically. "
                               "Ensure backbone and segmentation_head are compatible "
                               "or provide explicit dimensions.")
        
        print(f"Detected feature dimension after feature extractor: {self._input_dim_after_feature_extractor}")
        
        # 2. Determine the dimension right before Hopfield/Segmentation Head
        # This depends on the insertion point
        if self.insertion_point == 'after_backbone':
            dim_before_hopfield_or_seghead = self._input_dim_after_feature_extractor
            # If target_feature_dim is not specified, seg head must accept backbone output dim
            self._target_feature_dim = target_feature_dim if target_feature_dim is not None else dim_before_hopfield_or_seghead
        else: # 'after_seghead'
            # We need the dimension *after* the segmentation head
            self._input_dim_after_seghead = self._detect_feature_dimensions(after_seghead=True)
            if self._input_dim_after_seghead is None:
                 raise RuntimeError("Failed to detect feature dimensions after segmentation head.")
            print(f"Detected feature dimension after segmentation head: {self._input_dim_after_seghead}")
            dim_before_hopfield_or_seghead = self._input_dim_after_seghead
            # If target_feature_dim not specified, use the detected seg head output dim
            self._target_feature_dim = target_feature_dim if target_feature_dim is not None else dim_before_hopfield_or_seghead

        print(f"Effective dimension before Hopfield/SegHead/FinalClassifier: {dim_before_hopfield_or_seghead}")
        print(f"Target dimension for SegHead input (if after_backbone) or FinalClassifier input (if after_seghead): {self._target_feature_dim}")

        # 3. Add channel adapter if the detected dimension doesn't match the target
        self.needs_adapter = (dim_before_hopfield_or_seghead != self._target_feature_dim)
        self.channel_adapter = nn.Identity() # Default to no-op
        if self.needs_adapter:
            print(f"Adding Channel Adapter: {dim_before_hopfield_or_seghead} -> {self._target_feature_dim}")
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(dim_before_hopfield_or_seghead, self._target_feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self._target_feature_dim),
                # Using non-inplace ReLU is generally safer for gradient flow
                nn.ReLU(inplace=False) 
            )
        
        # Dimension going into the Hopfield input projection
        hopfield_proj_input_dim = self._target_feature_dim if self.needs_adapter else dim_before_hopfield_or_seghead
        
        # 4. Initialize Hopfield input projection
        self.hopfield_input_proj = nn.Conv2d(hopfield_proj_input_dim, hopfield_feature_dim, kernel_size=1)
        
        # 5. Initialize Hopfield layer
        self.hopfield = ModernHopfieldLayer(
            input_dim=hopfield_feature_dim,
            output_dim=hopfield_feature_dim, # Typically input_dim == output_dim
            num_heads=hopfield_num_heads,
            beta=hopfield_beta,
            memory_size=hopfield_memory_size,
            update_memory=True # Assume memory is updatable
        )
        
        # 6. Define subsequent layers based on insertion point
        if insertion_point == 'after_backbone':
            # Project Hopfield output back to the dimension expected by the segmentation head
            self.hopfield_output_proj = nn.Conv2d(hopfield_feature_dim, self._target_feature_dim, kernel_size=1)
            # The final classifier is implicitly within the segmentation_head
            self.final_classifier = None 
        else: # 'after_seghead'
            # Hopfield output is projected directly to class logits (+1 for OOD energy)
            self.final_classifier = nn.Conv2d(hopfield_feature_dim, self.num_classes + 1, kernel_size=1)
            # No separate output projection needed before the final classifier
            self.hopfield_output_proj = None 
            
        # 7. Energy computation head (operates on Hopfield output features)
        self.energy_head = nn.Sequential(
            nn.Conv2d(hopfield_feature_dim, hopfield_feature_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hopfield_feature_dim // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(hopfield_feature_dim // 2, 1, kernel_size=1) # Output a single energy channel
        )
        
        # 8. Initialize weights
        self._initialize_weights()

    def _detect_feature_dimensions(self, after_seghead: bool = False) -> Optional[int]:
        """Detect feature dimensions using a dummy forward pass."""
        try:
            # Use a reasonably small standard size to avoid OOM during detection
            # Using power-of-2 dimensions common in CNNs
            dummy_input = torch.zeros(1, 3, 64, 64, device=next(self.backbone.parameters()).device)
            
            self.eval() # Ensure model is in eval mode for detection (for dropout, BN)
            with torch.no_grad():
                features = self.backbone(dummy_input)
                
                # Handle backbones returning tuples/lists (e.g., intermediate features)
                if isinstance(features, (tuple, list)):
                    features = features[-1] # Assume the last feature map is the relevant one
                
                if after_seghead:
                    # Need to pass through segmentation head as well
                    seg_features = self.segmentation_head(features)
                    if isinstance(seg_features, (tuple, list)):
                       seg_features = seg_features[-1] # Assume last output
                    detected_dim = seg_features.shape[1]
                else:
                    detected_dim = features.shape[1]
            self.train() # Return model to train mode
            return detected_dim
        except Exception as e:
            print(f"Error during feature dimension detection ({'after seghead' if after_seghead else 'after backbone'}): {e}")
            print("Please ensure the backbone and segmentation head are correctly defined.")
            return None

    def _initialize_weights(self):
        """Initialize model weights for stability and better training dynamics."""
        print("Initializing weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
        # Special initialization for Hopfield-related layers
        if hasattr(self, 'hopfield') and self.hopfield is not None:
            # Initialize Hopfield projections with smaller std dev
            std_dev = 0.02 # Typical value for attention projections
            if hasattr(self.hopfield, 'query_proj'):
                 nn.init.normal_(self.hopfield.query_proj.weight, mean=0.0, std=std_dev)
            if hasattr(self.hopfield, 'key_proj'):
                 nn.init.normal_(self.hopfield.key_proj.weight, mean=0.0, std=std_dev)
            if hasattr(self.hopfield, 'value_proj'):
                 nn.init.normal_(self.hopfield.value_proj.weight, mean=0.0, std=std_dev)
            if hasattr(self.hopfield, 'out_proj') and self.hopfield.out_proj is not None:
                 nn.init.normal_(self.hopfield.out_proj.weight, mean=0.0, std=std_dev)

            # Initialize Hopfield memory with small random values
            with torch.no_grad():
                self.hopfield.memory.data.normal_(mean=0.0, std=0.01)
                
        # Initialize specific projections if they exist
        if hasattr(self, 'hopfield_input_proj'):
             nn.init.kaiming_normal_(self.hopfield_input_proj.weight, mode='fan_out', nonlinearity='relu')
             if self.hopfield_input_proj.bias is not None:
                 nn.init.constant_(self.hopfield_input_proj.bias, 0)
        if hasattr(self, 'hopfield_output_proj') and self.hopfield_output_proj is not None:
             nn.init.kaiming_normal_(self.hopfield_output_proj.weight, mode='fan_out', nonlinearity='relu')
             if self.hopfield_output_proj.bias is not None:
                 nn.init.constant_(self.hopfield_output_proj.bias, 0)
        if hasattr(self, 'final_classifier') and self.final_classifier is not None:
             nn.init.kaiming_normal_(self.final_classifier.weight, mode='fan_out', nonlinearity='relu')
             if self.final_classifier.bias is not None:
                 nn.init.constant_(self.final_classifier.bias, 0)
                 
        # Initialize channel adapter weights
        if isinstance(self.channel_adapter, nn.Sequential):
            for layer in self.channel_adapter:
                 if isinstance(layer, nn.Conv2d):
                     nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                     if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                 elif isinstance(layer, nn.BatchNorm2d):
                     nn.init.constant_(layer.weight, 1)
                     nn.init.constant_(layer.bias, 0)
                     
        print("Weight initialization complete.")

    def _apply_hopfield_processing(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function to apply Hopfield processing (projection, sampling, Hopfield, reshape).

        Args:
            features (torch.Tensor): Input features [B, C, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - retrieved features [B, C_hop, H_orig, W_orig]
                - hopfield energy [B, H_orig, W_orig]
                - feature energy [B, 1, H_orig, W_orig]
        """
        b, c_in, h_in, w_in = features.shape
        
        # 1. Project features for Hopfield
        hopfield_input = self.hopfield_input_proj(features)
        self.memory_tracker.log_memory_usage("After Hopfield Input Projection")
        
        c_hop = hopfield_input.shape[1] # Hopfield feature dimension
        
        # 2. Apply spatial sampling if features are large and memory efficiency is enabled
        h_proc, w_proc = h_in, w_in
        hopfield_input_sampled = hopfield_input
        needs_upsampling = False
        # Heuristic: Sample if number of spatial locations is large (e.g., > 64*64)
        if self.use_efficient_memory and h_in * w_in > 64 * 64: 
            stride = self.sampling_stride
            if h_in > stride and w_in > stride:
                hopfield_input_sampled = hopfield_input[:, :, ::stride, ::stride]
                h_proc, w_proc = hopfield_input_sampled.shape[2:]
                needs_upsampling = True
                self.memory_tracker.log_memory_usage(f"After Strided Sampling (stride {stride})")
            else:
                print(f"Warning: Input size {h_in}x{w_in} is large, but cannot apply stride {stride}. Processing full size.")


        # 3. Reshape for Hopfield Layer (expects [B, SequenceLength, FeatureDim])
        # SequenceLength = h_proc * w_proc
        hopfield_input_flat = hopfield_input_sampled.reshape(b, c_hop, -1).permute(0, 2, 1)  # [B, H_proc*W_proc, C_hop]
        
        # 4. Process with Hopfield Layer (potentially in chunks)
        retrieved_flat: torch.Tensor
        hopfield_energy_flat: torch.Tensor # Expected shape [B, H_proc*W_proc]

        num_tokens = hopfield_input_flat.shape[1]
        if self.use_efficient_memory and num_tokens > self.chunk_size:
            # Process in chunks
            retrieved_chunks = []
            energy_chunks = []
            print(f"Processing {num_tokens} tokens in chunks of size {self.chunk_size}")
            
            for i in range(0, num_tokens, self.chunk_size):
                end = min(i + self.chunk_size, num_tokens)
                chunk = hopfield_input_flat[:, i:end, :]
                
                chunk_retrieved, chunk_energy = self.hopfield(chunk) # [B, chunk_len, C_hop], [B, chunk_len]
                
                retrieved_chunks.append(chunk_retrieved)
                energy_chunks.append(chunk_energy)
                
                # Explicitly clear chunk variables and cache
                del chunk, chunk_retrieved, chunk_energy
                self.memory_tracker.clear_memory(f"Chunk {i//self.chunk_size + 1}")
            
            # Concatenate results
            retrieved_flat = torch.cat(retrieved_chunks, dim=1)
            hopfield_energy_flat = torch.cat(energy_chunks, dim=1)
            self.memory_tracker.log_memory_usage("After Chunked Hopfield Processing")
            
        else:
            # Process all at once
            retrieved_flat, hopfield_energy_flat = self.hopfield(hopfield_input_flat)
            self.memory_tracker.log_memory_usage("After Full Hopfield Processing")

        # 5. Reshape back to spatial dimensions [B, C_hop, H_proc, W_proc]
        retrieved_spatial = retrieved_flat.permute(0, 2, 1).reshape(b, c_hop, h_proc, w_proc)
        
        # 6. Calculate feature-based energy from retrieved features
        feature_energy_spatial = self.energy_head(retrieved_spatial) # [B, 1, H_proc, W_proc]
        
        # 7. Upsample retrieved features and energies if strided sampling was used
        if needs_upsampling:
            retrieved = F.interpolate(retrieved_spatial, 
                                      size=(h_in, w_in), 
                                      mode='bilinear', 
                                      align_corners=False)
            feature_energy = F.interpolate(feature_energy_spatial, 
                                           size=(h_in, w_in), 
                                           mode='bilinear', 
                                           align_corners=False)
            # Reshape and upsample hopfield energy
            hopfield_energy_map = hopfield_energy_flat.reshape(b, 1, h_proc, w_proc)
            hopfield_energy = F.interpolate(hopfield_energy_map,
                                            size=(h_in, w_in),
                                            mode='bilinear',
                                            align_corners=False).squeeze(1) # [B, H_in, W_in]
            self.memory_tracker.log_memory_usage("After Upsampling Hopfield Outputs")
        else:
            retrieved = retrieved_spatial
            feature_energy = feature_energy_spatial
            hopfield_energy = hopfield_energy_flat.reshape(b, h_in, w_in) # [B, H_in, W_in]
        
        # Clear intermediate flat tensors
        del hopfield_input_flat, retrieved_flat, hopfield_energy_flat
        del hopfield_input_sampled, hopfield_input # Also clear inputs
        self.memory_tracker.clear_memory("Hopfield Processing Cleanup")
            
        return retrieved, hopfield_energy, feature_energy

    def _check_and_handle_nan_inf(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Checks for NaN/Inf in a tensor and replaces them with zeros."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: NaN/Inf detected in '{name}'. Replacing with zeros.")
            # Keep track of how many NaNs/Infs were replaced
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            print(f"NaN count: {nan_count}, Inf count: {inf_count}")
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor),
                                 torch.zeros_like(tensor),
                                 tensor)
        return tensor

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Hopfield PEBAL model.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, H, W].

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'logits': Raw segmentation logits [B, NumClasses+1, H_out, W_out].
                - 'hopfield_energy': Energy from Hopfield associations [B, 1, H_out, W_out].
                - 'feature_energy': Energy from the convolutional energy head [B, 1, H_out, W_out].
                - 'pebal_energy': Energy derived from logits (logsumexp) [B, 1, H_out, W_out].
                - 'combined_energy': Weighted sum of energy components [B, 1, H_out, W_out].
        """
        self.memory_tracker.log_memory_usage("Forward Start")
        
        # --- Stage 1: Feature Extraction ---
        # Handle potential OOM in backbone with input resizing (if enabled)
        try:
            features = self.backbone(x)
            if isinstance(features, (tuple, list)):
                 features = features[-1] # Assume last feature map
            self.memory_tracker.log_memory_usage("After Backbone")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.use_efficient_memory:
                print(f"OOM Error in backbone: {e}. Trying with reduced input size (quality may degrade).")
                self.memory_tracker.clear_memory("OOM Fallback")
                # Reduce spatial size, run backbone, then upsample features
                # Note: This changes the effective receptive field and might impact results
                x_small = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                features_small = self.backbone(x_small)
                if isinstance(features_small, (tuple, list)):
                    features_small = features_small[-1]
                
                # Determine target feature map size (usually stride 4 or 8 from input)
                # This is a heuristic; actual stride depends on the backbone
                downsample_factor = x.shape[2] / features_small.shape[2] 
                target_h = x.shape[2] // downsample_factor
                target_w = x.shape[3] // downsample_factor
                
                features = F.interpolate(features_small, size=(int(target_h), int(target_w)), 
                                       mode='bilinear', align_corners=False)
                del x_small, features_small # Clean up intermediate tensors
                self.memory_tracker.clear_memory("OOM Fallback Cleanup")
                self.memory_tracker.log_memory_usage("After Backbone (OOM Fallback)")
            else:
                # Re-raise error if not OOM or efficiency disabled
                raise e

        # Store original feature shape for potential upsampling later
        b, _, h_feat, w_feat = features.shape
        
        # --- Stage 2: Apply Adapter (if needed) ---
        # Adapter is applied *before* Hopfield if insertion='after_backbone',
        # or *before* segmentation head if insertion='after_seghead'.
        # For simplicity, we apply it early if needed, regardless of insertion point,
        # assuming the seg head or Hopfield proj expects `self._target_feature_dim`.
        
        if self.insertion_point == 'after_backbone':
            features = self.channel_adapter(features)
            #print(f"Shape after adapter (pre-Hopfield): {features.shape}")
            self.memory_tracker.log_memory_usage("After Channel Adapter (pre-Hopfield)")
        
        # --- Stage 3: Hopfield or Segmentation Head Processing ---
        if self.insertion_point == 'after_backbone':
            # Apply Hopfield processing after backbone (and adapter)
            retrieved, hopfield_energy_map, feature_energy_map = self._apply_hopfield_processing(features)
            # retrieved shape: [B, C_hop, H_feat, W_feat]
            # hopfield_energy_map shape: [B, H_feat, W_feat]
            # feature_energy_map shape: [B, 1, H_feat, W_feat]
            
            # Project Hopfield output back to dimension expected by segmentation head
            hopfield_output = self.hopfield_output_proj(retrieved)
            self.memory_tracker.log_memory_usage("After Hopfield Output Projection")
            
            # Residual connection: Add Hopfield refinement to original (adapted) features
            updated_features = features + hopfield_output
            
            # Clear intermediate variables
            if self.use_efficient_memory:
                del features, retrieved, hopfield_output
                self.memory_tracker.clear_memory("After Backbone Path Cleanup")
            
            # Pass updated features through the segmentation head
            logits = self.segmentation_head(updated_features)
            if isinstance(logits, (tuple, list)):
                logits = logits[-1] # Assume last output is logits
            self.memory_tracker.log_memory_usage("After Segmentation Head")
            
            # Ensure energies match logit spatial dimensions
            h_out, w_out = logits.shape[-2:]
            if hopfield_energy_map.shape[-2:] != (h_out, w_out):
                 hopfield_energy = F.interpolate(hopfield_energy_map.unsqueeze(1), size=(h_out, w_out), mode='bilinear', align_corners=False) # Add channel dim, interp, remove
            else:
                 hopfield_energy = hopfield_energy_map.unsqueeze(1) # Add channel dim
                 
            if feature_energy_map.shape[-2:] != (h_out, w_out):
                 feature_energy = F.interpolate(feature_energy_map, size=(h_out, w_out), mode='bilinear', align_corners=False)
            else:
                 feature_energy = feature_energy_map

        else: # insertion_point == 'after_seghead'
            # Pass features through segmentation head first
            seg_features = self.segmentation_head(features)
            if isinstance(seg_features, (tuple, list)):
                seg_features = seg_features[-1] # Assume last output
            self.memory_tracker.log_memory_usage("After Segmentation Head")
            
            # Clean up backbone features
            if self.use_efficient_memory:
                del features
                self.memory_tracker.clear_memory("After SegHead Path Cleanup 1")
            
            # Apply adapter (if needed) to segmentation features
            seg_features = self.channel_adapter(seg_features)
            #print(f"Shape after adapter (pre-Hopfield): {seg_features.shape}")
            self.memory_tracker.log_memory_usage("After Channel Adapter (pre-Hopfield)")

            # Apply Hopfield processing to segmentation features
            retrieved, hopfield_energy_map, feature_energy_map = self._apply_hopfield_processing(seg_features)
            # retrieved shape: [B, C_hop, H_seg, W_seg]
            # hopfield_energy_map shape: [B, H_seg, W_seg]
            # feature_energy_map shape: [B, 1, H_seg, W_seg]

            # Clean up intermediate seg features
            if self.use_efficient_memory:
                del seg_features
                self.memory_tracker.clear_memory("After SegHead Path Cleanup 2")

            # Apply final classifier to Hopfield output
            logits = self.final_classifier(retrieved)
            self.memory_tracker.log_memory_usage("After Final Classifier")
            
            # Ensure energies match logit spatial dimensions
            h_out, w_out = logits.shape[-2:]
            if hopfield_energy_map.shape[-2:] != (h_out, w_out):
                 hopfield_energy = F.interpolate(hopfield_energy_map.unsqueeze(1), size=(h_out, w_out), mode='bilinear', align_corners=False)
            else:
                 hopfield_energy = hopfield_energy_map.unsqueeze(1)

            if feature_energy_map.shape[-2:] != (h_out, w_out):
                 feature_energy = F.interpolate(feature_energy_map, size=(h_out, w_out), mode='bilinear', align_corners=False)
            else:
                 feature_energy = feature_energy_map

        # --- Stage 4: Energy Calculation & NaN/Inf Handling ---
        # Calculate PEBAL energy (negative log-sum-exp over in-distribution classes)
        # Assumes the last logit channel corresponds to the OOD class/energy.
        pebal_energy = -torch.logsumexp(logits[:, :self.num_classes, :, :], dim=1, keepdim=True)
        
        # Check and handle NaN/Inf in individual components before combining
        logits = self._check_and_handle_nan_inf(logits, "logits")
        hopfield_energy = self._check_and_handle_nan_inf(hopfield_energy, "hopfield_energy")
        feature_energy = self._check_and_handle_nan_inf(feature_energy, "feature_energy")
        pebal_energy = self._check_and_handle_nan_inf(pebal_energy, "pebal_energy")

        # Combine energy terms (using simple averaging weights for now)
        # Weights could be learned parameters or configurable
        # Note: Ensure energies have compatible shapes [B, 1, H, W]
        # hopfield_energy is [B, 1, H, W], feature_energy is [B, 1, H, W], pebal_energy is [B, 1, H, W]
        combined_energy = pebal_energy + 0.5 * feature_energy + 0.5 * hopfield_energy
        
        # Clamp combined energy to prevent extreme values after combination
        combined_energy = torch.clamp(combined_energy, min=-100.0, max=100.0)
        combined_energy = self._check_and_handle_nan_inf(combined_energy, "combined_energy")

        self.memory_tracker.log_memory_usage("Forward End")
        
        return {
            'logits': logits,                     # [B, NumClasses+1, H_out, W_out]
            'hopfield_energy': hopfield_energy,   # [B, 1, H_out, W_out]
            'feature_energy': feature_energy,    # [B, 1, H_out, W_out]
            'pebal_energy': pebal_energy,        # [B, 1, H_out, W_out]
            'combined_energy': combined_energy   # [B, 1, H_out, W_out]
        }

    def _prepare_features_for_memory_update(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Prepares features (extraction, adapter, projection, sampling) for Hopfield memory update.
        Handles different insertion points and memory efficiency.

        Args:
            features (torch.Tensor): Input features, typically from the backbone [B, C, H, W].

        Returns:
            Optional[torch.Tensor]: Flattened features suitable for memory update [N, C_hop],
                                     or None if preparation fails.
        """
        self.memory_tracker.log_memory_usage("Start Memory Feature Prep")
        
        with torch.no_grad(): # Ensure no gradients are computed during prep
            # Determine the features to project based on insertion point
            if self.insertion_point == 'after_backbone':
                # Apply adapter if needed
                if self.needs_adapter:
                     features_to_project = self.channel_adapter(features)
                else:
                     features_to_project = features
            else: # 'after_seghead'
                # Pass through segmentation head first
                try:
                    seg_features = self.segmentation_head(features)
                    if isinstance(seg_features, (tuple, list)):
                         seg_features = seg_features[-1]
                except Exception as e:
                    print(f"Error in segmentation_head during memory prep: {e}")
                    return None
                    
                # Apply adapter if needed
                if self.needs_adapter:
                     features_to_project = self.channel_adapter(seg_features)
                else:
                     features_to_project = seg_features
            
            self.memory_tracker.log_memory_usage("Memory Prep: After SegHead/Adapter")

            # Project features to Hopfield dimension
            try:
                hopfield_input = self.hopfield_input_proj(features_to_project)
            except Exception as e:
                print(f"Error in hopfield_input_proj during memory prep: {e}")
                return None

            # Apply spatial sampling if needed (similar logic to forward pass)
            b, c_hop, h_in, w_in = hopfield_input.shape
            hopfield_input_sampled = hopfield_input
            
            if self.use_efficient_memory and h_in * w_in > 64 * 64: 
                 stride = self.sampling_stride
                 if h_in > stride and w_in > stride:
                     hopfield_input_sampled = hopfield_input[:, :, ::stride, ::stride]
                     self.memory_tracker.log_memory_usage(f"Memory Prep: After Sampling (stride {stride})")
                 else:
                     print(f"Warning (Memory Prep): Input size {h_in}x{w_in} large, but cannot apply stride {stride}.")

            # Reshape for memory update (flatten spatial dims)
            # [B, C_hop, H_proc*W_proc] -> [B*H_proc*W_proc, C_hop]
            hopfield_input_flat = hopfield_input_sampled.permute(0, 2, 3, 1).reshape(-1, c_hop)
            
            self.memory_tracker.log_memory_usage("End Memory Feature Prep")
            
            # Detach from computation graph before returning
            return hopfield_input_flat.detach()

    def update_memory(self, input_features: torch.Tensor, max_samples: int = 5000):
        """
        Updates the Hopfield memory bank with new features derived from input.

        Args:
            input_features (torch.Tensor): Input tensor to the model (e.g., image batch [B, 3, H, W])
                                          or pre-extracted features [B, C, H, W].
                                          If it's an image batch, backbone features will be extracted first.
            max_samples (int): Maximum number of feature vectors to sample for the update.
        """
        if not self.hopfield.update_memory:
            print("Warning: Hopfield layer is not configured to update memory.")
            return

        self.eval() # Use eval mode for feature extraction consistency
        with torch.no_grad():
            # Step 1: Extract features if raw input is given
            # Check if input_features look like image batch [B, 3, H, W]
            if len(input_features.shape) == 4 and input_features.shape[1] == 3:
                 try:
                     features = self.backbone(input_features)
                     if isinstance(features, (tuple, list)):
                         features = features[-1]
                 except Exception as e:
                     print(f"Error extracting backbone features for memory update: {e}")
                     self.train() # Return to train mode
                     return
            # Check if input features match backbone output dim (heuristic)
            elif len(input_features.shape) == 4 and input_features.shape[1] == self._input_dim_after_feature_extractor:
                 features = input_features # Assume pre-extracted features
            else:
                 print(f"Error: Unexpected input shape for memory update: {input_features.shape}. "
                       f"Expected [B, 3, H, W] or [B, {self._input_dim_after_feature_extractor}, H, W].")
                 self.train()
                 return

            # Step 2: Prepare features (adapter, projection, sampling, flattening)
            flat_features = self._prepare_features_for_memory_update(features) # [N, C_hop]
            
            if flat_features is None or flat_features.shape[0] == 0:
                 print("Memory update skipped: No valid features prepared.")
                 self.train() # Return to train mode
                 return

            num_available_features = flat_features.shape[0]
            
            # Step 3: Sample a subset of features for the update
            sample_size = min(num_available_features, max_samples, self.hopfield.memory_size)
            if sample_size < num_available_features:
                 indices = torch.randperm(num_available_features, device=flat_features.device)[:sample_size]
                 sampled_features = flat_features[indices]
            else:
                 sampled_features = flat_features
                 
            # Step 4: Update the Hopfield memory
            current_memory = self.hopfield.get_memory() # Get current memory [MemSize, C_hop]
            memory_device = current_memory.device
            num_memory_slots = current_memory.shape[0]
            
            # Determine indices in the memory bank to replace
            # Simple strategy: replace random slots
            replace_indices = torch.randperm(num_memory_slots, device=memory_device)[:sample_size]
            
            # Create a copy to modify (avoids potential in-place issues if memory is used elsewhere)
            # Perform update on the correct device
            new_memory = current_memory.clone()
            new_memory[replace_indices] = sampled_features.to(memory_device) # Ensure features are on same device
            
            # Set the updated memory in the Hopfield layer
            self.hopfield.set_memory(new_memory)
            
            print(f"Hopfield memory updated with {sample_size} feature vectors.")
            self.memory_tracker.log_memory_usage("End Memory Update")

            # Clean up large intermediate tensors explicitly if needed
            del features, flat_features, sampled_features, new_memory, current_memory
            self.memory_tracker.clear_memory("Memory Update Cleanup")

        self.train() # Return model to train mode after update


# --- Example Usage (requires dummy modules) ---
if __name__ == '__main__':

    # Dummy Backbone (e.g., simulating ResNet output)
    class DummyBackbone(nn.Module):
        def __init__(self, out_channels=2048):
            super().__init__()
            self.out_channels = out_channels
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer4 = nn.Conv2d(64, out_channels, kernel_size=1) # Simplified final layer

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            # Simulate more layers resulting in some feature map size
            # This dummy doesn't replicate actual ResNet feature sizes well
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False) 
            x = self.layer4(x)
            return x

    # Dummy Segmentation Head (e.g., simulating DeepLab head)
    class DummySegHead(nn.Module):
        def __init__(self, in_channels, num_classes, head_channels=256):
            super().__init__()
            # Simplified: just projects to num_classes + 1
            # A real head (like ASPP) would be more complex
            self.classifier = nn.Conv2d(in_channels, num_classes + 1, kernel_size=1) 
            # A more realistic head might output features first
            self.feature_conv = nn.Conv2d(in_channels, head_channels, kernel_size=1)
            self.final_classifier = nn.Conv2d(head_channels, num_classes + 1, kernel_size=1)
            self._output_features = False # Control output type

        def forward(self, x):
            if self._output_features:
                # Simulate outputting features before final classification
                return self.feature_conv(x) 
            else:
                # Normal operation: output logits
                # Use final_classifier path to be consistent with _output_features logic
                features = self.feature_conv(x)
                return self.final_classifier(features)

    # Configuration
    NUM_CLASSES = 19 # Cityscapes
    BACKBONE_OUT_DIM = 2048 # Example ResNet output
    SEG_HEAD_FEATURE_DIM = 256 # Example DeepLab head feature dim
    HOPFIELD_DIM = 128
    TARGET_DIM_AFTER_BACKBONE = 512 # Example target dim for seg head input
    
    # Create dummy modules
    backbone = DummyBackbone(out_channels=BACKBONE_OUT_DIM)
    seg_head = DummySegHead(in_channels=TARGET_DIM_AFTER_BACKBONE, num_classes=NUM_CLASSES, head_channels=SEG_HEAD_FEATURE_DIM) # Head expects TARGET_DIM

    # --- Test Case 1: Insert Hopfield after Backbone ---
    print("\n--- Testing Model: Hopfield after Backbone ---")
    model_after_backbone = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=seg_head, # Seg head expects TARGET_DIM
        num_classes=NUM_CLASSES,
        hopfield_feature_dim=HOPFIELD_DIM,
        target_feature_dim=TARGET_DIM_AFTER_BACKBONE, # Specify target for seg head
        insertion_point='after_backbone',
        use_efficient_memory=True,
        memory_log_verbose=True
    )

    # Create dummy input
    # Use larger size to potentially trigger memory optimizations
    dummy_input = torch.randn(2, 3, 256, 512) 
    if torch.cuda.is_available():
        print("Moving model and data to GPU")
        model_after_backbone.cuda()
        dummy_input = dummy_input.cuda()

    # Test forward pass
    print("\nTesting forward pass (after_backbone)...")
    model_after_backbone.train() # Set to train mode
    output_dict = model_after_backbone(dummy_input)
    print("Output keys:", output_dict.keys())
    for key, tensor in output_dict.items():
        print(f"  {key}: shape={tensor.shape}, device={tensor.device}")
        
    # Test memory update
    print("\nTesting memory update (after_backbone)...")
    # Use a smaller input for memory update if forward pass used large input
    mem_update_input = torch.randn(4, 3, 128, 256) 
    if torch.cuda.is_available():
        mem_update_input = mem_update_input.cuda()
    model_after_backbone.update_memory(mem_update_input) 
    
    del model_after_backbone, output_dict # Cleanup
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    # --- Test Case 2: Insert Hopfield after SegHead ---
    print("\n--- Testing Model: Hopfield after SegHead ---")
    # For this case, the seg head needs to output features, not logits
    seg_head_features = DummySegHead(in_channels=BACKBONE_OUT_DIM, num_classes=NUM_CLASSES, head_channels=SEG_HEAD_FEATURE_DIM)
    seg_head_features._output_features = True # Make it output features

    # TARGET_DIM is now the dim *after* the seg_head features, before the final classifier
    TARGET_DIM_AFTER_SEGHEAD = SEG_HEAD_FEATURE_DIM # We don't need an adapter here

    model_after_seghead = HopfieldPEBALModel(
        backbone=backbone, # Backbone outputs BACKBONE_OUT_DIM
        segmentation_head=seg_head_features, # Seg head outputs SEG_HEAD_FEATURE_DIM
        num_classes=NUM_CLASSES,
        hopfield_feature_dim=HOPFIELD_DIM,
        target_feature_dim=TARGET_DIM_AFTER_SEGHEAD, # Target dim before final classifier (matches seg head output)
        insertion_point='after_seghead',
        use_efficient_memory=True,
        memory_log_verbose=True
    )
    
    if torch.cuda.is_available():
        print("Moving model and data to GPU")
        model_after_seghead.cuda()
        # dummy_input is already on GPU if available

    # Test forward pass
    print("\nTesting forward pass (after_seghead)...")
    model_after_seghead.train()
    output_dict_seg = model_after_seghead(dummy_input)
    print("Output keys:", output_dict_seg.keys())
    for key, tensor in output_dict_seg.items():
        print(f"  {key}: shape={tensor.shape}, device={tensor.device}")
        
    # Test memory update
    print("\nTesting memory update (after_seghead)...")
    model_after_seghead.update_memory(mem_update_input)

    del model_after_seghead, output_dict_seg, dummy_input, mem_update_input # Cleanup
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    
    print("\nExample Usage Complete.")
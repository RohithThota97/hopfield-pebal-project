# hopfield_pebal_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import psutil
import time
from typing import Dict, Tuple, Optional, Union, List
import logging # Added logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use module-specific logger

# --- ModernHopfieldLayer Placeholder ---
class ModernHopfieldLayer(nn.Module):
    # (Code from previous response)
    def __init__(self, input_dim, output_dim, num_heads, beta, memory_size, update_memory):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.beta = beta
        self.memory_size = memory_size
        self.update_memory = update_memory
        self.register_buffer('memory', torch.randn(memory_size, input_dim) * 0.01)
        self.register_buffer('memory_ptr', torch.tensor(0, dtype=torch.long)) # Keep track of insertion point

        # Using ModuleList for projections is often cleaner
        self.projections = nn.ModuleDict({
            'query': nn.Linear(input_dim, input_dim),
            'key': nn.Linear(input_dim, input_dim),
            'value': nn.Linear(input_dim, input_dim),
            'out': nn.Linear(input_dim, output_dim)
        })

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dummy forward pass - returns input projected and dummy energy
        b, seq_len, d_in = x.shape
        mem = self.memory # [MemSize, InputDim]

        # Project query, key, value
        q = self.projections['query'](x)            # [B, SeqLen, InputDim]
        k = self.projections['key'](mem)            # [MemSize, InputDim]
        v = self.projections['value'](mem)          # [MemSize, InputDim]

        # Simplified attention score calculation (dot product as energy proxy)
        # [B, SeqLen, InputDim] @ [InputDim, MemSize] -> [B, SeqLen, MemSize]
        energy_proxy = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(d_in) # Add scaling

        # Take max energy over memory slots as a representative energy per token
        hopfield_energy = torch.max(energy_proxy, dim=-1)[0] # [B, SeqLen]

        # Dummy output (just projecting input) - Replace with actual attention output logic
        # E.g., Calculate attention weights (softmax over energy_proxy)
        # attn_weights = F.softmax(energy_proxy * self.beta, dim=-1) # Apply beta (temp scaling)
        # retrieved_attn = torch.matmul(attn_weights, v) # Apply attention to values
        # retrieved = self.projections['out'](retrieved_attn) # Project output

        # For now, just use the output projection on the input (acting more like a feed-forward)
        # This might not be the intended Hopfield behavior but matches the dummy structure
        retrieved = self.projections['out'](q) # Project the query itself

        return retrieved, hopfield_energy

    @torch.no_grad()
    def get_memory(self) -> torch.Tensor:
        return self.memory.data

    @torch.no_grad()
    def set_memory(self, new_memory: torch.Tensor):
        if not hasattr(self, 'memory'):
            logger.warning("Memory buffer not initialized in Hopfield.")
            return
        if new_memory.shape != self.memory.shape:
            logger.warning(f"New memory shape mismatch {new_memory.shape} vs {self.memory.shape}. Reinitializing.")
            # Re-register buffer if shape changes drastically
            self.register_buffer('memory', new_memory.clone())
        else:
            self.memory.copy_(new_memory)

    @torch.no_grad()
    def update_memory_bank(self, new_features: torch.Tensor):
        """Updates memory using a circular buffer strategy."""
        if not self.update_memory or not hasattr(self, 'memory') or not hasattr(self, 'memory_ptr'):
            return

        n = new_features.shape[0]
        mem_size = self.memory_size
        ptr = self.memory_ptr.item() # Get current pointer value

        if n == 0: return

        # Ensure features are on the same device as memory
        try:
            target_device = self.memory.device
            new_features = new_features.to(target_device)
        except Exception as e:
            logger.error(f"Failed to move features to memory device {target_device}: {e}")
            return

        # Calculate indices to insert/overwrite
        indices = torch.arange(ptr, ptr + n, device=target_device) % mem_size

        self.memory[indices] = new_features

        # Update pointer
        self.memory_ptr.copy_((ptr + n) % mem_size)
        logger.debug(f"Hopfield memory bank updated. Pointer at {self.memory_ptr.item()}. Input features: {n}")


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
        try: # Handle cases where psutil might not be available or fail
            self._process = psutil.Process()
        except (ImportError, psutil.NoSuchProcess):
            logger.warning("psutil unavailable/failed. CPU mem tracking disabled.")
            self._process = None

    def _bytes_to_mb(self, b: int) -> float:
        return b / (1024 * 1024)

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Returns current and peak GPU memory usage in MB."""
        current_gpu_mem = 0
        if torch.cuda.is_available():
            # Synchronization can be slow, use less frequently if needed
            # torch.cuda.synchronize()
            try:
                current_gpu_mem = self._bytes_to_mb(torch.cuda.memory_allocated())
                # Peak memory tracking in PyTorch can be complex; using current max as proxy
                self.peak_gpu_mem = max(self.peak_gpu_mem, current_gpu_mem)
            except Exception as e:
                 logger.warning(f"Could not get GPU mem: {e}")
                 # Maybe check torch.cuda.memory_stats() for more detailed info if needed
        return current_gpu_mem, self.peak_gpu_mem

    def get_cpu_memory_usage(self) -> Tuple[float, float]:
        """Returns current and peak CPU memory usage (RSS) in MB."""
        current_cpu_mem = 0
        if self._process:
            try:
                current_cpu_mem = self._bytes_to_mb(self._process.memory_info().rss)
                self.peak_cpu_mem = max(self.peak_cpu_mem, current_cpu_mem)
            except psutil.NoSuchProcess: # Handle case where process might terminate
                logger.warning("Process terminated. Disabling CPU memory tracking.")
                self._process = None # Stop trying if process is gone
            except Exception as e:
                 logger.warning(f"Could not get CPU mem: {e}") if not isinstance(e, psutil.NoSuchProcess) else None

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
            logger.info(f"[MemoryTracker]{op_str}: GPU {gpu_mem:.1f}MB (Peak: {peak_gpu:.1f}MB) | "
                        f"CPU {cpu_mem:.1f}MB (Peak: {peak_cpu:.1f}MB)") # Use logger.info
            self.last_log_time = current_time

    def clear_memory(self, operation_name: str = ""):
        """Performs garbage collection and clears CUDA cache."""
        op_str = f" Cleared after {operation_name}" if operation_name else " Clearing memory"
        self.log_memory_usage(f"Pre-Clear{op_str}") # Log before
        n = gc.collect() # Collect garbage
        if n > 0:
            logger.debug(f"[MemoryTracker] GC collected {n} objects.") # Use logger.debug
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
            except Exception as e:
                logger.warning(f"Could not empty CUDA cache: {e}")
        self.log_memory_usage(f"Post-Clear{op_str}") # Log after


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
                 target_feature_dim: Optional[int] = None, # IMPORTANT ARGUMENT
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
                                uses a default reduced value (e.g., 512).
            use_efficient_memory: Enable memory-saving techniques like chunking and sampling.
            chunk_size: Size of chunks for processing large sequences in Hopfield.
            sampling_stride: Stride for spatial sampling before Hopfield if input is large.
            memory_log_interval: Interval in seconds for logging memory usage.
            memory_log_verbose: Enable memory logging.
        """
        super().__init__()

        assert insertion_point in ['after_backbone', 'after_seghead'], \
            "insertion_point must be 'after_backbone' or 'after_seghead'"

        # Store base components
        self.backbone = backbone
        self.segmentation_head = segmentation_head # Store original head
        self.num_classes = num_classes
        self.insertion_point = insertion_point
        self.use_efficient_memory = use_efficient_memory
        self.chunk_size = chunk_size
        self.sampling_stride = sampling_stride

        # Initialize utilities
        self.memory_tracker = MemoryTracker(log_interval=memory_log_interval, verbose=memory_log_verbose)

        # --- Robust Device Detection ---
        self._model_device = self._get_module_device(backbone)
        if self._model_device is None:
            logger.warning("Could not detect device from backbone parameters. Attempting seg_head...")
            self._model_device = self._get_module_device(segmentation_head)
            if self._model_device is None:
                logger.warning("Could not detect device from seg_head parameters either. Falling back to CPU.")
                self._model_device = torch.device('cpu')
        logger.info(f"Determined model device: {self._model_device}")

        # --- Dimension Detection ---
        # Perform detection on the determined device
        self._input_dim_after_feature_extractor = self._detect_feature_dimensions(device=self._model_device)
        if self._input_dim_after_feature_extractor is None:
            raise RuntimeError("Failed to detect backbone feature dimensions.")
        logger.info(f"Detected feature dimension after feature extractor: {self._input_dim_after_feature_extractor}")

        # --- Determine Target Dimension and Need for Adapters ---
        if self.insertion_point == 'after_backbone':
            dim_before_modules = self._input_dim_after_feature_extractor
            # Default to a smaller dimension if target_feature_dim is not specified
            if target_feature_dim is None:
                 logger.warning(f"--target_feature_dim not specified. Defaulting to 512 for memory efficiency.")
                 self._target_feature_dim = 512
            else:
                 self._target_feature_dim = target_feature_dim
                 logger.info(f"Using specified --target_feature_dim: {self._target_feature_dim}")
        else: # 'after_seghead'
            self._input_dim_after_seghead = self._detect_feature_dimensions(after_seghead=True, device=self._model_device)
            if self._input_dim_after_seghead is None:
                raise RuntimeError("Failed to detect feature dimensions after segmentation head.")
            logger.info(f"Detected feature dimension after segmentation head: {self._input_dim_after_seghead}")
            dim_before_modules = self._input_dim_after_seghead
            # Use target_feature_dim if provided, else use the detected dim after seg head
            self._target_feature_dim = target_feature_dim if target_feature_dim is not None else dim_before_modules
            logger.info(f"Using target_feature_dim for final classifier input: {self._target_feature_dim}")

        logger.info(f"Effective dimension before downstream modules: {dim_before_modules}")
        logger.info(f"Target dimension after adapters/projections: {self._target_feature_dim}")

        # --- Adapters and Projections ---
        # Channel adapter (adapts backbone/seg_head output to target_feature_dim)
        self.needs_adapter = (dim_before_modules != self._target_feature_dim)
        self.channel_adapter = nn.Identity()
        if self.needs_adapter:
            logger.info(f"Adding Channel Adapter: {dim_before_modules} -> {self._target_feature_dim}")
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(dim_before_modules, self._target_feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self._target_feature_dim),
                nn.ReLU(inplace=False) # Use non-inplace ReLU
            )

        # Input projection for Hopfield layer (maps target_feature_dim to hopfield_feature_dim)
        self.hopfield_input_proj = nn.Conv2d(self._target_feature_dim, hopfield_feature_dim, kernel_size=1)

        # Hopfield Layer
        self.hopfield = ModernHopfieldLayer(
            input_dim=hopfield_feature_dim, output_dim=hopfield_feature_dim,
            num_heads=hopfield_num_heads, beta=hopfield_beta,
            memory_size=hopfield_memory_size, update_memory=True
        )
        logger.info(f"Initialized {type(self.hopfield).__name__} with Memory Size: {hopfield_memory_size}")

        # --- Output Projections and Final Layers ---
        self.final_seghead_proj = nn.Identity() # Projection before the original seg_head (if needed)
        self._original_seghead_in_channels = None # Store what the original head expected

        if insertion_point == 'after_backbone':
            # Projects Hopfield output (hopfield_feature_dim) -> target_feature_dim
            self.hopfield_output_proj = nn.Conv2d(hopfield_feature_dim, self._target_feature_dim, kernel_size=1)
            self.final_classifier = None # Final classification happens in seg_head

            # Check if the original seg_head needs a final projection before being called
            self._check_and_prepare_seghead_projection()

        else: # 'after_seghead'
            # Final classifier takes Hopfield output (hopfield_feature_dim) and maps to classes
            self.final_classifier = nn.Conv2d(hopfield_feature_dim, self.num_classes + 1, kernel_size=1)
            self.hopfield_output_proj = None # Output projection not needed before final classifier
            self._original_seghead_in_channels = None # Not relevant here

        # Energy head operates on Hopfield's internal features
        self.energy_head = nn.Sequential(
            nn.Conv2d(hopfield_feature_dim, hopfield_feature_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hopfield_feature_dim // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(hopfield_feature_dim // 2, 1, kernel_size=1)
        )

        # Initialize weights after defining all layers
        self._initialize_weights()
        # Ensure model is on the correct device AFTER initialization
        self.to(self._model_device)
        logger.info(f"HopfieldPEBALModel initialized and moved to device: {self._model_device}")


    def _get_module_device(self, module: nn.Module) -> Optional[torch.device]:
        """Safely gets the device of a module from its parameters."""
        try:
            # Handle ModuleDict parameters
            if isinstance(module, nn.ModuleDict):
                 for sub_module in module.values():
                      device = self._get_module_device(sub_module)
                      if device: return device
                 return None # No parameters found in ModuleDict submodules

            # Handle ModuleList parameters
            if isinstance(module, nn.ModuleList):
                 for sub_module in module:
                      device = self._get_module_device(sub_module)
                      if device: return device
                 return None # No parameters found in ModuleList submodules

            # Handle standard nn.Module parameters
            params = list(module.parameters())
            if params:
                 return params[0].device
            else: # Check buffers if no parameters
                 buffers = list(module.buffers())
                 if buffers:
                      return buffers[0].device
                 else:
                      # Cannot determine device if no params/buffers
                      # logger.warning(f"Module {type(module).__name__} has no parameters or buffers. Cannot determine device.")
                      return None
        except StopIteration: # Should not happen with the list check
             return None
        except Exception as e:
             logger.error(f"Error getting device for module {type(module).__name__}: {e}")
             return None


    def _detect_feature_dimensions(self, after_seghead: bool = False, device='cpu') -> Optional[int]:
        """Detect feature dimensions using a dummy forward pass on the specified device."""
        initial_backbone_device = self._get_module_device(self.backbone)
        initial_seghead_device = self._get_module_device(self.segmentation_head)

        # Use the provided device as the target device for the check
        target_device = device if device else torch.device('cpu') # Fallback to CPU if device is None

        # Ensure modules are on the target device for the check
        try:
            _backbone = self.backbone.to(target_device)
            _seg_head = self.segmentation_head.to(target_device)
            _backbone.eval(); _seg_head.eval() # Set to eval mode
        except Exception as e:
            logger.error(f"Error moving modules to {target_device} for dim detection: {e}", exc_info=True)
            return None # Cannot proceed if modules can't be moved

        detected_dim = None
        try:
            # Use a very small dummy input to minimize memory impact
            dummy_input = torch.zeros(1, 3, 32, 32, device=target_device) # Create dummy on target device
            with torch.no_grad():
                features = _backbone(dummy_input)
                if isinstance(features, (tuple, list)): features = features[-1]

                if after_seghead:
                    seg_head_input_dim = features.shape[1]
                    # Find first conv layer in seg_head to check its expected input
                    first_conv = None
                    for module in _seg_head.modules():
                         if isinstance(module, nn.Conv2d): first_conv = module; break

                    temp_proj = None
                    # Temporarily adapt input if seg_head expects different dim
                    if first_conv and first_conv.in_channels != seg_head_input_dim:
                         logger.debug(f"Temp adjusting seg head input for dim detection: {seg_head_input_dim} -> {first_conv.in_channels}")
                         temp_proj = nn.Conv2d(seg_head_input_dim, first_conv.in_channels, 1).to(target_device)
                         features = temp_proj(features) # Project features

                    seg_features = _seg_head(features)
                    if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                    detected_dim = seg_features.shape[1]
                    if temp_proj is not None: del temp_proj # Clean up temp proj
                else:
                    detected_dim = features.shape[1]

        except Exception as e:
            logger.error(f"Error during feature dimension detection forward pass ({'after seghead' if after_seghead else 'after backbone'}): {e}", exc_info=True)
            detected_dim = None
        finally:
            # Move modules back to original device *if they were moved* and *if we know the original device*
            # The self.backbone and self.segmentation_head were modified in-place by .to()
            # We must move them back to their original states.
            if initial_backbone_device is not None and target_device != initial_backbone_device:
                try:
                    self.backbone.to(initial_backbone_device)
                except Exception as move_e:
                    logger.warning(f"Could not move backbone back to {initial_backbone_device}: {move_e}")
            elif initial_backbone_device is None:
                 logger.warning("Original backbone device unknown, cannot move it back after dim detection.")

            if initial_seghead_device is not None and target_device != initial_seghead_device:
                try:
                    self.segmentation_head.to(initial_seghead_device)
                except Exception as move_e:
                    logger.warning(f"Could not move seg_head back to {initial_seghead_device}: {move_e}")
            elif initial_seghead_device is None:
                 logger.warning("Original seg_head device unknown, cannot move it back after dim detection.")

            # Clean up variables explicitly
            del _backbone, _seg_head # These might just be references, but deleting helps clarity
            if 'dummy_input' in locals(): del dummy_input
            if 'features' in locals(): del features
            if 'seg_features' in locals(): del seg_features
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return detected_dim


    def _check_and_prepare_seghead_projection(self):
        """Checks original seg head input and creates projection if needed."""
        if self.insertion_point != 'after_backbone':
            return # Only needed for this insertion point

        try:
            first_layer = None
            # Iterate through modules to find the first Conv or Linear layer
            # that is not part of the backbone or the Hopfield model itself
            modules_to_check = list(self.segmentation_head.modules())
            if not modules_to_check:
                 logger.warning("Segmentation head has no submodules to check.")
                 return

            for module in modules_to_check:
                # Skip self or immediate containers like Sequential, ModuleList
                if module is self.segmentation_head or isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                    continue
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    first_layer = module
                    break # Found the first relevant layer

            if first_layer is not None:
                input_dim_expected = None
                if isinstance(first_layer, nn.Conv2d):
                    input_dim_expected = first_layer.in_channels
                elif isinstance(first_layer, nn.Linear):
                    # Note: Using Linear head with Conv features requires flattening
                    input_dim_expected = first_layer.in_features

                if input_dim_expected is not None:
                    self._original_seghead_in_channels = input_dim_expected
                    logger.info(f"Original segmentation head's first layer ({type(first_layer).__name__}) expects input dimension: {self._original_seghead_in_channels}")

                    # Add projection if target dim != original head input dim
                    if self._original_seghead_in_channels != self._target_feature_dim:
                        logger.warning(f"Adding final projection before segmentation head: {self._target_feature_dim} -> {self._original_seghead_in_channels}")
                        # Ensure projection is Conv2d as input is spatial
                        self.final_seghead_proj = nn.Conv2d(self._target_feature_dim, self._original_seghead_in_channels, kernel_size=1)
                    else:
                        # Explicitly keep as Identity if dimensions match
                        self.final_seghead_proj = nn.Identity()
                else:
                    logger.warning("Could not determine input dimension for the first layer of the segmentation head. Assuming compatibility.")
                    self._original_seghead_in_channels = self._target_feature_dim
                    self.final_seghead_proj = nn.Identity()

            else:
                logger.warning("Could not find a Conv2d or Linear layer in the segmentation head to determine input dimension. Assuming it matches target_feature_dim.")
                self._original_seghead_in_channels = self._target_feature_dim
                self.final_seghead_proj = nn.Identity()

        except Exception as e:
            logger.error(f"Error while checking segmentation head input dim: {e}. Assuming compatibility.", exc_info=True)
            self._original_seghead_in_channels = self._target_feature_dim
            self.final_seghead_proj = nn.Identity()


    def _initialize_weights(self):
        """Initialize model weights."""
        logger.info("Initializing weights...")
        init_count = 0
        skipped_layers = 0
        initialized_modules = set() # Track initialized modules to avoid double init

        # Initialize main components first
        for component in [self.channel_adapter, self.hopfield_input_proj,
                          self.hopfield, self.final_seghead_proj,
                          self.final_classifier, self.energy_head]:
            if component is None or isinstance(component, nn.Identity):
                continue

            for m in component.modules():
                 if m in initialized_modules: continue # Skip if already initialized

                 if isinstance(m, (nn.Conv2d, nn.Linear)):
                     try:
                         if hasattr(m, 'weight') and m.weight is not None:
                             if isinstance(m, nn.Linear):
                                 nn.init.xavier_normal_(m.weight)
                             else: # Conv2d
                                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                             init_count += 1
                             initialized_modules.add(m)
                         if hasattr(m, 'bias') and m.bias is not None:
                             nn.init.constant_(m.bias, 0)
                     except Exception as e:
                         logger.warning(f"Could not initialize weights for {m.__class__.__name__}: {e}")
                         skipped_layers += 1
                 elif isinstance(m, nn.BatchNorm2d):
                     try:
                         if hasattr(m, 'weight') and m.weight is not None:
                             nn.init.constant_(m.weight, 1)
                         if hasattr(m, 'bias') and m.bias is not None:
                             nn.init.constant_(m.bias, 0)
                         init_count +=1
                         initialized_modules.add(m)
                     except Exception as e:
                         logger.warning(f"Could not initialize weights for {m.__class__.__name__}: {e}")
                         skipped_layers += 1

        # Specific initialization for Hopfield projections (handled within the loop above now)
        # std_dev = 0.02
        # if hasattr(self, 'hopfield') and isinstance(self.hopfield, ModernHopfieldLayer):
        #     logger.debug("Initializing Hopfield projection weights.")
        #     for name, proj in self.hopfield.projections.items():
        #         if proj in initialized_modules: continue # Skip if done
        #         try:
        #             if hasattr(proj, 'weight') and proj.weight is not None:
        #                 nn.init.normal_(proj.weight, mean=0.0, std=std_dev); init_count+=1
        #                 initialized_modules.add(proj)
        #             if hasattr(proj, 'bias') and proj.bias is not None:
        #                 nn.init.constant_(proj.bias, 0)
        #         except Exception as e:
        #             logger.warning(f"Could not initialize weights for Hopfield projection '{name}': {e}")
        #             skipped_layers += 1
        #     # Initialize memory buffer if it exists
        #     if hasattr(self.hopfield, 'memory') and self.hopfield.memory is not None:
        #         try:
        #             self.hopfield.memory.data.normal_(mean=0.0, std=0.01)
        #             logger.debug("Initialized Hopfield memory buffer.")
        #         except Exception as e:
        #             logger.warning(f"Could not initialize Hopfield memory: {e}")

        logger.info(f"Weight initialization complete ({init_count} modules initialized, {skipped_layers} skipped).")
        logger.info(f"Note: Backbone and original Segmentation Head weights are assumed to be pre-initialized/loaded.")


    def _apply_hopfield_processing(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function to apply Hopfield processing."""
        b, c_in, h_in, w_in = features.shape
        current_device = features.device
        output_dim_hop = self.hopfield.output_dim if hasattr(self.hopfield, 'output_dim') else self.hopfield.input_dim # Assume output=input if not specified

        # Initialize dummy outputs for error cases
        dummy_retrieved = torch.zeros(b, output_dim_hop, h_in, w_in, device=current_device)
        dummy_hop_energy = torch.zeros(b, h_in, w_in, device=current_device)
        dummy_feat_energy = torch.zeros(b, 1, h_in, w_in, device=current_device)

        try:
            hopfield_input = self.hopfield_input_proj(features)
            self.memory_tracker.log_memory_usage("After Hopfield Input Projection")
        except Exception as e:
             logger.error(f"Error in hopfield_input_proj: {e}", exc_info=True)
             return dummy_retrieved, dummy_hop_energy, dummy_feat_energy

        c_hop = hopfield_input.shape[1]
        h_proc, w_proc = h_in, w_in
        hopfield_input_sampled = hopfield_input
        needs_upsampling = False

        # Apply spatial sampling if efficient memory is on and input is large
        if self.use_efficient_memory and h_in * w_in > 4096 and self.sampling_stride > 1:
            stride = self.sampling_stride
            if h_in > stride and w_in > stride:
                hopfield_input_sampled = hopfield_input[:, :, ::stride, ::stride]
                h_proc, w_proc = hopfield_input_sampled.shape[2:]
                needs_upsampling = True
                logger.debug(f"Hopfield input sampled from {h_in}x{w_in} to {h_proc}x{w_proc} with stride {stride}")
                self.memory_tracker.log_memory_usage(f"After Strided Sampling (stride {stride})")
            else:
                logger.warning(f"Input size {h_in}x{w_in} large, cannot apply stride {stride}.")

        # Reshape for Hopfield layer
        # Need to handle empty dimensions after sampling if stride is too large
        if h_proc <= 0 or w_proc <= 0:
            logger.error(f"Hopfield input has zero spatial dimension after sampling: {h_proc}x{w_proc}. Skipping Hopfield.")
            return dummy_retrieved, dummy_hop_energy, dummy_feat_energy

        hopfield_input_flat = hopfield_input_sampled.reshape(b, c_hop, h_proc * w_proc).permute(0, 2, 1).contiguous() # [B, H*W, C_hop]
        num_tokens = hopfield_input_flat.shape[1]

        retrieved_flat: torch.Tensor
        hopfield_energy_flat: torch.Tensor

        # Process with Hopfield Layer (chunking if enabled and needed)
        if self.use_efficient_memory and num_tokens > self.chunk_size and self.chunk_size > 0:
            retrieved_chunks, energy_chunks = [], []
            logger.debug(f"Processing {num_tokens} tokens in chunks of size {self.chunk_size}")
            try:
                for i in range(0, num_tokens, self.chunk_size):
                    end = min(i + self.chunk_size, num_tokens)
                    chunk = hopfield_input_flat[:, i:end, :]
                    if chunk.shape[1] == 0: continue # Skip empty chunks
                    chunk_retrieved, chunk_energy = self.hopfield(chunk)
                    retrieved_chunks.append(chunk_retrieved)
                    energy_chunks.append(chunk_energy)
                    del chunk, chunk_retrieved, chunk_energy
                    # Clearing memory inside the loop can be very slow, do it less often if needed
                    # if i > 0 and i % (self.chunk_size * 10) == 0: self.memory_tracker.clear_memory(f"Hopfield Chunk {i}")

                if not retrieved_chunks: # Handle case where all chunks were empty or loop didn't run
                     logger.warning("No chunks processed in Hopfield layer. Returning zeros.")
                     retrieved_flat = torch.zeros(b, num_tokens, output_dim_hop, device=current_device)
                     hopfield_energy_flat = torch.zeros(b, num_tokens, device=current_device)
                else:
                     retrieved_flat = torch.cat(retrieved_chunks, dim=1)
                     hopfield_energy_flat = torch.cat(energy_chunks, dim=1)

                self.memory_tracker.log_memory_usage("After Chunked Hopfield Processing")
                del retrieved_chunks, energy_chunks
            except Exception as e:
                 logger.error(f"Error during chunked Hopfield processing: {e}. Returning zeros.", exc_info=True)
                 retrieved_flat = torch.zeros(b, num_tokens, output_dim_hop, device=current_device)
                 hopfield_energy_flat = torch.zeros(b, num_tokens, device=current_device)
        else:
            try:
                 retrieved_flat, hopfield_energy_flat = self.hopfield(hopfield_input_flat)
                 self.memory_tracker.log_memory_usage("After Full Hopfield Processing")
            except Exception as e:
                 logger.error(f"Error in Hopfield forward (full): {e}. Returning zeros.", exc_info=True)
                 retrieved_flat = torch.zeros(b, num_tokens, output_dim_hop, device=current_device)
                 hopfield_energy_flat = torch.zeros(b, num_tokens, device=current_device)

        # --- Post-Hopfield Processing ---
        # Ensure shapes are compatible before proceeding
        expected_retrieved_shape = (b, num_tokens, output_dim_hop)
        expected_energy_shape = (b, num_tokens)
        if retrieved_flat.shape != expected_retrieved_shape:
             logger.error(f"Shape mismatch for retrieved_flat: Got {retrieved_flat.shape}, expected {expected_retrieved_shape}. Adjusting.")
             # Attempt to recover if possible, otherwise use zeros
             if retrieved_flat.numel() == b * num_tokens * output_dim_hop:
                  retrieved_flat = retrieved_flat.reshape(expected_retrieved_shape)
             else:
                  retrieved_flat = torch.zeros(expected_retrieved_shape, device=current_device)
        if hopfield_energy_flat.shape != expected_energy_shape:
            logger.error(f"Shape mismatch for hopfield_energy_flat: Got {hopfield_energy_flat.shape}, expected {expected_energy_shape}. Adjusting.")
            if hopfield_energy_flat.numel() == b * num_tokens:
                 hopfield_energy_flat = hopfield_energy_flat.reshape(expected_energy_shape)
            else:
                 hopfield_energy_flat = torch.zeros(expected_energy_shape, device=current_device)


        # Reshape back to spatial dimensions
        retrieved_spatial = retrieved_flat.permute(0, 2, 1).reshape(b, output_dim_hop, h_proc, w_proc)

        # Calculate feature-based energy
        try:
            feature_energy_spatial = self.energy_head(retrieved_spatial)
        except Exception as e:
             logger.error(f"Error in energy_head: {e}. Returning zero energy.", exc_info=True)
             feature_energy_spatial = torch.zeros(b, 1, h_proc, w_proc, device=current_device)

        # Upsample if needed
        if needs_upsampling:
             try:
                 retrieved = F.interpolate(retrieved_spatial, size=(h_in, w_in), mode='bilinear', align_corners=False)
                 feature_energy = F.interpolate(feature_energy_spatial, size=(h_in, w_in), mode='bilinear', align_corners=False)
                 hopfield_energy_map = hopfield_energy_flat.reshape(b, 1, h_proc, w_proc)
                 hopfield_energy = F.interpolate(hopfield_energy_map, size=(h_in, w_in), mode='bilinear', align_corners=False).squeeze(1) # [B, H_in, W_in]
                 self.memory_tracker.log_memory_usage("After Upsampling Hopfield Outputs")
             except Exception as e:
                  logger.error(f"Error during upsampling: {e}. Returning non-upsampled.", exc_info=True)
                  # Ensure shapes are consistent even if not upsampled
                  retrieved = F.interpolate(retrieved_spatial, size=(h_in, w_in), mode='bilinear', align_corners=False) if h_in > 0 and w_in > 0 else retrieved_spatial
                  feature_energy = F.interpolate(feature_energy_spatial, size=(h_in, w_in), mode='bilinear', align_corners=False) if h_in > 0 and w_in > 0 else feature_energy_spatial
                  hopfield_energy = hopfield_energy_flat.reshape(b, h_proc, w_proc)
                  # Attempt to resize hopfield energy map if possible
                  if h_in > 0 and w_in > 0:
                       try:
                            hopfield_energy_map_resized = hopfield_energy_flat.reshape(b, 1, h_proc, w_proc)
                            hopfield_energy = F.interpolate(hopfield_energy_map_resized, size=(h_in, w_in), mode='bilinear', align_corners=False).squeeze(1)
                       except: # Fallback to original shape if resize fails
                            pass
        else:
            retrieved = retrieved_spatial
            feature_energy = feature_energy_spatial
            hopfield_energy = hopfield_energy_flat.reshape(b, h_in, w_in) # [B, H_in, W_in]

        # Cleanup intermediate tensors
        del hopfield_input_flat, retrieved_flat, hopfield_energy_flat
        del hopfield_input_sampled, hopfield_input, retrieved_spatial, feature_energy_spatial
        # Avoid clearing cache too often if efficiency is critical
        # self.memory_tracker.clear_memory("Hopfield Processing Cleanup")
        if self.use_efficient_memory: gc.collect()


        return retrieved, hopfield_energy, feature_energy

    def _check_and_handle_nan_inf(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Checks for NaN/Inf in a tensor and replaces them with zeros."""
        if not isinstance(tensor, torch.Tensor):
            logger.warning(f"Input '{name}' not a tensor ({type(tensor)}). Skipping check.")
            return tensor
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if has_nan or has_inf:
            nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
            inf_count = torch.isinf(tensor).sum().item() if has_inf else 0
            logger.warning(f"NaN/Inf detected in '{name}'. Replacing w/ zeros. (NaNs: {nan_count}, Infs: {inf_count}, Shape: {tensor.shape})")
            # Use torch.nan_to_num for efficient replacement
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the Hopfield PEBAL model."""
        self.memory_tracker.log_memory_usage("Forward Start")
        output_dict = {} # Initialize dictionary
        b, _, h_in_img, w_in_img = x.shape # Input image size
        current_device = x.device # Use input device

        # --- Stage 1: Feature Extraction ---
        try:
            features = self.backbone(x)
            if isinstance(features, (tuple, list)): features = features[-1] # Take last feature map if multiple returned
            features = self._check_and_handle_nan_inf(features, "Backbone Features")
            self.memory_tracker.log_memory_usage("After Backbone")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM Error in backbone: {e}. Trying with smaller input OR returning error.")
                self.memory_tracker.clear_memory("OOM Fallback")
                # Option 1: Try smaller input (disabled for simplicity, can add back if needed)
                # Option 2: Raise the error or return dummy output
                # Create dummy outputs matching expected shapes but on the CPU to save GPU memory
                dummy_logits = torch.zeros(b, self.num_classes + 1, h_in_img, w_in_img, device='cpu')
                dummy_energy = torch.zeros(b, 1, h_in_img, w_in_img, device='cpu')
                logger.critical(f"OOM in Backbone. Cannot proceed. Returning zero tensors on CPU.")
                return {
                    'seg_logits': dummy_logits,
                    'hopfield_energy': dummy_energy.clone(),
                    'feature_energy': dummy_energy.clone(),
                    'pebal_energy': dummy_energy.clone(),
                    'combined_energy': dummy_energy.clone()
                }
            else:
                logger.error(f"Runtime error in backbone: {e}", exc_info=True)
                raise e # Re-raise other runtime errors
        except Exception as e:
             logger.error(f"Error during backbone feature extraction: {e}", exc_info=True)
             raise e # Re-raise

        b, c_feat, h_feat, w_feat = features.shape

        # --- Stage 2&3: Adapter, Hopfield/SegHead ---
        logits = None
        # Initialize energy maps with default shapes matching features, will be overwritten or interpolated
        # Ensure they are on the correct device
        hopfield_energy_map = torch.zeros(b, h_feat, w_feat, device=current_device)
        feature_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)

        if self.insertion_point == 'after_backbone':
            try:
                features_adapted = self.channel_adapter(features) # Adapts to _target_feature_dim
                features_adapted = self._check_and_handle_nan_inf(features_adapted, "Features Adapted (after_backbone)")
                self.memory_tracker.log_memory_usage("After Channel Adapter (pre-Hopfield)")

                retrieved, hopfield_energy_map, feature_energy_map = self._apply_hopfield_processing(features_adapted)
                retrieved = self._check_and_handle_nan_inf(retrieved, "Hopfield Retrieved")
                hopfield_energy_map = self._check_and_handle_nan_inf(hopfield_energy_map, "Hopfield Energy Map")
                feature_energy_map = self._check_and_handle_nan_inf(feature_energy_map, "Feature Energy Map")

                hopfield_output = self.hopfield_output_proj(retrieved) # Projects C_hop -> _target_feature_dim
                hopfield_output = self._check_and_handle_nan_inf(hopfield_output, "Hopfield Output Projection")
                self.memory_tracker.log_memory_usage("After Hopfield Output Projection")

                # Residual connection
                updated_features = features_adapted + hopfield_output # Has _target_feature_dim channels
                updated_features = self._check_and_handle_nan_inf(updated_features, "Updated Features (Residual)")

                # Explicit cleanup of intermediate tensors
                del features, retrieved, hopfield_output, features_adapted
                if self.use_efficient_memory: self.memory_tracker.clear_memory("After Backbone Path Cleanup")

                # Project features before passing to original segmentation head
                seg_head_input = self.final_seghead_proj(updated_features)
                seg_head_input = self._check_and_handle_nan_inf(seg_head_input, "SegHead Input (Final Proj)")
                self.memory_tracker.log_memory_usage("After Final SegHead Projection")
                del updated_features # Cleanup

                logits = self.segmentation_head(seg_head_input) # Original head receives correct dim
                if isinstance(logits, (tuple, list)): logits = logits[-1]
                logits = self._check_and_handle_nan_inf(logits, "Segmentation Head Output")
                self.memory_tracker.log_memory_usage("After Segmentation Head")

            except Exception as e:
                 logger.error(f"Error in 'after_backbone' path: {e}", exc_info=True)
                 # Create dummy logits matching expected output size (h_feat, w_feat before interpolation)
                 logits = torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)

        else: # insertion_point == 'after_seghead'
             try:
                 seg_features = self.segmentation_head(features) # Original head expects backbone output
                 if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                 seg_features = self._check_and_handle_nan_inf(seg_features, "Segmentation Head Output (before Hopfield)")
                 self.memory_tracker.log_memory_usage("After Segmentation Head")

                 del features # Cleanup backbone features
                 if self.use_efficient_memory: self.memory_tracker.clear_memory("After SegHead Path Cleanup 1")

                 seg_features_adapted = self.channel_adapter(seg_features) # Adapts to _target_feature_dim
                 seg_features_adapted = self._check_and_handle_nan_inf(seg_features_adapted, "Seg Features Adapted (after_seghead)")
                 self.memory_tracker.log_memory_usage("After Channel Adapter (pre-Hopfield)")
                 del seg_features # Cleanup original seg features

                 retrieved, hopfield_energy_map, feature_energy_map = self._apply_hopfield_processing(seg_features_adapted)
                 retrieved = self._check_and_handle_nan_inf(retrieved, "Hopfield Retrieved")
                 hopfield_energy_map = self._check_and_handle_nan_inf(hopfield_energy_map, "Hopfield Energy Map")
                 feature_energy_map = self._check_and_handle_nan_inf(feature_energy_map, "Feature Energy Map")

                 del seg_features_adapted # Cleanup
                 if self.use_efficient_memory: self.memory_tracker.clear_memory("After SegHead Path Cleanup 2")

                 logits = self.final_classifier(retrieved) # Classifies Hopfield output
                 logits = self._check_and_handle_nan_inf(logits, "Final Classifier Output")
                 self.memory_tracker.log_memory_usage("After Final Classifier")
                 del retrieved # Cleanup

             except Exception as e:
                 logger.error(f"Error in 'after_seghead' path: {e}", exc_info=True)
                 # Create dummy outputs if this path fails
                 h_out, w_out = h_feat, w_feat # Assume output size matches feature size
                 logits = torch.zeros(b, self.num_classes+1, h_out, w_out, device=current_device)
                 # Use existing dummy energy maps if available, else create new ones
                 hopfield_energy_map = hopfield_energy_map if 'hopfield_energy_map' in locals() else torch.zeros(b, h_out, w_out, device=current_device)
                 feature_energy_map = feature_energy_map if 'feature_energy_map' in locals() else torch.zeros(b, 1, h_out, w_out, device=current_device)


        # Ensure logits exist
        if logits is None:
             logger.critical("Logits variable is None after main processing block. Creating zeros.")
             h_out, w_out = h_feat, w_feat # Default to feature map size
             logits = torch.zeros(b, self.num_classes+1, h_out, w_out, device=current_device)

        # --- Final Interpolation to Match Input Size ---
        h_logit_pre, w_logit_pre = logits.shape[-2:]
        # Ensure energy maps have the correct shape before interpolation/use
        # Hopfield energy map needs channel dim added, Feature energy map should already have it
        if hopfield_energy_map.dim() == 3: # Shape [B, H, W]
             hopfield_energy_map_spatial = hopfield_energy_map.unsqueeze(1) # Add channel dim -> [B, 1, H, W]
        elif hopfield_energy_map.dim() == 4 and hopfield_energy_map.shape[1] == 1: # Already [B, 1, H, W]
             hopfield_energy_map_spatial = hopfield_energy_map
        else: # Unexpected shape
             logger.warning(f"Unexpected Hopfield energy map shape: {hopfield_energy_map.shape}. Attempting to reshape or zero out.")
             hopfield_energy_map_spatial = torch.zeros(b, 1, h_logit_pre, w_logit_pre, device=current_device)

        if feature_energy_map.dim() != 4 or feature_energy_map.shape[1] != 1:
            logger.warning(f"Unexpected Feature energy map shape: {feature_energy_map.shape}. Attempting to reshape or zero out.")
            feature_energy_map_spatial = torch.zeros(b, 1, h_logit_pre, w_logit_pre, device=current_device)
        else:
            feature_energy_map_spatial = feature_energy_map


        # Perform interpolation if needed
        if h_logit_pre != h_in_img or w_logit_pre != w_in_img:
             logger.debug(f"Interpolating outputs from {(h_logit_pre, w_logit_pre)} to {(h_in_img, w_in_img)}")
             try:
                 logits_final = F.interpolate(logits, size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
                 hopfield_energy_final = F.interpolate(hopfield_energy_map_spatial, size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
                 feature_energy_final = F.interpolate(feature_energy_map_spatial, size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
             except Exception as e:
                 logger.error(f"Error interpolating final outputs: {e}. Using un-interpolated.", exc_info=True)
                 # Fallback: Use un-interpolated outputs
                 logits_final = logits
                 hopfield_energy_final = hopfield_energy_map_spatial
                 feature_energy_final = feature_energy_map_spatial
        else:
             logits_final = logits
             hopfield_energy_final = hopfield_energy_map_spatial
             feature_energy_final = feature_energy_map_spatial


        # --- Stage 4: Energy Calculation & NaN/Inf Handling ---
        # Use final (possibly interpolated) tensors
        logits_final = self._check_and_handle_nan_inf(logits_final, "Logits Final")
        hopfield_energy_final = self._check_and_handle_nan_inf(hopfield_energy_final, "Hopfield Energy Final")
        feature_energy_final = self._check_and_handle_nan_inf(feature_energy_final, "Feature Energy Final")

        output_dict['seg_logits'] = logits_final # Store final logits

        try:
            # Calculate PEBAL energy on final logits
            # Ensure stability by subtracting max logit before exp
            max_logits_in_class = torch.max(logits_final[:, :self.num_classes, :, :], dim=1, keepdim=True)[0]
            pebal_energy = max_logits_in_class - torch.logsumexp(logits_final[:, :self.num_classes, :, :] - max_logits_in_class, dim=1, keepdim=True)
            # Note: PEBAL energy is often defined as NEGATIVE logsumexp. Check convention.
            # Original code used -logsumexp, sticking to that.
            pebal_energy = -pebal_energy
            pebal_energy = self._check_and_handle_nan_inf(pebal_energy, "pebal_energy")
        except Exception as e:
            logger.error(f"Error calculating PEBAL energy: {e}. Using zero energy.", exc_info=True)
            pebal_energy = torch.zeros_like(feature_energy_final) # Match shape of other energies

        # Combine energy terms (ensure dimensions match before combining)
        # All should be [B, 1, H_out, W_out] after interpolation/unsqueeze
        try:
            combined_energy = pebal_energy + 0.5 * feature_energy_final + 0.5 * hopfield_energy_final
            combined_energy = torch.clamp(combined_energy, min=-100.0, max=100.0) # Clamp combined
            combined_energy = self._check_and_handle_nan_inf(combined_energy, "combined_energy")
        except Exception as e:
             logger.error(f"Error combining energy terms: {e}. Using zero combined energy.", exc_info=True)
             combined_energy = torch.zeros_like(pebal_energy)


        # Add final energies to dict
        output_dict['hopfield_energy'] = hopfield_energy_final
        output_dict['feature_energy'] = feature_energy_final
        output_dict['pebal_energy'] = pebal_energy
        output_dict['combined_energy'] = combined_energy

        self.memory_tracker.log_memory_usage("Forward End")

        # Final check for keys and shapes before returning
        expected_keys = ['seg_logits', 'hopfield_energy', 'feature_energy', 'pebal_energy', 'combined_energy']
        ref_shape = output_dict['seg_logits'].shape # Use final logit shape as reference [B, C, H, W]
        final_spatial_shape = ref_shape[-2:]

        for key in expected_keys:
             if key not in output_dict:
                  logger.critical(f"Output key '{key}' missing from forward pass result!")
                  # Create appropriate shape based on key
                  if key == 'seg_logits':
                      target_shape = ref_shape
                  else: # Energy maps
                      target_shape = (ref_shape[0], 1, final_spatial_shape[0], final_spatial_shape[1])
                  output_dict[key] = torch.zeros(target_shape, device=current_device)

             elif key != 'seg_logits': # Check energy map shapes
                 current_shape = output_dict[key].shape
                 expected_spatial = final_spatial_shape
                 if current_shape[-2:] != expected_spatial or current_shape[1] != 1:
                     logger.warning(f"Shape mismatch for '{key}': Got {current_shape} vs Expected Spatial {expected_spatial} with channel 1. Attempting resize.")
                     try:
                         # Ensure channel dim is 1 before resizing
                         tensor_to_resize = output_dict[key]
                         if tensor_to_resize.shape[1] != 1:
                              # Attempt to average or take first channel if multiple exist
                              logger.warning(f"Energy map '{key}' has {tensor_to_resize.shape[1]} channels, expected 1. Averaging.")
                              tensor_to_resize = torch.mean(tensor_to_resize, dim=1, keepdim=True)

                         output_dict[key] = F.interpolate(tensor_to_resize, size=expected_spatial, mode='bilinear', align_corners=False)
                         output_dict[key] = self._check_and_handle_nan_inf(output_dict[key], f"{key} (post-resize)")
                     except Exception as resize_e:
                         logger.error(f"Failed to resize '{key}' to match logits: {resize_e}. Setting to zeros.")
                         output_dict[key] = torch.zeros((ref_shape[0], 1, expected_spatial[0], expected_spatial[1]), device=current_device)

        return output_dict


    def _prepare_features_for_memory_update(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """Prepares features (adapter, projection, sampling, flattening) for Hopfield memory update."""
        self.memory_tracker.log_memory_usage("Start Memory Feature Prep")
        current_device = features.device
        self.eval() # Ensure modules are in eval mode for this prep step

        with torch.no_grad():
            features_to_project = None # Initialize
            # Determine features based on insertion point
            if self.insertion_point == 'after_backbone':
                try:
                    features_adapted = self.channel_adapter(features) # Adapts to _target_feature_dim
                    features_to_project = self._check_and_handle_nan_inf(features_adapted, "MemPrep Features Adapted (after_backbone)")
                except Exception as e:
                    logger.error(f"Error in channel_adapter (mem prep): {e}", exc_info=True); return None
            else: # 'after_seghead'
                try:
                    # Need to run seg_head first
                    seg_features = self.segmentation_head(features)
                    if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                    seg_features = self._check_and_handle_nan_inf(seg_features, "MemPrep Seg Head Output")

                    features_adapted = self.channel_adapter(seg_features) # Adapts to _target_feature_dim
                    features_to_project = self._check_and_handle_nan_inf(features_adapted, "MemPrep Features Adapted (after_seghead)")
                    del seg_features # Clean up intermediate
                except Exception as e:
                    logger.error(f"Error in seg_head/adapter (mem prep): {e}", exc_info=True); return None

            if features_to_project is None:
                logger.error("Failed to determine features_to_project in memory prep.")
                return None

            self.memory_tracker.log_memory_usage("Memory Prep: After SegHead/Adapter")

            # Project features to Hopfield dimension
            try:
                hopfield_input = self.hopfield_input_proj(features_to_project)
                hopfield_input = self._check_and_handle_nan_inf(hopfield_input, "MemPrep Hopfield Input Proj")
            except Exception as e:
                logger.error(f"Error in hopfield_input_proj during memory prep: {e}", exc_info=True)
                if features_to_project is not None: del features_to_project
                return None
            del features_to_project # Clean up

            # Apply spatial sampling if needed
            b, c_hop, h_in, w_in = hopfield_input.shape
            hopfield_input_sampled = hopfield_input
            if self.use_efficient_memory and h_in * w_in > 4096 and self.sampling_stride > 1:
                stride = self.sampling_stride
                if h_in > stride and w_in > stride:
                    hopfield_input_sampled = hopfield_input[:, :, ::stride, ::stride]
                    h_proc, w_proc = hopfield_input_sampled.shape[-2:]
                    logger.debug(f"MemPrep input sampled from {h_in}x{w_in} to {h_proc}x{w_proc} with stride {stride}")
                    self.memory_tracker.log_memory_usage(f"Memory Prep: After Sampling (stride {stride})")
                else:
                    logger.warning(f"Warning (Memory Prep): Cannot apply stride {stride} to {h_in}x{w_in}.")
            else:
                h_proc, w_proc = h_in, w_in # No sampling applied

            # Final check: Ensure C_hop matches Hopfield input dimension
            expected_hopfield_dim = getattr(self.hopfield, 'input_dim', None)
            if expected_hopfield_dim is None:
                 logger.error("ERROR (Memory Prep): Cannot determine Hopfield input dimension.")
                 return None
            if c_hop != expected_hopfield_dim:
                 logger.error(f"ERROR (Memory Prep): Projected feature dim {c_hop} != Hopfield input dim {expected_hopfield_dim}")
                 return None

            # Handle zero spatial dimensions after sampling
            if h_proc <= 0 or w_proc <= 0:
                logger.warning(f"Memory Prep: Zero spatial dimension after sampling ({h_proc}x{w_proc}). Skipping memory update.")
                return None

            # Reshape for memory update (flatten spatial dims)
            # Permute -> Reshape -> Detach
            hopfield_input_flat = hopfield_input_sampled.permute(0, 2, 3, 1).reshape(-1, c_hop).detach()
            self.memory_tracker.log_memory_usage("End Memory Feature Prep")
            return hopfield_input_flat


    def update_memory(self, input_data: torch.Tensor, max_samples: int = 5000):
        """
        Updates the Hopfield memory bank using features derived from input_data.

        Args:
            input_data: Batch of input images [B, 3, H, W] or pre-extracted backbone features
                        [B, C_backbone, H_feat, W_feat].
            max_samples: Maximum number of feature vectors to sample for the update.
        """
        if not hasattr(self.hopfield, 'update_memory') or not self.hopfield.update_memory:
            # logger.debug("Hopfield layer memory update disabled.") # Reduce noise
            return
        if not hasattr(self.hopfield, 'update_memory_bank') or not callable(self.hopfield.update_memory_bank):
             logger.warning("Hopfield layer missing callable 'update_memory_bank' method. Skipping update.")
             return


        self.eval() # Ensure model is in eval mode for feature extraction and prep
        original_model_device = self._model_device # Device the model normally lives on
        input_device = input_data.device # Device the input data is on

        with torch.no_grad():
            features = None
            input_data_on_model_device = input_data.to(original_model_device) # Move input to model device

            # --- Feature Extraction ---
            # Check if input_data looks like an image batch
            if len(input_data_on_model_device.shape) == 4 and input_data_on_model_device.shape[1] == 3:
                logger.debug("Extracting backbone features for memory update...")
                try:
                    features_extracted = self.backbone(input_data_on_model_device)
                    if isinstance(features_extracted, (tuple, list)): features_extracted = features_extracted[-1]
                    features = self._check_and_handle_nan_inf(features_extracted, "MemUpdate Backbone Features")
                except Exception as e:
                    logger.error(f"Error extracting backbone features for mem update: {e}", exc_info=True)
                    self.train() # Return model to train mode before exiting
                    return # Exit if extraction fails
            # Check if input_data looks like pre-extracted features matching backbone output dim
            elif len(input_data_on_model_device.shape) == 4 and hasattr(self, '_input_dim_after_feature_extractor') and input_data_on_model_device.shape[1] == self._input_dim_after_feature_extractor:
                logger.debug("Using pre-extracted features for memory update.")
                features = self._check_and_handle_nan_inf(input_data_on_model_device, "MemUpdate Pre-extracted Features")
            else:
                 logger.error(f"Error: Unexpected input shape/channels for memory update: {input_data.shape}. "
                              f"Expected [B, 3, H, W] or [B, {getattr(self, '_input_dim_after_feature_extractor', 'N/A')}, H, W].")
                 self.train(); return

            # Cleanup input tensor copy if it was created
            if input_data_on_model_device is not input_data:
                 del input_data_on_model_device

            if features is None:
                logger.error("Features for memory update are None."); self.train(); return

            # --- Feature Preparation ---
            flat_features = self._prepare_features_for_memory_update(features)
            del features # Clean up extracted features

            if flat_features is None or flat_features.shape[0] == 0:
                 logger.warning("Memory update skipped: No valid features prepared."); self.train(); return

            # --- Sampling ---
            num_available = flat_features.shape[0]
            # Determine sample size respecting max_samples and available memory size
            hopfield_mem_size = getattr(self.hopfield, 'memory_size', 0)
            sample_size = min(num_available, max_samples, hopfield_mem_size) if hopfield_mem_size > 0 else min(num_available, max_samples)

            logger.debug(f"Available features: {num_available}, Sampling: {sample_size} for memory update (Memory Size: {hopfield_mem_size}).")

            sampled_features = None
            if sample_size == 0:
                logger.warning("Memory update skipped: Sample size is 0."); self.train(); return
            elif sample_size >= num_available:
                sampled_features = flat_features # Use all features
            else:
                 # Efficiently sample using randperm
                 try:
                      indices = torch.randperm(num_available, device=flat_features.device)[:sample_size]
                      sampled_features = flat_features[indices]
                 except Exception as e:
                      logger.error(f"Error during feature sampling: {e}. Skipping update.", exc_info=True)
                      del flat_features; self.train(); return

            # --- Memory Update ---
            try:
                 # update_memory_bank should handle device placement internally if needed
                 self.hopfield.update_memory_bank(sampled_features)
                 logger.info(f"Hopfield memory updated via layer method with {sampled_features.shape[0]} vectors.")

            except Exception as e:
                logger.error(f"Error occurred during Hopfield memory update via layer method: {e}", exc_info=True)

            self.memory_tracker.log_memory_usage("End Memory Update")
            del flat_features, sampled_features # More cleanup
            # Avoid aggressive cache clearing here unless necessary
            # self.memory_tracker.clear_memory("Memory Update Cleanup")
            if self.use_efficient_memory: gc.collect()

        self.train() # Ensure model returns to train mode


# --- Example Usage ---
if __name__ == '__main__':
    # Configure logger for example script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger(__name__)

    # --- Dummy Modules ---
    class DummyBackbone(nn.Module):
        def __init__(self, out_channels=2048):
            super().__init__()
            self.out_channels = out_channels
            # Simplified layers
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # Final layer to reach target output dimension
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
            # Add a parameter to ensure device detection works
            self.dummy_param = nn.Parameter(torch.randn(1))

        def forward(self, x):
            # print(f"DummyBackbone input shape: {x.shape}, device: {x.device}")
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.final_conv(x)
            # print(f"DummyBackbone output shape: {x.shape}")
            return x

    class DummySegHead(nn.Module):
        def __init__(self, in_channels, num_classes, head_channels=256):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.head_channels = head_channels
            # Layer that consumes the input features
            self.feature_conv = nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(head_channels)
            self.relu = nn.ReLU(inplace=True)
            # Final classification layer
            self.final_classifier = nn.Conv2d(head_channels, num_classes + 1, kernel_size=1)
            self._output_features = False # Control whether to output features or logits

        def forward(self, x):
            # print(f"DummySegHead input shape: {x.shape}, expects: {self.in_channels}")
            features = self.feature_conv(x)
            features = self.bn(features)
            features = self.relu(features)
            # print(f"DummySegHead intermediate features shape: {features.shape}")

            if self._output_features:
                # Return intermediate features (e.g., for 'after_seghead' insertion)
                return features
            else:
                # Return final classification logits
                logits = self.final_classifier(features)
                # print(f"DummySegHead output logits shape: {logits.shape}")
                return logits

    # --- Configuration ---
    NUM_CLASSES = 19
    BACKBONE_OUT_DIM = 1024 # Adjusted example dim
    SEG_HEAD_FEATURE_DIM = 256
    HOPFIELD_DIM = 128
    HOPFIELD_MEM_SIZE = 500 # Smaller memory for faster example
    BATCH_SIZE = 2
    IMG_SIZE = (64, 128) # Smaller image size for faster example

    # *** SET EXPLICIT TARGET DIM FOR TESTING ***
    TARGET_DIM_AFTER_BACKBONE = 512 # Target dim if Hopfield is after backbone
    TARGET_DIM_AFTER_SEGHEAD = SEG_HEAD_FEATURE_DIM # Target dim if Hopfield is after seg_head features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_logger.info(f"Using device: {device}")

    # --- Instantiate Backbone ---
    backbone = DummyBackbone(out_channels=BACKBONE_OUT_DIM).to(device)

    # --- Test Case 1: Hopfield after Backbone ---
    main_logger.info("\n--- Testing Model: Hopfield after Backbone ---")
    # The seg head is defined to expect the final TARGET_DIM_AFTER_BACKBONE features
    # as its input, AFTER the Hopfield processing and projection.
    # The HopfieldPEBALModel will project the backbone output (BACKBONE_OUT_DIM)
    # down to TARGET_DIM_AFTER_BACKBONE via the channel_adapter, process it with Hopfield,
    # project Hopfield output back to TARGET_DIM_AFTER_BACKBONE, add residual,
    # and then potentially project again via final_seghead_proj if the DummySegHead's
    # feature_conv layer expects something different than TARGET_DIM_AFTER_BACKBONE.

    # This head's first layer (feature_conv) expects TARGET_DIM_AFTER_BACKBONE
    seg_head_1 = DummySegHead(in_channels=TARGET_DIM_AFTER_BACKBONE, num_classes=NUM_CLASSES, head_channels=SEG_HEAD_FEATURE_DIM).to(device)
    model_after_backbone = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=seg_head_1,
        num_classes=NUM_CLASSES,
        hopfield_feature_dim=HOPFIELD_DIM,
        target_feature_dim=TARGET_DIM_AFTER_BACKBONE, # Define target dim after backbone
        insertion_point='after_backbone',
        hopfield_memory_size=HOPFIELD_MEM_SIZE,
        use_efficient_memory=False, # Disable efficiency for easier debug
        memory_log_verbose=True
    ).to(device)

    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1], device=device)
    main_logger.info("\nTesting forward pass (after_backbone)...")
    model_after_backbone.train() # Set to train mode
    output_dict = model_after_backbone(dummy_input)
    main_logger.info("Output keys: %s", output_dict.keys())
    assert 'seg_logits' in output_dict, "FAIL: 'seg_logits' key missing!"
    for key, tensor in output_dict.items():
        main_logger.info(f"  {key}: shape={tensor.shape}, device={tensor.device}, has_nan={torch.isnan(tensor).any()}")

    main_logger.info("\nTesting memory update (after_backbone)...")
    mem_update_input = torch.randn(BATCH_SIZE * 2, 3, IMG_SIZE[0] // 2, IMG_SIZE[1] // 2, device=device) # Smaller input for mem update
    model_after_backbone.update_memory(mem_update_input, max_samples=HOPFIELD_MEM_SIZE // 2)

    del model_after_backbone, output_dict, seg_head_1
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Test Case 2: Hopfield after SegHead ---
    main_logger.info("\n--- Testing Model: Hopfield after SegHead ---")
    # The seg head here takes the BACKBONE_OUT_DIM as input and outputs features
    # of dimension SEG_HEAD_FEATURE_DIM. Hopfield is inserted after this.
    seg_head_features = DummySegHead(in_channels=BACKBONE_OUT_DIM, num_classes=NUM_CLASSES, head_channels=SEG_HEAD_FEATURE_DIM).to(device)
    seg_head_features._output_features = True # Configure the head to output intermediate features

    model_after_seghead = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=seg_head_features, # Use the feature-outputting head
        num_classes=NUM_CLASSES,
        hopfield_feature_dim=HOPFIELD_DIM,
        target_feature_dim=TARGET_DIM_AFTER_SEGHEAD, # Target dim is based on seg_head's output
        insertion_point='after_seghead',
        hopfield_memory_size=HOPFIELD_MEM_SIZE,
        use_efficient_memory=False,
        memory_log_verbose=True
    ).to(device)

    main_logger.info("\nTesting forward pass (after_seghead)...")
    model_after_seghead.train() # Set to train mode
    output_dict_seg = model_after_seghead(dummy_input)
    main_logger.info("Output keys: %s", output_dict_seg.keys())
    assert 'seg_logits' in output_dict_seg, "FAIL: 'seg_logits' key missing!"
    for key, tensor in output_dict_seg.items():
        main_logger.info(f"  {key}: shape={tensor.shape}, device={tensor.device}, has_nan={torch.isnan(tensor).any()}")

    main_logger.info("\nTesting memory update (after_seghead)...")
    # For memory update in this case, the model will run input -> backbone -> seg_head -> adapter -> proj -> flatten
    model_after_seghead.update_memory(mem_update_input, max_samples=HOPFIELD_MEM_SIZE // 2)

    del model_after_seghead, output_dict_seg, seg_head_features
    del backbone, dummy_input, mem_update_input
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    main_logger.info("\nExample Usage Complete.")
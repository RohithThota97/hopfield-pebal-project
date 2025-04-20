# hopfield_pebal_model.py (Modified)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import psutil
import time
from typing import Dict, Tuple, Optional, Union, List
import logging
import numpy as np
# Attempt to import faiss, but make it optional
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    print("WARNING: faiss library not found. FAISS acceleration disabled.") # Use print for early warning

# Configure logging
# Ensure logging is configured only once, preferably in the main script
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger instance

# --- Memory Tracker ---
class MemoryTracker:
    def __init__(self, log_interval: int = 10, verbose: bool = True):
        self.log_interval = log_interval
        self.verbose = verbose
        self.last_log_time = time.time()
        self.peak_gpu_mem = 0
        self.peak_cpu_mem = 0
        try:
            self._process = psutil.Process()
        except (ImportError, psutil.NoSuchProcess):
            logger.warning("psutil unavailable/failed. CPU mem tracking disabled.")
            self._process = None

    def _bytes_to_mb(self, b: int) -> float:
        return b / (1024 * 1024)

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        current_gpu_mem = 0
        if torch.cuda.is_available():
            try:
                current_gpu_mem = self._bytes_to_mb(torch.cuda.memory_allocated())
                self.peak_gpu_mem = max(self.peak_gpu_mem, current_gpu_mem)
            except Exception as e:
                 logger.warning(f"Could not get GPU allocated mem: {e}")
        return current_gpu_mem, self.peak_gpu_mem

    def get_cpu_memory_usage(self) -> Tuple[float, float]:
        current_cpu_mem = 0
        if self._process:
            try:
                current_cpu_mem = self._bytes_to_mb(self._process.memory_info().rss)
                self.peak_cpu_mem = max(self.peak_cpu_mem, current_cpu_mem)
            except psutil.NoSuchProcess:
                logger.warning("Process terminated. Disabling CPU memory tracking.")
                self._process = None
            except Exception as e:
                 logger.warning(f"Could not get CPU mem: {e}") if not isinstance(e, psutil.NoSuchProcess) else None
        return current_cpu_mem, self.peak_cpu_mem

    def log_memory_usage(self, operation_name: str = ""):
        if not self.verbose: return
        current_time = time.time()
        # Log unconditionally if operation_name is given, otherwise respect interval
        if operation_name or (current_time - self.last_log_time >= self.log_interval):
            gpu_mem, peak_gpu = self.get_gpu_memory_usage()
            cpu_mem, peak_cpu = self.get_cpu_memory_usage()
            op_str = f" [{operation_name}]" if operation_name else ""
            gpu_res_mem = 0
            if torch.cuda.is_available():
                try:
                    gpu_res_mem = self._bytes_to_mb(torch.cuda.memory_reserved())
                except: pass # Ignore errors here
            logger.info(f"[MemoryTracker]{op_str}: GPU Alloc {gpu_mem:.1f}MB (Peak: {peak_gpu:.1f}MB) | GPU Reserved {gpu_res_mem:.1f}MB | "
                        f"CPU {cpu_mem:.1f}MB (Peak: {peak_cpu:.1f}MB)")
            self.last_log_time = current_time

    def clear_memory(self, operation_name: str = ""):
        op_str = f" Cleared after {operation_name}" if operation_name else " Clearing memory"
        self.log_memory_usage(f"Pre-Clear{op_str}")
        n = gc.collect()
        if n > 0: logger.debug(f"[MemoryTracker] GC collected {n} objects.")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Could not empty CUDA cache: {e}")
        self.log_memory_usage(f"Post-Clear{op_str}")

# --- Efficient Memory Manager ---
class EfficientMemoryManager(nn.Module):
    """
    Implements sampling techniques and FAISS index for efficient memory bank management.
    """
    def __init__(self, feature_dim=256, memory_size=1024, pq_bytes=8,
                 sampling_ratio=0.25, num_classes=19, use_faiss=True,
                 device='cpu'):
        super(EfficientMemoryManager, self).__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.sampling_ratio = sampling_ratio # Currently unused in update logic
        self.num_classes = num_classes
        self.pq_bytes = pq_bytes
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self._device = torch.device(device) # Ensure device is a torch.device
        self.memory_tracker = MemoryTracker(log_interval=15, verbose=True)

        # Register memory_bank and memory_labels as buffers
        # This ensures they are moved correctly when the module's .to(device) is called.
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim, device=self._device))
        self.register_buffer('memory_labels', torch.full((memory_size,), -1, dtype=torch.long, device=self._device))

        # Pointer and counts can also be buffers if they need to be part of the state_dict
        # and moved with the module. Using tensor directly is also common if they aren't
        # strictly 'state' that needs saving/loading in the same way.
        # Let's register them too for consistency.
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long, device=self._device))
        self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.long, device=self._device))

        # Explicit .to(device) calls *after* registration might be redundant but ensure
        # the initial device is set correctly according to the 'device' argument.
        # The buffers are already initialized on self._device above.
        # self.memory_bank = self.memory_bank.to(self._device)
        # self.memory_labels = self.memory_labels.to(self._device)
        # self.memory_ptr = self.memory_ptr.to(self._device)
        # self.class_counts = self.class_counts.to(self._device)

        self.memory_initialized = False
        self.faiss_index = None
        self.faiss_res = None
        self._init_faiss_resources()

    def _init_faiss_resources(self):
        """Initialize FAISS GPU/CPU resources."""
        if not self.use_faiss: return
        try:
            if faiss.get_num_gpus() > 0 and 'cuda' in str(self._device):
                gpu_id = self._device.index if self._device.index is not None else 0
                self.faiss_res = faiss.StandardGpuResources()
                # self.faiss_res = faiss.GpuResources() # Simpler alternative?
                logger.info(f"FAISS: Initialized StandardGpuResources on device {gpu_id}.")
            else:
                logger.info("FAISS: Using CPU resources (No GPU detected by FAISS or CPU device specified).")
                self.use_faiss = False # Disable FAISS if GPU is not available or not requested
        except Exception as e:
            logger.warning(f"FAISS: Failed to initialize GPU resources ({e}). Falling back to CPU indexing.", exc_info=True)
            self.use_faiss = False
            self.faiss_res = None

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Safely convert tensor to numpy array."""
        return tensor.detach().cpu().numpy().astype(np.float32)

    # --- Sampling Methods ---
    def reservoir_sampling(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """Standard Reservoir Sampling."""
        n = features.shape[0]
        if n == 0: return features # Handle empty input
        k = min(n, k) # Cannot sample more than available
        if k == 0: return features.new_empty((0, features.shape[1]))
        reservoir = features[:k].clone()
        for i in range(k, n):
            j = torch.randint(0, i + 1, (1,), device=features.device).item() # Ensure randint is on correct device
            if j < k:
                reservoir[j] = features[i]
        return reservoir

    def class_balanced_sampling(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Samples features balancing across classes present in labels."""
        if labels is None or features.shape[0] != labels.shape[0]:
            logger.warning("Class balanced sampling requires valid labels matching features count. Falling back to reservoir.")
            return self.reservoir_sampling(features, self.memory_size)

        # Ensure labels are flat [N] and on the correct device
        labels = labels.view(-1).to(self._device)
        if features.device != self._device: features = features.to(self._device)

        unique_classes, counts = torch.unique(labels[labels >= 0], return_counts=True) # Ignore negative labels
        num_valid_classes = len(unique_classes)

        if num_valid_classes == 0:
             logger.warning("Class balanced sampling: No valid classes found in labels. Using reservoir sampling.")
             return self.reservoir_sampling(features, self.memory_size)

        # Calculate samples per class, ensuring at least 1 sample if class exists
        samples_per_class = max(1, self.memory_size // num_valid_classes)
        logger.debug(f"ClassBalSampling: {num_valid_classes} valid classes, aiming for {samples_per_class} samples/class.")

        balanced_features_list = []
        for cls_idx in unique_classes:
            cls_mask = (labels == cls_idx)
            cls_features = features[cls_mask]
            num_cls_features = cls_features.shape[0]

            if num_cls_features == 0: continue # Should not happen with unique_classes logic, but safe check

            # Sample from this class's features
            k_cls = min(num_cls_features, samples_per_class)
            sampled_cls_features = self.reservoir_sampling(cls_features, k_cls)
            balanced_features_list.append(sampled_cls_features)
            # logger.debug(f"  Class {cls_idx.item()}: Found {num_cls_features}, sampled {k_cls}.")

        if not balanced_features_list:
            logger.warning("Class balanced sampling resulted in zero features after per-class sampling. Using reservoir sampling on original.")
            return self.reservoir_sampling(features, self.memory_size) # Fallback

        balanced_features = torch.cat(balanced_features_list, dim=0)

        # If total samples are still less than memory_size, fill remaining with reservoir sampling from original
        num_balanced_samples = balanced_features.shape[0]
        if num_balanced_samples < self.memory_size:
            remaining_needed = self.memory_size - num_balanced_samples
            logger.debug(f"ClassBalSampling filled {num_balanced_samples}. Need {remaining_needed} more. Using reservoir.")
            # Reservoir sample from the *original* features to fill the gap
            # Avoid sampling already included features? Complex. Simple reservoir for now.
            additional_features = self.reservoir_sampling(features, remaining_needed)
            # Check if additional features were actually sampled
            if additional_features.shape[0] > 0:
                balanced_features = torch.cat([balanced_features, additional_features], dim=0)

        # Final check: Ensure we don't exceed memory_size (due to rounding/minimums)
        if balanced_features.shape[0] > self.memory_size:
            logger.warning(f"ClassBalSampling exceeded target size ({balanced_features.shape[0]} > {self.memory_size}). Truncating.")
            balanced_features = balanced_features[:self.memory_size]

        logger.debug(f"Class balanced sampling resulted in {balanced_features.shape[0]} features.")
        return balanced_features

    def kmeans_sampling(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """Samples k features using K-Means clustering centroids."""
        n_features = features.shape[0]
        if not self.use_faiss or n_features < k or not FAISS_AVAILABLE:
            logger.debug(f"KMeans sampling requirements not met (FAISS:{self.use_faiss}, N:{n_features}<k:{k}). Falling back to reservoir.")
            return self.reservoir_sampling(features, k)

        if features.device != self._device: features = features.to(self._device)
        features_np = self._to_numpy(features) # Convert to NumPy for FAISS Kmeans
        faiss.normalize_L2(features_np) # Normalize for K-means distance

        # Determine if GPU should be used for K-means
        use_gpu_kmeans = (self.faiss_res is not None)

        logger.info(f"Performing K-means clustering (k={k}) on {n_features} features (GPU: {use_gpu_kmeans})...")
        kmeans = faiss.Kmeans(d=self.feature_dim, k=k, niter=20, verbose=False, gpu=use_gpu_kmeans)
        try:
            kmeans.train(features_np)
            centroids = torch.from_numpy(kmeans.centroids).to(self._device) # Move centroids back to torch device
            logger.info("K-means clustering complete.")
            self.memory_tracker.log_memory_usage("After K-means")
            # Normalize centroids? Usually not needed if input was normalized.
            # centroids = F.normalize(centroids, p=2, dim=1)
            return centroids
        except Exception as e:
            logger.error(f"K-means failed: {e}. Falling back to reservoir sampling.", exc_info=True)
            return self.reservoir_sampling(features, k)
        finally:
             del features_np # Explicit cleanup
             gc.collect()

    # --- FAISS Indexing and Querying ---
    def create_faiss_index(self, features: Optional[torch.Tensor]=None):
        """Builds or rebuilds the FAISS index with current memory bank content."""
        if not self.use_faiss or not FAISS_AVAILABLE:
            self.faiss_index = None
            return

        if features is None:
            # Use only the initialized part of the memory bank
            if not self.memory_initialized:
                 logger.debug("FAISS: Memory bank not initialized, skipping index creation.")
                 self.faiss_index = None
                 return
            # Determine actual number of stored vectors (handle circular buffer)
            # This logic might be tricky if ptr wrapped around fully.
            # Simplification: Assume memory_bank contains valid features up to memory_size if initialized.
            # A better approach might track the number of valid entries separately.
            # For now, using the full bank if initialized.
            features_to_index = self.memory_bank # Potentially includes old/zero vectors if not full
            logger.debug(f"FAISS: Rebuilding index using current memory bank (size {features_to_index.shape[0]}).")
        else:
            features_to_index = features

        # --- Ensure features are on the correct device for FAISS ---
        # FAISS GPU index expects CPU numpy data for add, or direct GPU tensor (experimental?)
        # FAISS CPU index expects CPU numpy data
        # Our convention: Convert features to CPU numpy for index building
        if features_to_index.device != torch.device('cpu'):
             logger.debug(f"FAISS: Moving features from {features_to_index.device} to CPU for index creation.")
             features_to_index_cpu = features_to_index.cpu()
        else:
             features_to_index_cpu = features_to_index

        if features_to_index_cpu.shape[0] == 0:
             logger.warning("FAISS: Attempted to build index with 0 features. Skipping.")
             self.faiss_index = None
             return

        # Convert to numpy AFTER moving to CPU
        features_np = self._to_numpy(features_to_index_cpu)
        del features_to_index_cpu # Cleanup CPU copy if created
        faiss.normalize_L2(features_np) # Normalize data for L2 index

        logger.info(f"Creating FAISS index for {features_np.shape[0]} features (dim={self.feature_dim})...")
        # --- Index Creation Logic ---
        try:
            if self.faiss_res: # Use GPU index
                # Using IndexFlatL2 on GPU is often fast enough
                cpu_index = faiss.IndexFlatL2(self.feature_dim)
                gpu_id = self._device.index if self._device.index is not None else 0
                gpu_index = faiss.index_cpu_to_gpu(self.faiss_res, gpu_id, cpu_index)
                gpu_index.add(features_np) # Add CPU numpy data to GPU index
                self.faiss_index = gpu_index
                logger.info(f"FAISS: Created GpuIndexFlatL2 on device {self._device}.")
            else: # Use CPU index
                self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
                self.faiss_index.add(features_np)
                logger.info("FAISS: Created IndexFlatL2 (CPU).")

            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors.")
        except Exception as e:
             logger.error(f"FAISS: Failed to create index ({e}). Disabling FAISS.", exc_info=True)
             self.use_faiss = False # Disable if creation fails
             self.faiss_index = None
             self.faiss_res = None # Release GPU resources if failed
        finally:
            del features_np
            self.memory_tracker.clear_memory("FAISS Index Creation")
            self.memory_tracker.log_memory_usage("After FAISS index creation")


    def query_faiss(self, query_features: torch.Tensor, k: int = 5) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Queries the FAISS index."""
        if not self.use_faiss or self.faiss_index is None or not FAISS_AVAILABLE:
            # logger.debug("FAISS query skipped (disabled or index None).")
            return None, None
        if self.faiss_index.ntotal == 0:
            logger.warning("FAISS query: Index is empty.")
            return None, None
        if query_features.shape[0] == 0:
            logger.debug("FAISS query skipped (0 query features).")
            return None, None

        k = min(k, self.faiss_index.ntotal) # Ensure k is not larger than index size
        if k <= 0:
            logger.warning(f"FAISS query: k={k} is non-positive. Skipping.")
            return None, None

        # --- Prepare queries for FAISS ---
        # GPU Index Search: Can take GPU tensor directly if using standard resources? Docs are a bit unclear.
        # Safest bet compatible with both CPU/GPU index: Convert queries to CPU numpy.
        original_query_device = query_features.device
        if query_features.device != torch.device('cpu'):
            # logger.debug(f"FAISS Query: Moving query features from {query_features.device} to CPU.")
            query_features_cpu = query_features.cpu()
        else:
            query_features_cpu = query_features

        query_np = self._to_numpy(query_features_cpu)
        del query_features_cpu # Cleanup CPU copy if made
        faiss.normalize_L2(query_np) # Normalize queries to match indexed data

        try:
            distances, indices = self.faiss_index.search(query_np, k)
            # Convert results back to torch tensors on the *original* query device
            return torch.from_numpy(distances).to(original_query_device), \
                   torch.from_numpy(indices).to(original_query_device)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            return None, None
        finally:
            del query_np # Cleanup numpy array

    # --- Memory Update ---
    def update_memory(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Updates the memory bank with new features, applying normalization and sampling.

        Args:
            features (torch.Tensor): Input features [N, feature_dim]. MUST be on self._device.
            labels (Optional[torch.Tensor]): Corresponding labels [N] for class-balanced sampling.
                                             MUST be on self._device.
        """
        self.memory_tracker.log_memory_usage("Start Memory Update")
        with torch.no_grad():
            if features.shape[0] == 0:
                logger.debug("MemoryManager update: Received 0 features.")
                return
            if features.shape[1] != self.feature_dim:
                 logger.error(f"MemoryManager update: Feature dim mismatch ({features.shape[1]} vs {self.feature_dim}). Skipping.")
                 return
            # Check device and move if necessary (safeguard)
            if features.device != self._device:
                 logger.warning(f"MemoryManager update: Features received on {features.device}, expected {self._device}. Moving.")
                 features = features.to(self._device)
            if labels is not None and labels.device != self._device:
                 logger.warning(f"MemoryManager update: Labels received on {labels.device}, expected {self._device}. Moving.")
                 labels = labels.to(self._device)
            if labels is not None and features.shape[0] != labels.shape[0]:
                 logger.warning("MemoryManager update: Features and labels count mismatch. Disabling labels for this update.")
                 labels = None

            # --- Preprocessing ---
            features = F.normalize(features, p=2, dim=1) # Normalize features L2 norm

            # --- Sampling ---
            # Decide sampling strategy
            if labels is not None:
                logger.debug("MemoryManager: Applying Class-Balanced Sampling.")
                sampled_features = self.class_balanced_sampling(features, labels)
                # Note: Labels are not stored after class-balanced sampling by default.
            else:
                 if self.use_faiss and features.shape[0] > self.memory_size * 2: # Heuristic for Kmeans
                      logger.debug("MemoryManager: Applying K-means Sampling.")
                      num_to_sample = min(features.shape[0], self.memory_size)
                      sampled_features = self.kmeans_sampling(features, num_to_sample)
                 else:
                      logger.debug("MemoryManager: Applying Reservoir Sampling.")
                      num_to_sample = min(features.shape[0], self.memory_size)
                      sampled_features = self.reservoir_sampling(features, num_to_sample)
            # Labels are generally lost after sampling unless explicitly stored/handled

            num_features_to_add = sampled_features.shape[0]
            if num_features_to_add == 0:
                 logger.debug("MemoryManager update: 0 features after sampling.")
                 return

            logger.debug(f"MemoryManager: Adding {num_features_to_add} sampled features to bank (size {self.memory_size}).")

            # --- Update Memory Bank (Circular Buffer) ---
            ptr = self.memory_ptr[0].item()
            # Calculate indices to overwrite - ENSURE DEVICE MATCHES memory_bank
            indices = torch.arange(ptr, ptr + num_features_to_add, device=self._device) % self.memory_size

            # Ensure the number of features doesn't exceed indices available
            if len(indices) < num_features_to_add:
                 logger.warning(f"Indices calculation error? Got {len(indices)} indices for {num_features_to_add} features. Truncating features.")
                 num_features_to_add = len(indices)
                 sampled_features = sampled_features[:num_features_to_add]

            # --- **CRITICAL FIX AREA** ---
            # Ensure sampled_features are on the same device as memory_bank before assignment
            if sampled_features.device != self.memory_bank.device:
                 logger.warning(f"Device mismatch before memory assignment: features on {sampled_features.device}, memory_bank on {self.memory_bank.device}. Moving features.")
                 sampled_features = sampled_features.to(self.memory_bank.device)
            # --- End Critical Fix Area ---

            self.memory_bank[indices] = sampled_features
            # If labels were preserved through sampling, update self.memory_labels[indices] here
            # For now, assumes no labels are stored alongside features.
            # self.memory_labels[indices] = sampled_labels # Hypothetical

            # Update pointer
            new_ptr = (ptr + num_features_to_add) % self.memory_size
            self.memory_ptr[0] = new_ptr

            if not self.memory_initialized:
                if new_ptr > 0 or num_features_to_add >= self.memory_size:
                     self.memory_initialized = True
                     logger.info("MemoryManager marked as initialized.")

            # --- Rebuild FAISS Index ---
            if self.use_faiss:
                self.create_faiss_index(self.memory_bank)

        self.memory_tracker.log_memory_usage("After memory update")
        self.memory_tracker.clear_memory("Memory Update Cleanup")


# --- Efficient Segmentation Decoder ---
class EfficientSegmentationDecoder(nn.Module):
    """
    Decoder that applies low-resolution self-attention and progressive upsampling.
    """
    def __init__(self, in_channels: int, num_classes: int, feature_dim: int = 128, attention_heads: int = 8):
        super(EfficientSegmentationDecoder, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.attention_heads = attention_heads
        if feature_dim % attention_heads != 0:
            logger.warning(f"EfficientDecoder: feature_dim {feature_dim} not divisible by heads {attention_heads}. Adjusting feature_dim.")
            self.feature_dim = (feature_dim // attention_heads) * attention_heads
            logger.warning(f"Adjusted feature_dim to {self.feature_dim}")

        self.feature_projector = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1)
        self.head_dim = self.feature_dim // self.attention_heads

        # Use ModuleDict for QKV projections
        self.qkv_conv = nn.ModuleDict({
            'query': nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False),
            'key': nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False),
            'value': nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
        })
        self.out_proj = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1) # Output projection after attention
        self.classifier = nn.Conv2d(self.feature_dim, num_classes + 1, kernel_size=1) # Include OOD class
        self.memory_tracker = MemoryTracker(log_interval=10, verbose=True)

        # Max resolution for full attention calculation (e.g., 64x64 = 4096 tokens)
        self.attn_max_tokens = 64 * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        self.memory_tracker.log_memory_usage("EfficientDecoder start")

        features = self.feature_projector(x) # [B, feature_dim, H, W]

        # --- Determine Downscaling for Attention ---
        num_input_tokens = h * w
        downscale_factor = 1
        attn_features = features # Default: use original features
        if num_input_tokens > self.attn_max_tokens:
            downscale_factor = math.ceil(math.sqrt(num_input_tokens / self.attn_max_tokens))
            # Ensure factor is valid and pooling is possible
            downscale_factor = min(downscale_factor, h, w)
            if downscale_factor > 1:
                logger.debug(f"EfficientDecoder: Input {h}x{w} ({num_input_tokens} tokens) > max {self.attn_max_tokens}. Downscaling attention input by {downscale_factor}x.")
                # Use AvgPool for downscaling features for attention
                attn_features = F.avg_pool2d(features, kernel_size=downscale_factor, stride=downscale_factor)
            else:
                 logger.debug(f"EfficientDecoder: Calculated downscale factor is 1. Using original features for attention.")
                 downscale_factor = 1 # Reset if calculation leads to 1


        ah, aw = attn_features.size(2), attn_features.size(3)
        num_attn_tokens = ah * aw

        # --- Multi-Head Self-Attention ---
        queries = self.qkv_conv['query'](attn_features) # [B, feature_dim, ah, aw]
        keys = self.qkv_conv['key'](attn_features)
        values = self.qkv_conv['value'](attn_features)

        # Reshape for attention: B, heads, tokens, head_dim
        queries = queries.view(batch_size, self.attention_heads, self.head_dim, num_attn_tokens).permute(0, 1, 3, 2) # [B, heads, tokens, h_dim]
        # Reshape keys: B, heads, head_dim, tokens
        keys = keys.view(batch_size, self.attention_heads, self.head_dim, num_attn_tokens) # [B, heads, h_dim, tokens]
        # Reshape values: B, heads, tokens, head_dim
        values = values.view(batch_size, self.attention_heads, self.head_dim, num_attn_tokens).permute(0, 1, 3, 2) # [B, heads, tokens, h_dim]


        # --- Scaled Dot-Product Attention ---
        # Prefer PyTorch 2.0's optimized implementation if available
        use_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        if use_flash_attn:
             # logger.debug("EfficientDecoder: Using F.scaled_dot_product_attention")
             try:
                # Input shapes: query[B, H, Tq, D], key[B, H, Tk, D], value[B, H, Tv, D] (Tq=Tk=Tv=num_attn_tokens, D=head_dim)
                # Key needs permute to [B, H, Tk, D] from [B, H, D, Tk] before passing
                attention_output = F.scaled_dot_product_attention(
                    queries, keys.permute(0, 1, 3, 2), values,
                    attn_mask=None, dropout_p=0.0, is_causal=False
                )
                # Output shape: [B, heads, tokens, head_dim]
             except Exception as flash_e:
                 logger.warning(f"F.scaled_dot_product_attention failed: {flash_e}. Falling back to manual calculation.")
                 use_flash_attn = False # Fallback flag

        if not use_flash_attn: # Manual calculation or fallback
             # logger.debug("EfficientDecoder: Using manual attention calculation")
             # Key shape for matmul: [B, heads, h_dim, tokens] -> [B, heads, tokens, h_dim]
             # [B, heads, tokens, h_dim] @ [B, heads, h_dim, tokens] -> [B, heads, tokens, tokens]
             attention_scores = torch.matmul(queries, keys) / math.sqrt(self.head_dim)
             attention_weights = F.softmax(attention_scores, dim=-1) # Softmax over key tokens
             # [B, heads, tokens, tokens] @ [B, heads, tokens, h_dim] -> [B, heads, tokens, h_dim]
             attention_output = torch.matmul(attention_weights, values)
             del attention_scores, attention_weights # Cleanup

        # Reshape attention output back to spatial format [B, C, ah, aw]
        # [B, heads, tokens, h_dim] -> permute -> [B, tokens, heads, h_dim] -> reshape -> [B, tokens, C] -> view -> [B, C, ah, aw]
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, num_attn_tokens, self.feature_dim)
        attention_output = attention_output.view(batch_size, ah, aw, self.feature_dim).permute(0, 3, 1, 2) # [B, C, ah, aw]

        # --- Post-Attention Processing ---
        # Apply output projection
        attention_output = self.out_proj(attention_output)

        # Upsample if attention was downscaled
        if downscale_factor > 1:
            attention_output = F.interpolate(attention_output, size=(h, w), mode='bilinear', align_corners=False)
            final_features = attention_output + features # Add residual from initial projection
        else:
            final_features = attention_output + attn_features # Add residual from attention input features

        # Classifier
        output = self.classifier(final_features)

        self.memory_tracker.log_memory_usage("EfficientDecoder end")
        return output


# --- Hopfield PEBAL Model ---
class HopfieldPEBALModel(nn.Module):
    """
    Integrates EfficientMemoryManager into a PEBAL-like segmentation model.
    Optionally uses EfficientSegmentationDecoder.
    Handles feature extraction, adaptation, memory interaction, and final classification.
    """
    def __init__(self,
                 backbone: nn.Module,
                 segmentation_head: nn.Module,
                 num_classes: int = 19,
                 memory_feature_dim: int = 256,
                 memory_size: int = 1000,
                 insertion_point: str = 'after_backbone',
                 target_feature_dim: Optional[int] = None,
                 use_efficient_memory: bool = True,
                 use_faiss: bool = True,
                 pq_bytes: int = 8,
                 sampling_stride: int = 2,
                 memory_log_interval: int = 10,
                 memory_log_verbose: bool = True,
                 use_efficient_decoder: bool = False,
                 efficient_decoder_kwargs: Optional[Dict] = None,
                 memory_beta: float = 8.0
                 ):
        super().__init__()

        # --- Basic Assertions and Setup ---
        assert insertion_point in ['after_backbone', 'after_seghead'], \
            "insertion_point must be 'after_backbone' or 'after_seghead'"
        if use_faiss and not FAISS_AVAILABLE:
             logger.warning("use_faiss=True but FAISS library not found. Disabling FAISS.")
             use_faiss = False

        self.backbone = backbone
        self._original_segmentation_head = segmentation_head # Store original head
        self.num_classes = num_classes
        self.insertion_point = insertion_point
        self.use_efficient_memory = use_efficient_memory
        self.sampling_stride = sampling_stride if use_efficient_memory else 1
        self.memory_beta = memory_beta

        self.memory_tracker = MemoryTracker(log_interval=memory_log_interval, verbose=memory_log_verbose)

        # --- Device Detection ---
        self._model_device = self._get_module_device(backbone)
        if self._model_device is None:
             self._model_device = self._get_module_device(segmentation_head)
        if self._model_device is None:
             self._model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
             logger.warning(f"Could not detect device from backbone/head. Defaulting to {self._model_device}.")
        else:
             logger.info(f"Determined model device from components: {self._model_device}")


        # --- Instantiate Efficient Decoder (if requested) ---
        self.segmentation_head = self._original_segmentation_head # Default
        self.use_efficient_decoder = use_efficient_decoder
        self._efficient_decoder_instance = None
        if self.use_efficient_decoder:
            logger.info("Attempting to replace segmentation head with EfficientSegmentationDecoder.")
            backbone_dim = self._detect_feature_dimensions(after_seghead=False, device=self._model_device)
            if backbone_dim is None:
                raise RuntimeError("Cannot use EfficientSegmentationDecoder: Failed to detect backbone output dimension.")

            eff_decoder_defaults = {
                'in_channels': backbone_dim,
                'num_classes': self.num_classes,
                'feature_dim': memory_feature_dim, # Use memory dim for internal processing
                'attention_heads': 4 # Default attention heads
            }
            if efficient_decoder_kwargs:
                eff_decoder_defaults.update(efficient_decoder_kwargs)

            try:
                # Instantiate on the target device directly
                self._efficient_decoder_instance = EfficientSegmentationDecoder(**eff_decoder_defaults).to(self._model_device)
                if self.insertion_point == 'after_backbone':
                    logger.info(f"Using EfficientSegmentationDecoder (In: {backbone_dim} -> Feature: {eff_decoder_defaults['feature_dim']}) instead of provided segmentation head for logits.")
                    self.segmentation_head = self._efficient_decoder_instance # Replace the head reference
                else: # insertion_point == 'after_seghead'
                     # In this mode, the efficient decoder processes features *after* the original seg head & adapter
                     logger.warning("Using EfficientSegmentationDecoder with insertion_point='after_seghead'. It will process features from the *adapter* output.")
                     # We need to adjust the eff_decoder's in_channels to match the adapter's output (target_feature_dim)
                     adapter_out_dim = self._detect_feature_dimensions(after_seghead=True, device=self._model_device) # Get dim after original head
                     adapter_out_dim = target_feature_dim if target_feature_dim is not None else adapter_out_dim # Adapter maps to target_dim
                     if adapter_out_dim != eff_decoder_defaults['in_channels']:
                         logger.info(f"Re-configuring EfficientDecoder input channels for 'after_seghead': {adapter_out_dim}")
                         eff_decoder_defaults['in_channels'] = adapter_out_dim
                         # Re-instantiate with corrected input dim
                         del self._efficient_decoder_instance # Delete previous instance
                         self._efficient_decoder_instance = EfficientSegmentationDecoder(**eff_decoder_defaults).to(self._model_device)

                     # self.segmentation_head remains the original head. Eff decoder used later for logits.


            except Exception as e:
                logger.error(f"Failed to instantiate EfficientSegmentationDecoder: {e}. Falling back to original head.", exc_info=True)
                self.use_efficient_decoder = False # Disable if instantiation fails
                self._efficient_decoder_instance = None
                self.segmentation_head = self._original_segmentation_head # Ensure fallback


        # --- Dimension Detection ---
        self._input_dim_after_feature_extractor = self._detect_feature_dimensions(
            after_seghead=(insertion_point == 'after_seghead'),
            use_original_head=(insertion_point == 'after_seghead'), # Use original head for detection if inserting after it
            device=self._model_device
        )
        if self._input_dim_after_feature_extractor is None:
            raise RuntimeError("Failed to detect feature dimensions after primary feature extractor.")
        logger.info(f"Detected feature dimension after feature extractor ('{insertion_point}' stage): {self._input_dim_after_feature_extractor}")

        # --- Determine Target Dimension and Need for Adapters ---
        dim_before_modules = self._input_dim_after_feature_extractor
        if target_feature_dim is None:
            self._target_feature_dim = 512 if dim_before_modules > 512 else dim_before_modules
            logger.warning(f"--target_feature_dim not set, defaulting to {self._target_feature_dim}.")
        else:
            self._target_feature_dim = target_feature_dim

        logger.info(f"Effective dimension before memory modules: {dim_before_modules}")
        logger.info(f"Target dimension after adapter: {self._target_feature_dim}")
        logger.info(f"Memory interaction dimension: {memory_feature_dim}")


        # --- Adapters and Projections ---
        self.needs_adapter = (dim_before_modules != self._target_feature_dim)
        self.channel_adapter = nn.Identity()
        if self.needs_adapter:
            logger.info(f"Adding Channel Adapter: {dim_before_modules} -> {self._target_feature_dim}")
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(dim_before_modules, self._target_feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self._target_feature_dim),
                nn.ReLU(inplace=True)
            ).to(self._model_device)

        self.memory_input_proj = nn.Conv2d(self._target_feature_dim, memory_feature_dim, kernel_size=1).to(self._model_device)

        # --- Efficient Memory Manager ---
        self.memory_manager = EfficientMemoryManager(
            feature_dim=memory_feature_dim,
            memory_size=memory_size,
            pq_bytes=pq_bytes,
            num_classes=num_classes,
            use_faiss=use_faiss,
            device=self._model_device # Pass device
        )
        logger.info(f"Initialized EfficientMemoryManager (FAISS: {self.memory_manager.use_faiss}, Size: {memory_size}, Dim: {memory_feature_dim}) on device {self._model_device}")

        # --- Output Projections and Final Layers ---
        self.final_seghead_proj = nn.Identity()

        # Determine the final classification layer
        if self.insertion_point == 'after_backbone':
            if self.use_efficient_decoder:
                self.final_classifier = None # Logits from efficient decoder
                logger.info("Logits will be generated by the Efficient Decoder.")
            else:
                self.final_classifier = None # Logits from original head
                self._check_and_prepare_seghead_projection() # Prepare projection if needed
                logger.info("Logits will be generated by the original Segmentation Head (potentially after projection).")
        else: # insertion_point == 'after_seghead'
            if self.use_efficient_decoder:
                self.final_classifier = None # Logits from efficient decoder (processing adapted seg_head features)
                logger.info("Logits will be generated by the Efficient Decoder (processing adapted seg_head features).")
            else:
                 # Need final classifier after memory projection
                 logger.info(f"Adding Final Classifier: {memory_feature_dim} -> {self.num_classes + 1}")
                 self.final_classifier = nn.Conv2d(memory_feature_dim, self.num_classes + 1, kernel_size=1).to(self._model_device)


        self.energy_head = nn.Sequential(
            nn.Conv2d(memory_feature_dim, memory_feature_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(memory_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(memory_feature_dim // 2, 1, kernel_size=1)
        ).to(self._model_device)

        self._initialize_weights()
        # Final move to device (should handle all registered params/buffers including memory manager)
        self.to(self._model_device)
        logger.info(f"HopfieldPEBALModel initialized and moved to device: {self._model_device}")


    def _get_module_device(self, module: nn.Module) -> Optional[torch.device]:
        """Safely gets the device of a module from its parameters or buffers."""
        if not isinstance(module, nn.Module): return None # Handle cases where module might be None
        try:
            params = list(module.parameters())
            if params: return params[0].device
            buffers = list(module.buffers())
            if buffers: return buffers[0].device
            # Check submodules recursively
            for child in module.children():
                device = self._get_module_device(child)
                if device: return device
            return None
        except Exception as e:
            logger.error(f"Error getting device for module {type(module).__name__}: {e}", exc_info=False) # Less verbose log
            return None

    def _detect_feature_dimensions(self, after_seghead: bool = False, use_original_head: bool = False, device: Optional[Union[str, torch.device]] = None) -> Optional[int]:
        """Detect feature dimensions using a dummy forward pass."""
        target_device = device if device else self._model_device
        if isinstance(target_device, str): target_device = torch.device(target_device)

        original_backbone_device = self._get_module_device(self.backbone)
        _backbone = self.backbone.to(target_device)

        # Select the correct head to check based on flags
        if after_seghead and use_original_head:
            head_to_check = self._original_segmentation_head
        elif after_seghead and not use_original_head: # e.g. checking efficient decoder after original head
             head_to_check = self.segmentation_head # This might be the efficient decoder
        else: # after_backbone
            head_to_check = self.segmentation_head # This is either original or efficient if replaced

        original_seghead_device = self._get_module_device(head_to_check) if head_to_check else None
        _seg_head = head_to_check.to(target_device) if head_to_check else None

        _backbone.eval()
        if _seg_head: _seg_head.eval()

        detected_dim = None
        try:
            dummy_input = torch.zeros(1, 3, 64, 64, device=target_device)
            with torch.no_grad():
                features = _backbone(dummy_input)
                if isinstance(features, (tuple, list)): features = features[-1]

                if after_seghead:
                    if _seg_head is None:
                        raise ValueError("Segmentation head is None, cannot detect dimension after it.")

                    # Detect input dim of the specific head being checked
                    first_conv_or_proj = None
                    target_in_channels = None
                    for module in _seg_head.modules():
                        if isinstance(module, (nn.Conv2d, nn.Linear)): # Check Conv first usually
                            first_conv_or_proj = module
                            target_in_channels = getattr(module, 'in_channels', getattr(module, 'in_features', None))
                            break
                        elif isinstance(module, EfficientSegmentationDecoder): # Check if it's EffDec itself
                            # EffDec's first layer is feature_projector
                            first_conv_or_proj = module.feature_projector
                            target_in_channels = getattr(first_conv_or_proj, 'in_channels', None)
                            break

                    temp_proj = None
                    if target_in_channels is not None and target_in_channels != features.shape[1]:
                         logger.debug(f"Temp adapting input for dim detection: {features.shape[1]} -> {target_in_channels}")
                         temp_proj = nn.Conv2d(features.shape[1], target_in_channels, 1).to(target_device)
                         features = temp_proj(features)

                    # Run through the selected segmentation head
                    seg_features = _seg_head(features)
                    if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                    detected_dim = seg_features.shape[1] # Get channel dimension
                    if temp_proj is not None: del temp_proj # Clean up temp layer

                else: # Detect dimension directly after backbone
                    detected_dim = features.shape[1]

        except Exception as e:
            logger.error(f"Error during feature dimension detection forward pass: {e}", exc_info=True)
            detected_dim = None
        finally:
            # Move modules back safely
            if original_backbone_device is not None: self.backbone.to(original_backbone_device)
            else: self.backbone.cpu() # Fallback
            if original_seghead_device is not None and head_to_check: head_to_check.to(original_seghead_device)
            elif head_to_check: head_to_check.cpu()

            del _backbone # Remove references
            if '_seg_head' in locals() and _seg_head is not None: del _seg_head
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return detected_dim

    def _check_and_prepare_seghead_projection(self):
        """Checks if the original seg head needs an input projection and creates it."""
        if self.insertion_point != 'after_backbone' or self.use_efficient_decoder or self._original_segmentation_head is None:
            self.final_seghead_proj = nn.Identity()
            return

        try:
            first_layer = None
            expected_input_dim = None
            for module in self._original_segmentation_head.modules():
                if module is self._original_segmentation_head or isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                    continue
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    first_layer = module
                    expected_input_dim = getattr(first_layer, 'in_channels', getattr(first_layer, 'in_features', None))
                    if expected_input_dim is not None: break # Found first operational layer with input dim

            if expected_input_dim:
                 self._original_seghead_in_channels = expected_input_dim
                 logger.info(f"Original seg head's first layer expects input dim: {self._original_seghead_in_channels}")
                 dim_after_adapter = self._target_feature_dim
                 if self._original_seghead_in_channels != dim_after_adapter:
                     logger.warning(f"Adding final projection before original seg head: {dim_after_adapter} -> {self._original_seghead_in_channels}")
                     self.final_seghead_proj = nn.Conv2d(dim_after_adapter, self._original_seghead_in_channels, kernel_size=1).to(self._model_device)
                 else:
                     self.final_seghead_proj = nn.Identity()
            else:
                 logger.warning("Could not determine input dim for original seg head's first layer. Assuming no projection needed.")
                 self.final_seghead_proj = nn.Identity()

        except Exception as e:
             logger.error(f"Error checking original seg head input dim: {e}. Assuming no projection needed.", exc_info=True)
             self.final_seghead_proj = nn.Identity()


    def _initialize_weights(self):
        """Initialize weights for newly added adapter/projection/head layers."""
        logger.info("Initializing weights for newly added layers...")
        init_count = 0
        initialized_modules = set()

        components_to_init = [
            self.channel_adapter,
            self.memory_input_proj,
            self.final_seghead_proj,
            self.final_classifier,
            self.energy_head,
            self._efficient_decoder_instance
        ]

        for component in components_to_init:
            if component is None or isinstance(component, nn.Identity):
                continue

            for m in component.modules():
                 if m in initialized_modules or m == component or isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Identity)):
                     continue # Skip containers, identity, and already initialized

                 try:
                     if isinstance(m, (nn.Conv2d, nn.Linear)):
                         if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                             if m.weight.ndim > 1:
                                 if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight)
                                 else: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                                 init_count += 1
                                 initialized_modules.add(m)
                         if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                             nn.init.constant_(m.bias, 0)
                     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                         if getattr(m, 'affine', True):
                             if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                                 nn.init.constant_(m.weight, 1)
                             if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                                 nn.init.constant_(m.bias, 0)
                             init_count += 1
                             initialized_modules.add(m)
                 except Exception as init_e:
                       logger.warning(f"Could not initialize layer {m}: {init_e}", exc_info=False)


        logger.info(f"Weight initialization complete ({init_count} potential modules initialized).")
        logger.info("Note: Backbone and original SegHead weights are assumed pre-initialized/loaded.")


    def _apply_memory_interaction(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to interact with memory manager and calculate memory energy.
        Input: features projected to memory dimension [B, C_mem, H, W], already on self._model_device.
        Output: memory_energy_map [B, 1, H, W], flat_features_normalized [N, C_mem]
        """
        b, c_mem, h_in, w_in = features.shape
        current_device = features.device
        self.memory_tracker.log_memory_usage("Start Memory Interaction")

        dummy_energy = torch.zeros(b, 1, h_in, w_in, device=current_device)
        dummy_flat = torch.zeros(0, c_mem, device=current_device)

        h_proc, w_proc = h_in, w_in
        features_sampled = features
        needs_upsampling = False
        apply_spatial_sampling = (self.use_efficient_memory and h_in * w_in > 4096 and self.sampling_stride > 1)

        if apply_spatial_sampling:
            stride = self.sampling_stride
            if h_in >= stride and w_in >= stride:
                features_sampled = features[:, :, ::stride, ::stride]
                h_proc, w_proc = features_sampled.shape[2:]
                needs_upsampling = True
                logger.debug(f"Memory interaction spatially sampled input from {h_in}x{w_in} to {h_proc}x{w_proc}")
            else:
                logger.warning(f"Cannot apply spatial stride {stride} to features of size {h_in}x{w_in} for memory interaction.")

        if h_proc <= 0 or w_proc <= 0:
             logger.error("Zero spatial dimension after potential sampling for memory interaction. Skipping.")
             return dummy_energy, dummy_flat

        try:
            flat_features = features_sampled.permute(0, 2, 3, 1).contiguous().view(-1, c_mem)
        except Exception as e:
             logger.error(f"Error reshaping features for memory interaction: {e}", exc_info=True)
             return dummy_energy, dummy_flat

        if flat_features.shape[0] == 0:
             logger.debug("No features left after flattening for memory interaction.")
             return dummy_energy, dummy_flat

        flat_features_normalized = F.normalize(flat_features, p=2, dim=1)
        memory_energy_flat = torch.zeros(flat_features_normalized.shape[0], device=current_device)

        if self.memory_manager.memory_initialized:
            if self.memory_manager.use_faiss and self.memory_manager.faiss_index is not None:
                logger.debug("Querying FAISS index for memory energy.")
                k_neighbors = 1
                distances, indices = self.memory_manager.query_faiss(flat_features_normalized, k=k_neighbors)
                if distances is not None and indices is not None:
                    # Energy = distance * beta (L2 distance squared)
                    memory_energy_flat = distances[:, 0] * self.memory_beta # Shape [N]
                    self.memory_tracker.log_memory_usage("After FAISS Query")
                else:
                    logger.warning("FAISS query returned None. Using zero memory energy.")
            else: # Manual Path
                logger.debug("Calculating memory energy manually (dot product with memory bank).")
                try:
                    mem_bank_current = self.memory_manager.memory_bank
                    # Ensure memory bank is usable (not all zeros if just initialized)
                    if mem_bank_current.device != flat_features_normalized.device:
                        logger.warning("Memory bank and features on different devices! Moving bank temporarily.")
                        mem_bank_current = mem_bank_current.to(flat_features_normalized.device)

                    if mem_bank_current.abs().sum() > 1e-6: # Check if bank has non-zero values
                        similarities = torch.matmul(flat_features_normalized, mem_bank_current.t())
                        max_similarity, _ = torch.max(similarities, dim=1)
                        # Energy = (1 - max_similarity) * beta
                        memory_energy_flat = (1.0 - max_similarity.clamp(max=1.0)) * self.memory_beta # Clamp sim <= 1
                        self.memory_tracker.log_memory_usage("After Manual Memory Query")
                        del similarities, max_similarity
                    else:
                         logger.debug("Manual memory query skipped: Memory bank appears empty/zero.")
                except Exception as e:
                    logger.error(f"Manual memory energy calculation failed: {e}. Using zero energy.", exc_info=True)
        else:
            logger.debug("Memory not initialized. Using zero memory energy.")

        try:
            memory_energy_map = memory_energy_flat.view(b, h_proc, w_proc, 1).permute(0, 3, 1, 2)
            if needs_upsampling:
                memory_energy_map = F.interpolate(memory_energy_map, size=(h_in, w_in), mode='bilinear', align_corners=False)
        except Exception as e:
             logger.error(f"Error reshaping/upsampling memory energy map: {e}. Returning dummy energy.", exc_info=True)
             return dummy_energy, flat_features_normalized

        self.memory_tracker.log_memory_usage("End Memory Interaction")
        return memory_energy_map, flat_features_normalized


    def _check_and_handle_nan_inf(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Checks for NaN/Inf in a tensor and replaces them with zeros."""
        if not isinstance(tensor, torch.Tensor):
             logger.warning(f"Input '{name}' is not a tensor ({type(tensor)}). Cannot check for NaN/Inf.")
             return tensor

        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if has_nan or has_inf:
            nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
            inf_count = torch.isinf(tensor).sum().item() if has_inf else 0
            logger.warning(f"NaN/Inf detected in '{name}'. Replacing w/ zeros. (NaNs: {nan_count}, Infs: {inf_count}, Shape: {tensor.shape}, Device: {tensor.device})")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    # --- Forward Pass ---
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the Hopfield PEBAL model."""
        self.memory_tracker.log_memory_usage("Forward Start")
        output_dict = {}
        b, _, h_in_img, w_in_img = x.shape
        current_device = x.device

        # --- Stage 1: Feature Extraction ---
        features = None # Initialize
        try:
            # Ensure backbone is on the correct device relative to input
            if self._get_module_device(self.backbone) != current_device:
                 logger.warning(f"Backbone on {self._get_module_device(self.backbone)}, input on {current_device}. Moving backbone.")
                 self.backbone.to(current_device)

            features = self.backbone(x)
            if isinstance(features, (tuple, list)): features = features[-1]
            features = self._check_and_handle_nan_inf(features, "Backbone Features")
            self.memory_tracker.log_memory_usage("After Backbone")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.critical(f"OOM error during Backbone processing: {e}. Returning zero tensors on CPU.", exc_info=True)
                dummy_logits = torch.zeros(b, self.num_classes + 1, h_in_img, w_in_img, device='cpu')
                dummy_energy = torch.zeros(b, 1, h_in_img, w_in_img, device='cpu')
                self.memory_tracker.clear_memory("OOM Fallback")
                return {'seg_logits': dummy_logits, 'memory_energy': dummy_energy.clone(),
                        'feature_energy': dummy_energy.clone(), 'pebal_energy': dummy_energy.clone(),
                        'combined_energy': dummy_energy.clone(), 'is_ood': torch.zeros(b, dtype=torch.bool, device='cpu')}
            else:
                 logger.error(f"Runtime error in backbone: {e}", exc_info=True); raise e
        except Exception as e:
             logger.error(f"Error in backbone forward pass: {e}", exc_info=True); raise e

        # Ensure features were successfully computed
        if features is None:
             logger.critical("Backbone features are None after processing block. Cannot continue.")
             # Return dummy outputs
             dummy_logits = torch.zeros(b, self.num_classes + 1, h_in_img, w_in_img, device=current_device)
             dummy_energy = torch.zeros(b, 1, h_in_img, w_in_img, device=current_device)
             return {'seg_logits': dummy_logits, 'memory_energy': dummy_energy.clone(),
                     'feature_energy': dummy_energy.clone(), 'pebal_energy': dummy_energy.clone(),
                     'combined_energy': dummy_energy.clone(), 'is_ood': torch.zeros(b, dtype=torch.bool, device=current_device)}


        b, c_feat, h_feat, w_feat = features.shape
        memory_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)
        feature_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)
        logits = None
        features_adapted = None
        features_mem_proj = None

        # --- Stage 2 & 3: Processing based on Insertion Point ---
        if self.insertion_point == 'after_backbone':
            try:
                features_adapted = self.channel_adapter(features)
                features_adapted = self._check_and_handle_nan_inf(features_adapted, "Features Adapted (After Backbone)")

                features_mem_proj = self.memory_input_proj(features_adapted)
                features_mem_proj = self._check_and_handle_nan_inf(features_mem_proj, "Features Memory Proj (After Backbone)")
                self.memory_tracker.log_memory_usage("After Adapters/Projections")

                memory_energy_map, _ = self._apply_memory_interaction(features_mem_proj)
                memory_energy_map = self._check_and_handle_nan_inf(memory_energy_map, "Memory Energy Map")

                feature_energy_map = self.energy_head(features_mem_proj)
                feature_energy_map = self._check_and_handle_nan_inf(feature_energy_map, "Feature Energy Map")

                # --- Calculate Logits ---
                if self.use_efficient_decoder:
                    logger.debug("Calculating logits using Efficient Decoder (on raw backbone features).")
                    if self._efficient_decoder_instance is not None:
                         # Efficient decoder runs on raw backbone features in this mode
                         logits = self._efficient_decoder_instance(features)
                    else:
                         raise RuntimeError("use_efficient_decoder is True, but _efficient_decoder_instance is None.")
                else: # Original Segmentation Head Path
                    seg_head_input = self.final_seghead_proj(features_adapted) # Project adapted features if needed
                    seg_head_input = self._check_and_handle_nan_inf(seg_head_input, "SegHead Input")
                    logger.debug(f"Calculating logits using original Segmentation Head.")
                    if self._original_segmentation_head:
                       logits = self._original_segmentation_head(seg_head_input)
                       if isinstance(logits, (tuple, list)): logits = logits[-1]
                    else:
                       logger.error("Original segmentation head is None, cannot calculate logits.")
                       logits = torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)

            except Exception as e:
                 logger.error(f"Error in 'after_backbone' processing path: {e}", exc_info=True)
                 # Use existing values or create dummies
                 logits = logits if logits is not None else torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)
                 memory_energy_map = memory_energy_map if memory_energy_map is not None else torch.zeros(b, 1, h_feat, w_feat, device=current_device)
                 feature_energy_map = feature_energy_map if feature_energy_map is not None else torch.zeros(b, 1, h_feat, w_feat, device=current_device)


        else: # insertion_point == 'after_seghead'
            seg_features = None # Initialize
            try:
                if self._original_segmentation_head is None:
                     raise RuntimeError("Original segmentation head is None, required for 'after_seghead' insertion.")
                seg_features = self._original_segmentation_head(features)
                if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                seg_features = self._check_and_handle_nan_inf(seg_features, "Seg Head Features")

                features_adapted = self.channel_adapter(seg_features)
                features_adapted = self._check_and_handle_nan_inf(features_adapted, "Features Adapted (After SegHead)")

                features_mem_proj = self.memory_input_proj(features_adapted)
                features_mem_proj = self._check_and_handle_nan_inf(features_mem_proj, "Features Memory Proj (After SegHead)")

                memory_energy_map, _ = self._apply_memory_interaction(features_mem_proj)
                memory_energy_map = self._check_and_handle_nan_inf(memory_energy_map, "Memory Energy Map")

                feature_energy_map = self.energy_head(features_mem_proj)
                feature_energy_map = self._check_and_handle_nan_inf(feature_energy_map, "Feature Energy Map")

                # --- Calculate Logits ---
                if self.use_efficient_decoder and self._efficient_decoder_instance is not None:
                    # Efficient decoder processes the adapted seg_head features
                    logger.debug("Calculating logits using Efficient Decoder (on adapted seg_head features).")
                    logits = self._efficient_decoder_instance(features_adapted)
                elif self.final_classifier is not None:
                    logger.debug("Calculating logits using Final Classifier (on memory-projected features).")
                    logits = self.final_classifier(features_mem_proj)
                else:
                    logger.error("No final classifier or efficient decoder available for 'after_seghead' logits.")
                    logits = torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)

            except Exception as e:
                 logger.error(f"Error in 'after_seghead' processing path: {e}", exc_info=True)
                 logits = logits if logits is not None else torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)
                 memory_energy_map = memory_energy_map if memory_energy_map is not None else torch.zeros(b, 1, h_feat, w_feat, device=current_device)
                 feature_energy_map = feature_energy_map if feature_energy_map is not None else torch.zeros(b, 1, h_feat, w_feat, device=current_device)


        # --- Final Checks and Interpolation ---
        if logits is None:
             logger.critical("Logits tensor is None after processing blocks. Creating zeros.")
             h_out, w_out = memory_energy_map.shape[-2:] if memory_energy_map is not None else (h_in_img // 8, w_in_img // 8) # Guess spatial size
             logits = torch.zeros(b, self.num_classes+1, h_out, w_out, device=current_device)

        logits = self._check_and_handle_nan_inf(logits, "Logits Pre-Interpolation")
        h_pre_interp, w_pre_interp = logits.shape[-2:]

        if h_pre_interp != h_in_img or w_pre_interp != w_in_img:
             logger.debug(f"Interpolating outputs from {(h_pre_interp, w_pre_interp)} to {(h_in_img, w_in_img)}")
             try:
                 logits_final = F.interpolate(logits, size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
                 # Ensure energy maps have 4 dims [B, C=1, H, W] before interpolate
                 memory_energy_final = F.interpolate(memory_energy_map.view(b, 1, h_pre_interp, w_pre_interp), size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
                 feature_energy_final = F.interpolate(feature_energy_map.view(b, 1, h_pre_interp, w_pre_interp), size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
             except Exception as e:
                 logger.error(f"Output interpolation failed: {e}. Using un-interpolated outputs.", exc_info=True)
                 logits_final = logits
                 memory_energy_final = memory_energy_map.view(b, 1, h_pre_interp, w_pre_interp)
                 feature_energy_final = feature_energy_map.view(b, 1, h_pre_interp, w_pre_interp)
        else:
             logits_final = logits
             memory_energy_final = memory_energy_map.view(b, 1, h_pre_interp, w_pre_interp)
             feature_energy_final = feature_energy_map.view(b, 1, h_pre_interp, w_pre_interp)

        # --- PEBAL Energy Calculation (from logits) ---
        logits_final = self._check_and_handle_nan_inf(logits_final, "Logits Final")
        pebal_energy = torch.zeros_like(memory_energy_final) # Initialize with zeros
        try:
            num_logits_channels = logits_final.shape[1]
            # Determine number of in-distribution classes (exclude potential OOD channel)
            # Assuming OOD is the *last* channel if num_logits_channels == num_classes + 1
            in_class_channel_count = self.num_classes if num_logits_channels == self.num_classes + 1 else num_logits_channels

            if in_class_channel_count > 0 :
                 in_class_logits = logits_final[:, :in_class_channel_count, :, :]
                 # Stable logsumexp: Use float32 for stability if using mixed precision
                 with torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else contextlib.nullcontext():
                      in_class_logits_fp32 = in_class_logits.float()
                      max_logits_in_class = torch.max(in_class_logits_fp32, dim=1, keepdim=True)[0]
                      pebal_energy = -(torch.logsumexp(in_class_logits_fp32 - max_logits_in_class, dim=1, keepdim=True) + max_logits_in_class)
                 pebal_energy = self._check_and_handle_nan_inf(pebal_energy, "pebal_energy")
            else:
                 logger.warning("Cannot calculate PEBAL energy: No in-class logits available.")

        except Exception as e:
            logger.error(f"Error calculating PEBAL energy: {e}. Using zero energy.", exc_info=True)
            # pebal_energy already initialized to zeros


        # --- Combine Energies ---
        memory_energy_final = self._check_and_handle_nan_inf(memory_energy_final, "Memory Energy Final")
        feature_energy_final = self._check_and_handle_nan_inf(feature_energy_final, "Feature Energy Final")
        combined_energy = torch.zeros_like(pebal_energy) # Initialize with zeros
        try:
            # Simple averaging, weights can be adjusted
            combined_energy = pebal_energy + 0.5 * feature_energy_final + 0.5 * memory_energy_final
            combined_energy = torch.clamp(combined_energy, min=-100.0, max=100.0) # Clamp extreme values
            combined_energy = self._check_and_handle_nan_inf(combined_energy, "combined_energy")
        except Exception as e:
             logger.error(f"Error combining energy terms: {e}. Using PEBAL energy as combined.", exc_info=True)
             combined_energy = pebal_energy # Fallback

        # --- Prepare Output Dictionary ---
        output_dict['seg_logits'] = logits_final
        output_dict['memory_energy'] = memory_energy_final
        output_dict['feature_energy'] = feature_energy_final
        output_dict['pebal_energy'] = pebal_energy
        output_dict['combined_energy'] = combined_energy
        output_dict['is_ood'] = torch.zeros(b, dtype=torch.bool, device=current_device)


        # --- Final Output Shape Check ---
        ref_shape = logits_final.shape
        for key, tensor in output_dict.items():
             if key != 'is_ood' and isinstance(tensor, torch.Tensor):
                  expected_shape = (ref_shape[0], ref_shape[1] if key == 'seg_logits' else 1, ref_shape[2], ref_shape[3])
                  if tensor.shape != expected_shape:
                      logger.warning(f"Shape mismatch for output '{key}': Got {tensor.shape}, Expected {expected_shape}. Attempting resize/reshape.")
                      try:
                          tensor = tensor.view(ref_shape[0], -1, tensor.shape[-2], tensor.shape[-1]) # Reshape channel dim
                          resized_tensor = F.interpolate(tensor, size=ref_shape[-2:], mode='bilinear', align_corners=False)
                          # Average/Select channel if needed
                          if key != 'seg_logits' and resized_tensor.shape[1] != 1:
                               resized_tensor = torch.mean(resized_tensor, dim=1, keepdim=True) if resized_tensor.shape[1] > 0 else torch.zeros((ref_shape[0], 1, ref_shape[2], ref_shape[3]), device=current_device)
                          elif key == 'seg_logits' and resized_tensor.shape[1] != ref_shape[1]:
                               # Handle channel mismatch for logits (e.g., take first N channels or pad) - this indicates a bigger problem
                               logger.error(f"Logits channel mismatch after resize: {resized_tensor.shape[1]} vs {ref_shape[1]}. Cannot reliably fix.")
                               resized_tensor = resized_tensor[:, :ref_shape[1], :, :] # Crude fix: truncate/select
                          output_dict[key] = self._check_and_handle_nan_inf(resized_tensor, f"{key} (post-resize)")
                      except Exception as resize_e:
                          logger.error(f"Failed resize/reshape for '{key}': {resize_e}. Using zeros.", exc_info=True)
                          output_dict[key] = torch.zeros(expected_shape, device=current_device)
             elif key == 'is_ood' and isinstance(tensor, torch.Tensor) and tensor.shape[0] != ref_shape[0]:
                   logger.error(f"is_ood batch size mismatch! Expected {ref_shape[0]}, got {tensor.shape[0]}")

        # --- Cleanup ---
        del features, features_adapted, features_mem_proj, logits
        if 'seg_features' in locals(): del seg_features
        # Keep final outputs: logits_final, memory_energy_final, feature_energy_final, pebal_energy, combined_energy


        self.memory_tracker.log_memory_usage("Forward End")
        if self.use_efficient_memory: self.memory_tracker.clear_memory("End of Forward Pass")

        return output_dict


    def update_memory(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, max_samples: Optional[int] = None):
        """
        Updates the EfficientMemoryManager bank using pre-extracted features.
        This version expects features that are ALREADY projected to the memory dimension.

        Args:
            features (torch.Tensor): Pre-extracted features projected to memory dimension.
                                     Expected shape: [N_total, memory_feature_dim] or [B, memory_feature_dim, H, W].
                                     Should be on the model's device (self._model_device).
            labels (Optional[torch.Tensor]): Optional labels corresponding to features.
                                             Shape should match features spatial dims if provided
                                             [N_total] or [B, H, W]. Should be on model's device.
            max_samples (Optional[int]): Maximum number of *flattened* feature vectors to sample
                                         for the update. Defaults to memory manager size.
        """
        logger.debug("Starting memory update with pre-projected features...")
        if not hasattr(self, 'memory_manager'):
             logger.error("Memory Manager not initialized. Cannot update memory.")
             return

        self.eval() # Ensure model layers are in eval mode

        # --- Input Validation and Device Check ---
        if not isinstance(features, torch.Tensor) or features.shape[0] == 0:
            logger.warning("update_memory called with invalid or 0 features. Skipping.")
            return
        

        # --- Flatten Features and Labels if needed ---
        flat_features = None
        flat_labels = None
        c_mem = self.memory_manager.feature_dim

        if features.ndim == 4: # Input is [B, C, H, W]
            b, c_in, h_in, w_in = features.shape
            if c_in != c_mem:
                 logger.error(f"Feature dim mismatch: Input features ({c_in}) != Memory Manager dim ({c_mem}). Cannot update.")
                 return

            features_sampled = features
            labels_sampled = labels
            h_proc, w_proc = h_in, w_in
            needs_spatial_sampling = (self.use_efficient_memory and h_in * w_in > 4096 and self.sampling_stride > 1)
            if needs_spatial_sampling:
                stride = self.sampling_stride
                if h_in >= stride and w_in >= stride:
                    features_sampled = features[:, :, ::stride, ::stride]
                    h_proc, w_proc = features_sampled.shape[2:]
                    if labels is not None:
                        # Handle both [B, H, W] and [B, 1, H, W] label shapes
                        if labels.ndim == 3 and labels.shape == (b, h_in, w_in):
                            labels_sampled = labels[:, ::stride, ::stride]
                        elif labels.ndim == 4 and labels.shape[1] == 1 and labels.shape[-2:] == (h_in, w_in):
                             labels_sampled = labels[:, 0, ::stride, ::stride] # Squeeze channel dim
                        else:
                             logger.warning(f"Label shape {labels.shape if labels is not None else 'None'} incompatible with spatial sampling. Discarding labels.")
                             labels_sampled = None
                    logger.debug(f"MemUpdate spatially sampled features/labels from {h_in}x{w_in} to {h_proc}x{w_proc}")
                else:
                    logger.warning(f"Cannot apply spatial stride {stride} to {h_in}x{w_in} for memory update.")

            flat_features = features_sampled.permute(0, 2, 3, 1).contiguous().view(-1, c_mem)
            if labels_sampled is not None:
                 # Ensure labels_sampled has correct shape B, H_proc, W_proc before reshape
                 if labels_sampled.ndim == 3 and labels_sampled.shape == (b, h_proc, w_proc):
                      flat_labels = labels_sampled.reshape(-1)
                 else:
                      logger.warning(f"Final label shape {labels_sampled.shape if labels_sampled is not None else 'None'} mismatch sampled features B{b}, H{h_proc}, W{w_proc}. Discarding labels.")
            del features_sampled, labels_sampled

        elif features.ndim == 2: # Input is already [N, C]
             if features.shape[1] != c_mem:
                  logger.error(f"Feature dim mismatch: Input features ({features.shape[1]}) != Memory Manager dim ({c_mem}). Cannot update.")
                  return
             flat_features = features
             flat_labels = labels.view(-1) if labels is not None else None
             if flat_labels is not None and flat_labels.shape[0] != flat_features.shape[0]:
                 logger.warning("Flat features and labels count mismatch. Discarding labels.")
                 flat_labels = None
        else:
             logger.error(f"Unexpected feature shape for update_memory: {features.shape}. Expected [N, C] or [B, C, H, W].")
             return

        if flat_features is None or flat_features.shape[0] == 0:
            logger.warning("Memory update skipped: 0 features after flattening/processing.")
            return

        # --- Temporal/Feature Sampling (Random Subset) ---
        num_available = flat_features.shape[0]
        effective_max_samples = max_samples if max_samples is not None else self.memory_manager.memory_size
        sample_size = min(num_available, effective_max_samples) if effective_max_samples > 0 else num_available

        logger.debug(f"Available flat features: {num_available}, Sampling: {sample_size} (max: {effective_max_samples}) for memory manager update.")

        sampled_features = None
        sampled_labels = None
        if sample_size == 0:
             logger.warning("Memory update skipped: Sample size is 0."); return
        elif sample_size >= num_available:
            sampled_features = flat_features
            sampled_labels = flat_labels
        else:
             indices = torch.randperm(num_available, device=flat_features.device)[:sample_size]
             sampled_features = flat_features[indices]
             if flat_labels is not None:
                  sampled_labels = flat_labels[indices]

        del flat_features, flat_labels # Cleanup originals

        # --- Update Memory Manager ---
        # Ensure sampled features are on the correct device before passing
        if sampled_features.device != self.memory_manager._device:
             logger.warning(f"Moving sampled features from {sampled_features.device} to manager device {self.memory_manager._device} before update.")
             sampled_features = sampled_features.to(self.memory_manager._device)
        if sampled_labels is not None and sampled_labels.device != self.memory_manager._device:
             logger.warning(f"Moving sampled labels from {sampled_labels.device} to manager device {self.memory_manager._device} before update.")
             sampled_labels = sampled_labels.to(self.memory_manager._device)

        try:
             # Pass features already projected and on the correct device
             self.memory_manager.update_memory(sampled_features, sampled_labels)
             logger.info(f"EfficientMemoryManager update initiated with {sampled_features.shape[0]} vectors.")
        except Exception as e:
             logger.error(f"Error calling EfficientMemoryManager.update_memory: {e}", exc_info=True)
        finally:
             del sampled_features, sampled_labels
             if hasattr(self, 'memory_tracker'):
                 self.memory_tracker.log_memory_usage("End Memory Update Call")
                 if self.use_efficient_memory:
                     self.memory_tracker.clear_memory("Memory Update Cleanup")

        self.train() # Ensure model is back in train mode


# --- Example Usage (Minor contextlib import for AMP) ---
import contextlib # Added for AMP autocast example
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger(__name__)

    # --- Dummy Modules (Keep as before) ---
    class DummyBackbone(nn.Module):
        def __init__(self, out_channels=2048):
            super().__init__()
            self.out_channels = out_channels
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
            self.dummy_param = nn.Parameter(torch.randn(1)) # To detect device
        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.pool(x); x = self.final_conv(x)
            return x

    class DummySegHead(nn.Module):
         def __init__(self, in_channels, num_classes, head_channels=256):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.head_channels = head_channels
            self.feature_conv = nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(head_channels)
            self.relu = nn.ReLU(inplace=True)
            self.final_classifier = nn.Conv2d(head_channels, num_classes + 1, kernel_size=1)
         def forward(self, x):
            features = self.feature_conv(x); features = self.bn(features); features = self.relu(features)
            logits = self.final_classifier(features)
            return logits

    # --- Configuration ---
    NUM_CLASSES = 19
    BACKBONE_OUT_DIM = 512
    SEG_HEAD_INTERNAL_DIM = 128
    MEMORY_DIM = 64
    TARGET_DIM = 128
    MEM_SIZE = 200
    BATCH_SIZE = 2
    IMG_SIZE = (64, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_logger.info(f"Using device: {device}")

    # --- Instantiate Backbone ---
    backbone = DummyBackbone(out_channels=BACKBONE_OUT_DIM).to(device)

    # --- Test Case 1: After Backbone, Original Head ---
    main_logger.info("\n--- Test Case 1: After Backbone, Original Head ---")
    # Head expects TARGET_DIM as input because that's what comes out of the potential projection before it
    seg_head_orig = DummySegHead(in_channels=TARGET_DIM, num_classes=NUM_CLASSES, head_channels=SEG_HEAD_INTERNAL_DIM).to(device)
    model1 = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=seg_head_orig,
        num_classes=NUM_CLASSES,
        memory_feature_dim=MEMORY_DIM,
        memory_size=MEM_SIZE,
        insertion_point='after_backbone',
        target_feature_dim=TARGET_DIM,
        use_efficient_memory=True,
        use_faiss=FAISS_AVAILABLE,
        sampling_stride=2,
        use_efficient_decoder=False
    ).to(device) # Final move to device

    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1], device=device)
    dummy_labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1]), device=device)
    main_logger.info("\nTesting forward pass (Case 1)...")
    model1.train()
    output1 = model1(dummy_input)
    for key, tensor in output1.items(): main_logger.info(f"  Output {key}: shape={tensor.shape}, device={tensor.device}, has_nan={torch.isnan(tensor).any()}")

    # --- Simulate feature extraction for memory update (Case 1) ---
    main_logger.info("\nTesting memory update (Case 1)...")
    with torch.no_grad():
        model1.eval() # Set model to eval for feature extraction consistency
        features_raw = model1.backbone(dummy_input)
        features_adapted = model1.channel_adapter(features_raw)
        features_mem_proj = model1.memory_input_proj(features_adapted)
        # Pass the memory-projected features and labels to the model's update method
        model1.update_memory(features_mem_proj, labels=dummy_labels, max_samples=MEM_SIZE // 2)
    # --- End Simulation ---

    del model1, output1, seg_head_orig
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # --- Test Case 2: After Backbone, Efficient Decoder ---
    main_logger.info("\n--- Test Case 2: After Backbone, Efficient Decoder ---")
    seg_head_dummy_for_init = DummySegHead(in_channels=TARGET_DIM, num_classes=NUM_CLASSES).to(device) # Input dim irrelevant if replaced
    efficient_decoder_kwargs = {
        'in_channels': BACKBONE_OUT_DIM, # Eff decoder takes backbone output
        'feature_dim': MEMORY_DIM,
        'attention_heads': 4
    }
    model2 = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=seg_head_dummy_for_init, # Provided but replaced
        num_classes=NUM_CLASSES,
        memory_feature_dim=MEMORY_DIM,
        memory_size=MEM_SIZE,
        insertion_point='after_backbone',
        target_feature_dim=TARGET_DIM, # Still need target dim for memory path
        use_efficient_memory=True,
        use_faiss=FAISS_AVAILABLE,
        sampling_stride=2,
        use_efficient_decoder=True,
        efficient_decoder_kwargs=efficient_decoder_kwargs
    ).to(device) # Final move

    main_logger.info("\nTesting forward pass (Case 2)...")
    model2.train()
    output2 = model2(dummy_input)
    for key, tensor in output2.items(): main_logger.info(f"  Output {key}: shape={tensor.shape}, device={tensor.device}, has_nan={torch.isnan(tensor).any()}")

     # --- Simulate feature extraction for memory update (Case 2) ---
    main_logger.info("\nTesting memory update (Case 2)...")
    with torch.no_grad():
        model2.eval()
        features_raw = model2.backbone(dummy_input)
        features_adapted = model2.channel_adapter(features_raw) # Adapter still exists for memory path
        features_mem_proj = model2.memory_input_proj(features_adapted)
        model2.update_memory(features_mem_proj, labels=dummy_labels, max_samples=MEM_SIZE // 2)
    # --- End Simulation ---

    del model2, output2, seg_head_dummy_for_init
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()


    # --- Test Case 3: After SegHead, Original Classifier ---
    main_logger.info("\n--- Test Case 3: After SegHead, Original Classifier ---")
    # Original head takes backbone output
    seg_head_for_features = DummySegHead(in_channels=BACKBONE_OUT_DIM, num_classes=NUM_CLASSES, head_channels=SEG_HEAD_INTERNAL_DIM).to(device)
    # This head's classifier will be ignored; features before classifier are used.
    # The model adds its own final classifier in this mode.
    model3 = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=seg_head_for_features, # This head produces features
        num_classes=NUM_CLASSES,
        memory_feature_dim=MEMORY_DIM,
        memory_size=MEM_SIZE,
        insertion_point='after_seghead',
        target_feature_dim=TARGET_DIM,
        use_efficient_memory=True,
        use_faiss=FAISS_AVAILABLE,
        sampling_stride=2,
        use_efficient_decoder=False # Ensure efficient decoder is OFF
    ).to(device) # Final move

    main_logger.info("\nTesting forward pass (Case 3)...")
    model3.train()
    output3 = model3(dummy_input)
    for key, tensor in output3.items(): main_logger.info(f"  Output {key}: shape={tensor.shape}, device={tensor.device}, has_nan={torch.isnan(tensor).any()}")

    # --- Simulate feature extraction for memory update (Case 3) ---
    main_logger.info("\nTesting memory update (Case 3)...")
    with torch.no_grad():
        model3.eval()
        features_raw = model3.backbone(dummy_input)
        # Access the *original* seg head stored internally for feature extraction
        seg_features = model3._original_segmentation_head(features_raw)
        # Note: The DummySegHead returns logits. If we intended to use features *before* its classifier,
        # the DummySegHead or the model logic would need adjustment.
        # Assuming here we adapt the output of the provided seg_head.
        features_adapted = model3.channel_adapter(seg_features)
        features_mem_proj = model3.memory_input_proj(features_adapted)
        model3.update_memory(features_mem_proj, labels=dummy_labels, max_samples=MEM_SIZE // 2)
    # --- End Simulation ---

    del model3, output3, seg_head_for_features
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()


    main_logger.info("\nExample Usage Complete.")
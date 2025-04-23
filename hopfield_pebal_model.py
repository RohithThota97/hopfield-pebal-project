# hopfield_pebal_model.py (Modified V2)
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
import contextlib # Added for AMP autocast

# Attempt to import faiss, but make it optional
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    print("WARNING: faiss library not found. FAISS acceleration disabled.") # Use print for early warning

# Configure logging (Assume setup in main script)
logger = logging.getLogger(__name__) # Get logger instance

# --- Memory Tracker (Unchanged from previous version) ---
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
    MODIFIED V2: Implements FAISS PQ indexing.
    """
    def __init__(self, feature_dim=256, memory_size=1024, pq_bytes=8,
                 sampling_ratio=0.25, num_classes=19, use_faiss=True,
                 device='cpu', faiss_nprobe=16): # Added faiss_nprobe
        super(EfficientMemoryManager, self).__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.sampling_ratio = sampling_ratio # Currently unused in update logic
        self.num_classes = num_classes
        self.pq_bytes = pq_bytes # Corresponds to 'm' (num subquantizers) in FAISS PQ
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self._device = torch.device(device) # Ensure device is a torch.device
        self.memory_tracker = MemoryTracker(log_interval=15, verbose=True)
        self.faiss_nprobe = faiss_nprobe # Number of cells/lists to probe for IVFPQ (currently using IndexPQ, nprobe not needed)

        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim, device=self._device))
        self.register_buffer('memory_labels', torch.full((memory_size,), -1, dtype=torch.long, device=self._device))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long, device=self._device))
        self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.long, device=self._device))

        self.memory_initialized = False
        self.faiss_index = None
        self.faiss_res = None
        self._init_faiss_resources()
        self.index_type = None # Track the type of index created


    def _init_faiss_resources(self):
        """Initialize FAISS GPU/CPU resources."""
        if not self.use_faiss: return
        try:
            if faiss.get_num_gpus() > 0 and 'cuda' in str(self._device):
                gpu_id = self._device.index if self._device.index is not None else 0
                self.faiss_res = faiss.StandardGpuResources()
                logger.info(f"FAISS: Initialized StandardGpuResources on device {gpu_id}.")
            else:
                logger.info("FAISS: Using CPU resources (No GPU detected by FAISS or CPU device specified).")
                # Disable GPU indexing if resources failed or CPU requested
                self.faiss_res = None
                # Do not disable FAISS entirely, just use CPU index
        except Exception as e:
            logger.warning(f"FAISS: Failed to initialize GPU resources ({e}). Falling back to CPU indexing.", exc_info=True)
            self.use_faiss = True # Still try CPU FAISS
            self.faiss_res = None

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Safely convert tensor to numpy array."""
        return tensor.detach().cpu().numpy().astype(np.float32)

    # --- Sampling Methods (Unchanged) ---
    def reservoir_sampling(self, features: torch.Tensor, k: int) -> torch.Tensor:
        n = features.shape[0]
        if n == 0: return features
        k = min(n, k)
        if k == 0: return features.new_empty((0, features.shape[1]))
        reservoir = features[:k].clone()
        for i in range(k, n):
            j = torch.randint(0, i + 1, (1,), device=features.device).item()
            if j < k:
                reservoir[j] = features[i]
        return reservoir

    def class_balanced_sampling(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None or features.shape[0] != labels.shape[0]:
            logger.warning("Class balanced sampling requires valid labels matching features count. Falling back to reservoir.")
            return self.reservoir_sampling(features, self.memory_size)
        labels = labels.view(-1).to(self._device)
        if features.device != self._device: features = features.to(self._device)
        unique_classes, counts = torch.unique(labels[labels >= 0], return_counts=True)
        num_valid_classes = len(unique_classes)
        if num_valid_classes == 0:
             logger.warning("Class balanced sampling: No valid classes found in labels. Using reservoir sampling.")
             return self.reservoir_sampling(features, self.memory_size)
        samples_per_class = max(1, self.memory_size // num_valid_classes)
        balanced_features_list = []
        for cls_idx in unique_classes:
            cls_mask = (labels == cls_idx)
            cls_features = features[cls_mask]
            num_cls_features = cls_features.shape[0]
            if num_cls_features == 0: continue
            k_cls = min(num_cls_features, samples_per_class)
            sampled_cls_features = self.reservoir_sampling(cls_features, k_cls)
            balanced_features_list.append(sampled_cls_features)
        if not balanced_features_list:
            logger.warning("Class balanced sampling resulted in zero features. Using reservoir sampling on original.")
            return self.reservoir_sampling(features, self.memory_size)
        balanced_features = torch.cat(balanced_features_list, dim=0)
        num_balanced_samples = balanced_features.shape[0]
        if num_balanced_samples < self.memory_size:
            remaining_needed = self.memory_size - num_balanced_samples
            additional_features = self.reservoir_sampling(features, remaining_needed)
            if additional_features.shape[0] > 0:
                balanced_features = torch.cat([balanced_features, additional_features], dim=0)
        if balanced_features.shape[0] > self.memory_size:
            balanced_features = balanced_features[:self.memory_size]
        return balanced_features

    def kmeans_sampling(self, features: torch.Tensor, k: int) -> torch.Tensor:
        n_features = features.shape[0]
        if not self.use_faiss or n_features < k or not FAISS_AVAILABLE:
            logger.debug(f"KMeans sampling requirements not met (FAISS:{self.use_faiss}, N:{n_features}<k:{k}). Falling back to reservoir.")
            return self.reservoir_sampling(features, k)
        if features.device != self._device: features = features.to(self._device)
        features_np = self._to_numpy(features)
        faiss.normalize_L2(features_np)
        use_gpu_kmeans = (self.faiss_res is not None)
        logger.info(f"Performing K-means clustering (k={k}) on {n_features} features (GPU: {use_gpu_kmeans})...")
        kmeans = faiss.Kmeans(d=self.feature_dim, k=k, niter=20, verbose=False, gpu=use_gpu_kmeans)
        try:
            kmeans.train(features_np)
            centroids = torch.from_numpy(kmeans.centroids).to(self._device)
            logger.info("K-means clustering complete.")
            self.memory_tracker.log_memory_usage("After K-means")
            return centroids
        except Exception as e:
            logger.error(f"K-means failed: {e}. Falling back to reservoir sampling.", exc_info=True)
            return self.reservoir_sampling(features, k)
        finally:
             del features_np; gc.collect()

    # --- FAISS Indexing and Querying (MODIFIED for PQ) ---
    def create_faiss_index(self, features: Optional[torch.Tensor]=None):
        """Builds or rebuilds the FAISS index with PQ."""
        if not self.use_faiss or not FAISS_AVAILABLE:
            self.faiss_index = None
            return

        if features is None:
            if not self.memory_initialized:
                 logger.debug("FAISS: Memory bank not initialized, skipping index creation.")
                 self.faiss_index = None
                 return
            features_to_index = self.memory_bank
            logger.debug(f"FAISS: Rebuilding index using current memory bank (size {features_to_index.shape[0]}).")
        else:
            features_to_index = features

        # Convert features to CPU numpy for index building/training
        if features_to_index.device != torch.device('cpu'):
             logger.debug(f"FAISS: Moving features from {features_to_index.device} to CPU for index creation.")
             features_to_index_cpu = features_to_index.cpu()
        else:
             features_to_index_cpu = features_to_index

        if features_to_index_cpu.shape[0] == 0:
             logger.warning("FAISS: Attempted to build index with 0 features. Skipping.")
             self.faiss_index = None
             return

        features_np = self._to_numpy(features_to_index_cpu)
        del features_to_index_cpu # Cleanup CPU copy if created
        faiss.normalize_L2(features_np) # Normalize data for index

        # Warn if dataset is too small for PQ training
        min_pq_train_size = 256 * self.pq_bytes # Rough heuristic (e.g., 256 samples per subquantizer)
        if features_np.shape[0] < min_pq_train_size:
            logger.warning(f"FAISS: Dataset size ({features_np.shape[0]}) is potentially small for robust PQ training "
                           f"(m={self.pq_bytes}, recommend >= {min_pq_train_size}). Quantization might be suboptimal.")
            # Optional: Fallback to IndexFlatL2 if too small?
            # self.use_faiss = False # Or disable PQ specifically
            # self.create_faiss_index_flatl2(features_np) # Call a different method maybe
            # return # Exit this PQ creation if falling back

        logger.info(f"Creating FAISS PQ index for {features_np.shape[0]} features (dim={self.feature_dim}, PQ bytes (m)={self.pq_bytes})...")
        n_subquantizers = self.pq_bytes # Use pq_bytes for m (number of subquantizers)
        n_bits = 8 # Standard: 8 bits per subquantizer code

        # --- PQ Index Creation Logic ---
        try:
            if self.faiss_res: # Use GPU index
                gpu_id = self._device.index if self._device.index is not None else 0

                # --- Use GpuIndexPQ ---
                co = faiss.GpuPQIndexConfig() # Configuration object
                co.use_precomputed_tables = True # Common optimization
                # For GpuIndexPQ, typically uses METRIC_L2 implicitly for Euclidean
                gpu_index = faiss.GpuIndexPQ(self.faiss_res, self.feature_dim,
                                            n_subquantizers, n_bits,
                                            faiss.METRIC_L2, # Specify L2 distance
                                            config=co)

                logger.info(f"FAISS: Training GpuIndexPQ (GPU {gpu_id})...")
                gpu_index.train(features_np) # Train on numpy data using GPU resources
                logger.info("FAISS: GpuIndexPQ training complete. Adding vectors...")
                gpu_index.add(features_np) # Add numpy data to GPU index

                self.faiss_index = gpu_index
                self.index_type = 'GpuIndexPQ'
                logger.info(f"FAISS: Created {self.index_type} on device {self._device}.")
            else: # Use CPU index
                # --- Use IndexPQ ---
                cpu_index = faiss.IndexPQ(self.feature_dim, n_subquantizers, n_bits, faiss.METRIC_L2)
                logger.info(f"FAISS: Training IndexPQ (CPU)...")
                cpu_index.train(features_np) # Train on numpy data
                logger.info("FAISS: IndexPQ training complete. Adding vectors...")
                cpu_index.add(features_np) # Add numpy data to CPU index

                self.faiss_index = cpu_index
                self.index_type = 'IndexPQ'
                logger.info(f"FAISS: Created {self.index_type} (CPU).")

            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors.")
            # Note: For IndexIVFPQ (if used later), would need to set nprobe:
            # if hasattr(self.faiss_index, 'nprobe'): self.faiss_index.nprobe = self.faiss_nprobe

        except AttributeError as ae:
             # Catch common error if faiss gpu extensions are not compiled/available
            if "GpuIndexPQ" in str(ae) or "StandardGpuResources" in str(ae):
                 logger.error(f"FAISS: GPU extensions likely missing ('{ae}'). Disabling GPU FAISS & trying CPU PQ.")
                 self.faiss_res = None # Disable GPU res
                 self.use_faiss = True # Ensure we still try CPU
                 self.create_faiss_index(features) # Retry recursively (will now take CPU path)
            else:
                 logger.error(f"FAISS: Failed to create index ({ae}). Disabling FAISS.", exc_info=True)
                 self.use_faiss = False; self.faiss_index = None; self.faiss_res = None
        except Exception as e:
             logger.error(f"FAISS: Failed to create index ({e}). Disabling FAISS.", exc_info=True)
             self.use_faiss = False; self.faiss_index = None; self.faiss_res = None
        finally:
            del features_np
            self.memory_tracker.clear_memory("FAISS Index Creation")
            self.memory_tracker.log_memory_usage("After FAISS index creation")

    def query_faiss(self, query_features: torch.Tensor, k: int = 5) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Queries the FAISS index."""
        if not self.use_faiss or self.faiss_index is None or not FAISS_AVAILABLE:
            return None, None
        if not hasattr(self.faiss_index, 'ntotal') or self.faiss_index.ntotal == 0:
            logger.warning("FAISS query: Index is empty or not valid.")
            return None, None
        if query_features.shape[0] == 0:
            return None, None

        k = min(k, self.faiss_index.ntotal) # Ensure k is not larger than index size
        if k <= 0: return None, None

        original_query_device = query_features.device
        if query_features.device != torch.device('cpu'):
            query_features_cpu = query_features.cpu()
        else:
            query_features_cpu = query_features

        query_np = self._to_numpy(query_features_cpu)
        del query_features_cpu # Cleanup CPU copy if made
        faiss.normalize_L2(query_np) # Normalize queries to match indexed data

        try:
            # Note: If using IVFPQ index later, remember to set nprobe before search:
            # if hasattr(self.faiss_index, 'nprobe'): self.faiss_index.nprobe = self.faiss_nprobe
            distances, indices = self.faiss_index.search(query_np, k)
            # PQ returns L2 distance. Ensure non-negative.
            distances = np.maximum(distances, 0.0)
            return torch.from_numpy(distances).to(original_query_device), \
                   torch.from_numpy(indices).to(original_query_device)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            return None, None
        finally:
            del query_np # Cleanup numpy array

    # --- Memory Update (Unchanged - sampling logic uses updated Manager methods) ---
    def update_memory(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """ Updates the memory bank (expects features already projected) """
        self.memory_tracker.log_memory_usage("Start Memory Update")
        with torch.no_grad():
            if features.shape[0] == 0: return
            if features.shape[1] != self.feature_dim:
                 logger.error(f"MemMan update: Feat dim mismatch ({features.shape[1]} vs {self.feature_dim}).")
                 return
            if features.device != self._device:
                 features = features.to(self._device)
            if labels is not None and labels.device != self._device:
                 labels = labels.to(self._device)
            if labels is not None and features.shape[0] != labels.shape[0]:
                 labels = None

            features = F.normalize(features, p=2, dim=1)

            # Use existing sampling methods (these could be enhanced further)
            if labels is not None:
                sampled_features = self.class_balanced_sampling(features, labels)
            else:
                 if self.use_faiss and features.shape[0] > self.memory_size * 2: # Heuristic
                      num_to_sample = min(features.shape[0], self.memory_size)
                      sampled_features = self.kmeans_sampling(features, num_to_sample)
                 else:
                      num_to_sample = min(features.shape[0], self.memory_size)
                      sampled_features = self.reservoir_sampling(features, num_to_sample)

            num_features_to_add = sampled_features.shape[0]
            if num_features_to_add == 0: return

            logger.debug(f"MemMan: Adding {num_features_to_add} sampled features to bank.")

            ptr = self.memory_ptr[0].item()
            indices = torch.arange(ptr, ptr + num_features_to_add, device=self._device) % self.memory_size

            if len(indices) < num_features_to_add:
                 num_features_to_add = len(indices)
                 sampled_features = sampled_features[:num_features_to_add]

            if sampled_features.device != self.memory_bank.device:
                 sampled_features = sampled_features.to(self.memory_bank.device)

            self.memory_bank[indices] = sampled_features
            # NOTE: If storing labels in memory, update them here using sampled_labels (if preserved)

            new_ptr = (ptr + num_features_to_add) % self.memory_size
            self.memory_ptr[0] = new_ptr

            if not self.memory_initialized:
                if new_ptr > 0 or num_features_to_add >= self.memory_size:
                     self.memory_initialized = True; logger.info("MemoryManager initialized.")

            # Rebuild FAISS Index (Now with PQ)
            if self.use_faiss:
                # Only index the valid part of the bank if not full? For now, index all.
                # This rebuilds the PQ index including training. Might be slow.
                # Consider rebuilding less frequently or using techniques that allow adding to PQ indices.
                self.create_faiss_index(self.memory_bank)

        self.memory_tracker.log_memory_usage("After memory update")
        self.memory_tracker.clear_memory("Memory Update Cleanup")


# --- Efficient Segmentation Decoder (Unchanged from previous version) ---
class EfficientSegmentationDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, feature_dim: int = 128, attention_heads: int = 8):
        super(EfficientSegmentationDecoder, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.attention_heads = attention_heads
        if feature_dim % attention_heads != 0:
            logger.warning(f"EffDec: feature_dim {feature_dim} not divisible by heads {attention_heads}. Adjusting.")
            self.feature_dim = (feature_dim // attention_heads) * attention_heads
            logger.warning(f"Adjusted feature_dim to {self.feature_dim}")

        self.feature_projector = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1)
        self.head_dim = self.feature_dim // self.attention_heads
        self.qkv_conv = nn.ModuleDict({
            'query': nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False),
            'key': nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False),
            'value': nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
        })
        self.out_proj = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        self.classifier = nn.Conv2d(self.feature_dim, num_classes + 1, kernel_size=1)
        self.memory_tracker = MemoryTracker(log_interval=10, verbose=True)
        self.attn_max_tokens = 64 * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        self.memory_tracker.log_memory_usage("EfficientDecoder start")
        features = self.feature_projector(x)
        num_input_tokens = h * w
        downscale_factor = 1
        attn_features = features
        if num_input_tokens > self.attn_max_tokens:
            downscale_factor = math.ceil(math.sqrt(num_input_tokens / self.attn_max_tokens))
            downscale_factor = min(downscale_factor, h, w)
            if downscale_factor > 1:
                logger.debug(f"EffDec: Input {h}x{w} > max {self.attn_max_tokens}. Downscaling attn by {downscale_factor}x.")
                attn_features = F.avg_pool2d(features, kernel_size=downscale_factor, stride=downscale_factor)
            else: downscale_factor = 1
        ah, aw = attn_features.size(2), attn_features.size(3)
        num_attn_tokens = ah * aw
        queries = self.qkv_conv['query'](attn_features)
        keys = self.qkv_conv['key'](attn_features)
        values = self.qkv_conv['value'](attn_features)
        queries = queries.view(batch_size, self.attention_heads, self.head_dim, num_attn_tokens).permute(0, 1, 3, 2)
        keys = keys.view(batch_size, self.attention_heads, self.head_dim, num_attn_tokens)
        values = values.view(batch_size, self.attention_heads, self.head_dim, num_attn_tokens).permute(0, 1, 3, 2)

        use_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        if use_flash_attn:
             try:
                attention_output = F.scaled_dot_product_attention(
                    queries, keys.permute(0, 1, 3, 2), values,
                    attn_mask=None, dropout_p=0.0, is_causal=False
                )
             except Exception as flash_e:
                 logger.warning(f"F.scaled_dot_product_attention failed: {flash_e}. Falling back.")
                 use_flash_attn = False
        if not use_flash_attn:
             attention_scores = torch.matmul(queries, keys) / math.sqrt(self.head_dim)
             attention_weights = F.softmax(attention_scores, dim=-1)
             attention_output = torch.matmul(attention_weights, values)
             del attention_scores, attention_weights

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, num_attn_tokens, self.feature_dim)
        attention_output = attention_output.view(batch_size, ah, aw, self.feature_dim).permute(0, 3, 1, 2)
        attention_output = self.out_proj(attention_output)

        if downscale_factor > 1:
            attention_output = F.interpolate(attention_output, size=(h, w), mode='bilinear', align_corners=False)
            final_features = attention_output + features
        else:
            final_features = attention_output + attn_features
        output = self.classifier(final_features)
        self.memory_tracker.log_memory_usage("EfficientDecoder end")
        return output


# --- Hopfield PEBAL Model ---
class HopfieldPEBALModel(nn.Module):
    """
    MODIFIED V2: Adds lambda_feature, lambda_memory for energy weighting.
                 Adds comments for advanced spatial sampling placeholders.
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
                 pq_bytes: int = 8, # Passed to EfficientMemoryManager
                 faiss_nprobe: int = 16, # Passed to EfficientMemoryManager (for IVFPQ if used later)
                 sampling_stride: int = 2,
                 memory_log_interval: int = 10,
                 memory_log_verbose: bool = True,
                 use_efficient_decoder: bool = False,
                 efficient_decoder_kwargs: Optional[Dict] = None,
                 memory_beta: float = 8.0,
                 lambda_feature: float = 0.5, # NEW: Weight for feature energy
                 lambda_memory: float = 0.5   # NEW: Weight for memory (Hopfield) energy
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
        self.lambda_feature = lambda_feature # Store feature energy weight
        self.lambda_memory = lambda_memory   # Store memory energy weight

        logger.info(f"Initialized with Energy Weights: lambda_feature={self.lambda_feature}, lambda_memory={self.lambda_memory}")

        self.memory_tracker = MemoryTracker(log_interval=memory_log_interval, verbose=memory_log_verbose)
        self._model_device = self._get_module_device(backbone)
        if self._model_device is None: self._model_device = self._get_module_device(segmentation_head)
        if self._model_device is None:
             self._model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
             logger.warning(f"Could not detect device. Defaulting to {self._model_device}.")
        else: logger.info(f"Determined model device: {self._model_device}")


        # --- Instantiate Efficient Decoder (if requested) ---
        self.segmentation_head = self._original_segmentation_head
        self.use_efficient_decoder = use_efficient_decoder
        self._efficient_decoder_instance = None
        # (Instantiation logic for Efficient Decoder - Unchanged)
        if self.use_efficient_decoder:
            backbone_dim = self._detect_feature_dimensions(after_seghead=False, device=self._model_device)
            if backbone_dim is None: raise RuntimeError("Cannot use Efficient Decoder: Failed backbone dim detect.")
            eff_decoder_defaults = {'in_channels': backbone_dim,'num_classes': self.num_classes,
                                    'feature_dim': memory_feature_dim,'attention_heads': 4}
            if efficient_decoder_kwargs: eff_decoder_defaults.update(efficient_decoder_kwargs)
            try:
                adapter_out_dim = self._detect_feature_dimensions(after_seghead=True, device=self._model_device)
                adapter_out_dim = target_feature_dim if target_feature_dim is not None else adapter_out_dim

                if insertion_point == 'after_seghead' and adapter_out_dim != eff_decoder_defaults['in_channels']:
                     logger.info(f"Re-configuring EffDec input for 'after_seghead': {adapter_out_dim}")
                     eff_decoder_defaults['in_channels'] = adapter_out_dim

                self._efficient_decoder_instance = EfficientSegmentationDecoder(**eff_decoder_defaults).to(self._model_device)

                if insertion_point == 'after_backbone':
                    logger.info("Using EfficientSegmentationDecoder for logits (insertion='after_backbone').")
                    self.segmentation_head = self._efficient_decoder_instance # Replace head reference
                # Else ('after_seghead'), keep original head, eff decoder used later

            except Exception as e:
                logger.error(f"Failed EfficientSegmentationDecoder: {e}. Fallback.", exc_info=True)
                self.use_efficient_decoder = False; self._efficient_decoder_instance = None
                self.segmentation_head = self._original_segmentation_head # Ensure fallback


        # --- Dimension Detection ---
        self._input_dim_after_feature_extractor = self._detect_feature_dimensions(
            after_seghead=(insertion_point == 'after_seghead'),
            use_original_head=(insertion_point == 'after_seghead'),
            device=self._model_device
        )
        if self._input_dim_after_feature_extractor is None:
            raise RuntimeError("Failed to detect feature dimensions after extractor.")
        logger.info(f"Detected feature dim ('{insertion_point}' stage): {self._input_dim_after_feature_extractor}")

        # --- Determine Target Dimension and Need for Adapters ---
        dim_before_modules = self._input_dim_after_feature_extractor
        if target_feature_dim is None:
            self._target_feature_dim = 512 if dim_before_modules > 512 else dim_before_modules
            logger.warning(f"target_feature_dim default: {self._target_feature_dim}.")
        else: self._target_feature_dim = target_feature_dim
        logger.info(f"Adapter Dim: {dim_before_modules} -> {self._target_feature_dim}")
        logger.info(f"Memory Interaction Dim: {memory_feature_dim}")


        # --- Adapters and Projections ---
        self.needs_adapter = (dim_before_modules != self._target_feature_dim)
        self.channel_adapter = nn.Identity()
        if self.needs_adapter:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(dim_before_modules, self._target_feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self._target_feature_dim),
                nn.ReLU(inplace=True)
            ).to(self._model_device)
        self.memory_input_proj = nn.Conv2d(self._target_feature_dim, memory_feature_dim, kernel_size=1).to(self._model_device)

        # --- Efficient Memory Manager (Now with PQ enabled) ---
        self.memory_manager = EfficientMemoryManager(
            feature_dim=memory_feature_dim,
            memory_size=memory_size,
            pq_bytes=pq_bytes,             # Pass PQ parameter
            faiss_nprobe=faiss_nprobe,     # Pass nprobe parameter
            num_classes=num_classes,
            use_faiss=use_faiss,
            device=self._model_device      # Pass device
        )
        logger.info(f"Initialized EfficientMemoryManager (FAISS: {self.memory_manager.use_faiss} w/ PQ, "
                    f"Size: {memory_size}, Dim: {memory_feature_dim}) on device {self._model_device}")


        # --- Output Projections and Final Layers ---
        self.final_seghead_proj = nn.Identity()
        self.final_classifier = None # Reset

        if self.insertion_point == 'after_backbone':
            if not self.use_efficient_decoder: # Use original head path
                self._check_and_prepare_seghead_projection() # Prepare projection if needed
                logger.info("Logits from original Head (post potential projection).")
            else: # Use efficient decoder
                 logger.info("Logits from Efficient Decoder (input: backbone).")
        else: # insertion_point == 'after_seghead'
            if self.use_efficient_decoder: # Eff decoder runs on adapted seg_head features
                logger.info("Logits from Efficient Decoder (input: adapted seg_head feat).")
            else: # Need a final classifier after memory proj
                 logger.info(f"Adding Final Classifier: {memory_feature_dim} -> {self.num_classes + 1}")
                 self.final_classifier = nn.Conv2d(memory_feature_dim, self.num_classes + 1, kernel_size=1).to(self._model_device)


        # --- Energy Head (Unchanged) ---
        self.energy_head = nn.Sequential(
            nn.Conv2d(memory_feature_dim, memory_feature_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(memory_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(memory_feature_dim // 2, 1, kernel_size=1)
        ).to(self._model_device)

        self._initialize_weights()
        self.to(self._model_device)
        logger.info(f"HopfieldPEBALModel initialized on device: {self._model_device}")


    def _get_module_device(self, module: nn.Module) -> Optional[torch.device]:
        # (Helper function unchanged)
        if not isinstance(module, nn.Module): return None
        try:
            params = list(module.parameters())
            if params: return params[0].device
            buffers = list(module.buffers())
            if buffers: return buffers[0].device
            for child in module.children():
                device = self._get_module_device(child)
                if device: return device
            return None
        except Exception as e: return None

    def _detect_feature_dimensions(self, after_seghead: bool = False, use_original_head: bool = False, device: Optional[Union[str, torch.device]] = None) -> Optional[int]:
         # (Helper function unchanged)
        target_device = device if device else self._model_device
        if isinstance(target_device, str): target_device = torch.device(target_device)
        original_backbone_device = self._get_module_device(self.backbone)
        _backbone = self.backbone.to(target_device)

        if after_seghead and use_original_head: head_to_check = self._original_segmentation_head
        elif after_seghead and not use_original_head: head_to_check = self.segmentation_head # Might be eff decoder
        else: head_to_check = self.segmentation_head # Original or EffDec

        original_seghead_device = self._get_module_device(head_to_check) if head_to_check else None
        _seg_head = head_to_check.to(target_device) if head_to_check else None

        _backbone.eval();
        if _seg_head: _seg_head.eval()
        detected_dim = None
        try:
            dummy_input = torch.zeros(1, 3, 64, 64, device=target_device)
            with torch.no_grad():
                features = _backbone(dummy_input);
                if isinstance(features, (tuple, list)): features = features[-1]
                if after_seghead:
                    if _seg_head is None: raise ValueError("Seg head is None")
                    first_conv_or_proj = None; target_in_channels = None
                    for module in _seg_head.modules(): # Find first conv/linear layer's input
                         if isinstance(module, (nn.Conv2d, nn.Linear)):
                             target_in_channels = getattr(module, 'in_channels', getattr(module, 'in_features', None))
                             if target_in_channels: break
                         elif isinstance(module, EfficientSegmentationDecoder):
                              target_in_channels = getattr(module.feature_projector, 'in_channels', None)
                              if target_in_channels: break
                    temp_proj = None
                    if target_in_channels is not None and target_in_channels != features.shape[1]:
                         temp_proj = nn.Conv2d(features.shape[1], target_in_channels, 1).to(target_device)
                         features = temp_proj(features)
                    seg_features = _seg_head(features);
                    if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                    detected_dim = seg_features.shape[1]
                    if temp_proj is not None: del temp_proj
                else: detected_dim = features.shape[1]
        except Exception as e: logger.error(f"Dim detection failed: {e}", exc_info=True); detected_dim = None
        finally: # Move modules back
            if original_backbone_device:
                self.backbone.to(original_backbone_device)
            else:
                self.backbone.cpu()
            if original_seghead_device and head_to_check: head_to_check.to(original_seghead_device)
            elif head_to_check: head_to_check.cpu()
            del _backbone;
            if '_seg_head' in locals() and _seg_head is not None: del _seg_head
            gc.collect();
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return detected_dim

    def _check_and_prepare_seghead_projection(self):
        # (Helper function unchanged)
        if self.insertion_point != 'after_backbone' or self.use_efficient_decoder or self._original_segmentation_head is None:
            self.final_seghead_proj = nn.Identity(); return
        try:
            expected_input_dim = None
            for module in self._original_segmentation_head.modules(): # Find first layer
                if module is self._original_segmentation_head or isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)): continue
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    expected_input_dim = getattr(module, 'in_channels', getattr(module, 'in_features', None))
                    if expected_input_dim is not None: break
            if expected_input_dim:
                 self._original_seghead_in_channels = expected_input_dim
                 dim_after_adapter = self._target_feature_dim
                 if self._original_seghead_in_channels != dim_after_adapter:
                     logger.warning(f"Adding final projection: {dim_after_adapter} -> {self._original_seghead_in_channels}")
                     self.final_seghead_proj = nn.Conv2d(dim_after_adapter, self._original_seghead_in_channels, kernel_size=1).to(self._model_device)
                 else: self.final_seghead_proj = nn.Identity()
            else: self.final_seghead_proj = nn.Identity()
        except Exception as e: logger.error(f"Error checking seg head projection: {e}", exc_info=True); self.final_seghead_proj = nn.Identity()


    def _initialize_weights(self):
         # (Helper function largely unchanged - ensures new layers initialized)
        logger.info("Initializing weights for newly added layers...")
        init_count = 0; initialized_modules = set()
        components_to_init = [self.channel_adapter, self.memory_input_proj,
            self.final_seghead_proj, self.final_classifier, self.energy_head,
            self._efficient_decoder_instance ]
        for component in components_to_init:
            if component is None or isinstance(component, nn.Identity): continue
            for m in component.modules():
                 if m in initialized_modules or m == component or isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Identity)): continue
                 try:
                     if isinstance(m, (nn.Conv2d, nn.Linear)):
                         if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                             if m.weight.ndim > 1:
                                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                                 init_count += 1; initialized_modules.add(m)
                         if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad: nn.init.constant_(m.bias, 0)
                     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                         if getattr(m, 'affine', True):
                             if hasattr(m, 'weight') and m.weight is not None: nn.init.constant_(m.weight, 1)
                             if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)
                             init_count += 1; initialized_modules.add(m)
                 except Exception as init_e: logger.warning(f"Could not init {m}: {init_e}", exc_info=False)
        logger.info(f"Weight initialization complete ({init_count} modules).")

    def _apply_memory_interaction(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to interact with memory manager and calculate memory energy.
        MODIFIED V2: Added comments/placeholders for advanced spatial sampling.
        """
        b, c_mem, h_in, w_in = features.shape
        current_device = features.device
        self.memory_tracker.log_memory_usage("Start Memory Interaction")

        dummy_energy = torch.zeros(b, 1, h_in, w_in, device=current_device)
        dummy_flat = torch.zeros(0, c_mem, device=current_device)

        h_proc, w_proc = h_in, w_in
        features_sampled = features
        needs_upsampling = False
        # Check if spatial sampling should be applied before flattening
        apply_spatial_sampling = (self.use_efficient_memory and h_in * w_in > 4096 and self.sampling_stride > 1)

        if apply_spatial_sampling:
            stride = self.sampling_stride
            if h_in >= stride and w_in >= stride:
                # --- Simple Strided Spatial Sampling (Current) ---
                # features_sampled = features[:, :, ::stride, ::stride] # Original Line
                # --- Placeholder for Advanced Spatial Sampling ---
                # TODO: Implement class-balanced or anomaly-score-weighted spatial sampling here.
                # This would replace the simple strided sampling below.
                # It requires access to labels (during update) or an estimated anomaly map (general forward pass),
                # and logic to select pixel locations non-uniformly (e.g., weighted random sampling of indices,
                # or denser sampling in high-anomaly regions) *before* gathering features.
                # E.g., calculate anomaly map -> threshold/probabilities -> sample indices -> gather features[sampled_indices]
                # For now, falling back to simple strided sampling:
                logger.debug("Applying simple strided spatial sampling for memory interaction.")
                features_sampled = features[:, :, ::stride, ::stride]
                # -----------------------------------------------

                h_proc, w_proc = features_sampled.shape[2:]
                needs_upsampling = True
                logger.debug(f"Memory interaction spatially sampled input from {h_in}x{w_in} to {h_proc}x{w_proc} (Stride: {stride})")
            else:
                logger.warning(f"Cannot apply stride {stride} to features {h_in}x{w_in} for memory.")

        if h_proc <= 0 or w_proc <= 0:
             logger.error("Zero spatial dimension after sampling. Skipping mem interaction.")
             return dummy_energy, dummy_flat

        try:
            flat_features = features_sampled.permute(0, 2, 3, 1).contiguous().view(-1, c_mem)
        except Exception as e:
             logger.error(f"Error reshaping features for memory interaction: {e}", exc_info=True)
             return dummy_energy, dummy_flat

        if flat_features.shape[0] == 0:
             return dummy_energy, dummy_flat

        flat_features_normalized = F.normalize(flat_features, p=2, dim=1)
        memory_energy_flat = torch.zeros(flat_features_normalized.shape[0], device=current_device)

        # Memory Query Logic (Now potentially uses PQ index)
        if self.memory_manager.memory_initialized:
            if self.memory_manager.use_faiss and self.memory_manager.faiss_index is not None:
                k_neighbors = 1
                distances, indices = self.memory_manager.query_faiss(flat_features_normalized, k=k_neighbors)
                if distances is not None and indices is not None:
                    # FAISS PQ returns L2 distance, use directly
                    memory_energy_flat = distances[:, 0] * self.memory_beta # Shape [N]
                    self.memory_tracker.log_memory_usage("After FAISS Query")
                else: logger.warning("FAISS query returned None. Using zero memory energy.")
            else: # Manual Path (Dot Product with Memory Bank)
                 logger.debug("Calculating memory energy manually (dot product).")
                 try:
                    mem_bank_current = self.memory_manager.memory_bank
                    if mem_bank_current.device != flat_features_normalized.device:
                        mem_bank_current = mem_bank_current.to(flat_features_normalized.device)
                    if mem_bank_current.abs().sum() > 1e-6: # Check bank has values
                        similarities = torch.matmul(flat_features_normalized, mem_bank_current.t())
                        max_similarity, _ = torch.max(similarities, dim=1)
                        memory_energy_flat = (1.0 - max_similarity.clamp(max=1.0)) * self.memory_beta
                        self.memory_tracker.log_memory_usage("After Manual Memory Query")
                        del similarities, max_similarity
                    else: logger.debug("Manual query skipped: Memory bank empty/zero.")
                 except Exception as e:
                    logger.error(f"Manual memory energy calc failed: {e}. Zero energy.", exc_info=True)
        else: logger.debug("Memory not initialized. Using zero memory energy.")

        try:
            memory_energy_map = memory_energy_flat.view(b, h_proc, w_proc, 1).permute(0, 3, 1, 2)
            if needs_upsampling:
                memory_energy_map = F.interpolate(memory_energy_map, size=(h_in, w_in), mode='bilinear', align_corners=False)
        except Exception as e:
             logger.error(f"Error reshaping/upsampling memory energy map: {e}. Dummy energy.", exc_info=True)
             return dummy_energy, flat_features_normalized

        self.memory_tracker.log_memory_usage("End Memory Interaction")
        return memory_energy_map, flat_features_normalized


    def _check_and_handle_nan_inf(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
         # (Helper function unchanged)
        if not isinstance(tensor, torch.Tensor): return tensor
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if has_nan or has_inf:
            nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
            inf_count = torch.isinf(tensor).sum().item() if has_inf else 0
            logger.warning(f"NaN/Inf detected in '{name}'. Replacing w/ zeros. (NaNs: {nan_count}, Infs: {inf_count}, Shape: {tensor.shape}, Device: {tensor.device})")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    # --- Forward Pass (MODIFIED V2: Apply energy weights) ---
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.memory_tracker.log_memory_usage("Forward Start")
        output_dict = {}; b, _, h_in_img, w_in_img = x.shape
        current_device = x.device

        # --- Stage 1: Feature Extraction ---
        features = None
        try:
            if self._get_module_device(self.backbone) != current_device:
                 self.backbone.to(current_device) # Move if needed
            features = self.backbone(x)
            if isinstance(features, (tuple, list)): features = features[-1]
            features = self._check_and_handle_nan_inf(features, "Backbone Features")
            self.memory_tracker.log_memory_usage("After Backbone")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.critical(f"OOM error during Backbone: {e}.", exc_info=False)
                self.memory_tracker.clear_memory("OOM Fallback")
                # Return minimal dummy output on CPU to avoid crashing upstream
                return {'seg_logits': torch.zeros(b, self.num_classes + 1, 1, 1, device='cpu'),
                        'combined_energy': torch.zeros(b, 1, 1, 1, device='cpu'), 'is_ood': torch.ones(b, dtype=torch.bool, device='cpu')} # Indicate failure
            else: raise e
        except Exception as e: logger.error(f"Error in backbone: {e}", exc_info=True); raise e

        if features is None:
             logger.critical("Backbone features are None. Returning dummy outputs.")
             dummy_logits = torch.zeros(b, self.num_classes + 1, 1, 1, device=current_device)
             dummy_energy = torch.zeros(b, 1, 1, 1, device=current_device)
             return {'seg_logits': dummy_logits, 'memory_energy': dummy_energy.clone(), 'feature_energy': dummy_energy.clone(),
                     'pebal_energy': dummy_energy.clone(), 'combined_energy': dummy_energy.clone(), 'is_ood': torch.ones(b, dtype=torch.bool)}

        b, c_feat, h_feat, w_feat = features.shape
        memory_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)
        feature_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)
        logits = None; features_adapted = None; features_mem_proj = None

        # --- Stage 2 & 3: Processing based on Insertion Point ---
        if self.insertion_point == 'after_backbone':
            try:
                features_adapted = self.channel_adapter(features)
                features_adapted = self._check_and_handle_nan_inf(features_adapted, "FeatAdapt(PostBB)")
                features_mem_proj = self.memory_input_proj(features_adapted)
                features_mem_proj = self._check_and_handle_nan_inf(features_mem_proj, "FeatMemProj(PostBB)")
                memory_energy_map, _ = self._apply_memory_interaction(features_mem_proj)
                memory_energy_map = self._check_and_handle_nan_inf(memory_energy_map, "MemEnergy")
                feature_energy_map = self.energy_head(features_mem_proj)
                feature_energy_map = self._check_and_handle_nan_inf(feature_energy_map, "FeatEnergy")

                if self.use_efficient_decoder:
                    if self._efficient_decoder_instance: logits = self._efficient_decoder_instance(features)
                    else: raise RuntimeError("EffDec enabled but instance is None.")
                else: # Original Segmentation Head
                    seg_head_input = self.final_seghead_proj(features_adapted)
                    seg_head_input = self._check_and_handle_nan_inf(seg_head_input, "SegHeadInput")
                    if self._original_segmentation_head:
                        logits = self._original_segmentation_head(seg_head_input)
                        if isinstance(logits, (tuple, list)): logits = logits[-1]
                    else: raise RuntimeError("Original SegHead needed but is None.")
            except Exception as e:
                 logger.error(f"Error in 'after_backbone' path: {e}", exc_info=True)
                 # Ensure minimal outputs exist if error occurred mid-path
                 if logits is None: logits = torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)
                 if memory_energy_map is None: memory_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)
                 if feature_energy_map is None: feature_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)

        else: # insertion_point == 'after_seghead'
            seg_features = None
            try:
                if self._original_segmentation_head is None: raise RuntimeError("Original SegHead required for 'after_seghead'.")
                seg_features = self._original_segmentation_head(features)
                if isinstance(seg_features, (tuple, list)): seg_features = seg_features[-1]
                seg_features = self._check_and_handle_nan_inf(seg_features, "SegHeadFeat")
                features_adapted = self.channel_adapter(seg_features)
                features_adapted = self._check_and_handle_nan_inf(features_adapted, "FeatAdapt(PostSH)")
                features_mem_proj = self.memory_input_proj(features_adapted)
                features_mem_proj = self._check_and_handle_nan_inf(features_mem_proj, "FeatMemProj(PostSH)")
                memory_energy_map, _ = self._apply_memory_interaction(features_mem_proj)
                memory_energy_map = self._check_and_handle_nan_inf(memory_energy_map, "MemEnergy")
                feature_energy_map = self.energy_head(features_mem_proj)
                feature_energy_map = self._check_and_handle_nan_inf(feature_energy_map, "FeatEnergy")

                if self.use_efficient_decoder and self._efficient_decoder_instance:
                    logits = self._efficient_decoder_instance(features_adapted) # Input is adapted seg head features
                elif self.final_classifier:
                    logits = self.final_classifier(features_mem_proj) # Use added final classifier
                else: raise RuntimeError("No final classifier or eff. decoder for 'after_seghead'.")
            except Exception as e:
                 logger.error(f"Error in 'after_seghead' path: {e}", exc_info=True)
                 if logits is None: logits = torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)
                 if memory_energy_map is None: memory_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)
                 if feature_energy_map is None: feature_energy_map = torch.zeros(b, 1, h_feat, w_feat, device=current_device)


        # --- Final Checks, Interpolation, Energy Combination ---
        if logits is None: # Should not happen if errors handled above, but final safety check
             logits = torch.zeros(b, self.num_classes+1, h_feat, w_feat, device=current_device)
        logits = self._check_and_handle_nan_inf(logits, "Logits Pre-Interpolation")
        h_pre, w_pre = logits.shape[-2:]

        # Interpolate to original image size
        if h_pre != h_in_img or w_pre != w_in_img:
             logger.debug(f"Interpolating outputs { (h_pre, w_pre)} -> {(h_in_img, w_in_img)}")
             try:
                 logits_final = F.interpolate(logits, size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
                 # Ensure energy maps are [B, 1, H, W] before interpolating
                 mem_energy_in = memory_energy_map.view(b, 1, h_pre, w_pre) if memory_energy_map is not None else torch.zeros(b,1,h_pre, w_pre, device=current_device)
                 feat_energy_in = feature_energy_map.view(b, 1, h_pre, w_pre) if feature_energy_map is not None else torch.zeros(b,1,h_pre, w_pre, device=current_device)

                 memory_energy_final = F.interpolate(mem_energy_in, size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
                 feature_energy_final = F.interpolate(feat_energy_in, size=(h_in_img, w_in_img), mode='bilinear', align_corners=False)
             except Exception as e:
                 logger.error(f"Output interpolation failed: {e}. Using un-interpolated.", exc_info=True)
                 logits_final = logits; memory_energy_final = memory_energy_map; feature_energy_final = feature_energy_map
        else:
             logits_final = logits; memory_energy_final = memory_energy_map; feature_energy_final = feature_energy_map

        # Ensure final shapes are [B, C, H, W] or [B, 1, H, W]
        if logits_final.ndim == 3: logits_final = logits_final.unsqueeze(1)
        if memory_energy_final.ndim == 3: memory_energy_final = memory_energy_final.unsqueeze(1)
        if feature_energy_final.ndim == 3: feature_energy_final = feature_energy_final.unsqueeze(1)

        # --- PEBAL Energy Calculation ---
        logits_final = self._check_and_handle_nan_inf(logits_final, "Logits Final")
        pebal_energy = torch.zeros_like(memory_energy_final)
        try:
            num_logits = logits_final.shape[1]
            in_dist_classes = self.num_classes if num_logits == self.num_classes + 1 else num_logits
            if in_dist_classes > 0:
                with torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else contextlib.nullcontext(): # Use FP32 for stability
                   logits_fp32 = logits_final[:, :in_dist_classes, :, :].float()
                   max_logits_in = torch.max(logits_fp32, dim=1, keepdim=True)[0]
                   pebal_energy = -(torch.logsumexp(logits_fp32 - max_logits_in, dim=1, keepdim=True) + max_logits_in)
                pebal_energy = self._check_and_handle_nan_inf(pebal_energy, "pebal_energy")
            else: logger.warning("Cannot calc PEBAL energy: 0 in-dist classes.")
        except Exception as e: logger.error(f"PEBAL energy calc failed: {e}", exc_info=True)

        # --- Combine Energies (Using Lambdas) ---
        memory_energy_final = self._check_and_handle_nan_inf(memory_energy_final, "MemEnergy Final")
        feature_energy_final = self._check_and_handle_nan_inf(feature_energy_final, "FeatEnergy Final")
        combined_energy = torch.zeros_like(pebal_energy)
        try:
            # APPLY WEIGHTS HERE
            combined_energy = pebal_energy + self.lambda_feature * feature_energy_final + self.lambda_memory * memory_energy_final
            combined_energy = torch.clamp(combined_energy, min=-100.0, max=100.0) # Prevent extremes
            combined_energy = self._check_and_handle_nan_inf(combined_energy, "combined_energy")
        except Exception as e:
             logger.error(f"Error combining energies: {e}. Using PEBAL energy.", exc_info=True)
             combined_energy = pebal_energy # Fallback

        # --- Prepare Output Dictionary ---
        output_dict['seg_logits'] = logits_final
        output_dict['memory_energy'] = memory_energy_final
        output_dict['feature_energy'] = feature_energy_final
        output_dict['pebal_energy'] = pebal_energy
        output_dict['combined_energy'] = combined_energy # This is now weighted
        output_dict['is_ood'] = torch.zeros(b, dtype=torch.bool, device=current_device) # Placeholder


        # Final shape check - ensure all map-like outputs match logits spatial dims
        ref_shape = logits_final.shape[-2:]
        for key, tensor in output_dict.items():
             if key != 'is_ood' and isinstance(tensor, torch.Tensor):
                  if tensor.shape[-2:] != ref_shape:
                       logger.warning(f"Output '{key}' shape {tensor.shape} spatial dims mismatch logits {logits_final.shape}. Resizing.")
                       expected_channels = logits_final.shape[1] if key == 'seg_logits' else 1
                       try:
                           resized_tensor = F.interpolate(tensor.view(b, -1, tensor.shape[-2], tensor.shape[-1]), size=ref_shape, mode='bilinear', align_corners=False)
                           if resized_tensor.shape[1] != expected_channels:
                                if expected_channels == 1: resized_tensor = resized_tensor.mean(dim=1, keepdim=True) # Average if needed
                                else: logger.error(f"Channel mismatch for {key} after resize. Cannot fix."); pass # Error, keep as is
                           output_dict[key] = self._check_and_handle_nan_inf(resized_tensor, f"{key}(resized)")
                       except Exception as resize_e:
                           logger.error(f"Resize failed for {key}: {resize_e}")


        self.memory_tracker.log_memory_usage("Forward End")
        if self.use_efficient_memory: self.memory_tracker.clear_memory("End of Forward Pass")

        return output_dict


    # --- Memory Update Call ---
    def update_memory(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, max_samples: Optional[int] = None):
        """
        Updates the memory bank using pre-projected features [N, C_mem] or [B, C_mem, H, W].
        Handles spatial sampling internally if needed.
        MODIFIED V2: Added comments/placeholders for advanced spatial sampling.
        """
        logger.debug("Starting external memory update call...")
        if not hasattr(self, 'memory_manager') or not self.use_efficient_memory:
             logger.warning("Memory Manager not available or disabled. Skipping update.")
             return

        self.eval() # Ensure model layers used for pre-processing (if any) are in eval mode

        if not isinstance(features, torch.Tensor) or features.shape[0] == 0:
            logger.warning("update_memory called with invalid/empty features. Skipping.")
            return

        # --- Ensure features are on the correct device for processing ---
        if features.device != self._model_device:
             features = features.to(self._model_device)
        if labels is not None and labels.device != self._model_device:
             labels = labels.to(self._model_device)


        flat_features = None
        flat_labels = None
        c_mem = self.memory_manager.feature_dim

        # Handle spatial features [B, C, H, W] input - apply spatial sampling if configured
        if features.ndim == 4:
            b, c_in, h_in, w_in = features.shape
            if c_in != c_mem:
                 logger.error(f"Feature dim mismatch (update): Input {c_in} != Memory {c_mem}.")
                 return

            features_sampled = features
            labels_sampled = labels # Will be modified if labels exist & sampling applied
            h_proc, w_proc = h_in, w_in

            # Apply spatial stride if enabled and dims are large enough
            needs_spatial_sampling = (h_in * w_in > 4096 and self.sampling_stride > 1)
            if needs_spatial_sampling:
                stride = self.sampling_stride
                if h_in >= stride and w_in >= stride:
                    # --- Simple Strided Spatial Sampling (Current) ---
                    # features_sampled = features[:, :, ::stride, ::stride] # Original line
                    # --- Placeholder for Advanced Spatial Sampling ---
                    # TODO: Implement class-balanced or anomaly-score-weighted spatial sampling here.
                    # This requires access to labels (available in this function) or anomaly scores.
                    # It should select pixel indices non-uniformly.
                    # E.g., sample indices based on label distribution -> gather features[indices], labels[indices]
                    logger.debug("Applying simple strided spatial sampling for memory update.")
                    features_sampled = features[:, :, ::stride, ::stride]
                    # --- Simple Strided Sampling for Labels (matches features) ---
                    if labels is not None:
                         if labels.ndim == 3 and labels.shape == (b, h_in, w_in):
                              labels_sampled = labels[:, ::stride, ::stride]
                         elif labels.ndim == 4 and labels.shape[1] == 1 and labels.shape[-2:] == (h_in, w_in):
                              labels_sampled = labels[:, 0, ::stride, ::stride] # Squeeze channel dim
                         else:
                              logger.warning(f"Label shape {labels.shape} invalid for spatial sampling. Discarding labels.")
                              labels_sampled = None
                    # -------------------------------------------------------------
                    h_proc, w_proc = features_sampled.shape[2:]
                    logger.debug(f"MemUpdate spatially sampled from {h_in}x{w_in} to {h_proc}x{w_proc} (Stride: {stride})")
                else: logger.warning(f"Cannot apply stride {stride} to {h_in}x{w_in} for update.")

            # Flatten sampled spatial features
            flat_features = features_sampled.permute(0, 2, 3, 1).contiguous().view(-1, c_mem)
            if labels_sampled is not None:
                # Ensure labels match B, H_proc, W_proc before reshape
                if labels_sampled.ndim == 3 and labels_sampled.shape == (b, h_proc, w_proc):
                     flat_labels = labels_sampled.reshape(-1)
                else: logger.warning(f"Label shape mismatch after sampling. Discarding labels.")

        elif features.ndim == 2: # Input is already flat [N, C]
             if features.shape[1] != c_mem:
                  logger.error(f"Feature dim mismatch (flat update): Input {features.shape[1]} != Memory {c_mem}.")
                  return
             flat_features = features
             flat_labels = labels.view(-1) if labels is not None else None
             if flat_labels is not None and flat_labels.shape[0] != flat_features.shape[0]:
                 logger.warning("Flat features/labels count mismatch. Discarding labels.")
                 flat_labels = None
        else:
             logger.error(f"Unexpected feature shape for update_memory: {features.shape}.")
             return

        if flat_features is None or flat_features.shape[0] == 0:
            logger.warning("Mem update skipped: 0 features after processing/flattening.")
            return

        # --- Sub-sampling Features Before Passing to Manager (Optional Limit) ---
        num_available = flat_features.shape[0]
        effective_max_samples = max_samples if max_samples is not None else -1 # -1 means no limit besides mem size itself handled internally
        sample_size = num_available
        if effective_max_samples > 0 and num_available > effective_max_samples:
            sample_size = effective_max_samples
            logger.debug(f"Subsampling flat features: {num_available} -> {sample_size} (max: {effective_max_samples})")
            indices = torch.randperm(num_available, device=flat_features.device)[:sample_size]
            sampled_features = flat_features[indices]
            if flat_labels is not None: sampled_labels = flat_labels[indices]
            else: sampled_labels = None
        else:
            sampled_features = flat_features
            sampled_labels = flat_labels

        del flat_features, flat_labels # Cleanup intermediate

        # --- Update Memory Manager (which now handles internal sampling like class-balancing) ---
        # Pass features already projected and on the manager's device.
        if sampled_features.device != self.memory_manager._device:
             sampled_features = sampled_features.to(self.memory_manager._device)
        if sampled_labels is not None and sampled_labels.device != self.memory_manager._device:
             sampled_labels = sampled_labels.to(self.memory_manager._device)

        try:
             # Call manager's update, which will perform normalization & its own sampling strategy
             self.memory_manager.update_memory(sampled_features, sampled_labels)
             logger.info(f"EfficientMemoryManager update initiated with {sampled_features.shape[0]} candidate vectors.")
        except Exception as e:
             logger.error(f"Error calling EfficientMemoryManager.update_memory: {e}", exc_info=True)
        finally:
             del sampled_features, sampled_labels
             if hasattr(self, 'memory_tracker'):
                 self.memory_tracker.log_memory_usage("End External Memory Update Call")
                 if self.use_efficient_memory:
                     self.memory_tracker.clear_memory("MemUpdate Cleanup")

        self.train() # Ensure model is back in train mode if needed outside


# --- Example Usage (Modified V2 - Added lambda params) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger(__name__)

    # --- Dummy Modules (Keep as before) ---
    class DummyBackbone(nn.Module):
        def __init__(self, out_channels=2048):
            super().__init__(); self.out_channels=out_channels
            self.conv1 = nn.Conv2d(3, 64, 3, 2, 1); self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(True); self.pool = nn.MaxPool2d(3, 2, 1)
            self.final_conv = nn.Conv2d(64, out_channels, 1); self.p = nn.Parameter(torch.randn(1))
        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.pool(x); x = self.final_conv(x); return x

    class DummySegHead(nn.Module):
         def __init__(self, in_channels, num_classes, head_channels=256):
            super().__init__(); self.in_c=in_channels; self.n_cls=num_classes; self.h_c=head_channels
            self.f_conv = nn.Conv2d(in_channels, head_channels, 3, 1, 1); self.bn = nn.BatchNorm2d(head_channels)
            self.relu = nn.ReLU(True); self.final = nn.Conv2d(head_channels, num_classes + 1, 1)
         def forward(self, x):
            x = self.f_conv(x); x = self.bn(x); x = self.relu(x); x = self.final(x); return x

    # --- Configuration ---
    NUM_CLASSES = 19
    BACKBONE_OUT_DIM = 512
    SEG_HEAD_INTERNAL_DIM = 128
    MEMORY_DIM = 64        # Dimension for memory bank and PQ
    PQ_BYTES = 8           # Number of subquantizers (m for PQ)
    TARGET_DIM = 128       # Dimension after adapter
    MEM_SIZE = 2048        # Make slightly larger for better PQ training example
    BATCH_SIZE = 4
    IMG_SIZE = (128, 128) # Increase size a bit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_logger.info(f"Using device: {device}")
    if not FAISS_AVAILABLE: main_logger.warning("FAISS not found, memory acceleration disabled.")

    # --- Instantiate Backbone ---
    backbone = DummyBackbone(out_channels=BACKBONE_OUT_DIM).to(device)

    # --- Test Case 1: After Backbone, Original Head ---
    main_logger.info("\n--- Test Case 1: After Backbone, Original Head (w/ PQ & lambdas) ---")
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
        pq_bytes=PQ_BYTES,              # Pass PQ config
        sampling_stride=4,              # Increase stride
        use_efficient_decoder=False,
        lambda_feature=0.6,             # Set custom weights
        lambda_memory=0.4
    ).to(device)

    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1], device=device)
    dummy_labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1]), device=device)
    main_logger.info("\nTesting forward pass (Case 1)...")
    model1.train()
    output1 = model1(dummy_input)
    for key, tensor in output1.items(): main_logger.info(f"  Output {key}: shape={tensor.shape}, device={tensor.device}, has_nan={torch.isnan(tensor).any()}")

    main_logger.info("\nTesting memory update (Case 1) - This will build/train the PQ index...")
    # Increase update data size for better PQ training
    update_data_multiplier = 4
    dummy_input_update = torch.randn(BATCH_SIZE * update_data_multiplier, 3, IMG_SIZE[0], IMG_SIZE[1], device=device)
    dummy_labels_update = torch.randint(0, NUM_CLASSES, (BATCH_SIZE * update_data_multiplier, IMG_SIZE[0], IMG_SIZE[1]), device=device)

    with torch.no_grad():
        model1.eval()
        features_raw = model1.backbone(dummy_input_update)
        features_adapted = model1.channel_adapter(features_raw)
        features_mem_proj = model1.memory_input_proj(features_adapted)
        # Call model's update_memory which now handles spatial sampling internally if needed
        model1.update_memory(features_mem_proj, labels=dummy_labels_update, max_samples=MEM_SIZE * 2) # Allow more candidates than memory size for manager's sampling
    main_logger.info("Memory update complete. FAISS index should be trained and populated (check logs).")

    # Test forward pass again AFTER memory update
    main_logger.info("\nTesting forward pass (Case 1, after memory update)...")
    model1.train()
    with torch.no_grad(): # Often inference done in no_grad context
         output1_post = model1(dummy_input)
    for key, tensor in output1_post.items(): main_logger.info(f"  Output {key}: shape={tensor.shape}, device={tensor.device}, has_nan={torch.isnan(tensor).any()}")


    del model1, output1, output1_post, seg_head_orig, dummy_input_update, dummy_labels_update
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    main_logger.info("Case 1 cleanup done.")


    # --- Test Case 2: After Backbone, Efficient Decoder ---
    # (Keeping it simpler - run forward but maybe skip repeated PQ build unless needed)
    main_logger.info("\n--- Test Case 2: After Backbone, Efficient Decoder (w/ PQ & lambdas) ---")
    seg_head_dummy = DummySegHead(in_channels=TARGET_DIM, num_classes=NUM_CLASSES).to(device) # Input dim ignored if replaced
    eff_dec_kwargs = {'in_channels': BACKBONE_OUT_DIM, 'feature_dim': MEMORY_DIM, 'attention_heads': 4}
    model2 = HopfieldPEBALModel(
        backbone=backbone,
        segmentation_head=seg_head_dummy,
        num_classes=NUM_CLASSES, memory_feature_dim=MEMORY_DIM, memory_size=MEM_SIZE,
        insertion_point='after_backbone', target_feature_dim=TARGET_DIM,
        use_efficient_memory=True, use_faiss=FAISS_AVAILABLE, pq_bytes=PQ_BYTES,
        sampling_stride=4, use_efficient_decoder=True, efficient_decoder_kwargs=eff_dec_kwargs,
        lambda_feature=0.5, lambda_memory=0.5 # Default weights
    ).to(device)

    main_logger.info("\nTesting forward pass (Case 2)...")
    model2.train()
    output2 = model2(dummy_input)
    for key, tensor in output2.items(): main_logger.info(f"  Output {key}: shape={tensor.shape}, dev={tensor.device}, nan={torch.isnan(tensor).any()}")
    # Skip memory update here to save time, focus was PQ build in Case 1.

    del model2, output2, seg_head_dummy
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    main_logger.info("Case 2 cleanup done.")

    # --- Test Case 3: After SegHead, Original Classifier ---
    main_logger.info("\n--- Test Case 3: After SegHead, Orig Classifier (w/ PQ & lambdas) ---")
    seg_head_for_features = DummySegHead(in_channels=BACKBONE_OUT_DIM, num_classes=NUM_CLASSES, head_channels=SEG_HEAD_INTERNAL_DIM).to(device)
    model3 = HopfieldPEBALModel(
        backbone=backbone, segmentation_head=seg_head_for_features,
        num_classes=NUM_CLASSES, memory_feature_dim=MEMORY_DIM, memory_size=MEM_SIZE,
        insertion_point='after_seghead', target_feature_dim=TARGET_DIM,
        use_efficient_memory=True, use_faiss=FAISS_AVAILABLE, pq_bytes=PQ_BYTES,
        sampling_stride=4, use_efficient_decoder=False,
        lambda_feature=0.3, lambda_memory=0.7 # Different weights
    ).to(device)

    main_logger.info("\nTesting forward pass (Case 3)...")
    model3.train()
    output3 = model3(dummy_input)
    for key, tensor in output3.items(): main_logger.info(f"  Output {key}: shape={tensor.shape}, dev={tensor.device}, nan={torch.isnan(tensor).any()}")
    # Skip memory update here.

    del model3, output3, seg_head_for_features
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    main_logger.info("Case 3 cleanup done.")


    main_logger.info("\nExample Usage Complete.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import math
import gc
import psutil
import time
from torch.utils.checkpoint import checkpoint
import logging

logger = logging.getLogger("Hopfield-PEBAL")

# --------------------- Memory & Tracking Utilities --------------------- #

class MemoryTracker:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.peak_gpu_mem = 0
        self.peak_cpu_mem = 0

    def get_gpu_memory_usage(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current = torch.cuda.memory_allocated() / (1024 * 1024)
            self.peak_gpu_mem = max(self.peak_gpu_mem, current)
            return current
        return 0

    def get_cpu_memory_usage(self):
        current = psutil.Process().memory_info().rss / (1024 * 1024)
        self.peak_cpu_mem = max(self.peak_cpu_mem, current)
        return current

    def log_memory_usage(self, operation_name=""):
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            gpu_mem = self.get_gpu_memory_usage()
            cpu_mem = self.get_cpu_memory_usage()
            print(f"[MemoryTracker] {operation_name}: GPU {gpu_mem:.2f}MB, CPU {cpu_mem:.2f}MB | Peak: GPU {self.peak_gpu_mem:.2f}MB, CPU {self.peak_cpu_mem:.2f}MB")
            self.last_log_time = current_time

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# --------------------- Efficient Memory Manager --------------------- #

class EfficientMemoryManager(nn.Module):
    def __init__(self, feature_dim=256, memory_size=1024, pq_bytes=8, 
                 sampling_ratio=0.25, num_classes=19, use_faiss=True):
        super(EfficientMemoryManager, self).__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.sampling_ratio = sampling_ratio
        self.num_classes = num_classes
        self.pq_bytes = pq_bytes
        self.use_faiss = use_faiss
        self.memory_tracker = MemoryTracker()

        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_labels', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.memory_initialized = False
        self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.long))

        if use_faiss and faiss.get_num_gpus() > 0:
            self.faiss_res = faiss.StandardGpuResources()
            self.faiss_index = None
        else:
            self.faiss_res = None
            self.faiss_index = None

    def estimate_memory_requirements(self, features_shape):
        b, c, h, w = features_shape
        feature_mem_mb = (b * c * h * w * 4) / (1024 * 1024)
        if torch.cuda.is_available():
            available_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            used_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            free_mem_mb = available_mem_mb - used_mem_mb
        else:
            free_mem_mb = psutil.virtual_memory().available / (1024 * 1024)
        return {'feature_mem_mb': feature_mem_mb, 'free_mem_mb': free_mem_mb, 'ratio': feature_mem_mb / (free_mem_mb + 1e-6)}

    def get_optimal_pq_params(self, feature_dim, mem_estimate):
        if mem_estimate['ratio'] > 0.5:
            pq_bytes = min(4, feature_dim // 16)
            pq_bits = 4
        elif mem_estimate['ratio'] > 0.2:
            pq_bytes = min(8, feature_dim // 8)
            pq_bits = 8
        else:
            pq_bytes = min(16, feature_dim // 4)
            pq_bits = 8
        pq_bytes = max(1, min(pq_bytes, feature_dim // 4))
        print(f"Dynamic PQ params: bytes={pq_bytes}, bits={pq_bits}")
        return pq_bytes, pq_bits

    def strided_sampling(self, features, stride=2):
        B, C, H, W = features.shape
        return features[:, :, ::stride, ::stride]

    def reservoir_sampling(self, features, k):
        n = features.shape[0]
        if n <= k:
            return features
        reservoir = features[:k].clone()
        for i in range(k, n):
            j = torch.randint(0, i + 1, (1,)).item()
            if j < k:
                reservoir[j] = features[i]
        return reservoir

    def class_balanced_sampling(self, features, labels):
        if labels is None:
            return self.reservoir_sampling(features, self.memory_size)
        unique_classes = torch.unique(labels)
        samples_per_class = max(1, self.memory_size // unique_classes.numel())
        balanced_features = []
        for cls in unique_classes:
            if cls == 255:
                continue
            cls_mask = (labels == cls)
            cls_features = features[cls_mask]
            if cls_features.shape[0] > samples_per_class:
                cls_features = self.reservoir_sampling(cls_features, samples_per_class)
            balanced_features.append(cls_features)
        if not balanced_features:
            return torch.zeros(0, features.shape[1], device=features.device)
        return torch.cat(balanced_features, dim=0)

    def kmeans_sampling(self, features, k):
        if (not self.use_faiss) or (features.shape[0] < k) or (features.shape[0] < 5 * k):
            print("Warning: Not enough features for reliable K-means; using reservoir sampling.")
            return self.reservoir_sampling(features, k)
        try:
            import faiss
            features_np = features.detach().cpu().numpy().astype(np.float32)
            if not np.all(np.isfinite(features_np)):
                print("Warning: Non-finite values detected in features, cleaning.")
                features_np = np.where(np.isfinite(features_np), features_np, 0)
            faiss.normalize_L2(features_np)
            use_gpu = self.faiss_res is not None
            try:
                kmeans = faiss.Kmeans(d=features.shape[1], k=k, gpu=use_gpu)
                kmeans.train(features_np)
                centroids = torch.from_numpy(kmeans.centroids).to(features.device)
            except Exception as e:
                print(f"Warning: Error in GPU K-means: {e}, falling back to CPU.")
                kmeans = faiss.Kmeans(d=features.shape[1], k=k, gpu=False)
                kmeans.train(features_np)
                centroids = torch.from_numpy(kmeans.centroids).to(features.device)
            self.memory_tracker.log_memory_usage("After K-means")
            return centroids
        except Exception as e:
            print(f"Warning: Error in K-means sampling: {e}, falling back to reservoir sampling.")
            return self.reservoir_sampling(features, k)

    def create_faiss_index(self, features=None):
        if not self.use_faiss:
            return
        try:
            import faiss
            if features is None:
                features = self.memory_bank
            features_np = features.detach().cpu().numpy().astype(np.float32)
            mem_estimate = self.estimate_memory_requirements((1, features.shape[1],
                                                               int(math.sqrt(features.shape[0])),
                                                               int(math.sqrt(features.shape[0]))))
            pq_bytes, pq_bits = self.get_optimal_pq_params(self.feature_dim, mem_estimate)
            if pq_bytes > 0:
                m = min(pq_bytes, self.feature_dim // 8)
                nbits = pq_bits
                try:
                    if self.faiss_res:
                        cpu_index = faiss.IndexPQ(self.feature_dim, m, nbits)
                        cpu_index.train(features_np)
                        try:
                            self.faiss_index = faiss.index_cpu_to_gpu(self.faiss_res, 0, cpu_index)
                        except Exception as e:
                            print(f"Warning: GPU conversion failed ({e}). Using CPU PQ index.")
                            self.faiss_index = cpu_index
                    else:
                        self.faiss_index = faiss.IndexPQ(self.feature_dim, m, nbits)
                        self.faiss_index.train(features_np)
                except Exception as e:
                    print(f"Warning: Error creating PQ index: {e}, falling back to FlatL2")
                    if self.faiss_res:
                        try:
                            self.faiss_index = faiss.GpuIndexFlatL2(self.faiss_res, self.feature_dim)
                        except Exception as e2:
                            print(f"Warning: Error creating GPU FlatL2 index: {e2}, using CPU")
                            self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
                    else:
                        self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
            else:
                try:
                    if self.faiss_res:
                        self.faiss_index = faiss.GpuIndexFlatL2(self.faiss_res, self.feature_dim)
                    else:
                        self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
                except Exception as e:
                    print(f"Warning: Error creating FlatL2 index: {e}, falling back to CPU")
                    self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
            if len(features_np) > 0:
                self.faiss_index.add(features_np)
            self.memory_tracker.clear_memory()
            self.memory_tracker.log_memory_usage("After FAISS index creation")
        except Exception as e:
            print(f"Warning: Error in create_faiss_index: {e}")
            self.faiss_index = None

    def query_faiss(self, query_features, k=5):
        if not self.use_faiss or self.faiss_index is None:
            return None, None
        query_np = query_features.detach().cpu().numpy().astype(np.float32)
        distances, indices = self.faiss_index.search(query_np, k)
        return torch.from_numpy(distances).to(query_features.device), torch.from_numpy(indices).to(query_features.device)

    def update_memory(self, features, labels=None):
        self.memory_tracker.log_memory_usage("Before memory update")
        with torch.no_grad():
            if features.shape[0] == 0:
                return
            features = F.normalize(features, p=2, dim=1)
            if labels is not None:
                features = self.class_balanced_sampling(features, labels)
            if features.shape[0] > self.memory_size:
                if self.use_faiss:
                    features = self.kmeans_sampling(features, self.memory_size)
                else:
                    features = self.reservoir_sampling(features, self.memory_size)
            num_features = features.shape[0]
            if not self.memory_initialized:
                self.memory_bank[:num_features] = features
                self.memory_ptr[0] = num_features
                self.memory_initialized = (num_features >= self.memory_size)
                if self.use_faiss and num_features > 0:
                    self.create_faiss_index(features)
            else:
                ptr = self.memory_ptr[0].item()
                if ptr + num_features > self.memory_size:
                    space_left = self.memory_size - ptr
                    self.memory_bank[ptr:] = features[:space_left]
                    self.memory_bank[:num_features - space_left] = features[space_left:]
                    self.memory_ptr[0] = (num_features - space_left) % self.memory_size
                else:
                    self.memory_bank[ptr:ptr+num_features] = features
                    self.memory_ptr[0] = (ptr + num_features) % self.memory_size
                if self.use_faiss:
                    self.create_faiss_index()
        self.memory_tracker.log_memory_usage("After memory update")
        self.memory_tracker.clear_memory()

# --------------------- Efficient Decoder --------------------- #

class EfficientSegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, feature_dim=128, attention_heads=8):
        super(EfficientSegmentationDecoder, self).__init__()
        self.feature_projector = nn.Conv2d(in_channels, feature_dim, kernel_size=1)
        self.attention_heads = attention_heads
        # Use in-place ReLU for memory savings.
        self.query_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.memory_tracker = MemoryTracker()
    
    def forward(self, x):
        batch_size, _, h, w = x.shape
        self.memory_tracker.log_memory_usage("Decoder start")
        features = self.feature_projector(x)
        if h * w > 128 * 128:
            downscale_factor = min(4, math.ceil(math.sqrt(h * w / (128 * 128))))
            attn_features = F.avg_pool2d(features, downscale_factor)
        else:
            attn_features = features
        queries = self.query_conv(attn_features)
        keys = self.key_conv(attn_features)
        values = self.value_conv(attn_features)
        ah, aw = queries.size(2), queries.size(3)
        feature_dim = queries.size(1)
        head_dim = feature_dim // self.attention_heads
        # Reshape for multi-head attention.
        queries = queries.view(batch_size, self.attention_heads, head_dim, -1)
        keys = keys.view(batch_size, self.attention_heads, head_dim, -1)
        values = values.view(batch_size, self.attention_heads, head_dim, -1)
        chunk_size = 5000
        if keys.size(3) > chunk_size:
            attention_output = []
            for q_chunk in queries.split(chunk_size, dim=3):
                attention_scores = torch.matmul(q_chunk.transpose(2, 3), keys)
                attention_scores = attention_scores / math.sqrt(head_dim)
                attention_weights = F.softmax(attention_scores, dim=3)
                chunk_output = torch.matmul(attention_weights, values.transpose(2, 3))
                attention_output.append(chunk_output)
                self.memory_tracker.clear_memory()
            # Concatenate along the dimension representing the query positions.
            attention_output = torch.cat(attention_output, dim=3)
        else:
            attention_scores = torch.matmul(queries.transpose(2, 3), keys)
            attention_scores = attention_scores / math.sqrt(head_dim)
            attention_weights = F.softmax(attention_scores, dim=3)
            attention_output = torch.matmul(attention_weights, values.transpose(2, 3))
        attention_output = attention_output.view(batch_size, feature_dim, ah, aw)
        if attn_features.shape[2:] != (h, w):
            attention_output = F.interpolate(attention_output, size=(h, w), mode='bilinear', align_corners=False)
        output = self.classifier(attention_output)
        self.memory_tracker.log_memory_usage("Decoder end")
        return output

# --------------------- Enhanced Hopfield-PEBAL Model --------------------- #

class EnhancedHopfieldPEBAL(nn.Module):
    def __init__(self, num_classes=19, memory_size=1024, feature_dim=256, 
                 hopfield_beta=2.0, prototype_count=10, num_heads=4,
                 use_faiss=True, pq_bytes=8, efficient_attention=True):
        super(EnhancedHopfieldPEBAL, self).__init__()
        from utils import import_deepwv3plus
        DeepWV3Plus = import_deepwv3plus()
        self.segmentation_model = DeepWV3Plus(num_classes)
        self.num_classes = num_classes
        self.memory_tracker = MemoryTracker()
        self.feature_dim = feature_dim

        # For the feature projector and energy head, use in-place activations.
        self.feature_projector = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=1)
        )
        self.energy_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.log_beta = nn.Parameter(torch.log(torch.tensor(hopfield_beta, dtype=torch.float)))
        self.memory_manager = EfficientMemoryManager(
            feature_dim=feature_dim,
            memory_size=memory_size,
            pq_bytes=pq_bytes,
            num_classes=num_classes,
            use_faiss=use_faiss
        )
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.attention_heads = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_heads)])
        self.energy_fusion = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        self.efficient_attention = efficient_attention
        if efficient_attention:
            self.efficient_decoder = EfficientSegmentationDecoder(
                in_channels=128,
                num_classes=num_classes,
                feature_dim=feature_dim,
                attention_heads=num_heads
            )
        self.prototype_count = prototype_count
        if prototype_count > 0:
            self.register_buffer('prototypes', torch.zeros(prototype_count, feature_dim))
            self.register_buffer('prototype_labels', torch.zeros(prototype_count, dtype=torch.long))
            self.prototype_initialized = False
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var', torch.ones(feature_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    # Define a helper for feature extraction with gradient checkpointing.
    def _extract_features(self, x):
        # This function wraps mod1, pool2, and mod2.
        out1 = self.segmentation_model.mod1(x)
        # OPTIONAL: Offload out1 to CPU if needed.
        # out1 = out1.to('cpu')
        pooled = self.segmentation_model.pool2(out1)
        features = self.segmentation_model.mod2(pooled)
        return features

    def forward(self, x, return_all_outputs=False, use_efficient_attention=None):
        try:
            self.memory_tracker.log_memory_usage("Forward start")
            self.memory_tracker.clear_memory()
            duplicated = False
            if x.size(0) == 1:
                x = x.repeat(2, 1, 1, 1)
                duplicated = True

            # Use gradient checkpointing on the feature extraction block.
            try:
                features = torch.utils.checkpoint.checkpoint(self._extract_features, x)
                # Optionally, move features back to GPU if offloaded (e.g., features = features.to(x.device))
                if self.efficient_attention and (use_efficient_attention is not False):
                    logits = self.efficient_decoder(features)
                else:
                    logits = self.segmentation_model(x)
            except Exception as e:
                print(f"Feature extraction error: {e}")
                logits = self.segmentation_model(x)
                try:
                    features = self.segmentation_model.mod2(self.segmentation_model.pool2(self.segmentation_model.mod1(x)))
                except Exception as e2:
                    print(f"Second extraction attempt failed: {e2}")
                    if logits is not None:
                        features = logits
                    else:
                        raise ValueError("Could not extract features for energy calculation")
            
            if duplicated:
                features = features[0:1]
                logits = logits[0:1]

            print(f"Features shape before energy head: {features.shape}")
            print(f"Logits shape: {logits.shape}")

            energy = self.energy_head(features)
            self.memory_tracker.log_memory_usage("After feature extraction")

            B, _, _, _ = features.shape
            if features.shape[2] * features.shape[3] > 64 * 64:
                stride = max(1, min(4, features.shape[2] // 64))
                proj_features = self.feature_projector(features)[:, :, ::stride, ::stride]
            else:
                proj_features = self.feature_projector(features)
            B, C, H, W = proj_features.shape
            proj_features = proj_features.reshape(B, C, -1).transpose(1, 2)
            flat_features = proj_features.reshape(-1, C)
            flat_features = F.normalize(flat_features, p=2, dim=1)
            beta = torch.exp(self.log_beta)

            if self.memory_manager.memory_initialized:
                if self.memory_manager.use_faiss and self.memory_manager.faiss_index is not None:
                    chunk_size = 10000
                    num_flat = flat_features.size(0)
                    retrieved = torch.zeros_like(flat_features)
                    for i in range(0, num_flat, chunk_size):
                        end = min(i + chunk_size, num_flat)
                        chunk = flat_features[i:end]
                        distances, indices = self.memory_manager.query_faiss(chunk, k=5)
                        distances = distances.to(flat_features.device)
                        indices = indices.to(flat_features.device)
                        weights = F.softmax(-distances * beta, dim=1)
                        memory_vectors = self.memory_manager.memory_bank[indices.view(-1)].view(indices.size(0), indices.size(1), -1)
                        chunk_retrieved = torch.bmm(weights.unsqueeze(1), memory_vectors).squeeze(1)
                        retrieved[i:end] = chunk_retrieved
                        self.memory_tracker.clear_memory()
                else:
                    chunk_size = 10000
                    num_flat = flat_features.size(0)
                    if num_flat > chunk_size:
                        retrieved_chunks = []
                        for i in range(0, num_flat, chunk_size):
                            end = min(i + chunk_size, num_flat)
                            chunk = flat_features[i:end]
                            head_results = []
                            for head in self.attention_heads:
                                head_feat = head(chunk)
                                sim = torch.mm(head_feat, self.memory_manager.memory_bank.t())
                                sim = sim * beta
                                attn = F.softmax(sim, dim=1)
                                head_results.append(torch.mm(attn, self.memory_manager.memory_bank))
                                self.memory_tracker.clear_memory()
                            chunk_retrieved = torch.stack(head_results, dim=1).mean(dim=1)
                            retrieved_chunks.append(chunk_retrieved)
                        retrieved = torch.cat(retrieved_chunks, dim=0)
                    else:
                        head_results = []
                        for head in self.attention_heads:
                            head_feat = head(flat_features)
                            sim = torch.mm(head_feat, self.memory_manager.memory_bank.t())
                            sim = sim * beta
                            attn = F.softmax(sim, dim=1)
                            head_results.append(torch.mm(attn, self.memory_manager.memory_bank))
                            self.memory_tracker.clear_memory()
                        retrieved = torch.stack(head_results, dim=1).mean(dim=1)
                memory_energies = 1.0 - torch.sum(flat_features * retrieved, dim=1)
                memory_energies = memory_energies.view(B, H * W)
            else:
                memory_energies = torch.zeros(B, H * W, device=x.device)
            self.memory_tracker.log_memory_usage("After memory retrieval")
            base_energy = F.interpolate(energy, size=(H, W), mode='bilinear', align_corners=False)
            base_energy = base_energy.view(B, H * W)
            fusion_input = torch.cat([base_energy.unsqueeze(2), memory_energies.unsqueeze(2)], dim=2)
            fusion_chunk = 10000
            if fusion_input.numel() > fusion_chunk:
                fused_chunks = []
                for i in range(0, H * W, fusion_chunk // B):
                    end_i = min(i + fusion_chunk // B, H * W)
                    chunk = fusion_input[:, i:end_i, :]
                    chunk_shape = chunk.shape
                    chunk_flat = chunk.reshape(-1, 2)
                    fused_chunk = self.energy_fusion(chunk_flat).reshape(chunk_shape[0], chunk_shape[1], 1)
                    fused_chunks.append(fused_chunk)
                    self.memory_tracker.clear_memory()
                fused_energy = torch.cat(fused_chunks, dim=1)
            else:
                fusion_shape = fusion_input.shape
                fusion_input_flat = fusion_input.reshape(-1, 2)
                fused_energy = self.energy_fusion(fusion_input_flat).reshape(fusion_shape[0], fusion_shape[1], 1)
            fused_energy = fused_energy.squeeze(2).view(B, H, W, 1).permute(0, 3, 1, 2)
            self.memory_tracker.log_memory_usage("After energy fusion")
            self.memory_tracker.clear_memory()
            if return_all_outputs:
                return {
                    'logits': logits,
                    'energy': fused_energy,
                    'features': flat_features,
                    'memory_energies': memory_energies,
                    'raw_features': features
                }
            else:
                return logits, fused_energy
        except Exception as e:
            logger.error(f"Error in EnhancedHopfieldPEBAL forward: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            B = x.size(0)
            dummy_logits = torch.zeros(B, self.num_classes, x.shape[2], x.shape[3], device=x.device)
            dummy_energy = torch.zeros(B, 1, x.shape[2], x.shape[3], device=x.device)
            return {'logits': dummy_logits,
                    'energy': dummy_energy,
                    'features': None,
                    'memory_energies': None,
                    'raw_features': None}

    def update_memory(self, features, labels=None):
        with torch.no_grad():
            if features.shape[2] * features.shape[3] > 64 * 64:
                stride = max(1, min(4, features.shape[2] // 64))
                proj_features = self.feature_projector(features)[:, :, ::stride, ::stride]
            else:
                proj_features = self.feature_projector(features)
            B, C, H, W = proj_features.shape
            proj_features = proj_features.reshape(B, C, -1).transpose(1, 2)
            flat_features = proj_features.reshape(-1, C)
            if labels is not None:
                if stride > 1:
                    labels = F.interpolate(labels.float().unsqueeze(1), size=(H, W), mode='nearest').squeeze(1).long()
                flat_labels = labels.view(-1)
            else:
                flat_labels = None
            self.memory_manager.update_memory(flat_features, flat_labels)
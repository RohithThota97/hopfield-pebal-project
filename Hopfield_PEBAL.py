import torch
import torch.nn as nn
import torch.nn.functional as F

class HopfieldPEBALSegmentation(nn.Module):
    def __init__(self, num_classes=19, memory_size=1024, feature_dim=256, 
                 hopfield_beta=2.0, prototype_count=10, num_heads=4):
        super(HopfieldPEBALSegmentation, self).__init__()
        # Import DeepWV3Plus using our helper function.
        from utils import import_deepwv3plus
        DeepWV3Plus = import_deepwv3plus()  # Imports from /home/ha51dybi/hop-pebal/code
        self.segmentation_model = DeepWV3Plus(num_classes)
        
        # Enhanced Feature Projector: using 3x3 convolution for spatial context.
        # We assume that intermediate features (fallback: m2) have 128 channels.
        self.feature_dim = feature_dim
        self.feature_projector = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, kernel_size=1)
        )
        
        # Energy head for OOD detection
        self.energy_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        # Learnable temperature parameter (dynamic adjustment)
        self.log_beta = nn.Parameter(torch.log(torch.tensor(hopfield_beta, dtype=torch.float)))
        
        # Multi-head attention for memory retrieval.
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_heads)
        ])
        
        # Energy fusion network (learned combination of base energy and memory energy)
        self.energy_fusion = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Hopfield memory bank.
        self.memory_size = memory_size
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.memory_initialized = False
        
        # Prototype parameters (for class-aware memory organization).
        self.prototype_count = prototype_count
        self.register_buffer('class_prototypes', torch.zeros(num_classes, prototype_count, feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes, dtype=torch.long))
    
    def forward(self, x, return_all_outputs=False):
        # Free CUDA cache (optional, can help fragmentation)
        torch.cuda.empty_cache()
        
        # If input has batch size 1, duplicate for BatchNorm stability.
        duplicated = False
        if x.size(0) == 1:
            x = x.repeat(2, 1, 1, 1)
            duplicated = True
        
        # Call segmentation model normally (no extra keywords)
        logits = self.segmentation_model(x)
        # Manually extract intermediate features from a known branch.
        try:
            # Using mod1 and pool2, mod2 as fallback; adjust if needed.
            x1 = self.segmentation_model.mod1(x)
            m2 = self.segmentation_model.mod2(self.segmentation_model.pool2(x1))
            features = m2
        except Exception as e:
            features = logits  # Fallback
        
        if duplicated:
            features = features[0:1]
            logits = logits[0:1]
        
        # Compute base energy via the energy head.
        energy = self.energy_head(features)
        
        # Project features with the enhanced projector.
        proj_features = self.feature_projector(features)
        B, C, H, W = proj_features.shape
        proj_features = proj_features.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        flat_features = proj_features.reshape(-1, C)  # [B*H*W, C]
        flat_features = F.normalize(flat_features, p=2, dim=1)
        
        # Compute dynamic temperature.
        beta = torch.exp(self.log_beta)
        
        # Multi-head attention for memory retrieval (with chunking to control memory usage)
        if self.memory_initialized:
            chunk_size = 10000  # adjust chunk size based on your GPU capacity
            num_flat = flat_features.size(0)
            if num_flat > chunk_size:
                retrieved_chunks = []
                for i in range(0, num_flat, chunk_size):
                    end = min(i + chunk_size, num_flat)
                    chunk = flat_features[i:end]
                    head_results = []
                    for head in self.attention_heads:
                        head_feat = head(chunk)
                        sim = torch.mm(head_feat, self.memory_bank.t())
                        sim = sim * beta
                        attn = torch.softmax(sim, dim=1)
                        head_results.append(torch.mm(attn, self.memory_bank))
                    # Average across heads.
                    chunk_retrieved = torch.stack(head_results, dim=1).mean(dim=1)
                    retrieved_chunks.append(chunk_retrieved)
                retrieved = torch.cat(retrieved_chunks, dim=0)
            else:
                head_results = []
                for head in self.attention_heads:
                    head_feat = head(flat_features)
                    sim = torch.mm(head_feat, self.memory_bank.t())
                    sim = sim * beta
                    attn = torch.softmax(sim, dim=1)
                    head_results.append(torch.mm(attn, self.memory_bank))
                retrieved = torch.stack(head_results, dim=1).mean(dim=1)
            
            # Compute memory-based energy as 1 - cosine similarity.
            memory_energies = 1.0 - torch.sum(flat_features * retrieved, dim=1)
            memory_energies = memory_energies.view(B, H * W)
        else:
            memory_energies = torch.zeros(B, H * W, device=x.device)
        
        # Fusion: fuse base energy and memory energy using a learned network.
        base_energy = energy.view(B, H * W)  # flatten base energy.
        fusion_input = torch.cat([base_energy.unsqueeze(2), memory_energies.unsqueeze(2)], dim=2)  # [B, H*W, 2]
        
        # Process fusion input in chunks if needed.
        chunk_size = 10000
        if fusion_input.numel() > chunk_size:
            fused_energy = torch.zeros_like(base_energy)
            for b in range(B):
                for i in range(0, fusion_input.shape[1], chunk_size):
                    end_i = min(i + chunk_size, fusion_input.shape[1])
                    chunk = fusion_input[b, i:end_i, :]
                    chunk_out = self.energy_fusion(chunk).squeeze(1)
                    fused_energy[b, i:end_i] = chunk_out
        else:
            fused_energy = self.energy_fusion(fusion_input).squeeze(2)
        
        if return_all_outputs:
            return {
                'logits': logits,
                'energy': fused_energy,
                'features': flat_features,
                'memory_energies': memory_energies,
                'raw_features': features
            }
        else:
            return {
                'logits': logits,
                'energy': fused_energy,
                'features': flat_features,
                'memory_energies': memory_energies
            }
    
    def update_memory(self, features, is_anomaly=None, labels=None):
        """Enhanced memory bank update with diversity preservation."""
        with torch.no_grad():
            if is_anomaly is not None:
                features = features[~is_anomaly]
            if features.shape[0] == 0:
                return
            # Diversity-based sampling if already initialized.
            if self.memory_initialized and features.shape[0] > 10:
                chunk_size = 1000  # adjust as needed
                n = features.shape[0]
                avg_sim = torch.zeros(n, device=features.device)
                for i in range(0, n, chunk_size):
                    end_idx = min(i + chunk_size, n)
                    chunk = features[i:end_idx]
                    sim = torch.mm(chunk, self.memory_bank.t())
                    avg_sim[i:end_idx] = sim.mean(dim=1)
                k = min(n, self.memory_size // 10)
                _, indices = torch.topk(avg_sim, k=k, largest=False)
                features = features[indices]
            
            if features.shape[0] > self.memory_size:
                idx = torch.randperm(features.shape[0])[:self.memory_size]
                features = features[idx]
            
            if not self.memory_initialized:
                num_features = min(features.shape[0], self.memory_size)
                self.memory_bank[:num_features] = features[:num_features]
                self.memory_ptr[0] = num_features
                self.memory_initialized = (num_features == self.memory_size)
            else:
                num_features = features.shape[0]
                ptr = self.memory_ptr[0].item()
                if ptr + num_features > self.memory_size:
                    space_left = self.memory_size - ptr
                    self.memory_bank[ptr:] = features[:space_left]
                    self.memory_bank[:num_features - space_left] = features[space_left:]
                    self.memory_ptr[0] = (num_features - space_left) % self.memory_size
                else:
                    self.memory_bank[ptr:ptr+num_features] = features
                    self.memory_ptr[0] = (ptr + num_features) % self.memory_size

    def update_memory_by_class(self, features, labels):
        """Update memory ensuring balanced representation across classes."""
        with torch.no_grad():
            if labels is None or features.shape[0] == 0:
                return
            # Process features in manageable chunks if needed.
            chunk_size = 10000
            if features.shape[0] > chunk_size:
                for i in range(0, features.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, features.shape[0])
                    self.update_memory_by_class(features[i:end_idx], labels[i:end_idx])
                return
            
            unique_classes = torch.unique(labels)
            slots_per_class = self.memory_size // self.segmentation_model.num_classes
            for cls in unique_classes:
                if cls == 255:  # ignore index
                    continue
                cls_mask = (labels == cls)
                cls_features = features[cls_mask]
                if cls_features.shape[0] == 0:
                    continue
                cls_start = int(cls) * slots_per_class
                cls_end = cls_start + slots_per_class
                if cls_features.shape[0] > slots_per_class:
                    idx = torch.randperm(cls_features.shape[0])[:slots_per_class]
                    self.memory_bank[cls_start:cls_end] = cls_features[idx]
                else:
                    self.memory_bank[cls_start:cls_start + cls_features.shape[0]] = cls_features

def inference(model, image, device):
    """Run inference on a single image using the model.
    For large images, process in overlapping tiles.
    """
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        B, C, H, W = image.shape
        max_tile = 1024  # adjust maximum tile dimension if needed
        
        if H > max_tile or W > max_tile:
            stride = max_tile // 2  # 50% overlap.
            # Prepare empty tensors for logits; here we assume output logits shape is same as segmentation model output.
            # For simplicity, we run tile inference and aggregate logits by averaging overlapping regions.
            final_logits = torch.zeros((B, model.segmentation_model.num_classes, H, W), device=device)
            count = torch.zeros((B, 1, H, W), device=device)
            
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    y1 = i
                    x1 = j
                    y2 = min(i + max_tile, H)
                    x2 = min(j + max_tile, W)
                    tile = image[:, :, y1:y2, x1:x2]
                    outputs = model(tile, return_all_outputs=True)
                    logits_tile = outputs['logits']
                    # Resize logits_tile to the tile size (if needed)
                    if logits_tile.shape[2] != (y2 - y1) or logits_tile.shape[3] != (x2 - x1):
                        logits_tile = F.interpolate(logits_tile, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
                    final_logits[:, :, y1:y2, x1:x2] += logits_tile
                    count[:, :, y1:y2, x1:x2] += 1
            final_logits = final_logits / count.clamp(min=1)
            pred = torch.argmax(final_logits, dim=1)
            # For energy, you might simply set it to the fused energy computed from the full image
            outputs = model(image, return_all_outputs=True)
            energy = outputs['energy']
            return {
                'prediction': pred.cpu(),
                'energy': energy.cpu(),
                'logits': final_logits.cpu()
            }
        else:
            outputs = model(image, return_all_outputs=True)
            logits = outputs['logits']
            pred = torch.argmax(logits, dim=1)
            energy = outputs['energy']
            return {
                'prediction': pred.cpu(),
                'energy': energy.cpu(),
                'logits': logits.cpu()
            }
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
        
        # Base segmentation network.
        self.segmentation_model = DeepWV3Plus(num_classes)
        
        # Enhanced Feature Projector: using a 3x3 convolution for spatial context.
        # Note: We assume the fallback intermediate features have 128 channels.
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
        
        # Energy head for OOD detection: expects 128-channel features.
        self.energy_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        # Learnable temperature parameter.
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
        # Free CUDA cache (optional).
        torch.cuda.empty_cache()
        
        # If batch size is 1, duplicate for BatchNorm.
        duplicated = False
        if x.size(0) == 1:
            x = x.repeat(2, 1, 1, 1)
            duplicated = True
        
        # Call segmentation model normally.
        logits = self.segmentation_model(x)
        # Manually extract intermediate features using known submodules.
        try:
            x1 = self.segmentation_model.mod1(x)
            m2 = self.segmentation_model.mod2(self.segmentation_model.pool2(x1))
            features = m2
        except Exception as e:
            features = logits  # fallback
        
        if duplicated:
            features = features[0:1]
            logits = logits[0:1]
        
        # Compute base energy from the energy head.
        energy = self.energy_head(features)
        
        # Project features via enhanced projector.
        proj_features = self.feature_projector(features)
        B, C, H, W = proj_features.shape
        proj_features = proj_features.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        flat_features = proj_features.reshape(-1, C)  # [B*H*W, C]
        
        # Normalize features.
        flat_features = F.normalize(flat_features, p=2, dim=1)
        
        # Dynamic temperature.
        beta = torch.exp(self.log_beta)
        
        # Multi-head attention for memory retrieval.
        if self.memory_initialized:
            multi_head_retrieved = []
            for head in self.attention_heads:
                head_features = head(flat_features)  # [N, feature_dim]
                sim = torch.mm(head_features, self.memory_bank.t())
                sim = sim * beta
                attn = torch.softmax(sim, dim=1)
                retrieved = torch.mm(attn, self.memory_bank)
                multi_head_retrieved.append(retrieved)
            # Average retrieval over heads.
            retrieved = torch.stack(multi_head_retrieved, dim=1).mean(dim=1)
            # Compute memory-based energy as negative cosine similarity.
            memory_energies = 1.0 - torch.sum(flat_features * retrieved, dim=1)
            memory_energies = memory_energies.view(B, H * W)
        else:
            memory_energies = torch.zeros(B, H * W, device=x.device)
        
        # Fusion of base energy and memory energy.
        # First, flatten energy from the energy head.
        base_energy = energy.view(B, H * W)  # [B, H*W]
        fusion_input = torch.cat([base_energy.unsqueeze(2), memory_energies.unsqueeze(2)], dim=2)  # [B, H*W, 2]
        combined_energy = self.energy_fusion(fusion_input).squeeze(2)  # [B, H*W]
        
        if return_all_outputs:
            return {
                'logits': logits,
                'energy': combined_energy,  # fused energy
                'features': flat_features,
                'memory_energies': memory_energies,
                'raw_features': features
            }
        else:
            return {
                'logits': logits,
                'energy': combined_energy,
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
            # Diversity-based sampling if memory is already initialized.
            if self.memory_initialized and features.shape[0] > 10:
                similarity = torch.mm(features, self.memory_bank.t())
                avg_sim = similarity.mean(dim=1)
                # Select features with lower similarity (more novel).
                k = min(features.shape[0], self.memory_size // 10)
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

    # Optional: Class-aware memory update.
    def update_memory_by_class(self, features, labels):
        """Update memory ensuring balanced representation across classes."""
        with torch.no_grad():
            if labels is None or features.shape[0] == 0:
                return
            unique_classes = torch.unique(labels)
            slots_per_class = self.memory_size // self.segmentation_model.num_classes
            for cls in unique_classes:
                if cls == 255:  # skip ignore index
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
    """Run inference on a single image using the model."""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image.unsqueeze(0), return_all_outputs=True)
        logits = outputs['logits']
        pred = torch.argmax(logits, dim=1)
        energy = outputs['energy']
        return {
            'prediction': pred.cpu(),
            'energy': energy.cpu(),
            'logits': logits.cpu()
        }
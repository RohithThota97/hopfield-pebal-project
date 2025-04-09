import torch
import torch.nn as nn
import torch.nn.functional as F

class HopfieldPEBALSegmentation(nn.Module):
    def __init__(self, num_classes=19, memory_size=1024, feature_dim=304, 
                 hopfield_beta=2.0, prototype_count=10):
        super(HopfieldPEBALSegmentation, self).__init__()
        # Import DeepWV3Plus using our helper function.
        from utils import import_deepwv3plus
        DeepWV3Plus = import_deepwv3plus()  # This imports from /home/ha51dybi/hop-pebal/code
        # Base segmentation network.
        self.segmentation_model = DeepWV3Plus(num_classes)
        
        # Feature projector for intermediate features.
        self.feature_dim = feature_dim
        self.feature_projector = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),  # Updated to match fallback feature channels.
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1)
        )
        
        # Energy head for OOD detection.
        self.energy_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),  # Updated to match fallback feature channels.
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        # Hopfield memory bank.
        self.memory_size = memory_size
        self.beta = hopfield_beta  # Temperature parameter.
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.memory_initialized = False
        
        # Prototype parameters.
        self.prototype_count = prototype_count
        self.register_buffer('class_prototypes', torch.zeros(num_classes, prototype_count, feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes, dtype=torch.long))
    
    def forward(self, x, return_all_outputs=False):
        # Empty CUDA cache to help mitigate fragmentation.
        torch.cuda.empty_cache()
        
        # If input has batch size 1, duplicate it so BatchNorm layers have enough samples.
        duplicated = False
        if x.size(0) == 1:
            x = x.repeat(2, 1, 1, 1)
            duplicated = True

        # Always call segmentation_model normally (without 'return_features')
        logits = self.segmentation_model(x)
        # Manually extract intermediate features using model submodules.
        try:
            # Assuming that mod1 and pool2/mod2 produce the intermediate features you need.
            x1 = self.segmentation_model.mod1(x)
            m2 = self.segmentation_model.mod2(self.segmentation_model.pool2(x1))
            features = m2
        except Exception as e:
            # Fallback: if extraction fails, use logits as features (not ideal but avoids error).
            features = logits

        # If duplicated, only retain the first sample.
        if duplicated:
            features = features[0:1]
            logits = logits[0:1]

        # Compute OOD energy using the energy head.
        energy = self.energy_head(features)
        
        # Project features for memory operations.
        proj_features = self.feature_projector(features)
        B, C, H, W = proj_features.shape
        # Reshape to [B, C, H*W] and then transpose to [B, H*W, C]
        proj_features = proj_features.view(B, C, -1).transpose(1, 2)
        flat_features = proj_features.reshape(-1, C)
        
        # Compute memory-based energy if memory is initialized.
        if self.memory_initialized:
            similarity = torch.mm(flat_features, self.memory_bank.t())
            similarity = similarity * self.beta
            attention = torch.softmax(similarity, dim=1)
            retrieved = torch.mm(attention, self.memory_bank)
            memory_energies = -torch.sum(flat_features * retrieved, dim=1)
            memory_energies = memory_energies.view(B, H * W)
        else:
            memory_energies = torch.zeros(B, H * W, device=x.device)
        
        if return_all_outputs:
            return {
                'logits': logits,
                'energy': energy.squeeze(1),
                'features': flat_features,
                'memory_energies': memory_energies,
                'raw_features': features
            }
        else:
            return {
                'logits': logits,
                'energy': energy.squeeze(1),
                'features': flat_features,
                'memory_energies': memory_energies
            }
    
    def update_memory(self, features, is_anomaly=None):
        """Update memory bank with new features."""
        with torch.no_grad():
            if is_anomaly is not None:
                features = features[~is_anomaly]
            if features.shape[0] == 0:
                return
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
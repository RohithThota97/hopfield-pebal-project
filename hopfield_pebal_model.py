import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import psutil
import time
from hopfield_layer import ModernHopfieldLayer

class MemoryTracker:
    """
    Utility class to track and report GPU and CPU memory usage.
    """
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

class HopfieldPEBALModel(nn.Module):
    def __init__(self, backbone, segmentation_head, num_classes=19, 
                 feature_dim=256, hopfield_beta=8.0, memory_size=1000, 
                 num_heads=4, insertion_point='after_backbone',
                 use_efficient_memory=True, chunk_size=1000,
                 target_feature_dim=304):  # Added target_feature_dim parameter
        """
        Insert Hopfield layer into PEBAL architecture with efficient memory usage
        
        Args:
            backbone: CNN backbone for feature extraction
            segmentation_head: Segmentation head for classification
            num_classes: Number of segmentation classes (including background)
            feature_dim: Dimension for Hopfield memory features
            hopfield_beta: Beta parameter for Hopfield energy calculation
            memory_size: Size of Hopfield memory bank
            num_heads: Number of attention heads in Hopfield layer
            insertion_point: Where to insert Hopfield layer ('after_backbone' or 'after_seghead')
            use_efficient_memory: Use memory-efficient techniques
            chunk_size: Size of chunks for processing large inputs
            target_feature_dim: Target feature dimension for segmentation head input
        """
        super(HopfieldPEBALModel, self).__init__()
        
        self.backbone = backbone
        self.segmentation_head = segmentation_head
        self.num_classes = num_classes
        self.insertion_point = insertion_point
        self.use_efficient_memory = use_efficient_memory
        self.chunk_size = chunk_size
        self.target_feature_dim = target_feature_dim
        
        # Memory tracker for monitoring memory usage
        self.memory_tracker = MemoryTracker()
        
        # Determine the input feature dimension dynamically
        self.input_dim = None
        self.detect_feature_dimensions()
        
        if self.input_dim is None:
            # Fallback dimensions if detection failed
            if insertion_point == 'after_backbone':
                self.input_dim = 4096  # Based on previous error message
            else:
                self.input_dim = 128  # Typical segmentation head output
        
        print(f"Detected input dimension: {self.input_dim}")
        
        # Add channel adapter to match expected dimensions
        self.needs_adapter = (self.input_dim != self.target_feature_dim)
        if self.needs_adapter:
            print(f"Adding channel adapter: {self.input_dim} -> {self.target_feature_dim}")
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(self.input_dim, self.target_feature_dim, kernel_size=1),
                nn.BatchNorm2d(self.target_feature_dim),
                nn.ReLU(inplace=False)  # Changed to non-inplace to avoid gradient errors
            )
        
        # Initialize input projection (using target_feature_dim if adapter is in use)
        projection_input_dim = self.target_feature_dim if self.needs_adapter else self.input_dim
        self.hopfield_input_proj = nn.Conv2d(projection_input_dim, feature_dim, kernel_size=1)
        
        # Initialize Hopfield layer
        self.hopfield = ModernHopfieldLayer(
            input_dim=feature_dim,
            output_dim=feature_dim,
            num_heads=num_heads,
            beta=hopfield_beta,
            memory_size=memory_size,
            update_memory=True
        )
        
        # Final classifier after Hopfield processing
        if insertion_point == 'after_backbone':
            # Project Hopfield output back to segmentation head input dimension
            self.hopfield_output_proj = nn.Conv2d(feature_dim, self.target_feature_dim, kernel_size=1)
        else:  # after_seghead
            # Direct classifier from Hopfield output to class logits
            self.final_classifier = nn.Conv2d(feature_dim, num_classes + 1, kernel_size=1)
            
        # Energy computation head
        self.energy_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=False),  # Changed to non-inplace
            nn.Conv2d(feature_dim // 2, 1, kernel_size=1)
        )
        
    def _initialize_weights(self):
        """Initialize model weights for better numerical stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
            # Kaiming initialization for Conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
            # Standard initialization for BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
            # Xavier initialization for Linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # Special initialization for Hopfield-specific layers
        if hasattr(self, 'hopfield'):
        # Initialize attention projection layers with smaller weights
            nn.init.normal_(self.hopfield.query_proj.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.hopfield.key_proj.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.hopfield.value_proj.weight, mean=0.0, std=0.01)
        
        # Initialize Hopfield memory with smaller values to prevent extreme attention values
            with torch.no_grad():
                self.hopfield.memory.data.normal_(mean=0.0, std=0.01)
    
    def detect_feature_dimensions(self):
        """Detect feature dimensions using a dummy forward pass"""
        try:
            # Create a small dummy input
            dummy_input = torch.zeros(1, 3, 64, 64)
            
            # Pass through backbone
            with torch.no_grad():
                backbone_features = self.backbone(dummy_input)
                
                if isinstance(backbone_features, tuple) or isinstance(backbone_features, list):
                    backbone_features = backbone_features[0]  # Take the first element if it's a tuple/list
                
                # Get the number of channels from the backbone output
                if self.insertion_point == 'after_backbone':
                    self.input_dim = backbone_features.shape[1]
                else:
                    # Pass through segmentation head to get its output dimension
                    seg_features = self.segmentation_head(backbone_features)
                    self.input_dim = seg_features.shape[1]
                
        except Exception as e:
            print(f"Error detecting feature dimensions: {e}")
            # Dimensions will be set to default values
    
    def forward(self, x):
        self.memory_tracker.log_memory_usage("Forward start")
        
        # Get features from backbone - divide into smaller chunks if memory is concern
        if self.use_efficient_memory and x.shape[2] * x.shape[3] > 128 * 256:
            try:
                # Sequential processing with explicit memory management
                features = self.backbone(x)
                self.memory_tracker.log_memory_usage("After backbone")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in backbone, trying sequential processing: {e}")
                    # Fallback approach with smaller input
                    x_small = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                    features = self.backbone(x_small)
                    features = F.interpolate(features, size=(x.shape[2]//4, x.shape[3]//4), 
                                           mode='bilinear', align_corners=False)
                else:
                    raise e
        else:
            features = self.backbone(x)
        
        # Log feature shape before adapter for debugging
        #print(f"Backbone output shape: {features.shape}")
        
        # Apply channel adapter if needed
        if self.needs_adapter:
            try:
                features = self.channel_adapter(features)
                #print(f"After adapter shape: {features.shape}")
            except RuntimeError as e:
                print(f"Error in channel adapter: {e}")
                raise e
        
        if self.insertion_point == 'after_backbone':
            # Insert Hopfield after backbone
            
            # Project features for Hopfield processing
            hopfield_input = self.hopfield_input_proj(features)
            self.memory_tracker.log_memory_usage("After projection")
            
            # Apply strided sampling if features are too large
            if hopfield_input.shape[2] * hopfield_input.shape[3] > 64 * 64 and self.use_efficient_memory:
                stride = max(1, min(4, hopfield_input.shape[2] // 32))
                hopfield_input_sampled = hopfield_input[:, :, ::stride, ::stride]
                self.memory_tracker.log_memory_usage("After strided sampling")
            else:
                hopfield_input_sampled = hopfield_input
            
            # Reshape for Hopfield processing
            b, c, h, w = hopfield_input_sampled.shape
            hopfield_input_flat = hopfield_input_sampled.view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]
            
            # Process in chunks if needed
            if hopfield_input_flat.shape[1] > self.chunk_size and self.use_efficient_memory:
                # Process in chunks to save memory
                retrieved_chunks = []
                energy_chunks = []
                
                for i in range(0, hopfield_input_flat.shape[1], self.chunk_size):
                    end = min(i + self.chunk_size, hopfield_input_flat.shape[1])
                    chunk = hopfield_input_flat[:, i:end, :]
                    
                    # Apply Hopfield association to chunk
                    chunk_retrieved, chunk_energy = self.hopfield(chunk)
                    
                    retrieved_chunks.append(chunk_retrieved)
                    energy_chunks.append(chunk_energy)
                    
                    # Clear cache after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Combine chunks
                retrieved = torch.cat(retrieved_chunks, dim=1)
                hopfield_energy = torch.cat(energy_chunks, dim=1)
                
                self.memory_tracker.log_memory_usage("After chunked Hopfield")
            else:
                # Apply Hopfield association
                retrieved, hopfield_energy = self.hopfield(hopfield_input_flat)
            
            # Reshape back to spatial dimensions
            retrieved = retrieved.permute(0, 2, 1).view(b, c, h, w)
            
            # Calculate additional energy from feature map
            feature_energy = self.energy_head(retrieved)
            
            # If we used strided sampling, need to upsample back to original size
            if hopfield_input_sampled.shape != hopfield_input.shape:
                retrieved = F.interpolate(retrieved, 
                                         size=hopfield_input.shape[2:], 
                                         mode='bilinear', 
                                         align_corners=False)
                feature_energy = F.interpolate(feature_energy, 
                                              size=hopfield_input.shape[2:], 
                                              mode='bilinear', 
                                              align_corners=False)
            
            # Project back to original feature dimensions
            hopfield_output = self.hopfield_output_proj(retrieved)
            
            # Add residual connection
            updated_features = features + hopfield_output
            
            # Clear intermediate variables to save memory
            if self.use_efficient_memory:
                del features, hopfield_input, retrieved, hopfield_output
                self.memory_tracker.clear_memory()
            
            # Continue with segmentation head
            logits = self.segmentation_head(updated_features)
            
            # Reshape energy to match spatial dimensions
            hopfield_energy = hopfield_energy.view(b, h, w)
            
        else:  # after_seghead
            # Get segmentation features
            seg_features = self.segmentation_head(features)
            
            # Clear backbone features to save memory
            if self.use_efficient_memory:
                del features
                self.memory_tracker.clear_memory()
            
            # Project seg_features for Hopfield processing
            hopfield_input = self.hopfield_input_proj(seg_features)
            
            # Apply strided sampling if features are too large
            if hopfield_input.shape[2] * hopfield_input.shape[3] > 64 * 64 and self.use_efficient_memory:
                stride = max(1, min(4, hopfield_input.shape[2] // 32))
                hopfield_input_sampled = hopfield_input[:, :, ::stride, ::stride]
            else:
                hopfield_input_sampled = hopfield_input
            
            # Reshape for Hopfield processing
            b, c, h, w = hopfield_input_sampled.shape
            hopfield_input_flat = hopfield_input_sampled.view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]
            
            # Process in chunks if needed
            if hopfield_input_flat.shape[1] > self.chunk_size and self.use_efficient_memory:
                # Process in chunks to save memory
                retrieved_chunks = []
                energy_chunks = []
                
                for i in range(0, hopfield_input_flat.shape[1], self.chunk_size):
                    end = min(i + self.chunk_size, hopfield_input_flat.shape[1])
                    chunk = hopfield_input_flat[:, i:end, :]
                    
                    # Apply Hopfield association to chunk
                    chunk_retrieved, chunk_energy = self.hopfield(chunk)
                    
                    retrieved_chunks.append(chunk_retrieved)
                    energy_chunks.append(chunk_energy)
                    
                    # Clear cache after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Combine chunks
                retrieved = torch.cat(retrieved_chunks, dim=1)
                hopfield_energy = torch.cat(energy_chunks, dim=1)
            else:
                # Apply Hopfield association
                retrieved, hopfield_energy = self.hopfield(hopfield_input_flat)
            
            # Reshape back to spatial dimensions
            retrieved = retrieved.permute(0, 2, 1).view(b, c, h, w)
            
            # Calculate additional energy from feature map
            feature_energy = self.energy_head(retrieved)
            
            # If we used strided sampling, need to upsample back to original size
            if hopfield_input_sampled.shape != hopfield_input.shape:
                retrieved = F.interpolate(retrieved, 
                                         size=hopfield_input.shape[2:], 
                                         mode='bilinear', 
                                         align_corners=False)
                feature_energy = F.interpolate(feature_energy, 
                                              size=hopfield_input.shape[2:], 
                                              mode='bilinear', 
                                              align_corners=False)
            
            # Clear intermediate variables to save memory
            if self.use_efficient_memory:
                del hopfield_input, hopfield_input_sampled
                self.memory_tracker.clear_memory()
            
            # Apply final classifier
            logits = self.final_classifier(retrieved)
            
            # Reshape energy to match spatial dimensions
            hopfield_energy = hopfield_energy.view(b, h, w)
        
        # Resize energy to match logits spatial dimensions if needed
        if feature_energy.shape[-2:] != logits.shape[-2:]:
            feature_energy = F.interpolate(
                feature_energy, 
                size=logits.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        if hopfield_energy.shape[-2:] != logits.shape[-2:]:
            hopfield_energy = F.interpolate(
                hopfield_energy.unsqueeze(1), 
                size=logits.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # Calculate PEBAL energy (log-sum-exp formulation)
        pebal_energy = -torch.logsumexp(logits[:, :-1], dim=1, keepdim=True)
        
        # Combined energy (weighted sum of different energy terms)
        combined_energy = pebal_energy + 0.5 * feature_energy + 0.5 * hopfield_energy.unsqueeze(1)
        
        combined_energy = torch.clamp(combined_energy, min=-100.0, max=100.0)

        if torch.isnan(hopfield_energy).any() or torch.isinf(hopfield_energy).any():
            print("Warning: NaN/Inf detected in hopfield_energy, replacing with zeros")
        hopfield_energy = torch.where(torch.isnan(hopfield_energy) | torch.isinf(hopfield_energy),
                                  torch.zeros_like(hopfield_energy),
                                  hopfield_energy)

        if torch.isnan(feature_energy).any() or torch.isinf(feature_energy).any():
            print("Warning: NaN/Inf detected in feature_energy, replacing with zeros")
        feature_energy = torch.where(torch.isnan(feature_energy) | torch.isinf(feature_energy),
                                 torch.zeros_like(feature_energy),
                                 feature_energy)

        if torch.isnan(pebal_energy).any() or torch.isinf(pebal_energy).any():
            print("Warning: NaN/Inf detected in pebal_energy, replacing with zeros")
            pebal_energy = torch.where(torch.isnan(pebal_energy) | torch.isinf(pebal_energy),
                              torch.zeros_like(pebal_energy),
                              pebal_energy)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/Inf detected in logits, replacing with zeros")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                         torch.zeros_like(logits),
                         logits)

        return {
            'logits': logits,
            'hopfield_energy': hopfield_energy.unsqueeze(1),
            'feature_energy': feature_energy,
            'pebal_energy': pebal_energy,
            'combined_energy': combined_energy
        }
    
    def update_memory(self, features):
        """
    Update the Hopfield memory with new features
    Args:
        features: Tensor of features to use for memory update [B, C, H, W]
    """
        with torch.no_grad():
        # Print feature shape for debugging
            #print(f"Feature shape for memory update: {features.shape}")
        
        # Apply channel adapter for memory update if needed
            if self.needs_adapter:
                features = self.channel_adapter(features)
                #print(f"Feature shape after adapter for memory update: {features.shape}")
        
            self.memory_tracker.log_memory_usage("Start memory update")
        
            if self.insertion_point == 'after_backbone':
            # Memory efficient projection
                if features.shape[2] * features.shape[3] > 128 * 128 and self.use_efficient_memory:
                # Process in smaller chunks or with stride
                    stride = max(1, min(4, features.shape[2] // 64))
                    hopfield_input = self.hopfield_input_proj(features)[:, :, ::stride, ::stride]
                else:
                    hopfield_input = self.hopfield_input_proj(features)
            else:
            # First get segmentation features
                if features.shape[2] * features.shape[3] > 128 * 128 and self.use_efficient_memory:
                # Process in smaller chunks or with stride
                    features_small = F.interpolate(features, scale_factor=0.5, mode='bilinear', align_corners=False)
                    seg_features = self.segmentation_head(features_small)
                    seg_features = F.interpolate(seg_features, size=(features.shape[2]//4, features.shape[3]//4), 
                                          mode='bilinear', align_corners=False)
                else:
                    seg_features = self.segmentation_head(features)
            
            # Then project
                if seg_features.shape[2] * seg_features.shape[3] > 128 * 128 and self.use_efficient_memory:
                    stride = max(1, min(4, seg_features.shape[2] // 64))
                    hopfield_input = self.hopfield_input_proj(seg_features)[:, :, ::stride, ::stride]
                else:
                    hopfield_input = self.hopfield_input_proj(seg_features)
        
        # Reshape for Hopfield processing
        b, c, h, w = hopfield_input.shape
        hopfield_input_flat = hopfield_input.view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]
        
        # Use a smaller subset for memory update to avoid memory issues
        # Combine batch and spatial dimensions
        flat_features = hopfield_input_flat.reshape(-1, c)
        
        # Sample a subset if too large
        if flat_features.shape[0] > self.hopfield.memory_size:
            subset_size = min(self.hopfield.memory_size, 1000)  # Limit to 1000 samples maximum
            indices = torch.randperm(flat_features.shape[0])[:subset_size]
            flat_features = flat_features[indices]
        
        # Update memory directly using the safe methods
        # First get the current memory
        current_memory = self.hopfield.get_memory()
        device = current_memory.device
        
        # Prepare indices for memory update
        memory_idx = torch.randperm(self.hopfield.memory_size)[:len(flat_features)]
        
        # Create a new memory
        new_memory = current_memory.cpu().clone()
        
        # Update the memory with the new features
        new_memory[memory_idx] = flat_features.detach().cpu()
        
        # Set the updated memory
        self.hopfield.set_memory(new_memory.to(device))
        
        self.memory_tracker.log_memory_usage("End memory update")
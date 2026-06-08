import os
import random
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Dict, Tuple, Optional, List

try:
    from model.network import Network
    from engine.engine import Engine
    from config.config import config
except ImportError as e:
    warnings.warn(f"External import failed: {e}. Using fallbacks.")
    # Dummy placeholders if import fails
    class Network(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.branch1 = nn.Identity()
    
    class Engine:
        def __init__(self, *args, **kwargs):
            pass
    
    # Mock config
    class MockConfig:
        pretrained_weight_path = None
    config = MockConfig()

# Add safe global for NumPy scalar to fix unpickling error
import torch.serialization
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        resize_resolution: Tuple[int, int] = (768, 1536),
        device: Optional[torch.device] = None,
        num_classes: int = 19,
        load_threshold: float = 0.9,
        amp: bool = False,
        verbose_logging: bool = True
    ):
        super(FeatureExtractor, self).__init__()
        self.load_threshold = load_threshold
        self.amp = amp
        self.verbose_logging = verbose_logging
        
        if not self.verbose_logging:
            logging.basicConfig(level=logging.WARNING)
        
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(resize_resolution, (list, tuple)) and len(resize_resolution) == 2:
            self.resize_resolution = tuple(int(x) for x in resize_resolution)
        else:
            self.resize_resolution = (768, 1536)
        
        self.num_classes = num_classes
        
        # Multi-scale feature extraction setup
        self.target_layers = ['mod3', 'mod4', 'mod5', 'mod6', 'aspp']
        self.layer_paths = {
            'mod3': 'mod3',
            'mod4': 'mod4',
            'mod5': 'mod5',
            'mod6': 'mod6',
            'aspp': 'aspp'
        }
        
        self.engine = self._setup_engine()
        model = self._load_model(model_path)
        self.add_module('model', model)
        self.model.eval()
        self.projection = None
        self.features = {}
        self._register_hooks()
        
        # Feature fusion modules
        self.feature_fusion = nn.ModuleDict({
            'mod3_proj': nn.Conv2d(256, 256, 1, bias=True),
            'mod4_proj': nn.Conv2d(512, 256, 1, bias=True),
            'mod5_proj': nn.Conv2d(1024, 256, 1, bias=True),
            'mod6_proj': nn.Conv2d(2048, 256, 1, bias=True),
            'aspp_proj': nn.Conv2d(1280, 256, 1, bias=True),
            'fusion': nn.Conv2d(256 * len(self.target_layers), 1280, 1, bias=True)
        }).to(self.device)
        
        # Attention block
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, 1280//16, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(1280//16, 1280, 1, bias=True),
            nn.Softplus()
        ).to(self.device)

    def eval(self):
        super().eval()
        self.model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # Keep backbone in eval mode even during training
        return self

    def to(self, device):
        super().to(device)
        self.device = device
        self.feature_fusion.to(device)
        self.attention.to(device)
        return self

    def _setup_engine(self) -> 'Engine':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('-l', '--local_rank', default=-1, type=int)
        parser.add_argument('-n', '--nodes', default=1, type=int)
        parser.add_argument('--ddp', action='store_true')
        args = parser.parse_args([])
        args.world_size = args.nodes * args.gpus
        return Engine(custom_arg=args, logger=None, continue_state_object=getattr(config, 'pretrained_weight_path', None))

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        try:
            model = Network(self.num_classes, wide=True)
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = self._extract_state_dict(checkpoint)
                filtered_dict = self._filter_state_dict_flexible(state_dict, model.branch1)
                missing, unexpected = model.branch1.load_state_dict(filtered_dict, strict=False)
                
                if self.verbose_logging:
                    logger.info(f"Missing keys: {missing}")
                    logger.info(f"Unexpected keys: {unexpected}")
                
                loaded_ratio = 1 - (len(missing) + len(unexpected)) / len(state_dict)
                if self.verbose_logging:
                    logger.info(f"Loaded {loaded_ratio*100:.1f}% of parameters successfully")
                
                if loaded_ratio < self.load_threshold:
                    raise ValueError("Low parameter load ratio; check checkpoint compatibility")
            
            dtype = torch.float32
            model = model.to(self.device, dtype=dtype)
            for param in model.parameters():
                param.requires_grad = False
            return model
        except Exception as e:
            logger.warning(f"Model load failed: {e}. Using dummy model.")
            return nn.Identity()

    def _extract_state_dict(self, checkpoint: Dict) -> Dict:
        if 'model' in checkpoint:
            return checkpoint['model']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        return checkpoint

    def _filter_state_dict_flexible(self, state_dict: Dict, target_module: nn.Module) -> Dict:
        filtered_dict = {}
        model_keys = set(target_module.state_dict().keys())
        loaded_keys = 0
        skipped_keys = []
        
        for key, value in state_dict.items():
            clean_key = key.replace('module.', '')
            if clean_key.startswith('criterion.'):
                skipped_keys.append(clean_key)
                continue
            
            target_key = clean_key
            if target_key in model_keys and value.shape == target_module.state_dict()[target_key].shape:
                filtered_dict[target_key] = value.to(dtype=torch.float32)
                loaded_keys += 1
        
        if self.verbose_logging:
            logger.info(f"Loaded {loaded_keys}/{len(model_keys)} model parameters flexibly")
            if skipped_keys:
                logger.warning(f"Skipped keys: {skipped_keys}")
        
        if loaded_keys == 0:
            logger.warning("No parameters loaded from checkpoint")
        
        return filtered_dict

    def _hook_fn(self, layer_name: str):
        def hook(module, input, output):
            self.features[layer_name] = output.to(dtype=torch.float32)
        return hook

    def _register_hooks(self):
        model_module = self.model.module.branch1 if isinstance(self.model, nn.DataParallel) else self.model.branch1
        for layer_name, path in self.layer_paths.items():
            current_module = model_module
            parts = path.split('.')
            for part in parts[:-1]:
                current_module = getattr(current_module, part)
            target_module = getattr(current_module, parts[-1])
            target_module.register_forward_hook(self._hook_fn(layer_name))

    def _validate_labels(self, labels: torch.Tensor, num_classes: int = 19) -> torch.Tensor:
        """FIXED: Simplified validation that preserves OOD labels during evaluation"""
        if labels is None:
            raise ValueError("Labels cannot be None")
        
        if labels.dim() not in [3, 4]:
            raise ValueError(f"Labels must have 3 or 4 dimensions, got {labels.dim()}")
        
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        if not labels.dtype == torch.long:
            if self.verbose_logging:
                logger.warning(f"Converting labels from {labels.dtype} to torch.long")
            labels = labels.long()
        
        # During evaluation, preserve all labels including OOD (254)
        if not self.training:
            return labels
            
        # Only validate during training
        valid_mask = (labels >= 0) & (labels < num_classes)
        ood_mask = (labels == 254)
        ignore_mask = (labels == 255)
        invalid_mask = ~(valid_mask | ood_mask | ignore_mask)
        invalid_count = invalid_mask.sum().item()
        total_pixels = labels.numel()
        
        if invalid_count > 0:
            if self.verbose_logging:
                logger.warning(f"Found {invalid_count} invalid label values ({invalid_count/total_pixels:.2%})")
                logger.warning(f"Invalid indices: {torch.unique(labels[invalid_mask])}")
            labels[invalid_mask] = 255
        
        if invalid_count / total_pixels > 0.5:
            raise ValueError("Too many invalid labels—check data")
        
        return labels

    def _validate_input_tensor(self, tensor: torch.Tensor, name: str = "input") -> torch.Tensor:
        if tensor is None:
            raise ValueError(f"{name} tensor cannot be None")
        
        if tensor.dim() not in [3, 4]:
            raise ValueError(f"{name} must have 3 or 4 dimensions, got {tensor.dim()}")
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        if name == "image" and tensor.shape[1] not in [1, 3]:
            raise ValueError(f"Image must have 1 or 3 channels, got {tensor.shape[1]}")
        
        if name == "image" and tensor.shape[1] == 1:
            if self.verbose_logging:
                logger.warning(f"Grayscale image detected, repeating to 3 channels")
            tensor = tensor.repeat(1, 3, 1, 1)
        
        if torch.isnan(tensor).any():
            if self.verbose_logging:
                logger.warning(f"{name} contains NaN values")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        
        if torch.isinf(tensor).any():
            if self.verbose_logging:
                logger.warning(f"{name} contains inf values")
            tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
        
        dtype = torch.float32
        if tensor.dtype != dtype:
            if name == "image":
                if tensor.dtype == torch.uint8:
                    tensor = tensor.to(dtype) / 255.0
                else:
                    tensor = tensor.to(dtype)
        
        return tensor

    def resize_images(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if not isinstance(self.resize_resolution, (tuple, list)) or len(self.resize_resolution) != 2:
            self.resize_resolution = (768, 1536)
        
        target_h, target_w = self.resize_resolution
        images = self._validate_input_tensor(images, "image")
        images = images.to(self.device)
        
        # Direct resize without any cropping or padding
        images = F.interpolate(
            images,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        if labels is not None:
            labels = labels.to(self.device)
            
            # Handle label dimensions
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            elif labels.dim() == 2:
                labels = labels.unsqueeze(0)
            
            # Preserve OOD labels during resize
            ood_mask_orig = (labels == 254).float()
            
            # Resize labels using nearest neighbor to preserve discrete values
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(target_h, target_w),
                mode='nearest'
            ).squeeze(1).long()
            
            # Restore OOD pixels that might have been lost during resize
            ood_mask_resized = F.interpolate(
                ood_mask_orig.unsqueeze(1),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Use threshold 0.5 to preserve OOD pixels
            labels_resized[ood_mask_resized > 0.5] = 254
            
            labels = self._validate_labels(labels_resized, self.num_classes)
        
        return images, labels

    def forward(self, images: torch.Tensor, domain_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = torch.cuda.amp.autocast(enabled=self.amp) if self.device.type == 'cuda' else torch.no_grad()
        with context:
            try:
                self.features.clear()
                _ = self.model(images)
                
                # Multi-scale feature fusion
                if 'aspp' in self.features:
                    target_size = self.features['aspp'].shape[-2:]
                else:
                    target_size = (images.shape[2] // 8, images.shape[3] // 8)
                
                # Project and resize all features to same resolution
                fused_features = []
                if 'mod3' in self.features:
                    f3 = self.feature_fusion['mod3_proj'](self.features['mod3'])
                    f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=True)
                    fused_features.append(f3)
                if 'mod4' in self.features:
                    f4 = self.feature_fusion['mod4_proj'](self.features['mod4'])
                    f4 = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=True)
                    fused_features.append(f4)
                if 'mod5' in self.features:
                    f5 = self.feature_fusion['mod5_proj'](self.features['mod5'])
                    f5 = F.interpolate(f5, size=target_size, mode='bilinear', align_corners=True)
                    fused_features.append(f5)
                if 'mod6' in self.features:
                    f6 = self.feature_fusion['mod6_proj'](self.features['mod6'])
                    f6 = F.interpolate(f6, size=target_size, mode='bilinear', align_corners=True)
                    fused_features.append(f6)
                if 'aspp' in self.features:
                    fa = self.feature_fusion['aspp_proj'](self.features['aspp'])
                    fused_features.append(fa)
                
                # FPN-style fusion with lateral connections
                if len(fused_features) > 0:
                    final_features = torch.cat(fused_features, dim=1)
                    final_features = self.feature_fusion['fusion'](final_features)
                else:
                    fallback_shape = (images.shape[0], 1280, target_size[0], target_size[1])
                    final_features = torch.zeros(fallback_shape, device=self.device, dtype=torch.float32)
                
                # Apply attention
                attn = self.attention(final_features)
                final_features = final_features * attn
                
                return final_features
            except Exception as e:
                import traceback
                logger.error(f"Error in forward: {e}")
                logger.error(traceback.format_exc())
                fallback_shape = (images.shape[0], 1280, images.shape[2] // 8, images.shape[3] // 8)
                fallback_features = torch.zeros(fallback_shape, device=self.device, dtype=torch.float32)
                return fallback_features

    def align_labels_to_features(self, labels: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """CRITICAL FIX: Resize labels to match feature map resolution"""
        if labels is None:
            return None
            
        B, C, H, W = features.shape
        
        # Handle different label input formats
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1).float()
        elif labels.dim() == 4:
            if labels.shape[1] != 1:
                labels = labels[:, 0:1].float()
            else:
                labels = labels.float()
        
        # Preserve OOD information during resize
        ood_mask = (labels == 254).float()
        
        # Resize to feature resolution using nearest neighbor
        labels_resized = F.interpolate(
            labels,
            size=(H, W),
            mode='nearest'
        ).squeeze(1).long()
        
        # Restore OOD pixels that might have been lost during resize
        if ood_mask.sum() > 0:
            ood_mask_resized = F.interpolate(
                ood_mask,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Use threshold 0.5 to preserve OOD pixels
            labels_resized[ood_mask_resized > 0.5] = 254
        
        return labels_resized

    def extract_features_batch(self, batch) -> Dict[str, torch.Tensor]:
        """CRITICAL FIX: Properly handle label resizing to match feature dimensions"""
        was_training = self.training
        self.eval()
        
        try:
            # Extract data from batch
            if isinstance(batch, dict):
                images_raw = batch.get('data')
                labels_raw = batch.get('label')
                is_ood = batch.get('is_ood', torch.zeros(images_raw.shape[0], dtype=torch.bool) if images_raw is not None else None)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images_raw = batch[0]
                labels_raw = batch[1] if len(batch) > 1 else None
                is_ood = torch.zeros(images_raw.shape[0], dtype=torch.bool) if images_raw is not None else None
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            if images_raw is None:
                logger.warning("No image data in batch")
                return {}
            if images_raw.shape[0] == 0:
                return {}

            images_raw = self._validate_input_tensor(images_raw, "image")
            images_raw = images_raw.to(self.device)
            is_ood = is_ood.to(self.device) if is_ood is not None else None

            # Only validate labels if they exist
            if labels_raw is not None:
                labels_raw = labels_raw.to(self.device)

            # Resize images to target resolution
            images, labels_at_image_res = self.resize_images(images_raw, labels_raw)
            
            # Extract features
            raw_features = self.forward(images)

            # CRITICAL FIX: Align labels with feature dimensions
            if labels_at_image_res is not None:
                labels = self.align_labels_to_features(labels_at_image_res, raw_features)
                
                # Create masks at feature resolution
                ood_mask = (labels == 254).float()
                ignore_mask = (labels == 255).float()
                is_ood = (ood_mask.sum(dim=[1, 2]) > 0).bool()
            else:
                # No labels - create dummy masks at feature resolution
                feat_h, feat_w = raw_features.shape[-2:]
                batch_size = raw_features.shape[0]
                labels = None
                ood_mask = torch.zeros(batch_size, feat_h, feat_w, device=self.device)
                ignore_mask = torch.zeros(batch_size, feat_h, feat_w, device=self.device)
                is_ood = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            # Feature statistics for OOD awareness
            feature_stats = {
                'feature_mean': raw_features.mean().item(),
                'feature_std': raw_features.std().item(),
                'feature_max': raw_features.max(dim=3)[0].max(dim=2)[0],
                'feature_min': raw_features.min(dim=3)[0].min(dim=2)[0]
            }
            
            if self.verbose_logging:
                logger.info(f"Feature stats: mean={feature_stats['feature_mean']:.4f}, std={feature_stats['feature_std']:.4f}")
                if labels is not None:
                    logger.info(f"Label shape: {labels.shape}, Feature shape: {raw_features.shape}")
                    logger.info(f"OOD pixels: {ood_mask.sum().item()}, Ignore pixels: {ignore_mask.sum().item()}")

            return {
                'images': images,
                'labels': labels,  # Labels aligned to feature resolution
                'features': raw_features,
                'ood_mask': ood_mask,
                'ignore_mask': ignore_mask,
                'is_ood': is_ood,
                **feature_stats
            }
            
        except Exception as e:
            logger.error(f"Error in extract_features_batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if isinstance(batch, dict) and 'fn' in batch:
                if self.verbose_logging:
                    logger.info(f"Possibly bad files: {batch['fn']}")
            
            # Return valid fallback with consistent dimensions
            fallback_feat_shape = (1, 1280, 96, 192)  # For (768, 1536) input
            fallback_mask_shape = (1, 96, 192)
            
            return {
                'images': torch.zeros((1, 3, 768, 1536), device=self.device, dtype=torch.float32),
                'labels': None,
                'features': torch.zeros(fallback_feat_shape, device=self.device, dtype=torch.float32),
                'ood_mask': torch.zeros(fallback_mask_shape, device=self.device, dtype=torch.float32),
                'ignore_mask': torch.zeros(fallback_mask_shape, device=self.device, dtype=torch.float32),
                'is_ood': torch.zeros(1, dtype=torch.bool, device=self.device)
            }
        finally:
            if was_training:
                self.train()
#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import gc
import numpy as np
from torch.utils.data import DataLoader, Dataset  # Added Dataset import
import logging
import time
import wandb
import math
import torch.cuda.amp as amp
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Any, Dict, Tuple
import traceback
import sklearn.metrics  # FIXED: Added missing sklearn import
import json  # FIXED: Added missing json import

# Imports from provided programs
from segmentation_head import SegmentationClassifierHead
from feature_extractor import FeatureExtractor
from projection_head import SimpleProjectionHead
from hopfield_memory_builder import MemoryBuilder
from hopfield_weight_updater import HopfieldBoostingManager
from abc import ABC, abstractmethod
import matplotlib.cm as cm
from torch.cuda.amp import autocast
from pixel_energy import compute_hopfield_ood_loss, PixelWiseBorderEnergy, PixelWiseInferenceScore, lse
from dataset.data_loader import Fishyscapes,LostAndFound,RoadAnomaly,Cityscapes,CityscapesCocoMix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (unchanged)
CITYSCAPES_COLORMAP = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

CITYSCAPES_CLASSES = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

class PILToTensorTransform:
    """Convert PIL Images to tensors with normalization"""
    def __init__(self, target_size=(512, 1024)):
        self.target_size = target_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, target=None):
        # Convert image
        if isinstance(image, Image.Image):
            image = transforms.Resize(self.target_size, interpolation=InterpolationMode.BILINEAR)(image)
            image = transforms.ToTensor()(image)
            image = self.normalize(image)
        
        if target is not None and isinstance(target, Image.Image):
            target = transforms.Resize(self.target_size, interpolation=InterpolationMode.NEAREST)(target)
            target = torch.tensor(np.array(target), dtype=torch.long)
        
        return image, target

class ImprovedOODSegmentationTrainer:
    def __init__(self, config_dict, train_loader, val_loader, fixed_batches=None, resume_from=None):
        self.config = config_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fixed_batches = fixed_batches or []
        self.resume_from = resume_from
        
        # Training parameters
        self.total_epochs = 50
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints_improved")
        self.beta = 128.0
        self.lambda_ood = 5.0
        self.memory_subsample = 100000
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize wandb
        wandb.init(
            project="ood-seg-improved",
            config=self.config,
            name="frozen-backbone-fixed-hopfield-pixelwise",
            mode="online"
        )

        # Initialize models
        logger.info("Initializing models...")
        self._init_models()

        # Build or load memories
        if self.resume_from and os.path.exists(self.resume_from):
            self._load_checkpoint(self.resume_from)
        else:
            self._build_initial_memories()
        
        self._init_training_components()

        # Mixed precision scaler
        self.scaler = amp.GradScaler()
        self.best_val_miou = 0.0
        self.best_fpr95 = 1.0
        self.patience = 100
        self.patience_counter = 0
        self.global_step = 0
        self.accum_steps = 1
        self.warmup_steps = 500

        # Performance monitoring
        self.ema_alpha = 0.99
        self.ema_seg_loss = 0.0
        self.ema_ood_loss = 0.0
        self.ema_total_loss = 0.0
        self.grad_norm_history_seg = []
        self.grad_norm_history_proj = []
        self.memory_eval_interval = 5

        # Memory and time tracking
        self.max_memory_threshold = 0.8 * torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        self.memory_metrics = {}
        self.time_metrics = {}

    def _load_checkpoint(self, path):
        """Load checkpoint and move memories to GPU"""
        checkpoint = torch.load(path, map_location='cpu')
        self.segmentation_head.load_state_dict(checkpoint['segmentation_head_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        self.optimizer_seg.load_state_dict(checkpoint['optimizer_seg_state_dict'])
        self.optimizer_proj.load_state_dict(checkpoint['optimizer_proj_state_dict'])
        self.id_memory = checkpoint['id_memory'].to(self.device).float()
        self.aux_memory = checkpoint['aux_memory'].to(self.device).float()
        self.global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
        self.best_fpr95 = checkpoint.get('best_fpr95', 1.0)
        logger.info(f"Resumed from epoch {start_epoch}, global_step {self.global_step}")
        self._init_training_components()
        return start_epoch

    def _init_models(self):
        """Initialize models with fully frozen backbone."""
        self.feature_extractor = FeatureExtractor(
            model_path=self.config['model_path'],
            device=self.device,
            num_classes=self.config['num_classes'],
        ).to(self.device)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor.train()
        logger.info("Backbone fully frozen.")
        
        self.segmentation_head = SegmentationClassifierHead(
            1280, self.config['num_classes']
        ).to(self.device)
        
        self.projection_head = SimpleProjectionHead(
            input_dim=1280, output_dim=128
        ).to(self.device)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights carefully to prevent gradient explosion"""
        for module in [self.segmentation_head, self.projection_head]:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if isinstance(m, nn.Conv2d):
                        nn.init.orthogonal_(m.weight)
                    else:
                        nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _build_initial_memories(self):
        """Build initial Hopfield memories"""
        logger.info("Building initial memories...")
        memory_builder = MemoryBuilder(
            feature_extractor=self.feature_extractor,
            projection_pipeline=self.projection_head,
            device=self.device,
            id_memory_size=100000,
            aux_memory_size=100000,
            num_in_dist_classes=self.config['num_classes'],
            ood_label=254,
        )
        
        id_memory, aux_memory, warnings = memory_builder.process_images(self.train_loader)
        if warnings:
            logger.warning(f"Memory building warnings: {warnings}")
        
        self.id_memory = id_memory.to(self.device).float()
        self.aux_memory = aux_memory.to(self.device).float()
        logger.info(f"Initial memories: ID={self.id_memory.shape}, AUX={self.aux_memory.shape}")

    def _init_training_components(self):
        """Initialize training components with higher LR for projection head"""
        base_lr = 5e-5
        
        self.optimizer_seg = torch.optim.AdamW(
            self.segmentation_head.parameters(),
            lr=base_lr,
            weight_decay=5e-4,
            eps=1e-6,
            betas=(0.9, 0.999)
        )
        
        self.optimizer_proj = torch.optim.AdamW(
            self.projection_head.parameters(),
            lr=base_lr * 10,
            weight_decay=5e-4,
            eps=1e-6
        )

        self.scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_seg, mode='min', factor=0.5, patience=3, min_lr=1e-7, verbose=True
        )
        
        self.scheduler_proj = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_proj, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )

        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=255, reduction='mean'
        )

        # Initialize Hopfield manager
        self.hopfield_manager = HopfieldBoostingManager(
            id_features_full=self.id_memory,
            aux_features_full=self.aux_memory,
            beta_sampling=self.beta,
            lambda_ood=self.lambda_ood,
            device=self.device,
            memory_subset_size=min(10000, self.memory_subsample * self.config['num_classes']),
            positive_shift=False,
            num_boosting_iters=5
        )

    def _prepare_batch(self, batch):
        """Prepare batch for GPU"""
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == 'label':
                    batch_gpu[k] = v.to(self.device).long()
                else:
                    batch_gpu[k] = v.to(self.device).float()
        return batch_gpu

    def _compute_losses(self, batch):
        """Fixed loss computation with comprehensive stability checks"""
        batch_gpu = self._prepare_batch(batch)
        
        # Extract features
        extracted = self.feature_extractor.extract_features_batch(batch_gpu)
        features = extracted['features'].float()
        labels = extracted['labels']

        # Validate and fix labels
        if labels is not None:
            labels = labels.clone()
            valid_mask = (labels >= 0) & (labels < self.config['num_classes'])
            ood_mask = (labels == 254)
            ignore_mask = (labels == 255)
            invalid_mask = ~(valid_mask | ood_mask | ignore_mask)
            
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} invalid labels, setting to ignore")
                labels[invalid_mask] = 255

        # Segmentation loss with stability
        seg_loss = torch.tensor(0.0, device=self.device)
        
        with torch.cuda.amp.autocast():
            seg_logits = self.segmentation_head(features)
            seg_logits = torch.clamp(seg_logits, min=-10, max=10)
            
            if seg_logits.shape[-2:] != labels.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=True
                )

            # Prepare labels for CE loss
            labels_for_ce = labels.clone()
            labels_for_ce[labels == 254] = 255
            labels_for_ce[labels >= self.config['num_classes']] = 255

            # Compute CE loss with numerical stability
            ce_unreduced = F.cross_entropy(
                seg_logits, labels_for_ce, ignore_index=255, reduction='none'
            )

            # Filter out inf/nan values
            valid_loss_mask = torch.isfinite(ce_unreduced)
            if not valid_loss_mask.all():
                logger.warning(f"Inf/NaN in {(~valid_loss_mask).sum()} pixels")
            
            ce_unreduced = torch.where(
                valid_loss_mask, ce_unreduced, torch.zeros_like(ce_unreduced)
            )

            # Take mean only over valid pixels
            if valid_loss_mask.any():
                seg_loss = ce_unreduced[valid_loss_mask].mean()
            else:
                seg_loss = torch.tensor(0.0, device=self.device)

            # Final stability check
            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                logger.warning("Loss is NaN/Inf, setting to 0")
                seg_loss = torch.tensor(0.0, device=self.device)

        # OOD loss computation with stability
        ood_loss = torch.tensor(0.0, device=self.device)
        
        with torch.cuda.amp.autocast():
            projected = self.projection_head(features)
            projected = torch.clamp(projected, min=-10, max=10)
            
            B, C, H_feat, W_feat = projected.shape
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(), size=(H_feat, W_feat), mode='nearest'
            ).squeeze(1).long()
            
            pixel_features = projected.permute(0, 2, 3, 1).reshape(-1, C)
            pixel_labels = labels_resized.view(-1)
            
            valid_mask = (pixel_labels != 255)
            
            if valid_mask.any():
                valid_pixels = pixel_features[valid_mask]
                valid_labels = pixel_labels[valid_mask]
                
                id_mask = (valid_labels < self.config['num_classes'])
                ood_mask = (valid_labels == 254)
                
                id_pixels = valid_pixels[id_mask] if id_mask.any() else torch.empty(0, C, device=self.device)
                ood_pixels = valid_pixels[ood_mask] if ood_mask.any() else torch.empty(0, C, device=self.device)

                if id_mask.any():
                    num_to_sample = min(128, len(valid_pixels))
                    id_batch, aux_batch = self.hopfield_manager.sample_batch(num_to_sample)
                    
                    boosted_id = torch.cat([id_pixels, id_batch.to(self.device).float()]) if len(id_batch) > 0 else id_pixels
                    boosted_ood = torch.cat([ood_pixels, aux_batch.to(self.device).float()]) if len(aux_batch) > 0 else ood_pixels
                    
                    if len(boosted_id) > 0 and len(boosted_ood) > 0:
                        raw_ood_loss = self.hopfield_manager.compute_boosted_ood_loss(boosted_ood, boosted_id)
                        ood_loss = torch.clamp(raw_ood_loss, min=-20.0, max=20.0)
                        
                        if torch.isnan(ood_loss) or torch.isinf(ood_loss):
                            logger.warning("OOD loss is NaN/Inf, setting to 0")
                            ood_loss = torch.tensor(0.0, device=self.device)
                    else:
                        ood_loss = torch.tensor(0.0, device=self.device)

        # Add small L2 reg loss to always update projection head
        reg_loss = 0.0
        for param in self.projection_head.parameters():
            if param.requires_grad:
                reg_loss += torch.norm(param) * 1e-5
        
        ood_loss += reg_loss

        return {
            'seg_loss': seg_loss,
            'ood_loss': ood_loss,
            'has_id': (labels_for_ce < 255).any() if labels is not None else False,
        }

    def _train_epoch(self, epoch):
        """Train with weight updates every epoch"""
        self.hopfield_manager.update_sampling_weights(memory_size=self.memory_subsample)
        self.segmentation_head.train()
        self.projection_head.train()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_metrics = {
            'seg_losses': [],
            'ood_losses': [],
        }
        
        accum_count = 0
        epoch_grad_norms_seg = []
        epoch_grad_norms_proj = []

        # Time tracking
        start_time = time.time()
        preprocess_time = 0
        extract_time = 0
        project_time = 0
        energy_time = 0
        post_time = 0

        for batch_idx, batch in enumerate(progress_bar):
            batch_start = time.time()
            
            if self.global_step < self.warmup_steps:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
                for pg in self.optimizer_seg.param_groups:
                    pg['lr'] = lr_scale * 5e-5
                for pg in self.optimizer_proj.param_groups:
                    pg['lr'] = lr_scale * 5e-5

            # Check memory constraint before processing
            current_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            if current_mem > self.max_memory_threshold:
                logger.warning(f"Memory exceeded threshold, skipping batch {batch_idx}")
                continue

            try:
                preprocess_end = time.time()
                preprocess_time += (preprocess_end - batch_start)
                
                loss_dict = self._compute_losses(batch)
                
                extract_end = time.time()
                extract_time += (extract_end - preprocess_end)
            except RuntimeError as e:
                logger.warning(f"Error computing losses at batch {batch_idx}: {e}")
                continue

            seg_loss = loss_dict['seg_loss'] / self.accum_steps
            ood_loss = loss_dict['ood_loss'] / self.accum_steps

            if accum_count == 0:
                self.optimizer_seg.zero_grad(set_to_none=True)
                self.optimizer_proj.zero_grad(set_to_none=True)

            try:
                self.scaler.scale(seg_loss).backward(retain_graph=True)
                proj_loss = self.lambda_ood * ood_loss
                self.scaler.scale(proj_loss).backward()
            except RuntimeError as e:
                logger.warning(f"Backward error at batch {batch_idx}: {e}")
                self.optimizer_seg.zero_grad(set_to_none=True)
                self.optimizer_proj.zero_grad(set_to_none=True)
                continue

            accum_count += 1

            if accum_count == self.accum_steps or batch_idx == len(self.train_loader) - 1:
                # Gradient clipping
                for param_group in [self.segmentation_head.parameters(), self.projection_head.parameters()]:
                    for param in param_group:
                        if param.grad is not None:
                            param.grad.data = torch.clamp(param.grad.data, -5.0, 5.0)

                # Unscale and clip gradients
                self.scaler.unscale_(self.optimizer_seg)
                grad_norm_seg = torch.nn.utils.clip_grad_norm_(
                    self.segmentation_head.parameters(), max_norm=2.0
                )

                self.scaler.unscale_(self.optimizer_proj)
                grad_norm_proj = torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(), max_norm=2.0
                )

                epoch_grad_norms_seg.append(grad_norm_seg.item())
                epoch_grad_norms_proj.append(grad_norm_proj.item())

                # Step optimizers with gradient norm checks
                if grad_norm_seg <= 10.0:
                    self.scaler.step(self.optimizer_seg)
                else:
                    logger.warning(f"Skipping seg update due to large gradients: {grad_norm_seg}")
                    self.optimizer_seg.zero_grad(set_to_none=True)

                if grad_norm_proj <= 10.0:
                    self.scaler.step(self.optimizer_proj)
                else:
                    logger.warning(f"Skipping proj update due to large gradients: {grad_norm_proj}")
                    self.optimizer_proj.zero_grad(set_to_none=True)

                self.scaler.update()
                accum_count = 0

            project_end = time.time()
            project_time += (project_end - extract_end)
            
            energy_end = time.time()
            energy_time += (energy_end - project_end)

            epoch_metrics['seg_losses'].append(loss_dict['seg_loss'].item())
            epoch_metrics['ood_losses'].append(loss_dict['ood_loss'].item())

            progress_bar.set_postfix({
                'Seg': f"{loss_dict['seg_loss'].item():.4f}",
                'OOD': f"{loss_dict['ood_loss'].item():.4f}",
            })

            if batch_idx % 10 == 0:
                self._debug_loss_values(loss_dict['seg_loss'], loss_dict['ood_loss'], batch_idx)

            log_dict = {
                'seg_loss': loss_dict['seg_loss'].item(),
                'ood_loss': loss_dict['ood_loss'].item(),
                'lr_seg': self.optimizer_seg.param_groups[0]['lr'],
                'lr_proj': self.optimizer_proj.param_groups[0]['lr'],
                'global_step': self.global_step
            }
            wandb.log(log_dict)
            self.global_step += 1

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            post_end = time.time()
            post_time += (post_end - energy_end)

        total_time = time.time() - start_time
        avg_seg_loss = np.mean(epoch_metrics['seg_losses']) if epoch_metrics['seg_losses'] else 0.0
        avg_ood_loss = np.mean(epoch_metrics['ood_losses']) if epoch_metrics['ood_losses'] else 0.0

        # Update EMAs
        self.ema_seg_loss = self.ema_alpha * self.ema_seg_loss + (1 - self.ema_alpha) * avg_seg_loss if self.global_step > 0 else avg_seg_loss
        self.ema_ood_loss = self.ema_alpha * self.ema_ood_loss + (1 - self.ema_alpha) * avg_ood_loss if self.global_step > 0 else avg_ood_loss
        total_loss = avg_seg_loss + self.lambda_ood * avg_ood_loss
        self.ema_total_loss = self.ema_alpha * self.ema_total_loss + (1 - self.ema_alpha) * total_loss if self.global_step > 0 else total_loss

        # Log EMAs to wandb
        wandb.log({
            'ema_seg_loss': self.ema_seg_loss,
            'ema_ood_loss': self.ema_ood_loss,
            'ema_total_loss': self.ema_total_loss
        })

        # Compute and log average grad norms
        avg_grad_norm_seg = np.mean(epoch_grad_norms_seg) if epoch_grad_norms_seg else 0.0
        avg_grad_norm_proj = np.mean(epoch_grad_norms_proj) if epoch_grad_norms_proj else 0.0
        self.grad_norm_history_seg.append(avg_grad_norm_seg)
        self.grad_norm_history_proj.append(avg_grad_norm_proj)

        wandb.log({
            'avg_grad_norm_seg': avg_grad_norm_seg,
            'avg_grad_norm_proj': avg_grad_norm_proj
        })

        # Compute memory metrics
        self.memory_metrics = self._compute_memory_usage()

        # Compute time metrics (average per batch)
        num_batches = len(self.train_loader)
        self.time_metrics = {
            'preprocessing': (preprocess_time / num_batches) * 1000,  # ms
            'feature_extraction': (extract_time / num_batches) * 1000,
            'projection': (project_time / num_batches) * 1000,
            'energy_computation': (energy_time / num_batches) * 1000,
            'post_processing': (post_time / num_batches) * 1000,
            'total': (total_time / num_batches) * 1000,
            'fps': num_batches / total_time if total_time > 0 else 0
        }

        return {
            'avg_seg_loss': avg_seg_loss,
            'avg_ood_loss': avg_ood_loss,
        }

    def _compute_memory_usage(self):
        """FIXED: Compute memory usage for components safely"""
        def get_param_count(model):
            return sum(p.numel() for p in model.parameters())

        def bytes_to_mb(bytes_size):
            return bytes_size / (1024 * 1024)

        def get_model_memory_mb(model):
            """Safely compute model memory usage"""
            total_params = get_param_count(model)
            if total_params == 0:
                return 0.0
            
            # Get first parameter to determine element size
            first_param = next(iter(model.parameters()), None)
            if first_param is None:
                return 0.0
            
            element_size = first_param.element_size()
            return bytes_to_mb(total_params * element_size)

        mem_dict = {}

        # Model components - use safe memory calculation
        mem_dict['feature_extractor'] = {
            'params': get_param_count(self.feature_extractor), 
            'memory_mb': get_model_memory_mb(self.feature_extractor)
        }
        
        mem_dict['projection_head'] = {
            'params': get_param_count(self.projection_head), 
            'memory_mb': get_model_memory_mb(self.projection_head)
        }
        
        mem_dict['segmentation_head'] = {
            'params': get_param_count(self.segmentation_head), 
            'memory_mb': get_model_memory_mb(self.segmentation_head)
        }

        # Memories (assuming float32, 4 bytes/element)
        mem_size = self.memory_subsample * 128 * 4  # dims=128
        mem_dict['id_memory'] = {'params': self.memory_subsample * 128, 'memory_mb': bytes_to_mb(mem_size)}
        mem_dict['aux_memory'] = {'params': self.memory_subsample * 128, 'memory_mb': bytes_to_mb(mem_size)}

        # Activations (estimate for batch=2, features 1280x32x64 approx)
        activation_size = 2 * 1280 * 32 * 64 * 4  # Rough estimate
        mem_dict['activation_maps'] = {'params': '-', 'memory_mb': bytes_to_mb(activation_size)}

        # Total
        total_params = sum(v['params'] for v in mem_dict.values() if v['params'] != '-')
        total_mem = sum(v['memory_mb'] for v in mem_dict.values())
        mem_dict['total_training'] = {'params': total_params, 'memory_mb': total_mem}
        mem_dict['total_inference'] = {'params': total_params, 'memory_mb': total_mem - 2 * mem_dict['id_memory']['memory_mb']}

        wandb.log(mem_dict)
        return mem_dict

    def _debug_loss_values(self, seg_loss, ood_loss, batch_idx):
        """Debug helper to track loss values"""
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}:")
            logger.info(f" Seg loss: {seg_loss.item():.6f}")
            logger.info(f" OOD loss: {ood_loss.item():.6f}")
            
            with torch.no_grad():
                for name, param in self.projection_head.named_parameters():
                    if 'projection.3' in name and 'weight' in name:
                        logger.info(f" Proj final weight stats: mean={param.mean():.6f}, std={param.std():.6f}")

    def _evaluate_ood(self, epoch=None):
        """Evaluate OOD metrics using PixelOODEvaluator"""
        evaluator = PixelOODEvaluator(self.device, segmentation_head=self.segmentation_head)
        metrics = evaluator.evaluate(
            self.feature_extractor,
            self.projection_head,
            self.id_memory,
            self.aux_memory,
            beta_border=128.0,
            epoch=epoch
        )
        logger.info(f"OOD Metrics: {metrics}")
        return metrics

    def _evaluate_semantic(self, epoch=None, num_images=30, output_dir="semantic_results"):
        """Evaluate semantic segmentation on Cityscapes val images"""
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Evaluating semantic segmentation on {num_images} Cityscapes val images...")
        
        self.segmentation_head.eval()
        self.feature_extractor.eval()
        
        confusion_matrix = np.zeros((self.config['num_classes'], self.config['num_classes']))
        processed = 0
        
        for batch_idx, batch in enumerate(self.val_loader):
            if processed >= num_images:
                break
                
            batch_gpu = self._prepare_batch(batch)
            images = batch_gpu['data']
            labels = batch_gpu['label']
            B = images.size(0)
            
            for b in range(B):
                if processed >= num_images:
                    break
                    
                img = images[b:b+1]
                gt = labels[b:b+1]
                
                with torch.no_grad():
                    extracted = self.feature_extractor.extract_features_batch({'data': img})
                    features = extracted['features']
                    seg_logits = self.segmentation_head(features)
                    seg_logits = F.interpolate(
                        seg_logits, size=gt.shape[-2:], mode='bilinear', align_corners=True
                    )
                    pred = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()
                    gt_np = gt[0].cpu().numpy()
                    
                    # Ignore OOD/ignore labels in mIoU
                    valid_mask = (gt_np < self.config['num_classes']) & (gt_np != 255) & (gt_np != 254)
                    
                    if np.any(valid_mask):
                        pred_valid = pred[valid_mask]
                        gt_valid = gt_np[valid_mask]
                        self._update_confusion_matrix(confusion_matrix, pred_valid, gt_valid)
                    
                    # Colorize prediction
                    seg_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for cls, color in enumerate(CITYSCAPES_COLORMAP):
                        seg_color[pred == cls] = color
                    
                    # Save to folder
                    save_path = os.path.join(output_dir, f"epoch{epoch}_img{processed:03d}.png")
                    Image.fromarray(seg_color).save(save_path)
                    logger.info(f"Saved semantic segmentation: {save_path}")
                
                processed += 1
        
        # Compute mIoU
        iou_per_class = []
        for i in range(self.config['num_classes']):
            if confusion_matrix[i, i] == 0 and np.sum(confusion_matrix[i, :]) == 0:
                continue
            
            iou = confusion_matrix[i, i] / (
                np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i] + 1e-10
            )
            iou_per_class.append(iou)
        
        miou = np.mean(iou_per_class) if iou_per_class else 0.0
        logger.info(f"Semantic mIoU (19 classes): {miou:.4f}")
        return {'miou': miou}

    def _update_confusion_matrix(self, cm, pred, gt):
        """Update confusion matrix efficiently"""
        n = cm.shape[0]
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        indices = gt_flat * n + pred_flat
        unique, counts = np.unique(indices, return_counts=True)
        cm.flat[unique] += counts

    def _compute_memory_quality(self):
        """Compute diversity and separability of memory banks"""
        if len(self.id_memory) == 0 or len(self.aux_memory) == 0:
            return {}
        
        sample_size = min(1000, min(len(self.id_memory), len(self.aux_memory)))
        id_sample = self.id_memory[torch.randperm(len(self.id_memory))[:sample_size]].cpu().numpy()
        aux_sample = self.aux_memory[torch.randperm(len(self.aux_memory))[:sample_size]].cpu().numpy()
        
        def mean_cos_sim(feats):
            if len(feats) < 2:
                return 0.0
            feats_norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            sim = np.dot(feats_norm, feats_norm.T)
            np.fill_diagonal(sim, 0)
            return np.mean(sim)
        
        id_div = mean_cos_sim(id_sample)
        aux_div = mean_cos_sim(aux_sample)
        
        all_feats = np.concatenate([id_sample, aux_sample])
        all_labels = np.concatenate([np.zeros(len(id_sample)), np.ones(len(aux_sample))])
        
        try:
            sil_score = sklearn.metrics.silhouette_score(all_feats, all_labels) if len(set(all_labels)) > 1 else 0.0
        except:
            sil_score = 0.0
        
        return {
            'id_diversity': 1 - id_div,
            'aux_diversity': 1 - aux_div,
            'sil_separability': sil_score
        }

    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        # Move memories to CPU for saving
        id_mem_cpu = self.id_memory.cpu()
        aux_mem_cpu = self.aux_memory.cpu()
        
        checkpoint = {
            'epoch': epoch,
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'segmentation_head_state_dict': self.segmentation_head.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_seg_state_dict': self.optimizer_seg.state_dict(),
            'optimizer_proj_state_dict': self.optimizer_proj.state_dict(),
            'id_memory': id_mem_cpu,
            'aux_memory': aux_mem_cpu,
            'best_val_miou': self.best_val_miou,
            'best_fpr95': self.best_fpr95,
            'global_step': self.global_step,
            'metrics': metrics
        }
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
        # Log checkpoint as artifact
        artifact = wandb.Artifact(name=f"checkpoint-epoch-{epoch}", type="checkpoint")
        artifact.add_file(path)
        wandb.run.log_artifact(artifact)
        
        # Save best model
        if 'fpr95' in metrics and metrics['fpr95'] < self.best_fpr95:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
            
            best_artifact = wandb.Artifact(name="best-model", type="checkpoint")
            best_artifact.add_file(best_path)
            wandb.run.log_artifact(best_artifact)

    def _save_performance_tables(self, epoch):
        """Save memory and time tables to .txt as LaTeX"""
        txt_path = f"epoch_{epoch}_metrics.txt"
        
        with open(txt_path, 'w') as f:
            # Memory Table
            f.write("\\begin{table}[htbp]\n\\centering\n\\caption{Memory requirements}\n\\label{tab:memory_requirements}\n\\begin{tabular}{lcc}\n\\toprule\n\\textbf{Component} & \\textbf{Parameters} & \\textbf{Memory (MB)} \\\\\n\\midrule\n")
            
            for comp, vals in self.memory_metrics.items():
                params = vals['params'] if vals['params'] != '-' else '-'
                mem = f"{vals['memory_mb']:.1f}"
                f.write(f"{comp.replace('_', ' ').title()} & {params} & {mem} \\\\\n")
            
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
            
            # Time Table
            f.write("\\begin{table}[htbp]\n\\centering\n\\caption{Inference time breakdown}\n\\label{tab:inference_time}\n\\begin{tabular}{lcc}\n\\toprule\n\\textbf{Component} & \\textbf{Time (ms)} & \\textbf{Percentage (\\%)} \\\\\n\\midrule\n")
            
            total_ms = self.time_metrics['total']
            for comp, ms in self.time_metrics.items():
                if comp == 'fps':
                    continue
                perc = (ms / total_ms * 100) if total_ms > 0 else 0
                f.write(f"{comp.replace('_', ' ').title()} & {ms:.1f} & {perc:.1f} \\\\\n")
            
            f.write(f"\\midrule\n\\textbf{{Total}} & {total_ms:.1f} & 100.0 \\\\\n\\textbf{{FPS}} & {self.time_metrics['fps']:.1f} & - \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        
        logger.info(f"Saved performance tables to {txt_path}")

    def train(self):
        """Main training loop"""
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING (Pixel-Wise Hopfield Boosting)")
        logger.info(f"Total epochs: {self.total_epochs}")
        logger.info(f"Frozen backbone: Yes")
        logger.info("="*80 + "\n")
        
        start_epoch = 1
        if self.resume_from:
            start_epoch = self._load_checkpoint(self.resume_from) + 1
        
        for epoch in range(start_epoch, self.total_epochs + 1):
            epoch_start = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch}/{self.total_epochs}")
            logger.info(f"{'='*60}")
            
            # Training
            train_metrics = self._train_epoch(epoch)
            logger.info(f"\nEpoch {epoch} Training Summary:")
            logger.info(f" Avg Seg Loss: {train_metrics['avg_seg_loss']:.4f}")
            logger.info(f" Avg OOD Loss: {train_metrics['avg_ood_loss']:.4f}")
            
            # Memory bank quality evaluation
            if epoch % self.memory_eval_interval == 0:
                mem_metrics = self._compute_memory_quality()
                wandb.log(mem_metrics)
            
            # OOD evaluation
            logger.info("\nRunning OOD evaluation...")
            ood_metrics = self._evaluate_ood(epoch=epoch)
            
            # Semantic evaluation
            logger.info("\nRunning semantic segmentation evaluation...")
            sem_metrics = self._evaluate_semantic(epoch=epoch)
            
            # Scheduler step
            val_metric = ood_metrics.get('fpr95', 1.0)
            self.scheduler_seg.step(val_metric)
            self.scheduler_proj.step(val_metric)
            
            # Save checkpoint
            combined_metrics = {**ood_metrics, **sem_metrics}
            self.save_checkpoint(epoch, combined_metrics)
            
            # Save performance tables
            self._save_performance_tables(epoch)
            
            # Early stopping check
            if 'fpr95' in ood_metrics and ood_metrics['fpr95'] < self.best_fpr95:
                self.best_fpr95 = ood_metrics['fpr95']
                logger.info(f"New best FPR95: {self.best_fpr95:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Log to wandb
            log_dict = {'epoch': epoch}
            
            if ood_metrics:
                log_dict.update({
                    'ood_auroc': ood_metrics.get('auroc', 0.0),
                    'ood_fpr95': ood_metrics.get('fpr95', 1.0),
                    'ood_auprs': ood_metrics.get('auprs', 0.0),
                })
            
            if sem_metrics:
                log_dict.update({
                    'semantic_miou': sem_metrics.get('miou', 0.0),
                })
            
            wandb.log(log_dict)
            
            # Log epoch time
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch} completed in {epoch_time:.1f} seconds")
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best FPR95: {self.best_fpr95:.4f}")
        logger.info("="*80)
        
        wandb.finish()
        return self.best_fpr95


# Data loading helper functions
def val_joint_transform(img, gt):
    """Validation transformation"""
    size = (512, 1024)
    img = transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)(img)
    if gt is not None:
        gt = transforms.Resize(size, interpolation=InterpolationMode.NEAREST)(gt)
    
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    
    if gt is not None:
        gt = np.array(gt, dtype=np.uint8)
    
    return img, gt


class DictWrapperDataset:
    """Wrapper to convert tuple dataset to dict format"""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) >= 1:
            return {'data': item[0], 'label': item[1] if len(item) > 1 else None}
        return item
    
    def __len__(self):
        return len(self.dataset)


# Metric classes
class PixelMetric(ABC):
    @abstractmethod
    def __call__(self, in_scores, out_scores):
        pass


class AUROCMetric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        if len(in_scores) == 0 or len(out_scores) == 0:
            print("❌ Empty scores for AUROC, returning 0.0")
            return 0.0
        targets_np = np.concatenate([np.zeros_like(in_scores, dtype=int), np.ones_like(out_scores, dtype=int)])
        scores_np = np.concatenate([in_scores, out_scores])
        return sklearn.metrics.roc_auc_score(targets_np, scores_np)


class FPR95Metric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        if len(in_scores) == 0 or len(out_scores) == 0:
            print("❌ Empty scores for FPR95, returning 1.0")
            return 1.0
        targets_np = np.concatenate([np.zeros_like(in_scores, dtype=int), np.ones_like(out_scores, dtype=int)])
        scores_np = np.concatenate([in_scores, out_scores])
        return self._fpr_at_tpr(targets_np, scores_np, tpr_level=0.95)

    def _fpr_at_tpr(self, y_true, y_score, tpr_level=0.95):
        y_true = (y_true == 1)
        desc_indices = np.argsort(-y_score)
        y_score_sorted = y_score[desc_indices]
        y_true_sorted = y_true[desc_indices]
    
        distinct_indices = np.where(np.diff(y_score_sorted))[0]
        threshold_indices = np.r_[distinct_indices, y_true_sorted.size - 1]
    
        tps = np.cumsum(y_true_sorted)[threshold_indices]
        fps = 1 + threshold_indices - tps
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    
        if len(tpr) == 0 or tpr[-1] == 0:
            return 1.0
    
        cutoff = np.argmin(np.abs(tpr - tpr_level))
        n_negatives = np.sum(~y_true)
        if n_negatives == 0:
            return 0.0
        return fps[cutoff] / n_negatives


class AUPRSMetric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        if len(in_scores) == 0 or len(out_scores) == 0:
            print("❌ Empty scores for AUPRS, returning 0.0")
            return 0.0
        targets_np = np.concatenate([np.zeros_like(in_scores, dtype=int), np.ones_like(out_scores, dtype=int)])
        scores_np = np.concatenate([in_scores, out_scores])
        return sklearn.metrics.average_precision_score(targets_np, scores_np)

# FIXED: Inherit from Dataset instead of DataLoader
class FishyscapesDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Warning: Dataset directories not found: {image_dir}, {label_dir}")
            self.images = []
        else:
            self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        
        # Resize to 512x1024
        image = transforms.Resize((512, 1024), interpolation=InterpolationMode.BILINEAR)(image)
        label = transforms.Resize((512, 1024), interpolation=InterpolationMode.NEAREST)(label)
        
        image = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
        label = np.array(label).astype(np.int64)
        
        image = torch.tensor(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        return image, torch.tensor(label)


class PixelOODEvaluator:
    def __init__(self, device, segmentation_head=None):
        self.device = device
        self.segmentation_head = segmentation_head
        self.transform = PILToTensorTransform(target_size=(512, 1024))
        
        # Define datasets WITH transform
        self.datasets = [
            ('static', Fishyscapes(split='Static', root="/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static", transform=self.transform)),
            ('lf', Fishyscapes(split='LostAndFound', root="/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/cityscapes_processed", transform=self.transform)),
            ('road_anomaly', RoadAnomaly(root="/home/ha51dybi/PEBAL/fishyscapes_lostandfound/final_dataset/road_anomaly", transform=self.transform)),
            ('lost_and_found', LostAndFound(root="/home/ha51dybi/PEBAL/fishyscapes_lostandfound/LostAndFound", transform=self.transform))
        ]
        
        self.results_dir = "ood_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def safe_subsample(self, scores, labels, max_pixels=None):
        """No subsampling - use all pixels"""
        total_pixels = len(scores)
        if total_pixels == 0:
            return scores, labels
        
        if max_pixels is None or total_pixels <= max_pixels:
            return scores, labels
        
        # If we need to subsample, balance in/out
        in_mask = (labels == 0)
        out_mask = (labels == 1)
        
        in_scores = scores[in_mask]
        out_scores = scores[out_mask]
        in_labels = labels[in_mask]
        out_labels = labels[out_mask]
        
        half_max = max_pixels // 2
        
        if len(in_scores) > half_max:
            in_perm = torch.randperm(len(in_scores))[:half_max]
            in_scores = in_scores[in_perm]
            in_labels = in_labels[in_perm]
        
        if len(out_scores) > half_max:
            out_perm = torch.randperm(len(out_scores))[:half_max]
            out_scores = out_scores[out_perm]
            out_labels = out_labels[out_perm]
        
        return torch.cat([in_scores, out_scores]), torch.cat([in_labels, out_labels])

    def evaluate(self, feature_extractor, projection_pipeline, id_memory, aux_memory, beta_border=128.0, epoch=None):
        print(f"Starting multi-dataset OOD evaluation...")
        
        if id_memory is None or aux_memory is None:
            print("Memories are None - cannot evaluate")
            return {}
        
        feature_extractor.eval()
        projection_pipeline.eval()
        
        id_memory = id_memory.to(self.device)
        aux_memory = aux_memory.to(self.device)
        
        score_calc = PixelWiseInferenceScore(id_memory, aux_memory, beta=beta_border)
        
        metrics_calcs = {
            'auroc': AUROCMetric(),
            'fpr95': FPR95Metric(),
            'auprs': AUPRSMetric()
        }
        
        all_metrics = {}
        avg_fpr95 = 0.0
        num_valid_datasets = 0
        
        for ds_name, ds in self.datasets:
            print(f"Evaluating on {ds_name}...")
            print(f"Dataset size: {len(ds)}")
            
            ds_loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
            
            all_in_scores = []
            all_out_scores = []
            total_images = 0
            images_with_ood = 0
            processed_images = 0
            
            ds_vis_dir = os.path.join(self.results_dir, ds_name)
            os.makedirs(ds_vis_dir, exist_ok=True)
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(ds_loader):
                    total_images += images.size(0)
                    
                    try:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        labels[(labels > 0) & (labels != 255)] = 1  # Binarize labels
                        
                        if batch_idx == 0:
                            unique_labels = torch.unique(labels)
                            ood_pixels = (labels == 1).sum().item()
                            total_pixels = labels.numel()
                            print(f"First batch - unique labels: {unique_labels.cpu().numpy()}")
                            print(f"First batch - OOD pixels: {ood_pixels}/{total_pixels} ({100*ood_pixels/total_pixels:.2f}%)")
                        
                        batch_dict = {'data': images, 'label': labels}
                        extracted = feature_extractor.extract_features_batch(batch_dict)
                        
                        if 'features' not in extracted:
                            print(f"No features extracted for batch {batch_idx}")
                            continue
                        
                        features = extracted['features']
                        
                        if batch_idx == 0:
                            print(f"DEBUG - Extracted feature shape: {features.shape}")
                        
                        projected = projection_pipeline(features)
                        B, C, H, W = projected.shape
                        
                        pixel_features = projected.permute(0, 2, 3, 1).contiguous().view(-1, C)
                        labels_resized = F.interpolate(
                            labels.unsqueeze(1).float(), size=(H, W), mode='nearest'
                        ).squeeze(1).long()
                        pixel_labels = labels_resized.view(-1)
                        
                        assert pixel_features.shape[0] == pixel_labels.shape[0], f"Shape mismatch: features {pixel_features.shape[0]}, labels {pixel_labels.shape[0]}"
                        
                        ood_scores = self._compute_ood_scores(pixel_features, score_calc)
                        
                        if len(ood_scores) == 0:
                            print(f"No OOD scores computed for batch {batch_idx}")
                            continue
                        
                        if not torch.isfinite(ood_scores).all():
                            print(f"Non-finite OOD scores in batch {batch_idx}, skipping")
                            continue
                        
                        # Compute segmentation if available
                        pred = None
                        if self.segmentation_head:
                            seg_logits = self.segmentation_head(features)
                            seg_logits = F.interpolate(
                                seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                            )
                            pred = torch.argmax(seg_logits, dim=1)
                        
                        # Visualization for first batch of every dataset
                        save_vis = (epoch is not None)
                        if save_vis and batch_idx < 1:
                            try:
                                for b in range(B):
                                    # Upsample scores to input size
                                    ood_scores_up = F.interpolate(
                                        ood_scores.view(B, 1, H, W),
                                        size=labels.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze(1)
                                    
                                    scores_map = ood_scores_up[b].cpu().numpy()
                                    scores_norm = (scores_map - scores_map.min()) / (scores_map.max() - scores_map.min() + 1e-5)
                                    
                                    colormap = plt.colormaps['inferno']
                                    scores_color = colormap(scores_norm)[:, :, :3]
                                    
                                    orig_img = images[b].cpu().numpy().transpose(1,2,0)
                                    # Denormalize orig_img
                                    mean = np.array([0.485, 0.456, 0.406])
                                    std = np.array([0.229, 0.224, 0.225])
                                    orig_img = (orig_img * std + mean).clip(0, 1)
                                    
                                    label_np = labels[b].cpu().numpy()
                                    
                                    # Anomaly GT
                                    anomaly_gt = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                                    anomaly_gt[label_np == 0] = [0,0,0]  # black for in
                                    anomaly_gt[label_np == 1] = [1,0,0]  # red
                                    anomaly_gt[label_np == 255] = [0.5,0.5,0.5]  # gray
                                    
                                    # OOD mask
                                    threshold = np.quantile(scores_map.flatten(), 0.95)
                                    ood_mask = (scores_map > threshold)
                                    ood_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                                    ood_color[ood_mask] = [1,1,0]  # yellow
                                    
                                    # Seg map including OOD
                                    if pred is not None:
                                        seg_with_ood = pred[b].cpuseg_with_ood = pred[b].cpu().numpy()
                                        seg_with_ood[ood_mask] = 19  # Set detected OOD to class 19
                                        seg_color = np.zeros((seg_with_ood.shape[0], seg_with_ood.shape[1], 3))
                                        for cls, color in enumerate(CITYSCAPES_COLORMAP):
                                            seg_color[seg_with_ood == cls] = [c/255.0 for c in color]
                                    else:
                                        # Dummy if no seg head
                                        seg_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                                        for cls, color in enumerate(CITYSCAPES_COLORMAP[:19]):
                                            seg_color[label_np == cls] = [c/255.0 for c in color]
                                        seg_color[label_np == 254] = [0,0,1]
                                    
                                    fig, axs = plt.subplots(1,5, figsize=(25,5))
                                    axs[0].imshow(orig_img)
                                    axs[0].set_title('Original Image')
                                    axs[1].imshow(anomaly_gt)
                                    axs[1].set_title('Anomaly Ground Truth')
                                    axs[2].imshow(seg_color)
                                    axs[2].set_title('Segmentation incl. OOD')
                                    axs[3].imshow(ood_color)
                                    axs[3].set_title('OOD Map (Mask)')
                                    axs[4].imshow(scores_color)
                                    axs[4].set_title('OOD Score Map')
                                    
                                    vis_path = os.path.join(ds_vis_dir, f'epoch{epoch}_{batch_idx}_{b}.png')
                                    plt.savefig(vis_path)
                                    plt.close()
                                    print(f"Saved evaluation visualization for {ds_name}, epoch {epoch}, batch {batch_idx}, image {b} to {vis_path}")
                            
                            except Exception as e:
                                print(f"Error in visualization for batch {batch_idx}: {e}")
                                traceback.print_exc()
                            finally:
                                plt.close()
                        
                        valid_mask = (pixel_labels != 255)
                        valid_ood = ood_scores[valid_mask]
                        valid_labels = pixel_labels[valid_mask]
                        
                        if len(valid_ood) == 0:
                            print(f"No valid pixels after masking for batch {batch_idx}")
                            continue
                        
                        sub_ood, sub_labels = self.safe_subsample(
                            valid_ood, valid_labels, max_pixels=None  # Use all pixels
                        )
                        
                        in_mask = (sub_labels == 0)
                        out_mask = (sub_labels == 1)
                        
                        in_count = in_mask.sum().item()
                        out_count = out_mask.sum().item()
                        
                        if out_count > 0:
                            images_with_ood += 1
                        
                        if batch_idx < 3:
                            print(f"Batch {batch_idx}: in={in_count}, ood={out_count}")
                            print(f" Scores shape: {sub_ood.shape}, Labels shape: {sub_labels.shape}")
                        
                        if in_count > 0:
                            in_scores_batch = sub_ood[in_mask].cpu().numpy()
                            if len(in_scores_batch) > 0:
                                all_in_scores.append(in_scores_batch)
                        
                        if out_count > 0:
                            out_scores_batch = sub_ood[out_mask].cpu().numpy()
                            if len(out_scores_batch) > 0:
                                all_out_scores.append(out_scores_batch)
                        
                        processed_images += images.size(0)
                        
                        if batch_idx % 5 == 0:
                            torch.cuda.empty_cache()
                        
                        del projected, features, ood_scores, sub_ood, sub_labels
                    
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        traceback.print_exc()
                        continue
            
            print(f"Processed {processed_images}/{total_images} images for {ds_name}, {images_with_ood} had OOD pixels")
            print(f"Total in-distribution batches: {len(all_in_scores)}")
            print(f"Total OOD batches: {len(all_out_scores)}")
            
            if not all_in_scores or not all_out_scores:
                print(f"No valid scores collected for {ds_name}")
                print(f"In-distribution batches: {len(all_in_scores)}")
                print(f"OOD batches: {len(all_out_scores)}")
                continue
            
            try:
                in_scores = np.concatenate(all_in_scores) if all_in_scores else np.array([])
                out_scores = np.concatenate(all_out_scores) if all_out_scores else np.array([])
            except Exception as e:
                print(f"Error concatenating scores for {ds_name}: {e}")
                continue
            
            print(f"Final scores for {ds_name} - In: {len(in_scores)}, Out: {len(out_scores)}")
            print(f"In-scores range: [{in_scores.min():.3f}, {in_scores.max():.3f}]" if len(in_scores) > 0 else "Empty in_scores")
            print(f"Out-scores range: [{out_scores.min():.3f}, {out_scores.max():.3f}]" if len(out_scores) > 0 else "Empty out_scores")
            
            ds_metrics = {}
            for metric_name, metric in metrics_calcs.items():
                try:
                    score = metric(in_scores, out_scores)
                    ds_metrics[metric_name] = float(score)
                    print(f"✅ {ds_name.upper()} {metric_name.upper()}: {score:.4f}")
                except Exception as e:
                    print(f"Error computing {metric_name} for {ds_name}: {e}")
                    ds_metrics[metric_name] = 0.0
            
            # Log histograms to wandb
            if len(in_scores) > 0:
                wandb.log({f"energy_hist_id_{ds_name}": wandb.Histogram(in_scores)})
            if len(out_scores) > 0:
                wandb.log({f"energy_hist_ood_{ds_name}": wandb.Histogram(out_scores)})
            
            # Prefix and add to all_metrics
            for k, v in ds_metrics.items():
                all_metrics[f"{k}_{ds_name}"] = v
            
            if 'fpr95' in ds_metrics and ds_metrics['fpr95'] > 0:
                avg_fpr95 += ds_metrics['fpr95']
                num_valid_datasets += 1
            
            # Clear memory after each dataset
            del in_scores, out_scores
            torch.cuda.empty_cache()
            gc.collect()
        
        if num_valid_datasets > 0:
            avg_fpr95 /= num_valid_datasets
            all_metrics['fpr95'] = avg_fpr95
            print(f"Average FPR95 across datasets: {avg_fpr95:.4f}")
        else:
            avg_fpr95 = 1.0
            all_metrics['fpr95'] = 1.0
        
        # Save all metrics to JSON
        json_path = os.path.join(self.results_dir, f"epoch{epoch}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        print(f"Saved metrics to {json_path}")
        
        # Move memories to CPU immediately after use
        id_memory = id_memory.cpu()
        aux_memory = aux_memory.cpu()
        
        return all_metrics

    def _compute_ood_scores(self, pixel_features, score_calc):
        """Compute OOD scores with chunking for memory efficiency"""
        if len(pixel_features) == 0:
            return torch.tensor([], device=self.device)
        
        chunk_size = 10000
        num_pixels = len(pixel_features)
        all_scores = torch.empty(num_pixels, device=self.device, dtype=torch.float32)
        
        min_chunk_size = 100
        max_retries = 3
        
        try:
            start_idx = 0
            for i in range(0, num_pixels, chunk_size):
                end_i = min(i + chunk_size, num_pixels)
                chunk = pixel_features[i:end_i]
                
                if len(chunk) == 0:
                    continue
                
                retries = 0
                while retries < max_retries:
                    try:
                        with autocast():
                            chunk_scores = score_calc(chunk)
                            chunk_scores = chunk_scores.squeeze()
                            if chunk_scores.dim() == 0:
                                chunk_scores = chunk_scores.unsqueeze(0)
                        
                        all_scores[start_idx:start_idx + len(chunk_scores)] = chunk_scores
                        start_idx += len(chunk_scores)
                        break  # Success, exit retry loop
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e) and chunk_size > min_chunk_size:
                            chunk_size = max(chunk_size // 2, min_chunk_size)
                            print(f"Reducing chunk_size to {chunk_size}")
                            retries += 1
                        else:
                            raise  # Re-raise if not OOM or chunk too small
                    finally:
                        torch.cuda.empty_cache()
                
                if retries >= max_retries:
                    print(f"Failed to process chunk after {max_retries} retries")
                    break
            
            return all_scores
        
        except Exception as e:
            print(f"Error in _compute_ood_scores: {e}")
            return torch.tensor([], device=self.device)


def main():
    """Main training script"""
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Training configuration
    train_config = {
        'model_path': "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth",
        'checkpoint_dir': "./checkpoints_improved",
        'num_classes': 19,
        'learning_rate': 1e-5,
        'weight_decay': 5e-5,
        'batch_size': 2,
        'num_workers': 0
    }
    
    # Check model path
    if not os.path.exists(train_config['model_path']):
        logger.error(f"Model checkpoint not found: {train_config['model_path']}")
        return
    
    # Setup data loaders
    logger.info("Setting up data loaders...")
    
    cityscapes_root = "/home/ha51dybi/PEBAL/cityscapes"
    
    # Verify directories exist
    images_dir = os.path.join(cityscapes_root, "images", "city_gt_fine", "train")
    labels_dir = os.path.join(cityscapes_root, "annotation", "city_gt_fine", "train")
    
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        logger.error(f"Labels directory not found: {labels_dir}")
        return
    
    logger.info(f"Found images in: {images_dir}")
    logger.info(f"Found labels in: {labels_dir}")
    
    # Check file counts
    img_count = len([f for f in os.listdir(images_dir) if f.endswith('_leftImg8bit.png')])
    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('_gtFine.png')])
    logger.info(f"Image files: {img_count}, Label files: {label_count}")
    
    # Create engine for data loading
    class CustomArgs:
        def __init__(self):
            self.ddp = False
            self.local_rank = -1
            self.gpus = 1
            self.world_size = 1
    
    from engine.engine import Engine
    from config.config import config as global_config
    from dataset.data_loader import get_mix_loader, Cityscapes
    
    custom_args = CustomArgs()
    engine_instance = Engine(
        custom_arg=custom_args,
        logger=logger,
        continue_state_object=train_config['model_path']
    )
    
    global_config.batch_size = train_config['batch_size']
    
    # Mixed loader for training
    train_loader, _, _ = get_mix_loader(
        engine=engine_instance,
        augment=True,
        cs_root="/home/ha51dybi/PEBAL/cityscapes",
        coco_root="/home/ha51dybi/PEBAL/coco"
    )
    
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    
    val_transform = PILToTensorTransform(target_size=(512, 1024))
    val_dataset = Cityscapes(
        root="/home/ha51dybi/PEBAL/cityscapes",
        split='val',
        transform=val_transform
    )
    
    wrapped_val = DictWrapperDataset(val_dataset)
    val_loader = DataLoader(
        wrapped_val,
        batch_size=2,
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )
    
    # Get fixed batches for visualization
    val_iter = iter(val_loader)
    fixed_batches = []
    try:
        for _ in range(3):
            fixed_batches.append(next(val_iter))
    except StopIteration:
        pass
    
    # Log dataset statistics
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Batch size: {train_config['batch_size']}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Check for OOD pixels in training data
    sample_batch = next(iter(train_loader))
    if 'label' in sample_batch:
        labels = sample_batch['label']
        ood_count = (labels == 254).sum().item()
        total_pixels = labels.numel()
        unique_labels = torch.unique(labels)
        logger.info(f"Sample batch unique labels: {unique_labels}")
        logger.info(f"Sample batch OOD ratio: {ood_count}/{total_pixels} = {ood_count/total_pixels:.4%}")
    
    logger.info("="*60 + "\n")
    
    # Create trainer
    trainer = ImprovedOODSegmentationTrainer(
        train_config,
        train_loader,
        val_loader,
        fixed_batches=fixed_batches,
        resume_from=None  # Set to path if resuming
    )
    
    # Start training
    best_fpr95 = trainer.train()
    
    logger.info(f"\nFinal Results:")
    logger.info(f" Best FPR95: {best_fpr95:.4f}")
    
    return best_fpr95


if __name__ == "__main__":
    main()
                                        
                                        
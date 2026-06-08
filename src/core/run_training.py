#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import gc
import numpy as np
from torch.utils.data import DataLoader
import logging
import time
import wandb
import math
import torch.cuda.amp as amp
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Any, Dict, Tuple
import traceback
# Imports from provided programs
from segmentation_head import SegmentationClassifierHead  # Assuming this exists
from feature_extractor import FeatureExtractor  # Assuming this exists
from projection_head import SimpleProjectionHead
from hopfield_memory_builder import MemoryBuilder  # Assuming this exists
from hopfield_weight_updater import HopfieldBoostingManager  # Assuming this exists
import sklearn.metrics
from abc import ABC, abstractmethod
import matplotlib.cm as cm
from torch.cuda.amp import autocast
from pixel_energy import compute_hopfield_ood_loss, PixelWiseBorderEnergy, PixelWiseInferenceScore, lse  # From PEBAL-inspired
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
        # Convert target if target is not None and isinstance(target, Image.Image):
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

        # Training parameters (FIXED: Increased subsample for diverse memories; higher lambda for OOD emphasis)
        self.total_epochs = 50
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints_improved")
        self.beta =128.0
        self.lambda_ood = 5.0  # FIXED: Increased to 1.5 to amplify OOD loss signal for projection head updates
        self.memory_subsample = 100000

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize wandb (unchanged)
        wandb.init(
            project="ood-seg-improved",
            config=self.config,
            name="frozen-backbone-fixed-hopfield-pixelwise",
            mode="online"
        )

        # Initialize models
        logger.info("Initializing models...")
        self._init_models()

        # Build or load memories (unchanged)
        if self.resume_from and os.path.exists(self.resume_from):
            self._load_checkpoint(self.resume_from)
        else:
            self._build_initial_memories()

        self._init_training_components()

        # Mixed precision scaler (unchanged)
        self.scaler = amp.GradScaler()

        self.best_val_miou = 0.0
        self.best_fpr95 = 1.0
        self.patience = 100
        self.patience_counter = 0
        self.global_step = 0
        self.accum_steps = 1  # FIXED: Reduced to 1 to avoid diluting small gradients from OOD loss
        self.warmup_steps = 500

    def _load_checkpoint(self, path):
        """Load checkpoint and move memories to GPU; recreate optimizers if needed"""
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

        # FIXED: Recreate optimizers after load to ensure they include current head params (in case mismatch)
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

        # FIXED: Recreate optimizers after adding head to include its params (in case mismatch)
        # REMOVED: self._init_training_components() # This call is moved to after memories are set

    def _init_weights(self):
        """Initialize weights carefully to prevent gradient explosion"""
        for module in [self.segmentation_head, self.projection_head]:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    # FIXED: Use orthogonal init for better separation in projection head
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
            lr=base_lr * 10,  # FIXED: 10x higher LR for projection head to encourage updates on similar features
            weight_decay=5e-4,
            eps=1e-6
        )

        # FIXED: Use ReduceLROnPlateau on FPR95 for better control
        self.scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_seg, mode='min', factor=0.5, patience=3, min_lr=1e-7, verbose=True
        )
        self.scheduler_proj = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_proj, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )

        # FIXED: Standard CE without smoothing for simplicity/stability
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=255, reduction='mean'
        )

        # Initialize Hopfield manager (FIXED: Increase iterations for more boosting on hard samples)
        self.hopfield_manager = HopfieldBoostingManager(
            id_features_full=self.id_memory,
            aux_features_full=self.aux_memory,
            beta_sampling=self.beta,
            lambda_ood=self.lambda_ood,
            device=self.device,
            memory_subset_size=min(10000, self.memory_subsample * self.config['num_classes']),
            positive_shift=False,
            num_boosting_iters=5  # FIXED: Increased to 3 for better hard-sample focus (from Hopfield paper)
        )

    def _prepare_batch(self, batch):
        """Prepare batch for GPU (unchanged)"""
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == 'label':
                    batch_gpu[k] = v.to(self.device).long()
                else:
                    batch_gpu[k] = v.to(self.device).float()
        return batch_gpu

    def _compute_losses(self, batch):
        """Fixed loss computation with comprehensive stability checks and reg for head"""
        batch_gpu = self._prepare_batch(batch)

        # Extract features (unchanged)
        extracted = self.feature_extractor.extract_features_batch(batch_gpu)
        features = extracted['features'].float()
        labels = extracted['labels']

        # Validate and fix labels (unchanged, but added logging for debug)
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
            # FIXED: Soften clamp to allow more range
            seg_logits = torch.clamp(seg_logits, min=-10, max=10)

            if seg_logits.shape[-2:] != labels.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=True
                )

            # Prepare labels for CE loss (unchanged)
            labels_for_ce = labels.clone()
            labels_for_ce[labels == 254] = 255
            labels_for_ce[labels >= self.config['num_classes']] = 255

            # Compute CE loss with numerical stability (unchanged)
            ce_unreduced = F.cross_entropy(
                seg_logits, labels_for_ce, ignore_index=255, reduction='none'
            )

            # Filter out inf/nan values (unchanged)
            valid_loss_mask = torch.isfinite(ce_unreduced)
            if not valid_loss_mask.all():
                logger.warning(f"Inf/NaN in {(~valid_loss_mask).sum()} pixels")
            ce_unreduced = torch.where(
                valid_loss_mask, ce_unreduced, torch.zeros_like(ce_unreduced)
            )

            # Take mean only over valid pixels (unchanged)
            if valid_loss_mask.any():
                seg_loss = ce_unreduced[valid_loss_mask].mean()
            else:
                seg_loss = torch.tensor(0.0, device=self.device)

            # Final stability check (unchanged)
            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                logger.warning("Loss is NaN/Inf, setting to 0")
                seg_loss = torch.tensor(0.0, device=self.device)

        # OOD loss computation with stability
        ood_loss = torch.tensor(0.0, device=self.device)
        with torch.cuda.amp.autocast():
            projected = self.projection_head(features)
            # FIXED: Soften clamp for more range
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

                # FIXED: Log feature std for debug (check if similar)
                if id_mask.any():
                    logger.info(f"ID feature std: {id_pixels.std().item():.4f}")
                if ood_mask.any():
                    logger.info(f"OOD feature std: {ood_pixels.std().item():.4f}")

                if id_mask.any():
                    num_to_sample = min(128, len(valid_pixels))
                    id_batch, aux_batch = self.hopfield_manager.sample_batch(num_to_sample)

                    boosted_id = torch.cat([id_pixels, id_batch.to(self.device).float()]) if len(id_batch) > 0 else id_pixels
                    boosted_ood = torch.cat([ood_pixels, aux_batch.to(self.device).float()]) if len(aux_batch) > 0 else ood_pixels

                    if len(boosted_id) > 0 and len(boosted_ood) > 0:
                        raw_ood_loss = self.hopfield_manager.compute_boosted_ood_loss(boosted_ood, boosted_id)
                        # FIXED: Soften clamp for OOD loss
                        ood_loss = torch.clamp(raw_ood_loss, min=-20.0, max=20.0)

                        if torch.isnan(ood_loss) or torch.isinf(ood_loss):
                            logger.warning("OOD loss is NaN/Inf, setting to 0")
                            ood_loss = torch.tensor(0.0, device=self.device)
                    else:
                        ood_loss = torch.tensor(0.0, device=self.device)  # Skip if no OOD

        # FIXED: Add small L2 reg loss to always update projection head (encourage weight growth even if ood_loss=0)
        reg_loss = 0.0
        for param in self.projection_head.parameters():
            if param.requires_grad:
                reg_loss += torch.norm(param) * 1e-5  # Small L2 to nudge weights
        ood_loss += reg_loss

        return {
            'seg_loss': seg_loss,
            'ood_loss': ood_loss,
            'has_id': (labels_for_ce < 255).any() if labels is not None else False,
        }

    def _train_epoch(self, epoch):
        """UPDATED: Train with weight updates every epoch (paper Sec. 2: frequent for dynamic hard focus)"""
        # CHANGED: Update sampling weights every epoch (more frequent updates)
        self.hopfield_manager.update_sampling_weights(memory_size=self.memory_subsample)

        self.segmentation_head.train()
        self.projection_head.train()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_metrics = {
            'seg_losses': [],
            'ood_losses': [],
        }
        accum_count = 0

        for batch_idx, batch in enumerate(progress_bar):
            if self.global_step < self.warmup_steps:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
                for pg in self.optimizer_seg.param_groups:
                    pg['lr'] = lr_scale * 5e-5
                for pg in self.optimizer_proj.param_groups:
                    pg['lr'] = lr_scale * 5e-5

            try:
                loss_dict = self._compute_losses(batch)
            except RuntimeError as e:
                logger.warning(f"Error computing losses at batch {batch_idx}: {e}")
                continue

            seg_loss = loss_dict['seg_loss'] / self.accum_steps
            ood_loss = loss_dict['ood_loss'] / self.accum_steps

            if accum_count == 0:
                self.optimizer_seg.zero_grad(set_to_none=True)
                self.optimizer_proj.zero_grad(set_to_none=True)

            try:
                # Always run backward for both losses (zero loss = zero gradient, which is fine)
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
                # Gradient clipping for both optimizers
                for param_group in [self.segmentation_head.parameters(), self.projection_head.parameters()]:
                    for param in param_group:
                        if param.grad is not None:
                            param.grad.data = torch.clamp(param.grad.data, -5.0, 5.0)

                # Unscale and clip gradients for segmentation
                self.scaler.unscale_(self.optimizer_seg)
                grad_norm_seg = torch.nn.utils.clip_grad_norm_(
                    self.segmentation_head.parameters(), max_norm=2.0
                )

                # Unscale and clip gradients for projection
                self.scaler.unscale_(self.optimizer_proj)
                grad_norm_proj = torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(), max_norm=2.0
                )

                if batch_idx % 10 == 0:
                    logger.info(f"Grad norms - Seg: {grad_norm_seg:.4f}, Proj: {grad_norm_proj:.4f}")

                # Step optimizers with gradient norm checks
                if grad_norm_seg > 10.0:
                    logger.warning(f"Skipping seg update due to large gradients: {grad_norm_seg}")
                    self.optimizer_seg.zero_grad(set_to_none=True)
                else:
                    self.scaler.step(self.optimizer_seg)

                if grad_norm_proj > 10.0:
                    logger.warning(f"Skipping proj update due to large gradients: {grad_norm_proj}")
                    self.optimizer_proj.zero_grad(set_to_none=True)
                else:
                    self.scaler.step(self.optimizer_proj)

                self.scaler.update()
                accum_count = 0

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

        avg_seg_loss = np.mean(epoch_metrics['seg_losses']) if epoch_metrics['seg_losses'] else 0.0
        avg_ood_loss = np.mean(epoch_metrics['ood_losses']) if epoch_metrics['ood_losses'] else 0.0

        return {
            'avg_seg_loss': avg_seg_loss,
            'avg_ood_loss': avg_ood_loss,
        }

    def _debug_loss_values(self, seg_loss, ood_loss, batch_idx):
        """Debug helper to track loss values (unchanged)"""
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}:")
            logger.info(f" Seg loss: {seg_loss.item():.6f}")
            logger.info(f" OOD loss: {ood_loss.item():.6f}")
            with torch.no_grad():
                for name, param in self.projection_head.named_parameters():
                    if 'projection.3' in name and 'weight' in name:
                        # FIXED: Log projection head final layer stats
                        logger.info(f" Proj final weight stats: mean={param.mean():.6f}, std={param.std():.6f}")

    def _evaluate_ood(self, epoch=None):
        """Evaluate OOD metrics using PixelOODEvaluator"""
        evaluator = PixelOODEvaluator(self.device, segmentation_head=self.segmentation_head)
        metrics = evaluator.evaluate(
            self.feature_extractor,
            self.projection_head,
            self.id_memory,
            self.aux_memory,
            beta_border=128.0,  # As per paper
            epoch=epoch
        )
        logger.info(f"OOD Metrics: {metrics}")
        return metrics

    def _evaluate_semantic(self, epoch=None, num_images=30, output_dir="semantic_results"):
        """Evaluate semantic segmentation on Cityscapes val images (19 classes) (unchanged)"""
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

                    # Ignore OOD/ignore labels (254/255) in mIoU
                    valid_mask = (gt_np < self.config['num_classes']) & (gt_np != 255) & (gt_np != 254)
                    if np.any(valid_mask):
                        pred_valid = pred[valid_mask]
                        gt_valid = gt_np[valid_mask]
                        self._update_confusion_matrix(confusion_matrix, pred_valid, gt_valid)

                    # Colorize prediction (19 classes only)
                    seg_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for cls, color in enumerate(CITYSCAPES_COLORMAP):
                        seg_color[pred == cls] = color

                    # Save to folder
                    save_path = os.path.join(output_dir, f"epoch{epoch}_img{processed:03d}.png")
                    Image.fromarray(seg_color).save(save_path)
                    logger.info(f"Saved semantic segmentation: {save_path}")

                processed += 1

        # Compute mIoU (unchanged)
        iou_per_class = []
        for i in range(self.config['num_classes']):
            if confusion_matrix[i, i] == 0 and np.sum(confusion_matrix[i, :]) == 0:
                continue  # Skip empty classes
            iou = confusion_matrix[i, i] / (
                np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i] + 1e-10
            )
            iou_per_class.append(iou)

        miou = np.mean(iou_per_class) if iou_per_class else 0.0
        logger.info(f"Semantic mIoU (19 classes): {miou:.4f}")

        return {'miou': miou}

    def _update_confusion_matrix(self, cm, pred, gt):
        """Update confusion matrix efficiently (unchanged)"""
        n = cm.shape[0]
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        indices = gt_flat * n + pred_flat
        unique, counts = np.unique(indices, return_counts=True)
        cm.flat[unique] += counts

    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint - Memories to CPU to avoid OOM on load (unchanged)"""
        # Move memories to CPU for saving
        id_mem_cpu = self.id_memory.cpu()
        aux_mem_cpu = self.aux_memory.cpu()

        checkpoint = {
            'epoch': epoch,
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),  # Frozen weights
            'segmentation_head_state_dict': self.segmentation_head.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_seg_state_dict': self.optimizer_seg.state_dict(),
            'optimizer_proj_state_dict': self.optimizer_proj.state_dict(),
            'id_memory': id_mem_cpu,  # CPU for saving
            'aux_memory': aux_mem_cpu,  # CPU for saving
            'best_val_miou': self.best_val_miou,
            'best_fpr95': self.best_fpr95,
            'global_step': self.global_step,
            'metrics': metrics
        }

        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        # Save best model
        if 'fpr95' in metrics and metrics['fpr95'] < self.best_fpr95:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def train(self):
        """Main training loop (unchanged)"""
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING (Pixel-Wise Hopfield Boosting)")
        logger.info(f"Total epochs: {self.total_epochs}")
        logger.info(f"Frozen backbone: Yes (as per request)")
        logger.info(f"Always train proj head via boosted samples")
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

            # OOD eval every epoch (vis only every 5) (unchanged)
            logger.info("\nRunning OOD evaluation...")
            ood_metrics = self._evaluate_ood(epoch=epoch)

            # Semantic eval every epoch (unchanged)
            logger.info("\nRunning semantic segmentation evaluation...")
            sem_metrics = self._evaluate_semantic(epoch=epoch)

            # Scheduler step here (FIXED: Step on FPR95)
            val_metric = ood_metrics.get('fpr95', 1.0)
            self.scheduler_seg.step(val_metric)
            self.scheduler_proj.step(val_metric)

            # Save checkpoint (unchanged)
            combined_metrics = {**ood_metrics, **sem_metrics}
            self.save_checkpoint(epoch, combined_metrics)

            # Early stopping check (on FPR95) (unchanged)
            if 'fpr95' in ood_metrics and ood_metrics['fpr95'] < self.best_fpr95:
                self.best_fpr95 = ood_metrics['fpr95']
                logger.info(f"New best FPR95: {self.best_fpr95:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Log to wandb (unchanged)
            log_dict = {
                'epoch': epoch,
            }
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

            # Log epoch time (unchanged)
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch} completed in {epoch_time:.1f} seconds")

            # Clear cache (unchanged)
            torch.cuda.empty_cache()
            gc.collect()

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best FPR95: {self.best_fpr95:.4f}")
        logger.info("="*80)

        wandb.finish()
        return self.best_fpr95


# Data loading helper functions (unchanged)
def val_joint_transform(img, gt):
    """Validation transformation"""
    size = (512, 1024)  # Increased slightly
    img = transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)(img)
    if gt is not None:
        gt = transforms.Resize(size, interpolation=InterpolationMode.NEAREST)(gt)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    if gt is not None:
        gt = np.array(gt, dtype=np.uint8)
    return img, gt


class DictWrapperDataset:
    """Wrapper to convert tuple dataset to dict format (unchanged)"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) >= 1:
            return {'data': item[0], 'label': item[1] if len(item) > 1 else None}
        return item

    def __len__(self):
        return len(self.dataset)


# PixelOODEvaluator class (FIXED: No subsample, upsample vis, use PixelWiseInferenceScore)
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


class FPR95Metric:
    def __call__(self, in_scores, out_scores):
        if len(in_scores) == 0 or len(out_scores) == 0:
            print("❌ Empty scores for FPR95, returning 1.0")
            return 1.0
        targets_np = np.concatenate([np.zeros_like(in_scores, dtype=int), np.ones_like(out_scores, dtype=int)])
        scores_np = np.concatenate([in_scores, out_scores])
        return self._fpr_at_tpr(targets_np, scores_np, tpr_level=0.95)
    
    def _fpr_at_tpr(self, y_true, y_score, tpr_level=0.95):
        y_true = (y_true == 1)  # True for positives (OOD/anomalies)
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
        
        # Correct: Find first (minimal) index where tpr >= tpr_level
        valid_cutoffs = np.where(tpr >= tpr_level)[0]
        if len(valid_cutoffs) == 0:
            return 1.0  # Cannot reach tpr_level
        cutoff = np.min(valid_cutoffs)
        
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


class FishyscapesDataset(DataLoader):
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
        # Resize to 512x1024 as requested (unchanged)
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
        # Create transform for all datasets
        self.transform = PILToTensorTransform(target_size=(512, 1024)) # Make it instance variable
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
        # FIXED: No subsample (set to None for all pixels)
        total_pixels = len(scores)
        if total_pixels == 0:
            return scores, labels

        if max_pixels is None or total_pixels <= max_pixels:
            return scores, labels

        # Balance in/out (unchanged, but skipped since max_pixels=None)
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
            print("❌ Memories are None - cannot evaluate")
            return {}

        feature_extractor.eval()
        projection_pipeline.eval()
        id_memory = id_memory.to(self.device)
        aux_memory = aux_memory.to(self.device)
        score_calc = PixelWiseInferenceScore(id_memory, aux_memory, beta=beta_border)  # FIXED: Use PixelWiseInferenceScore for PEBAL energy

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
                        labels[(labels > 0) & (labels != 255)] = 1  # Binarize labels early

                        if batch_idx == 0:
                            unique_labels = torch.unique(labels)
                            ood_pixels = (labels == 1).sum().item()
                            total_pixels = labels.numel()
                            print(f"First batch - unique labels: {unique_labels.cpu().numpy()}")
                            print(f"First batch - OOD pixels: {ood_pixels}/{total_pixels} ({100*ood_pixels/total_pixels:.2f}%)")

                        batch_dict = {'data': images, 'label': labels}
                        extracted = feature_extractor.extract_features_batch(batch_dict)
                        if 'features' not in extracted:
                            print(f"❌ No features extracted for batch {batch_idx}")
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
                            print(f"❌ No OOD scores computed for batch {batch_idx}")
                            continue
                        if not torch.isfinite(ood_scores).all():
                            print(f"❌ Non-finite OOD scores in batch {batch_idx}, skipping")
                            continue

                        # Compute segmentation if available (interpolated to input size) (unchanged)
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
                                    # FIXED: Upsample scores to input size
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

                                    # Anomaly GT: black for in (0), cyan for anomaly (1), blue for ignore (255)
                                    anomaly_gt = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                                    anomaly_gt[label_np == 0] = [0,0,0]  # black for in
                                    anomaly_gt[label_np == 1] = [1,0,0]  # red
                                    anomaly_gt[label_np == 255] = [0.5,0.5,0.5]  # gray

                                    # OOD mask (threshold scores > quantile 0.85, green for detected)
                                    threshold = np.quantile(scores_map.flatten(), 0.95)
                                    ood_mask = (scores_map > threshold)
                                    ood_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                                    ood_color[ood_mask] = [1,1,0]  # yellow

                                    # Seg map including OOD (if available)
                                    if pred is not None:
                                        seg_with_ood = pred[b].cpu().numpy()
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
                                print(f"❌ Error in visualization for batch {batch_idx}: {e}")
                                traceback.print_exc()
                            finally:
                                plt.close()

                        valid_mask = (pixel_labels != 255)
                        valid_ood = ood_scores[valid_mask]
                        valid_labels = pixel_labels[valid_mask]

                        if len(valid_ood) == 0:
                            print(f"❌ No valid pixels after masking for batch {batch_idx}")
                            continue

                        sub_ood, sub_labels = self.safe_subsample(
                            valid_ood, valid_labels, max_pixels=None  # FIXED: Use all pixels
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

                        if batch_idx % 5 == 0:  # More frequent
                            torch.cuda.empty_cache()

                        del projected, features, ood_scores, sub_ood, sub_labels

                    except Exception as e:
                        print(f"❌ Error processing batch {batch_idx}: {e}")
                        traceback.print_exc()
                        continue

            print(f"Processed {processed_images}/{total_images} images for {ds_name}, {images_with_ood} had OOD pixels")
            print(f"Total in-distribution batches: {len(all_in_scores)}")
            print(f"Total OOD batches: {len(all_out_scores)}")

            if not all_in_scores or not all_out_scores:
                print(f"❌ No valid scores collected for {ds_name}")
                print(f"In-distribution batches: {len(all_in_scores)}")
                print(f"OOD batches: {len(all_out_scores)}")
                continue

            try:
                in_scores = np.concatenate(all_in_scores) if all_in_scores else np.array([])
                out_scores = np.concatenate(all_out_scores) if all_out_scores else np.array([])
            except Exception as e:
                print(f"❌ Error concatenating scores for {ds_name}: {e}")
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
                    print(f"❌ Error computing {metric_name} for {ds_name}: {e}")
                    ds_metrics[metric_name] = 0.0

            # Prefix and add to all_metrics
            for k, v in ds_metrics.items():
                all_metrics[f"{k}_{ds_name}"] = v

            if 'fpr95' in ds_metrics and ds_metrics['fpr95'] > 0:
                avg_fpr95 += ds_metrics['fpr95']
                num_valid_datasets += 1

            # Explicit GPU memory clearing after each dataset evaluation
            del in_scores, out_scores
            torch.cuda.empty_cache()
            gc.collect()

        if num_valid_datasets > 0:
            avg_fpr95 /= num_valid_datasets
            all_metrics['fpr95'] = avg_fpr95  # Average for scheduler
            print(f"Average FPR95 across datasets: {avg_fpr95:.4f}")
        else:
            avg_fpr95 = 1.0
            all_metrics['fpr95'] = 1.0

        # Save all metrics to JSON
        import json
        json_path = os.path.join(self.results_dir, f"epoch{epoch}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved metrics to {json_path}")

        # Move memories to CPU immediately after use
        id_memory = id_memory.cpu()
        aux_memory = aux_memory.cpu()

        return all_metrics

    def _compute_ood_scores(self, pixel_features, score_calc):
        if len(pixel_features) == 0:
            return torch.tensor([], device=self.device)

        chunk_size = 10000
        num_pixels = len(pixel_features)
        all_scores = torch.empty(num_pixels, device=self.device, dtype=torch.float32)
    
        min_chunk_size = 100  # Prevent infinite reduction
        max_retries = 3  # Limit retry attempts

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
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target_path = self.targets[idx]
        image = Image.open(img_path).convert('RGB')
        if os.path.exists(target_path):
            target = Image.open(target_path).convert('L')
        else:
            # Create dummy target if missing
            target = Image.new('L', image.size, 255)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


def main():
    """Main training script (FIXED: Lower BS for OOM, num_workers=0 for stability)"""
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Set random seeds (unchanged)
    torch.manual_seed(42)
    np.random.seed(42)

    # Training configuration (unchanged)
    train_config = {
        'model_path': "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth",
        'checkpoint_dir': "./checkpoints_improved",
        'num_classes': 19,
        'learning_rate': 1e-5,
        'weight_decay': 5e-5,
        'batch_size': 2,  # FIXED: Reduced to avoid OOM
        'num_workers': 0  # FIXED: Set to 0 for stability
    }

    # Check model path (unchanged)
    if not os.path.exists(train_config['model_path']):
        logger.error(f"Model checkpoint not found: {train_config['model_path']}")
        return

    # Setup data loaders (FIXED: Use custom CityscapesCocoMix)
    logger.info("Setting up data loaders...")

    # Remove the symlink creation code and replace with:
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
        transform=val_transform  # Pass the transform here
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

    # Get fixed batches for visualization (unchanged)
    val_iter = iter(val_loader)
    fixed_batches = []
    try:
        for _ in range(3):
            fixed_batches.append(next(val_iter))
    except StopIteration:
        pass

    # Log dataset statistics (unchanged)
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Batch size: {train_config['batch_size']}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Check for OOD pixels in training data (unchanged)
    sample_batch = next(iter(train_loader))
    if 'label' in sample_batch:
        labels = sample_batch['label']
        ood_count = (labels == 254).sum().item()
        total_pixels = labels.numel()
        unique_labels = torch.unique(labels)
        logger.info(f"Sample batch unique labels: {unique_labels}")
        logger.info(f"Sample batch OOD ratio: {ood_count}/{total_pixels} = {ood_count/total_pixels:.4%}")

    logger.info("="*60 + "\n")

    # Create trainer (unchanged)
    trainer = ImprovedOODSegmentationTrainer(
        train_config,
        train_loader,
        val_loader,
        fixed_batches=fixed_batches,
        resume_from=None  # Set to path if resuming
    )

    # Start training (unchanged)
    best_fpr95 = trainer.train()
    logger.info(f"\nFinal Results:")
    logger.info(f" Best FPR95: {best_fpr95:.4f}")

    return best_fpr95


if __name__ == "__main__":
    main()
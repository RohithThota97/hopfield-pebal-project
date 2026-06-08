import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from collections import defaultdict
import sklearn.metrics
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.cuda.amp import autocast
import traceback
from torchvision.transforms import Normalize
from pixel_energy import PixelWiseHopfieldEnergyCalculator

class PixelMetric(ABC):
    @abstractmethod
    def __call__(self, in_scores, out_scores):
        # Access in_scores and out_scores to avoid unused variable warnings
        _ = in_scores, out_scores
        pass

class AUROCMetric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.zeros_like(in_scores, dtype=torch.int),
            torch.ones_like(out_scores, dtype=torch.int)
        ])
        scores = torch.cat([in_scores, out_scores])
       
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
       
        return sklearn.metrics.roc_auc_score(targets_np, scores_np)

class FPR95Metric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.zeros_like(in_scores, dtype=torch.int),
            torch.ones_like(out_scores, dtype=torch.int)
        ])
        scores = torch.cat([in_scores, out_scores])
       
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
       
        return self._fpr_at_tpr(targets_np, scores_np, tpr_level=0.95)
   
    def _fpr_at_tpr(self, y_true, y_score, tpr_level=0.95):
        y_true = (y_true == 1)
       
        desc_indices = np.argsort(y_score)[::-1]
        y_score = y_score[desc_indices]
        y_true = y_true[desc_indices]
       
        distinct_indices = np.where(np.diff(y_score))[0]
        threshold_indices = np.r_[distinct_indices, y_true.size - 1]
       
        tps = np.cumsum(y_true)[threshold_indices]
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
        targets = torch.cat([
            torch.zeros_like(in_scores, dtype=torch.int),
            torch.ones_like(out_scores, dtype=torch.int)
        ])
        scores = torch.cat([in_scores, out_scores])
       
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
       
        return sklearn.metrics.average_precision_score(targets_np, scores_np)

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
       
        image = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
        label = np.array(label).astype(np.int64)
       
        image = torch.tensor(image)
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
       
        return image, torch.tensor(label)

class PixelOODEvaluator:
    def __init__(self, device):
        self.device = device
       
        fishyscapes_dir = "/home/ha51dybi/PEBAL/fishyscapes_lostandfound/Static"
        image_dir = os.path.join(fishyscapes_dir, "original")
        label_dir = os.path.join(fishyscapes_dir, "labels")
       
        self.ood_dataset = FishyscapesDataset(image_dir, label_dir)
        self.ood_loader = DataLoader(self.ood_dataset, batch_size=1, shuffle=False)
   
    def safe_subsample(self, scores, labels, max_pixels=None):
        """Safely subsample scores and labels - use all if max_pixels=None"""
        total_pixels = len(scores)
       
        if total_pixels == 0:
            return scores, labels
       
        if max_pixels is None or total_pixels <= max_pixels:
            return scores, labels
       
        perm = torch.randperm(total_pixels, device=scores.device)
        selected_indices = perm[:max_pixels]
       
        selected_indices = torch.clamp(selected_indices, 0, total_pixels - 1)
       
        return scores[selected_indices], labels[selected_indices]
   
    def evaluate(self, feature_extractor, projection_pipeline, id_memory, aux_memory, beta_border=64.0):
        from pixel_energy import PixelWiseHopfieldEnergyCalculator
       
        print(f"Starting OOD evaluation...")
        print(f"Dataset size: {len(self.ood_dataset)}")
        print(f"ID memory shape: {id_memory.shape if id_memory is not None else 'None'}")
        print(f"AUX memory shape: {aux_memory.shape if aux_memory is not None else 'None'}")
       
        if id_memory is None or aux_memory is None:
            print("❌ Memories are None - cannot evaluate")
            return {}
       
        # FIXED: Set to eval mode to avoid BN issues with batch_size=1
        feature_extractor.eval()
        projection_pipeline.eval()
       
        id_memory = id_memory.to(self.device)
        aux_memory = aux_memory.to(self.device)
       
        energy_calc = PixelWiseHopfieldEnergyCalculator.create_border_energy(
            id_memory, aux_memory, beta=beta_border, positive_shift=True
        )
       
        metrics = {
            'auroc': AUROCMetric(),
            'fpr95': FPR95Metric(),
            'auprs': AUPRSMetric()
        }
       
        all_in_scores = []
        all_out_scores = []
        total_images = 0
        images_with_ood = 0
        processed_images = 0
       
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ood_loader):
                total_images += 1
               
                try:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                   
                    # Debug first image
                    if batch_idx == 0:
                        unique_labels = torch.unique(labels)
                        ood_pixels = (labels == 1).sum().item()
                        total_pixels = labels.numel()
                        print(f"First image - unique labels: {unique_labels.cpu().numpy()}")
                        print(f"First image - OOD pixels: {ood_pixels}/{total_pixels} ({100*ood_pixels/total_pixels:.2f}%)")
                   
                    # Extract features
                    batch_dict = {'data': images, 'label': labels}
                    extracted = feature_extractor.extract_features_batch(batch_dict)
                   
                    if 'features' not in extracted:
                        print(f"❌ No features extracted for image {batch_idx}")
                        continue
                   
                    features = extracted['features']
                   
                    # Debug feature dimensions
                    if batch_idx == 0:
                        print(f"DEBUG - Extracted feature shape: {features.shape}")
                   
                    # Now project the features
                    projected = projection_pipeline(features)
                   
                    # Get dimensions and reshape safely
                    B, C, H, W = projected.shape
                    pixel_features = projected.permute(0, 2, 3, 1).contiguous().view(-1, C)
                   
                    # Reshape labels to match features
                    if labels.shape[-2:] != (H, W):
                        labels_resized = F.interpolate(
                            labels.unsqueeze(1).float(),
                            size=(H, W),
                            mode='nearest'
                        ).squeeze(1).long()
                    else:
                        labels_resized = labels
                   
                    pixel_labels = labels_resized.view(-1)
                   
                    # Compute full OOD scores
                    ood_scores = self._compute_ood_scores(pixel_features, energy_calc)
                   
                    if len(ood_scores) == 0:
                        print(f"❌ No OOD scores computed for image {batch_idx}")
                        continue
                   
                    # For visualization (first 3 images)
                    if batch_idx < 3:
                        scores_map = ood_scores.view(H, W).cpu().numpy()
                        scores_norm = (scores_map - scores_map.min()) / (scores_map.max() - scores_map.min() + 1e-5)
                        colormap = plt.colormaps['plasma']
                        scores_color = colormap(scores_norm)[:, :, :3]  # RGB
                       
                        orig_img = images[0].cpu().numpy().transpose(1,2,0)
                       
                        label_np = labels[0].cpu().numpy()
                        label_color = np.zeros((label_np.shape[0], label_np.shape[1], 3))
                        label_color[label_np == 0] = [0,0,0]
                        label_color[label_np == 1] = [1,0,0]
                        label_color[label_np == 255] = [0,0,1]
                       
                        fig, axs = plt.subplots(1,3, figsize=(15,5))
                        axs[0].imshow(orig_img)
                        axs[0].set_title('Original')
                        axs[1].imshow(label_color)
                        axs[1].set_title('Labels')
                        axs[2].imshow(scores_color)
                        axs[2].set_title('OOD Scores')
                        plt.savefig(f'ood_vis_{batch_idx}.png')
                        plt.close()
                        print(f"Saved visualization for image {batch_idx} to ood_vis_{batch_idx}.png")
                   
                    # For metrics: exclude ignore and subsample
                    valid_mask = (pixel_labels != 255)
                    valid_ood = ood_scores[valid_mask]
                    valid_labels = pixel_labels[valid_mask]
                   
                    if len(valid_ood) == 0:
                        print(f"❌ No valid pixels after masking for image {batch_idx}")
                        continue
                   
                    # Subsample for metrics
                    sub_ood, sub_labels = self.safe_subsample(
                        valid_ood, valid_labels, max_pixels=1000000
                    )
                   
                    # Masks
                    in_mask = (sub_labels == 0)
                    out_mask = (sub_labels == 1)
                   
                    in_count = in_mask.sum().item()
                    out_count = out_mask.sum().item()
                   
                    if out_count > 0:
                        images_with_ood += 1
                   
                    # Debug for first few images
                    if batch_idx < 3:
                        print(f"Image {batch_idx}: in={in_count}, ood={out_count}")
                        print(f" Scores shape: {sub_ood.shape}, Labels shape: {sub_labels.shape}")
                   
                    # Collect scores
                    if in_count > 0:
                        in_scores_batch = sub_ood[in_mask]
                        if len(in_scores_batch) > 0:
                            all_in_scores.append(in_scores_batch)
                   
                    if out_count > 0:
                        out_scores_batch = sub_ood[out_mask]
                        if len(out_scores_batch) > 0:
                            all_out_scores.append(out_scores_batch)
                   
                    processed_images += 1
                   
                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
               
                except Exception as e:
                    print(f"❌ Error processing image {batch_idx}: {e}")
                    traceback.print_exc()
                    continue
       
        print(f"Processed {processed_images}/{total_images} images, {images_with_ood} had OOD pixels")
        print(f"Total in-distribution batches: {len(all_in_scores)}")
        print(f"Total OOD batches: {len(all_out_scores)}")
       
        if not all_in_scores or not all_out_scores:
            print("❌ No valid scores collected")
            print(f"In-distribution batches: {len(all_in_scores)}")
            print(f"OOD batches: {len(all_out_scores)}")
            return {}
       
        try:
            in_scores = torch.cat(all_in_scores)
            out_scores = torch.cat(all_out_scores)
        except Exception as e:
            print(f"❌ Error concatenating scores: {e}")
            return {}
       
        print(f"Final scores - In: {len(in_scores)}, Out: {len(out_scores)}")
        print(f"In-scores range: [{in_scores.min():.3f}, {in_scores.max():.3f}]")
        print(f"Out-scores range: [{out_scores.min():.3f}, {out_scores.max():.3f}]")
       
        results = {}
        for metric_name, metric in metrics.items():
            try:
                score = metric(in_scores, out_scores)
                results[metric_name] = float(score)
                print(f"✅ {metric_name.upper()}: {score:.4f}")
            except Exception as e:
                print(f"❌ Error computing {metric_name}: {e}")
                results[metric_name] = 0.0
       
        return results
   
    def _compute_ood_scores(self, pixel_features, energy_calc):
        """Compute OOD scores with proper error handling - use positive energies (higher for OOD)"""
        if len(pixel_features) == 0:
            return torch.tensor([], device=self.device)
       
        chunk_size = 10000 # Increased chunk size for efficiency
        all_scores = []
       
        try:
            for i in range(0, len(pixel_features), chunk_size):
                chunk = pixel_features[i:i+chunk_size]
               
                if len(chunk) == 0:
                    continue
               
                # Add spatial dimensions for energy calculation
                chunk_4d = chunk.unsqueeze(-1).unsqueeze(-1) # [N, C, 1, 1]
               
                with autocast():
                    energies = energy_calc(chunk_4d)
               
                # Ensure energies is 1D
                energies = energies.squeeze()
                if energies.dim() == 0:
                    energies = energies.unsqueeze(0)
               
                # Use positive energy as OOD score (higher = more OOD)
                all_scores.append(energies)
               
            if all_scores:
                return torch.cat(all_scores)
            else:
                return torch.tensor([], device=self.device)
               
        except Exception as e:
            print(f"❌ Error in _compute_ood_scores: {e}")
            return torch.tensor([], device=self.device)
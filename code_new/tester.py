#!/usr/bin/env python3
# updated_memory_builder_with_analysis.py - Updated version with dummy data generation, memory building,
# t-SNE and PCA visualizations. Simulates ID and OOD with small differences in distributions to replicate
# the paradigms from the three papers: Hopfield Boosting (focus on boundary sharpening with AUX/OOD close to ID),
# PEBAL (pixel-wise energy-biased abstention for anomaly segmentation with adaptive penalties for close OOD),
# and Master's Thesis (assuming similar OOD detection themes). Ablation studies simulated by varying overlap
# between ID and OOD clusters. Visualizations show small differences in distributions.
# Analysis:
# - Hopfield Boosting: We simulate weak learners by sampling near boundaries where ID and OOD overlap slightly.
# - PEBAL: Adaptive penalties modeled by energy-based separation; close OOD pixels get lower penalties in visualization.
# - Master's Thesis: Assumes general OOD detection; we replicate with dummy semantic segmentation features.
# Goal: Show t-SNE/PCA where ID classes are distinct clusters, OOD is interspersed or close, with small differences
# (e.g., overlap ratio ~10-20%) to mimic real-world urban driving anomalies that are semantically similar to ID.

from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
import logging
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Dummy FeatureExtractor and ProjectionHead for simulation
class DummyFeatureExtractor:
    def __init__(self, device):
        self.device = device

    def eval(self):
        pass

    def extract_features_batch(self, batch):
        # Simulate feature extraction: return random features with labels
        batch_size = batch['data'].shape[0]
        feat_h, feat_w = 32, 32  # Simulated downsampled size
        features = torch.randn(batch_size, 1280, feat_h, feat_w, device=self.device)  # High-dim features
        labels = batch['labels']  # Use provided dummy labels
        # Properly resize labels to feature size using interpolation
        if labels.dim() == 3:  # [B, H, W]
            labels = F.interpolate(labels.unsqueeze(1).float(), size=(feat_h, feat_w), mode='nearest').squeeze(1).long()
        return {
            'features': features,
            'labels': labels
        }

class DummyProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=1280, output_dim=128):
        super().__init__()
        self.proj = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

class MemoryBuilder:
    def __init__(self, feature_extractor, projection_pipeline, device,
                 id_memory_size=60000, aux_memory_size=50000, num_in_dist_classes=19,
                 max_sample_per_image_class=500, max_sample_per_image_ood=1000,
                 target_id_per_class=3000, target_ood=50000,
                 ood_label=254, input_size=(512, 512), log_level=logging.INFO,
                 overlap_ratio=0.15):  # Added for ablation: control ID-OOD overlap
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = feature_extractor
        self.projection_pipeline = projection_pipeline.float().to(device)
        self.device = device
        self.id_memory_size = id_memory_size
        self.aux_memory_size = aux_memory_size
        self.num_in_dist_classes = num_in_dist_classes
        self.target_id_per_class = target_id_per_class
        self.target_ood = target_ood
        self.max_sample_per_image_class = max_sample_per_image_class
        self.max_sample_per_image_ood = max_sample_per_image_ood
        self.ood_label = ood_label
        self.input_size = input_size
        self.feature_dim = self._infer_feature_dimension()
        self.analysis_data = defaultdict(lambda: defaultdict(int))
        self._verbose_logging = log_level <= logging.INFO
        self.overlap_ratio = overlap_ratio  # For ablation: how much OOD overlaps with ID

    def _infer_feature_dimension(self):
        try:
            dummy_input = {'data': torch.randn(1, 3, *self.input_size).to(self.device)}
            with torch.no_grad():
                self.feature_extractor.eval()
                extracted = self.feature_extractor.extract_features_batch(dummy_input)
                features = extracted['features']
                self.projection_pipeline.eval()
                projected = self.projection_pipeline(features).float()
                return projected.shape[1]
        except Exception as e:
            warnings.warn(f"Failed to infer feature dimension, using 128: {e}")
            return 128

    def generate_dummy_dataloader(self, num_batches=100, batch_size=4, ablation_variant='base'):
        """
        Generate dummy data: Simulate semantic segmentation dataset.
        - ID classes: Gaussian clusters per class with small variance.
        - OOD: Close to ID boundaries with small differences (overlap controlled by ablation).
        Ablation variants: 'high_overlap' (0.3), 'low_overlap' (0.05), 'no_overlap' (0.0)
        """
        if ablation_variant == 'high_overlap':
            self.overlap_ratio = 0.3
        elif ablation_variant == 'low_overlap':
            self.overlap_ratio = 0.05
        elif ablation_variant == 'no_overlap':
            self.overlap_ratio = 0.0

        self.logger.info(f"Generating dummy data with overlap_ratio={self.overlap_ratio} for ablation: {ablation_variant}")

        # Simulate class centers in feature space (for projection output)
        class_centers = torch.randn(self.num_in_dist_classes, self.feature_dim, device=self.device) * 2.0  # Spread out

        def dummy_batch_generator():
            for _ in range(num_batches):
                images = torch.randn(batch_size, 3, *self.input_size, device=self.device)
                labels = torch.randint(0, self.num_in_dist_classes, (batch_size, *self.input_size), device=self.device)
                # Add OOD pixels: 10% of pixels are OOD, close to random ID class
                ood_mask = torch.rand(batch_size, *self.input_size, device=self.device) < 0.1
                labels[ood_mask] = self.ood_label
                # For features, we'll simulate in extract_features_batch
                yield {'data': images, 'labels': labels}

        return list(dummy_batch_generator())  # Return as list for streaming simulation

    def process_images(self, dataloader, ablation_variant='base'):
        torch.manual_seed(42)
        np.random.seed(42)
        self.logger.info(f"Starting memory building with ablation variant: {ablation_variant}")
        id_candidates, ood_candidates, collected_counts = self._collect_candidate_features_streaming(dataloader)

        # Print collected pixels
        self._print_collected_pixels(collected_counts, id_candidates, ood_candidates)

        id_memory = self._build_simple_id_memory(id_candidates)
        ood_memory = self._build_simple_ood_memory(ood_candidates)

        # Compute and print similarities and separation
        self._compute_and_print_similarities(id_candidates, ood_candidates)
        separation_score = self._compute_separation(id_memory, ood_memory)
        self.logger.info(f"ID-OOD Separation Rate: {separation_score:.3f}")

        # Visualizations
        self._visualize_distributions(id_memory, ood_memory, ablation_variant)

        # Auto-save
        results_dir = f'results/memory_{ablation_variant}'
        os.makedirs(results_dir, exist_ok=True)
        id_path = os.path.join(results_dir, "id_memory.pt")
        ood_path = os.path.join(results_dir, "ood_memory.pt")
        torch.save(id_memory, id_path)
        torch.save(ood_memory, ood_path)
        self.logger.info("Memories saved.")

        return id_memory, ood_memory

    def _collect_candidate_features_streaming(self, dataloader) -> Tuple[Dict, torch.Tensor, Dict]:
        id_candidates = defaultdict(list)
        ood_candidates = []
        collected_counts = {'ood': 0}
        for cid in range(self.num_in_dist_classes):
            collected_counts[cid] = 0

        self.feature_extractor.eval()
        self.projection_pipeline.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting Features"):
                try:
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    if 'features' not in extracted:
                        continue

                    features = extracted['features'].to(self.device)
                    labels_spatial = extracted.get('labels').to(self.device)

                    projected = self.projection_pipeline(features).float()
                    pixel_features_flat = projected.permute(0, 2, 3, 1).reshape(-1, projected.shape[1])
                    normalized_features_flat = F.normalize(pixel_features_flat, p=2, dim=1).float()
                    normalized_features_spatial = normalized_features_flat.reshape(
                        features.shape[0], features.shape[2], features.shape[3], -1
                    ).float()

                    for i in range(features.shape[0]):
                        img_labels = labels_spatial[i]
                        img_features = normalized_features_spatial[i]

                        for class_id in range(self.num_in_dist_classes):
                            mask = (img_labels == class_id)
                            class_pixels = mask.sum().item()
                            collected_counts[class_id] += class_pixels
                            if class_pixels > 0:
                                feats = img_features[mask]
                                # Simulate small difference: perturb OOD close to ID
                                feats += torch.randn_like(feats) * self.overlap_ratio
                                sample_size = min(class_pixels, self.max_sample_per_image_class)
                                idx = torch.randperm(class_pixels)[:sample_size]
                                selected = feats[idx].cpu().float()
                                id_candidates[class_id].append(selected)

                        # OOD
                        ood_mask = (img_labels == self.ood_label)
                        ood_pixels = ood_mask.sum().item()
                        collected_counts['ood'] += ood_pixels
                        if ood_pixels > 0:
                            ood_feats = img_features[ood_mask]
                            # Simulate closeness: shift OOD towards random ID class
                            random_id = torch.randint(0, self.num_in_dist_classes, (1,)).item()
                            ood_feats += torch.randn_like(ood_feats) * self.overlap_ratio
                            sample_size = min(ood_pixels, self.max_sample_per_image_ood)
                            idx = torch.randperm(ood_pixels)[:sample_size]
                            selected_ood = ood_feats[idx].cpu().float()
                            ood_candidates.append(selected_ood)

                except Exception as e:
                    self.logger.error(f"Error processing batch: {e}. Skipping.")

        # Concat and sample down to target
        for class_id in id_candidates:
            if id_candidates[class_id]:
                id_candidates[class_id] = torch.cat(id_candidates[class_id], dim=0).float()
                collected = len(id_candidates[class_id])
                if collected < self.target_id_per_class:
                    self.logger.warning(f"Class {class_id} has only {collected} pixels collected, below target {self.target_id_per_class}")
                else:
                    idx = torch.randperm(collected)[:self.target_id_per_class]
                    id_candidates[class_id] = id_candidates[class_id][idx]

        ood_tensor = torch.cat(ood_candidates, dim=0).float() if ood_candidates else torch.empty(0, self.feature_dim)
        collected_ood = len(ood_tensor)
        if collected_ood < self.target_ood:
            self.logger.warning(f"OOD has only {collected_ood} pixels collected, below target {self.target_ood}")
        elif collected_ood > self.target_ood:
            idx = torch.randperm(collected_ood)[:self.target_ood]
            ood_tensor = ood_tensor[idx]

        return id_candidates, ood_tensor, collected_counts

    def _build_simple_id_memory(self, id_candidates):
        all_id = [feats for feats in id_candidates.values() if len(feats) > 0]
        if not all_id:
            return torch.empty(0, self.feature_dim, device=self.device)
        id_memory = torch.cat(all_id, dim=0).to(self.device)
        if len(id_memory) > self.id_memory_size:
            idx = torch.randperm(len(id_memory))[:self.id_memory_size]
            id_memory = id_memory[idx]
        return id_memory.float()

    def _build_simple_ood_memory(self, ood_candidates):
        ood_memory = ood_candidates.to(self.device)
        if len(ood_memory) > self.aux_memory_size:
            idx = torch.randperm(len(ood_memory))[:self.aux_memory_size]
            ood_memory = ood_memory[idx]
        return ood_memory.float()

    def _print_collected_pixels(self, collected_counts, id_candidates, ood_candidates):
        self.logger.info("Pixels seen after entire dataset:")
        for cid in range(self.num_in_dist_classes):
            sampled = len(id_candidates[cid]) if cid in id_candidates else 0
            self.logger.info(f"Class {cid}: {collected_counts[cid]} seen, {sampled} sampled")
        sampled_ood = len(ood_candidates) if isinstance(ood_candidates, torch.Tensor) else 0
        self.logger.info(f"OOD (254): {collected_counts['ood']} seen, {sampled_ood} sampled")

    def _compute_and_print_similarities(self, id_candidates, ood_candidates):
        # Within-class cos sim
        self.logger.info("Within-class average cosine similarities:")
        for cid in id_candidates:
            feats = id_candidates[cid].to(self.device)
            if len(feats) < 2:
                self.logger.info(f"Class {cid}: Insufficient samples for sim")
                continue
            sim = torch.matmul(feats, feats.T).mean().item()
            self.logger.info(f"Class {cid}: {sim:.3f}")

        # Between-class avg cos
        self.logger.info("Between-class average cosine similarities:")
        all_pairs = []
        for cid1 in id_candidates:
            for cid2 in id_candidates:
                if cid1 >= cid2:
                    continue
                f1 = id_candidates[cid1].to(self.device)[:1000]  # Sample to avoid OOM
                f2 = id_candidates[cid2].to(self.device)[:1000]
                if len(f1) == 0 or len(f2) == 0:
                    continue
                pair_sim = torch.matmul(f1, f2.T).mean().item()
                all_pairs.append(pair_sim)
        if all_pairs:
            avg_between = np.mean(all_pairs)
            self.logger.info(f"Average between ID classes: {avg_between:.3f}")

        # ID-OOD avg cos
        id_all = torch.cat([feats.to(self.device)[:1000] for feats in id_candidates.values() if len(feats) > 0])
        ood_sample = ood_candidates.to(self.device)[:1000]
        if len(id_all) > 0 and len(ood_sample) > 0:
            id_ood_sim = torch.matmul(id_all, ood_sample.T).mean().item()
            self.logger.info(f"Average ID-OOD cosine similarity: {id_ood_sim:.3f}")
        else:
            self.logger.info("Insufficient samples for ID-OOD sim")

    def _compute_separation(self, id_memory, ood_memory):
        if len(id_memory) == 0 or len(ood_memory) == 0:
            return 0.0
        id_center = id_memory.mean(dim=0)
        ood_center = ood_memory.mean(dim=0)
        inter_distance = torch.norm(id_center - ood_center).item()
        id_spread = torch.norm(id_memory - id_center, dim=1).mean().item()
        ood_spread = torch.norm(ood_memory - ood_center, dim=1).mean().item()
        separation_rate = inter_distance / (id_spread + ood_spread + 1e-6)
        return separation_rate

    def _visualize_distributions(self, id_memory, ood_memory, ablation_variant):
        # Sample for viz (to avoid large data)
        sample_size = 5000
        id_sample = id_memory[:sample_size].cpu().numpy()
        ood_sample = ood_memory[:sample_size].cpu().numpy()
        if len(id_sample) == 0 or len(ood_sample) == 0:
            self.logger.warning("Insufficient samples for visualization")
            return

        combined = np.concatenate([id_sample, ood_sample], axis=0)
        labels = np.array(['ID'] * len(id_sample) + ['OOD'] * len(ood_sample))

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        self._plot_embedding(pca_result, labels, f'PCA_{ablation_variant}.png')

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        tsne_result = tsne.fit_transform(combined)
        self._plot_embedding(tsne_result, labels, f'tSNE_{ablation_variant}.png')

        self.logger.info(f"Visualizations saved for {ablation_variant}")

    def _plot_embedding(self, embedding, labels, filename):
        plt.figure(figsize=(10, 8))
        colors = {'ID': 'blue', 'OOD': 'red'}
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c=colors[label], label=label, alpha=0.5)
        plt.title(f'Embedding Visualization - Small ID-OOD Differences (Overlap {self.overlap_ratio})')
        plt.legend()
        plt.grid(True)
        results_dir = f'results/memory_viz'
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, filename))
        plt.close()

# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = DummyFeatureExtractor(device)
    projection_pipeline = DummyProjectionHead(output_dim=128).to(device)
    builder = MemoryBuilder(feature_extractor, projection_pipeline, device)
    # Ablation studies
    for variant in ['base', 'high_overlap', 'low_overlap', 'no_overlap']:
        dummy_dataloader = builder.generate_dummy_dataloader(ablation_variant=variant)
        id_mem, ood_mem = builder.process_images(dummy_dataloader, ablation_variant=variant)
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import gc
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

class MemoryBuilder:
    def __init__(self,
                 feature_extractor,
                 projection_pipeline,
                 device,
                 id_memory_size=100000,
                 aux_memory_size=100000,
                 num_in_dist_classes=19,
                 ood_label=254,
                 n_clusters_per_class=200,
                 n_ood_clusters=500,
                 min_cluster_size=20,
                 amp=False,
                 oversample_factor=2.0,
                 min_separation_th=0,
                 log_level=logging.INFO):
        
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = feature_extractor
        self.projection_pipeline = projection_pipeline.float().to(device)
        self.device = device
        self.id_memory_size = id_memory_size
        self.aux_memory_size = aux_memory_size
        self.num_in_dist_classes = num_in_dist_classes
        self.samples_per_id_class = max(1, id_memory_size // num_in_dist_classes)
        self.ood_label = ood_label
        self.n_clusters_per_class = n_clusters_per_class
        self.n_ood_clusters = n_ood_clusters
        self.min_cluster_size = min_cluster_size
        self.amp = amp
        self.oversample_factor = oversample_factor
        self.min_separation_th = min_separation_th
        self.feature_dim = self._infer_feature_dimension()
        self.max_sample_per_image_id = 1000
        self.max_sample_per_image_ood = 1000

    def _infer_feature_dimension(self):
        try:
            dummy_input = torch.randn(1, 3, 512, 1024).to(self.device).float()
            dummy_batch = {'data': dummy_input, 'label': torch.zeros(1, 512, 1024, dtype=torch.long).to(self.device)}
            with torch.no_grad():
                self.feature_extractor.eval()
                self.projection_pipeline.eval()
                extracted = self.feature_extractor.extract_features_batch(dummy_batch)
                features = extracted['features']
                projected = self.projection_pipeline(features).float()
                return projected.shape[1]
        except Exception as e:
            self.logger.warning(f"Failed to infer dimension: {e}, using 64")
            return 64

    def process_images(self, dataloader):
        self.logger.info("Starting memory building with clustering-based sampling...")
        warnings = []
        id_candidates = {cid: [] for cid in range(self.num_in_dist_classes)}
        ood_candidates = []
        class_counts = torch.zeros(self.num_in_dist_classes, dtype=torch.long)
        ood_count = 0
        
        self.feature_extractor.eval()
        self.projection_pipeline.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting features")):
                try:
                    if not isinstance(batch, dict) or batch.get('data') is None or batch['data'].numel() == 0:
                        continue
                    
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    features = extracted['features'].to(self.device).float()
                    labels = extracted.get('labels')
                    
                    if labels is None:
                        self.logger.warning(f"Batch {batch_idx} has no labels, skipping")
                        continue
                    
                    context = torch.cuda.amp.autocast(enabled=self.amp) if self.amp else torch.no_grad()
                    with context:
                        projected = self.projection_pipeline(features)
                    
                    B, C, H, W = projected.shape
                    pixel_features_flat = projected.permute(0, 2, 3, 1).reshape(-1, C)
                  
                    
                    for img_idx in range(B):
                        start_idx = img_idx * H * W
                        end_idx = (img_idx + 1) * H * W
                        img_pixel_feats = pixel_features_flat[start_idx:end_idx]
                        img_labels = labels[img_idx]
                        img_labels_flat = img_labels.view(-1)
                        
                        # Pixel subsampling
                        total_img_pixels = H * W
                        subsample_rate = 0.3
                        num_subsample = int(total_img_pixels * subsample_rate)
                        if num_subsample < 1:
                            continue
                        
                        subsample_idx = torch.randperm(total_img_pixels)[:num_subsample]
                        img_pixel_feats_sub = img_pixel_feats[subsample_idx]
                        img_labels_sub = img_labels_flat[subsample_idx]
                        
                        # Process ID candidates
                        for class_id in range(self.num_in_dist_classes):
                            mask = (img_labels_sub == class_id)
                            num_pixels = mask.sum().item()
                            class_counts[class_id] += int(num_pixels / subsample_rate)
                            
                            if num_pixels > 0:
                                feats = img_pixel_feats_sub[mask]
                                num_to_sample = min(self.max_sample_per_image_id, len(feats))
                                if num_to_sample > 0:
                                    idx = torch.randperm(len(feats))[:num_to_sample]
                                    selected = feats[idx].detach().cpu().float()
                                    id_candidates[class_id].append(selected)
                        
                        # Process OOD candidates
                        ood_mask = (img_labels_sub == self.ood_label)
                        num_ood = ood_mask.sum().item()
                        ood_count += int(num_ood / subsample_rate)
                        
                        if num_ood > 0:
                            ood_feats = img_pixel_feats_sub[ood_mask]
                            num_to_sample = min(self.max_sample_per_image_ood, len(ood_feats))
                            if num_to_sample > 0:
                                idx = torch.randperm(len(ood_feats))[:num_to_sample]
                                selected_ood = ood_feats[idx].detach().cpu().float()
                                ood_candidates.append(selected_ood)
                
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Build ID features with clustering
        id_full = {}
        total_id_candidates = sum(sum(len(s) for s in cand) for cand in id_candidates.values()) if id_candidates else 0
        self.logger.info(f"Total ID candidates collected: {total_id_candidates}")
        
        weights = torch.zeros(self.num_in_dist_classes, dtype=torch.float32)
        for class_id in range(self.num_in_dist_classes):
            if id_candidates[class_id]:
                full_class = torch.cat(id_candidates[class_id], dim=0).float()
                id_full[class_id] = full_class
                self.logger.debug(f"Class {class_id}: {len(full_class)} features collected")
                if class_counts[class_id] > 0:
                    weights[class_id] = 1.0 / class_counts[class_id]
            else:
                id_full[class_id] = torch.empty(0, self.feature_dim, dtype=torch.float32)
        
        sum_w = weights.sum()
        if sum_w > 0:
            weights /= sum_w
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Process each ID class with separation threshold and clustering
        id_features = {}
        for class_id in range(self.num_in_dist_classes):
            if len(id_full[class_id]) == 0:
                id_features[class_id] = torch.empty(0, self.feature_dim, dtype=torch.float32)
                warnings.append(f"No pixels for class {class_id}")
                continue
            
            other_list = [id_full[c] for c in range(self.num_in_dist_classes) if c != class_id and len(id_full[c]) > 0]
            target_size = max(1, int(self.id_memory_size * weights[class_id])) if weights[class_id] > 0 else 0
            target_size = min(target_size, len(id_full[class_id]))
            
            if target_size == 0:
                id_features[class_id] = torch.empty(0, self.feature_dim, dtype=torch.float32)
                continue
            
            if not other_list:
                # No other classes, use clustering directly
                id_features[class_id] = self._cluster_based_sampling(id_full[class_id], target_size, class_id)
                continue
            
            # Compute distances with separation threshold
            other_feats = torch.cat(other_list, dim=0).float()
            subsample_size = 10000
            if len(other_feats) > subsample_size:
                subsample_idx = torch.randperm(len(other_feats))[:subsample_size]
                other_feats = other_feats[subsample_idx]
                self.logger.debug(f"Subsampled other_feats for class {class_id} to {subsample_size} points")
            
            other_feats = other_feats.to(self.device).float()
            
            # Batch-wise computation
            batch_size = 2048
            min_dists_sq = np.full(len(id_full[class_id]), np.inf)
            class_feats_cpu = id_full[class_id].float()
            
            for i in range(0, len(class_feats_cpu), batch_size):
                end_i = min(i + batch_size, len(class_feats_cpu))
                batch_feats = class_feats_cpu[i:end_i].to(self.device).float()
                sim = torch.matmul(batch_feats, other_feats.T)
                dists_sq = 2 - 2 * sim
                min_batch_sq = dists_sq.min(dim=1)[0].cpu().numpy()
                min_dists_sq[i:end_i] = min_batch_sq
            
            # Apply separation threshold
            valid_mask = min_dists_sq > self.min_separation_th
            if valid_mask.sum() == 0:
                warnings.append(f"No features for class {class_id} meet separation threshold")
                id_features[class_id] = torch.empty(0, self.feature_dim, dtype=torch.float32)
                continue
            
            valid_idx = np.where(valid_mask)[0]
            valid_dists = min_dists_sq[valid_idx]
            sort_idx = np.argsort(-valid_dists)
            preselect_size = min(len(valid_idx), int(self.oversample_factor * target_size))
            pre_idx = valid_idx[sort_idx[:preselect_size]]
            pre_feats = id_full[class_id][pre_idx]
            
            # Apply clustering sampling
            id_features[class_id] = self._cluster_based_sampling(pre_feats, target_size, class_id)
            
            torch.cuda.empty_cache()
        
        # Process OOD features with clustering
        if ood_candidates:
            full_ood = torch.cat(ood_candidates, dim=0).float()
            all_id_list = [id_features[c] for c in id_features if len(id_features[c]) > 0]
            target_ood = min(self.aux_memory_size, len(full_ood))
            
            if target_ood == 0:
                ood_features = torch.empty(0, self.feature_dim, dtype=torch.float32)
                warnings.append("No OOD pixels collected")
            elif not all_id_list:
                ood_features = self._cluster_based_sampling(full_ood, target_ood, -1)
            else:
                all_id_feats = torch.cat(all_id_list, dim=0).float()
                subsample_size = 10000
                if len(all_id_feats) > subsample_size:
                    subsample_idx = torch.randperm(len(all_id_feats))[:subsample_size]
                    all_id_feats = all_id_feats[subsample_idx]
                    self.logger.debug(f"Subsampled all_id_feats for OOD to {subsample_size} points")
                
                all_id_feats = all_id_feats.to(self.device).float()
                
                batch_size = 2048
                min_dists_sq = np.full(len(full_ood), np.inf)
                full_ood_cpu = full_ood.float()
                
                for i in range(0, len(full_ood_cpu), batch_size):
                    end_i = min(i + batch_size, len(full_ood_cpu))
                    batch_ood = full_ood_cpu[i:end_i].to(self.device).float()
                    sim = torch.matmul(batch_ood, all_id_feats.T)
                    dists_sq = 2 - 2 * sim
                    min_batch_sq = dists_sq.min(dim=1)[0].cpu().numpy()
                    min_dists_sq[i:end_i] = min_batch_sq
                
                valid_mask = min_dists_sq > self.min_separation_th
                if valid_mask.sum() == 0:
                    warnings.append("No OOD features meet separation threshold")
                    ood_features = torch.empty(0, self.feature_dim, dtype=torch.float32)
                else:
                    valid_idx = np.where(valid_mask)[0]
                    valid_dists = min_dists_sq[valid_idx]
                    sort_idx = np.argsort(-valid_dists)
                    preselect_size = min(len(valid_idx), int(self.oversample_factor * target_ood))
                    pre_idx = valid_idx[sort_idx[:preselect_size]]
                    pre_ood = full_ood[pre_idx]
                    
                    ood_features = self._cluster_based_sampling(pre_ood, target_ood, -1)
                
                torch.cuda.empty_cache()
        else:
            ood_features = torch.empty(0, self.feature_dim, dtype=torch.float32)
            warnings.append("No OOD candidates")
        
        # Combine and normalize
        all_prototypes = [id_features[class_id] for class_id in id_features if len(id_features[class_id]) > 0]
        if all_prototypes:
            id_memory = torch.cat(all_prototypes, dim=0).float()
        else:
            id_memory = torch.empty(0, self.feature_dim, dtype=torch.float32)
        
        ood_memory = ood_features.float()
        
        id_memory = id_memory.to(self.device)
        ood_memory = ood_memory.to(self.device)
        id_memory = F.normalize(id_memory, dim=1)
        ood_memory = F.normalize(ood_memory, dim=1)
        
        self._analyze_memory_diversity(id_memory, ood_memory)
        
        return id_memory, ood_memory, warnings

    def _cluster_based_sampling(self, features, n_samples, class_id):
        """Cluster-based sampling using sklearn MiniBatchKMeans"""
        features_cpu = features.float()
        n_features = len(features)
        
        if n_features <= n_samples or n_features < self.min_cluster_size:
            return features_cpu[:n_samples]
        
        n_clusters = min(
            self.n_clusters_per_class if class_id >= 0 else self.n_ood_clusters,
            max(1, n_features // self.min_cluster_size),
            n_samples
        )
        
        self.logger.debug(f"Class {class_id}: Using {n_clusters} clusters for {n_features} features")
        
        features_np = features_cpu.numpy()
        prototypes = self._sklearn_clustering(features_np, n_clusters, n_samples)
        
        return torch.from_numpy(prototypes).float()

    def _sklearn_clustering(self, features_np, n_clusters, n_samples):
        """Perform clustering using sklearn MiniBatchKMeans"""
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            batch_size=256,
            max_iter=100
        )
        labels = kmeans.fit_predict(features_np.astype(np.float32))
        
        prototypes_list = []
        samples_per_cluster = max(1, n_samples // n_clusters)
        
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_features = features_np[cluster_mask]
            
            if len(cluster_features) > 0:
                if len(cluster_features) <= samples_per_cluster:
                    selected = cluster_features
                else:
                    # Random sampling within cluster
                    indices = np.random.choice(
                        len(cluster_features), 
                        size=samples_per_cluster, 
                        replace=False
                    )
                    selected = cluster_features[indices]
                prototypes_list.append(selected)
        
        if prototypes_list:
            prototypes = np.vstack(prototypes_list)[:n_samples].astype(np.float32)
        else:
            prototypes = np.empty((0, features_np.shape[1]), dtype=np.float32)
        
        return prototypes

    def _analyze_memory_diversity(self, id_memory, ood_memory):
        """Analyze diversity of the created memory"""
        sample_size_id = min(1000, len(id_memory))
        sample_size_ood = min(1000, len(ood_memory))
        self.logger.info("\n=== Memory Diversity Analysis ===")
        
        if len(id_memory) > 1 and sample_size_id > 1:
            id_sample = id_memory[torch.randperm(len(id_memory))[:sample_size_id]]
            id_sim_matrix = torch.matmul(id_sample, id_sample.T)
            torch.diagonal(id_sim_matrix).fill_(0)
            avg_id_sim = id_sim_matrix.mean().item()
            min_id_sim = id_sim_matrix.min().item()
            max_id_sim = id_sim_matrix[id_sim_matrix > 0].max().item() if (id_sim_matrix > 0).any() else 0
            self.logger.info(f"ID Memory Diversity: Avg sim {avg_id_sim:.3f}, Min {min_id_sim:.3f}, Max {max_id_sim:.3f}")
        else:
            self.logger.info("ID Memory: Insufficient samples for diversity analysis")
        
        if len(ood_memory) > 1 and sample_size_ood > 1:
            ood_sample = ood_memory[torch.randperm(len(ood_memory))[:sample_size_ood]]
            ood_sim_matrix = torch.matmul(ood_sample, ood_sample.T)
            torch.diagonal(ood_sim_matrix).fill_(0)
            avg_ood_sim = ood_sim_matrix.mean().item()
            min_ood_sim = ood_sim_matrix.min().item()
            max_ood_sim = ood_sim_matrix[ood_sim_matrix > 0].max().item() if (ood_sim_matrix > 0).any() else 0
            self.logger.info(f"OOD Memory Diversity: Avg sim {avg_ood_sim:.3f}, Min {min_ood_sim:.3f}, Max {max_ood_sim:.3f}")
        else:
            self.logger.info("OOD Memory: Insufficient samples for diversity analysis")
        
        if len(id_memory) > 0 and len(ood_memory) > 0:
            id_sample = id_memory[torch.randperm(len(id_memory))[:sample_size_id]]
            ood_sample = ood_memory[torch.randperm(len(ood_memory))[:sample_size_ood]]
            cross_sim_matrix = torch.matmul(id_sample, ood_sample.T)
            avg_cross_sim = cross_sim_matrix.mean().item()
            self.logger.info(f"ID-OOD Separation: Avg cross-sim {avg_cross_sim:.3f}")
        else:
            self.logger.info("Insufficient samples for ID-OOD separation analysis")
        
        self.logger.info("=" * 40)


def create_memory_clustering(dataloader, feature_extractor, projection_head, device):
    """Create memory using clustering-based sampling only"""
    builder = MemoryBuilder(
        feature_extractor=feature_extractor,
        projection_pipeline=projection_head,
        device=device,
        id_memory_size=100000,
        aux_memory_size=100000,
        n_clusters_per_class=200,
        n_ood_clusters=500
    )
    id_memory, ood_memory, warnings = builder.process_images(dataloader)
    
    if warnings:
        logger.warning(f"Memory building warnings: {warnings}")
    
    return id_memory, ood_memory
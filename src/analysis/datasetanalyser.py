#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import os
import json
from datetime import datetime
import warnings
import sys
from scipy.spatial.distance import pdist, squareform

sys.path.append('/home/ha51dybi/PEBAL/code') 
from config.config import config
from engine.engine import Engine
from dataset.data_loader import get_mix_loader
from feature_extractor import FeatureExtractor
from projection_head import SimpleProjectionHead

warnings.filterwarnings("ignore")


class RealMemoryQualityTester:
    """Comprehensive tester for real memory building quality and dataset analysis"""
    
    def __init__(self, feature_extractor, projection_pipeline, device, save_dir="quality_test_results"):
        self.feature_extractor = feature_extractor
        self.projection_pipeline = projection_pipeline
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Quality analysis results
        self.test_results = {
            'dataset_quality': {},
            'class_diversity': {},
            'ood_diversity': {},
            'feature_quality': {},
            'memory_efficiency': {},
            'recommendations': []
        }
        
        # Semantic importance weights for validation
        self.importance_weights = {
            11: 4.0, 12: 4.0, 13: 3.0, 14: 5.0, 15: 5.0, 16: 6.0, 17: 5.0,
            0: 2.0, 1: 2.0, 5: 3.0, 6: 3.0, 7: 3.0
        }
        
        # Semantically inconsistent pairs for OOD detection
        self.inconsistent_pairs = [
            (0, 10), (2, 13), (8, 13), (10, 1), (0, 2), (10, 13)
        ]
        
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 
            'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 
            'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]

    def run_quality_tests(self, dataloader, max_batches=200):
        """Run comprehensive quality tests on the dataset and memory building process"""
        print("Starting Real Memory Quality Testing")
        print("=" * 70)
        
        # Test 1: Dataset Quality Assessment
        print("\nTest 1: Dataset Quality Assessment")
        self._test_dataset_quality(dataloader, max_batches)
        
        # Test 2: Class Diversity Analysis
        print("\nTest 2: Class Diversity Analysis")
        self._test_class_diversity(dataloader, max_batches)
        
        # Test 3: OOD Diversity Assessment
        print("\nTest 3: OOD Diversity Assessment")
        self._test_ood_diversity(dataloader, max_batches)
        
        # Test 4: Feature Quality Validation
        print("\nTest 4: Feature Quality Validation")
        self._test_feature_quality(dataloader, max_batches)
        
        # Test 5: Memory Efficiency Analysis
        print("\nTest 5: Memory Efficiency Analysis")
        self._test_memory_efficiency(dataloader, max_batches)
        
        # Generate comprehensive report
        print("\nTest 6: Generating Quality Reports")
        self._generate_quality_reports()
        
        print(f"\nQuality testing complete! Results saved to {self.save_dir}")
        return self.test_results

    def _test_dataset_quality(self, dataloader, max_batches):
        """Test overall dataset quality and balance"""
        print("    Testing dataset quality...")
        
        total_images = 0
        class_pixel_counts = defaultdict(int)
        ood_pixel_counts = defaultdict(int)
        spatial_consistency_scores = []
        label_quality_issues = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Dataset Quality", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                
                try:
                    if not isinstance(batch, dict) or 'data' not in batch:
                        continue
                    
                    labels = batch.get('label')
                    if labels is None:
                        continue
                    
                    for img_idx in range(labels.shape[0]):
                        total_images += 1
                        label = labels[img_idx]
                        
                        # Count pixels per class
                        unique_labels, counts = torch.unique(label, return_counts=True)
                        for class_id, count in zip(unique_labels, counts):
                            class_id = int(class_id)
                            if 0 <= class_id < 19:
                                class_pixel_counts[class_id] += int(count)
                            else:
                                ood_pixel_counts[class_id] += int(count)
                        
                        # Test spatial consistency
                        consistency = self._calculate_spatial_consistency(label)
                        spatial_consistency_scores.append(consistency)
                        
                        # Check for labeling issues
                        issues = self._detect_labeling_issues(label)
                        if issues:
                            label_quality_issues.extend(issues)
                            
                except Exception as e:
                    continue
        
        # Calculate dataset quality metrics
        quality_metrics = {
            'total_images_tested': total_images,
            'class_balance_score': self._calculate_class_balance_score(class_pixel_counts),
            'spatial_consistency_mean': np.mean(spatial_consistency_scores) if spatial_consistency_scores else 0,
            'labeling_issues_count': len(label_quality_issues),
            'rare_classes_identified': self._identify_rare_classes(class_pixel_counts),
            'ood_label_diversity': len(ood_pixel_counts)
        }
        
        self.test_results['dataset_quality'] = quality_metrics
        self._print_dataset_quality_summary(quality_metrics)

    def _test_class_diversity(self, dataloader, max_batches):
        """Test intra-class and inter-class diversity"""
        print("    Testing class diversity...")
        
        class_features = defaultdict(list)
        boundary_pixel_counts = defaultdict(int)
        context_diversity_scores = defaultdict(list)
        
        self.feature_extractor.model.eval()
        self.projection_pipeline.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Class Diversity", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                
                try:
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    if not extracted or 'features' not in extracted:
                        continue
                    
                    features, labels = extracted['features'], extracted['labels']
                    projected = self.projection_pipeline(features)
                    
                    for j in range(features.shape[0]):
                        diversity_metrics = self._analyze_class_diversity(
                            projected[j], labels[j], class_features, 
                            boundary_pixel_counts, context_diversity_scores
                        )
                        
                except Exception as e:
                    continue
        
        # Calculate diversity metrics
        diversity_metrics = {
            'intra_class_diversity': self._calculate_intra_class_diversity(class_features),
            'inter_class_separation': self._calculate_inter_class_separation(class_features),
            'boundary_richness': self._calculate_boundary_richness(boundary_pixel_counts),
            'context_diversity': {k: np.mean(v) if v else 0 for k, v in context_diversity_scores.items()},
            'similar_class_confusion': self._assess_similar_class_confusion(class_features)
        }
        
        self.test_results['class_diversity'] = diversity_metrics
        self._print_class_diversity_summary(diversity_metrics)

    def _test_ood_diversity(self, dataloader, max_batches):
        """Test OOD sample diversity and quality"""
        print("    Testing OOD diversity...")
        
        ood_sources = {
            'true_anomalies': [],
            'boundary_anomalies': [],
            'uncertainty_regions': [],
            'semantic_inconsistencies': []
        }
        
        ood_spatial_distributions = []
        ood_intensity_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="OOD Diversity", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                
                try:
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    if not extracted or 'features' not in extracted:
                        continue
                    
                    features, labels = extracted['features'], extracted['labels']
                    projected = self.projection_pipeline(features)
                    
                    for j in range(features.shape[0]):
                        ood_analysis = self._analyze_ood_diversity(
                            projected[j], labels[j], ood_sources,
                            ood_spatial_distributions, ood_intensity_scores
                        )
                        
                except Exception as e:
                    continue
        
        # Calculate OOD diversity metrics
        ood_metrics = {
            'source_diversity': {k: len(v) for k, v in ood_sources.items()},
            'total_ood_samples': sum(len(v) for v in ood_sources.values()),
            'spatial_distribution_score': np.mean(ood_spatial_distributions) if ood_spatial_distributions else 0,
            'ood_intensity_mean': np.mean(ood_intensity_scores) if ood_intensity_scores else 0,
            'source_balance': self._calculate_ood_source_balance(ood_sources),
            'anomaly_type_coverage': self._assess_anomaly_type_coverage(ood_sources)
        }
        
        self.test_results['ood_diversity'] = ood_metrics
        self._print_ood_diversity_summary(ood_metrics)

    def _test_feature_quality(self, dataloader, max_batches):
        """Test feature extraction and projection quality"""
        print("    Testing feature quality...")
        
        feature_stats = []
        projection_stats = []
        normalization_scores = []
        gradient_magnitudes = []
        
        self.feature_extractor.model.eval()
        self.projection_pipeline.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Feature Quality", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                
                try:
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    if not extracted or 'features' not in extracted:
                        continue
                    
                    features = extracted['features']
                    projected = self.projection_pipeline(features)
                    
                    # Analyze feature quality
                    feature_stats.append({
                        'mean': float(features.mean()),
                        'std': float(features.std()),
                        'range': float(features.max() - features.min())
                    })
                    
                    projection_stats.append({
                        'mean': float(projected.mean()),
                        'std': float(projected.std()),
                        'range': float(projected.max() - projected.min())
                    })
                    
                    # Check normalization quality
                    norms = torch.norm(projected, dim=1, keepdim=True)
                    normalization_scores.append(float(torch.mean(torch.abs(norms - 1.0))))
                    
                    # Analyze feature discriminability
                    grad_mag = torch.norm(torch.gradient(projected, dim=(2, 3))[0], dim=1).mean()
                    gradient_magnitudes.append(float(grad_mag))
                    
                except Exception as e:
                    continue
        
        # Calculate feature quality metrics
        feature_quality = {
            'feature_stability': np.std([s['std'] for s in feature_stats]),
            'projection_quality': np.mean([s['range'] for s in projection_stats]),
            'normalization_error': np.mean(normalization_scores),
            'discriminability_score': np.mean(gradient_magnitudes),
            'feature_range_consistency': np.std([s['range'] for s in feature_stats])
        }
        
        self.test_results['feature_quality'] = feature_quality
        self._print_feature_quality_summary(feature_quality)

    def _test_memory_efficiency(self, dataloader, max_batches):
        """Test memory building efficiency and quality"""
        print("    Testing memory efficiency...")
        
        # Simulate memory building with quality tracking
        id_memory_quality = defaultdict(list)
        ood_memory_quality = []
        sampling_efficiency = defaultdict(int)
        
        target_id_size = 3000
        target_ood_size = 1500
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Memory Efficiency", total=max_batches)):
                if batch_idx >= max_batches:
                    break
                
                try:
                    extracted = self.feature_extractor.extract_features_batch(batch)
                    if not extracted or 'features' not in extracted:
                        continue
                    
                    features, labels = extracted['features'], extracted['labels']
                    projected = self.projection_pipeline(features)
                    
                    for j in range(features.shape[0]):
                        efficiency_metrics = self._analyze_memory_efficiency(
                            projected[j], labels[j], id_memory_quality,
                            ood_memory_quality, sampling_efficiency
                        )
                        
                except Exception as e:
                    continue
        
        # Calculate efficiency metrics
        efficiency_metrics = {
            'id_memory_utilization': sum(len(v) for v in id_memory_quality.values()) / target_id_size,
            'ood_memory_utilization': len(ood_memory_quality) / target_ood_size,
            'class_balance_efficiency': self._calculate_memory_balance_efficiency(id_memory_quality),
            'sampling_success_rate': self._calculate_sampling_success_rate(sampling_efficiency),
            'memory_diversity_score': self._calculate_memory_diversity_score(id_memory_quality, ood_memory_quality)
        }
        
        self.test_results['memory_efficiency'] = efficiency_metrics
        self._print_memory_efficiency_summary(efficiency_metrics)

    # Helper methods for quality calculations
    
    def _calculate_spatial_consistency(self, label):
        """Calculate spatial consistency of labels"""
        h, w = label.shape
        consistency_score = 0
        total_pixels = 0
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = label[i, j]
                neighbors = label[i-1:i+2, j-1:j+2]
                same_class_ratio = (neighbors == center).float().mean()
                consistency_score += same_class_ratio
                total_pixels += 1
        
        return consistency_score / total_pixels if total_pixels > 0 else 0

    def _detect_labeling_issues(self, label):
        """Detect potential labeling issues"""
        issues = []
        h, w = label.shape
        
        # Check for impossible adjacencies
        for i in range(h-1):
            for j in range(w-1):
                current = int(label[i, j])
                right = int(label[i, j+1])
                bottom = int(label[i+1, j])
                
                for pair in self.inconsistent_pairs:
                    if ((current == pair[0] and right == pair[1]) or
                        (current == pair[1] and right == pair[0]) or
                        (current == pair[0] and bottom == pair[1]) or
                        (current == pair[1] and bottom == pair[0])):
                        issues.append(f"Inconsistent adjacency: {pair} at ({i},{j})")
        
        return issues[:10]  # Limit to prevent overflow

    def _calculate_class_balance_score(self, class_counts):
        """Calculate how balanced the classes are"""
        if not class_counts:
            return 0
        
        counts = [class_counts.get(i, 0) for i in range(19)]
        total = sum(counts)
        if total == 0:
            return 0
        
        expected_ratio = 1.0 / 19
        actual_ratios = [c / total for c in counts]
        balance_score = 1.0 - np.std(actual_ratios) / expected_ratio
        return max(0, balance_score)

    def _identify_rare_classes(self, class_counts):
        """Identify classes that are significantly underrepresented"""
        if not class_counts:
            return []
        
        total_pixels = sum(class_counts.values())
        expected_per_class = total_pixels / 19
        threshold = expected_per_class * 0.1  # Less than 10% of expected
        
        rare_classes = []
        for class_id in range(19):
            count = class_counts.get(class_id, 0)
            if count < threshold:
                rare_classes.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'count': count,
                    'expected': expected_per_class
                })
        
        return rare_classes

    def _analyze_class_diversity(self, img_features, img_labels, class_features, 
                               boundary_counts, context_scores):
        """Analyze diversity within and between classes for a single image"""
        features_flat = img_features.permute(1, 2, 0).reshape(-1, img_features.shape[0])
        labels_flat = img_labels.reshape(-1)
        h, w = img_labels.shape
        
        # Sample features for diversity analysis (limit to prevent memory issues)
        for class_id in range(19):
            class_mask = (labels_flat == class_id)
            if class_mask.sum() > 0:
                indices = torch.where(class_mask)[0]
                # Sample up to 50 features per class per image
                if len(indices) > 50:
                    sampled_indices = indices[torch.randperm(len(indices))[:50]]
                else:
                    sampled_indices = indices
                
                features_sample = features_flat[sampled_indices].cpu().numpy()
                class_features[class_id].extend(features_sample.tolist())
                
                # Count boundary pixels
                boundary_count = self._count_boundary_pixels(labels_flat, class_id, h, w)
                boundary_counts[class_id] += boundary_count
                
                # Calculate context diversity
                if len(features_sample) > 1:
                    diversity = np.mean(pdist(features_sample))
                    context_scores[class_id].append(diversity)

    def _count_boundary_pixels(self, labels_flat, class_id, h, w):
        """Count boundary pixels for a specific class"""
        labels_2d = labels_flat.reshape(h, w).cpu().numpy()
        boundary_count = 0
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if labels_2d[i, j] == class_id:
                    # Check if any neighbor is different
                    neighbors = labels_2d[i-1:i+2, j-1:j+2]
                    if not np.all(neighbors == class_id):
                        boundary_count += 1
        
        return boundary_count

    def _calculate_intra_class_diversity(self, class_features):
        """Calculate diversity within each class"""
        diversity_scores = {}
        
        for class_id in range(19):
            features = class_features.get(class_id, [])
            if len(features) > 1:
                # Limit features to prevent memory issues
                if len(features) > 1000:
                    features = np.random.choice(len(features), 1000, replace=False)
                    features = [class_features[class_id][i] for i in features]
                
                feature_array = np.array(features)
                if feature_array.shape[0] > 1:
                    distances = pdist(feature_array)
                    diversity_scores[class_id] = np.mean(distances)
                else:
                    diversity_scores[class_id] = 0
            else:
                diversity_scores[class_id] = 0
        
        return diversity_scores

    def _calculate_inter_class_separation(self, class_features):
        """Calculate separation between different classes"""
        separation_scores = {}
        
        # Test separation for critical class pairs
        critical_pairs = [(13, 14), (14, 15), (13, 15)]  # car-truck, truck-bus, car-bus
        
        for class1, class2 in critical_pairs:
            features1 = class_features.get(class1, [])
            features2 = class_features.get(class2, [])
            
            if len(features1) > 0 and len(features2) > 0:
                # Sample features to prevent memory issues
                f1_sample = features1[:100] if len(features1) > 100 else features1
                f2_sample = features2[:100] if len(features2) > 100 else features2
                
                f1_array = np.array(f1_sample)
                f2_array = np.array(f2_sample)
                
                # Calculate minimum distance between classes
                min_distances = []
                for f1 in f1_array:
                    distances_to_f2 = [np.linalg.norm(f1 - f2) for f2 in f2_array]
                    min_distances.append(min(distances_to_f2))
                
                separation_scores[f"{class1}-{class2}"] = np.mean(min_distances)
            else:
                separation_scores[f"{class1}-{class2}"] = 0
        
        return separation_scores

    def _calculate_boundary_richness(self, boundary_counts):
        """Calculate how rich the boundary information is"""
        total_boundaries = sum(boundary_counts.values())
        if total_boundaries == 0:
            return 0
        
        # Classes that should have rich boundaries
        important_boundaries = [0, 1, 2, 13, 11]  # road, sidewalk, building, car, person
        important_boundary_count = sum(boundary_counts.get(c, 0) for c in important_boundaries)
        
        return important_boundary_count / total_boundaries

    def _assess_similar_class_confusion(self, class_features):
        """Assess potential confusion between similar classes"""
        confusion_risks = {}
        
        # Define similar class groups
        similar_groups = [
            [13, 14, 15],  # car, truck, bus
            [12, 17, 18],  # rider, motorcycle, bicycle
            [5, 6, 7]      # pole, traffic_light, traffic_sign
        ]
        
        for group in similar_groups:
            group_features = []
            group_labels = []
            
            for class_id in group:
                features = class_features.get(class_id, [])
                if features:
                    sample = features[:50] if len(features) > 50 else features
                    group_features.extend(sample)
                    group_labels.extend([class_id] * len(sample))
            
            if len(group_features) > 0 and len(set(group_labels)) > 1:
                # Calculate within-group vs between-group distances
                feature_array = np.array(group_features)
                confusion_score = self._calculate_confusion_score(feature_array, group_labels)
                confusion_risks[f"group_{'-'.join(map(str, group))}"] = confusion_score
        
        return confusion_risks

    def _calculate_confusion_score(self, features, labels):
        """Calculate confusion score for a group of similar classes"""
        if len(features) < 2:
            return 0
        
        within_class_distances = []
        between_class_distances = []
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                distance = np.linalg.norm(features[i] - features[j])
                if labels[i] == labels[j]:
                    within_class_distances.append(distance)
                else:
                    between_class_distances.append(distance)
        
        if not within_class_distances or not between_class_distances:
            return 0
        
        # Higher score means more confusion (within-class > between-class distances)
        return np.mean(within_class_distances) / np.mean(between_class_distances)

    def _analyze_ood_diversity(self, img_features, img_labels, ood_sources,
                             spatial_distributions, intensity_scores):
        """Analyze OOD diversity for a single image"""
        features_flat = img_features.permute(1, 2, 0).reshape(-1, img_features.shape[0])
        labels_flat = img_labels.reshape(-1)
        h, w = img_labels.shape
        
        # True anomalies (label 254)
        true_anomaly_mask = (labels_flat == 254)
        if true_anomaly_mask.sum() > 0:
            anomaly_features = features_flat[true_anomaly_mask].cpu().numpy()[:50]  # Limit samples
            ood_sources['true_anomalies'].extend(anomaly_features.tolist())
        
        # Boundary anomalies
        boundary_anomalies = self._extract_boundary_anomalies(features_flat, labels_flat, h, w)
        ood_sources['boundary_anomalies'].extend(boundary_anomalies)
        
        # Uncertainty regions
        uncertainty_features = self._extract_uncertainty_features(features_flat, labels_flat, h, w)
        ood_sources['uncertainty_regions'].extend(uncertainty_features)
        
        # Calculate spatial distribution score
        if true_anomaly_mask.sum() > 0:
            spatial_score = self._calculate_spatial_distribution_score(true_anomaly_mask, h, w)
            spatial_distributions.append(spatial_score)
        
        # Calculate intensity score
        if true_anomaly_mask.sum() > 0:
            intensity = float(true_anomaly_mask.sum()) / (h * w)
            intensity_scores.append(intensity)

    def _extract_boundary_anomalies(self, features_flat, labels_flat, h, w):
        """Extract features from semantically inconsistent boundaries"""
        labels_2d = labels_flat.reshape(h, w).cpu().numpy()
        boundary_features = []
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center_label = labels_2d[i, j]
                
                # Check neighbors for inconsistent pairs
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        neighbor_label = labels_2d[i + di, j + dj]
                        
                        for pair in self.inconsistent_pairs:
                            if ((center_label == pair[0] and neighbor_label == pair[1]) or
                                (center_label == pair[1] and neighbor_label == pair[0])):
                                
                                idx = i * w + j
                                boundary_features.append(features_flat[idx].cpu().numpy().tolist())
                                break
        
        return boundary_features[:20]  # Limit samples

    def _extract_uncertainty_features(self, features_flat, labels_flat, h, w):
        """Extract features from high uncertainty regions"""
        labels_2d = labels_flat.reshape(h, w).cpu().numpy()
        uncertainty_features = []
        
        kernel_size = 5
        for i in range(kernel_size//2, h - kernel_size//2):
            for j in range(kernel_size//2, w - kernel_size//2):
                patch = labels_2d[i-kernel_size//2:i+kernel_size//2+1, 
                               j-kernel_size//2:j+kernel_size//2+1]
                
                unique_labels = np.unique(patch)
                if len(unique_labels) > 3:  # High diversity
                    idx = i * w + j
                    uncertainty_features.append(features_flat[idx].cpu().numpy().tolist())
        
        return uncertainty_features[:15]  # Limit samples

    def _calculate_spatial_distribution_score(self, anomaly_mask, h, w):
        """Calculate how well distributed anomalies are spatially"""
        anomaly_2d = anomaly_mask.reshape(h, w).cpu().numpy()
        
        # Divide image into quadrants and check distribution
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            anomaly_2d[:mid_h, :mid_w],
            anomaly_2d[:mid_h, mid_w:],
            anomaly_2d[mid_h:, :mid_w],
            anomaly_2d[mid_h:, mid_w:]
        ]
        
        quadrant_counts = [np.sum(q) for q in quadrants]
        total_anomalies = sum(quadrant_counts)
        
        if total_anomalies == 0:
            return 0
        
        # Calculate distribution uniformity
        expected_per_quadrant = total_anomalies / 4
        distribution_score = 1.0 - np.std(quadrant_counts) / expected_per_quadrant
        return max(0, distribution_score)

    def _calculate_ood_source_balance(self, ood_sources):
        """Calculate balance between different OOD sources"""
        counts = [len(v) for v in ood_sources.values()]
        total = sum(counts)
        
        if total == 0:
            return 0
        
        expected_ratio = 1.0 / len(ood_sources)
        actual_ratios = [c / total for c in counts]
        balance_score = 1.0 - np.std(actual_ratios) / expected_ratio
        return max(0, balance_score)

    def _assess_anomaly_type_coverage(self, ood_sources):
        """Assess how well different types of anomalies are covered"""
        coverage_score = 0
        total_sources = len(ood_sources)
        
        for source, samples in ood_sources.items():
            if len(samples) > 0:
                coverage_score += 1
        
        return coverage_score / total_sources

    def _analyze_memory_efficiency(self, img_features, img_labels, id_memory_quality,
                                 ood_memory_quality, sampling_efficiency):
        """Analyze memory building efficiency for a single image"""
        features_flat = img_features.permute(1, 2, 0).reshape(-1, img_features.shape[0])
        labels_flat = img_labels.reshape(-1)
        
        # Simulate intelligent sampling for ID classes
        for class_id in range(19):
            class_mask = (labels_flat == class_id)
            if class_mask.sum() > 0:
                sampling_efficiency['class_encounters'] += 1
                
                # Simulate rare class handling
                if class_id in [14, 15, 16, 17]:  # truck, bus, train, motorcycle
                    sample_count = min(int(class_mask.sum()), 50)  # Aggressive sampling
                    sampling_efficiency['rare_class_samples'] += sample_count
                else:
                    sample_count = min(int(class_mask.sum()), 10)  # Standard sampling
                    sampling_efficiency['common_class_samples'] += sample_count
                
                # Add sample features for quality analysis
                if len(id_memory_quality[class_id]) < 200:  # Limit per class
                    indices = torch.where(class_mask)[0][:sample_count]
                    sample_features = features_flat[indices].cpu().numpy()
                    id_memory_quality[class_id].extend(sample_features.tolist())
        
        # Simulate OOD collection
        ood_mask = (labels_flat == 254) | (labels_flat < 0) | (labels_flat >= 19)
        if ood_mask.sum() > 0:
            sampling_efficiency['ood_encounters'] += 1
            ood_sample_count = min(int(ood_mask.sum()), 30)
            sampling_efficiency['ood_samples'] += ood_sample_count
            
            # Add OOD features for quality analysis
            if len(ood_memory_quality) < 500:  # Limit total OOD samples
                indices = torch.where(ood_mask)[0][:ood_sample_count]
                ood_features = features_flat[indices].cpu().numpy()
                ood_memory_quality.extend(ood_features.tolist())

    def _calculate_memory_balance_efficiency(self, id_memory_quality):
        """Calculate how efficiently memory is balanced across classes"""
        class_counts = {k: len(v) for k, v in id_memory_quality.items()}
        if not class_counts:
            return 0
        
        # Apply importance weighting
        weighted_scores = []
        for class_id in range(19):
            count = class_counts.get(class_id, 0)
            importance = self.importance_weights.get(class_id, 1.0)
            
            # Score based on importance-weighted representation
            if class_id in [14, 15, 16, 17]:  # Rare but critical classes
                target = 200  # Higher target for rare classes
            else:
                target = 100  # Standard target
            
            efficiency = min(count / target, 1.0) * importance
            weighted_scores.append(efficiency)
        
        return np.mean(weighted_scores)

    def _calculate_sampling_success_rate(self, sampling_efficiency):
        """Calculate overall sampling success rate"""
        total_encounters = sampling_efficiency.get('class_encounters', 1)
        successful_samples = (sampling_efficiency.get('rare_class_samples', 0) + 
                            sampling_efficiency.get('common_class_samples', 0))
        
        return successful_samples / total_encounters if total_encounters > 0 else 0

    def _calculate_memory_diversity_score(self, id_memory_quality, ood_memory_quality):
        """Calculate overall diversity score of the memory"""
        diversity_scores = []
        
        # ID memory diversity
        for class_id, features in id_memory_quality.items():
            if len(features) > 1:
                # Sample to prevent memory issues
                sample_size = min(len(features), 100)
                sample_features = np.array(features[:sample_size])
                if sample_features.shape[0] > 1:
                    distances = pdist(sample_features)
                    diversity_scores.append(np.mean(distances))
        
        # OOD memory diversity
        if len(ood_memory_quality) > 1:
            sample_size = min(len(ood_memory_quality), 100)
            ood_array = np.array(ood_memory_quality[:sample_size])
            if ood_array.shape[0] > 1:
                ood_distances = pdist(ood_array)
                diversity_scores.append(np.mean(ood_distances))
        
        return np.mean(diversity_scores) if diversity_scores else 0

    # Summary printing methods
    
    def _print_dataset_quality_summary(self, metrics):
        print(f"    Dataset Quality Summary:")
        print(f"        Images tested: {metrics.get('total_images_tested', 0)}")
        print(f"        Class balance score: {metrics.get('class_balance_score', 0):.3f}")
        print(f"        Spatial consistency: {metrics.get('spatial_consistency_mean', 0):.3f}")
        print(f"        Labeling issues found: {metrics.get('labeling_issues_count', 0)}")
        print(f"        Rare classes identified: {len(metrics.get('rare_classes_identified', []))}")

    def _print_class_diversity_summary(self, metrics):
        print(f"    Class Diversity Summary:")
        intra_div = metrics.get('intra_class_diversity', {})
        inter_sep = metrics.get('inter_class_separation', {})
        print(f"        Average intra-class diversity: {np.mean(list(intra_div.values())):.3f}")
        print(f"        Critical class separations: {len(inter_sep)} pairs analyzed")
        print(f"        Boundary richness: {metrics.get('boundary_richness', 0):.3f}")
        confusion = metrics.get('similar_class_confusion', {})
        print(f"        Similar class confusion risks: {len(confusion)} groups")

    def _print_ood_diversity_summary(self, metrics):
        print(f"    OOD Diversity Summary:")
        sources = metrics.get('source_diversity', {})
        print(f"        Total OOD samples: {metrics.get('total_ood_samples', 0)}")
        print(f"        Source diversity: {len([s for s in sources.values() if s > 0])}/4 sources active")
        print(f"        Spatial distribution: {metrics.get('spatial_distribution_score', 0):.3f}")
        print(f"        Source balance: {metrics.get('source_balance', 0):.3f}")
        print(f"        Type coverage: {metrics.get('anomaly_type_coverage', 0):.3f}")

    def _print_feature_quality_summary(self, metrics):
        print(f"    Feature Quality Summary:")
        print(f"        Feature stability: {metrics.get('feature_stability', 0):.3f}")
        print(f"        Projection quality: {metrics.get('projection_quality', 0):.3f}")
        print(f"        Normalization error: {metrics.get('normalization_error', 0):.4f}")
        print(f"        Discriminability score: {metrics.get('discriminability_score', 0):.3f}")

    def _print_memory_efficiency_summary(self, metrics):
        print(f"    Memory Efficiency Summary:")
        print(f"        ID memory utilization: {metrics.get('id_memory_utilization', 0):.1%}")
        print(f"        OOD memory utilization: {metrics.get('ood_memory_utilization', 0):.1%}")
        print(f"        Class balance efficiency: {metrics.get('class_balance_efficiency', 0):.3f}")
        print(f"        Sampling success rate: {metrics.get('sampling_success_rate', 0):.3f}")
        print(f"        Memory diversity score: {metrics.get('memory_diversity_score', 0):.3f}")

    def _generate_quality_reports(self):
        """Generate comprehensive quality reports"""
        
        # Generate recommendations based on test results
        self._generate_recommendations()
        
        # Save detailed JSON report
        report_path = os.path.join(self.save_dir, f"quality_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=4, default=str)
        
        # Generate visualizations
        self._create_quality_visualizations()
        
        # Generate summary report
        self._generate_quality_summary()
        
        print(f"    Quality reports saved to {self.save_dir}")

    def _generate_recommendations(self):
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Dataset quality recommendations
        dataset_quality = self.test_results.get('dataset_quality', {})
        if dataset_quality.get('class_balance_score', 0) < 0.5:
            recommendations.append("Poor class balance detected. Consider using importance-weighted sampling.")
        
        if dataset_quality.get('labeling_issues_count', 0) > 100:
            recommendations.append("High number of labeling issues found. Review dataset quality.")
        
        # Class diversity recommendations
        class_diversity = self.test_results.get('class_diversity', {})
        confusion_risks = class_diversity.get('similar_class_confusion', {})
        high_confusion = [k for k, v in confusion_risks.items() if v > 1.5]
        if high_confusion:
            recommendations.append(f"High confusion risk for groups: {', '.join(high_confusion)}. Use contrastive learning.")
        
        # OOD diversity recommendations
        ood_diversity = self.test_results.get('ood_diversity', {})
        if ood_diversity.get('source_balance', 0) < 0.3:
            recommendations.append("Poor OOD source balance. Implement multi-source collection strategy.")
        
        # Feature quality recommendations
        feature_quality = self.test_results.get('feature_quality', {})
        if feature_quality.get('normalization_error', 1) > 0.1:
            recommendations.append("High normalization error. Add explicit L2 normalization to projection head.")
        
        # Memory efficiency recommendations
        memory_efficiency = self.test_results.get('memory_efficiency', {})
        if memory_efficiency.get('id_memory_utilization', 0) < 0.5:
            recommendations.append("Low ID memory utilization. Increase sampling rates or reduce target size.")
        
        if memory_efficiency.get('class_balance_efficiency', 0) < 0.4:
            recommendations.append("Poor class balance efficiency. Implement semantic importance weighting.")
        
        self.test_results['recommendations'] = recommendations

    def _create_quality_visualizations(self):
        """Create quality assessment visualizations"""
        
        # Class diversity visualization
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Class balance
        plt.subplot(2, 3, 1)
        class_diversity = self.test_results.get('class_diversity', {})
        intra_diversity = class_diversity.get('intra_class_diversity', {})
        if intra_diversity:
            classes = [self.class_names[i] for i in range(19)]
            diversity_scores = [intra_diversity.get(i, 0) for i in range(19)]
            plt.bar(classes, diversity_scores)
            plt.title('Intra-Class Diversity')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Diversity Score')
        
        # Subplot 2: Inter-class separation
        plt.subplot(2, 3, 2)
        inter_separation = class_diversity.get('inter_class_separation', {})
        if inter_separation:
            pairs = list(inter_separation.keys())
            scores = list(inter_separation.values())
            plt.bar(pairs, scores)
            plt.title('Inter-Class Separation')
            plt.ylabel('Separation Score')
            plt.xticks(rotation=45)
        
        # Subplot 3: OOD source distribution
        plt.subplot(2, 3, 3)
        ood_diversity = self.test_results.get('ood_diversity', {})
        source_diversity = ood_diversity.get('source_diversity', {})
        if source_diversity:
            sources = list(source_diversity.keys())
            counts = list(source_diversity.values())
            plt.pie(counts, labels=sources, autopct='%1.1f%%')
            plt.title('OOD Source Distribution')
        
        # Subplot 4: Feature quality metrics
        plt.subplot(2, 3, 4)
        feature_quality = self.test_results.get('feature_quality', {})
        if feature_quality:
            metrics = ['stability', 'projection_quality', 'discriminability_score']
            values = [feature_quality.get(f'feature_{m}', 0) if m == 'stability' 
                     else feature_quality.get(m, 0) for m in metrics]
            plt.bar(metrics, values)
            plt.title('Feature Quality Metrics')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
        
        # Subplot 5: Memory efficiency
        plt.subplot(2, 3, 5)
        memory_efficiency = self.test_results.get('memory_efficiency', {})
        if memory_efficiency:
            metrics = ['id_memory_utilization', 'ood_memory_utilization', 'class_balance_efficiency']
            values = [memory_efficiency.get(m, 0) for m in metrics]
            labels = ['ID Memory', 'OOD Memory', 'Class Balance']
            plt.bar(labels, values)
            plt.title('Memory Efficiency')
            plt.ylabel('Efficiency Score')
            plt.xticks(rotation=45)
        
        # Subplot 6: Overall quality score
        plt.subplot(2, 3, 6)
        overall_scores = {
            'Dataset': self.test_results.get('dataset_quality', {}).get('class_balance_score', 0),
            'Diversity': np.mean(list(intra_diversity.values())) if intra_diversity else 0,
            'OOD': ood_diversity.get('source_balance', 0),
            'Features': feature_quality.get('discriminability_score', 0),
            'Memory': memory_efficiency.get('class_balance_efficiency', 0)
        }
        plt.bar(overall_scores.keys(), overall_scores.values())
        plt.title('Overall Quality Scores')
        plt.ylabel('Quality Score')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'quality_assessment.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_quality_summary(self):
        """Generate a comprehensive quality summary report"""
        summary_path = os.path.join(self.save_dir, 'quality_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" Real Memory Builder - Quality Assessment Report\n")
            f.write("="*80 + "\n\n")
            
            # Overall assessment
            f.write("OVERALL ASSESSMENT\n")
            f.write("-"*50 + "\n")
            
            # Calculate overall quality score
            scores = []
            scores.append(self.test_results.get('dataset_quality', {}).get('class_balance_score', 0))
            scores.append(self.test_results.get('ood_diversity', {}).get('source_balance', 0))
            scores.append(self.test_results.get('feature_quality', {}).get('discriminability_score', 0))
            scores.append(self.test_results.get('memory_efficiency', {}).get('class_balance_efficiency', 0))
            
            overall_score = np.mean([s for s in scores if s > 0])
            f.write(f"Overall Quality Score: {overall_score:.3f}/1.000\n")
            
            if overall_score >= 0.8:
                f.write("Assessment: EXCELLENT - Ready for production use\n")
            elif overall_score >= 0.6:
                f.write("Assessment: GOOD - Minor optimizations recommended\n")
            elif overall_score >= 0.4:
                f.write("Assessment: FAIR - Significant improvements needed\n")
            else:
                f.write("Assessment: POOR - Major restructuring required\n")
            
            f.write("\n")
            
            # Detailed breakdown
            f.write("DETAILED BREAKDOWN\n")
            f.write("-"*50 + "\n")
            
            # Dataset Quality
            dataset_quality = self.test_results.get('dataset_quality', {})
            f.write(f"Dataset Quality:\n")
            f.write(f"  - Class Balance: {dataset_quality.get('class_balance_score', 0):.3f}\n")
            f.write(f"  - Spatial Consistency: {dataset_quality.get('spatial_consistency_mean', 0):.3f}\n")
            f.write(f"  - Labeling Issues: {dataset_quality.get('labeling_issues_count', 0)}\n\n")
            
            # Class Diversity
            class_diversity = self.test_results.get('class_diversity', {})
            f.write(f"Class Diversity:\n")
            intra_div = class_diversity.get('intra_class_diversity', {})
            if intra_div:
                f.write(f"  - Average Intra-Class Diversity: {np.mean(list(intra_div.values())):.3f}\n")
            f.write(f"  - Boundary Richness: {class_diversity.get('boundary_richness', 0):.3f}\n\n")
            
            # OOD Quality
            ood_diversity = self.test_results.get('ood_diversity', {})
            f.write(f"OOD Quality:\n")
            f.write(f"  - Total OOD Samples: {ood_diversity.get('total_ood_samples', 0)}\n")
            f.write(f"  - Source Balance: {ood_diversity.get('source_balance', 0):.3f}\n")
            f.write(f"  - Type Coverage: {ood_diversity.get('anomaly_type_coverage', 0):.3f}\n\n")
            
            # Memory Efficiency
            memory_efficiency = self.test_results.get('memory_efficiency', {})
            f.write(f"Memory Efficiency:\n")
            f.write(f"  - ID Utilization: {memory_efficiency.get('id_memory_utilization', 0):.1%}\n")
            f.write(f"  - OOD Utilization: {memory_efficiency.get('ood_memory_utilization', 0):.1%}\n")
            f.write(f"  - Balance Efficiency: {memory_efficiency.get('class_balance_efficiency', 0):.3f}\n\n")
            
            # Recommendations
            recommendations = self.test_results.get('recommendations', [])
            if recommendations:
                f.write("RECOMMENDATIONS\n")
                f.write("-"*50 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            f.write("Next Steps:\n")
            f.write("1. Review quality assessment visualizations\n")
            f.write("2. Implement recommended improvements\n")
            f.write("3. Re-run quality tests after modifications\n")
            f.write("4. Proceed with memory building if scores are satisfactory\n")


def main():
    """Main function to run the quality testing pipeline"""
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    model_path = "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth"
    cs_root = "/home/ha51dybi/PEBAL/cityscapes"
    coco_root = "/home/ha51dybi/PEBAL/coco"
    
    # Setup configuration
    config.batch_size = 4
    config.image_height = 512
    config.image_width = 1024
    config.num_workers = 4
    config.train_scale_array = [0.75, 1.0, 1.25]
    config.image_mean = [0.485, 0.456, 0.406]
    config.image_std = [0.229, 0.224, 0.225]
    
    print("Initializing quality testing components...")
    
    # Initialize components
    feature_extractor = FeatureExtractor(
        model_path=model_path,
        device=device,
        num_classes=19
    )
    
    projection_head = SimpleProjectionHead(1280, 128).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'projection_head_state_dict' in checkpoint:
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        print("Loaded projection head weights from checkpoint.")
    
    # Setup engine and data loader
    class Args:
        gpus, local_rank, nodes, ddp, world_size = 1, -1, 1, False, 1
    
    engine = Engine(custom_arg=Args(), logger=None, continue_state_object=checkpoint)
    train_loader, _, _ = get_mix_loader(engine=engine, augment=True, cs_root=cs_root, coco_root=coco_root)
    
    print("Initialization complete.")
    
    # Run quality tests
    tester = RealMemoryQualityTester(
        feature_extractor=feature_extractor,
        projection_pipeline=projection_head,
        device=device
    )
    
    results = tester.run_quality_tests(train_loader, max_batches=200)
    
    print("\n" + "="*70)
    print("Quality testing completed successfully!")
    print("Check the 'quality_test_results' directory for detailed reports.")


if __name__ == "__main__":
    main()
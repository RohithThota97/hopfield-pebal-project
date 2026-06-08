#!/usr/bin/env python3
# deepwv3_layer_explorer.py - Evaluate all layers of DeepWV3Plus for OOD detection

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns
from datetime import datetime

from model.mynn import *
from model.wide_resnet_base import WiderResNetA2
from model.wide_network import DeepWV3Plus
class DeepWV3PlusLayerExplorer:
    def __init__(self, model_path=None, num_classes=19, device=None):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        self.num_classes = num_classes
        
        # Create timestamp for output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"layer_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup model and hooks
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Dictionary to store features from different layers
        self.layer_features = {}
        
        # Define all layers to explore in DeepWV3Plus
        self.layers_config = [
            # Main backbone layers
            ('mod1', self.model.mod1),
            ('mod2', self.model.mod2),
            ('mod3', self.model.mod3),
            ('mod4', self.model.mod4),
            ('mod5', self.model.mod5),
            ('mod6', self.model.mod6),
            ('mod7', self.model.mod7),
            # ASPP module
            ('aspp', self.model.aspp),
            # Decoder parts
            ('bot_fine', self.model.bot_fine),
            ('bot_aspp', self.model.bot_aspp),
            ('final', self.model.final)
        ]
        
        # Register hooks for all layers
        self._register_hooks()
        
        # Sub-layer hooks for ASPP (since it's critical)
        if hasattr(self.model, 'aspp') and hasattr(self.model.aspp, 'features'):
            for i, feature in enumerate(self.model.aspp.features):
                layer_name = f'aspp.feature{i}'
                self.layers_config.append((layer_name, feature))
                feature.register_forward_hook(self._get_hook_fn(layer_name))
            
            # Also hook the image pooling branch
            if hasattr(self.model.aspp, 'img_conv'):
                layer_name = 'aspp.img_conv'
                self.layers_config.append((layer_name, self.model.aspp.img_conv))
                self.model.aspp.img_conv.register_forward_hook(self._get_hook_fn(layer_name))
        
        # Metrics storage
        self.results = {}
    
    def _load_model(self, model_path):
        model = DeepWV3Plus(self.num_classes)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Clean state dict keys if needed
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]
                    cleaned_state_dict[k] = v
                
                model.load_state_dict(cleaned_state_dict, strict=False)
                print(f"Successfully loaded checkpoint from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        
        model = model.to(self.device)
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def _get_hook_fn(self, layer_name):
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                self.layer_features[layer_name] = output[0].detach()
            else:
                self.layer_features[layer_name] = output.detach()
        return hook
    
    def _register_hooks(self):
        for name, layer in self.layers_config:
            try:
                layer.register_forward_hook(self._get_hook_fn(name))
                print(f"Registered hook on {name}")
            except Exception as e:
                print(f"Error registering hook for {name}: {e}")
    
    def generate_dummy_data(self, batch_size=8, img_size=(512, 512), ood_percentage=0.3):
        """Generate dummy data with known ID and OOD pixels"""
        # Create synthetic input images
        images = torch.randn(batch_size, 3, img_size[0], img_size[1], device=self.device)
        
        # Create synthetic segmentation labels
        # 0-18: ID classes, 254: OOD, 255: ignore
        labels = torch.randint(0, self.num_classes, 
                               (batch_size, img_size[0], img_size[1]), 
                               device=self.device)
        
        # Create OOD regions (simulate anomalies) with realistic shapes
        for i in range(batch_size):
            # Create 1-3 random OOD regions per image
            num_regions = np.random.randint(1, 4)
            for _ in range(num_regions):
                # Random center
                cy = np.random.randint(0, img_size[0])
                cx = np.random.randint(0, img_size[1])
                
                # Random size
                size = np.random.randint(30, 100)
                
                # Create irregular OOD region (more realistic)
                y_indices, x_indices = torch.meshgrid(
                    torch.arange(img_size[0], device=self.device),
                    torch.arange(img_size[1], device=self.device)
                )
                
                # Base mask
                base_mask = ((y_indices - cy)**2 + (x_indices - cx)**2 < size**2)
                
                # Add noise to the boundary for irregularity
                noise = torch.randn(img_size[0], img_size[1], device=self.device) * (size * 0.2)
                noisy_mask = ((y_indices - cy)**2 + (x_indices - cx)**2 < (size**2 + noise))
                
                # Combine for final mask
                mask = base_mask | (noisy_mask & (torch.rand_like(noisy_mask.float()) > 0.5))
                
                labels[i][mask] = 254  # OOD label
                
                # Make the OOD region visually different in the image
                # Use more structured patterns for realism
                if torch.rand(1).item() > 0.5:
                    # Striped pattern
                    stripe_pattern = torch.sin(x_indices[mask].float() * 0.2) > 0
                    images[i, 0, mask] = stripe_pattern.float() * 2 - 1
                    images[i, 1, mask] = (~stripe_pattern).float() * 2 - 1
                    images[i, 2, mask] = torch.randn_like(images[i, 2, mask]) * 0.5
                else:
                    # Gradient pattern
                    dist_from_center = torch.sqrt((y_indices[mask] - cy).float()**2 + (x_indices[mask] - cx).float()**2)
                    normalized_dist = dist_from_center / size
                    images[i, 0, mask] = normalized_dist * 2 - 1
                    images[i, 1, mask] = 1 - normalized_dist * 2
                    images[i, 2, mask] = torch.randn_like(images[i, 2, mask]) * 0.5
        
        # Add some ignore regions
        ignore_mask = torch.rand_like(labels.float()) < 0.05
        labels[ignore_mask] = 255
        
        return {'data': images, 'label': labels}
    
    def extract_and_evaluate(self, batch):
        """Extract features from all layers and evaluate for OOD detection"""
        images = batch['data']
        labels = batch['label']
        
        # Forward pass through model to trigger hooks
        with torch.no_grad():
            _ = self.model(images)
        
        results = {}
        
        # Analyze features from each layer
        for layer_name in self.layer_features.keys():
            features = self.layer_features[layer_name]
            
            # Handle different output dimensions 
            if len(features.shape) == 4:  # [B, C, H, W]
                # Resize labels to match feature size if needed
                if labels.shape[-2:] != features.shape[-2:]:
                    resized_labels = F.interpolate(
                        labels.unsqueeze(1).float(), 
                        size=features.shape[-2:], 
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    resized_labels = labels
                
                # Calculate metrics
                layer_metrics = self._calculate_metrics(features, resized_labels, layer_name)
                results[layer_name] = layer_metrics
                
                # Visualize feature distribution
                self._visualize_feature_distribution(features, resized_labels, layer_name)
                
                # Visualize energy maps
                self._visualize_energy_maps(features, resized_labels, layer_name)
                
                # Visualize feature maps
                self._visualize_feature_maps(features, resized_labels, layer_name)
            
            else:
                print(f"Skipping {layer_name} with shape {features.shape} (not spatial features)")
        
        self.results = results
        return results
    
    def _calculate_metrics(self, features, labels, layer_name):
        """Calculate metrics for OOD detection quality"""
        batch_size, channels, height, width = features.shape
        
        # Create masks
        id_mask = (labels >= 0) & (labels < self.num_classes)
        ood_mask = (labels == 254)
        ignore_mask = (labels == 255)
        
        # Skip if no OOD pixels
        if ood_mask.sum() == 0:
            return {
                'error': 'No OOD pixels in labels',
                'feature_dim': channels,
                'spatial_dim': (height, width)
            }
        
        # Flatten spatial dimensions for analysis
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)  # [B*H*W, C]
        id_mask_flat = id_mask.reshape(-1)
        ood_mask_flat = ood_mask.reshape(-1)
        valid_mask = (id_mask_flat | ood_mask_flat)
        
        # Skip ignore pixels
        features_valid = features_flat[valid_mask]
        labels_valid = torch.zeros_like(id_mask_flat[valid_mask], dtype=torch.float32)
        labels_valid[ood_mask_flat[valid_mask]] = 1.0  # 1 for OOD, 0 for ID
        
        # Calculate intra-class and inter-class distances
        id_features = features_valid[labels_valid == 0]
        ood_features = features_valid[labels_valid == 1]
        
        if len(id_features) == 0 or len(ood_features) == 0:
            return {
                'error': 'Missing ID or OOD features after masking',
                'feature_dim': channels,
                'spatial_dim': (height, width)
            }
        
        # Sample if too many features (for efficiency)
        max_samples = 10000
        if len(id_features) > max_samples:
            id_indices = torch.randperm(len(id_features))[:max_samples]
            id_features = id_features[id_indices]
            
        if len(ood_features) > max_samples:
            ood_indices = torch.randperm(len(ood_features))[:max_samples]
            ood_features = ood_features[ood_indices]
        
        # Normalize features for fair comparison
        id_features_norm = F.normalize(id_features, p=2, dim=1)
        ood_features_norm = F.normalize(ood_features, p=2, dim=1)
        
        # Calculate intra-class cosine similarity
        id_sim = torch.mm(id_features_norm, id_features_norm.t())
        id_sim_mean = (torch.sum(id_sim) - torch.sum(torch.diag(id_sim))) / (id_sim.numel() - id_sim.shape[0])
        
        # Calculate inter-class cosine similarity
        inter_sim = torch.mm(id_features_norm, ood_features_norm.t())
        inter_sim_mean = torch.mean(inter_sim)
        
        # Separation ratio (higher is better)
        separation_ratio = id_sim_mean / (inter_sim_mean + 1e-8)
        
        # Energy-based scoring for OOD detection (multiple methods)
        
        # 1. Basic logsumexp energy
        energy_scores_basic = -torch.logsumexp(features_valid, dim=1)
        
        # 2. Scaled logsumexp energy (temperature scaling)
        temperature = 1.0  # Can be tuned
        energy_scores_temp = -torch.logsumexp(features_valid / temperature, dim=1)
        
        # 3. L2 norm-based energy (deviation from mean)
        id_mean = id_features.mean(dim=0, keepdim=True)
        l2_dist = torch.norm(features_valid - id_mean, dim=1)
        
        # 4. Mahalanobis distance (if enough samples)
        mahalanobis_scores = None
        if len(id_features) > channels:  # Need more samples than dimensions
            try:
                # Calculate covariance matrix (add small regularization)
                id_centered = id_features - id_mean
                cov = torch.mm(id_centered.t(), id_centered) / (id_centered.size(0) - 1)
                cov += torch.eye(cov.size(0), device=cov.device) * 1e-5
                
                # Calculate inverse covariance matrix
                try:
                    inv_cov = torch.inverse(cov)
                    # Calculate Mahalanobis distance
                    centered = features_valid - id_mean
                    mahalanobis_scores = torch.sqrt(torch.sum(torch.mm(centered, inv_cov) * centered, dim=1))
                except:
                    pass  # Skip if inverse fails
            except:
                pass  # Skip if calculation fails
        
        # Calculate AUC and AP for each energy method
        energy_metrics = {}
        
        # Basic energy
        energy_auc = roc_auc_score(labels_valid.cpu().numpy(), energy_scores_basic.cpu().numpy())
        energy_ap = average_precision_score(labels_valid.cpu().numpy(), energy_scores_basic.cpu().numpy())
        energy_metrics['basic'] = {'auc': energy_auc, 'ap': energy_ap}
        
        # Temperature scaled energy
        temp_energy_auc = roc_auc_score(labels_valid.cpu().numpy(), energy_scores_temp.cpu().numpy())
        temp_energy_ap = average_precision_score(labels_valid.cpu().numpy(), energy_scores_temp.cpu().numpy())
        energy_metrics['temp_scaled'] = {'auc': temp_energy_auc, 'ap': temp_energy_ap}
        
        # L2 norm energy
        l2_auc = roc_auc_score(labels_valid.cpu().numpy(), l2_dist.cpu().numpy())
        l2_ap = average_precision_score(labels_valid.cpu().numpy(), l2_dist.cpu().numpy())
        energy_metrics['l2_norm'] = {'auc': l2_auc, 'ap': l2_ap}
        
        # Mahalanobis energy
        if mahalanobis_scores is not None:
            try:
                mahalanobis_auc = roc_auc_score(labels_valid.cpu().numpy(), mahalanobis_scores.cpu().numpy())
                mahalanobis_ap = average_precision_score(labels_valid.cpu().numpy(), mahalanobis_scores.cpu().numpy())
                energy_metrics['mahalanobis'] = {'auc': mahalanobis_auc, 'ap': mahalanobis_ap}
            except:
                pass
        
        # Find best energy method for this layer
        best_energy_method = max(energy_metrics.items(), key=lambda x: x[1]['auc'])
        
        # Feature statistics
        feature_mean = features_valid.mean().item()
        feature_std = features_valid.std().item()
        feature_min = features_valid.min().item()
        feature_max = features_valid.max().item()
        
        return {
            'feature_dim': channels,
            'spatial_dim': (height, width),
            'id_sim_mean': id_sim_mean.item(),
            'inter_sim_mean': inter_sim_mean.item(),
            'separation_ratio': separation_ratio.item(),
            'energy_metrics': energy_metrics,
            'best_energy': {
                'method': best_energy_method[0],
                'auc': best_energy_method[1]['auc'],
                'ap': best_energy_method[1]['ap']
            },
            'feature_stats': {
                'mean': feature_mean,
                'std': feature_std,
                'min': feature_min,
                'max': feature_max
            }
        }
    
    def _visualize_feature_distribution(self, features, labels, layer_name):
        """Visualize feature distribution using t-SNE"""
        # Sample pixels for visualization (t-SNE is slow for large datasets)
        max_pixels = 5000
        batch_size, channels, height, width = features.shape
    
    # Create masks and flatten
        id_mask = (labels >= 0) & (labels < self.num_classes)
        ood_mask = (labels == 254)
        valid_mask = (id_mask | ood_mask).reshape(-1)
    
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)  # [B*H*W, C]
        features_valid = features_flat[valid_mask]
        labels_valid = torch.zeros_like(valid_mask[valid_mask], dtype=torch.long)
        labels_valid[ood_mask.reshape(-1)[valid_mask]] = 1  # 1 for OOD, 0 for ID
    
    # Sample if too many features
        if len(features_valid) > max_pixels:
            indices = torch.randperm(len(features_valid))[:max_pixels]
            features_sample = features_valid[indices].cpu().numpy()
            labels_sample = labels_valid[indices].cpu().numpy()
        else:
            features_sample = features_valid.cpu().numpy()
            labels_sample = labels_valid.cpu().numpy()
    
    # Skip if too few samples (t-SNE/PCA not meaningful, avoids errors)
        if len(features_sample) < 30:
            print(f"Skipping t-SNE for {layer_name} due to small number of samples ({len(features_sample)})")
            return
    
    # Apply PCA first to reduce dimensionality for very high-dim features
        n_samples, n_features = features_sample.shape
        if n_features > 50:
            from sklearn.decomposition import PCA
            n_comp = min(50, min(n_samples, n_features))  # Dynamic to avoid ValueError
            pca = PCA(n_components=n_comp)
            features_sample = pca.fit_transform(features_sample)
    
    # Apply t-SNE (adjust perplexity to avoid invalid values like 0)
        print(f"Running t-SNE for {layer_name} features...")
        perplexity = min(30, max(1, len(features_sample) // 3))  # Safe range: 1 to 30, scales with size
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features_sample)
    
    # Plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_sample, cmap='coolwarm', 
                         alpha=0.7, s=10)
        legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
        ax = plt.gca()
        ax.add_artist(legend1)
        plt.title(f'Feature Distribution for {layer_name}')
        plt.tight_layout()
    
    # Save plot
        plt.savefig(f'{self.output_dir}/{layer_name.replace(".", "_")}_tsne.png')
        plt.close()
    
    def _visualize_energy_maps(self, features, labels, layer_name):
        """Visualize energy maps for a batch of images"""
        batch_size, channels, height, width = features.shape
        
        # Take first image in batch for visualization
        image_features = features[0]  # [C, H, W]
        image_labels = labels[0]      # [H, W]
        
        # Calculate energy map
        energy_map = -torch.logsumexp(image_features, dim=0)  # [H, W]
        
        # Create OOD mask
        ood_mask = (image_labels == 254)
        id_mask = (image_labels >= 0) & (image_labels < self.num_classes)
        
        # Plot
        plt.figure(figsize=(15, 5))
        
        # Original labels
        plt.subplot(1, 3, 1)
        label_vis = image_labels.clone().cpu().numpy()
        # Create a custom colormap
        cmap = plt.cm.viridis.copy()
        cmap.set_bad('black')
        cmap.set_over('red')  # OOD pixels (254)
        plt.imshow(label_vis, cmap=cmap, vmin=0, vmax=self.num_classes)
        plt.title('Ground Truth Labels')
        plt.colorbar()
        
        # Energy map
        plt.subplot(1, 3, 2)
        plt.imshow(energy_map.cpu().numpy(), cmap='plasma')
        plt.title('Energy Map')
        plt.colorbar()
        
        # Energy map with OOD overlay
        plt.subplot(1, 3, 3)
        energy_viz = energy_map.cpu().numpy()
        plt.imshow(energy_viz, cmap='plasma', alpha=0.7)
        
        # Overlay OOD mask
        ood_overlay = np.zeros_like(energy_viz)
        ood_overlay[ood_mask.cpu().numpy()] = 1
        plt.imshow(ood_overlay, cmap='binary', alpha=0.3)
        plt.title('Energy Map with OOD Overlay')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{layer_name.replace(".", "_")}_energy.png')
        plt.close()
        
        # Save histogram of energy values for ID vs OOD
        plt.figure(figsize=(10, 6))
        
        energy_id = energy_map[id_mask].cpu().numpy().flatten()
        energy_ood = energy_map[ood_mask].cpu().numpy().flatten()
        
        if len(energy_id) > 0 and len(energy_ood) > 0:
            plt.hist([energy_id, energy_ood], bins=50, 
                    label=['ID', 'OOD'], alpha=0.7)
            plt.xlabel('Energy Value')
            plt.ylabel('Frequency')
            plt.title(f'Energy Distribution for {layer_name}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{layer_name.replace(".", "_")}_energy_hist.png')
        plt.close()
    
    def _visualize_feature_maps(self, features, labels, layer_name):
        """Visualize individual feature maps for a selected layer"""
        # Only do this for important layers to avoid too many plots
        if layer_name not in ['aspp', 'mod7', 'final']:
            return
            
        batch_size, channels, height, width = features.shape
        
        # Take first image for visualization
        image_features = features[0]  # [C, H, W]
        
        # Select subset of channels to visualize (max 16)
        num_channels = min(16, channels)
        channel_indices = np.linspace(0, channels-1, num_channels, dtype=int)
        
        # Create grid plot
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        plt.figure(figsize=(15, 15))
        
        for i, idx in enumerate(channel_indices):
            if i >= grid_size * grid_size:
                break
                
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(image_features[idx].cpu().numpy(), cmap='viridis')
            plt.title(f'Channel {idx}')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{layer_name.replace(".", "_")}_feature_maps.png')
        plt.close()
    
    def _generate_correlation_matrix(self):
        """Generate correlation matrix between layer performance metrics"""
        if not self.results:
            print("No results to analyze!")
            return
            
        # Extract metrics for each layer
        layers = []
        metrics = {
            'feature_dim': [],
            'separation_ratio': [],
            'energy_auc': [],
            'feature_mean': [],
            'feature_std': []
        }
        
        for layer_name, result in self.results.items():
            if 'error' in result:
                continue
                
            layers.append(layer_name)
            metrics['feature_dim'].append(result['feature_dim'])
            metrics['separation_ratio'].append(result['separation_ratio'])
            metrics['energy_auc'].append(result['best_energy']['auc'])
            metrics['feature_mean'].append(result['feature_stats']['mean'])
            metrics['feature_std'].append(result['feature_stats']['std'])
        
        # Convert to numpy arrays
        for k, v in metrics.items():
            metrics[k] = np.array(v)
        
        # Calculate correlation matrix
        corr_data = np.vstack([metrics[k] for k in metrics.keys()]).T
        corr_matrix = np.corrcoef(corr_data, rowvar=False)
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   xticklabels=list(metrics.keys()), 
                   yticklabels=list(metrics.keys()))
        plt.title('Correlation Between Layer Metrics')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/metrics_correlation.png')
        plt.close()
    
    def _generate_performance_chart(self):
        """Generate bar chart comparing layer performance"""
        if not self.results:
            return
            
        layers = []
        aucs = []
        aps = []
        separations = []
        
        for layer_name, metrics in self.results.items():
            if 'error' in metrics:
                continue
                
            layers.append(layer_name)
            aucs.append(metrics['best_energy']['auc'])
            aps.append(metrics['best_energy']['ap'])
            separations.append(metrics['separation_ratio'])
        
        # Sort by AUC
        sorted_indices = np.argsort(aucs)[::-1]
        layers = [layers[i] for i in sorted_indices]
        aucs = [aucs[i] for i in sorted_indices]
        aps = [aps[i] for i in sorted_indices]
        separations = [separations[i] for i in sorted_indices]
        
        # Truncate long layer names
        display_layers = [l[:20] if len(l) > 20 else l for l in layers]
        
        plt.figure(figsize=(12, 8))
        x = np.arange(len(layers))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width/2, aucs, width, label='AUC', color='royalblue')
        ax.bar(x + width/2, aps, width, label='AP', color='lightcoral')
        
        # Add a second y-axis for separation ratio
        ax2 = ax.twinx()
        ax2.plot(x, separations, 'go-', label='Separation Ratio', linewidth=2)
        
        ax.set_xlabel('Layers')
        ax.set_ylabel('AUC / AP')
        ax2.set_ylabel('Separation Ratio')
        
        ax.set_title('Layer Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(display_layers, rotation=45, ha='right')
        
        # Create combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/layer_performance_comparison.png')
        plt.close()
    
    
    
    def print_summary(self):
        """Print summary of layer analysis results"""
        print("\n" + "="*100)
        print("DEEPWV3PLUS LAYER ANALYSIS SUMMARY")
        print("="*100)
        
        # Create comparison table
        headers = ["Layer", "Feature Dim", "ID-ID Sim", "ID-OOD Sim", "Separation", "Best Energy", "AUC", "AP"]
        row_format = "{:<20} {:<12} {:<12} {:<12} {:<12} {:<15} {:<8} {:<8}"
        
        print(row_format.format(*headers))
        print("-"*100)
        
        # Sort layers by best energy AUC (higher is better)
        sorted_layers = sorted(
            [(name, metrics) for name, metrics in self.results.items() if 'error' not in metrics],
            key=lambda x: x[1].get('best_energy', {}).get('auc', 0),
            reverse=True
        )
        
        # First print layers with metrics
        for layer_name, metrics in sorted_layers:
            print(row_format.format(
                layer_name,
                str(metrics['feature_dim']),
                f"{metrics['id_sim_mean']:.4f}",
                f"{metrics['inter_sim_mean']:.4f}",
                f"{metrics['separation_ratio']:.4f}",
                metrics['best_energy']['method'],
                f"{metrics['best_energy']['auc']:.4f}",
                f"{metrics['best_energy']['ap']:.4f}"
            ))
        
        # Then print layers with errors
        error_layers = [(name, metrics) for name, metrics in self.results.items() if 'error' in metrics]
        if error_layers:
            print("\nLayers with errors:")
            for layer_name, metrics in error_layers:
                print(f"{layer_name}: {metrics['error']}")
        
        # Find best layer
        if sorted_layers:
            best_layer = sorted_layers[0]
            
            print("\n" + "="*100)
            print(f"BEST LAYER FOR OOD DETECTION: {best_layer[0]}")
            print(f"  Feature Dimensions: {best_layer[1]['feature_dim']}")
            print(f"  Spatial Dimensions: {best_layer[1]['spatial_dim']}")
            print(f"  Separation Ratio: {best_layer[1]['separation_ratio']:.4f}")
            print(f"  Best Energy Method: {best_layer[1]['best_energy']['method']}")
            print(f"  AUC: {best_layer[1]['best_energy']['auc']:.4f}")
            print(f"  AP: {best_layer[1]['best_energy']['ap']:.4f}")
            
            # Show why ASPP is special (if it's not the best layer)
            if 'aspp' in self.results and 'error' not in self.results['aspp'] and best_layer[0] != 'aspp':
                aspp_metrics = self.results['aspp']
                print("\nComparison with ASPP layer:")
                print(f"  ASPP Feature Dimensions: {aspp_metrics['feature_dim']}")
                print(f"  ASPP Separation Ratio: {aspp_metrics['separation_ratio']:.4f}")
                print(f"  ASPP AUC: {aspp_metrics['best_energy']['auc']:.4f}")
                
                # Explain why ASPP might be better despite not having highest AUC
                if aspp_metrics['feature_dim'] < best_layer[1]['feature_dim']:
                    print("  Note: While ASPP has lower AUC, its lower feature dimensionality")
                    print("  makes it more efficient for memory-based methods like Hopfield Networks.")
                
                if hasattr(self.model.aspp, 'features') and len(self.model.aspp.features) > 0:
                    print("  ASPP combines multi-scale context through different dilation rates,")
                    print("  which is particularly valuable for detecting anomalies at different scales.")
        else:
            print("\nNo valid layers found for OOD detection!")
        
        print("="*100)
        
        # Generate correlation matrix
        self._generate_correlation_matrix()
    
    # Create performance comparison chart
        self._generate_performance_chart()
    
        print(f"\nAnalysis complete! Visualizations saved to: {self.output_dir}")
    
    def _visualize_best_layer(self, layer_name):
        """Create additional visualizations for the best layer"""
        # This would generate additional plots specifically for the best layer
        pass

    def run_analysis(self, num_batches=5):
        """Run complete analysis pipeline"""
        print("Starting layer analysis for OOD detection...")
        
        for i in tqdm(range(num_batches), desc="Processing batches"):
            # Generate dummy data with different random seeds
            batch = self.generate_dummy_data(batch_size=4, img_size=(256, 256))
            
            # Extract and evaluate
            batch_results = self.extract_and_evaluate(batch)
            
            # Aggregate results (average across batches)
            if i == 0:
                self.results = batch_results
            else:
                for layer_name, metrics in batch_results.items():
                    if layer_name not in self.results:
                        self.results[layer_name] = metrics
                    else:
                        for k, v in metrics.items():
                            if k in self.results[layer_name] and isinstance(v, (int, float)):
                                self.results[layer_name][k] = (self.results[layer_name][k] * i + v) / (i + 1)
        
        # Print summary
        self.print_summary()
        
        return self.results


if __name__ == "__main__":
    # Path to your checkpoint
    model_path = "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth"  # Change this
    
    # Create explorer and run analysis
    explorer = DeepWV3PlusLayerExplorer(model_path=model_path)
    results = explorer.run_analysis(num_batches=5)
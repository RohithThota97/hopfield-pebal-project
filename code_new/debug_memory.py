import torch
import torch.nn.functional as F
from feature_extractor import FeatureExtractor
from projection_head import SimpleProjectionHead
from hopfield_memory_builder import SimplifiedMemoryBuilder
from engine.engine import Engine
from dataset.data_loader import get_mix_loader
import logging
import os
import numpy as np
from tqdm import tqdm
import argparse
import torch.utils.data as data_utils
import cv2
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_debug")
def debug_memory_builder():
    # Setup paths
    cs_root = "/home/ha51dybi/PEBAL/cityscapes"
    coco_root = "/home/ha51dybi/PEBAL/coco"
    model_path = "/home/ha51dybi/PEBAL/code/ckpts/pretrained_ckpts/cityscapes_best.pth"
   
    # Create engine with proper args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('-l', '--local_rank', default=-1, type=int)
    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args([])
   
    engine_instance = Engine(custom_arg=args, logger=logger, continue_state_object=model_path)
   
    # Create full dataloaders (no subset)
    train_loader, _, _ = get_mix_loader(
        engine=engine_instance, augment=True,
        cs_root=cs_root,
        coco_root=coco_root,
    )
   
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Create feature extractor - UPDATED for ASPP only (1280 channels)
    feature_extractor = FeatureExtractor(
        model_path=model_path,
        resize_resolution=(512, 512),
        device=device,
        num_classes=19,
        hybrid=False, # Disable hybrid mode to use ASPP only
        aspp_select=None, # Not needed for ASPP-only mode
        project_dim=None # No projection in feature extractor
    )
   
    # Create projection head - Input dim already 1280 for ASPP
    projection_head = SimpleProjectionHead(
        input_dim=1280, # ASPP output is 1280 channels
        output_dim=128
    ).to(device)
   
    # Debug the feature dimensions
    try:
        dummy_input = {'data': torch.randn(1, 3, 512, 512).to(device)}
        with torch.no_grad():
            feature_extractor.eval()
            extracted = feature_extractor.extract_features_batch(dummy_input)
            aspp_features = extracted['features']
            logger.info(f"ASPP feature shape: {aspp_features.shape}")
           
            projection_head.eval()
            projected = projection_head(aspp_features)
            logger.info(f"Projected feature shape: {projected.shape}")
    except Exception as e:
        logger.error(f"Error during dimension debug: {e}")
   
    # Create simplified memory builder
    memory_builder = SimplifiedMemoryBuilder(
        feature_extractor=feature_extractor,
        projection_pipeline=projection_head,
        device=device,
        id_memory_size=50000,
        aux_memory_size=50000,
        num_in_dist_classes=19,
        target_id_per_class=3000,
        target_ood=50000
    )
   
    # Validate dataset entries
    dataset = train_loader.dataset
    valid_indices = []
    bad_files = []
    for i in tqdm(range(len(dataset)), desc="Validating dataset"):
        try:
            img_path = str(dataset.images[i])
            gt_path = str(dataset.targets[i])
            if not os.path.exists(img_path) or not os.path.exists(gt_path):
                bad_files.append(img_path)
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if img is None or gt is None or img.ndim != 3 or gt.ndim != 2 or img.shape[2] != 3 or img.shape[:2] != gt.shape[:2] or min(img.shape[:2]) == 0:
                bad_files.append(img_path)
                logger.warning(f"Corrupted image skipped during validation: {img_path} (img: {getattr(img, 'shape', 'None')}, gt: {getattr(gt, 'shape', 'None')})")
                continue
            valid_indices.append(i)
        except Exception as e:
            bad_files.append(str(dataset.images[i]) if hasattr(dataset, 'images') else f"Index {i}")
            logger.error(f"Validation error at {i}: {e}")
   
    if bad_files:
        logger.info(f"Found {len(bad_files)} bad files: {bad_files[:10]}...")
    if len(valid_indices) == 0:
        raise ValueError("No valid entries in dataset after filtering.")
   
    # Full dataset analysis with filtered single-threaded loader
    logger.info("Analyzing filtered dataset...")
    filtered_dataset = data_utils.Subset(dataset, valid_indices)
    analysis_loader = data_utils.DataLoader(
        filtered_dataset,
        batch_size=1,  # Set to 1 to isolate bad items
        num_workers=0,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        collate_fn=train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None,
    )
   
    class_counts = {i: 0 for i in range(20)}
    ood_label_count = 0
    ignore_label_count = 0
    batch_stats = []
   
    iterator = iter(analysis_loader)
    batch_idx = 0
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        except Exception as e:
            logger.warning(f"Skipping bad batch load {batch_idx}: {str(e)}")
            batch_idx += 1
            continue
   
        try:
            logger.info(f"=== Batch {batch_idx+1} Analysis ===")
           
            if 'label' in batch:
                label = batch['label']
                unique_labels = torch.unique(label).cpu().numpy()
               
                has_ood = 254 in unique_labels
                ood_pixels = (label == 254).sum().item() if has_ood else 0
                ood_label_count += ood_pixels
               
                has_ignore = 255 in unique_labels
                ignore_pixels = (label == 255).sum().item() if has_ignore else 0
                ignore_label_count += ignore_pixels
               
                batch_class_counts = {}
                for c in range(19):
                    class_pixels = (label == c).sum().item()
                    class_counts[c] += class_pixels
                    batch_class_counts[c] = class_pixels
               
                stats = {
                    'batch_idx': batch_idx,
                    'image_count': len(label),
                    'has_ood': has_ood,
                    'ood_pixels': ood_pixels,
                    'ignore_pixels': ignore_pixels,
                    'unique_labels': unique_labels.tolist(),
                    'class_counts': batch_class_counts
                }
                batch_stats.append(stats)
               
                logger.info(f"Images in batch: {len(label)}")
                logger.info(f"Has OOD label (254): {has_ood}")
                logger.info(f"OOD pixels: {ood_pixels}")
                logger.info(f"Ignore pixels (255): {ignore_pixels}")
                logger.info(f"Unique labels: {unique_labels}")
               
                sorted_classes = sorted(batch_class_counts.items(), key=lambda x: x[1], reverse=True)
                logger.info("Top 5 classes in batch:")
                for cls, count in sorted_classes[:5]:
                    logger.info(f" Class {cls}: {count} pixels")
            else:
                logger.error(f"Batch {batch_idx} does not contain 'label' key")
            batch_idx += 1
        except Exception as e:
            logger.warning(f"Skipping corrupted batch {batch_idx}: {str(e)}")
            if 'fn' in batch:
                logger.info(f"Possibly bad files: {batch['fn']}")
            batch_idx += 1
            continue
   
    logger.info("\n=== Full Dataset Summary ===")
    logger.info(f"Total OOD pixels (254): {ood_label_count}")
    logger.info(f"Total ignore pixels (255): {ignore_label_count}")
   
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    logger.info("Class distribution:")
    for cls, count in sorted_class_counts:
        if count > 0:
            logger.info(f" Class {cls}: {count} pixels")
   
    # Process memory over full loader
    logger.info("\n=== Building Memory Banks ===")
    id_memory, ood_memory = memory_builder.process_images(train_loader)
   
    # Analyze separation
    logger.info("\n=== Memory Separation Analysis ===")
   
    if id_memory is not None and ood_memory is not None:
        logger.info(f"ID memory size: {id_memory.shape}")
        logger.info(f"OOD memory size: {ood_memory.shape}")
       
        id_center = id_memory.mean(dim=0)
        ood_center = ood_memory.mean(dim=0)
       
        center_dist = torch.norm(id_center - ood_center).item()
        logger.info(f"Distance between memory centers: {center_dist:.4f}")
       
        id_spread = torch.norm(id_memory - id_center, dim=1).mean().item()
        ood_spread = torch.norm(ood_memory - ood_center, dim=1).mean().item()
        logger.info(f"ID memory spread: {id_spread:.4f}")
        logger.info(f"OOD memory spread: {ood_spread:.4f}")
       
        separation_ratio = center_dist / (id_spread + ood_spread)
        logger.info(f"Separation ratio: {separation_ratio:.4f}")
       
        id_sample = id_memory[:1000] if len(id_memory) > 1000 else id_memory
        ood_sample = ood_memory[:1000] if len(ood_memory) > 1000 else ood_memory
       
        sim_matrix = torch.matmul(F.normalize(id_sample, p=2, dim=1),
                                  F.normalize(ood_sample, p=2, dim=1).T)
        mean_sim = sim_matrix.mean().item()
        max_sim = sim_matrix.max().item()
        min_sim = sim_matrix.min().item()
       
        logger.info(f"Cross-similarity - Mean: {mean_sim:.4f}, Min: {min_sim:.4f}, Max: {max_sim:.4f}")
       
        sim_flat = sim_matrix.flatten()
        percentiles = [10, 25, 50, 75, 90, 95]
        sim_percentiles = {p: torch.quantile(sim_flat, p/100).item() for p in percentiles}
        logger.info("Similarity percentiles:")
        for p, val in sim_percentiles.items():
            logger.info(f" {p}%: {val:.4f}")
       
        thresholds = [0.7, 0.8, 0.9]
        for thresh in thresholds:
            overlap = (sim_matrix > thresh).float().mean().item() * 100
            logger.info(f"Features with similarity > {thresh}: {overlap:.2f}%")
   
    logger.info("\n=== Memory Building Process Complete ===")
    return id_memory, ood_memory
if __name__ == "__main__":
    id_mem, ood_mem = debug_memory_builder()
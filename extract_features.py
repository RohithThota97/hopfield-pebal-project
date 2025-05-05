import os
import torch
import numpy as np
import random
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.network import Network
from config.config import config
import sys
from dataset.data_loader import Cityscapes, COCO, extract_bboxes
from engine.engine import Engine

def extract_synthetic_ood_features():
    output_dir = '/home/ha51dybi/PEBAL/extracted_synthetic_ood'
    sample_limit = 50
    
   
    dirs = {
        'synthetic_ood': {
            'mod7': os.path.join(output_dir, "synthetic_ood", "mod7"),
            'decoder': os.path.join(output_dir, "synthetic_ood", "decoder"),
            'segmap': os.path.join(output_dir, "synthetic_ood", "segmap"),
        }
    }

    for category in dirs.values():
        for dir_path in category.values():
            os.makedirs(dir_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class Args:
        def __init__(self):
            self.local_rank = -1
            self.ddp = False
            self.gpus = 1
            self.world_size = 1
            self.nodes = 1

    engine_args = Args()
    checkpoint_path = config.pretrained_weight_path

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    import logging
    logger = logging.getLogger("pebal_extractor")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    engine = Engine(custom_arg=engine_args, logger=logger, continue_state_object=checkpoint_path)

    # Create separate datasets for Cityscapes and COCO
    try:
        # Create Cityscapes dataset
        cityscapes_dataset = Cityscapes(root=config.city_root_path, split='train')
        
        # Create COCO dataset
        coco_dataset = COCO(root=config.coco_root_path, proxy_size=1000, split='train')
        
        print(f"Cityscapes dataset size: {len(cityscapes_dataset)}")
        print(f"COCO dataset size: {len(coco_dataset)}")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return

    
    model = Network(config.num_classes, wide=True)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    model.to(device)
    model.eval()

 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.image_mean, std=config.image_std)
    ])

    
    captured_features = {}
    
    def hook_fn(name):
        def fn(module, input, output):
            feature_data = output[0] if isinstance(output, tuple) else output
            captured_features[name] = feature_data.detach()
        return fn

    hooks = []
    try:
        hooks.append(model.branch1.mod7.register_forward_hook(hook_fn('mod7')))
        hooks.append(model.branch1.final.register_forward_hook(hook_fn('decoder')))
        print("Registered hooks on standard layers")
    except AttributeError:
        print("Standard layers not found, searching model structure...")
        for name, module in model.named_modules():
            if 'mod7' in name and 'mod7' not in captured_features:
                hooks.append(module.register_forward_hook(hook_fn('mod7')))
                print(f"Found mod7: {name}")
            elif ('final' in name or 'decoder' in name) and 'decoder' not in captured_features:
                hooks.append(module.register_forward_hook(hook_fn('decoder')))
                print(f"Found decoder: {name}")
    
    if len(captured_features) == 0:
        print("No features captured, check model structure")
        return 
    print("Hooks registered successfully")
    
    def manual_mix_object(city_image, city_label, coco_image, coco_label, save_debug_images=False, debug_index=0):
        
        import cv2
        import os
        

        debug_dir = os.path.join(output_dir, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        
    
        city_image_copy = np.copy(city_image)
        city_label_copy = np.copy(city_label)
        coco_image_copy = np.copy(coco_image)
        coco_label_copy = np.copy(coco_label)
        
        # Print shapes for debugging
        print(f"City image shape: {city_image_copy.shape}")
        print(f"City label shape: {city_label_copy.shape}")
        print(f"COCO image shape: {coco_image_copy.shape}")
        print(f"COCO label shape: {coco_label_copy.shape}")
        print(f"COCO label unique values: {np.unique(coco_label_copy)}")
        
    
        coco_label_copy[coco_label_copy > 0] = 254
        
       
        mask = coco_label_copy == 254
     
        if save_debug_images:
            
            cmap = np.zeros((255, 3), dtype=np.uint8)
            cmap[254] = [255, 0, 0]  # Red for OOD
            
  
            coco_label_viz = np.zeros_like(coco_image_copy)
            for i in range(3):
                coco_label_viz[:, :, i] = np.take(cmap[:, i], coco_label_copy)
                
         
            alpha = 0.5
            coco_overlay = cv2.addWeighted(coco_image_copy, 1-alpha, coco_label_viz, alpha, 0)
            
           
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_coco_image.png"), 
                        cv2.cvtColor(coco_image_copy, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_coco_mask.png"), 
                        coco_label_copy)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_coco_overlay.png"), 
                        cv2.cvtColor(coco_overlay, cv2.COLOR_RGB2BGR))
            
            print(f"Saved COCO debug images with index {debug_index}")

        if not np.any(mask):
            print("No OOD pixels found in COCO image")
            return city_image_copy, city_label_copy
        
        print(f"Number of OOD pixels in COCO: {np.sum(mask)}")
        
  
        mask_3d = np.stack([mask] * 3, axis=2)
        

        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            print("No valid indices found for OOD object")
            return city_image_copy, city_label_copy
            
        y1, y2 = np.min(y_indices), np.max(y_indices) + 1
        x1, x2 = np.min(x_indices), np.max(x_indices) + 1
        
        print(f"OOD object bounding box: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
        

        ood_object = coco_image_copy[y1:y2, x1:x2]
        ood_mask = mask[y1:y2, x1:x2]
        ood_mask_3d = mask_3d[y1:y2, x1:x2]
        
        h_ood, w_ood = ood_object.shape[:2]
        h_city, w_city = city_image_copy.shape[:2]
        
        print(f"OOD object dimensions: h={h_ood}, w={w_ood}")
        print(f"City image dimensions: h={h_city}, w={w_city}")
        
      
        if h_ood > h_city or w_ood > w_city:
            print("OOD object is larger than city image")
            return city_image_copy, city_label_copy
    
        if save_debug_images:
   
            city_label_viz = np.zeros_like(city_image_copy)
            for i in range(3):
            
                if i == 0:  
                    city_label_viz[:, :, i] = (city_label_copy * 10) % 255
                else:
                    city_label_viz[:, :, i] = (city_label_copy * 20) % 255
                
 
            alpha = 0.5
            city_overlay = cv2.addWeighted(city_image_copy, 1-alpha, city_label_viz, alpha, 0)
            
            # Save Cityscapes debug images
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_city_image.png"), 
                        cv2.cvtColor(city_image_copy, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_city_mask.png"), 
                        city_label_copy)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_city_overlay.png"), 
                        cv2.cvtColor(city_overlay, cv2.COLOR_RGB2BGR))
            
            print(f"Saved Cityscapes debug images with index {debug_index}")
        
       
        h_start = random.randint(0, h_city - h_ood)
        w_start = random.randint(0, w_city - w_ood)
        h_end = h_start + h_ood
        w_end = w_start + w_ood
        
        print(f"Placing OOD object at: h_start={h_start}, h_end={h_end}, w_start={w_start}, w_end={w_end}")
        

        region_image = city_image_copy[h_start:h_end, w_start:w_end].copy()
        region_label = city_label_copy[h_start:h_end, w_start:w_end].copy()

        region_image[ood_mask_3d] = ood_object[ood_mask_3d]
        region_label[ood_mask] = 254  
        

        city_image_copy[h_start:h_end, w_start:w_end] = region_image
        city_label_copy[h_start:h_end, w_start:w_end] = region_label
        
        print(f"Number of OOD pixels in mixed image: {np.sum(city_label_copy == 254)}")
   
        if save_debug_images:
            # Create visualization of mixed mask
            mixed_label_viz = np.zeros_like(city_image_copy)
            # Special colormap for mixed image - highlight OOD pixels in red
            for y in range(city_label_copy.shape[0]):
                for x in range(city_label_copy.shape[1]):
                    if city_label_copy[y, x] == 254:  # OOD pixel
                        mixed_label_viz[y, x, 0] = 255  # Red
                    else:
                        # Regular classes
                        mixed_label_viz[y, x, 1] = (city_label_copy[y, x] * 20) % 255  # Green
                        mixed_label_viz[y, x, 2] = (city_label_copy[y, x] * 10) % 255  # Blue
            
            # Create visualization of mixed mask overlaid on image
            alpha = 0.5
            mixed_overlay = cv2.addWeighted(city_image_copy, 1-alpha, mixed_label_viz, alpha, 0)
            
            # Save mixed debug images
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_mixed_image.png"), 
                        cv2.cvtColor(city_image_copy, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_mixed_mask.png"), 
                        city_label_copy)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{debug_index}_mixed_overlay.png"), 
                        cv2.cvtColor(mixed_overlay, cv2.COLOR_RGB2BGR))
            
            print(f"Saved mixed debug images with index {debug_index}")
        
        return city_image_copy, city_label_copy
    
    processed = 0
    

    print("Processing synthetic OOD data...")

    sample_index = 0
    while processed < sample_limit and sample_index < min(len(cityscapes_dataset), len(coco_dataset)):
        try:
            # Get a Cityscapes image
            city_image, city_label = cityscapes_dataset[sample_index % len(cityscapes_dataset)]
            
            # Get a COCO image
            coco_image, coco_label = coco_dataset[sample_index % len(coco_dataset)]
            
            # Convert PIL images to numpy arrays if needed
            if isinstance(city_image, Image.Image):
                city_image_np = np.array(city_image)
            else:
                city_image_np = city_image
                
            if isinstance(city_label, Image.Image):
                city_label_np = np.array(city_label)
            else:
                city_label_np = city_label
                
            if isinstance(coco_image, Image.Image):
                coco_image_np = np.array(coco_image)
            else:
                coco_image_np = coco_image
                
            if isinstance(coco_label, Image.Image):
                coco_label_np = np.array(coco_label)
            else:
                coco_label_np = coco_label
            

            if len(city_label_np.shape) > 2:
                city_label_np = city_label_np[:, :, 0]
            if len(coco_label_np.shape) > 2:
                coco_label_np = coco_label_np[:, :, 0]
            
        
            save_debug = processed < 2 
            mixed_image_np, mixed_label_np = manual_mix_object(
                city_image_np, city_label_np, coco_image_np, coco_label_np,
                save_debug_images=save_debug, debug_index=processed
            )
            
          
            if 254 not in np.unique(mixed_label_np):
                print(f"Failed to mix OOD objects in sample {sample_index}")
                sample_index += 1
                continue
            
      
            mixed_image_tensor = transform(mixed_image_np).unsqueeze(0).to(device)
            
       
            with torch.no_grad():
                _ = model(mixed_image_tensor)
            

            mod7_features = captured_features.get('mod7')
            decoder_features = captured_features.get('decoder')
            
            if mod7_features is None or decoder_features is None:
                print(f"Features not captured correctly for sample {sample_index}")
                sample_index += 1
                continue
            

            features_mod7 = mod7_features[0].cpu().numpy()
            features_decoder = decoder_features[0].cpu().numpy()
            

            mod7_size = (features_mod7.shape[1], features_mod7.shape[2])
            decoder_size = (features_decoder.shape[1], features_decoder.shape[2])
            

            mixed_label_tensor = torch.from_numpy(mixed_label_np).unsqueeze(0).unsqueeze(0).float()
            
            # Downsample using nearest neighbor interpolation
            mixed_label_resized = F.interpolate(
                mixed_label_tensor,
                size=decoder_size,
                mode='nearest'
            ).squeeze().long().numpy()
            
            # Create binary OOD mask
            ood_mask = (mixed_label_resized == 254).astype(np.uint8)
            
            # Verify OOD pixels are present after resizing
            if np.sum(ood_mask) == 0:
                print(f"No OOD pixels found after resizing in sample {sample_index}")
                sample_index += 1
                continue
            
            # Save the extracted features and masks
            base_name = f"synthetic_ood_{processed}"
            metadata = {
                'filename': base_name,
                'has_ood': True,
                'ood_class_id': 254
            }
            
            np.save(os.path.join(dirs['synthetic_ood']['mod7'], f"{base_name}_mod7.npy"), features_mod7)
            np.save(os.path.join(dirs['synthetic_ood']['decoder'], f"{base_name}_decoder.npy"), features_decoder)
            np.save(os.path.join(dirs['synthetic_ood']['segmap'], f"{base_name}_segmap.npy"), mixed_label_resized)
            np.save(os.path.join(dirs['synthetic_ood']['segmap'], f"{base_name}_ood_mask.npy"), ood_mask)
            np.save(os.path.join(dirs['synthetic_ood']['segmap'], f"{base_name}_meta.npy"), metadata)
            
            # Also save the original resolution mask
            np.save(os.path.join(dirs['synthetic_ood']['segmap'], f"{base_name}_original_mask.npy"), mixed_label_np)
            
            processed += 1
            print(f"Saved SYNTHETIC OOD ({processed}/{sample_limit}): {base_name}")
            
        except Exception as e:
            print(f"Error processing sample {sample_index}: {e}")
        
        sample_index += 1
    

    for hook in hooks:
        hook.remove()
    
    print("\n" + "="*30)
    print("Feature extraction complete!")
    print(f"Total images processed: {processed}")
    print(f"Results saved in directory: {output_dir}")
    print("="*30 + "\n")

if __name__ == "__main__":
    extract_synthetic_ood_features()
import os
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

from model.network import Network
from utils.img_utils import Compose, Normalize, ToTensor
from config.config import config
from dataset.data_loader import Cityscapes 

output_dir = "/home/ha51dybi/PEBAL/extracted_cityscapes" 
batch_size = 1
split = "train"
checkpoint_path = config.pretrained_weight_path

IMAGE_LIMIT = 50 

def extract_features():

    mod7_dir = os.path.join(output_dir, "mod7")
    decoder_dir = os.path.join(output_dir, "decoder")
    os.makedirs(mod7_dir, exist_ok=True)
    os.makedirs(decoder_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])
    dataset = Cityscapes(root=config.city_root_path, split=split, transform=transform) # Use Cityscapes dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Cityscapes dataset loaded with {len(dataset)} images (processing max {IMAGE_LIMIT})")

    model = Network(config.num_classes, wide=True)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    processed_count = 0
    captured_features = {}

    def hook_mod7(module, input, output):
            captured_features['mod7'] = output.detach().cpu().numpy()
            print("Mod7 Features:")
            print(f"Shape: {output.shape}")
            print(f"Dtype: {output.dtype}")
            print(f"Min: {output.min()}")
            print(f"Max: {output.max()}")

            

    def capture_final_input(module, input):
        if input and isinstance(input[0], torch.Tensor):
             captured_features['decoder'] = input[0].detach().cpu().numpy()
             print("Decoder Input Features:")
             print(f"Shape: {input[0].shape}")
             print(f"Dtype: {input[0].dtype}")
             print(f"Min: {input[0].min()}")
             print(f"Max: {input[0].max()}")
        else:
             print(f"Warning: Unexpected input format to final layer hook: {type(input)}")
             captured_features.pop('decoder', None)

    for batch_idx, batch in enumerate(dataloader):
        if processed_count >= IMAGE_LIMIT:
            print(f"\nReached image limit ({IMAGE_LIMIT}). Stopping.")
            break
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}")

        images = batch[0].to(device)

        captured_features.clear()

        hooks = []
   
        hooks.append(model.module.branch1.mod7.register_forward_hook(hook_mod7))
        hooks.append(model.module.branch1.final.register_forward_pre_hook(capture_final_input))
        
        
        with torch.no_grad():
            _ = model(images)

        for hook in hooks:
            hook.remove()

        if processed_count < IMAGE_LIMIT:
            prefix = "cityscapes" 
            filename_base = f"{prefix}_{processed_count:06d}"

            mod7_data = captured_features.get('mod7')
            decoder_data = captured_features.get('decoder')

            if mod7_data is not None:
                save_path_mod7 = os.path.join(mod7_dir, f"{filename_base}_mod7.npy")
               
                np.save(save_path_mod7, mod7_data[0])
                
            else:
                print(f"Warning: mod7 features not captured for image {processed_count}")

            if decoder_data is not None:
                save_path_decoder = os.path.join(decoder_dir, f"{filename_base}_decoder.npy")
                
                np.save(save_path_decoder, decoder_data[0])
                
            else:
                print(f"Warning: Decoder features not captured for image {processed_count}")

            processed_count += 1

    print(f"\nFinished processing. Total images processed: {processed_count}")

if __name__ == "__main__":
    extract_features()
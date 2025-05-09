import torch
import torch.nn.functional as F
import logging
import argparse


from config.config import config
from model.network import Network
from engine.engine import Engine
from dataset.data_loader import get_mix_loader

logger = logging.getLogger("feature_extractor")

class FeatureExtractor:
    def __init__(self, model_path=None, target_layer='dec0'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.target_layer = target_layer
        self.available_layers = ['mod7', 'aspp', 'dec0', 'final_output']
        
        if self.target_layer not in self.available_layers:
            logger.warning(f"Invalid target layer '{target_layer}'. Using 'dec0' instead.")
            self.target_layer = 'dec0'
            
        logger.info(f"Using target layer: {self.target_layer}")
        

        self.engine = self._setup_engine()
        self.model = self.load_model(model_path)
        self.model.eval()
        
     
        self.features = {}
        
        
        self.register_hooks()
    
    def _setup_engine(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('-l', '--local_rank', default=-1, type=int)
        parser.add_argument('-n', '--nodes', default=1, type=int)
        parser.add_argument("--ddp", action="store_true")
        
        args = parser.parse_args([])
        args.world_size = args.nodes * args.gpus
        
        engine = Engine(
            custom_arg=args, 
            logger=logger,
            continue_state_object=config.pretrained_weight_path
        )
        
        return engine
    
    def load_model(self, model_path=None):
       
        logger.info("Loading model...")
        model = Network(config.num_classes, wide=True)
        model = torch.nn.DataParallel(model, device_ids=self.engine.devices)
        model.to(self.device)
        
        
        
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info("Model loaded and parameters frozen")
        return model
    
    def hook_fn(self, layer_name):
        
        def hook(module, input, output):
            self.features[layer_name] = output.detach()
        return hook
    
    def register_hooks(self):
        
        logger.info("Registering hooks for feature extraction...")
        
        
        self.model.module.branch1.mod7.register_forward_hook(self.hook_fn('mod7'))
        

        self.model.module.branch1.aspp.register_forward_hook(self.hook_fn('aspp'))
        

        self.model.module.branch1.bot_aspp.register_forward_hook(self.hook_fn('bot_aspp'))
        self.model.module.branch1.bot_fine.register_forward_hook(self.hook_fn('bot_fine'))
        
      
        def hook_dec0(module, input, output):
            if 'bot_aspp' in self.features and 'bot_fine' in self.features:
                dec0_up = self.features['bot_aspp']
                dec0_fine = self.features['bot_fine']
                dec0_up_resized = F.interpolate(
                    dec0_up, 
                    size=dec0_fine.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
                dec0 = torch.cat([dec0_fine, dec0_up_resized], dim=1)
                self.features['dec0'] = dec0
        
      
        self.model.module.branch1.final[0].register_forward_hook(hook_dec0)
        logger.info("Hooks registered successfully")
    
    def get_dataloader(self):
     
        
        train_loader, _, _ = get_mix_loader(
            engine=self.engine,
            augment=True,
            cs_root=config.city_root_path,
            coco_root=config.coco_root_path
        )
        logger.info(f"Dataloader created with {len(train_loader)} batches")
        return train_loader
    
    def extract_features(self, images):
       
        with torch.no_grad():
         
            self.features.clear()
            
 
            _ = self.model(images)
            
            if self.target_layer in self.features:                          
                
                if 'dec0' in self.features:
                    return self.features['dec0']
                else:
                    logger.error("Features from dec0 layer could not be extracted")
                    logger.error(f"Available feature keys: {list(self.features.keys())}")
                    raise RuntimeError("Features from dec0 layer could not be extracted")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import glob

# 1. First, create a Cityscapes Dataset class
class CityscapesDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, split='train', transform=None):
        """
        Args:
            image_dir: Path to the Cityscapes images
            annotation_dir: Path to the Cityscapes annotations
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on the images
        """
        self.image_dir = os.path.join(image_dir, split)  # Adjust if your path structure is different
        self.annotation_dir = os.path.join(annotation_dir, split)  # Adjust if your path structure is different
        self.transform = transform
        
        # Get list of image files
        self.images = sorted(glob.glob(os.path.join(self.image_dir, '**', '*.png'), recursive=True))
        
        # Get corresponding annotation files
        self.annotations = []
        for img_path in self.images:
            # Construct the corresponding annotation path based on your directory structure
            # You may need to adjust this based on your file naming convention
            img_name = os.path.basename(img_path)
            city_name = img_path.split('/')[-2]  # Assuming the city name is the parent directory
            
            # For Cityscapes, annotations typically have _gtFine_labelIds.png suffix
            ann_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            ann_path = os.path.join(self.annotation_dir, city_name, ann_name)
            
            if os.path.exists(ann_path):
                self.annotations.append(ann_path)
            else:
                print(f"Warning: Annotation not found for {img_path}")
        
        # Filter images to only those with annotations
        if len(self.images) != len(self.annotations):
            self.images = [self.images[i] for i in range(len(self.images)) if i < len(self.annotations)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load annotation
        ann_path = self.annotations[idx]
        target = Image.open(ann_path)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert target to tensor
        target = torch.from_numpy(np.array(target)).long()
        
        # For this example, we don't have anomaly masks
        # You would need to create these based on your specific anomaly definition
        # For simplicity, we'll set is_anomaly to None
        is_anomaly = None
        
        return {
            'image': image,
            'target': target,
            'is_anomaly': is_anomaly,
            'path': img_path
        }

# 2. Set up data transforms
train_transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # Adjust resolution based on your GPU memory
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Create datasets
train_dataset = CityscapesDataset(
    image_dir='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/',
    annotation_dir='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine',
    split='train',
    transform=train_transform
)

val_dataset = CityscapesDataset(
    image_dir='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/',
    annotation_dir='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine',
    split='val',
    transform=val_transform
)

# 4. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# 5. Set up the model, criterion, optimizer as in the code you provided
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = HopfieldPEBALModel(
    num_classes=19,  # Cityscapes has 19 classes
    backbone_name='resnet101',
    pretrained=True,
    hopfield_dim=512,
    memory_size=1024
).to(device)

# Create loss function
criterion = HopfieldPEBALLoss(
    num_classes=19,
    energy_weight=0.1,
    hopfield_weight=0.1,
    anomaly_margin=10.0,
    known_margin=1.0
)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Create scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)

# 6. Train the model
trained_model = train_hopfield_pebal(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=30,  # Adjust based on your needs
    device=device,
    scheduler=scheduler
)

# 7. Save the trained model
torch.save(trained_model.state_dict(), "hopfield_pebal_cityscapes.pth")

# 8. Code for testing on the test set
def test_model():
    model.eval()
    
    test_dataset = CityscapesDataset(
        image_dir='/home/ha51dybi/PEBAL/cityscapes/images/city_gt_fine/',
        annotation_dir='/home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine',
        split='test',
        transform=val_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Initialize metrics
    results = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            images = batch['image'].to(device)
            paths = batch['path']
            
            # Run inference
            outputs = model(images)
            
            # Get segmentation and anomaly predictions
            logits = outputs['logits']
            combined_energy = outputs['combined_energy']
            
            # Segmentation prediction (excluding anomaly class)
            seg_probs = torch.softmax(logits[:, :-1], dim=1)
            seg_pred = torch.argmax(seg_probs, dim=1)
            
            # Anomaly prediction (using combined energy with a threshold)
            anomaly_threshold = 0.5  # This should be calibrated
            anomaly_pred = (combined_energy > anomaly_threshold).float()
            
            # Store results for each image
            for i in range(len(paths)):
                results.append({
                    'path': paths[i],
                    'segmentation': seg_pred[i].cpu().numpy(),
                    'anomaly_map': anomaly_pred[i].cpu().numpy()
                })
    
    return results

# Run test after training if needed
# test_results = test_model()
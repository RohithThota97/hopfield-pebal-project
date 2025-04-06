import torch
from torch.utils.data import DataLoader
from models.hopfield_pebal import (
    HopfieldPEBALModel,
    HopfieldPEBALLoss,
    train_hopfield_pebal,
    inference
)
from datasets.custom_dataset import DummyDataset
import torch.optim as optim

def main():
    # 1) Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # 2) Make a dummy dataset
    train_dataset = DummyDataset(length=20)
    val_dataset = DummyDataset(length=5)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # 3) Create model
    #    num_classes=19 is typical for Cityscapes, but let's just keep it as is
    model = HopfieldPEBALModel(
        num_classes=19,
        backbone_name='resnet101',
        pretrained=False,  # We can't do pretrained easily in this dummy scenario
        hopfield_dim=512,
        memory_size=1024
    ).to(device)
    
    # 4) Create the loss function
    criterion = HopfieldPEBALLoss(
        num_classes=19,
        energy_weight=0.1,
        hopfield_weight=0.1,
        anomaly_margin=10.0,
        known_margin=1.0
    )
    
    # 5) Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 6) (Optional) create a scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # 7) Train
    model = train_hopfield_pebal(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3,  # let's do 3 epochs for this example
        device=device,
        scheduler=scheduler
    )
    
    # 8) Save the model
    torch.save(model.state_dict(), "hopfield_pebal_final.pth")
    print("Training complete, model saved!")
    
    # 9) Do a quick inference
    # Just take 1 batch from the val_dataset
    sample = val_dataset[0]  # single item
    sample_image = sample['image'].unsqueeze(0).to(device)  # add batch dimension
    
    result = inference(model, sample_image[0], device=device)
    print("Inference result keys:", result.keys())

if __name__ == "__main__":
    main()
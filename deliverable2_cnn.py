import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import RobotControlDataset

class PositionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stream 1: Image Feature Extractor
        # Input: [Batch, 3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: [16, 64, 64]
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: [32, 32, 32]
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: [64, 16, 16]
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [64, 8, 8]
        )
        
        self.flatten = nn.Flatten()
        
        # 64 channels * 8 height * 8 width = 4096 features
        # Add the 4 action features -> 4100
        
        # Stream 2: Fusion and Prediction Head
        self.mlp = nn.Sequential(
            nn.Linear(4100, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output: 2D target position (x, y)
        )

    def forward(self, img, action):
        # Extract visual features
        visual_features = self.cnn(img)
        visual_features = self.flatten(visual_features)
        
        # Fuse with action
        fused_vector = torch.cat([visual_features, action], dim=1)
        
        # Predict position
        return self.mlp(fused_vector)

def train(model, train_loader, test_loader, criterion, optimizer, device, epochs=50):
    epoch_train_losses = []
    epoch_test_losses = []
    
    for epoch in tqdm(range(epochs)):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for imgs_before, actions, target_pos, _ in train_loader:
            imgs_before, actions, target_pos = imgs_before.to(device), actions.to(device), target_pos.to(device)
            
            optimizer.zero_grad()
            predictions = model(imgs_before, actions)
            loss = criterion(predictions, target_pos)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for imgs_before, actions, target_pos, _ in test_loader:
                imgs_before, actions, target_pos = imgs_before.to(device), actions.to(device), target_pos.to(device)
                predictions = model(imgs_before, actions)
                loss = criterion(predictions, target_pos)
                running_test_loss += loss.item()
                
        avg_test_loss = running_test_loss / len(test_loader)
        epoch_test_losses.append(avg_test_loss)
        
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
            
    return epoch_train_losses, epoch_test_losses

def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for imgs_before, actions, target_pos, _ in dataloader:
            imgs_before, actions, target_pos = imgs_before.to(device), actions.to(device), target_pos.to(device)
            predictions = model(imgs_before, actions)
            loss = criterion(predictions, target_pos)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(dataloader)
    print(f"\nFinal Test Error (MSE): {avg_test_loss:.6f}")
    return avg_test_loss

if __name__ == "__main__":
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load and Split Data
    full_dataset = RobotControlDataset(data_dir="data")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = PositionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train and Test
    print("\nStarting CNN Training...")
    train_losses, test_losses = train(model, train_loader, test_loader, criterion, optimizer, device, epochs=50)
    test(model, test_loader, criterion, device)

    # Save the model weights
    torch.save(model.state_dict(), "models/cnn_model.pth")
    print("Model saved to models/cnn_model.pth")

    # Plot and save the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Deliverable 2: CNN Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/cnn_loss_curve.png")
    print("Loss curves saved to assets/cnn_loss_curve.png")

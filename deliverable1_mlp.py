import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import RobotControlDataset

class PositionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 channels * 128 height * 128 width = 49152 + 4 action features = 49156
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            nn.Linear(49156, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Helps prevent overfitting on our small dataset
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output: 2D target position (x, y)
        )

    def forward(self, img, action):
        # Flatten the image: [Batch, 3, 128, 128] -> [Batch, 49152]
        img_flat = self.flatten(img)
        # Concatenate with action: [Batch, 4] -> [Batch, 49156]
        x = torch.cat([img_flat, action], dim=1)

        return self.network(x)

def train(model, dataloader, criterion, optimizer, device, epochs=50):
    model.train()
    epoch_losses = []
    
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for imgs_before, actions, target_pos, _ in dataloader:
            imgs_before = imgs_before.to(device)
            actions = actions.to(device)
            target_pos = target_pos.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(imgs_before, actions)
            loss = criterion(predictions, target_pos)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_epoch_loss:.6f}")
            
    return epoch_losses

def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for imgs_before, actions, target_pos, _ in dataloader:
            imgs_before = imgs_before.to(device)
            actions = actions.to(device)
            target_pos = target_pos.to(device)
            
            predictions = model(imgs_before, actions)
            loss = criterion(predictions, target_pos)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(dataloader)
    print(f"\nFinal Test Error (MSE): {avg_test_loss:.6f}")
    return avg_test_loss

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load and Split Data (80% Train, 20% Test)
    full_dataset = RobotControlDataset(data_dir="data")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Fix the generator seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = PositionMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the Model
    print("\nStarting Training...")
    train_losses = train(model, train_loader, criterion, optimizer, device, epochs=50)

    # Test the Model
    test(model, test_loader, criterion, device)

    # Save the model weights
    torch.save(model.state_dict(), "models/mlp_model.pth")
    print("Model saved to models/mlp_model.pth")

    # Plot and save the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Deliverable 1: MLP Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/mlp_loss_curve.png")
    print("Loss curve saved to assets/mlp_loss_curve.png")

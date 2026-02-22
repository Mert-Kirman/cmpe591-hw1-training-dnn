import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

from dataset import RobotControlDataset

class ActionConditionedEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder: Compress 128x128 image down to 8x8 feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # [Batch, 16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [Batch, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [Batch, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# [Batch, 128, 8, 8]
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        # 128 channels * 8 * 8 = 8192 + 4 action features = 8196
        
        # Fusion bottleneck
        self.fc_fusion = nn.Sequential(
            nn.Linear(8196, 8192),
            nn.ReLU()
        )
        
        # Decoder: Upsample 8x8 feature maps back to 128x128 image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # [64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # [3, 128, 128]
            nn.Sigmoid() # Force output pixels to be between 0.0 and 1.0
        )

    def forward(self, img, action):
        # Encode
        x = self.encoder(img)
        x_flat = self.flatten(x)
        
        # Fuse
        fused = torch.cat([x_flat, action], dim=1)
        fused = self.fc_fusion(fused)
        
        # Reshape back to spatial dimensions: [Batch, 128, 8, 8]
        x_reshaped = fused.view(-1, 128, 8, 8)
        
        # Decode
        return self.decoder(x_reshaped)

def train(model, train_loader, test_loader, criterion, optimizer, device, epochs=50):
    epoch_train_losses = []
    epoch_test_losses = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        running_train_loss = 0.0
        for imgs_before, actions, _, imgs_after in train_loader:
            imgs_before, actions, imgs_after = imgs_before.to(device), actions.to(device), imgs_after.to(device)
            
            optimizer.zero_grad()
            reconstructed_imgs = model(imgs_before, actions)
            loss = criterion(reconstructed_imgs, imgs_after)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for imgs_before, actions, _, imgs_after in test_loader:
                imgs_before, actions, imgs_after = imgs_before.to(device), actions.to(device), imgs_after.to(device)
                reconstructed_imgs = model(imgs_before, actions)
                loss = criterion(reconstructed_imgs, imgs_after)
                running_test_loss += loss.item()
                
        avg_test_loss = running_test_loss / len(test_loader)
        epoch_test_losses.append(avg_test_loss)
        
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
            
    return epoch_train_losses, epoch_test_losses

def test_and_save_images(model, dataloader, device):
    model.eval()
    print("\nGenerating reconstruction comparisons...")
    
    with torch.no_grad():
        # Grab just one batch
        imgs_before, actions, _, imgs_after = next(iter(dataloader))
        imgs_before, actions, imgs_after = imgs_before.to(device), actions.to(device), imgs_after.to(device)
        
        # Generate predictions
        reconstructed_imgs = model(imgs_before, actions)
        
        # Select the first 8 images from the batch to visualize
        n_images = 8
        real_imgs = imgs_after[:n_images].cpu()
        fake_imgs = reconstructed_imgs[:n_images].cpu()
        
        # Create a grid: Top row = Ground Truth, Bottom row = Predictions
        comparison_grid = torch.cat([real_imgs, fake_imgs], dim=0)
        
        # Save the grid
        vutils.save_image(comparison_grid, "assets/reconstruction_comparison.png", nrow=n_images, padding=2, normalize=False)
        print("Saved comparison grid to assets/reconstruction_comparison.png")

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

    full_dataset = RobotControlDataset(data_dir="data")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ActionConditionedEncoderDecoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("\nStarting Encoder-Decoder Training...")
    train_losses, test_losses = train(model, train_loader, test_loader, criterion, optimizer, device, epochs=50)
    
    # Run the custom test function to save images
    test_and_save_images(model, test_loader, device)

    # Save the model weights
    torch.save(model.state_dict(), "models/encoder_decoder_model.pth")
    print("Model saved to models/encoder_decoder_model.pth")

    # Plot and save the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Deliverable 3: Encoder-Decoder Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/encoder_decoder_loss_curve.png")
    print("Loss curves saved to assets/encoder_decoder_loss_curve.png")

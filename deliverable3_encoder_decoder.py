import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import RobotControlDataset


class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),   # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU() # Stay at 32x32
        )
        
        # Action Embedding
        self.action_embed = nn.Sequential(
            nn.Linear(4, 64), 
            nn.ReLU(), 
            nn.Linear(64, 128)
        )
        
        # Fusion Bottleneck
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32 -> 64
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64 -> 128
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img, action):
        # Encode to 32x32 (higher resolution bottleneck keeps more spatial detail)
        x = self.enc(img)
        
        # Fuse action information
        act_emb = self.action_embed(action)
        act_map = act_emb.view(-1, 128, 1, 1).expand(-1, -1, 32, 32)
        fused = self.fusion(torch.cat([x, act_map], dim=1))
        
        # Decode to full resolution
        return self.dec(fused)


# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, predictions, targets):
        # MSE for overall pixel accuracy
        pixel_loss = self.mse(predictions, targets)
        
        # Extra focus on red channel (object) without hard thresholds
        red_channel_loss = self.l1(predictions[:, 0], targets[:, 0]) * 2.0
        
        return pixel_loss + red_channel_loss


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=50):
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
        
        # Update learning rate based on test loss
        scheduler.step(avg_test_loss)
        
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
        
        n_images = min(8, imgs_before.size(0))
        
        # Move tensors to CPU and extract the first n_images
        imgs_b = imgs_before[:n_images].cpu()
        imgs_a = imgs_after[:n_images].cpu()
        imgs_pred = reconstructed_imgs[:n_images].cpu()
        
        # Revert the one-hot encoding back to standard integer action IDs
        action_ids = torch.argmax(actions[:n_images], dim=1).cpu()
        
        # Set up a Matplotlib grid (n_images rows, 3 columns)
        fig, axes = plt.subplots(nrows=n_images, ncols=3, figsize=(9, 3 * n_images))
        
        for i in range(n_images):
            # PyTorch images are [C, H, W], Matplotlib expects [H, W, C]
            img_b_np = imgs_b[i].permute(1, 2, 0).numpy()
            img_a_np = imgs_a[i].permute(1, 2, 0).numpy()
            img_pred_np = imgs_pred[i].permute(1, 2, 0).numpy()
            
            # Column 1: Before Image
            axes[i, 0].imshow(img_b_np)
            axes[i, 0].set_title(f"Before (Action: {action_ids[i].item()})")
            axes[i, 0].axis('off')
            
            # Column 2: True After Image
            axes[i, 1].imshow(img_a_np)
            axes[i, 1].set_title("True After")
            axes[i, 1].axis('off')
            
            # Column 3: Predicted After Image
            axes[i, 2].imshow(img_pred_np)
            axes[i, 2].set_title("Predicted After")
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        plt.savefig("assets/reconstruction_comparison.png", bbox_inches='tight')
        plt.close()
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

    model = EncoderDecoderModel().to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Add learning rate scheduler to prevent overfitting
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print("\nStarting Encoder-Decoder Training...")
    train_losses, test_losses = train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=120)
    
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
    plt.ylabel('Combined Loss (Overall + Red Channel)')
    plt.title('Deliverable 3: Encoder-Decoder Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/encoder_decoder_loss_curve.png")
    print("Loss curves saved to assets/encoder_decoder_loss_curve.png")

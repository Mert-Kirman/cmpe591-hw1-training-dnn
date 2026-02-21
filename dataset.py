import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RobotControlDataset(Dataset):
    def __init__(self, data_dir="data", num_parts=4):
        """
        Loads and preprocesses the collected robotic manipulation data.
        """
        super().__init__()
        
        pos_list = []
        act_list = []
        img_b_list = []
        img_a_list = []
        
        # Load all parts from the data directory
        for i in range(num_parts):
            pos_path = os.path.join(data_dir, f"positions_{i}.pt")
            act_path = os.path.join(data_dir, f"actions_{i}.pt")
            img_b_path = os.path.join(data_dir, f"imgs_before_{i}.pt")
            img_a_path = os.path.join(data_dir, f"imgs_after_{i}.pt")
            
            pos_list.append(torch.load(pos_path, weights_only=True))
            act_list.append(torch.load(act_path, weights_only=True))
            img_b_list.append(torch.load(img_b_path, weights_only=True))
            img_a_list.append(torch.load(img_a_path, weights_only=True))
            
        # Concatenate the lists into single tensors
        self.positions = torch.cat(pos_list, dim=0).float()
        raw_actions = torch.cat(act_list, dim=0).to(torch.int64)
        
        # One-hot encode actions (4 possible directions) and convert to float
        self.actions = F.one_hot(raw_actions, num_classes=4).float()
        
        # Concatenate and normalize images to 0.0 - 1.0 range
        self.imgs_before = torch.cat(img_b_list, dim=0).float() / 255.0
        self.imgs_after = torch.cat(img_a_list, dim=0).float() / 255.0
        
        print(f"Successfully loaded {len(self.positions)} samples from '{data_dir}'.")
        print(f"Image shape: {self.imgs_before.shape[1:]} | Action shape: {self.actions.shape[1:]} | Pos shape: {self.positions.shape[1:]}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Returns a tuple of:
        (input_image, input_action, target_position, target_image)
        """
        return (
            self.imgs_before[idx],
            self.actions[idx],
            self.positions[idx],
            self.imgs_after[idx]
        )

# Test block to ensure everything works
if __name__ == "__main__":
    # Create the dataset
    dataset = RobotControlDataset(data_dir="data")
    
    # Wrap it in a DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Fetch one batch to verify shapes
    img_b, act, pos, img_a = next(iter(loader))
    print("\nBatch shapes verification:")
    print(f"Input Images: {img_b.shape}")  # Expected: [32, 3, 128, 128]
    print(f"Input Actions: {act.shape}")   # Expected: [32, 4]
    print(f"Target Positions: {pos.shape}") # Expected: [32, 2]
    print(f"Target Images: {img_a.shape}")  # Expected: [32, 3, 128, 128]

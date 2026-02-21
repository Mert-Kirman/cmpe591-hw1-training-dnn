from multiprocessing import Process
import numpy as np
import torch
from homework1 import Hw1Env
import os

def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")
    
    # Pre-allocate tensors for all required data
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    
    for i in range(N):
        # Capture the initial state before the action
        _, pixels_before = env.state()
        
        # Sample and take the action
        action_id = np.random.randint(4)
        env.step(action_id)
        
        # Capture the resulting state after the action
        obj_pos, pixels_after = env.state()
        
        # Store everything
        positions[i] = torch.tensor(obj_pos)
        actions[i] = action_id
        imgs_before[i] = pixels_before
        imgs_after[i] = pixels_after
        
        # Reset for the next iteration
        env.reset()
        
    # Save the tensors
    torch.save(positions, f"data/positions_{idx}.pt")
    torch.save(actions, f"data/actions_{idx}.pt")
    torch.save(imgs_before, f"data/imgs_before_{idx}.pt")
    torch.save(imgs_after, f"data/imgs_after_{idx}.pt")
    print(f"Process {idx} finished collecting {N} samples.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    processes = []
    for i in range(4):
        p = Process(target=collect, args=(i, 250))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("All 1000 data points successfully collected and saved in data folder.")
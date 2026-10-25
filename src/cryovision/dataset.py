import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class CryovisionDataset(Dataset):
    def __init__(self, patches_dir, manifest_path, split="train"):
        self.patches_dir = patches_dir
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.patch_ids = [k for k, v in self.manifest.items() if v["split"] == split]
        self.is_train = (split == "train")

    def __len__(self):
        return len(self.patch_ids)
        
    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        data = np.load(os.path.join(self.patches_dir, f"{pid}.npz"))
        x = data['x'] # (4, H, W)
        y = data['y'] # (H, W)
        
        if self.is_train:
            # Random flip H
            if random.random() > 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=1)
            # Random flip V
            if random.random() > 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=0)
            # Random Rot90
            k = random.randint(0, 3)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k, axes=(0, 1))
            
            # Brightness jitter uniformly across optical channels (0,1,2)
            # DO NOT jitter NDWI independently!
            if random.random() > 0.5:
                jitter = random.uniform(0.8, 1.2)
                # Clip to prevent overflow if they were 0-255 scaled
                x[:3] = np.clip(x[:3] * jitter, 0, 255)
                # Recompute NDWI to maintain physical relationship
                nir = x[0]
                green = x[2]
                denom = green + nir
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndwi = np.where(denom == 0, 0, (green - nir) / denom)
                x[3] = ndwi

        x = torch.from_numpy(x.copy()).float()
        y = torch.from_numpy(y.copy()).float().unsqueeze(0) # (1, H, W)
        return x, y

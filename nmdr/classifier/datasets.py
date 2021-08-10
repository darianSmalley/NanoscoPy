import os
import glob
from pathlib import Path
import numpy as np 
import torch
from torch.utils.data import Dataset, Subset

class STSDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        # get all filenames in root with filetype .npy
        self.data_root =  os.path.join(data_path, "**/*.npy")
        self.data_paths = glob.glob(self.data_root,
                                    recursive = True)
        self.data_paths = np.array(self.data_paths)
        
        # get classes from directory structure
        self.targets = os.listdir(data_path)
        self.num_classes = len(self.targets)
        
        # get number of samples
        self.n = len(self.data_paths)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get filepath using indexes 
        sts_path = self.data_paths[idx]
        
        # Get sts sample class from path
        label = Path(sts_path).parts[-2]
        
        # Convert label to index
        label_i = self.targets.index(label)
        
        # Use index to set active bit of onehot vector
        label_onehot = np.zeros(self.num_classes)
        label_onehot[label_i] = 1
        
        # Read STS data from disk
        sts = np.load(sts_path)
        
        # data has dI/dV and V with shape (2, 1200)
        # only use dI/dV
        sts = sts[0].copy().reshape( 1, -1)
        
        if self.transform:
            sts = self.transform(sts)
        if self.target_transform:
            label_onehot = self.target_transform(label_onehot)
            
        return torch.tensor(sts, dtype=float),  torch.tensor(label_i, dtype=torch.long)
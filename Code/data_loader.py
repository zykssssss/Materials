import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
import operator

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        #image = Image.open(image_path).convert("RGB")
        image = np.array(io.imread(image_path))
        chs = []
        for i in range(0,4):
            idrr = image[:,:,i]
            idrr[idrr < 0] = 0.0
            idrr[idrr == 65536] = 0.0
            idrr[idrr > 1000] = 1000.0
            idrr = 1.0 * (idrr / 1000.0)
            chs.append(idrr)
        image = np.stack(chs)
        if self.transform:
            image = self.transform(image)
        
        return image_path, image, label

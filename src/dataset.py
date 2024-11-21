import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class Geochem_Dataset_Train(Dataset):
    """
    Dataset for training
    """
    def __init__(self, datadir: str, labeldir: str, transform=None):
        """
        Args:
            datadir (str): data path
            labeldir (str): label path
        """
        assert os.path.exists(datadir), f"{datadir} not exists!"
        assert os.path.exists(labeldir), f"{labeldir} not exists!"
        
        img_path = sorted(glob.glob(os.path.join(datadir, "*.npy")))
        lab_path = sorted(glob.glob(os.path.join(labeldir, "*.npy")))
        
        if len(img_path) != len(lab_path):
            raise ValueError("The number of data and label is not equal!")
        
        self.img = []
        self.label = []
        
        for i, j in zip(img_path, lab_path):
            self.img.append(np.load(i))
            self.label.append(np.load(j))
        
        self.transform = transform

    def __getitem__(self, index):
        image = self.img[index]
        label = self.label[index]
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # 将numpy数组转换为torch张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

    def __len__(self):
        return len(self.img)


class GeochemDatasetTest(Dataset):
    def __init__(self, datalist: list):
        self.img = datalist

    def __getitem__(self, index):
        return self.img[index]

    def __len__(self):
        return len(self.img)

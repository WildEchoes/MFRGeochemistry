import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

import lmdb


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
        # image = np.load(self.img_path[index])
        # label = np.load(self.lab_path[index])
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


class Geochem_Dataset_Trian_lmdb(Dataset):
    def __init__(self, data_lmdb_dir:str, label_lmdb_dir:str, transform=None):
        self.env_data = lmdb.open(data_lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
        self.env_label = lmdb.open(label_lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env_data.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        with self.env_label.begin(write=False) as txn:
            assert txn.stat()['entries'] == self.length, "数据和标签的数量不一致"
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env_data.begin(write=False) as txn:
            byteflow = txn.get(f"{index+1}".encode())  # 确保索引从1开始
        buffer = np.frombuffer(byteflow, dtype=np.float32)
        data = np.reshape(buffer, (12, 50, 50))  # 需要根据实际情况调整shape
        image = torch.from_numpy(data.copy()).float()

        # 此处假设标签也以同样方式存储，如果不是，需要额外处理
        with self.env_label.begin(write=False) as txn:
            byteflow = txn.get(f"{index+1}".encode())
        buffer = np.frombuffer(byteflow, dtype=np.float32)
        data1 = np.reshape(buffer, (1, 50, 50))
        label = torch.from_numpy(data1.copy()).float()

        # 应用任何传递的变换
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label

if __name__ == "__main__":
    img_path = "D:\\MyProject\\ETS_data\\trian\\imglmdb\\50"
    lab_path = "D:\\MyProject\\ETS_data\\trian\\geolmdb\\50\\Al2O3"
    # dataset = Geochem_Dataset_Train(img_path, lab_path)
    # print(len(dataset))
    # DataLoader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # for j, (data, label) in enumerate(DataLoader):
    #     data = data.cuda()
    #     label = label.cuda()
    #     pass
    
    dataset = Geochem_Dataset_Trian_lmdb(img_path, lab_path)
    DataLoader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    for j, (data, label) in enumerate(DataLoader):
        data = data.cuda()
        label = label.cuda()
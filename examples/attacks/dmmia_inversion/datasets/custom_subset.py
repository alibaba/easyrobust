import torch
import numpy as np


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = np.array(dataset.targets)[self.indices]

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, targets

    def __len__(self):
        return len(self.indices)


class SingleClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_class):
        self.dataset = dataset
        self.indices = np.where(np.array(dataset.targets) == target_class)[0]
        self.targets = np.array(dataset.targets)[self.indices]
        self.target_class = target_class

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)


class ClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_classes):
        self.dataset = dataset
        self.indices = np.where(
            np.isin(np.array(dataset.targets), np.array(target_classes)))[0]
        self.targets = np.array(dataset.targets)[self.indices]
        self.target_classes = target_classes

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)

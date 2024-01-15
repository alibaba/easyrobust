import os

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder


class FaceScrub(Dataset):
    def __init__(self,
                 group,
                 train,
                 split_seed=42,
                 transform=None,
                 cropped=True,
                 root='data/facescrub'):

        if group == 'actors':
            if cropped:
                root = os.path.join(root, 'actors/faces')
            else:
                root = os.path.join(root, 'actors/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actors'

        elif group == 'actresses':
            if cropped:
                root = os.path.join(root, 'actresses/faces')
            else:
                root = os.path.join(root, 'actresses/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actresses'

        elif group == 'all':
            if cropped:
                root_actors = os.path.join(root, 'actors/faces')
                root_actresses = os.path.join(root, 'actresses/faces')
            else:
                root_actors = os.path.join(root, 'actors/images')
                root_actresses = os.path.join(root, 'actresses/images')
            dataset_actors = ImageFolder(root=root_actors, transform=None)
            target_transform_actresses = lambda x: x + len(dataset_actors.
                                                           classes)
            dataset_actresses = ImageFolder(
                root=root_actresses,
                transform=None,
                target_transform=target_transform_actresses)
            dataset_actresses.class_to_idx = {
                key: value + len(dataset_actors.classes)
                for key, value in dataset_actresses.class_to_idx.items()
            }
            self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
            self.classes = dataset_actors.classes + dataset_actresses.classes
            self.class_to_idx = {
                **dataset_actors.class_to_idx,
                **dataset_actresses.class_to_idx
            }
            self.targets = dataset_actors.targets + [
                t + len(dataset_actors.classes)
                for t in dataset_actresses.targets
            ]
            self.name = 'facescrub_all'

        else:
            raise ValueError(
                f'Dataset group {group} not found. Valid arguments are \'all\', \'actors\' and \'actresses\'.'
            )

        self.transform = transform
        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        if train:
            self.dataset = Subset(self.dataset, train_idx)
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = Subset(self.dataset, test_idx)
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]

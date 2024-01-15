from torchvision.datasets import CelebA
from torch.utils.data import Dataset, Subset
from collections import Counter
import torchvision.transforms as T
import numpy as np
from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

class CelebA1000(Dataset):
    def __init__(self,
                 train,
                 split_seed=42,
                 transform=None,
                 root='data/celeba',
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity")
        celeba.targets = celeba.identity

        # Select the 1,000 most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(),
                   key=lambda item: item[1],
                   reverse=True))
        sorted_targets = list(ordered_dict.keys())[:1000]

        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }

        self.target_transform = T.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if train:
            self.dataset = Subset(celeba, train_idx)
            train_targets = np.array(targets)[train_idx]
            self.targets = [self.target_transform(t) for t in train_targets]
            self.name = 'CelebA1000_train'
        else:
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebA1000_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CustomCelebA(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root: str,
            split: str = "all",
            target_type: Union[List[str], str] = "identity",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(CustomCelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, "img_align_celeba", self.filename[index])
        file_path = file_path.replace('.jpg', '.png')
        file_path = './' + file_path
        X = PIL.Image.open(file_path)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import os
import numpy as np
import scipy.io
from PIL import Image

# Code Snippets by https://github.com/zrsmithson/Stanford-dogs/blob/master/data/stanford_dogs_data.py

class StanfordDogs(Dataset):
    def __init__(self,
                 train,
                 cropped,
                 split_seed=42,
                 transform=None,
                 root='data/stanford_dogs'):

        self.image_path = os.path.join(root, 'Images')
        dataset = ImageFolder(root=self.image_path, transform=None)
        self.dataset = dataset
        self.cropped = cropped
        self.root = root

        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self.breeds = os.listdir(self.image_path)

        self.classes = [cls.split('-', 1)[-1] for cls in self.dataset.classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        self.targets = self.dataset.targets
        self.name = 'stanford_dogs'

        split_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
        labels_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        split_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
        labels_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split_train] + [item[0][0] for item in split_test]
        labels = [item[0]-1 for item in labels_train] + [item[0]-1 for item in labels_test]

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                                        for annotation, idx in zip(split, labels)]
            self._flat_breed_annotations = [t[0] for t in self._breed_annotations]
            self.targets = [t[-1][-1] for t in self._breed_annotations]
            self._flat_breed_images = [(annotation+'.jpg', box, idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in zip(split, labels)]
            self.targets = [t[-1] for t in self._breed_images]
            self._flat_breed_images = self._breed_images

        self.transform = transform
        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        if train:
            self.dataset = np.array(self._flat_breed_images, dtype=object)[train_idx].tolist()
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = np.array(self._flat_breed_images, dtype=object)[test_idx].tolist()
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # im, _ = self.dataset[idx]
        image_name, target = self.dataset[idx][0], self.dataset[idx][-1]
        image_path = os.path.join(self.image_path, image_name)
        im = Image.open(image_path).convert('RGB')

        if self.cropped:
            im = im.crop(self.dataset[idx][1])
        if self.transform:
            return self.transform(im), target
        else:
            return im, target

    def get_boxes(self, path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

from types import SimpleNamespace
from typing import List

import numpy as np
import torch.optim as optim
import torchvision.transforms as T
import yaml
from models.classifier import Classifier
from rtpt.rtpt import RTPT
from torchvision.datasets import *

from datasets.celeba import CelebA1000
from datasets.custom_subset import Subset
from datasets.facescrub import FaceScrub
from datasets.stanford_dogs import StanfordDogs
from utils.datasets import get_normalization


class TrainingConfigParser:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def create_model(self):
        model_config = self._config['model']
        print(model_config)
        model = Classifier(**model_config)
        return model

    def create_datasets(self):
        dataset_config = self._config['dataset']
        name = dataset_config['type'].lower()
        train_set, valid_set, test_set = None, None, None

        data_transformation_train = self.create_transformations(
            mode='training', normalize=True)
        data_transformation_test = self.create_transformations(mode='test',
                                                               normalize=True)
        # Load datasets
        if name == 'facescrub':
            if 'facescrub_group' in dataset_config:
                group = dataset_config['facescrub_group']
            else:
                group = 'all'
            train_set = FaceScrub(group=group, train=True, cropped=True)
            test_set = FaceScrub(group=group,
                                 train=False,
                                 cropped=True,
                                 transform=data_transformation_test)
        elif name == 'celeba_identities':
            train_set = CelebA1000(train=True)
            test_set = CelebA1000(train=False,
                                  transform=data_transformation_test)
        elif name == 'stanford_dogs_uncropped':
            train_set = StanfordDogs(train=True, cropped=False)
            test_set = StanfordDogs(train=False, cropped=False,
                                        transform=data_transformation_test)
        elif name == 'stanford_dogs_cropped':
            train_set = StanfordDogs(train=True, cropped=True)
            test_set = StanfordDogs(train=False, cropped=True,
                                        transform=data_transformation_test)

        else:
            raise Exception(
                f'{name} is no valid dataset. Please use one of [\'facescrub\', \'celeba_identities\', \'stanford_dogs_uncropped\', \'stanford_dogs_cropped\'].'
            )

        if 'training_set_size' in dataset_config:
            train_set_size = dataset_config['training_set_size']
        else:
            train_set_size = len(train_set)

        # Set train and valid split sizes if specified
        validation_set_size = 0
        if 'validation_set_size' in dataset_config:
            validation_set_size = dataset_config['validation_set_size']
        elif 'validation_split_ratio' in dataset_config:
            validation_set_size = int(
                dataset_config['validation_split_ratio'] * train_set_size)
        if validation_set_size + train_set_size > len(train_set):
            print(
                f'Specified training and validation sets are larger than full dataset. \n\tTaking validation samples from training set.'
            )
            train_set_size = len(train_set) - validation_set_size
        # Split datasets into train and test split and set transformations
        indices = list(range(len(train_set)))
        np.random.seed(self._config['seed'])
        np.random.shuffle(indices)
        train_idx = indices[:train_set_size]
        if validation_set_size > 0:
            valid_idx = indices[train_set_size:train_set_size +
                                validation_set_size]
            valid_set = Subset(train_set, valid_idx, data_transformation_test)

            # Assert that there are no overlapping datasets
            assert len(set.intersection(set(train_idx), set(valid_idx))) == 0

        train_set = Subset(train_set, train_idx, data_transformation_train)

        # Compute dataset lengths
        train_len, valid_len, test_len = len(train_set), 0, 0
        if valid_set:
            valid_len = len(valid_set)
        if test_set:
            test_len = len(test_set)

        print(
            f'Created {name} datasets with {train_len:,} training, {valid_len:,} validation and {test_len:,} test samples.\n',
            f'Transformations during training: {train_set.transform}\n',
            f'Transformations during evaluation: {test_set.transform}')
        return train_set, valid_set, test_set

    def create_transformations(self, mode, normalize=True):
        """
        mode: 'training' or 'test'
        """
        dataset_config = self._config['dataset']
        dataset_name = self._config['dataset']['type'].lower()
        image_size = dataset_config['image_size']

        transformation_list = []
        # resize images to the expected size
        transformation_list.append(T.Resize(image_size))
        transformation_list.append(T.ToTensor())
        if mode == 'training' and 'transformations' in self._config:
            transformations = self._config['transformations']
            if transformations != None:
                for transform, args in transformations.items():
                    if not hasattr(T, transform):
                        raise Exception(
                            f'{transform} is no valid transformation. Please write the type exactly as the Torchvision class'
                        )
                    else:
                        transformation_class = getattr(T, transform)
                        transformation_list.append(transformation_class(**args))

        elif mode == 'test' and 'celeba' in dataset_name:
            if isinstance(image_size, list):
                transformation_list.append(T.CenterCrop(image_size))
            else:
                transformation_list.append(T.CenterCrop((image_size, image_size)))
        elif mode == 'test':
            pass
        else:
            raise Exception(f'{mode} is no valid mode for augmentation')

        if normalize:
            transformation_list.append(get_normalization())
        data_transformation = T.Compose(transformation_list)

        return data_transformation

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def create_rtpt(self):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=self.training['num_epochs'])
        return rtpt

    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def model(self):
        return self._config['model']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def wandb(self):
        return self._config['wandb']

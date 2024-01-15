import torch
import functools

from jax.experimental import optimizers
import jax
import jax.config
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
from jax import numpy as jnp
from jax import scipy as sp
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline


ARCHITECTURE = 'FC' #@param ['FC', 'Conv', 'Myrtle']; choice of neural network architecture yielding the corresponding NTK
DEPTH =  1#@param {'type': int}; depth of neural network
WIDTH = 1024 #@param {'type': int}; width of finite width neural network; only used if parameterization is 'standard'
PARAMETERIZATION = 'ntk' #@param ['ntk', 'standard']; whether to use standard or NTK parameterization, see https://arxiv.org/abs/2001.07301

# dataset
DATASET = 'cifar10' #@param ['cifar10', 'mnist']

# training params
LEARNING_RATE = 4e-2 #@param {'type': float};
SUPPORT_SIZE = 100  #@param {'type': int}; number of images to learn
TARGET_BATCH_SIZE = 5000  #@param {'type': int}; number of target images to use in KRR for each step
LEARN_LABELS = False #@param {'type': bool}; whether to optimize over support labels during training


def FullyConnectedNetwork( 
    depth,
    width,
    W_std = np.sqrt(2), 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk',
    activation = 'relu'):
    """Returns neural_tangents.stax fully connected network."""
    activation_fn = stax.Relu()
    dense = functools.partial(
      stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    for _ in range(depth):
        layers += [dense(width), activation_fn]
        layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, 
                        parameterization=parameterization)]

    return stax.serial(*layers)


def get_kernel_fn(architecture, depth, width, parameterization):
    return FullyConnectedNetwork(depth=depth, width=width, parameterization=parameterization)

_, _, kernel_fn = get_kernel_fn(ARCHITECTURE, DEPTH, WIDTH, PARAMETERIZATION)
KERNEL_FN = functools.partial(kernel_fn, get='ntk')

def class_balanced_sample(sample_size: int, 
                          labels: np.ndarray,
                          *arrays: np.ndarray, **kwargs: int):
 
    if labels.ndim != 1:
        raise ValueError(f'Labels should be one-dimensional, got shape {labels.shape}')
    n = len(labels)
    if not all([n == len(arr) for arr in arrays[1:]]):
        raise ValueError(f'All arrays to be subsampled should have the same length. Got lengths {[len(arr) for arr in arrays]}')
    classes = np.unique(labels)
    n_classes = len(classes)
    n_per_class, remainder = divmod(sample_size, n_classes)
    if remainder != 0:
        raise ValueError(
        f'Number of classes {n_classes} in labels must divide sample size {sample_size}.'
        )
    if kwargs.get('seed') is not None:
        np.random.seed(kwargs['seed'])
    inds = np.concatenate([
        np.random.choice(np.where(labels == c)[0], n_per_class, replace=False)
        for c in classes
    ])
    return (inds, labels[inds].copy()) + tuple([arr[inds].copy() for arr in arrays])






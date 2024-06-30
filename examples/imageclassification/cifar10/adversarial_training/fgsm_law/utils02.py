'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
from TinyImageNet import TinyImageNet
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.utils.data as data
from TinyImageNet import TinyImageNet

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)



def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def ImageNet_get_loaders(dir_, batch_size):
    num_workers = {'train' : 0,'val'   : 0,'test'  : 0}
    data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
    ])}
    num_workers = 0
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x), data_transforms[x]) 
                  for x in ['train', 'val','test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=num_workers)
                  for x in ['train', 'val', 'test']}
    return dataloaders

def ImageNet_get_loaders_32(dir_, batch_size):
    num_workers = {'train' : 0,'val'   : 0,'test'  : 0}
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(32),  
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(32),  
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(32),  
        transforms.ToTensor(),
    ])}
    num_workers = 0
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x), data_transforms[x]) 
                  for x in ['train', 'val','test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=num_workers)
                  for x in ['train', 'val', 'test']}
    return dataloaders
    
    
def New_ImageNet_get_loaders_32(dir_, batch_size):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    trainset = TinyImageNet(dir_, 'train', transform=transform_train, in_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    


    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
                  
    return trainloader, testloader

def New_ImageNet_get_loaders_32_testloader(dir_, batch_size):
    

    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
   
    


    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
                  
    return  testloader

def New_ImageNet_get_loaders_64(dir_, batch_size):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    trainset = TinyImageNet(dir_, 'train', transform=transform_train, in_memory=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    


    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
                  
    return trainloader, testloader



def New_ImageNet_get_loaders_64_testloader(dir_, batch_size):
   

    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
   


    testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
                  
    return testloader


  
def get_loaders_cifar100(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR100(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader







def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta
def evaluate_pgd(test_loader, model, attack_iters, restarts,epsilon= (8 / 255.) / std):
    print(epsilon)
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n



def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    # y_true=np.eye(10)[y.cuda().data.cpu().numpy()]
    # y_true=torch.from_numpy(y_true).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta



def evaluate_pgd_cw(test_loader, model, attack_iters, restarts):
    alpha = (2 / 255.) / std
    epsilon = (8 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta  = cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters=attack_iters, restarts=restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n








import numpy as np
from torch.autograd import Variable
def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out
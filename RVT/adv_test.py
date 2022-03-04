#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from json import load
import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from advertorch.attacks import LinfPGDAttack
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
import robust_models

from timm.models import create_model

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: dpn92)')
parser.add_argument('--ckpt_path', default='', type=str,
                    help='model architecture (default: dpn92)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-gpu', type=int, default=8,
                    help='Number of GPUS to use')

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
    
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def main():
    args = parser.parse_args()

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000
    )

    state_dict = torch.load(args.ckpt_path)

    model.load_state_dict(state_dict)
    normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model)
    

    dataset = datasets.ImageFolder('/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/val', transform=transforms.Compose([
        transforms.Resize(249, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    model.eval()

    end = time.time()
    # features_list = []

    total_num = 0
    correct_num = 0
    adv_num = 0

    adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=4/255,
    nb_iter=3, eps_iter=8/255/3, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

    for input, targets in tqdm(loader):
        input = input.cuda()
        labels = model(input)
        targets = targets.cuda()
        total_num += input.shape[0]
        correct = labels.max(1)[1].eq(targets).sum().item()
        correct_num += correct

        adv_untargeted = adversary.perturb(input, targets)
        labels_adv = model(adv_untargeted.detach())
        adv_num += labels_adv.max(1)[1].eq(targets).sum().item()

        print('acc: {} adv_acc:{} '.format(correct_num/total_num, adv_num/total_num))



if __name__ == '__main__':
    main()

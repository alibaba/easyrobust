#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils import model_zoo
import torch.nn as nn
import torchvision.transforms as transforms

from timm.models import create_model
from easyrobust.benchmarks import *
from easyrobust.models import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--data_dir', default='benchmarks/data', 
                    help='benchmark datasets')
parser.add_argument('--ckpt_path', default='', type=str, required=True,
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--mean', type=float, nargs='+', default=[0.485, 0.456, 0.406], metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=[0.229, 0.224, 0.225], metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default=3, type=int,
                    help='1: lanczos 2: bilinear 3: bicubic')
parser.add_argument('--input-size', default=224, type=int, 
                    help='images input size')
parser.add_argument('--crop-pct', default=0.875, type=float,
                metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--use_ema', action='store_true', default=False,
                    help='use use_ema model state_dict')
parser.add_argument('--adv', action='store_true',
                    default=False, help='')
parser.add_argument('--num-gpu', type=int, default=1,
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

    #############################################################
    #         Load Model
    #############################################################
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes
    )
    normalize = NormalizeByChannelMeanStd(
            mean=args.mean, std=args.std)

    if args.ckpt_path.startswith('http'):
        ckpt = model_zoo.load_url(args.ckpt_path)
    else:
        ckpt = torch.load(args.ckpt_path)

    if args.use_ema:
        assert 'state_dict_ema' in ckpt.keys() and ckpt['state_dict_ema'] is not None, 'no ema state_dict found!'
        state_dict = ckpt['state_dict_ema']
    else:
        if 'state_dict' in ckpt.keys():
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

    if '0.mean' in state_dict.keys() and '0.std' in state_dict.keys():
        model = nn.Sequential(normalize, model)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt)
        model = nn.Sequential(normalize, model)
    
    model.eval()
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()
        
    #############################################################
    #         Define Data Transform
    #############################################################
    test_transform = transforms.Compose([
        transforms.Resize(int(args.input_size/args.crop_pct), interpolation=args.interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor()
    ])

    #############################################################
    #         Evaluate on Benchmarks
    #############################################################
    
    if args.adv:
        # Adversarial Robust Models Benchmark
        evaluate_imagenet_autoattack(model, os.path.join(args.data_dir, 'imagenet-val'), test_batchsize=args.batch_size, test_transform=test_transform)
    else:
        # Non-Adversarial Robust Models Benchmark
        evaluate_imagenet_val(model, os.path.join(args.data_dir, 'imagenet-val'), test_batchsize=args.batch_size, test_transform=test_transform)
        evaluate_imagenet_a(model, os.path.join(args.data_dir, 'imagenet-a'), test_batchsize=args.batch_size, test_transform=test_transform)
        evaluate_imagenet_r(model, os.path.join(args.data_dir, 'imagenet-r'), test_batchsize=args.batch_size, test_transform=test_transform)
        evaluate_imagenet_sketch(model, os.path.join(args.data_dir, 'imagenet-sketch'), test_batchsize=args.batch_size, test_transform=test_transform)
        evaluate_imagenet_v2(model, os.path.join(args.data_dir, 'imagenetv2'), test_batchsize=args.batch_size, test_transform=test_transform)
        evaluate_stylized_imagenet(model, os.path.join(args.data_dir, 'imagenet-style'), test_batchsize=args.batch_size, test_transform=test_transform)
        evaluate_imagenet_c(model, os.path.join(args.data_dir, 'imagenet-c'), test_batchsize=args.batch_size, test_transform=test_transform)

if __name__ == '__main__':
    main()
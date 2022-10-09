#!/usr/bin/env python3
import argparse
import os
import torch

from timm.models import create_model
from easyrobust.benchmarks import *
from easyrobust.models import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data_dir', metavar='DIR', default='benchmarks/data',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='clip_vit_base_patch16_224',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--load', type=lambda x: x.split(","), default=None, 
                    help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )

def _merge(alpha, theta_0, theta_1):
    # interpolate between all weights in the checkpoints
    return {
        'module.'+key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }

def main():
    args = parser.parse_args()

    #############################################################
    #         Load Model
    #############################################################
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False
    )
    val_transform = model.val_preprocess

    assert len(args.load) == 2
    zeroshot_checkpoint, finetuned_checkpoint = args.load
    
    theta_0 = torch.load(zeroshot_checkpoint)
    theta_1 = torch.load(finetuned_checkpoint)

    model.eval()

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

        theta = _merge(alpha, theta_0, theta_1)

        # update the model (in-place) acccording to the new weights
        model.load_state_dict(theta)

        evaluate_imagenet_val(model, os.path.join(args.data_dir, 'imagenet-val'), test_batchsize=args.batch_size, test_transform=val_transform)
        evaluate_imagenet_a(model, os.path.join(args.data_dir, 'imagenet-a'), test_batchsize=args.batch_size, test_transform=val_transform)
        evaluate_imagenet_r(model, os.path.join(args.data_dir, 'imagenet-r'), test_batchsize=args.batch_size, test_transform=val_transform)
        evaluate_imagenet_sketch(model, os.path.join(args.data_dir, 'imagenet-sketch'), test_batchsize=args.batch_size, test_transform=val_transform)
        evaluate_imagenet_v2(model, os.path.join(args.data_dir, 'imagenetv2'), test_batchsize=args.batch_size, test_transform=val_transform)


if __name__ == '__main__':
    main()



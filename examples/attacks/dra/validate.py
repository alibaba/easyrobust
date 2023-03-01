from __future__ import print_function, division, absolute_import
import argparse
import os
import time
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np
import pretrainedmodels
import pretrainedmodels.utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFile
import torchvision.models as models
from robustness import datasets, defaults, model_utils, train
from robustness.tools import helpers

ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--input_dir', metavar='DIR', default="./SubImageNet224",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='target model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--defense_methods', default='None', type=str, metavar='PATH',
                    help='None,Augmix,SIN,SIN-IN,Linf-0.5,Linf-1.0,L2-0.05,L2-0.1,L2-0.5,L2-1.0')
# parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0

class SubsetImageNet(Dataset):
    def __init__(self, root, class_to_idx='./imagenet_class_to_idx.npy', transform=None):
        super(SubsetImageNet, self).__init__()
        self.root = root
        self.transform = transform
        img_path = os.listdir(root)
        img_path = sorted(img_path)
        self.img_path = [item for item in img_path if 'png' in item]
        self.class_to_idx = np.load(class_to_idx, allow_pickle=True)[()]

    def __getitem__(self, item):
        filepath = os.path.join(self.root, self.img_path[item])
        sample = Image.open(filepath, mode='r')

        if self.transform:
            sample = self.transform(sample)

        class_name = self.img_path[item].split('_')[0]
        label = self.class_to_idx[class_name]

        return sample, label, item

    def __len__(self):
        return len(self.img_path)
    

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t
#####################################   

def main():
    global args, best_prec1
    args = parser.parse_args()
    print("This arch:",args.arch)
    # create model
    # print("=> creating model '{}'".format(args.arch))
    robustness_flag=0
    if args.defense_methods=="None":
        model = pretrainedmodels.__dict__[args.arch](num_classes=1000,pretrained=args.pretrained)
    elif args.defense_methods=="Augmix":
        model = models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('./defense_models/checkpoint.pth.tar')
        model.load_state_dict(checkpoint["state_dict"])     
    elif args.defense_methods=="SIN":
        model = models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('./defense_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
        model.load_state_dict(checkpoint["state_dict"])     
    elif args.defense_methods=="SIN-IN":
        model = models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('./defense_models/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar')
        model.load_state_dict(checkpoint["state_dict"]) 
    elif args.defense_methods=="Linf-0.5":
        ds = datasets.ImageNet("")
        robustness_flag=1
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,resume_path="./robust_models/resnet50_linf_eps0.5.ckpt", parallel=False, add_custom_forward=True)    
    elif args.defense_methods=="Linf-1.0":
        ds = datasets.ImageNet("")
        robustness_flag=1
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,resume_path="./robust_models/resnet50_linf_eps1.0.ckpt", parallel=False, add_custom_forward=True)       
    elif args.defense_methods=="L2-0.05":
        ds = datasets.ImageNet("")
        robustness_flag=1
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,resume_path="./robust_models/resnet50_l2_eps0.05.ckpt", parallel=False, add_custom_forward=True)    
    elif args.defense_methods=="L2-0.1":
        ds = datasets.ImageNet("")
        robustness_flag=1
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,resume_path="./robust_models/resnet50_l2_eps0.1.ckpt", parallel=False, add_custom_forward=True)    
    elif args.defense_methods=="L2-0.5":
        ds = datasets.ImageNet("")
        robustness_flag=1
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,resume_path="./robust_models/resnet50_l2_eps0.5.ckpt", parallel=False, add_custom_forward=True)    
    elif args.defense_methods=="L2-1.0":
        ds = datasets.ImageNet("")
        robustness_flag=1
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,resume_path="./robust_models/resnet50_l2_eps1.0.ckpt", parallel=False, add_custom_forward=True)        
    else:
        print("!!!Make sure you set right value for defense_methods!!!")
        

    
    # Data loading code
    valdir = os.path.join(args.input_dir)    
    
    if args.arch=="inceptionv3" or args.arch=="inceptionv4" or args.arch=="inceptionresnetv2":
        val_tf = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
#             normalize,
        ])
    else:
        val_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
#             normalize,
        ])       
    val_set = SubsetImageNet(root=valdir, transform=val_tf)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()

    validate(val_loader, model, criterion,robustness_flag)


def validate(val_loader, model, criterion,robustness_flag):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        end = time.time()
        for i, raw_data in enumerate(val_loader):
            input = raw_data[0]
            target = raw_data[1]
#             target = torch.ones(target.shape).long()*30
#             target = torch.ones(target.shape).long().cuda()*919
    
            target = target.cuda()
            input = input.cuda()
        
            
            # compute output
            if robustness_flag==1:
                output,_ = model(input) #robustness lib has added normalization layer 
            else:    
                output = model(normalize(input))
            
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
#     print("size:",target.shape)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
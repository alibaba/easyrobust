from utils import log
import torch
import torchvision as tv
import time

import numpy as np

from utils.test_utils import arg_parser, get_measures,get_measures2
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.autograd import Variable
import torch.nn as nn
import math

import copy
from torchvision import transforms, utils
from timm.models import create_model
from cifar_resnet import *
        
from PIL import Image
class ImageListDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, root_path, imglist, transform=None, target_transform=None):
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform
        with open(imglist) as f:
            self._indices = f.readlines()
    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index].strip().split()
        img_path = os.path.join(self.root_path, img_path)
        img = Image.open(img_path).convert('RGB')             
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
        
        
def make_id_ood_ImageNet(args, logger):
    """Returns train and validation datasets."""

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((384, 384)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

   
    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    return in_set, out_set, in_loader, out_loader

def make_id_ood_CIFAR(args, logger):
    """Returns train and validation datasets."""
    # crop = 480
    # crop = 32
 
    imagesize = 32
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((imagesize, imagesize)),
        tv.transforms.CenterCrop(imagesize),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])        
        
    in_set = tv.datasets.CIFAR10("./ID_OOD_dataset/", 
                                   train=False, 
                                   transform=val_tx, 
                                   download=True)
    in_loader = torch.utils.data.DataLoader(in_set, batch_size=args.batch, shuffle=False, num_workers=4)
    if "SVHN" in args.out_datadir:
        out_set = tv.datasets.SVHN(
            root="./ID_OOD_dataset/", 
            split="test",
            download=True, 
            transform=val_tx)
    else:
        out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    return in_set, out_set, in_loader, out_loader



def iterate_data_msp(data_loader, model, lam, feature_std, feature_mean, bats=False):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            features = model.forward_features(x)
            
            if bast:
                features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
                features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
            
            logits = model.forward_head(features)
            # logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger, lam, feature_std, feature_mean, bats=False):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        
        features = model.forward_features(x)
        if bats:
            features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
            features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
        
        outputs = model.forward_head(features)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
                
        features = model.forward_features(Variable(tempInputs))
        
        if bats:
            features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
            features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
        
        outputs = model.forward_head(features)
        
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

    return np.array(confs)

def iterate_data_energy(data_loader, model, temper, lam, feature_std, feature_mean, bats=False):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            features = model.forward_features(x)
            if bats:
                features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
                features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
            
            logits = model.forward_head(features)
            
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)



def iterate_data_gradnorm(data_loader, model, temperature, num_classes, lam, feature_std, feature_mean, bats=False):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        
        features = model.forward_features(inputs)
        if bats:
            features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
            features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)        
        outputs = model.forward_head(features)
        
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward(retain_graph=True)
        
        if num_classes==1000:
            layer_grad = model.head.weight.grad.data
        else:
            layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


def run_eval(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()
    print("Dataset:",args.dataset,"Method:",args.score,"Using BATS:", args.bats)
    
    
    bats = args.bats
    
    if args.dataset == 'ImageNet':
        feature_std=torch.load("vit_features_std.pt").cuda()
        feature_mean=torch.load("vit_features_mean.pt").cuda()        
        lam = 1.05
    elif args.dataset == 'CIFAR':
        feature_std=torch.load("cifar_features_std.pt").cuda()
        feature_mean=torch.load("cifar_features_mean.pt").cuda()
        lam = 3.25
        
    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model,lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model,lam, feature_std, feature_mean, bats)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger,lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger,lam, feature_std, feature_mean, bats)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy,lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy,lam, feature_std, feature_mean, bats)      
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes,lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes,lam, feature_std, feature_mean, bats)     
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    fpr_list = get_measures2(in_examples, out_examples)
    print("fpr_list:",fpr_list)

    logger.info('============Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()
    

def main(args):
    logger = log.setup_logger(args)

    torch.backends.cudnn.benchmark = True

    if args.score == 'GradNorm':
        args.batch = 1

    if args.dataset == 'ImageNet':
        model = create_model("vit_base_patch16_384",pretrained=True,num_classes=1000)
        model = model.cuda()
        in_set, out_set, in_loader, out_loader = make_id_ood_ImageNet(args, logger)
        numc=1000
    elif args.dataset == 'CIFAR':    
        model = resnet18_cifar(num_classes=10)
        model.load_state_dict(torch.load("./checkpoints/resnet18_cifar10.pth")['state_dict'])
        model = model.cuda()
        in_set, out_set, in_loader, out_loader = make_id_ood_CIFAR(args, logger)
        numc=10

    start_time = time.time()
    run_eval(model, in_loader, out_loader, logger, args, num_classes=numc)
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'GradNorm'], default='Energy')
    
    parser.add_argument('--dataset', choices=['CIFAR', 'ImageNet'], default='ImageNet')
    parser.add_argument('--bats', default=0, type=int, help='Using BATS to boost the performance or not.')
    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.005, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')

    main(parser.parse_args())

    
#### Using the attack methods from advertorch
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import pretrainedmodels
from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils_data import *
from utils_sgm import *
from models.resnet import *
from models.densenet import *
from models.vgg import *

from attackutils import clamp
from attackutils import normalize_by_pnorm
from attackutils import clamp_by_pnorm
from attackutils import is_float_or_torch_tensor
from attackutils import batch_multiply
from attackutils import batch_clamp
from attackutils import replicate_input
from attackutils import batch_l1_proj
from attackutils import Attack
from attackutils import LabelMixin
from attackutils import rand_init_delta

import pdb  
 
    
############### PGD attack ########################################     
def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False,
                      clip_min=0.0, clip_max=1.0,
                      l1_sparsity=None):

    delta = torch.zeros_like(xvar)
    delta.requires_grad_()
    loss_fn=nn.CrossEntropyLoss(reduction="sum")

    if minimize:
        yvar = torch.ones(yvar.shape).long().cuda()*targetidx ### target class: 24	99	245	344	471	555	661	701	802	919
    
    for ii in range(nb_iter):
            outputs = predict(xvar + delta)
            loss = -1*outputs.gather(1,yvar.unsqueeze(1)).squeeze(1).sum()  
            if minimize:
                loss = -loss
            loss.backward()
            norm = delta.grad.data.abs().std(dim=[2,3], keepdim=True)
            grad_sign = (delta.grad.data) / norm

            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv
class PGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1., l1_sparsity=None, targeted=False):
        """
        Create an instance of the PGDAttack.
        """
        super(PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)
    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )
        return rval.data
###################################################################       
    
    
    
    

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack Evaluation')
parser.add_argument('--input_dir', default='./SubImageNet224', help='the path of original dataset')
parser.add_argument('--output_dir', default='./save', help='the path of the saved dataset')
parser.add_argument('--arch', default='resnet18',
                    help='source model for black-box attack evaluation',
                    choices=model_names)
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=16, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps. For targeted attack steps=300.')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--targetidx', default=24, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--DRA', default=0, type=int,help='Set DRA True to use our DRA models.')
parser.add_argument('--advertorch', default=0, type=int,help="using the lib advertorch.")

args = parser.parse_args()
print(args)
# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]
 
    
def generate_adversarial_example(model, data_loader, adversary, img_path):
    """
    evaluate model by black-box attack
    """
    model.eval()

    for batch_idx, (inputs, true_class, idx) in enumerate(data_loader):
        inputs, true_class = \
            inputs.to(device), true_class.to(device)

        # attack
        inputs_adv = adversary.perturb(inputs, true_class)

        save_images(inputs_adv.detach().cpu().numpy(), img_list=img_path,
                    idx=idx, output_dir=args.output_dir)
        # assert False
        if batch_idx % args.print_freq == 0:
            print('generating: [{0}/{1}]'.format(batch_idx, len(data_loader)))

def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

def main():

    try_make_dir(args.output_dir)
    if args.arch=="inceptionv3" or args.arch=="inceptionv4" or args.arch=="inceptionresnetv2":
        transform_test = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])       
    
    data_set = SubsetImageNet(root=args.input_dir, transform=transform_test)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    # create models
    if args.DRA==0:
        net = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
        model = nn.Sequential(Normalize(mean=net.mean, std=net.std), net)
        model = model.to(device)
        model.eval()

        '''
        # create adversary attack
        epsilon = args.epsilon / 255.0
        if args.step_size < 0:
            step_size = epsilon / args.num_steps
        else:
            step_size = args.step_size / 255.0
        '''

        # Skip Gradient Method (SGM)
        if args.gamma < 1.0:
            if args.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                register_hook_for_resnet(model, arch=args.arch, gamma=args.gamma)
            elif args.arch in ['densenet121', 'densenet169', 'densenet201']:
                register_hook_for_densenet(model, arch=args.arch, gamma=args.gamma)
            else:
                raise ValueError('Current code only supports resnet/densenet. '
                                 'You can extend this code to other architectures.')
    else:
        archstr = str(args.arch)
        net = pretrainedmodels.__dict__[archstr](num_classes=1000,pretrained='imagenet') 
        net = torch.nn.DataParallel(net).cuda()

        ckpt = torch.load("./DRA/DRA_"+archstr+".pth")
        if "model_state_dict" in ckpt:
            net.load_state_dict(ckpt["model_state_dict"])
            if "accuracy" in ckpt:
                print("The loaded model has Validation accuracy of: {:.2f} %\n".format(ckpt["accuracy"]))
        else:
            net.load_state_dict(ckpt)  

        model = modelsdir[archstr]
        model = nn.DataParallel(model).cuda()
        model_dict = model.state_dict()
        pre_dict = net.state_dict()
        state_dict = {k:v for k,v in pre_dict.items() if k in model_dict.keys()}
        print("Loaded pretrained weight. Len :",len(pre_dict.keys()),len(state_dict.keys()))  
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)     
        model = nn.Sequential(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), model)
        model.eval()

        
    epsilon = args.epsilon / 255.0
    if args.step_size < 0:
        step_size = epsilon / args.num_steps
    else:
        step_size = args.step_size / 255.0  
        
    if args.advertorch==1:    
        ####using the attack methods from advertorch lib.
        if args.momentum > 0.0:
            print('using PGD attack with momentum = {}'.format(args.momentum))
            adversary = MomentumIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,decay_factor=args.momentum,clip_min=0.0, clip_max=1.0, targeted=False)
        else:
            print('using linf PGD attack')
            adversary = LinfPGDAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        #### using the attack method in this code.
        print('using linf PGD attack')
        adversary = PGDAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)

    generate_adversarial_example(model=model, data_loader=data_loader,
                                 adversary=adversary, img_path=data_set.img_path)


    
if __name__ == '__main__':
    modelsdir = {
        'resnet50': resnet50(),
        'resnet152': resnet152(),
        'densenet121': densenet121(),
        'densenet201': densenet201(),
        'vgg19_bn': vgg19_bn(),
    }
    main()

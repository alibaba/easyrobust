import argparse
from PIL import Image

import torch
from torch.utils import model_zoo
from torchvision import utils
from timm.models import create_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--ckpt_path', default='', type=str, required=True,
                    help='model architecture (default: dpn92)')
parser.add_argument('--use_ema', dest='use_ema', action='store_true',
                    help='use use_ema model state_dict')
parser.add_argument('--vis_num', default=64, type=int)
                    
args = parser.parse_args()

def visTensor(tensor, kernel_num=64, nrow=8, padding=1): 
    print(tensor.shape)
    n,c,w,h = tensor.shape
    assert kernel_num <= n and c == 3

    if kernel_num == n:
        tensor = tensor
    else:
        cc = torch.pca_lowrank(tensor.reshape(n,-1).transpose(0,1), q=kernel_num)
        dd = torch.matmul(tensor.reshape(n,-1).transpose(0,1), cc[2])
        tensor = dd.transpose(0,1).reshape(kernel_num,3,w,h)

    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding).numpy().transpose((1, 2, 0))
    Image.fromarray((grid*255.).astype('uint8')).save('vis_filters.png')
    
if __name__ == "__main__":
    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000
    )

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
        state_dict = {k[2:]:v for k,v in state_dict.items()}
        del state_dict['mean']
        del state_dict['std']
            
    model.load_state_dict(state_dict)

    if args.model in ['resnet50']:
        filter = model.conv1.weight.data.clone()
    elif args.model == ['vit_small_patch16_224', 'vit_base_patch16_224']:
        filter = model.patch_embed.proj.weight.data.clone()
    else:
        raise Exception('{} is not supported now!'.format(args.model))
    
    visTensor(filter, kernel_num=args.vis_num)
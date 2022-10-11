"""
steal from https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
"""

import argparse
from PIL import Image
import requests
import io
import numpy as np

import torch
from torch.utils import model_zoo
from torchvision import transforms
from timm.models import create_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='vit_base_patch16_224',
                    help='model architecture (default: resnet50)')
parser.add_argument('--ckpt_path', default='', type=str, required=True,
                    help='model architecture (default: dpn92)')
parser.add_argument('--input_image', default='', type=str, help='url or path')

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
parser.add_argument('--use_ema', dest='use_ema', action='store_true',
                    help='use use_ema model state_dict')
                    
args = parser.parse_args()

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

activation = {}


def get_attn_softmax(name):
    def hook(model, input, output):
        with torch.no_grad():
            input = input[0]
            B, N, C = input.shape
            qkv = (
                model.qkv(input)
                .detach()
                .reshape(B, N, 3, model.num_heads, C // model.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * model.scale
            attn = attn.softmax(dim=-1)
            activation[name] = attn

    return hook

# expects timm vis transformer model
def add_attn_vis_hook(model):
    for idx, module in enumerate(list(model.blocks.children())):
        module.attn.register_forward_hook(get_attn_softmax(f"attn{idx}"))

def get_mask(im,att_mat):
    # Average the attention weights across all heads.
    # att_mat,_ = torch.max(att_mat, dim=1)
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).unsqueeze(0).unsqueeze(0)
    mask = torch.nn.functional.interpolate(mask / mask.max(), size=im.size[::-1]).squeeze().detach().numpy()[..., np.newaxis]
    result = (mask * im).astype("uint8")
    return result, joint_attentions, grid_size

def show_attention_map(model, args):
    add_attn_vis_hook(model)

    if args.input_image.startswith('http'):
        im = download_image(args.input_image).convert('RGB')
    else:
        im = Image.open(args.input_image).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(int(args.input_size/args.crop_pct), interpolation=args.interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])

    model(transform(im).unsqueeze(0))

    attn_weights_list = list(activation.values())

    result, _, _ = get_mask(im,torch.cat(attn_weights_list))

    Image.fromarray(result).save('images/vit_attn.jpg')

def main(args):
    if args.input_image == '':
        return

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes
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
    model.eval()
    show_attention_map(model, args)
        

    
if __name__ == "__main__":

    main(args)

    

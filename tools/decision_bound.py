import argparse
from PIL import Image
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import model_zoo
from torchvision import transforms
from timm.models import create_model

np.random.seed(1)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
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
parser.add_argument('--max-eps', default=6, type=float,
                metavar='N', help='attack epsilon')
parser.add_argument('--eps-step', default=0.3, type=float,
                metavar='N', help='attack step')
                    
args = parser.parse_args()

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def get_decision_boundary(model, xi, max_eps=6, eps_step=0.3, num_classes=1000):
    lab2color = np.random.uniform(0,1,size=(num_classes,3))
    xi.requires_grad = True
    pred = model(xi)

    _, out = torch.max(pred, dim=1)
    print('pred: {}'.format(out.item()))
    xi.grad = None
    pred[0][out.item()].backward()
    g1 = xi.grad.data.detach()
    g1 = g1 / g1.norm()

    g2 = torch.randn(*g1.shape).cuda()
    g2 = g2 / g2.norm()
    g2 = g2 - torch.dot(g1.view(-1),g2.view(-1)) * g1
    g2 = g2 / g2.norm()
    assert torch.dot(g1.view(-1),g2.view(-1)) < 1e-6
    
    x_epss = y_epss = np.arange(-max_eps, max_eps, eps_step)
    to_plt = []
    with torch.no_grad():
        for j, x_eps in enumerate(x_epss):
            x_inp = xi + x_eps * g1 + torch.FloatTensor(y_epss.reshape(-1, 1, 1, 1)).cuda() * g2
            pred = model(x_inp)
            pred_c = torch.max(pred, dim=1)[1].cpu().detach().numpy()
            to_plt.append(lab2color[pred_c])

    to_plt = np.array(to_plt)

    plt.imshow(to_plt)
    plt.plot((len(x_epss)-1)/2,(len(y_epss)-1)/2,'ro')
    plt.axvline((len(x_epss)-1)/2, ymin=0.5, color='k', ls='--')
    plt.axis('off')
    plt.savefig('images/vis_decision_bound.jpg', bbox_inches='tight', pad_inches=-0.1)

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
    model.cuda()
    
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

    xi = transform(im).unsqueeze(0).cuda()
    
    get_decision_boundary(model, xi, num_classes=args.num_classes, max_eps=args.max_eps, eps_step=args.eps_step)

if __name__ == "__main__":
    main(args)
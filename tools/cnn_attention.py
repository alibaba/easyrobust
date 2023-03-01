# @misc{jacobgilpytorchcam,
#   title={PyTorch library for CAM methods},
#   author={Jacob Gildenblat and contributors},
#   year={2021},
#   publisher={GitHub},
#   howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}},
# }

import argparse
import PIL
from PIL import Image
import requests
import io
import numpy as np

import torch
from torch.utils import model_zoo
from torchvision import transforms
from timm.models import create_model

from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
    

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--ckpt_path', default='', type=str, required=True,
                    help='model architecture (default: dpn92)')
parser.add_argument('--input_image', default='', type=str, help='url or path')
parser.add_argument('--target_class', default=None, type=int)

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
parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'])
                    
args = parser.parse_args()

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

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

    target_layers = [model.layer4]

    preprocess2tensor = transforms.Compose([
        transforms.Resize(int(args.input_size/args.crop_pct), interpolation=args.interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])
    preprocess = transforms.Compose([
        transforms.Resize(int(args.input_size/args.crop_pct), interpolation=args.interpolation),
        transforms.CenterCrop(args.input_size)
    ])

    methods = { "gradcam": GradCAM,
                "hirescam":HiResCAM,
                "scorecam": ScoreCAM,
                "gradcam++": GradCAMPlusPlus,
                "ablationcam": AblationCAM,
                "xgradcam": XGradCAM,
                "eigencam": EigenCAM,
                "eigengradcam": EigenGradCAM,
                "layercam": LayerCAM,
                "fullgrad": FullGrad,
                "gradcamelementwise": GradCAMElementWise}

    targets = args.target_class

    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=True) as cam:


        cam.batch_size = 1

        if args.input_image.startswith('http'):
            raw_image = download_image(args.input_image).convert('RGB')
        else:
            raw_image = Image.open(args.input_image).convert('RGB')

        input_tensor = preprocess2tensor(raw_image).unsqueeze(0)
        np_img = np.array(preprocess(raw_image))/255.

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image).save('images/cnn_attn.jpg')

    
if __name__ == "__main__":
    main(args)
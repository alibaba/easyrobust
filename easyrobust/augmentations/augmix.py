"""
@article{hendrycks2020augmix,
  title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
  author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2020}
}
"""

import numpy as np
from PIL import Image
from PIL import Image, ImageOps, ImageEnhance
import torch
import torch.nn as nn

class AugMixAugmentations(object):
    def __init__(self, img_size=32, all_ops=False):
        self.img_size = img_size
        self.all_ops = all_ops

        def int_parameter(level, maxval):
            return int(level * maxval / 10)
        def float_parameter(level, maxval):
            return float(level) * maxval / 10.
        def sample_level(n):
            return np.random.uniform(low=0.1, high=n)
        def autocontrast(pil_img, _):
            return ImageOps.autocontrast(pil_img)
        def equalize(pil_img, _):
            return ImageOps.equalize(pil_img)
        def posterize(pil_img, level):
            level = int_parameter(sample_level(level), 4)
            return ImageOps.posterize(pil_img, 4 - level)
        def rotate(pil_img, level):
            degrees = int_parameter(sample_level(level), 30)
            if np.random.uniform() > 0.5:
                degrees = -degrees
            return pil_img.rotate(degrees, resample=Image.BILINEAR)
        def solarize(pil_img, level):
            level = int_parameter(sample_level(level), 256)
            return ImageOps.solarize(pil_img, 256 - level)
        def shear_x(pil_img, level):
            level = float_parameter(sample_level(level), 0.3)
            if np.random.uniform() > 0.5:
                level = -level
            return pil_img.transform((self.img_size, self.img_size),
                                    Image.AFFINE, (1, level, 0, 0, 1, 0),
                                    resample=Image.BILINEAR)
        def shear_y(pil_img, level):
            level = float_parameter(sample_level(level), 0.3)
            if np.random.uniform() > 0.5:
                level = -level
            return pil_img.transform((self.img_size, self.img_size),
                                    Image.AFFINE, (1, 0, 0, level, 1, 0),
                                    resample=Image.BILINEAR)
        def translate_x(pil_img, level):
            level = int_parameter(sample_level(level), self.img_size / 3)
            if np.random.random() > 0.5:
                level = -level
            return pil_img.transform((self.img_size, self.img_size),
                                    Image.AFFINE, (1, 0, level, 0, 1, 0),
                                    resample=Image.BILINEAR)
        def translate_y(pil_img, level):
            level = int_parameter(sample_level(level), self.img_size / 3)
            if np.random.random() > 0.5:
                level = -level
            return pil_img.transform((self.img_size, self.img_size),
                                    Image.AFFINE, (1, 0, 0, 0, 1, level),
                                    resample=Image.BILINEAR)
        # operation that overlaps with ImageNet-C's test set
        def color(pil_img, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Color(pil_img).enhance(level)
        # operation that overlaps with ImageNet-C's test set
        def contrast(pil_img, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Contrast(pil_img).enhance(level)
        # operation that overlaps with ImageNet-C's test set
        def brightness(pil_img, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Brightness(pil_img).enhance(level)
        # operation that overlaps with ImageNet-C's test set
        def sharpness(pil_img, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Sharpness(pil_img).enhance(level)
        augmentations = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y
        ]
        augmentations_all = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y, color, contrast, brightness, sharpness
        ]
        if self.all_ops:
            self.aug_list = augmentations_all
        else:
            self.aug_list = augmentations

class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, mixture_width=3, aug_severity=1, img_tuple_num=1, img_size=32, all_ops=False, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.img_tuple_num = img_tuple_num
        self.mixture_width = mixture_width
        self.aug_severity = aug_severity

        augmix_aug = AugMixAugmentations(img_size=img_size, all_ops=all_ops)
        self.aug_list = augmix_aug.aug_list

    def aug(self, image, preprocess):
        """Perform augmentation operations on PIL.Images.
        Args:
            image: PIL.Image input image
            preprocess: Preprocessing function which should return a torch tensor.
        Returns:
            image_aug: Augmented image.
        """
        image_aug = image.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(self.aug_list)
            image_aug = op(image_aug, self.aug_severity)

        image_aug = preprocess(image_aug)

        return image_aug

    def __getitem__(self, i):
        x, y = self.dataset[i]

        img_tuples = []
        for _ in range(self.img_tuple_num):
            img_tuple = [self.preprocess(x)]
            for _ in range(self.mixture_width):
                img_tuple.append(
                    self.aug(x, self.preprocess)
                )
            img_tuples.append(img_tuple)
        return img_tuples, y

    def __len__(self):
        return len(self.dataset)

class AugMixModule(nn.Module):
    def __init__(self, mixture_width, device='cuda'):
        super(AugMixModule, self).__init__()

        self.mixture_width = mixture_width
        self.w_dist = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.] * mixture_width)) # Dirichlet parameter must be float in pytorch
        self.m_dist = torch.distributions.beta.Beta(1, 1)

        self.device = device

    def forward(self, xs):
        '''
        Args:
            xs: tuple of Tensors. len(x)=3. xs = (x_ori, x_aug1, x_aug2, x_aug3). x_ori.size()=(N,W,H,C)
        '''
        # mixture_width = len(xs) - 1

        x_ori = xs[0]
        N = x_ori.size()[0]

        w = self.w_dist.sample([N]).to(self.device)
        m = self.m_dist.sample([N]).to(self.device)

        x_mix = torch.zeros_like(x_ori).to(self.device)
        for i, x_aug in enumerate(xs[1:]):
            wi = w[:,i].view((N,1,1,1)).expand_as(x_aug)
            x_mix += wi * x_aug 

        m = m.view((N,1,1,1)).expand_as(x_ori)
        x_mix = (1-m) * x_ori + m * x_mix

        return x_mix 

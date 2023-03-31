import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from util import task
from .image_folder import make_dataset
import random
import numpy as np
import copy
import skimage.morphology as sm
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True


######################################################################################
# Create the dataloader
######################################################################################
class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_paths, self.img_size = make_dataset(opt.img_file)
        if opt.mask_file != 'none':         # load the random mask files for training and testing
            self.mask_paths, self.mask_size = make_dataset(opt.mask_file)
        self.transform = get_transform(opt, convert=False, augment=False)
        fixed_opt = copy.deepcopy(opt)
        fixed_opt.preprocess = 'scale_longside'
        fixed_opt.load_size = fixed_opt.fixed_size
        fixed_opt.no_flip = True
        self.transform_fixed = get_transform(fixed_opt, convert=True, augment=False)

    def __len__(self):
        """return the total number of examples in the dataset"""
        return self.img_size

    def __getitem__(self, item):
        """return a data point and its metadata information"""
        # load the image and conditional input
        img_org, img, img_path = self._load_img(item)
        if self.opt.batch_size > 1: # padding the image to the same size for batch training
            img_org = transforms.functional.pad(img_org, (0, 0, self.opt.fine_size-self.img_h, self.opt.fine_size-self.img_w))
            img = transforms.functional.pad(img, (0, 0, self.opt.fixed_size - img.size(-1), self.opt.fixed_size - img.size(-2)))
        pad_mask = torch.zeros_like(img_org)
        pad_mask[:, :self.img_w, :self.img_h] = 1
        # load the mask
        mask, mask_type = self._load_mask(item, img_org)
        if self.opt.reverse_mask:
            if self.opt.isTrain:
                mask = 1 - mask if random.random() > 0.8 else mask
            else:
                mask = 1 - mask
        return {'img_org': img_org, 'img': img, 'img_path': img_path, 'mask': mask, 'pad_mask': pad_mask}

    def name(self):
        return ""

    def _load_img(self, item):
        """load the original image and preprocess image"""
        img_path = self.img_paths[item % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img_org = self.transform(img_pil)
        img = self.transform_fixed(img_org)
        img_org = transforms.ToTensor()(img_org)
        img_pil.close()
        self.img_c, self.img_w, self.img_h = img_org.size()
        return img_org, img, img_path

    def _mask_dilation(self, mask):
        """mask erosion for different region"""
        mask = np.array(mask)
        pixel = np.random.randint(3, 25)
        mask = sm.erosion(mask, sm.square(pixel)).astype(np.uint8)

        return mask

    def _load_mask(self, item, img):
        """load the mask for image completion task"""
        c, h, w = img.size()
        if isinstance(self.opt.mask_type, list):
            mask_type_index = random.randint(0, len(self.opt.mask_type) - 1)
            mask_type = self.opt.mask_type[mask_type_index]
        else:
            mask_type = self.opt.mask_type

        if mask_type == 0:                                  # center mask
            if random.random() > 0.3 and self.opt.isTrain:
                return task.random_regular_mask(img), mask_type # random regular mask
            return task.center_mask(img), mask_type
        elif mask_type == 1:                                # random regular mask
            return task.random_regular_mask(img), mask_type
        elif mask_type == 2:                                # random irregular mask
            return task.random_irregular_mask(img), mask_type
        elif mask_type == 3:
            # external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"
            if self.opt.isTrain:
                mask_index = random.randint(0, self.mask_size-1)
                mask_transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.RandomCrop([self.opt.fine_size + 64, self.opt.fine_size + 64]),
                        transforms.Resize([h, w])
                    ]
                )
            else:
                mask_index = item
                mask_transform = transforms.Compose(
                    [
                        transforms.Resize([h, w])
                    ]
                )
            mask_pil = Image.open(self.mask_paths[mask_index]).convert('L')
            mask = mask_transform(mask_pil)
            mask_pil.close()
            if self.opt.isTrain:
                mask = self._mask_dilation(mask)
            else:
                mask = np.array(mask) < 128
            mask = torch.tensor(mask).view(1, h, w).float()
            return mask, mask_type
        else:
            raise NotImplementedError('mask type [%s] is not implemented' % str(mask_type))


def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batch_size, shuffle=not opt.no_shuffle,
                              num_workers=int(opt.nThreads), drop_last=True)

    return dataset


######################################################################################
# Basic image preprocess function
######################################################################################
def _make_power_2(img, power, method=Image.BICUBIC):
    """resize the image to the size of log2(base) times"""
    ow, oh = img.size
    base = 2 ** power
    nw, nh = int(max(1, round(ow / base)) * base), int(max(1, round(oh / base)) * base)
    if nw == ow and nh == oh:
        return img
    return img.resize((nw, nh), method)


def _random_zoom(img, target_width, method=Image.BICUBIC):
    """random resize the image scale"""
    zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    ow, oh = img.size
    nw, nh = int(round(max(target_width, ow * zoom_level[0]))), int(round(max(target_width, oh * zoom_level[1])))
    return img.resize((nw, nh), method)


def _scale_shortside(img, target_width, method=Image.BICUBIC):
    """resize the short side to the target width"""
    ow, oh = img.size
    shortsize = min(ow, oh)
    scale = target_width / shortsize
    return img.resize((round(ow * scale), round(oh * scale)), method)


def _scale_longside(img, target_width, method=Image.BICUBIC):
    """resize the long side to the target width"""
    ow, oh = img.size
    longsize = max(ow, oh)
    scale = target_width / longsize
    return img.resize((round(ow * scale), round(oh * scale)), method)


def _scale_randomside(img, target_width, method=Image.BICUBIC):
    """resize the side to the target width with random side"""
    if random.random() > 0.5:
        return _scale_shortside(img, target_width, method)
    else:
        return _scale_longside(img, target_width, method)


def _crop(img, pos=None, size=None):
    """crop the image based on the given pos and size"""
    ow, oh = img.size
    if size is None:
        return img
    nw = min(ow, size)
    nh = min(oh, size)
    if (ow > nw or oh > nh):
        if pos is None:
            x1 = np.random.randint(0, int(ow-nw)+1)
            y1 = np.random.randint(0, int(oh-nh)+1)
        else:
            x1, y1 = pos
        return img.crop((x1, y1, x1 + nw, y1 + nh))
    return img


def _pad(img):
    """expand the image to the square size"""
    ow, oh = img.size
    size = max(ow, oh)
    return ImageOps.pad(img, (size, size), centering=(0, 0))


def _flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_transform(opt, params=None, method=Image.BICUBIC, convert=True, augment=False):
    """get the transform functions"""
    transforms_list = []
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transforms_list.append(transforms.Resize(osize))
    elif 'scale_shortside' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _scale_shortside(img, opt.load_size, method)))
    elif 'scale_longside' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _scale_longside(img, opt.load_size, method)))
    elif "scale_randomside" in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _scale_randomside(img, opt.load_size, method)))

    if 'zoom' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _random_zoom(img, opt.load_size, method)))

    if 'crop' in opt.preprocess and opt.isTrain:
        transforms_list.append(transforms.Lambda(lambda img: _crop(img, size=opt.fine_size)))
    if 'pad' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _pad(img)))     # padding image to square

    transforms_list.append(transforms.Lambda(lambda img: _make_power_2(img, opt.data_powers, method)))

    if not opt.no_flip and opt.isTrain:
        transforms_list.append(transforms.RandomHorizontalFlip())

    if augment and opt.isTrain:
        transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))

    if convert:
        transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)
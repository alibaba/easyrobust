import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import random


###################################################################
# random mask generation
###################################################################
def random_regular_mask(img):
    """Generate a random regular mask
    :param img: original image size  C*H*W
    :return: mask
    """
    mask = torch.ones_like(img)[0:1, :, :]
    s = img.size()
    N_mask = random.randint(1, 5)
    lim_x = s[1] - s[1] / (N_mask + 1)
    lim_y = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(lim_x))
        y = random.randint(0, int(lim_y))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), min(int(s[1] - x), int(s[1] / 2)))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), min(int(s[2] - y), int(s[2] / 2)))
        mask[:, int(x) : int(range_x), int(y) : int(range_y)] = 0
    return mask


def center_mask(img):
    """Generate a center hole with 1/4*W and 1/4*H
    :param img: original image size C*H*W
    :return: mask
    """
    mask = torch.ones_like(img)[0:1, :, :]
    s = img.size()
    mask[:, int(s[1]/4):int(s[1]*3/4), int(s[2]/4):int(s[2]*3/4)] = 0
    return mask


def random_irregular_mask(img):
    """Generate a random irregular mask with lines, circles and ellipses
    :param img: original image size C*H*W
    :return: mask
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)[0:1, :, :]
    s = mask.size()
    img = np.zeros((s[1], s[2], 1), np.uint8)

    max_width = int(min(s[1]/10, s[2]/10))
    N_mask = random.randint(16, 64)
    for _ in range(N_mask):
        model = random.random()
        if model < 0.2: # Draw random lines
            x1, x2 = random.randint(1, s[1]), random.randint(1, s[1])
            y1, y2 = random.randint(1, s[2]), random.randint(1, s[2])
            thickness = random.randint(2, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)
        elif (model > 0.2 and model < 0.5): # Draw random circles
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            radius = random.randint(2, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)
        else: # draw random ellipses
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            s1, s2 = random.randint(1, s[1]), random.randint(1, s[2])
            a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
            thickness = random.randint(2, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(s[2], s[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(s[0]):
        mask[j, :, :] = img_mask

    return mask


def scale_img(img, size):
    h_ratio = img.size(-1) // size[-1]
    w_ratio = img.size(-2) // size[-2]
    scaled_img = F.avg_pool2d(img, kernel_size=(w_ratio, h_ratio), stride=(w_ratio, h_ratio))
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    for i in range(1, num_scales):
        ratio = 2**i
        scaled_img = F.avg_pool2d(img, kernel_size=ratio, stride=ratio)
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs


def jacobian(y, x, point=None, create_graph=True):
    """Calculate the jacobian matrix for given point"""
    jac = []
    flat_y = y.reshape(-1)
    b, c, h, w = y.size()
    if point is not None:
        i = point[0] * h + point[1]
        input_y = flat_y[i]
        grad_x = torch.autograd.grad(input_y, x, retain_graph=True, grad_outputs=torch.ones(input_y.size()).to(x.device),
                                     create_graph=create_graph, only_inputs=True)[0]
        jac.append(grad_x.reshape(x.shape))
        return jac
    else:
        for i in range(len(flat_y)):
            input_y = flat_y[i]
            grad_x = torch.autograd.grad(input_y, x, retain_graph=True, grad_outputs=torch.ones(input_y.size()).to(x.device),
                                         create_graph=create_graph, only_inputs=True)[0]
            jac.append(grad_x.reshape(x.shape))
        return torch.stack(jac).reshape(y.shape + x.shape)
#!/usr/bin/python
#****************************************************************#
# ScriptName: change_place.py
# Author: Anonymous_123
# Create Date: 2022-08-26 14:13
# Modify Author: Anonymous_123
# Modify Date: 2022-08-26 14:13
# Function: 
#***************************************************************#

import os
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
import cv2
from PIL import Image
import numpy as np
import random
# random.seed(0)
import pdb
import imutils
from tqdm import tqdm

def change_place(img, mask, bbox, invert_mask):
    '''
    img: N,C,H,W
    '''
    if invert_mask:
        mask = 1-mask

    device = img.device
    x,y,new_x,new_y,w,h = bbox

    img_ori = img.clone()
    mask_ori = mask.clone()
    img_ori = img_ori.to(device)
    mask_ori = mask_ori.to(device)

    img[:,:, new_y:new_y+h, new_x:new_x+w] = img_ori[:,:, y:y+h, x:x+w]
    mask_new = torch.zeros(mask.shape).to(device)
    mask_new[:,:, new_y:new_y+h, new_x:new_x+w] = mask_ori[:,:, y:y+h, x:x+w]
    mask_ = mask_new > 0.5
    img = img*mask_ + (~mask_)*img_ori

    if invert_mask:
        mask_new = 1 - mask_new

    return img, mask_new

def find_bbox(mask):
    mask_copy = mask.copy()
    
    contours, _ = cv2.findContours(mask[:,:,0],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(mask_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if max_area < w*h:
            max_area = w*h
            bbox = [x,y,w,h]

    if bbox == []:
        return None
    else:
        H,W,C = mask.shape
        x,y,w,h = bbox
        new_x = random.randint(0, W-w)
        new_y = random.randint(0, H-h)
        return [x, y, new_x, new_y, w,h]


if __name__ == '__main__':
    mask_path = 'n01440764/ILSVRC2012_val_00000293.png'

    ori_img_path_root = 'ImageNet-S/ImageNetS919/validation/'
    outpainting_root = 'TFill/results/imagenet_2/test_latest/img_ref_out/'
    padding_root = 'ImageNet-S/ImageNetS919/validation-size-0.05-padding-4901/'
    mask_root = 'ImageNet-S/ImageNetS919/validation-segmentation-label-mask/'


    imgs = os.listdir(outpainting_root)

    shape = (256,256)
    for cls in tqdm(os.listdir(mask_root)):
        for img_name in os.listdir(os.path.join(mask_root, cls)):
            if not img_name.split('.')[0]+'_0.png' in imgs:
                continue
            img_path = os.path.join(ori_img_path_root, cls, img_name.split('.')[0]+'.JPEG')
            img_path_init = os.path.join(outpainting_root, img_name.split('.')[0]+'_0.png')
            img_path_2 = os.path.join(padding_root, cls, img_name.split('.')[0]+'.JPEG')
            mask_path = os.path.join(mask_root, cls, img_name)
            if os.path.exists(img_path) and os.path.exists(img_path_init) and os.path.exists(img_path_2) and os.path.exists(mask_path):
                img = Image.open(img_path_2).convert('RGB')
                img = img.resize(shape, Image.LANCZOS)
                img = TF.to_tensor(img).unsqueeze(0).mul(2).sub(1)
            
                mask = Image.open(mask_path).convert('RGB')
                mask = mask.resize(shape, Image.NEAREST)
                bbox = find_bbox(np.array(mask))
            
                mask = ((np.array(mask) > 0.5) * 255).astype(np.uint8)
            
                mask = TF.to_tensor(Image.fromarray(mask))
                mask = mask[0, ...].unsqueeze(0).unsqueeze(0)
            
                if bbox is not None:
                    img, mask = change_place(img, mask, bbox)
            
                img_init = Image.open(img_path_init).convert('RGB')
                img_init = img_init.resize(shape, Image.LANCZOS)
                img_init = TF.to_tensor(img_init).unsqueeze(0).mul(2).sub(1)
                img_new = img_init*(1-mask) + img*mask
            
                img_new = np.transpose(((img_new+1)/2*255)[0].numpy(), (1,2,0))[:,:,::-1]
                img_init = cv2.imread(img_path)
                img_init = cv2.resize(img_init, shape)
                # cv2.imwrite('tmp/'+img_name, cv2.hconcat([img_init, img_new.astype('uint8')]))
                cv2.imwrite('tmp/'+img_name, img_new.astype('uint8'))
            
            

#!/usr/bin/python
#****************************************************************#
# ScriptName: fft_pytorch.py
# Author: Anonymous_123
# Create Date: 2022-08-15 11:33
# Modify Author: Anonymous_123
# Modify Date: 2022-08-18 17:46
# Function: 
#***************************************************************#

import torch
import torch.nn as nn
import torch.fft as fft
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def lowpass(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)
    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input*kernel, s=input.shape[-2:])

class HighFrequencyLoss(nn.Module):
    def __init__(self, size=(224,224)):
        super(HighFrequencyLoss, self).__init__()
        '''
        self.h,self.w = size
        self.lpf = torch.zeros((self.h,1))
        R = (self.h+self.w)//8
        for x in range(self.w):
            for y in range(self.h):
                if ((x-(self.w-1)/2)**2 + (y-(self.h-1)/2)**2) < (R**2):
                    self.lpf[y,x] = 1
        self.hpf = 1-self.lpf
        '''

    def forward(self, x):
        f = fft.fftn(x, dim=(2,3))
        loss = torch.abs(f).mean()

        # f = torch.roll(f,(self.h//2,self.w//2),dims=(2,3))
        # f_l = torch.mean(f * self.lpf)
        # f_h = torch.mean(f * self.hpf)

        return loss

if __name__  == '__main__':
    import pdb
    pdb.set_trace()
    HF = HighFrequencyLoss()
    transform = transforms.Compose([transforms.ToTensor()])

    # img = cv2.imread('test_imgs/ILSVRC2012_val_00001935.JPEG')
    img = cv2.imread('../tmp.jpg')
    H,W,C = img.shape
    imgs = []
    for i in range(10):
        img_ = img[:, 224*i:224*(i+1), :]
        print(img_.shape)
        img_tensor = transform(Image.fromarray(img_[:,:,::-1])).unsqueeze(0)
        loss = HF(img_tensor).item()
        cv2.putText(img_, str(loss)[:6], (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        imgs.append(img_)

    cv2.imwrite('tmp.jpg', cv2.hconcat(imgs))





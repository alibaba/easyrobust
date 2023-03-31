#!/usr/bin/python
#****************************************************************#
# ScriptName: analysis_data.py
# Author: Anonymous_123
# Create Date: 2022-07-25 19:54
# Modify Author: Anonymous_123
# Modify Date: 2022-09-25 12:04
# Function: 
#***************************************************************#

import os
import sys
import numpy as np
import cv2
import torch
from tqdm import tqdm
import shutil
import pdb

import argparse

parser = argparse.ArgumentParser(description='resize object')
parser.add_argument('--scale', type=float, default=None, help='object scale')
parser.add_argument('--img_path', type=str, help='image path')
parser.add_argument('--mask_path', type=str, help='mask path')


def get_bbox_and_rate(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None, None
    max_area = 0
    max_idx = 0
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h > max_area:
            max_idx = i
            max_area = w*h
    # 外接矩形
    x,y,w,h = cv2.boundingRect(contours[max_idx])
    mask_new = np.zeros(mask.shape, dtype='uint8')
    mask_new[y:y+h, x:x+w, :] = mask[y:y+h, x:x+w, :]

    rate = (mask_new[:,:,0]>127.5).sum()/mask.shape[0]/mask.shape[1]

    return (x,y,w,h), rate

def resize_around_the_center(img, mask, bbox, operation, scale_step=1.2):
    x,y,w,h = bbox
    H,W,C = mask.shape
    obj_mask = mask[y:y+h, x:x+w, :].copy()
    # obj_mask = cv2.resize(obj_mask, (int(w*scale_step),int(h*scale_step)) if operation == 'upsample' else (int(w/scale_step), int(h/scale_step)))
    obj_mask = cv2.resize(obj_mask, (int(w*scale_step),int(h*scale_step)))
    start_point_x = max(x+w//2 - obj_mask.shape[1]//2, 0) # center - w
    start_point_y = max(y+h//2 - obj_mask.shape[0]//2, 0) # center - h
    end_point_x = min(x+w//2 + obj_mask.shape[1]//2, W) # center+w
    end_point_y = min(y+h//2 + obj_mask.shape[0]//2, H) # center+h

    start_point_x_obj = max(0,obj_mask.shape[1]//2-(x+w//2))
    start_point_y_obj = max(0, obj_mask.shape[0]//2-(y+h//2))
    mask[:] = 0
    mask[start_point_y:end_point_y, start_point_x:end_point_x] = obj_mask[start_point_y_obj:start_point_y_obj+(end_point_y-start_point_y), start_point_x_obj:start_point_x_obj+(end_point_x-start_point_x)]

    obj_img = img[y:y+h, x:x+w, :].copy()
    # obj_img = cv2.resize(obj_img, (int(w*scale_step),int(h*scale_step)) if operation == 'upsample' else (int(w/scale_step), int(h/scale_step)))
    obj_img = cv2.resize(obj_img, (int(w*scale_step),int(h*scale_step)))
    img = cv2.GaussianBlur(img, (49, 49), 0)
    img[start_point_y:end_point_y, start_point_x:end_point_x] = obj_img[start_point_y_obj:start_point_y_obj+(end_point_y-start_point_y), start_point_x_obj:start_point_x_obj+(end_point_x-start_point_x)]

    return img, mask

def resize_around_the_center_padding(img, mask, bbox, scale_step=1.2):
    x,y,w,h = bbox
    H,W,C = mask.shape
    mask_new = np.zeros((int(H/scale_step), int(W/scale_step), 3), dtype='uint8')
    mask_new_full = np.zeros((int(H/scale_step), int(W/scale_step), 3), dtype='uint8')
    # img_new = np.zeros((int(H/scale_step), int(W/scale_step), 3), dtype='uint8')
    img_new = cv2.resize(img, (int(W/scale_step), int(H/scale_step)))

    if scale_step < 1:
        mask_new[int((y+h/2)*(1/scale_step-1)):int((y+h/2)*(1/scale_step-1)+H), int((x+w/2)*(1/scale_step-1)):int((x+w/2)*(1/scale_step-1)+W)] = mask
        mask_new_full[int((y+h/2)*(1/scale_step-1)):int((y+h/2)*(1/scale_step-1)+H), int((x+w/2)*(1/scale_step-1)):int((x+w/2)*(1/scale_step-1)+W)] = mask.max()*np.ones(mask.shape, dtype='uint8')

        img_new[int((y+h/2)*(1/scale_step-1)):int((y+h/2)*(1/scale_step-1)+H), int((x+w/2)*(1/scale_step-1)):int((x+w/2)*(1/scale_step-1)+W)] = img

    else:
        mask_new = mask[int((y+h/2)*(1-1/scale_step)):int((y+h/2)*(1-1/scale_step))+int(H/scale_step), int((x+w/2)*(1-1/scale_step)):int((x+w/2)*(1-1/scale_step))+int(W/scale_step)]
        mask_new_full = mask[int((y+h/2)*(1-1/scale_step)):int((y+h/2)*(1-1/scale_step))+int(H/scale_step), int((x+w/2)*(1-1/scale_step)):int((x+w/2)*(1-1/scale_step))+int(W/scale_step)]
        img_new = img[int((y+h/2)*(1-1/scale_step)):int((y+h/2)*(1-1/scale_step))+int(H/scale_step), int((x+w/2)*(1-1/scale_step)):int((x+w/2)*(1-1/scale_step))+int(W/scale_step)]

    img_new = cv2.resize(img_new, (W,H))
    mask_new = cv2.resize(mask_new, (W,H))
    mask_new_full = cv2.resize(mask_new_full, (W,H))
   
    return img_new, mask_new, mask_new_full

def rescale(img, mask, scale=None, max_steps=50):
    bbox, rate = get_bbox_and_rate(mask)
    if bbox is None:
        return None, None, None
    num_steps = 0
    mask_full = mask.copy()
    while np.floor(rate*100) != scale*100. and abs(rate-scale) > 0.015:
    # while not (abs(bbox[0]-0)<10 or abs(bbox[1]-0)<10 or abs(bbox[0]+bbox[2]-img.shape[1])<10 or abs(bbox[1]+bbox[3]-img.shape[0])<10):
        operation = 'upsample' if np.floor(rate*100) < scale*100. else 'downsample'
        scale_step = np.sqrt(scale/rate)
        # img, mask = resize_around_the_center(img, mask, bbox, operation, scale_step=scale_step)
        img, mask, mask_full = resize_around_the_center_padding(img, mask, bbox, scale_step=scale_step)
        bbox, rate_ = get_bbox_and_rate(mask)
        if (operation == 'upsample' and rate_ < rate) or (operation == 'downsample' and rate_ > rate):
            return None, None, None
        num_steps += 1
        rate = rate_
        print(rate)
        if num_steps > max_steps:
            return None, None, None
    return img, mask_full, mask


def rescale_maximum(img, mask, scale=None, max_steps=50):
    bbox, rate = get_bbox_and_rate(mask)
    if bbox is None:
        return None, None, None
    x,y,w,h = bbox
    H,W,C = img.shape
    if H/h < W/w:
        y_start, y_end = y, y+h
        new_w = w/H*h
        c_x = x + w//2
        c_x_new = new_w*c_x/W
        x_start = c_x - c_x_new
        x_end = x_start + new_w
    else:
        x_start, x_end = x, x+w
        new_h = h/W*w
        c_y = y+h//2
        c_y_new = new_h*c_y/H
        y_start = c_y - c_y_new
        y_end = y_start + new_h
    img_new = img[min(y, int(y_start)):max(int(y_end), y+h), min(x, int(x_start)):max(int(x_end),x+w), :]
    mask_new = mask[min(y, int(y_start)):max(int(y_end),y+h),min(x, int(x_start)):max(int(x_end),x+w),:]

    img_new = cv2.resize(img_new, (W,H))
    mask_new = cv2.resize(mask_new, (W,H))

    return img_new, mask_new, mask_new
   

if __name__ == '__main__':
    args = parser.parse_args()
    scale = args.scale
    img_path_save = 'results/img_rescaled.png'
    mask_path_save = 'results/mask_rescaled.png'
    if scale == None:
        shutil.copy(args.img_path, img_path_save)
        shutil.copy(args.mask_path, mask_path_save)
    else:
        try:
            finals = []
            img = cv2.imread(args.img_path)
            mask = cv2.imread(args.mask_path)

            img_rescale, mask_rescale, mask_obj = rescale_maximum(img.copy(), mask.copy(), scale=scale)
            bbox, max_rate = get_bbox_and_rate(mask_obj)
            if scale < max_rate:
                img_rescale, mask_rescale, mask_obj = rescale(img.copy(), mask.copy(), scale=scale)
            if img_rescale is None:
                print('Invalid size')
                shutil.copy(args.img_path, img_path_save)
                shutil.copy(args.mask_path, mask_path_save)
                sys.exit()
            final = [img, img_rescale, mask, mask_rescale, mask_obj]
            # cv2.imwrite('tmp.png', cv2.hconcat(final))

            cv2.imwrite(img_path_save, img_rescale)
            cv2.imwrite(mask_path_save, mask_obj)
            # cv2.imwrite(mask_path_save_full, mask_rescale)
        except:
            print('Invalid size, using the original one')
            shutil.copy(args.img_path, img_path_save)
            shutil.copy(args.mask_path, mask_path_save)
        
        
    
    


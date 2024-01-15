'''
    Source: https://github.com/mseitzer/pytorch-fid
    Modified code to be compatible with our attack pipeline

    Copyright [2021] [Maximilian Seitzer]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

'''

import numpy as np
import pytorch_fid.fid_score
import torch
from pytorch_fid.inception import InceptionV3
from utils.stylegan import create_image

IMAGE_EXTENSIONS = ('bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp')


class FID_Score:
    def __init__(self, dataset_1, dataset_2, device, crop_size=None, generator=None, batch_size=128, dims=2048, num_workers=8, gpu_devices=[]):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device
        self.generator = generator
        self.crop_size = crop_size

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx]).to(self.device)
        if 0:
        #if len(gpu_devices) > 1:
            self.inception_model = torch.nn.DataParallel(inception_model, device_ids=gpu_devices)
        else:
            self.inception_model = inception_model
        self.inception_model.to(device)
            
    def compute_fid(self, rtpt=None):
        m1, s1 = self.compute_statistics(self.dataset_1, rtpt)
        m2, s2 = self.compute_statistics(self.dataset_2, rtpt)
        fid_value = pytorch_fid.fid_score.calculate_frechet_distance(
            m1, s1, m2, s2)
        return fid_value

    def compute_statistics(self, dataset, rtpt=None):
        self.inception_model.eval()
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        max_iter = int(len(dataset) / self.batch_size)
        for step, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                if x.shape[1] != 3:
                    x = create_image(x, self.generator,
                                     crop_size=self.crop_size, resize=299, batch_size=int(self.batch_size / 2), device = self.device)

                x = x.to(self.device)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

            if rtpt:
                rtpt.step(
                    subtitle=f'FID Score Computation step {step} of {max_iter}')

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma

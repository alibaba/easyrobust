import sys

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3

sys.path.insert(0, '/workspace')
from datasets.custom_subset import SingleClassSubset
from utils.stylegan import create_image


class PRCD:
    def __init__(self, dataset_real, dataset_fake, device, crop_size=None, generator=None, batch_size=128, dims=2048, num_workers=16, gpu_devices=[]):
        self.dataset_real = dataset_real
        self.dataset_fake = dataset_fake
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device
        self.generator = generator
        self.crop_size = crop_size

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx])
        if len(gpu_devices) > 1:
            self.inception_model = torch.nn.DataParallel(inception_model, device_ids=gpu_devices)
        else:
            self.inception_model = inception_model
        self.inception_model.to(self.device)
        
    def compute_metric(self, num_classes, k=3, rtpt=None):
        precision_list = []
        recall_list = []
        density_list = []
        coverage_list = []
        for step, cls in enumerate(range(num_classes)):
            with torch.no_grad():
                embedding_fake = self.compute_embedding(self.dataset_fake, cls)
                embedding_real = self.compute_embedding(self.dataset_real, cls)
                pair_dist_real = torch.cdist(embedding_real, embedding_real, p=2)
                print(pair_dist_real.shape)
                pair_dist_real = torch.sort(pair_dist_real, dim=1, descending=False)[0]
                print(pair_dist_real.shape)
                pair_dist_fake = torch.cdist(embedding_fake, embedding_fake, p=2)
                pair_dist_fake = torch.sort(pair_dist_fake, dim=1, descending=False)[0]
                radius_real = pair_dist_real[:, k]
                radius_fake = pair_dist_fake[:, k]

                # Compute precision
                distances_fake_to_real = torch.cdist(embedding_fake, embedding_real, p=2)
                min_dist_fake_to_real, nn_real = distances_fake_to_real.min(dim=1)
                precision = (min_dist_fake_to_real <= radius_real[nn_real]).float().mean()
                precision_list.append(precision.cpu().item())

                # Compute recall
                distances_real_to_fake = torch.cdist(embedding_real, embedding_fake, p=2)
                min_dist_real_to_fake, nn_fake = distances_real_to_fake.min(dim=1)
                recall = (min_dist_real_to_fake <= radius_fake[nn_fake]).float().mean()
                recall_list.append(recall.cpu().item())

                # Compute density
                num_samples = distances_fake_to_real.shape[0]
                sphere_counter = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0).mean()
                density = sphere_counter / k
                density_list.append(density.cpu().item())

                # Compute coverage
                num_neighbors = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0)
                coverage = (num_neighbors > 0).float().mean()
                coverage_list.append(coverage.cpu().item())
                # Update rtpt
                if rtpt:
                    rtpt.step(
                        subtitle=f'PRCD Computation step {step} of {num_classes}')

        # Compute mean over targets
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        density = np.mean(density_list)
        coverage = np.mean(coverage_list)
        return precision, recall, density, coverage

    def compute_embedding(self, dataset, cls=None):
        self.inception_model.eval()
        if cls:
            dataset = SingleClassSubset(dataset, cls)
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

        return torch.from_numpy(pred_arr)
